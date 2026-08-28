import torch
import copy
from supervisor_controller.utils.util import soft_update, huber_loss, mse_loss, to_torch, log_loss, update_linear_schedule
from supervisor_controller.utils.valuenorm import ValueNorm
import numpy as np
import torch.nn.functional as F

class R_MAPPO:
    def __init__(self, args, num_agents, policies, policy_mapping_fn, device=torch.device("cpu"), episode_length=25,vdn=False):
        """
        Trainer class for QMix with MLP policies. See parent class for more information.
        :param vdn: (bool) whether the algorithm in use is VDN.
        """
        self.args = args
        self.use_popart = self.args.use_popart
        self.use_value_active_masks = self.args.use_value_active_masks
        self.use_per = self.args.use_per
        self.per_eps = self.args.per_eps
        self.use_huber_loss = self.args.use_huber_loss
        self.huber_delta = self.args.huber_delta
        self.clip_param = self.args.clip_param
        self.use_vfunction = self.args.use_vfunction
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.lr = self.args.lr
        self.critic_lr = self.args.critic_lr
        self.adj_lr = self.args.adj_lr
        self.tau = self.args.tau
        self.opti_eps = self.args.opti_eps
        self.weight_decay = self.args.weight_decay
        self.episode_length = episode_length
        self.num_agents = num_agents
        self.highest_orders = self.args.highest_orders
        self.use_dyn_graph = self.args.use_dyn_graph
        self.entropy_coef = self.args.entropy_coef
        self._use_valuenorm = self.args.use_valuenorm
        self.adj_max_grad_norm = self.args.adj_max_grad_norm
        self.policies = policies
        self.policy_mapping_fn = policy_mapping_fn
        self.use_popart = self.args.use_popart
        self.policy_ids = sorted(list(self.policies.keys()))
        self.pinn_coef = float(getattr(self.args, 'pinn_coef', 1.0))
        self.max_pinn_loss = float(getattr(self.args, 'max_pinn_loss', 10.0))
        self.max_return_abs = float(getattr(self.args, 'max_return_abs', 200.0))
        self.policy_agents = {policy_id: sorted(
            [agent_id for agent_id in range(self.num_agents) if self.policy_mapping_fn(agent_id) == policy_id]) for policy_id in
            self.policies.keys()}
        if self._use_valuenorm:
            # --- 方案 A: 使用外部 ValueNorm ---
            self.value_normalizer = {policy_id: ValueNorm(1).to(self.device) for policy_id in self.policies.keys()}
            print("Value Normalization: Using external ValueNorm module.")

        elif self.use_popart:
            # --- 方案 B: 使用内置 PopArt ---
            self.value_normalizer = {policy_id: self.policies[policy_id].v_network.output_layer for policy_id in self.policies.keys()}
            print("Value Normalization: Using built-in PopArt layer.")

        else:
            # --- 方案 C: 不使用任何归一化 ---
            self.value_normalizer = {policy_id: None for policy_id in self.policies.keys()}
            print("Value Normalization: Disabled.")
   

        multidiscrete_list = None
        if any([isinstance(policy.act_dim, np.ndarray) for policy in self.policies.values()]):
            # multidiscrete
            multidiscrete_list = [len(self.policies[p_id].act_dim) *
                                  len(self.policy_agents[p_id]) for p_id in self.policy_ids]

        # target policies/networks
        # self.adj_network = adj_network
        self.actor_parameters = []
        self.critic_parameters = []
        for policy in self.policies.values():
            self.actor_parameters += policy.actor_parameters()
            self.critic_parameters += policy.critic_parameters()
        self.actor_optimizer = torch.optim.Adam(params=self.actor_parameters, lr=self.lr, eps=self.opti_eps)
        self.critic_optimizer = torch.optim.Adam(params=self.critic_parameters, lr=self.critic_lr, eps=self.opti_eps)
        # self.adj_parameters = []
        # self.adj_parameters += self.adj_network.parameters()
        # self.adj_optimizer = torch.optim.Adam(params=self.adj_parameters, lr=self.adj_lr, eps=self.opti_eps)
        
        # if args.use_double_q:
        #     print("double Q learning will be used")
    
    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        # update_linear_schedule(self.adj_optimizer, episode, episodes, self.adj_lr)
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def _sanitize_tensor(self, tensor, clip_abs=None):
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        if clip_abs is not None:
            tensor = torch.clamp(tensor, -clip_abs, clip_abs)
        return tensor

    def _params_are_finite(self, params):
        return all(torch.isfinite(p).all().item() for p in params)

    def _snapshot_params(self, params):
        return [p.detach().clone() for p in params]

    def _restore_params(self, params, snapshot):
        with torch.no_grad():
            for param, old_param in zip(params, snapshot):
                param.copy_(old_param)

    def _repair_nonfinite_params(self, params, optimizer=None):
        repaired = False
        with torch.no_grad():
            for param in params:
                if not torch.isfinite(param).all():
                    param.copy_(torch.nan_to_num(param, nan=0.0, posinf=1.0, neginf=-1.0))
                    param.clamp_(-10.0, 10.0)
                    repaired = True

        if optimizer is not None:
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value) and not torch.isfinite(value).all():
                        value.copy_(torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))
                        repaired = True

        return repaired

    def _safe_item(self, value, default=0.0):
        if torch.is_tensor(value):
            value = value.detach()
            if value.numel() != 1:
                value = value.mean()
            value = value.item()
        value = float(value)
        if not np.isfinite(value):
            return float(default)
        return value

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch, p_id):
        """
        Calculate value function loss using ONLY basic torch operations for maximum safety.
        """
        with torch.no_grad():
            valid_returns = return_batch[active_masks_batch == 1]
            if len(valid_returns) > 0:
                self.value_normalizer[p_id].update(valid_returns)

            returns_target = self.value_normalizer[p_id].normalize(return_batch).detach()

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)

        error_clipped = returns_target - value_pred_clipped
        error_original = returns_target - values

        value_loss_clipped = mse_loss(error_clipped)
        value_loss_original = mse_loss(error_original)

        value_loss_tensor = torch.max(value_loss_original, value_loss_clipped)

        active_masks_sum = active_masks_batch.sum()
        if active_masks_sum > 0:
            final_value_loss = (value_loss_tensor * active_masks_batch).sum() / active_masks_sum
        else:
            final_value_loss = (values * 0.0).sum()
            
        return final_value_loss
    
    def _calculate_pinn_loss(self, obs_batch, pde_params, active_masks):
   
        obs_batch = self._sanitize_tensor(obs_batch, clip_abs=10.0)
        B, T, N, D = obs_batch.shape
        if active_masks is not None:
            active_masks = active_masks.view(B, T, N, 1)
        else:
            active_masks = torch.ones((B, T, N, 1), device=obs_batch.device)

        w_target_act = self._sanitize_tensor(pde_params['w_target'].view(B, T, N, 1), clip_abs=0.13)
        w_center_act = self._sanitize_tensor(pde_params['w_center'].view(B, T, N, 1), clip_abs=0.13)
        w_rand_act   = self._sanitize_tensor(pde_params['w_rand'].view(B, T, N, 1), clip_abs=0.13)
        k_diff_act   = self._sanitize_tensor(pde_params['k_diff'].view(B, T, N, 1), clip_abs=0.035)

        lambda_anchor  = pde_params['lambda_anchor'].view(B, T, N, 1)
        lambda_release = pde_params['lambda_release'].view(B, T, N, 1)

        env_rho_norm = obs_batch[:, :, :, 2:3]       # 局部密度 rho
        dist_target = obs_batch[:, :, :, 3:4]        # 目标势场 (exp(-d_target))
        vec_target = obs_batch[:, :, :, 4:6]         # 指向救援目标的单位向量
        dist_base = obs_batch[:, :, :, 6:7]          # 基地势场 (exp(-d_base))
        vec_center = obs_batch[:, :, :, 7:9]         # 指向中继驻守轴线的回归向量
        grad_rho = obs_batch[:, :, :, 9:11]          # 避障排斥密度梯度

        phase_searcher = obs_batch[:, :, :, 11:12]   # Phase A: 自由盲搜
        phase_responder = obs_batch[:, :, :, 12:13]  # Phase B: 起锚冲锋
        phase_relay = obs_batch[:, :, :, 13:14]      # Phase C: 抛锚中继筑链

        max_speed = 0.12
        eps_rho = 0.05

        steer_angles = (torch.rand((B, T, N, 1), device=obs_batch.device) * 2.0 - 1.0) * (3.1415926 / 3.0)
        vec_rand = torch.cat([torch.cos(steer_angles), torch.sin(steer_angles)], dim=-1)

        repulsion_term = - grad_rho / (env_rho_norm + eps_rho)
        
        # 提取 T-1 步的相态指示器
        p_search = phase_searcher[:, :-1]
        p_respond = phase_responder[:, :-1]
        p_relay = phase_relay[:, :-1]

        v_target_col = vec_target[:, :-1].unsqueeze(-1) * p_respond.unsqueeze(-1)
        v_center_col = vec_center[:, :-1].unsqueeze(-1) * p_relay.unsqueeze(-1)
        v_rand_col = vec_rand[:, :-1].unsqueeze(-1) * p_search.unsqueeze(-1)
        
        v_rep_col = repulsion_term[:, :-1].unsqueeze(-1)
        
        A = torch.cat([v_target_col, v_center_col, v_rand_col, v_rep_col], dim=-1) # (B, T-1, N, 2, 4)

        v_ideal_search = vec_rand     # Phase A: 随机盲搜
        v_ideal_respond = vec_target  # Phase B: 响应冲锋
        v_ideal_relay = vec_center    # Phase C: 轴线筑链中继
        
        v_ideal_multi_phase = (
            phase_searcher * v_ideal_search
            + phase_responder * v_ideal_respond
            + phase_relay * v_ideal_relay
        ) * max_speed
        
        Y_advection = v_ideal_multi_phase[:, :-1].unsqueeze(-1) # (B, T-1, N, 2, 1)

        k_ideal = 0.025 * torch.clamp(env_rho_norm[:, :-1] / 0.08, 0.0, 1.0)
        
        Y_repulsion = k_ideal.unsqueeze(-1) * v_rep_col  # (B, T-1, N, 2, 1)

        Y = Y_advection + Y_repulsion

        # 求解正规方程 (4x4 高精度求解)
        A_flat = A.reshape(-1, 2, 4)            # (M, 2, 4)
        Y_flat = Y.reshape(-1, 2, 1)            # (M, 2, 1)

        At = A_flat.transpose(1, 2)              # (M, 4, 2)
        AtA = torch.bmm(At, A_flat)              # (M, 4, 4)
        AtY = torch.bmm(At, Y_flat)              # (M, 4, 1)

        reg_eps = 1e-4
        eye = torch.eye(4, device=A.device).unsqueeze(0) * reg_eps # (1, 4, 4)
        
        sol_flat = torch.linalg.solve(AtA + eye, AtY)
        sol = sol_flat.reshape(B, T-1, N, 4)
        
        target_w_target = torch.clamp(sol[:, :, :, 0:1], 0.0, 0.13)
        target_w_center = torch.clamp(sol[:, :, :, 1:2], 0.0, 0.13)
        target_w_rand   = torch.clamp(sol[:, :, :, 2:3], 0.0, 0.13)
        target_k_diff   = torch.clamp(sol[:, :, :, 3:4], 0.0, 0.035)


        u_actual = (
            w_target_act * vec_target 
            + w_center_act * vec_center 
            + w_rand_act * vec_rand
        )
        u_theory = (
            target_w_target * vec_target[:, :-1]
            + target_w_center * vec_center[:, :-1]
            + target_w_rand * vec_rand[:, :-1]
        )

        v_des_actual = u_actual[:, :-1] - (k_diff_act[:, :-1] / (env_rho_norm[:, :-1] + eps_rho)) * grad_rho[:, :-1]
        v_des_theory = u_theory - (target_k_diff / (env_rho_norm[:, :-1] + eps_rho)) * grad_rho[:, :-1]

        # 使用一阶 L1-Norm 范数替代 L2 平方
        loss_micro_map = torch.clamp(
            torch.norm(v_des_actual - v_des_theory, dim=-1, keepdim=True),
            0.0,
            1.0,
        )

        adv_diff_drift = torch.sum((u_actual[:, :-1] - u_theory) * grad_rho[:, :-1], dim=-1, keepdim=True)
        
        r_target = -torch.log(torch.clamp(dist_target[:, :-1], min=1e-5))
        visible_target_mask = torch.where(dist_target[:, :-1] > 0.05, 1.0, 0.0)
        
        div_target = -1.0 / (r_target + 1e-3) * visible_target_mask
        
        div_u_actual = w_target_act[:, :-1] * div_target
        div_u_theory = target_w_target * div_target
        adv_diff_compress = env_rho_norm[:, :-1] * (div_u_actual - div_u_theory)
        
        adv_diff_phase = (adv_diff_drift + adv_diff_compress)

        closer_count_obs = obs_batch[:, :-1, :, 14:15]

        TRAIN_AGENT_COUNT = 8.0
        closer_count_t = closer_count_obs * TRAIN_AGENT_COUNT
        
        knows_target_all_t = 1.0 - phase_searcher[:, :-1]

        target_anchor = ((closer_count_t >= 2.99) & (knows_target_all_t > 0.5)).float()
        target_release = ((closer_count_t < 2.99) & (knows_target_all_t > 0.5)).float()

        weight_anchor = 1.0 + 3.0 * target_anchor
        weight_release = 1.0 + 3.0 * target_release

        loss_react_search = (lambda_anchor[:, :-1] + lambda_release[:, :-1])

        react_diff_active = (
            weight_anchor * (lambda_anchor[:, :-1] - target_anchor)
            + weight_release * (lambda_release[:, :-1] - target_release)
        )

        reaction_diff = phase_searcher[:, :-1] * loss_react_search + (1.0 - phase_searcher[:, :-1]) * react_diff_active

        macro_residual_diff = adv_diff_phase + reaction_diff
        loss_macro_map = torch.clamp(macro_residual_diff ** 2, 0.0, 25.0)

        count_search = phase_searcher[:, :-1].sum() + 1e-5
        count_respond = phase_responder[:, :-1].sum() + 1e-5
        count_relay = phase_relay[:, :-1].sum() + 1e-5
        total_count = count_search + count_respond + count_relay

        weight_search = total_count / (3.0 * count_search)
        weight_respond = total_count / (3.0 * count_respond)
        weight_relay = total_count / (3.0 * count_relay)

        dynamic_weight = (
            phase_searcher[:, :-1] * weight_search
            + phase_responder[:, :-1] * weight_respond
            + phase_relay[:, :-1] * weight_relay
        )

        loss_micro_map = loss_micro_map * dynamic_weight
        loss_macro_map = loss_macro_map * dynamic_weight

        valid_transition_masks = active_masks[:, :-1] * active_masks[:, 1:]
        loss_micro = (loss_micro_map * valid_transition_masks).sum() / (valid_transition_masks.sum() + 1e-5)
        loss_macro = (loss_macro_map * active_masks[:, :-1]).sum() / (active_masks[:, :-1].sum() + 1e-5)

        COEF_MICRO = 40.0
        COEF_MACRO = 0.5 
        
        total_pinn_loss = torch.clamp(COEF_MACRO * loss_macro + COEF_MICRO * loss_micro, 0.0, self.max_pinn_loss)
        total_pinn_loss = self._sanitize_tensor(total_pinn_loss, clip_abs=self.max_pinn_loss)

        return total_pinn_loss, {
            'pinn/loss_macro': loss_macro.item(),
            'pinn/loss_micro': loss_micro.item(),
            'pinn/v_des_norm_diff': torch.norm(v_des_actual - v_des_theory, dim=-1).mean().item(),
            'pinn/active_target_w_target': target_w_target[phase_responder[:, :-1] > 0.5].mean().item() if (phase_responder > 0.5).any() else 0.0,
            'pinn/active_target_w_center': target_w_center[phase_relay[:, :-1] > 0.5].mean().item() if (phase_relay > 0.5).any() else 0.0,
            'pinn/active_target_w_rand': target_w_rand[phase_searcher[:, :-1] > 0.5].mean().item() if (phase_searcher > 0.5).any() else 0.0,
        }
    
    def _torch_map_actions_to_params(self, raw_action_batch):
        raw_action_torch = to_torch(raw_action_batch)

        TOTAL_W_BUDGET = 0.13
        MAX_K = 0.035
        TEMPERATURE = 5.0

        advection_logits = raw_action_torch[..., :3] * TEMPERATURE
        advection_weights = torch.nn.functional.softmax(advection_logits, dim=-1)
        
        w_target = advection_weights[..., 0:1] * TOTAL_W_BUDGET
        w_center = advection_weights[..., 1:2] * TOTAL_W_BUDGET
        w_rand   = advection_weights[..., 2:3] * TOTAL_W_BUDGET

        k_raw = torch.clamp(raw_action_torch[..., 3:4], -1.0, 1.0)
        k_diff = ((k_raw + 1.0) / 2.0) * MAX_K

        raw_lambda_anchor = torch.clamp(raw_action_torch[..., 4:5], -1.0, 1.0)
        raw_lambda_release = torch.clamp(raw_action_torch[..., 5:6], -1.0, 1.0)

        lambda_anchor = (raw_lambda_anchor + 1.0) / 2.0
        lambda_release = (raw_lambda_release + 1.0) / 2.0

        return {
            'w_target': w_target,
            'w_center': w_center,
            'w_rand': w_rand,
            'k_diff': k_diff,
            'lambda_anchor': lambda_anchor,
            'lambda_release': lambda_release,
            'advection_weights': advection_weights 
        }

    def train_policy_on_batch(self, batch, epoch=0, mini_batch_idx=0):
        """
        Trains the actor and critic networks on a mini-batch of sequence data.
        This version correctly handles RNN states for recurrent policies.
        """
        obs_batch, obs_neighbor_num_batch, share_obs_batch, dones_batch, \
        dones_env_batch, acts_batch, old_log_probs_batch, \
        advantages_batch, rnn_obs_batch, rnn_share_obs_batch, \
        returns_batch, value_preds_batch, _ = batch

        obs_batch = self._sanitize_tensor(to_torch(obs_batch).to(**self.tpdv), clip_abs=10.0)
        share_obs_batch = self._sanitize_tensor(to_torch(share_obs_batch).to(**self.tpdv), clip_abs=10.0)
        acts_batch = self._sanitize_tensor(to_torch(acts_batch).to(**self.tpdv), clip_abs=5.0)
        old_log_probs_batch = self._sanitize_tensor(to_torch(old_log_probs_batch).to(**self.tpdv), clip_abs=50.0)
        advantages_batch = self._sanitize_tensor(to_torch(advantages_batch).to(**self.tpdv), clip_abs=10.0)
        returns_batch = self._sanitize_tensor(
            to_torch(returns_batch).to(**self.tpdv),
            clip_abs=self.max_return_abs,
        )
        value_preds_batch = self._sanitize_tensor(to_torch(value_preds_batch).to(**self.tpdv), clip_abs=self.max_return_abs)
        active_masks_batch = to_torch(1.0 - dones_batch).to(**self.tpdv)
        
        rnn_obs_batch = self._sanitize_tensor(to_torch(rnn_obs_batch).to(**self.tpdv), clip_abs=10.0)
        rnn_share_obs_batch = self._sanitize_tensor(to_torch(rnn_share_obs_batch).to(**self.tpdv), clip_abs=10.0)
        
        obs_neighbor_num_batch = to_torch(obs_neighbor_num_batch).to(**self.tpdv).long()

        data_chunk_length = obs_batch.shape[1]
        batch_chunks = obs_batch.shape[0]
        batch_agents = obs_batch.shape[2]

        for p_id in self.policy_ids:
            policy = self.policies[p_id]
            repaired_actor_params = float(self._repair_nonfinite_params(
                self.actor_parameters,
                self.actor_optimizer,
            ))
            repaired_critic_params = float(self._repair_nonfinite_params(
                self.critic_parameters,
                self.critic_optimizer,
            ))

            obs_for_rnn = obs_batch.permute(1, 0, 2, 3).reshape(data_chunk_length, -1, obs_batch.shape[-1])
            rnn_states_for_rnn = rnn_obs_batch.reshape(-1, rnn_obs_batch.shape[-1])
            obs_neighbor_num_for_rnn = obs_neighbor_num_batch.permute(1, 0, 2, 3).reshape(-1, obs_neighbor_num_batch.shape[-1])
            actor_hidden_states, _, _ = policy.get_hidden_states(
                obs_for_rnn, 
                None, # prev_actions
                rnn_states_for_rnn, 
                obs_neighbor_num_for_rnn 
            )
            actor_hidden_states_flat = (
                actor_hidden_states
                .reshape(data_chunk_length, batch_chunks, batch_agents, -1)
                .permute(1, 0, 2, 3)
                .contiguous()
                .reshape(-1, actor_hidden_states.shape[-1])
            )

            acts_batch_flat = acts_batch.reshape(-1, acts_batch.shape[-1])
            obs_batch_flat = obs_batch.reshape(-1, obs_batch.shape[-1])
            old_log_probs_batch_flat = old_log_probs_batch.reshape(-1, 1)
            advantages_batch_flat = advantages_batch.reshape(-1, 1)
            active_masks_batch_flat = active_masks_batch.reshape(-1, 1)

            new_log_probs, dist_entropy, action_dist = policy.evaluate_actions(
                actor_hidden_states_flat, 
                acts_batch_flat,
                obs_batch_flat
                )
        
            ratio = torch.exp(new_log_probs - old_log_probs_batch_flat)
            surr1 = ratio * advantages_batch_flat
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch_flat

            active_masks_sum = active_masks_batch_flat.sum()
            if active_masks_sum > 0:
                actor_loss = -torch.sum(torch.min(surr1, surr2) * active_masks_batch_flat) / active_masks_sum
            else:
                actor_loss = (surr1 * 0.0).sum()

            action_logits = action_dist.loc
            pde_params_torch = self._torch_map_actions_to_params(action_logits)
            
            pinn_loss, pinn_logs = self._calculate_pinn_loss(
                obs_batch=obs_batch,                 # 形状：(B, T, N, D)
                pde_params=pde_params_torch,         # 内部各 Tensor 在计算时会自动 reshape
                active_masks=active_masks_batch      # 形状：(B, T, N, 1) 或 (B, T, 1)
            )

            final_actor_loss = actor_loss - dist_entropy * self.entropy_coef + self.pinn_coef * pinn_loss

            self.actor_optimizer.zero_grad()
            if torch.isfinite(final_actor_loss):
                actor_snapshot = self._snapshot_params(self.actor_parameters)
                final_actor_loss.backward()
                grad_norm_actor = torch.nn.utils.clip_grad_norm_(
                    self.actor_parameters,
                    self.args.max_grad_norm,
                    error_if_nonfinite=False,
                )
                if torch.isfinite(grad_norm_actor) and self._params_are_finite(self.actor_parameters):
                    self.actor_optimizer.step()
                    if self._params_are_finite(self.actor_parameters):
                        skipped_actor_update = 0.0
                    else:
                        self._restore_params(self.actor_parameters, actor_snapshot)
                        self._repair_nonfinite_params(self.actor_parameters, self.actor_optimizer)
                        skipped_actor_update = 1.0
                else:
                    self._restore_params(self.actor_parameters, actor_snapshot)
                    self.actor_optimizer.zero_grad()
                    skipped_actor_update = 1.0
            else:
                grad_norm_actor = torch.as_tensor(0.0, device=self.device)
                skipped_actor_update = 1.0
            
            share_obs_for_rnn = share_obs_batch.permute(1, 0, 2, 3) 
            rnn_states_for_rnn = rnn_share_obs_batch
            critic_features_seq, _, _ = policy.get_hidden_critic(share_obs_for_rnn, rnn_states_for_rnn)

            critic_features_flat = (
                critic_features_seq
                .reshape(data_chunk_length, batch_chunks, batch_agents, -1)
                .permute(1, 0, 2, 3)
                .contiguous()
                .reshape(-1, critic_features_seq.shape[-1])
            )

            values = policy.get_v_value(critic_features_flat.shape[0], critic_features_flat)
            returns_batch_flat = returns_batch.reshape(-1, 1)
            value_preds_batch_flat = value_preds_batch.reshape(-1, 1)
            active_masks_batch_flat = active_masks_batch.reshape(-1, 1)

            critic_loss = self.cal_value_loss(
                values,
                value_preds_batch_flat,
                returns_batch_flat,
                active_masks_batch_flat,
                p_id
            )

            with torch.no_grad():
                y_true = returns_batch_flat
                y_pred_norm = values      
                valid_mask = active_masks_batch_flat == 1 

                normalizer = self.value_normalizer.get(p_id)
                if normalizer is not None:
                    y_pred_np = normalizer.denormalize(y_pred_norm)
                    y_pred_raw = torch.from_numpy(y_pred_np).to(y_true.device)
                else:
                    y_pred_raw = y_pred_norm

                if valid_mask.sum() > 1:
                    var_y_true = torch.var(y_true[valid_mask])
                    var_diff = torch.var(y_true[valid_mask] - y_pred_raw[valid_mask])
                    explained_var = 1 - var_diff / (var_y_true + 1e-8)
                    critic_ev = explained_var.item()
                else:
                    critic_ev = float('nan')

            self.critic_optimizer.zero_grad()
            if torch.isfinite(critic_loss):
                critic_snapshot = self._snapshot_params(self.critic_parameters)
                critic_loss.backward()
                grad_norm_critic = torch.nn.utils.clip_grad_norm_(
                    self.critic_parameters,
                    self.args.max_grad_norm,
                    error_if_nonfinite=False,
                )
                if torch.isfinite(grad_norm_critic) and self._params_are_finite(self.critic_parameters):
                    self.critic_optimizer.step()
                    if self._params_are_finite(self.critic_parameters):
                        skipped_critic_update = 0.0
                    else:
                        self._restore_params(self.critic_parameters, critic_snapshot)
                        self._repair_nonfinite_params(self.critic_parameters, self.critic_optimizer)
                        skipped_critic_update = 1.0
                else:
                    self._restore_params(self.critic_parameters, critic_snapshot)
                    self.critic_optimizer.zero_grad()
                    skipped_critic_update = 1.0
            else:
                grad_norm_critic = torch.as_tensor(0.0, device=self.device)
                skipped_critic_update = 1.0

        train_info = {}
        train_info['actor_loss'] = self._safe_item(actor_loss)
        train_info['critic_loss'] = self._safe_item(critic_loss)
        train_info['entropy'] = self._safe_item(dist_entropy)
        train_info['grad_norm_actor'] = self._safe_item(grad_norm_actor)
        train_info['grad_norm_critic'] = self._safe_item(grad_norm_critic)
        train_info['ratio'] = self._safe_item(ratio.mean())
        train_info['advantage'] = self._safe_item(advantages_batch_flat.mean())
        clamp_mask = (ratio < 1.0 - self.clip_param) | (ratio > 1.0 + self.clip_param)
        train_info['clamp_ratio'] = self._safe_item(clamp_mask.float().mean())
        train_info['explained_variance'] = self._safe_item(critic_ev)
        train_info['pinn_loss'] = self._safe_item(pinn_loss)
        train_info['pinn_coef'] = self.pinn_coef
        train_info['skipped_actor_update'] = skipped_actor_update
        train_info['skipped_critic_update'] = skipped_critic_update
        train_info['repaired_actor_params'] = repaired_actor_params
        train_info['repaired_critic_params'] = repaired_critic_params
        train_info.update({
            key: self._safe_item(value)
            for key, value in pinn_logs.items()
        })

        return train_info, None, None

    def prep_training(self):
        """See parent class."""
        for p_id in self.policy_ids:
            self.policies[p_id].rnn_network.train()
            self.policies[p_id].act.train()
            if self.use_vfunction:
                self.policies[p_id].rnn_critic_network.train()
                self.policies[p_id].v_network.train()

    def prep_rollout(self):
        """See parent class."""
        for p_id in self.policy_ids:
            self.policies[p_id].rnn_network.eval()
            self.policies[p_id].act.eval()
            if self.use_vfunction:
                self.policies[p_id].rnn_critic_network.eval()
                self.policies[p_id].v_network.eval()
        #self.mixer.eval()
        #self.target_mixer.eval()
