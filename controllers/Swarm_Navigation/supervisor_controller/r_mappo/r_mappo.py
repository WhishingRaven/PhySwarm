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
    
    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        # update_linear_schedule(self.adj_optimizer, episode, episodes, self.adj_lr)
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch, p_id):
        """
        Calculate value function loss using ONLY basic torch operations for maximum safety.
        """
        normalizer = self.value_normalizer.get(p_id)
        with torch.no_grad():
            if normalizer is not None and hasattr(normalizer, 'update'):
                valid_returns = return_batch[active_masks_batch == 1]
                if len(valid_returns) > 0:
                    normalizer.update(valid_returns)
                returns_target = normalizer.normalize(return_batch).detach()
            else:
                returns_target = return_batch.detach()

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
    
    def _sanitize_tensor(self, tensor, clip_abs=10.0):
        """防止数值溢出的安全过滤器"""
        return torch.clamp(tensor, -clip_abs, clip_abs)

    def _calculate_pinn_loss(self, obs_batch, pde_params, active_masks):

        obs_batch = self._sanitize_tensor(obs_batch, clip_abs=10.0)
        B, T, N, D = obs_batch.shape
        active_masks = active_masks.view(B, T, N, 1)

        w_flow_act = self._sanitize_tensor(pde_params['w_flow'].view(B, T, N, 1), clip_abs=0.13)
        w_shape_act = self._sanitize_tensor(pde_params['w_shape'].view(B, T, N, 1), clip_abs=0.13)
        k_diff_act = self._sanitize_tensor(pde_params['k_diff'].view(B, T, N, 1), clip_abs=0.035)
        beta_act = self._sanitize_tensor(pde_params['beta'].view(B, T, N, 1), clip_abs=5.0)

        env_rho_norm = obs_batch[:, :, :, 2:3]  # 混合归一化密度
        corridor_width = obs_batch[:, :, :, 12:13]
        phase_indicator = obs_batch[:, :, :, 13:16]

        vec_centroid = obs_batch[:, :, :, 5:7]
        dist_centroid = obs_batch[:, :, :, 7:8]
        goal_body = obs_batch[:, :, :, 9:11]

        max_speed = 0.12
        SHAPE_GAIN = 50.0
        alpha = 1.0

        pos_rel = vec_centroid * dist_centroid                 # (B, T, N, 2)
        diff_pos = pos_rel.unsqueeze(3) - pos_rel.unsqueeze(2) # (B, T, N, N, 2)
        dist_matrix_all = self._sanitize_tensor(torch.norm(diff_pos, dim=-1), clip_abs=5.0) # (B, T, N, N)

        h_agent = 0.15
        h2 = h_agent ** 2
        h4 = h_agent ** 4
        
        W_agent = torch.exp(-dist_matrix_all**2 / (2.0 * h2))
        mask_eye = torch.eye(N, device=obs_batch.device).view(1, 1, N, N)
        W_agent = W_agent * (1.0 - mask_eye)  # (B, T, N, N)
        
        pure_agent_rho = torch.sum(W_agent, dim=-1, keepdim=True) # (B, T, N, 1)
        pure_rho_norm = torch.clamp(pure_agent_rho / 2.0, 0.0, 1.0)

        grad_rho_analytical = torch.sum((-diff_pos / h2) * W_agent.unsqueeze(-1), dim=3)
        grad_rho_analytical = self._sanitize_tensor(grad_rho_analytical, clip_abs=3.0)

        laplacian_analytical = torch.sum((dist_matrix_all**2 / h4 - 2.0 / h2) * W_agent, dim=3, keepdim=True)
        laplacian_analytical = self._sanitize_tensor(laplacian_analytical, clip_abs=5.0)

        def get_shape_force_and_div(beta_val):
            """
            解析推导椭圆向心力场与其对应的空间散度 (Divergence)
            """
            x_rel = vec_centroid[:, :, :, 0:1] * dist_centroid
            y_rel = vec_centroid[:, :, :, 1:2] * dist_centroid
            
            d_w = torch.sqrt(torch.clamp(alpha * x_rel**2 + beta_val * y_rel**2, min=1e-8))
            dynamic_R0 = 0.2 - (1.0 - corridor_width) * 0.05
            err_w = torch.clamp(d_w - dynamic_R0, -0.5, 0.5)
            
            # 1. 向心矢量力场
            fx = SHAPE_GAIN * (err_w / d_w) * (alpha * x_rel)
            fy = SHAPE_GAIN * (err_w / d_w) * (beta_val * y_rel)
            V_force = self._sanitize_tensor(torch.cat([fx, fy], dim=-1), clip_abs=5.0)
            
            # 2. 解析散度计算：\nabla \cdot V
            term1 = SHAPE_GAIN * (alpha + beta_val) * (1.0 - dynamic_R0 / d_w)
            term2 = SHAPE_GAIN * (dynamic_R0 / (d_w**3 + 1e-6)) * (alpha**2 * x_rel**2 + beta_val**2 * y_rel**2)
            div_V = self._sanitize_tensor(term1 + term2, clip_abs=10.0)
            
            return V_force, div_V


        target_beta_shape = torch.clamp(1.2 + (1.0 - corridor_width) * 5.0, 1.2, 5.0)
        
        target_beta = (1.0 - phase_indicator[:, :, :, 1:2]) * 1.2 + phase_indicator[:, :, :, 1:2] * target_beta_shape
        
        V_shape_theory, div_V_theory = get_shape_force_and_div(target_beta)

        eps_rho = 0.05
        repulsion_term = -grad_rho_analytical / (env_rho_norm + eps_rho)  # 扩散项 (B, T, N, 2)
        
        goal_t = goal_body[:, :-1]
        V_shape_t = V_shape_theory[:, :-1]
        rep_t = repulsion_term[:, :-1]
        
        goal_flat = goal_t.reshape(B, T-1, N*2, 1)
        V_shape_flat = V_shape_t.reshape(B, T-1, N*2, 1)
        rep_flat = rep_t.reshape(B, T-1, N*2, 1)
        
        A = torch.cat([goal_flat, V_shape_flat, rep_flat], dim=-1)

        err_radius = dist_centroid - 0.20
        v_ideal_gathering = 0.8 * torch.clamp(err_radius / 0.10, -1.0, 1.0) * vec_centroid
        
        v_actual = obs_batch[:, :, :, 0:2] * max_speed
        
        v_ideal_transit_x = v_actual[:, :, :, 0:1]
        y_rel = vec_centroid[:, :, :, 1:2] * dist_centroid
        v_ideal_transit_y = 0.8 * (1.0 - corridor_width) * y_rel

        v_ideal_transit = torch.cat([v_ideal_transit_x, v_ideal_transit_y], dim=-1)
        
        v_ideal_multi_phase = (
            phase_indicator[:, :, :, 0:1] * v_ideal_gathering 
            + phase_indicator[:, :, :, 1:2] * v_ideal_transit
            + phase_indicator[:, :, :, 2:3] * v_ideal_gathering
        )
        
        Y = v_ideal_multi_phase[:, 1:].reshape(B, T-1, N*2, 1)
        
        sol = torch.linalg.lstsq(A, Y).solution
        
        target_w_flow = torch.clamp(sol[:, :, 0:1, :], 0.0, 0.13).expand(B, T-1, N, 1)
        target_w_shape = torch.clamp(sol[:, :, 1:2, :], 0.0, 0.13).expand(B, T-1, N, 1)
        target_k_diff = torch.clamp(sol[:, :, 2:3, :], 0.0, 0.035).expand(B, T-1, N, 1)

        V_shape_actual, div_V_actual = get_shape_force_and_div(beta_act)

        u_actual = w_flow_act * goal_body + w_shape_act * V_shape_actual
        u_theory = target_w_flow * goal_body[:, :-1] + target_w_shape * V_shape_t

        div_u_actual = w_shape_act * div_V_actual
        div_u_theory = target_w_shape * div_V_theory[:, :-1]

        v_des_actual = u_actual[:, :-1] - (k_diff_act[:, :-1] / (env_rho_norm[:, :-1] + eps_rho)) * grad_rho_analytical[:, :-1]
        v_des_theory = u_theory - (target_k_diff / (env_rho_norm[:, :-1] + eps_rho)) * grad_rho_analytical[:, :-1]

        loss_micro_map = torch.clamp(
            torch.sum((v_des_actual - v_des_theory) ** 2, dim=-1, keepdim=True),
            0.0,
            1.0,
        )
        valid_transition_masks = active_masks[:, :-1] * active_masks[:, 1:]
        loss_micro = (loss_micro_map * valid_transition_masks).sum() / (valid_transition_masks.sum() + 1e-5)

        adv_diff_drift = torch.sum((u_actual[:, :-1] - u_theory) * grad_rho_analytical[:, :-1], dim=-1, keepdim=True)
        adv_diff_compress = pure_rho_norm[:, :-1] * (div_u_actual[:, :-1] - div_u_theory)
        adv_diff_phase = (adv_diff_drift + adv_diff_compress) * phase_indicator[:, :-1]

        rho_phase_laplacian = laplacian_analytical[:, :-1] * phase_indicator[:, :-1]
        diff_diff_phase = (k_diff_act[:, :-1] - target_k_diff) * rho_phase_laplacian

        macro_residual_diff = adv_diff_phase - diff_diff_phase
        loss_macro_map = torch.clamp(torch.sum(macro_residual_diff ** 2, dim=-1, keepdim=True), 0.0, 25.0)
        loss_macro = (loss_macro_map * active_masks[:, :-1]).sum() / (active_masks[:, :-1].sum() + 1e-5)

        COEF_MICRO = 1.0
        COEF_MACRO = 1.0
        total_pinn_loss = torch.clamp(COEF_MACRO * loss_macro + COEF_MICRO * loss_micro, 0.0, self.max_pinn_loss)
        total_pinn_loss = self._sanitize_tensor(total_pinn_loss, clip_abs=self.max_pinn_loss)

        return total_pinn_loss, {
            'pinn/loss_macro': loss_macro.item(),
            'pinn/loss_micro': loss_micro.item(),
            'pinn/v_des_norm_diff': torch.norm(v_des_actual - v_des_theory, dim=-1).mean().item()
        }
    
    def _torch_map_actions_to_params(self, raw_action_batch):
        """
        用于在训练过程中保留梯度，以便计算 PINN Loss。
        修正后的动作空间维度为 4:[w_flow, w_shape, k_diff, beta]
        """
        raw_action_torch = to_torch(raw_action_batch)

        TOTAL_W_BUDGET = 0.13  
        MAX_K = 0.035

        advection_weights = torch.softmax(raw_action_torch[..., 0:2], dim=-1)
        w_flow = advection_weights[..., 0:1] * TOTAL_W_BUDGET
        w_shape = advection_weights[..., 1:2] * TOTAL_W_BUDGET

        k_raw = torch.clamp(raw_action_torch[..., 2:3], -1.0, 1.0)
        k_diff = ((k_raw + 1.0) / 2.0) * MAX_K

        raw_beta = torch.clamp(raw_action_torch[..., 3:4], -1.0, 1.0)

        beta_min = 1.0
        beta_max = 5.0

        beta_unit = (raw_beta + 1.0) / 2.0

        beta = beta_min + (beta_max - beta_min) * torch.clamp(beta_unit, 0.0, 1.0)

        alpha = torch.ones_like(beta)

        return {
            'w_flow': w_flow,
            'w_shape': w_shape,
            'k_diff': k_diff,
            'alpha': alpha,
            'beta': beta
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
