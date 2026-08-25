import random
import numpy as np
from deepbots.supervisor.controllers.csv_supervisor_env import CSVSupervisorEnv
from gym.spaces import Discrete, Box
import utilities
import math
import random
import torch
from controller import Supervisor
from scipy.spatial.distance import pdist, squareform
from supervisor_controller.utils.util import to_torch
import matplotlib.pyplot as plt
import os
import pandas as pd


class Epuck2Supervisor(CSVSupervisorEnv):
    def __init__(self, all_args=None):
        self.args = all_args
        self.num_agents = self.args.num_agents
        self.num_targets = self.args.num_target
        self.num_envs = self.args.n_rollout_threads
        self.timestep = self.args.timestep
        self.interval = self.args.interval

        super().__init__(timestep=self.timestep)

        self.world_size = np.array([3.0, 1.0])
        # self.world_size = np.array([4.0, 2.0])
        
        #感知与通信
        self.num_obs_agents = self.args.num_obs_agents  # 3
        self.num_obs_targets = self.args.num_obs_targets  # 1
        self.obs_view = self.args.obs_view
        self.comm_view = self.args.comm_view
        
        #任务与奖励
        self.collision_distance = self.args.collision_distance
        self.episode_length = self.args.episode_length
        self.catch_distance = self.args.catch_distance

        #观测与状态空间维度
        self_dim = 2       # 2(vel_x, vel_y)
        field_dim = 9      # 3(F_base, vec_base_x, vec_base_y) + 3(F_target, vec_target) + 2(grad_rho)
        role_dim = 3       # 3(is_searcher, is_responder, is_relay) 

        self.num_observations = self_dim + field_dim + role_dim + 1

        # 状态空间
        single_num_state = self.num_observations
        self.num_states = self.num_agents * single_num_state
        self.obs_neighbor_dim = self.num_obs_targets + self.num_obs_agents

        # [w_base, w_target, w_rand, k_diff, lambda_respond, lambda_relay]
        self.num_actions = 6  

        #PDE框架与宏观物理场相关参数
        self.kde_bandwidth = 0.05     # KDE核函数带宽 (米)

        self.emitter, self.receiver = self.initialize_comms(
            "emitter", "receiver")

        self.arena_pos = self.add_arenas()
        self.target_pos, self.base_pos = self.add_nodes(self.arena_pos) 

        self.robots = []
        for i in range(1,self.num_envs+1):
            robot_env = []
            for j in range(1,self.num_agents+1):
                robot_env.append(self.getFromDef(f"epuck{i}-{j}"))
            self.robots.append(robot_env)

        self.targets = []
        for i in range(1, self.num_envs+1):
            target_env = []
            for j in range(1,self.num_targets+1):
                target_env.append(self.getFromDef(f"target{i}-{j}"))
            self.targets.append(target_env)

        self.bases =[]
        for i in range(1, self.num_envs + 1):
            base_node = self.getFromDef(f"base{i}") 
            self.bases.append(base_node)

        #强化学习空间定义
        self.action_space = []
        self.observation_space = []
        self.state_space = []
        for i in range(self.num_agents):
            self.action_space.append(Box(low=-1.0, high=1.0, shape=(self.num_actions,)))
            self.observation_space.append([self.num_observations])
            self.state_space.append([self.num_states])

        #物理与归一化常量
        self.signal_strength = 80
        self.ps_sensor_mm = {'min': 50, 'max': 85}
        self.tof_sensor_mm = {'min': 0, 'max': 160}
        self.angle_mm = {'min': -np.pi, 'max': np.pi}
        # self.dis_mm = {'min': 0, 'max': 2.24}
        self.pos_x_mm = {'min': self.world_size[0] / -2.0, 'max': self.world_size[0] / 2.0}
        self.pos_y_mm = {'min': self.world_size[1] / -2.0, 'max': self.world_size[1] / 2.0}
        self.max_speed = 0.12

        self.steps = np.zeros((self.num_envs), dtype=np.int32)

        #数据存储
        self.save_data = False
        self.episode_data_log = []
        self.save_path = f"data/trial.csv"

        self.is_state_machine_baseline = False  # 是否使用状态机基准控制逻辑
        self.is_robust_test = True  # 是否不重置目标

        self.index = 0

        self.cleanup()

    def step(self, action):

        control_observations = self.get_observations()
        pde_params = self._map_actions_to_params(action, control_observations)

        self.update_role_and_task_state(pde_params)

        if getattr(self, 'is_state_machine_baseline', False):
            target_wheel_speeds = self._calculate_baseline_wheel_speeds(control_observations)
        else:
            target_wheel_speeds = self._calculate_target_wheel_speeds(pde_params, control_observations)

        self.handle_emitter(target_wheel_speeds)

        for i in range(self.interval):
            if super(Supervisor, self).step(self.timestep//self.interval) == -1:
                exit()
            self.handle_receiver(i)

        observations = self.get_observations()
        state = self.get_state()
        reward, reward_components_info = self.get_reward(pde_params, observations)
        done = self.is_done()
    
        if self.save_data == True:
            self.save_test_data(pde_params, done)
        
        obs_corr = (observations, self.obs_neighbor_num_buf)

        return obs_corr, state, reward, done, reward_components_info

    def save_test_data(self, pde_params, done):
        self.save_path = f'data/test_{self.index}.csv'

        for env_idx in range(self.num_envs):
            current_base_pos = self.base_pos[env_idx] 
            current_target_pos = self.target_pos[env_idx]

            for i in range(self.num_agents):
                l_speed = float(self.last_wheel_speeds[env_idx, i, 0]) if hasattr(self, 'last_wheel_speeds') else 0.0
                r_speed = float(self.last_wheel_speeds[env_idx, i, 1]) if hasattr(self, 'last_wheel_speeds') else 0.0
                # 提取标量值
                agent_log = {
                    'step': int(self.steps[env_idx]),
                    'agent_id': i,
                    'pos_x': float(self.agent_pos[env_idx, i, 0]),
                    'pos_y': float(self.agent_pos[env_idx, i, 1]),
                    'angle': float(self.agent_angle[env_idx, i]),

                    'left_speed': l_speed,
                    'right_speed': r_speed,

                    'role': int(self.roles[env_idx, i]),
                    'is_anchored': int(self.is_anchored[env_idx, i]),
                    'knows_target': int(self.agent_knows_target[env_idx,i]),

                    'assigned_target': int(self.agent_target_map[env_idx,i]),

                    'w_target': float(pde_params['w_target'][env_idx, i ]),
                    'w_center': float(pde_params['w_center'][env_idx,i]),
                    'w_rand': float( pde_params['w_rand'][ env_idx,i]),
                    'k_diff': float( pde_params['k_diff'][env_idx, i] ),

                    'lambda_anchor': float( pde_params['lambda_anchor'][env_idx,i]),
                    'lambda_release': float(pde_params['lambda_release'][ env_idx,i ]),

                    'base_x': float(current_base_pos[0]),
                    'base_y': float(current_base_pos[1]),

                    'chain_connected': int(self.is_chain_connected[env_idx]),
                    'rescued_count': int(self.rescued_targets_count[env_idx]),

                    'is_alive': int( self.alive_agent_buf[ env_idx, i ]),
                    'is_colliding': int( self.last_collision_masks[env_idx, i]
                    ) if hasattr(self,'last_collision_masks') else 0,

                    # 建议保存实验阈值
                    'comm_view': float(self.comm_view),
                    'obs_view': float(self.obs_view),
                    'catch_distance': float( self.catch_distance),
                }

                for t_idx in range(self.num_targets):
                    agent_log[f'target_{t_idx}_x'] = float(current_target_pos[t_idx][0])
                    agent_log[f'target_{t_idx}_y'] = float(current_target_pos[t_idx][1])

                self.episode_data_log.append(agent_log)
        
        if self.steps[0] >= self.episode_length or done[0].all():
            
            if len(self.episode_data_log) > self.num_agents:
                self.save_log_to_csv()
                print(f"Rescue Data saved to {self.save_path} with {len(self.episode_data_log)} rows.")
                
                self.index += 1 

            self.episode_data_log = []

    def save_log_to_csv(self):
        if len(self.episode_data_log) > 0:
            df = pd.DataFrame(self.episode_data_log)
            df.to_csv(self.save_path, index=False)

    def update_role_and_task_state(self, pde_params):
   
        lambda_anchor = pde_params['lambda_anchor']
        lambda_release = pde_params['lambda_release']

        if not hasattr(self, 'is_chain_connected'):
            self.is_chain_connected = np.zeros(self.num_envs, dtype=bool)

        if not hasattr(self, 'is_anchored'):
            self.is_anchored = np.zeros((self.num_envs, self.num_agents), dtype=bool)

        for n in range(self.num_envs):
            alive_mask = self.alive_agent_buf[n].astype(bool)
            dead_mask = ~alive_mask
            self.is_chain_connected[n] = False

            # 清除死亡机器人的残留任务状态
            self.agent_knows_target[n, dead_mask] = False
            self.agent_target_map[n, dead_mask] = -1
            self.is_anchored[n, dead_mask] = False
            self.roles[n, dead_mask] = 0

            # 构建仅包含存活机器人的局部通信图
            dist_matrix = squareform(pdist(self.agent_pos[n, :, :2]))
            communication_adj = (dist_matrix < self.comm_view) & alive_mask[:, None] & alive_mask[None, :]
            np.fill_diagonal(communication_adj, False)

            newly_informed_mask = np.zeros(self.num_agents, dtype=bool)

            # 随机处理多个目标，避免 target_0 始终优先招募
            target_indices = list(range(self.num_targets))
            np.random.shuffle(target_indices)

            for t_idx in target_indices:
                target_pos = self.target_pos[n][t_idx]
                dist_base_target = np.linalg.norm(self.base_pos[n][:2] - target_pos[:2])

                required_responders = 3
                desired_relay_spacing = max(0.8 * float(self.comm_view), 1e-6)
                required_relays = max(1, int(np.ceil(dist_base_target / desired_relay_spacing)) - 1) * 2 - 2
                required_total = required_responders + required_relays

                assigned_mask = alive_mask & self.agent_knows_target[n] & (self.agent_target_map[n] == t_idx)
                current_assigned = int(np.sum(assigned_mask))
                remaining_quota = max(0, required_total - current_assigned)

                if remaining_quota <= 0:
                    continue

                dist_to_target = np.linalg.norm(self.agent_pos[n, :, :2] - target_pos[:2], axis=1)
                sources = []

                # 已经知道该目标的机器人均可作为消息源
                for i in range(self.num_agents):
                    if alive_mask[i] and self.agent_knows_target[n, i] and self.agent_target_map[n, i] == t_idx:
                        sources.append(i)

                # 未分配 Searcher 只有直接进入 obs_view 才能发现目标
                for i in range(self.num_agents):
                    if remaining_quota <= 0:
                        break

                    if not alive_mask[i]:
                        continue

                    if self.agent_knows_target[n, i]:
                        continue

                    if dist_to_target[i] >= self.obs_view:
                        continue

                    self.agent_knows_target[n, i] = True
                    self.agent_target_map[n, i] = t_idx
                    self.is_anchored[n, i] = False
                    newly_informed_mask[i] = True
                    sources.append(i)
                    remaining_quota -= 1

                if not sources or remaining_quota <= 0:
                    continue

                # 从已知目标的机器人出发，沿通信网络逐跳传播目标消息
                queue = list(dict.fromkeys(sources))
                visited = set(queue)

                while queue and remaining_quota > 0:
                    current_agent = queue.pop(0)
                    neighbors = np.where(communication_adj[current_agent])[0]

                    for neighbor in neighbors:
                        if remaining_quota <= 0:
                            break

                        if not alive_mask[neighbor]:
                            continue

                        if neighbor in visited:
                            continue

                        visited.add(neighbor)

                        if self.agent_knows_target[n, neighbor]:
                            continue

                        self.agent_knows_target[n, neighbor] = True
                        self.agent_target_map[n, neighbor] = t_idx
                        self.is_anchored[n, neighbor] = False
                        newly_informed_mask[neighbor] = True
                        queue.append(neighbor)
                        remaining_quota -= 1

            for i in range(self.num_agents):
                if not alive_mask[i]:
                    continue

                if not self.agent_knows_target[n, i]:
                    self.is_anchored[n, i] = False
                    continue

                if newly_informed_mask[i]:
                    continue

                anchor_signal = float(lambda_anchor[n, i] - lambda_release[n, i])
                release_signal = float(lambda_release[n, i] - lambda_anchor[n, i])
                prob_anchor = np.clip(5.0 * max(anchor_signal, 0.0), 0.0, 1.0)
                prob_release = np.clip(2.0 * max(release_signal, 0.0), 0.0, 1.0)

                if not self.is_anchored[n, i]:
                    if np.random.rand() < prob_anchor:
                        self.is_anchored[n, i] = True
                else:
                    if np.random.rand() < prob_release:
                        self.is_anchored[n, i] = False

            for i in range(self.num_agents):
                if not alive_mask[i]:
                    self.agent_knows_target[n, i] = False
                    self.agent_target_map[n, i] = -1
                    self.is_anchored[n, i] = False
                    self.roles[n, i] = 0
                elif not self.agent_knows_target[n, i]:
                    self.is_anchored[n, i] = False
                    self.roles[n, i] = 0
                elif not self.is_anchored[n, i]:
                    self.roles[n, i] = 1
                else:
                    self.roles[n, i] = 2

            for t_idx in range(self.num_targets):
                target_pos = self.target_pos[n][t_idx]

                responder_mask = alive_mask & (self.agent_target_map[n] == t_idx) & (self.roles[n] == 1)
                responder_ids = np.where(responder_mask)[0]

                if len(responder_ids) > 0:
                    responder_distances = np.linalg.norm(self.agent_pos[n, responder_ids, :2] - target_pos[:2], axis=1)
                    responders_near_this_target = int(np.sum(responder_distances < self.catch_distance))
                else:
                    responders_near_this_target = 0

                if responders_near_this_target < 3:
                    continue

                chain_mask = alive_mask & (self.agent_target_map[n] == t_idx) & (self.roles[n] > 0)
                num_nodes = self.num_agents + 2
                base_node = self.num_agents
                target_node = self.num_agents + 1
                graph_adj = np.zeros((num_nodes, num_nodes), dtype=bool)

                agent_adj = (dist_matrix < self.comm_view) & chain_mask[:, None] & chain_mask[None, :]
                np.fill_diagonal(agent_adj, False)
                graph_adj[:self.num_agents, :self.num_agents] = agent_adj

                dist_to_base = np.linalg.norm(self.agent_pos[n, :, :2] - self.base_pos[n][:2], axis=1)
                base_edges = (dist_to_base < self.comm_view) & chain_mask
                graph_adj[base_node, :self.num_agents] = base_edges
                graph_adj[:self.num_agents, base_node] = base_edges

                dist_to_target_all = np.linalg.norm(self.agent_pos[n, :, :2] - target_pos[:2], axis=1)
                target_edges = (dist_to_target_all < self.comm_view) & chain_mask
                graph_adj[target_node, :self.num_agents] = target_edges
                graph_adj[:self.num_agents, target_node] = target_edges

                queue = [base_node]
                visited_nodes = np.zeros(num_nodes, dtype=bool)
                visited_nodes[base_node] = True

                while queue:
                    current_node = queue.pop(0)

                    if current_node == target_node:
                        break

                    neighbors = np.where(graph_adj[current_node])[0]

                    for neighbor in neighbors:
                        if not visited_nodes[neighbor]:
                            visited_nodes[neighbor] = True
                            queue.append(neighbor)

                if not visited_nodes[target_node]:
                    continue

                self.is_chain_connected[n] = True

                if getattr(self, 'is_robust_test', False):
                    continue

                self.rescued_targets_count[n] += 1
                rescued_agents_mask = alive_mask & (self.agent_target_map[n] == t_idx)

                self.agent_knows_target[n, rescued_agents_mask] = False
                self.agent_target_map[n, rescued_agents_mask] = -1
                self.is_anchored[n, rescued_agents_mask] = False
                self.roles[n, rescued_agents_mask] = 0

                new_x = np.random.uniform(0.4, 0.6)
                new_y = np.random.uniform(-0.35, 0.35)
                self.target_pos[n][t_idx] = np.array([new_x, new_y])
                self.targets[n][t_idx].getField('translation').setSFVec3f([new_x, new_y, 0.01])

    def random_pos(self, arena_pos):
 
        x_grid_len = 0.15
        y_grid_len = 0.12
        
        n_rows = math.floor(self.world_size[1] / y_grid_len)
        n_cols = math.floor(self.world_size[0] / x_grid_len)
        
        all_slots =[]
        for r in range(n_rows):
            for c in range(n_cols):
                all_slots.append((r, c))

        base_local_x = -1.2
        base_local_y = 0.0
        base_pos = np.array([base_local_x, base_local_y]) + arena_pos[:2]
        
        base_c = int((base_local_x + self.world_size[0] / 2) / x_grid_len)
        base_r = int((-base_local_y + self.world_size[1] / 2) / y_grid_len)
        if (base_r, base_c) in all_slots:
            all_slots.remove((base_r, base_c))

        target_x_min = 0.4
        target_x_max = 0.5
        
        target_col_min = int((target_x_min + self.world_size[0] / 2) / x_grid_len)
        target_col_max = int((target_x_max + self.world_size[0] / 2) / x_grid_len)
        
        target_slots = [(r, c) for r, c in all_slots if target_col_min <= c <= target_col_max]
        
        if len(target_slots) < self.num_targets:
            print("[Warning] Target zone is too small, falling back to global random.")
            target_slots = all_slots

        target_pos = np.zeros((self.num_targets, 2))
        chosen_target_indices = np.random.choice(len(target_slots), size=self.num_targets, replace=False)
        
        for i, idx in enumerate(chosen_target_indices):
            r_t, c_t = target_slots[idx]
            if (r_t, c_t) in all_slots:
                all_slots.remove((r_t, c_t)) # 占用该点
                
            target_pos[i, 0] = c_t * x_grid_len - self.world_size[0] / 2 + x_grid_len / 2
            target_pos[i, 1] = -r_t * y_grid_len + self.world_size[1] / 2 - y_grid_len / 2

        target_pos[0,0] = 0.825
        target_pos[0,1] = -0.2
        agent_x_min = -0.15
        agent_x_max = 0.35
        
        agent_col_min = int((agent_x_min + self.world_size[0] / 2) / x_grid_len)
        agent_col_max = int((agent_x_max + self.world_size[0] / 2) / x_grid_len)
        
        agent_slots = [(r, c) for r, c in all_slots if agent_col_min <= c <= agent_col_max]
        
        if len(agent_slots) < self.num_agents:
            print("[Warning] Agent spawn zone is too small, falling back to global random.")
            agent_slots = all_slots

        agent_pos = np.zeros((self.num_agents, 2))
        chosen_agent_indices = np.random.choice(len(agent_slots), size=self.num_agents, replace=False)
        
        for i, idx in enumerate(chosen_agent_indices):
            r_a, c_a = agent_slots[idx]
            agent_pos[i, 0] = c_a * x_grid_len - self.world_size[0] / 2 + x_grid_len / 2
            agent_pos[i, 1] = -r_a * y_grid_len + self.world_size[1] / 2 - y_grid_len / 2

        # 加上 Arena 的多环境物理偏移量
        agent_pos += arena_pos[:2]
        target_pos += arena_pos[:2]

        return agent_pos, target_pos, base_pos

    def add_nodes(self, arena_pos):
        target_env_pos = []
        base_env_pos = []
        for i in range(1, self.num_envs+1):
            agent_pos, target_pos, base_pos = self.random_pos(arena_pos[i-1])
            target_env_pos.append(target_pos)
            base_env_pos.append(base_pos)
            for j in range(1, self.num_agents + 1):
                self.importRobot(i, j, agent_pos[j - 1, 0], agent_pos[j - 1, 1], 0.05, random.uniform(-np.pi, np.pi))
            for k in range(1, self.num_targets + 1):
                self.importTarget(i, k, target_pos[k - 1, 0], target_pos[k - 1, 1], 0.01)

            self.importBase(i, base_pos[0], base_pos[1], 0.01)

        return np.array(target_env_pos), np.array(base_env_pos)

    def add_arenas(self):
        column = int(math.ceil(math.sqrt(self.num_envs)))
        wall_len = 0.1
        wall_high = 0.1
        x_arena = self.world_size[0] * column + (column-1)*wall_len
        y_arena = self.world_size[1] * column + (column-1)*wall_len
        self.importArena(x_arena ,y_arena)
        x_begin = -x_arena / 2
        y_begin = -y_arena / 2
        x_tmp = 0
        y_tmp = 0
        for i in range(column-1):
            if i == 0:
                x_tmp = x_begin + self.world_size[0] + wall_len/2
            else:
                x_tmp += (self.world_size[0] + wall_len)
            self.importWall(x_tmp,0,wall_high/2,wall_len,y_arena,wall_high)

        for i in range(column-1):
            if i == 0:
                y_tmp = y_begin + self.world_size[1] + wall_len/2
            else:
                y_tmp += (self.world_size[1] + wall_len)

            for j in range(column):
                if j == 0:
                    x_tmp = x_begin + self.world_size[0]/2
                else:
                    x_tmp += (self.world_size[0] + wall_len)
                self.importWall(x_tmp,y_tmp,wall_high/2,self.world_size[0],wall_len,wall_high)

        arena_pos = np.zeros((column, column, 3), dtype=np.float32)
        x = 0
        y = 0
        for i in range(column):
            if i==0:
                x = x_begin+self.world_size[0]/2
            else:
                x += (self.world_size[0]+wall_len)
            arena_pos[:,i,0] = x

        for i in range(column):
            if i==0:
                y = y_begin+self.world_size[1]/2
            else:
                y += (self.world_size[1]+wall_len)
            arena_pos[i,:,1] = y
        arena_pos_re = arena_pos.reshape(-1,3)
        return arena_pos_re

    def importArena(self, len_x, len_y):
        root = self.getRoot()
        chFd = root.getField("children")
        line_String = """
                        RectangleArena{
                            translation 0 0 0
                            name "rectangle arena"
                            floorSize %f %f        
                        }
                        """ % (len_x, len_y)
        chFd.importMFNodeFromString(-1, line_String)

    def importWall(self,x, y, z, len_x, len_y, len_z):
        root = self.getRoot()
        chFd = root.getField("children")
        line_String = """
                        Solid {
                          translation %f %f %f
                          children [
                            Shape {
                              appearance Appearance {
                                material Material {
                                  diffuseColor 0.8 0.8 0.8
                                }
                              }
                              geometry Box {
                                size %f %f %f
                              }
                            }
                          ]
                          boundingObject Box {
                            size %f %f %f
                          }
                          physics Physics {
                            density 1000  # 使用默认密度
                          }
                        }
                        """ % (x, y, z, len_x, len_y, len_z, len_x, len_y, len_z)
        chFd.importMFNodeFromString(-1, line_String)

    def importRobot(self, arena_id, id, x, y, z, ro):
        root = self.getRoot()
        chFd = root.getField("children")
        line_String = """      
                        DEF epuck%d-%d E-puck{
                            translation %f %f %f
                            rotation 0 0 1 %f
                            name "e-puck%d-%d"
                            controller "epuck_rlcontroller_nonros"
                            supervisor FALSE
                            version "2"
                            emitter_channel %d
                            receiver_channel %d
                        }
                        """ % (arena_id, id, x, y, z, ro, arena_id, id, (arena_id-1)*(self.num_agents+3)+1, (arena_id-1)*(self.num_agents+3)+2)
        chFd.importMFNodeFromString(-1, line_String)

    def importBase(self, arena_id, x, y, z):
        root = self.getRoot()
        chFd = root.getField("children")
        line_string = f"""
            DEF base{arena_id} Solid {{
                translation {x} {y} {z}
                children [
                    Shape {{
                        appearance Appearance {{
                            material Material {{
                                diffuseColor 0 1 0
                            }}
                        }}
                        geometry Box {{
                            size 0.04 0.04 0.01
                        }}
                    }}
                ]
            }}
        """
        chFd.importMFNodeFromString(-1, line_string)

    def importTarget(self, arena_id, id, x, y, z, color=(1, 0, 0)):
        root = self.getRoot()
        chFd = root.getField("children")
        line_string = f"""
            DEF target{arena_id}-{id} Solid {{
                translation {x} {y} {z}
                children [
                    Shape {{
                        appearance Appearance {{
                            material Material {{
                                diffuseColor {color[0]} {color[1]} {color[2]}
                            }}
                        }}
                        geometry Box {{
                            size 0.04 0.04 0.01
                        }}
                    }}
                ]
            }}
        """
        chFd.importMFNodeFromString(-1, line_string)

    def handle_emitter(self, wheel_speeds):
        flat_wheel_speeds = wheel_speeds.reshape(self.num_envs, -1)
        wheel_speeds_list = flat_wheel_speeds.tolist()
        for n in range(self.num_envs):
            env_speeds = wheel_speeds_list[n]
            message = (",".join(map(str, env_speeds))).encode("utf-8")
            self.emitter[n].send(message)
        
    def handle_receiver(self, is_obs):
        message = np.zeros((self.num_envs, self.num_agents,16))

        for i in range(self.num_envs):
            for j in range(self.num_agents):
                if self.receiver[i].getQueueLength() > 0:
                    try:
                        string_message = self.receiver[i].getString().split(',')
                    except AttributeError:
                        string_message = self.receiver[i].getData().decode("utf-8")
                    self.receiver[i].nextPacket()
                    idx = int(string_message[0][1])-1
                    message[i,idx] = np.array(string_message[1:]).astype(np.float32)

        if is_obs == -1:
            return None

        if is_obs == self.interval-1:
            self.message[:] = message[...,8:]

        return message

    def _map_actions_to_params(self, raw_action_batch, observations=None):

        raw_action_torch = to_torch(raw_action_batch)
        if raw_action_torch.is_cuda: 
            raw_action_torch = raw_action_torch.cpu()

        TOTAL_W_BUDGET = 0.13 
        MAX_K = 0.035       
        TEMPERATURE = 5.0   

        advection_logits = (raw_action_torch[..., :3]) * TEMPERATURE
        advection_weights = torch.nn.functional.softmax(advection_logits, dim=-1)

        w_target = advection_weights[..., 0].detach().numpy() * TOTAL_W_BUDGET
        w_center = advection_weights[..., 1].detach().numpy() * TOTAL_W_BUDGET
        w_rand   = advection_weights[..., 2].detach().numpy() * TOTAL_W_BUDGET

        k_raw = torch.clamp(raw_action_torch[..., 3], -1.0, 1.0)
        k_diff = ((k_raw + 1.0) / 2.0).detach().numpy() * MAX_K

        raw_lambda_anchor = torch.clamp(raw_action_torch[..., 4], -1.0, 1.0)
        raw_lambda_release = torch.clamp(raw_action_torch[..., 5], -1.0, 1.0)

        lambda_anchor = ((raw_lambda_anchor + 1.0) / 2.0).detach().numpy()
        lambda_release = ((raw_lambda_release + 1.0) / 2.0).detach().numpy()

        pde_params = {
            'w_target': w_target,       # 对应寻觅目标
            'w_center': w_center,       # 对应中继平衡点/筑链
            'w_rand': w_rand,           # 对应随机搜索
            'k_diff': k_diff,           # 对应物理排斥
            'lambda_anchor': lambda_anchor, 
            'lambda_release': lambda_release  
        }

        return pde_params

    def _calculate_target_wheel_speeds(self, pde_params, observations):
        all_wheel_speeds = np.zeros((self.num_envs, self.num_agents, 2))

        wheel_radius = 0.02
        wheel_sep = 0.05685
        max_wheel_speed = 6.28

        rho_norm = observations[:, :, 2]
        vec_target = observations[:, :, 4:6]   
        vec_center = observations[:, :, 7:9] 
        grad_rho = observations[:, :, 9:11]  

        NOISE_AMPLITUDE = np.pi / 3 

        angle_delta = np.random.uniform(-np.pi/10, np.pi/10, size=(self.num_envs, self.num_agents))
        self.steer_noise = np.clip(self.steer_noise + angle_delta, -NOISE_AMPLITUDE, NOISE_AMPLITUDE)
        
        vec_rand = np.stack([np.cos(self.steer_noise), np.sin(self.steer_noise)], axis=-1)

        def ensure_dim(tensor):
            if tensor.ndim == 2:
                return tensor[:, :, np.newaxis]
            return tensor

        w_target = ensure_dim(pde_params['w_target']) # 救援意图强度
        w_center = ensure_dim(pde_params['w_center']) # 中继驻守意图强度
        w_rand   = ensure_dim(pde_params['w_rand'])   # 随机搜索意图强度
        k_diff   = ensure_dim(pde_params['k_diff'])   # 扩散排斥强度

        V_advection = w_rand * vec_rand + w_target * vec_target + w_center * vec_center
     
        eps_rho = 0.05
        V_interaction = - (k_diff / (rho_norm[:, :, np.newaxis] + eps_rho)) * grad_rho 

        final_v_body = V_advection + V_interaction
        
        if final_v_body.ndim == 4:
            final_v_body = final_v_body.squeeze(2)

        v_body_x = final_v_body[:, :, 0]
        v_body_y = final_v_body[:, :, 1]

        target_angle = np.arctan2(v_body_y, v_body_x)
        
        kp_omega = 2.0 
        omega_max = 4.0
        omega_cmd = np.clip(kp_omega * target_angle, -omega_max, omega_max)

        max_speed_limit = getattr(self, 'max_speed', 0.12)
        direction_alignment = np.clip(np.cos(target_angle), 0.0, 1.0)
        v_cmd = np.clip(np.linalg.norm(final_v_body, axis=-1), 0.0, max_speed_limit) * (0.25 + 0.75 * direction_alignment)

        SAFE_AGENT_DISTANCE = 0.15
        EMERGENCY_DISTANCE = 0.075
        AVOID_OMEGA_MAX = 1.6
        MIN_FORWARD_SCALE = 0.20
        AVOID_FILTER_ALPHA = 0.25

        if not hasattr(self, 'avoid_omega_state'):
            self.avoid_omega_state = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)

        for n in range(self.num_envs):
            alive_mask = self.alive_agent_buf[n].astype(bool)

            if not alive_mask.any():
                continue

            v = v_cmd[n].copy()
            w = omega_cmd[n].copy()
            positions = self.agent_pos[n, :, :2]
            alive_ids = np.where(alive_mask)[0]

            target_avoid_omega = np.zeros(self.num_agents, dtype=np.float32)

            for i in alive_ids:
                other_ids = alive_ids[alive_ids != i]

                if len(other_ids) == 0:
                    continue

                relative_vectors = positions[other_ids] - positions[i]
                neighbor_distances = np.linalg.norm(relative_vectors, axis=1)
                nearest_local_idx = int(np.argmin(neighbor_distances))
                nearest_id = int(other_ids[nearest_local_idx])
                nearest_distance = float(neighbor_distances[nearest_local_idx])

                if nearest_distance >= SAFE_AGENT_DISTANCE:
                    continue

                relative_vector = relative_vectors[nearest_local_idx]
                neighbor_global_angle = float(np.arctan2(relative_vector[1], relative_vector[0]))
                bearing_error = (neighbor_global_angle - float(self.agent_angle[n, i]) + np.pi) % (2.0 * np.pi) - np.pi
                closeness = np.clip((SAFE_AGENT_DISTANCE - nearest_distance) / max(SAFE_AGENT_DISTANCE - EMERGENCY_DISTANCE, 1e-6), 0.0, 1.0)
                frontal_factor = np.clip(np.cos(bearing_error), 0.0, 1.0)
                risk_factor = closeness * (0.35 + 0.65 * frontal_factor)
                side_sign = float(np.sign(np.sin(bearing_error)))

                if abs(side_sign) < 1e-3:
                    side_sign = 1.0 if ((i + nearest_id) % 2 == 0) else -1.0

                target_avoid_omega[i] = -side_sign * AVOID_OMEGA_MAX * risk_factor
                forward_scale = np.clip(1.0 - 0.80 * risk_factor, MIN_FORWARD_SCALE, 1.0)
                v[i] *= forward_scale

                if nearest_distance < EMERGENCY_DISTANCE and frontal_factor > 0.3:
                    v[i] = min(v[i], 0.005)

            self.avoid_omega_state[n] = (1.0 - AVOID_FILTER_ALPHA) * self.avoid_omega_state[n] + AVOID_FILTER_ALPHA * target_avoid_omega
            self.avoid_omega_state[n, ~alive_mask] = 0.0
            w = np.clip(w + self.avoid_omega_state[n], -omega_max, omega_max)

            v_l = v - (w * wheel_sep) / 2.0
            v_r = v + (w * wheel_sep) / 2.0
            omega_l = v_l / wheel_radius
            omega_r = v_r / wheel_radius

            wheel_speeds_env = np.stack([omega_l, omega_r], axis=1)
            wheel_speeds_env = np.clip(wheel_speeds_env, -max_wheel_speed, max_wheel_speed)
            all_wheel_speeds[n] = np.where(alive_mask[:, np.newaxis], wheel_speeds_env, 0.0)

        return all_wheel_speeds
                    
    def cleanup(self) -> None:
        self.states_buf = np.zeros((self.num_envs, self.num_states), dtype=np.float32)
        self.alive_agent_buf = np.ones((self.num_envs, self.num_agents), dtype=np.bool_) 
        self.message = np.zeros((self.num_envs, self.num_agents, 8), dtype=np.float32) 
        self.early_stop = np.zeros((self.num_envs, 1), dtype=np.bool_)
        self.num_collision = np.zeros((self.num_envs, 1), dtype=np.float32)
        self.last_collision_masks = np.zeros((self.num_envs, self.num_agents), dtype=bool)

        self.agent_pos = np.zeros((self.num_envs, self.num_agents, 3), dtype=np.float32)
        self.agent_angle = np.zeros((self.num_envs, self.num_agents), dtype=np.float32) 
        self.agent_vec = np.zeros((self.num_envs, self.num_agents, 3), dtype=np.float32) 

        self.obs_neighbor_num_buf = np.zeros((self.num_envs, self.num_agents, self.obs_neighbor_dim), dtype=np.int32) 

        self.prev_dist_to_goals = [None] * self.num_envs
        
        self.roles = np.zeros((self.num_envs, self.num_agents), dtype=np.int32)
        
        self.steer_noise = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
        self.is_anchored = np.zeros((self.num_envs, self.num_agents), dtype=bool)
        
        self.agent_knows_target = np.zeros((self.num_envs, self.num_agents), dtype=bool)
        self.prev_target_discovered = np.zeros(self.num_envs, dtype=bool) # 记录全队是否已发现
        self.prev_agent_knows_target = np.zeros((self.num_envs, self.num_agents), dtype=bool) # 记录个人是否已发现
        self.rescued_targets_count = np.zeros(self.num_envs, dtype=np.int32)
        self.prev_rescued_targets_count = np.zeros(self.num_envs, dtype=np.int32)
        self.agent_target_map = np.full((self.num_envs, self.num_agents), -1, dtype=int)

        self.robust = 0

        self.steps = np.zeros((self.num_envs), dtype=np.int32)

    def reset_position(self):
        for i in range(self.num_envs):
            agent_pos, target_pos, base_pos = self.random_pos(self.arena_pos[i])
            for j in range(self.num_agents):
                epuck_default_pos = self.robots[i][j].getField('translation').getSFVec3f()
                epuck_default_pos[:2] = agent_pos[j][:2]
                robot_default_rotation = [0,0,1,0]
                robot_default_rotation[3] = random.uniform(-np.pi, np.pi)
                self.robots[i][j].getField('translation').setSFVec3f(epuck_default_pos)
                
                self.robots[i][j].getField('rotation').setSFRotation(robot_default_rotation)
                
            self.target_pos[i] = target_pos
            self.base_pos[i] = base_pos

            for k in range(self.num_targets):
                target_default_pos = self.targets[i][k].getField('translation').getSFVec3f()
                target_default_pos[:2] = target_pos[k][:2]
                self.targets[i][k].getField('translation').setSFVec3f(target_default_pos)
            
            base_default_pos = self.bases[i].getField('translation').getSFVec3f()
            base_default_pos[:2] = base_pos[:2]
            self.bases[i].getField('translation').setSFVec3f(base_default_pos)

        return None

    def get_state(self):
        return self.states_buf

    def initialize_comms(self, emitter_name, receiver_name):
        emitter = []
        receiver = []
        for i in range(self.num_envs):
            emitter_tmp = self.getDevice(emitter_name+"{}".format(i+1))
            receiver_tmp = self.getDevice(receiver_name+"{}".format(i+1))
            receiver_tmp.enable(self.timestep//self.interval)
            emitter.append(emitter_tmp)
            receiver.append(receiver_tmp)
        return emitter, receiver

    def get_observations(self):
        for n in range(self.num_envs):
            for i in range(self.num_agents):
                if self.alive_agent_buf[n, i]:
                    self.agent_pos[n, i] = self.robots[n][i].getField('translation').getSFVec3f()
                    robot_rotation = self.robots[n][i].getField('rotation').getSFRotation()
                    self.agent_angle[n, i] = robot_rotation[3] if robot_rotation[2] > 0 else -robot_rotation[3]
                    vel = self.robots[n][i].getVelocity()
                    self.agent_vec[n, i, 0] = vel[0]
                    self.agent_vec[n, i, 1] = vel[1]

        observation = np.zeros((self.num_envs, self.num_agents, self.num_observations), dtype=np.float32)

        # 确保通信或环境状态正常
        if self.message is not None:
            for n in range(self.num_envs):
                if not self.alive_agent_buf[n].any():
                    continue

                local_pos_env = self.agent_pos[n, :, :2] - self.arena_pos[n][:2]
                global_vel_env = self.agent_vec[n, :, :2] 

                for i in range(self.num_agents):
                    if not self.alive_agent_buf[n, i]:
                        continue

                    my_pos = local_pos_env[i]
                    my_vel_global = global_vel_env[i] 
                    my_angle = self.agent_angle[n, i]
                    my_rot_vec = np.array([np.cos(my_angle), np.sin(my_angle)])
                    
                    my_vel_body_x = my_vel_global[0] * my_rot_vec[0] + my_vel_global[1] * my_rot_vec[1]
                    my_vel_body_y = -my_vel_global[0] * my_rot_vec[1] + my_vel_global[1] * my_rot_vec[0]
                    my_vel_body_norm = utilities.normalize_to_range(
                        np.array([my_vel_body_x, my_vel_body_y]),
                        -self.max_speed, self.max_speed, -1, 1, clip=True
                    )
                    self_state_part = my_vel_body_norm 

                    local_base_pos = (
                        self.base_pos[n][:2]
                        - self.arena_pos[n][:2]
                    )

                    my_t_idx = -1
                    local_target_pos = None
                    min_dist_to_target = float('inf')

                    has_assigned_target = False
                    target_visible_now = False
                    target_info_available = False

                    assigned_t_idx = int(
                        self.agent_target_map[n, i]
                    )

                    has_assigned_target = (
                        self.agent_knows_target[n, i]
                        and 0 <= assigned_t_idx < self.num_targets
                    )

                    if has_assigned_target:
      
                        my_t_idx = assigned_t_idx

                        local_target_pos = (
                            self.target_pos[n][my_t_idx][:2]
                            - self.arena_pos[n][:2]
                        )

                        min_dist_to_target = np.linalg.norm(
                            local_target_pos - my_pos
                        )

                        target_info_available = True

                    else:
        
                        for t_idx in range(self.num_targets):
                            candidate_target_pos = (
                                self.target_pos[n][t_idx][:2]
                                - self.arena_pos[n][:2]
                            )

                            candidate_distance = np.linalg.norm(
                                candidate_target_pos - my_pos
                            )

                            if (
                                candidate_distance < self.obs_view
                                and candidate_distance < min_dist_to_target
                            ):
                                min_dist_to_target = candidate_distance
                                local_target_pos = candidate_target_pos
                                my_t_idx = t_idx
                                target_visible_now = True

                        target_info_available = target_visible_now

                    if self.roles[n, i] == 2:
                        h = 0.22  
                    else:
                        h = self.kde_bandwidth  

                    h_sq = h**2
                    h_wall = 0.05  
                    h_wall_sq = h_wall**2

                    wall_rho = 0.0
                    wall_grad_global = np.zeros(2)

                    potential_walls = [
                        np.array([self.pos_x_mm['min'], my_pos[1]]), 
                        np.array([self.pos_x_mm['max'], my_pos[1]]), 
                        np.array([my_pos[0], self.pos_y_mm['min']]), 
                        np.array([my_pos[0], self.pos_y_mm['max']])
                    ]

                    # 墙体排斥场
                    for wp in potential_walls:
                        diff = wp - my_pos 
                        dist = np.linalg.norm(diff)
                        if dist < 0.2:
                            weight = 2.0 * np.exp(-(dist**2) / (2 * h_wall_sq)) # 权重对齐
                            wall_rho += weight
                            wall_grad_global += (weight / h_wall_sq) * diff

                    # 邻居排斥场
                    all_dists = np.linalg.norm(local_pos_env - my_pos, axis=1)
                    neighbor_mask = (all_dists < self.comm_view) & self.alive_agent_buf[n]
                    neighbor_mask[i] = False
                    
                    neighbor_indices = np.where(neighbor_mask)[0]
                    neighbors = local_pos_env[neighbor_mask]

                    agent_rho = 0.0
                    agent_grad_global = np.zeros(2)
                    if len(neighbors) > 0:
                        diffs = neighbors - my_pos 
                        d_sqs = np.sum(diffs**2, axis=1)
                        weights = np.exp(-d_sqs / (2 * h_sq))
                        agent_rho = np.sum(weights)
                        agent_grad_global = np.sum((weights[:, np.newaxis] / h_sq) * diffs, axis=0)

                    t_grad_g = agent_grad_global + wall_grad_global
                    grad_rho_body_x = t_grad_g[0] * my_rot_vec[0] + t_grad_g[1] * my_rot_vec[1]
                    grad_rho_body_y = -t_grad_g[0] * my_rot_vec[1] + t_grad_g[1] * my_rot_vec[0]

                    MAX_EXPECTED_DENSITY = 2.0
                    local_rho_norm = np.clip((agent_rho + wall_rho) / MAX_EXPECTED_DENSITY, 0.0, 1.0)

                    vec_base_global = local_base_pos - my_pos
                    dist_to_base = np.linalg.norm(vec_base_global)
                    F_base = np.exp(-(dist_to_base**2) / 1.5)

                    F_target = 0.0

                    vec_target_body = np.zeros(
                        2,
                        dtype=np.float32
                    )

                    vec_center_body = np.zeros(
                        2,
                        dtype=np.float32
                    )

                    closer_count = 0.0

                    if target_info_available:
                        vec_target_global = (
                            local_target_pos - my_pos
                        )

                        dist_to_target = np.linalg.norm(
                            vec_target_global
                        )

                        F_target = np.exp(
                            -(dist_to_target ** 2) / 1.5
                        )

                        if dist_to_target > 1e-6:
                            vec_target_norm_global = (
                                vec_target_global / dist_to_target
                            )

                            vec_target_body[0] = (
                                vec_target_norm_global[0] * my_rot_vec[0]
                                + vec_target_norm_global[1] * my_rot_vec[1]
                            )

                            vec_target_body[1] = (
                                -vec_target_norm_global[0] * my_rot_vec[1]
                                + vec_target_norm_global[1] * my_rot_vec[0]
                            )

                    if has_assigned_target:
                        dist_to_target = np.linalg.norm(
                            local_target_pos - my_pos
                        )

                        for nb_idx in range(self.num_agents):
                            if nb_idx == i:
                                continue

                            if not self.alive_agent_buf[n, nb_idx]:
                                continue

                            same_target_group = (
                                self.agent_knows_target[n, nb_idx]
                                and
                                self.agent_target_map[n, nb_idx]
                                == my_t_idx
                            )

                            if not same_target_group:
                                continue

                            nb_pos = local_pos_env[nb_idx]

                            nb_dist_to_target = np.linalg.norm(
                                local_target_pos - nb_pos
                            )

                            if nb_dist_to_target < dist_to_target:
                                closer_count += 1.0

                        axis_vector = (
                            local_target_pos - local_base_pos
                        )

                        axis_length = np.linalg.norm(
                            axis_vector
                        )

                        if axis_length > 1e-6:
                            axis_norm = (
                                axis_vector / axis_length
                            )

                            agent_to_base = (
                                my_pos - local_base_pos
                            )

                            proj_length = np.dot(
                                agent_to_base,
                                axis_norm
                            )

                            proj_length_clamped = np.clip(
                                proj_length,
                                0.0,
                                axis_length
                            )

                            closest_point_on_line = (
                                local_base_pos
                                + proj_length_clamped * axis_norm
                            )

                            vec_to_line = (
                                closest_point_on_line - my_pos
                            )

                            dist_to_line = np.linalg.norm(
                                vec_to_line
                            )

                            if dist_to_line > 1e-6:
                                scale_factor = np.clip(
                                    dist_to_line / 0.10,
                                    0.0,
                                    1.0
                                )

                                vec_center_norm_global = (
                                    vec_to_line / dist_to_line
                                ) * scale_factor
                            else:
                                vec_center_norm_global = np.zeros(
                                    2,
                                    dtype=np.float32
                                )

                            vec_center_body[0] = (
                                vec_center_norm_global[0] * my_rot_vec[0]
                                + vec_center_norm_global[1] * my_rot_vec[1]
                            )

                            vec_center_body[1] = (
                                -vec_center_norm_global[0] * my_rot_vec[1]
                                + vec_center_norm_global[1] * my_rot_vec[0]
                            )

                    macro_field_part = np.array([
                        local_rho_norm,              # (1) 索引 2：归一化局部密度
                        F_target,                    # (1) 索引 3：目标场强
                        vec_target_body[0],          # (1) 索引 4：目标方向 X
                        vec_target_body[1],          # (1) 索引 5：目标方向 Y
                        F_base,                      # (1) 索引 6：基地场强
                        vec_center_body[0],          # (1) 索引 7：中心驻波方向 X
                        vec_center_body[1],          # (1) 索引 8：中心驻波方向 Y
                        np.tanh(grad_rho_body_x),    # (1) 索引 9：密度排斥梯度 X
                        np.tanh(grad_rho_body_y)     # (1) 索引 10：密度排斥梯度 Y
                    ], dtype=np.float32)

                    phase_indicator = np.zeros(3, dtype=np.float32)
                    
                    if not self.agent_knows_target[n, i]:
                        phase_indicator[0] = 1.0  # Phase A: Searcher 
                    elif not self.is_anchored[n, i]:
                        phase_indicator[1] = 1.0  # Phase B: Responder 
                    else:
                        phase_indicator[2] = 1.0  # Phase C: Relay

   
                    RANK_NORM_DENOM = 8.0

                    norm_closer_count = np.clip(
                        closer_count / RANK_NORM_DENOM,
                        0.0,
                        1.0
                    )

                    closer_part = np.array(
                        [norm_closer_count],
                        dtype=np.float32
                    )

                    observation[n, i] = np.concatenate([
                        self_state_part,    # 2 自身状态 (0:2)
                        macro_field_part,   # 9 宏观场 (2:11)
                        phase_indicator,    # 3 真实角色 (11:14)
                        closer_part         # 1 局域排位 (14)
                    ])

            self.states_buf = np.zeros((self.num_envs, self.num_agents, self.num_states), dtype=np.float32)
            all_agent_obs = observation.reshape(self.num_envs, -1) 
            for n in range(self.num_envs):
                for i in range(self.num_agents):
                    self.states_buf[n, i] = all_agent_obs[n]

        return observation

    def get_reward(self, pde_params, observations):

        rew_all = np.zeros((self.num_envs, self.num_agents, 1))
        infos = [{} for _ in range(self.num_envs)]

        REWARD_RESCUE = 100.0         # 完成连通且3人到达 
        REWARD_DISCOVERY = 30.0       # 发现目标的全队激励
        COLLISION_PENALTY = -1.0      # 碰撞惩罚

        SCALE_EXPLORE = 0.3           # 搜索者远离基地的步分 (0.3/step)
        SCALE_RESPOND_VEL = 1.0       # 响应者向目标冲锋的步分 (0.3/step)
        SCALE_RESPOND_DIST = 0.2      # 响应者在目标点的驻留奖励 (最高 1.0/step)
        
        SCALE_RELAY_CENTER = 0.4      # 中继者在驻波中点的对齐步分 (0.4/step)
        SCALE_RELAY_OVERLAP = 1.0     # 中继者在双场交叠/轴线附近的驻留奖 (最高 1.0/step)
        PENALTY_SPACING = 0.3         # 中继者间距惩罚权重
        IDEAL_COMM_DIST = 0.6         

        vel_body_obs   = observations[:, :, 0:2]   # 自身速度 (0:2)
        F_target       = observations[:, :, 3]     # 目标场强 (3)
        vec_target_obs = observations[:, :, 4:6]   # 指向目标的向量 (4:6)
        F_base         = observations[:, :, 6]     # 基地场强 (6)
        vec_center_obs = observations[:, :, 7:9]   # 指向中继驻守轴线的回归向量 (7:9)
        roles          = observations[:, :, 11:14] # Phase Indicator (11:14)

        for n in range(self.num_envs):
            if not self.alive_agent_buf[n].any(): continue

            r_explore, r_respond, r_relay, r_event, r_collision = [np.zeros(self.num_agents) for _ in range(5)]

            mask_searcher = (roles[n, :, 0] == 1)
            mask_responder = (roles[n, :, 1] == 1)
            mask_relay = (roles[n, :, 2] == 1)

            if np.any(mask_searcher):
                r_explore[mask_searcher] += (1.0 - F_base[n][mask_searcher]) * SCALE_EXPLORE
                r_explore[mask_searcher] += pde_params['k_diff'][n][mask_searcher] * 2.0

            if np.any(mask_responder):
                v_t = vec_target_obs[n][mask_responder]
                vel = vel_body_obs[n][mask_responder]
                
                proj_toward_target = np.sum(vel * v_t, axis=1)
                r_respond[mask_responder] += np.maximum(proj_toward_target, 0) * SCALE_RESPOND_VEL
                
                f_t_resp = F_target[n][mask_responder]
                resp_indices = np.where(mask_responder)[0]
                sorted_idx_in_resp = np.argsort(-f_t_resp) 
                
                for rank, order_idx in enumerate(sorted_idx_in_resp):
                    idx = resp_indices[order_idx]
                    if rank < 3:
                        r_respond[idx] += F_target[n][idx] * SCALE_RESPOND_DIST
                    else:
                        r_respond[idx] -= 0.2  

            if np.any(mask_relay):
                v_c = vec_center_obs[n][mask_relay]
                vel = vel_body_obs[n][mask_relay]
                proj_toward_center = np.sum(vel * v_c, axis=1)
                r_relay[mask_relay] += np.maximum(proj_toward_center, 0) * SCALE_RELAY_CENTER
                
                relay_indices = np.where(mask_relay)[0]
                true_dist_to_axis = np.zeros(len(relay_indices))
                base_pos_2d = self.base_pos[n][:2]
                
                for idx_in_mask, idx_global in enumerate(relay_indices):
                    my_pos = self.agent_pos[n, idx_global, :2]
                    my_t_idx = self.agent_target_map[n, idx_global]
                    
                    if my_t_idx >= 0:
                        target_pos_2d = self.target_pos[n][my_t_idx][:2]
                    else:
                        target_pos_2d = self.target_pos[n][0][:2] 
                    
                    axis_vec = target_pos_2d - base_pos_2d
                    axis_len = np.linalg.norm(axis_vec)
                    if axis_len > 1e-6:
                        axis_norm = axis_vec / axis_len
                        agent_to_base = my_pos - base_pos_2d
                        proj = np.dot(agent_to_base, axis_norm)
                        proj_clamped = np.clip(proj, 0, axis_len)
                        closest_pt = base_pos_2d + proj_clamped * axis_norm
                        true_dist_to_axis[idx_in_mask] = np.linalg.norm(my_pos - closest_pt)
                    else:
                        true_dist_to_axis[idx_in_mask] = np.linalg.norm(my_pos - base_pos_2d)
                
                r_relay[mask_relay] += np.exp(-true_dist_to_axis * 5.0) * SCALE_RELAY_OVERLAP 

                for idx in relay_indices:
                    my_t_idx = self.agent_target_map[n, idx]
                    if my_t_idx >= 0:
                        target_pos_2d = self.target_pos[n][my_t_idx][:2]
                    else:
                        target_pos_2d = self.target_pos[n][0][:2]  

                    my_pos = self.agent_pos[n, idx, :2]
                    dist_to_target = np.linalg.norm(my_pos - target_pos_2d)
                    
                    if dist_to_target < 0.5:
                        r_relay[idx] += 0.2 
                    else:
                        current_pos = self.agent_pos[n, :, :2]
                        dist_matrix = squareform(pdist(current_pos))
                        min_dist = np.min(dist_matrix[idx])
                        if min_dist < self.comm_view:
                            r_relay[idx] += 0.1 
                            spacing_error = abs(min_dist - IDEAL_COMM_DIST)
                            r_relay[idx] -= spacing_error * PENALTY_SPACING
                        else:
                            r_relay[idx] -= 0.2

            any_know_now = np.any(self.agent_knows_target[n])

            if any_know_now and not self.prev_target_discovered[n]:
                r_event += REWARD_DISCOVERY  
            
            just_learned_mask = self.agent_knows_target[n] & ~self.prev_agent_knows_target[n]
            if np.any(just_learned_mask):
                r_event[just_learned_mask] += 1.0  

            if self.rescued_targets_count[n] > self.prev_rescued_targets_count[n]:
                r_event += REWARD_RESCUE     

            total_collision_mask = np.zeros(self.num_agents, dtype=bool)

            if self.num_agents > 1:
                current_pos = self.agent_pos[n, :, :2]
                dist_matrix = squareform(pdist(current_pos))
                np.fill_diagonal(dist_matrix, np.inf)
                collision_mask = np.any(dist_matrix < self.collision_distance, axis=1)

                agent_y = self.agent_pos[n, :, 1] - self.arena_pos[n][1]
                half_width = 0.5 * (self.pos_y_mm['max'] - self.pos_y_mm['min'])
                robot_radius = 0.035
                wall_collision = np.abs(agent_y) > (half_width - robot_radius)

                total_collision_mask = collision_mask | wall_collision
                
                penalty_eligible = total_collision_mask & (~mask_responder)
                r_collision[penalty_eligible] = COLLISION_PENALTY

            self.last_collision_masks[n] = total_collision_mask

            total_reward = r_explore + r_respond + r_relay + r_event + r_collision
            rew_all[n, :, 0] = total_reward

            self.prev_target_discovered[n] = any_know_now
            self.prev_agent_knows_target[n] = self.agent_knows_target[n].copy()
            self.prev_rescued_targets_count[n] = self.rescued_targets_count[n]

            searchers_mask = ~self.agent_knows_target[n]
            if np.any(searchers_mask):
                searcher_positions = self.agent_pos[n, searchers_mask, :2] - self.arena_pos[n][:2]
                all_target_dists = []
                for s_pos in searcher_positions:
                    for t_idx in range(self.num_targets):
                        t_pos = self.target_pos[n][t_idx][:2] - self.arena_pos[n][:2]
                        all_target_dists.append(np.linalg.norm(s_pos - t_pos))
                min_dist_to_target = float(np.min(all_target_dists))
            else:
                min_dist_to_target = 9.9

            num_knows = np.sum(self.agent_knows_target[n])
            num_anchored = np.sum(self.is_anchored[n])

            infos[n] = {
                'Reward/Total': np.mean(total_reward),
                'Reward/r_explore': np.mean(r_explore[mask_searcher]) if np.any(mask_searcher) else 0,
                'Reward/r_relay': np.mean(r_relay[mask_relay]) if np.any(mask_relay) else 0,
                'Reward/r_respond': np.mean(r_respond[mask_responder]) if np.any(mask_responder) else 0,
                'Reward/r_event': np.mean(r_event),
                'Reward/r_collision': np.mean(r_collision),
                
                'Metric/Is_Connected': 1.0 if self.is_chain_connected[n] else 0.0,

                'Diag/Min_Dist_To_Target_M': min_dist_to_target,        
                'Diag/Count_Knows_Target': float(num_knows),            
                'Diag/Count_Anchored': float(num_anchored),             
                'Diag/Count_Searchers': float(np.sum(mask_searcher)),   
                'Diag/Count_Responders': float(np.sum(mask_responder)), 
                'Diag/Count_Relays': float(np.sum(mask_relay)),         
                
                'Diag/Avg_Lambda_Anchor': float(np.mean(pde_params['lambda_anchor'][n])),
                'Diag/Avg_Lambda_Release': float(np.mean(pde_params['lambda_release'][n])),
            }

            infos[n].update({
                'Param_Searcher/W_Rand': np.mean(pde_params['w_rand'][n][mask_searcher]) if np.any(mask_searcher) else 0.0,
                'Param_Searcher/W_center': np.mean(pde_params['w_center'][n][mask_searcher]) if np.any(mask_searcher) else 0.0,
                'Param_Searcher/W_Target': np.mean(pde_params['w_target'][n][mask_searcher]) if np.any(mask_searcher) else 0.0,
                'Param_Searcher/K_Diff': np.mean(pde_params['k_diff'][n][mask_searcher]) if np.any(mask_searcher) else 0.0,
            })
            infos[n].update({
                'Param_Responder/W_Target': np.mean(pde_params['w_target'][n][mask_responder]) if np.any(mask_responder) else 0.0,
                'Param_Responder/W_center': np.mean(pde_params['w_center'][n][mask_responder]) if np.any(mask_responder) else 0.0,
                'Param_Responder/W_Rand': np.mean(pde_params['w_rand'][n][mask_responder]) if np.any(mask_responder) else 0.0,
                'Param_Responder/K_Diff': np.mean(pde_params['k_diff'][n][mask_responder]) if np.any(mask_responder) else 0.0,
            })
            infos[n].update({
                'Param_Relay/W_center': np.mean(pde_params['w_center'][n][mask_relay]) if np.any(mask_relay) else 0.0,
                'Param_Relay/W_Target': np.mean(pde_params['w_target'][n][mask_relay]) if np.any(mask_relay) else 0.0,
                'Param_Relay/W_Rand': np.mean(pde_params['w_rand'][n][mask_relay]) if np.any(mask_relay) else 0.0,
                'Param_Relay/K_Diff': np.mean(pde_params['k_diff'][n][mask_relay]) if np.any(mask_relay) else 0.0,
            })

        return rew_all, infos

    def is_done(self):
        self.steps += 1

        dies = np.ones((self.num_envs, self.num_agents, 1), dtype=np.bool_)
        alives = np.zeros((self.num_envs, self.num_agents, 1), dtype=np.bool_)
        alives[..., 0] = (~self.alive_agent_buf) 
        all_agents_done = ~self.alive_agent_buf.any(-1)
        time_is_up = self.steps >= self.episode_length
        cond = all_agents_done | time_is_up | self.early_stop[:, 0]

        return np.where(cond[:, np.newaxis, np.newaxis], dies, alives)

    def reset(self):
        self.cleanup()

        #self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(self.timestep//self.interval)

        self.reset_position()  
        
        for _ in range(self.interval-1):
            super(Supervisor, self).step(self.timestep//self.interval)
            self.handle_receiver(-1)

        return None

    def get_info(self):
        return None
