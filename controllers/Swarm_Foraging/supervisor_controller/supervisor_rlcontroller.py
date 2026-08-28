import random
import numpy as np
from gymnasium.spaces import Box, Tuple as SpaceTuple
import utilities
import math
import torch
from controller import Supervisor
from common.gymnasium_api import episode_end_flags
from common.webots_env import WebotsSupervisorEnv
from scipy.spatial.distance import pdist, squareform
from supervisor_controller.utils.util import to_torch
import matplotlib.pyplot as plt
import os
import pandas as pd


class Epuck2Supervisor(WebotsSupervisorEnv):
    def __init__(self, all_args=None):
        self.args = all_args
        self.num_agents = self.args.num_agents
        self.num_targets = self.args.num_target
        self.num_envs = self.args.n_rollout_threads
        self.timestep = self.args.timestep
        self.interval = self.args.interval

        super().__init__(timestep=self.timestep)

        # self.world_size = np.array([4.0, 2.0])
        self.world_size = np.array([2.0, 1.0])
        
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
        self_dim = 2  # 2(vel) 
        macro_field_dim =  12 # 1(local_rho) + 2(grad_rho) + 2(vec_food) + 2(vec_nest) + 2(vec_info) + 1(dist_food) + 1(dist_nest) + 1(dist_info)
        state_dim = 3 # 1(search) + 1(approach) + 1(nest)
        self.num_observations = self_dim + macro_field_dim + state_dim  

        # 状态空间
        single_num_state = self.num_observations
        self.num_states = self.num_agents * single_num_state
        self.obs_neighbor_dim = self.num_obs_targets + self.num_obs_agents

        self.num_actions = 7  # w_food, w_nest, w_rand, w_info, k_diff, lambda_pick, lambda_drop

        self.kde_bandwidth = 0.05  # KDE核函数带宽 (米)

        self.emitter, self.receiver = self.initialize_comms(
            "emitter", "receiver")

        self.arena_pos = self.add_arenas()
        self.target_pos, self.nest_pos = self.add_nodes(self.arena_pos)

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

        self.nests = []
        for i in range(1, self.num_envs+1):
            nest = self.getFromDef(f"nest{i}")
            self.nests.append(nest)

        #强化学习空间定义
        self.agent_action_spaces = tuple(
            Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)
            for _ in range(self.num_agents)
        )
        self.agent_observation_spaces = tuple(
            Box(low=-np.inf, high=np.inf, shape=(self.num_observations,), dtype=np.float32)
            for _ in range(self.num_agents)
        )
        self.agent_state_spaces = tuple(
            Box(low=-np.inf, high=np.inf, shape=(self.num_states,), dtype=np.float32)
            for _ in range(self.num_agents)
        )
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_envs, self.num_agents, self.num_actions),
            dtype=np.float32,
        )
        self.observation_space = SpaceTuple((
            SpaceTuple((
                Box(-np.inf, np.inf, (self.num_envs, self.num_agents, self.num_observations), dtype=np.float32),
                Box(0, self.num_agents + self.num_targets, (self.num_envs, self.num_agents, self.obs_neighbor_dim), dtype=np.int32),
            )),
            Box(-np.inf, np.inf, (self.num_envs, self.num_agents, self.num_states), dtype=np.float32),
        ))

        #物理与归一化常量
        self.signal_strength = 80
        # self.reward_signal_strength = 1500
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
        self.save_path = f"data/food_test_episode_data_{self.num_agents}agents_run5.csv"

        self.is_state_machine_baseline = False  # 是否使用状态机基准控制逻辑

        self.index = 0

        self.cleanup()

    def step(self, action):

        control_observations = self.get_observations()
        pde_params = self._map_actions_to_params(action, control_observations)

        self.update_carrying_state(pde_params)

        if getattr(self, 'is_state_machine_baseline', False):
            target_wheel_speeds = self._calculate_baseline_wheel_speeds(control_observations)
        else:
            target_wheel_speeds = self._calculate_target_wheel_speeds(pde_params, control_observations)

        self.handle_emitter(target_wheel_speeds)

        for i in range(self.interval):
            if super(Supervisor, self).step(self.timestep//self.interval) == -1:
                exit()
            self.handle_receiver(i)

        if self.steps[0] == 300:
                    # 1. 获取当前所有还活着的机器人索引
                    alive_indices = np.where(self.alive_agent_buf[0] == True)[0]
                    
                    # 2. 确保至少有两个机器人可以被设置为失效
                    if len(alive_indices) >= 4:
                        # 3. 从活着的机器人中随机抽取 2 个
                        # replace=False 确保不会抽到同一个机器人
                        to_kill = np.random.choice(alive_indices, size=0, replace=False)
                        
                        # 4. 执行失效设置
                        self.alive_agent_buf[0, to_kill] = False

        observations = self.get_observations()
        state = self.get_state()
        reward, reward_components_info = self.get_reward(pde_params, observations)
        terminated, truncated = self._get_episode_end_flags()
        done = np.logical_or(terminated, truncated)
    
        if self.save_data == True:
            self.save_test_data(pde_params, done)
        
        observation = ((observations, self.obs_neighbor_num_buf.copy()), state)
        info = {"per_env": reward_components_info}
        return observation, reward, terminated, truncated, info

    def save_test_data(self, pde_params, done):
        self.save_path = f'data/expand_3_16.csv'

        for env_idx in range(self.num_envs):
            current_nest_pos = self.nest_pos[env_idx] 
            current_target_pos = self.target_pos[env_idx]
            current_target_caps = self.target_capacities[env_idx]

            for i in range(self.num_agents):
                l_speed = float(self.last_wheel_speeds[env_idx, i, 0]) if hasattr(self, 'last_wheel_speeds') else 0.0
                r_speed = float(self.last_wheel_speeds[env_idx, i, 1]) if hasattr(self, 'last_wheel_speeds') else 0.0
                # 提取标量值
                agent_log = {
                    'step': self.steps[env_idx],
                    'agent_id': i,
                    # --- 智能体位置 ---
                    'pos_x': self.agent_pos[env_idx, i, 0],
                    'pos_y': self.agent_pos[env_idx, i, 1],
                    'angle': self.agent_angle[env_idx, i],
                    # --- 状态 ---
                    'is_carrying': int(self.agent_carrying_state[env_idx, i]),
                    # --- 权重参数 ---
                    'w_food': float(pde_params['w_food'][env_idx, i]),
                    'w_nest': float(pde_params['w_nest'][env_idx, i]),
                    'w_rand': float(pde_params['w_rand'][env_idx, i]),
                    'w_info': float(pde_params['w_info'][env_idx, i]), 
                    'k_diff': float(pde_params['k_diff'][env_idx, i]),
                    # --- 概率 ---
                    'lambda_pick': float(pde_params['lambda_pick'][env_idx, i]),
                    'lambda_drop': float(pde_params['lambda_drop'][env_idx, i]),

                    'left_speed': l_speed,
                    'right_speed': r_speed,

                    # --- 环境固定特征 (巢穴) ---
                    'nest_x': float(current_nest_pos[0]),
                    'nest_y': float(current_nest_pos[1]),
                    
                    # 记录这一步是否发生了捡起(1)或没有(0)
                    'pickup_event': int(self.last_pickup_masks[i]), 
                    # 记录这一步是否成功运回了食物(1)或没有(0)
                    'delivery_event': int(self.last_delivery_masks[i]),
                    # 记录这一步是否撞车了(1)或没有(0)
                    'is_colliding': int(self.last_collision_masks[i]),

                    'is_alive': int(self.alive_agent_buf[env_idx, i])
                }

                # --- 动态添加所有食物的位置和剩余量 ---
                for t_idx in range(self.num_targets):
                    agent_log[f'target_{t_idx}_x'] = float(current_target_pos[t_idx][0])
                    agent_log[f'target_{t_idx}_y'] = float(current_target_pos[t_idx][1])
                    agent_log[f'target_{t_idx}_cap'] = float(current_target_caps[t_idx])

                self.episode_data_log.append(agent_log)
        
        if self.steps[0] >= self.episode_length or done[0].all():
            
            if len(self.episode_data_log) > self.num_agents:
                self.save_log_to_csv()
                print(f"Data saved to {self.save_path} with {len(self.episode_data_log)} rows.")
                
                self.index += 1 

            self.episode_data_log =[]

    def save_log_to_csv(self):
        if len(self.episode_data_log) > 0:
            df = pd.DataFrame(self.episode_data_log)
            df.to_csv(self.save_path, index=False)

    def update_carrying_state(self, pde_params):
        """
        更新智能体的携粮状态，引入随机概率采样以保留强化学习的动作探索梯度。
        """
        lambda_pick = pde_params['lambda_pick']
        lambda_drop = pde_params['lambda_drop']

        if self.target_capacities.dtype != np.float32 and self.target_capacities.dtype != np.float64:
            self.target_capacities = self.target_capacities.astype(np.float32)

        for n in range(self.num_envs):
            for i in range(self.num_agents):
                if not self.alive_agent_buf[n, i]: 
                    continue
                
                current_pos = self.agent_pos[n, i, :2]
                is_carrying = self.agent_carrying_state[n, i]

                # --- 1. 未携粮状态：尝试拾取 ---
                if not is_carrying:
                    prob_pick = lambda_pick[n, i]
                    
                    for t_idx in range(self.num_targets):
                        if self.target_capacities[n, t_idx] > 0:
                            t_pos = self.target_pos[n][t_idx]
                            dist = np.linalg.norm(current_pos - t_pos)

                            if dist < self.catch_distance:
                                if np.random.rand() < prob_pick:
                                    self.agent_carrying_state[n, i] = True 
                                    self.target_capacities[n, t_idx] -= 0.5
                                    self.agent_last_food_pos[n, i] = t_pos
                                break 

                # --- 2. 已携粮状态：尝试放下 ---
                else:
                    prob_drop = lambda_drop[n, i]
                    
                    nest_pos = self.nest_pos[n]
                    dist = np.linalg.norm(current_pos - nest_pos)
                    
                    if dist < self.catch_distance:
                        if np.random.rand() < prob_drop:
                            self.agent_carrying_state[n, i] = False 
        
        # print(self.target_capacities[0, 0], self.target_capacities[0, 1])

    def random_pos(self, arena_pos):
        x_grid_len = 0.15
        y_grid_len = 0.12
        
        # 1. 计算全地图的行数和列数
        n_rows = math.floor(self.world_size[1] / y_grid_len)
        n_cols = math.floor(self.world_size[0] / x_grid_len)
        
        # 生成全场所有可用的网格索引池
        all_slots = []
        for r in range(n_rows):
            for c in range(n_cols):
                all_slots.append((r, c))

        # --- 2. 设置 Nest 位置 (固定在中心) ---
        nest_pos = np.array([0.0, 0.0]) + arena_pos[:2]
        # 计算 Nest 所在的网格索引，以便从池中剔除，防止机器人叠在巢穴上
        nest_grid_r = n_rows // 2
        nest_grid_c = n_cols // 2
        if (nest_grid_r, nest_grid_c) in all_slots:
            all_slots.remove((nest_grid_r, nest_grid_c))

        # --- 3. 生成 Target (食物) 位置 ---
        # 依然保持在四个角落生成，这样比较符合觅食任务逻辑
        corner_rows = max(1, n_rows // 4)  
        corner_cols = max(1, n_cols // 5)  
        
        # 极大缩小 Margin (留白)，只留出 1 个格子的防撞距离，把区域彻底推向四周
        margin_y = 1  
        margin_x = 1

        # 0:左上, 1:右上, 2:左下, 3:右下
        # 统一格式: (r_start, r_end, c_start, c_end)
        zones = {
            # 左上
            0: (margin_y, margin_y + corner_rows, 
                margin_x, margin_x + corner_cols),
                
            # 右上
            1: (margin_y, margin_y + corner_rows, 
                n_cols - margin_x - corner_cols, n_cols - margin_x),
                
            # 左下
            2: (n_rows - margin_y - corner_rows, n_rows - margin_y, 
                margin_x, margin_x + corner_cols),
                
            # 右下
            3: (n_rows - margin_y - corner_rows, n_rows - margin_y, 
                n_cols - margin_x - corner_cols, n_cols - margin_x)
        }

        target_pos = np.zeros((self.num_targets, 2))
        occupied_by_targets =[]

        # 均匀分配四个角落
        base_corners = [0, 1, 2, 3]
        pool = base_corners * math.ceil(self.num_targets / 4)
        chosen_target_zones = np.random.choice(pool, size=self.num_targets, replace=False)

        for i in range(self.num_targets):
            zone_id = chosen_target_zones[i]
            z = zones[zone_id]
            
            # 在指定的绝对角落 zone 内找可用点
            zone_slots = [(r, c) for r, c in all_slots if z[0] <= r < z[1] and z[2] <= c < z[3]]
            
            # 安全校验：万一某个角落被占满了，全局随机找空位
            if len(zone_slots) == 0:
                print(f"[Warning] Corner Zone {zone_id} is full! Falling back to any available slot.")
                if len(all_slots) == 0:
                    raise ValueError("No available slots anywhere on the map!")
                zone_slots = all_slots
                
            idx = np.random.choice(len(zone_slots))
            r_t, c_t = zone_slots[idx]
            
            # 记录位置并从总池中移除
            occupied_by_targets.append((r_t, c_t))
            if (r_t, c_t) in all_slots:
                all_slots.remove((r_t, c_t))
            
            target_pos[i, 0] = c_t * x_grid_len - self.world_size[0] / 2 + x_grid_len / 2
            target_pos[i, 1] = -r_t * y_grid_len + self.world_size[1] / 2 - y_grid_len / 2


        # --- 4. 生成 Agent (机器人) 位置 (全场随机) ---
        if len(all_slots) < self.num_agents:
            raise ValueError("地图太小，无法容纳所有机器人和食物！")

        chosen_agent_indices = np.random.choice(len(all_slots), size=self.num_agents, replace=False)
        agent_pos = np.zeros((self.num_agents, 2))
        
        for i, idx in enumerate(chosen_agent_indices):
            r_a, c_a = all_slots[idx]
            # 将网格索引转换为 Webots 坐标
            agent_pos[i, 0] = c_a * x_grid_len - self.world_size[0] / 2 + x_grid_len / 2
            agent_pos[i, 1] = -r_a * y_grid_len + self.world_size[1] / 2 - y_grid_len / 2

        # 加上 Arena 的物理偏移量
        agent_pos += arena_pos[:2]
        target_pos += arena_pos[:2]

        return agent_pos, target_pos, nest_pos

    def get_zone_slots_static(self, zone_idx, zones):
        r_min, r_max, c_min, c_max = zones[zone_idx]
        slots = []
        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                slots.append((r, c))
        return slots

    def add_nodes(self, arena_pos):
        target_env_pos = []
        nest_env_pos = []
        for i in range(1, self.num_envs+1):
            agent_pos, target_pos, nest_pos = self.random_pos(arena_pos[i-1])
            target_env_pos.append(target_pos)
            nest_env_pos.append(nest_pos)
            for j in range(1, self.num_agents + 1):
                self.importRobot(i, j, agent_pos[j - 1, 0], agent_pos[j - 1, 1], 0.05, random.uniform(-np.pi, np.pi))
            for k in range(1, self.num_targets + 1):
                self.importTarget(i, k, target_pos[k - 1, 0], target_pos[k - 1, 1], 0.01)

            self.importNest(i, nest_pos[0], nest_pos[1], 0.01)

        return np.array(target_env_pos), np.array(nest_env_pos)
    
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
                            controller "epuck_controller"
                            supervisor FALSE
                            version "2"
                            emitter_channel %d
                            receiver_channel %d
                        }
                        """ % (arena_id, id, x, y, z, ro, arena_id, id, (arena_id-1)*(self.num_agents+3)+1, (arena_id-1)*(self.num_agents+3)+2)
        chFd.importMFNodeFromString(-1, line_String)

    def importNest(self, arena_id, x, y, z):
        root = self.getRoot()
        chFd = root.getField("children")
        line_string = f"""
            DEF nest{arena_id} Solid {{
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
                            size 0.01 0.01 0.01
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

    def _map_actions_to_params(self,raw_action_batch,observations=None,):
        raw_action_torch = to_torch(raw_action_batch).detach().to(
            device="cpu", dtype=torch.float32
        )

        TOTAL_W_BUDGET = 0.13
        MAX_K = 0.085
        TEMPERATURE = 5.0

        # 小权重抑制强度
        W_SUPPRESS_THRESHOLD = 0.05

        # --------------------------------------------------
        # 1. 四个竞争性对流权重
        # --------------------------------------------------
        advection_logits = (
            raw_action_torch[..., :4]
            * TEMPERATURE
        )

        original_weights = torch.nn.functional.softmax(
            advection_logits,
            dim=-1,
        )

        # 从每个权重中减去固定阈值
        suppressed_weights = torch.relu(
            original_weights
            - W_SUPPRESS_THRESHOLD
        )

        suppressed_sum = suppressed_weights.sum(
            dim=-1,
            keepdim=True,
        )

        normalized_sparse_weights = (
            suppressed_weights
            / suppressed_sum.clamp_min(1e-8)
        )

        # 安全回退
        advection_weights = torch.where(
            suppressed_sum > 1e-8,
            normalized_sparse_weights,
            original_weights,
        )

        w_food = (
            advection_weights[..., 0]
            .detach()
            .numpy()
            * TOTAL_W_BUDGET
        )

        w_nest = (
            advection_weights[..., 1]
            .detach()
            .numpy()
            * TOTAL_W_BUDGET
        )

        w_rand = (
            advection_weights[..., 2]
            .detach()
            .numpy()
            * TOTAL_W_BUDGET
        )

        w_info = (
            advection_weights[..., 3]
            .detach()
            .numpy()
            * TOTAL_W_BUDGET
        )

        # --------------------------------------------------
        # 2. k_diff
        # --------------------------------------------------
        k_raw = torch.clamp(
            raw_action_torch[..., 4],
            -1.0,
            1.0,
        )

        k_diff = (
            (k_raw + 1.0)
            / 2.0
        ).detach().numpy() * MAX_K

        # --------------------------------------------------
        # 3. lambda_pick
        # --------------------------------------------------
        raw_lambda_pick = torch.clamp(
            raw_action_torch[..., 5],
            -1.0,
            1.0,
        )

        lambda_pick = (
            (raw_lambda_pick + 1.0)
            / 2.0
        ).detach().numpy()

        # --------------------------------------------------
        # 4. lambda_drop
        # --------------------------------------------------
        raw_lambda_drop = torch.clamp(
            raw_action_torch[..., 6],
            -1.0,
            1.0,
        )

        lambda_drop = (
            (raw_lambda_drop + 1.0)
            / 2.0
        ).detach().numpy()

        return {
            "w_food": w_food,
            "w_nest": w_nest,
            "w_rand": w_rand,
            "w_info": w_info,
            "k_diff": k_diff,
            "lambda_pick": lambda_pick,
            "lambda_drop": lambda_drop,
        }

    def _calculate_target_wheel_speeds(self, pde_params, observations):
        all_wheel_speeds = np.zeros((self.num_envs, self.num_agents, 2))

        wheel_radius = 0.02
        wheel_sep = 0.05685
        max_wheel_speed = 6.28

        # --- 1. 探索噪声扰动流场计算 ---
        NOISE_AMPLITUDE = np.pi / 2.5  
        random_steer_angles = np.random.uniform(-NOISE_AMPLITUDE, NOISE_AMPLITUDE, size=(self.num_envs, self.num_agents))
        vec_rand = np.stack([np.cos(random_steer_angles), np.sin(random_steer_angles)], axis=-1)

        # --- 2. 从观测中提取特征 ---
        rho_norm = observations[:, :, 2]
        grad_rho = observations[:, :, 3:5]
        vec_food = observations[:, :, 5:7]
        vec_nest = observations[:, :, 8:10] 
        vec_info = observations[:, :, 11:13] 

        def ensure_dim(tensor):
            if tensor.ndim == 2:
                return tensor[:, :, np.newaxis]
            return tensor

        w_food = ensure_dim(pde_params['w_food'])
        w_nest = ensure_dim(pde_params['w_nest'])
        w_rand = ensure_dim(pde_params['w_rand'])
        w_info = ensure_dim(pde_params['w_info'])
        k_diff = ensure_dim(pde_params['k_diff'])

        # --- 3. 期望速度场合成 ---
        V_advection = (w_food * vec_food + w_nest * vec_nest + w_info * vec_info)
        V_diffusion = w_rand * vec_rand
    
        eps_rho = 0.05
        V_interaction = - (k_diff / (rho_norm[:, :, np.newaxis] + eps_rho)) * grad_rho 

        # 4. 机体合成速度
        final_v_body = V_advection + V_diffusion + V_interaction
        
        if final_v_body.ndim == 4:
            final_v_body = final_v_body.squeeze(2)

        v_body_x = final_v_body[:, :, 0]
        v_body_y = final_v_body[:, :, 1]
        
        # --- 5. 轮速运动学控制逻辑 ---
        target_angle = np.arctan2(v_body_y, v_body_x)
        
        kp_omega = 2.0 
        omega_max = 4.0
        omega_cmd = np.clip(kp_omega * target_angle, -omega_max, omega_max)

        max_speed_limit = getattr(self, 'max_speed', 0.12)
        v_cmd = np.clip(np.linalg.norm(final_v_body, axis=-1), 0.0, max_speed_limit)

        for n in range(self.num_envs):
            if not self.alive_agent_buf[n].any():
                continue

            v = v_cmd[n]      
            w = omega_cmd[n]  

            v_l = v - (w * wheel_sep) / 2.0
            v_r = v + (w * wheel_sep) / 2.0
            
            omega_l = v_l / wheel_radius
            omega_r = v_r / wheel_radius
        
            wheel_speeds_env = np.stack([omega_l, omega_r], axis=1)
            wheel_speeds_env = np.clip(wheel_speeds_env, -max_wheel_speed, max_wheel_speed)
            
            all_wheel_speeds[n] = np.where(self.alive_agent_buf[n, :, np.newaxis], wheel_speeds_env, 0)
            
        return all_wheel_speeds

    def cleanup(self) -> None:
        self.states_buf = np.zeros((self.num_envs, self.num_states), dtype=np.float32)
        self.alive_agent_buf = np.ones((self.num_envs, self.num_agents), dtype=np.bool_) 
        self.message = np.zeros((self.num_envs, self.num_agents, 8), dtype=np.float32) 
        self.early_stop = np.zeros((self.num_envs, 1), dtype=np.bool_)
        self.num_collision = np.zeros((self.num_envs, 1), dtype=np.float32)

        self.agent_pos = np.zeros((self.num_envs, self.num_agents, 3), dtype=np.float32)
        self.agent_angle = np.zeros((self.num_envs, self.num_agents), dtype=np.float32) 
        self.agent_vec = np.zeros((self.num_envs, self.num_agents, 3), dtype=np.float32) 

        self.obs_neighbor_num_buf = np.zeros((self.num_envs, self.num_agents, self.obs_neighbor_dim), dtype=np.int32) 

        self.prev_dist_to_goals = [None] * self.num_envs
        self.prev_carrying_state = [np.zeros(self.num_agents, dtype=bool) for _ in range(self.num_envs)]
        self.agent_carrying_state = np.zeros((self.num_envs, self.num_agents), dtype=bool)

        self.agent_last_food_pos = np.zeros((self.num_envs, self.num_agents, 2), dtype=np.float32)
        INITIAL_CAPACITY = 5.0 
        self.target_capacities = np.full((self.num_envs, self.num_targets), INITIAL_CAPACITY, dtype=np.float32)

        self.steps = np.zeros((self.num_envs), dtype=np.int32)

    def reset_position(self):
        for i in range(self.num_envs):
            agent_pos, target_pos, nest_pos = self.random_pos(self.arena_pos[i])
            for j in range(self.num_agents):
                epuck_default_pos = self.robots[i][j].getField('translation').getSFVec3f()
                epuck_default_pos[:2] = agent_pos[j][:2]
                robot_default_rotation = [0,0,1,0]
                robot_default_rotation[3] = random.uniform(-np.pi, np.pi)
                self.robots[i][j].getField('translation').setSFVec3f(epuck_default_pos)
                self.robots[i][j].getField('rotation').setSFRotation(robot_default_rotation)
                
            self.target_pos[i] = target_pos

            for k in range(self.num_targets):
                target_default_pos = self.targets[i][k].getField('translation').getSFVec3f()
                target_default_pos[:2] = target_pos[k][:2]
                self.targets[i][k].getField('translation').setSFVec3f(target_default_pos)

            self.nest_pos[i] = nest_pos
            nest_default_pos = self.nests[i].getField('translation').getSFVec3f()
            nest_default_pos[:2] = nest_pos[:2]
            self.nests[i].getField('translation').setSFVec3f(nest_default_pos)

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

        if self.message is not None:
            for n in range(self.num_envs):
                if not self.alive_agent_buf[n].any():
                    continue

                local_pos_env = self.agent_pos[n, :, :2] - self.arena_pos[n][:2]
                global_vel_env = self.agent_vec[n, :, :2] 
                all_targets_pos = self.target_pos[n] - self.arena_pos[n][:2] 
                local_nest_pos = self.nest_pos[n] - self.arena_pos[n][:2]

                sensors_env = self.message[n].copy()
                carrying_states = self.agent_carrying_state[n] 

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

                    my_pos_norm_x = utilities.normalize_to_range(my_pos[0], self.pos_x_mm['min'], self.pos_x_mm['max'], -1, 1, clip=True)
                    my_pos_norm_y = utilities.normalize_to_range(my_pos[1], self.pos_y_mm['min'], self.pos_y_mm['max'], -1, 1, clip=True)
                    my_pos_norm = np.array([my_pos_norm_x, my_pos_norm_y])
                    
                    ir_sensors = sensors_env[i, :8]
                    ir_sensors_norm = utilities.normalize_to_range(ir_sensors, self.ps_sensor_mm['min'], self.ps_sensor_mm['max'], 0, 1, clip=True)

                    self_state_part = np.concatenate([
                        # my_pos_norm,        #（2）位置   
                        # my_rot_vec,         #（2）朝向向量
                        my_vel_body_norm,   #（2）速度（机体坐标）
                        # ir_sensors_norm     #（8）红外传感器归一化
                    ]) 

                    h_agent = self.kde_bandwidth  
                    h_agent_sq = h_agent**2
                    
                    all_dists = np.linalg.norm(local_pos_env - my_pos, axis=1)
                    neighbor_mask = (all_dists < self.comm_view) & self.alive_agent_buf[n]
                    neighbor_mask[i] = False
                    
                    neighbor_indices = np.where(neighbor_mask)[0] 
                    neighbors = local_pos_env[neighbor_mask]

                    agent_rho = 0.0
                    agent_grad_global = np.zeros(2)
                    if len(neighbors) > 0:
                        diffs_agent = neighbors - my_pos 
                        dists_sq_agent = np.sum(diffs_agent**2, axis=1)
                        
                        # 计算智能体间的密度贡献
                        weights_agent = np.exp(-dists_sq_agent / (2 * h_agent_sq))
                        agent_rho = np.sum(weights_agent)
                        
                        # 计算智能体间的梯度贡献
                        agent_grad_global = np.sum((weights_agent[:, np.newaxis] / h_agent_sq) * diffs_agent, axis=0)

                    # --- 2. 障碍物与墙壁的避障 SDF 计算 ---
                    h_wall = 0.05
                    h_wall_sq = h_wall**2
                    wall_rho = 0.0
                    wall_grad_global = np.zeros(2)

                    x_min, x_max = self.pos_x_mm['min'], self.pos_x_mm['max']
                    y_min, y_max = self.pos_y_mm['min'], self.pos_y_mm['max']
                    
                    potential_walls = [
                        np.array([x_min, my_pos[1]]), # 左墙最近点
                        np.array([x_max, my_pos[1]]), # 右墙最近点
                        np.array([my_pos[0], y_min]), # 下墙最近点
                        np.array([my_pos[0], y_max])  # 上墙最近点
                    ]
                    
                    # # 🚀 扩展支持：如果觅食地图中设计了内部障碍物（如中继隔墙），自动计算最近点并加入排斥
                    # if hasattr(self, 'internal_walls'):
                    #     for seg_start, seg_end in self.internal_walls[n]:
                    #         cp = self.get_closest_point_on_segment(my_pos, seg_start, seg_end)
                    #         potential_walls.append(cp)

                    # 计算来自障碍物墙壁的排斥场密度与梯度
                    for wp in potential_walls:
                        diff_wall = wp - my_pos  # 向量由机器人指向墙壁
                        dist_wall = np.linalg.norm(diff_wall)
                        if dist_wall < 0.2:
                            # 增加近距离排斥权重
                            weight_wall = 2.0 * np.exp(-(dist_wall**2) / (2 * h_wall_sq))
                            wall_rho += weight_wall
                            wall_grad_global += (weight_wall / h_wall_sq) * diff_wall

                    # --- 3. 物理场汇总与机体坐标系投影转换 ---
                    # 汇总总密度梯度 (智能体斥力 + 墙体斥力)
                    local_grad_rho_global = agent_grad_global + wall_grad_global
                    
                    # 投影转换到机体坐标系 (Body Frame)
                    grad_rho_body_x = local_grad_rho_global[0] * my_rot_vec[0] + local_grad_rho_global[1] * my_rot_vec[1]
                    grad_rho_body_y = -local_grad_rho_global[0] * my_rot_vec[1] + local_grad_rho_global[1] * my_rot_vec[0]

                    # 汇总总局部密度并进行裁剪，防止数值爆炸
                    MAX_EXPECTED_DENSITY = 2.0
                    local_rho_norm = np.clip((agent_rho + wall_rho) / MAX_EXPECTED_DENSITY, 0.0, 1.0)

                    # 计算 vec_food: 指向最近可见有效食物的单位向量
                    vec_food_body = np.zeros(2)
                    val_dist_food = 0.0 
                    
                    valid_target_indices = np.where(self.target_capacities[n] > 0)[0]
                    if len(valid_target_indices) > 0:
                        valid_targets_pos = all_targets_pos[valid_target_indices]
                        dists_to_targets = np.linalg.norm(valid_targets_pos - my_pos, axis=1)
                        visible_mask = dists_to_targets < self.obs_view
                        visible_indices = np.where(visible_mask)[0]
                        
                        if len(visible_indices) > 0:
                            nearest_idx = visible_indices[np.argmin(dists_to_targets[visible_indices])]
                            real_target_pos = valid_targets_pos[nearest_idx]
                            target_vec = real_target_pos - my_pos
                            raw_dist_to_food = np.linalg.norm(target_vec)
                            
                            if raw_dist_to_food > 1e-6:
                                val_dist_food = np.exp(-raw_dist_to_food)
                                
                                vec_food_global = target_vec / raw_dist_to_food
                                vec_food_body[0] = vec_food_global[0] * my_rot_vec[0] + vec_food_global[1] * my_rot_vec[1]
                                vec_food_body[1] = -vec_food_global[0] * my_rot_vec[1] + vec_food_global[1] * my_rot_vec[0]

                    # 计算 vec_nest: 指向巢穴的单位向量
                    vec_nest_body = np.zeros(2)
                    nest_vec = local_nest_pos - my_pos
                    raw_dist_to_nest = np.linalg.norm(nest_vec) 
                    val_dist_nest = np.exp(-raw_dist_to_nest) 
                    
                    if raw_dist_to_nest > 1e-6:
                        vec_nest_global = nest_vec / raw_dist_to_nest
                        vec_nest_body[0] = vec_nest_global[0] * my_rot_vec[0] + vec_nest_global[1] * my_rot_vec[1]
                        vec_nest_body[1] = -vec_nest_global[0] * my_rot_vec[1] + vec_nest_global[1] * my_rot_vec[0]

                    # 计算 vec_info: 指向携带食物的邻居共享信息的单位向量
                    vec_info_body = np.zeros(2)
                    val_dist_info = 0.0
                    if len(neighbor_indices) > 0:
                        neighbors_carrying = carrying_states[neighbor_indices]
                        informative_neighbors_idx = neighbor_indices[neighbors_carrying] 
                        
                        if len(informative_neighbors_idx) > 0:
                            shared_food_positions = self.agent_last_food_pos[n, informative_neighbors_idx]
                            diff_vecs = shared_food_positions - my_pos
                            dists = np.linalg.norm(diff_vecs, axis=1)
                            
                            valid_mask = dists > 1e-6
                            if np.any(valid_mask):
                                norm_vecs = diff_vecs[valid_mask] / dists[valid_mask][:, np.newaxis]
                                
                                avg_global_vec = np.mean(norm_vecs, axis=0)
                                avg_global_vec = avg_global_vec / (np.linalg.norm(avg_global_vec) + 1e-6)
                                val_dist_info = 1.0 
                                vec_info_body[0] = avg_global_vec[0] * my_rot_vec[0] + avg_global_vec[1] * my_rot_vec[1]
                                vec_info_body[1] = -avg_global_vec[0] * my_rot_vec[1] + avg_global_vec[1] * my_rot_vec[0]

                    macro_field_part = np.array([
                        local_rho_norm,              # (1) 局部密度
                        np.tanh(grad_rho_body_x),    # (1) 密度梯度 X (机体) - tanh限制到[-1,1]
                        np.tanh(grad_rho_body_y),    # (1) 密度梯度 Y (机体) - tanh限制到[-1,1]
                        vec_food_body[0],            # (1) 食物方向 X
                        vec_food_body[1],            # (1) 食物方向 Y
                        val_dist_food,               # (1) 食物距离值
                        vec_nest_body[0],            # (1) 巢穴方向 X
                        vec_nest_body[1],            # (1) 巢穴方向 Y
                        val_dist_nest,               # (1) 巢穴距离值
                        vec_info_body[0],            # (1) 信息方向 X
                        vec_info_body[1],            # (1) 信息方向 Y
                        val_dist_info                # (1) 信息强度值
                    ], dtype=np.float32)

                    phase_indicator = np.zeros(3, dtype=np.float32)

                    is_carrying_bool = carrying_states[i]
                    sees_food_bool = (val_dist_food > 0.05) 
                    if is_carrying_bool:
                        # Phase C: Carry
                        phase_indicator[2] = 1.0
                    elif sees_food_bool:
                        # Phase B: Approach (没搬运，且看到食物)
                        phase_indicator[1] = 1.0
                    else:
                        # Phase A: Search (没搬运，且没看到食物)
                        phase_indicator[0] = 1.0

                    observation[n, i] = np.concatenate([
                        self_state_part, 
                        macro_field_part,
                        phase_indicator
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

        REWARD_EVENT = 200.0          
        SCALE_ALIGNMENT = 1.5        
        SCALE_SEARCH_VEL = 0.3        
        SCALE_SEARCH_DIFF = 0.05       
        SCALE_SOCIAL = 1.5            
        COLLISION_PENALTY = -2.0      

        vel_body_obs = observations[:, :, 0:2]
        vel_speed = np.linalg.norm(vel_body_obs, axis=2) + 1e-6 
        
        vec_food_obs = observations[:, :, 5:7]   
        val_dist_food = observations[:, :, 7]    
        
        vec_nest_obs = observations[:, :, 8:10]  
        val_dist_nest = observations[:, :, 10]   

        vec_info_obs = observations[:, :, 11:13] 
        val_dist_info = observations[:, :, 13]

        sees_food_mask = val_dist_food > 0.05    
        has_info_mask = val_dist_info > 0.1

        k_diff = pde_params['k_diff']

        def extract_phase_metrics(params_dict, mask, env_idx):
            metrics = {}
            target_keys = ['w_food', 'w_nest', 'w_rand', 'w_info', 'k_diff', 'lambda_pick', 'lambda_drop']
            for key in target_keys:
                if key in params_dict:
                    # 提取该环境下该参数的张量/数组并转为 NumPy
                    val_array = params_dict[key]
                    if torch.is_tensor(val_array):
                        val_array = val_array.detach().cpu().numpy()
                    
                    # 仅在 mask 中有激活个体时计算均值，否则退避为 0.0
                    metrics[key] = float(np.mean(val_array[env_idx][mask])) if np.any(mask) else 0.0
                else:
                    metrics[key] = 0.0
            return metrics

        for n in range(self.num_envs):
            if not self.alive_agent_buf[n].any():
                continue
            
            is_carrying = self.agent_carrying_state[n].astype(bool)

            mask_carry = is_carrying                                   
            mask_approach = (~is_carrying) & sees_food_mask[n]         
            mask_explore_base = (~is_carrying) & (~sees_food_mask[n])
            mask_social = mask_explore_base & has_info_mask[n]
            mask_pure_explore = mask_explore_base & (~has_info_mask[n])

            r_efficiency = np.zeros(self.num_agents) 
            r_dispersion = np.zeros(self.num_agents) 
            r_collision = np.zeros(self.num_agents)   
            r_event = np.zeros(self.num_agents)       

            # ==========================================================
            # 1. 纯随机探索阶段 (Random Search)
            # ==========================================================
            if np.any(mask_pure_explore):
                r_efficiency[mask_pure_explore] += vel_speed[n][mask_pure_explore] * SCALE_SEARCH_VEL
                r_efficiency[mask_pure_explore] += k_diff[n][mask_pure_explore] * SCALE_SEARCH_DIFF
                
                dist_factor = val_dist_nest[n][mask_pure_explore]
                r_dispersion[mask_pure_explore] += (1.0 - dist_factor) * 0.2  
                nest_congestion_penalty = np.maximum(dist_factor - 0.7, 0) 
                r_dispersion[mask_pure_explore] -= nest_congestion_penalty * 0.5 

            # ==========================================================
            # 2. 社交响应阶段 (Social Response)
            # ==========================================================
            if np.any(mask_social):
                v_i = vec_info_obs[n][mask_social]
                vel = vel_body_obs[n][mask_social]
                
                proj_speed = np.sum(vel * v_i, axis=1)
                
                r_efficiency[mask_social] += np.maximum(proj_speed, 0) * SCALE_SOCIAL
                r_dispersion[mask_social] += k_diff[n][mask_social] * (SCALE_SEARCH_DIFF * 0.5)

            # ==========================================================
            # 3. 靠近食物阶段 (Visual Approach)
            # ==========================================================
            if np.any(mask_approach):
                v_f = vec_food_obs[n][mask_approach]
                vel = vel_body_obs[n][mask_approach]
                proj_speed = np.sum(vel * v_f, axis=1)
                r_efficiency[mask_approach] += np.maximum(proj_speed, 0) * SCALE_ALIGNMENT

            # ==========================================================
            # 4. 搬运回巢阶段 (Carrying)
            # ==========================================================
            if np.any(mask_carry):
                v_n = vec_nest_obs[n][mask_carry]
                vel = vel_body_obs[n][mask_carry]
                proj_speed = np.sum(vel * v_n, axis=1)
                r_efficiency[mask_carry] += np.maximum(proj_speed, 0) * SCALE_ALIGNMENT

            # ==========================================================
            # 事件与碰撞检测
            # ==========================================================
            if self.num_agents > 1:
                current_pos = self.agent_pos[n, :, :2]
                dist_matrix = squareform(pdist(current_pos))
                np.fill_diagonal(dist_matrix, np.inf)
                collision_mask = np.any(dist_matrix < self.collision_distance, axis=1)
                r_collision[collision_mask] = COLLISION_PENALTY
                self.num_collision[n, 0] += np.sum(collision_mask)
            
            prev_carry = self.prev_carrying_state[n].astype(bool)
            curr_carry = is_carrying
            pickup_mask = (~prev_carry) & curr_carry
            delivery_mask = prev_carry & (~curr_carry)
            
            r_event[pickup_mask] += REWARD_EVENT
            r_event[delivery_mask] += REWARD_EVENT
            
            self.prev_carrying_state[n] = is_carrying.copy()

            total_reward = r_efficiency + r_dispersion + r_collision + r_event 
            rew_all[n, :, 0] = total_reward

            # ==========================================================
            # 🚀 全相位控制权参数提取 
            # ==========================================================
            explore_p = extract_phase_metrics(pde_params, mask_pure_explore, n)
            social_p = extract_phase_metrics(pde_params, mask_social, n)
            approach_p = extract_phase_metrics(pde_params, mask_approach, n)
            carry_p = extract_phase_metrics(pde_params, mask_carry, n)

            # ==========================================================
            # 🚀 物理瓶颈诊断指标计算 
            # ==========================================================
            physical_dist_food = -np.log(np.clip(val_dist_food[n], 1e-5, 1.0))
            physical_dist_nest = -np.log(np.clip(val_dist_nest[n], 1e-5, 1.0))
            
            # 计算当前环境下，非携粮个体离最近食物的平均距离和绝对最近距离
            searchers_mask = ~is_carrying
            min_food_dist_m = float(np.min(physical_dist_food[searchers_mask])) if np.any(searchers_mask) else 9.9
            
            # 统计有多少未携粮个体成功进入了物理拾取判定区 (dist < catch_distance)
            is_near_food_zone = searchers_mask & (physical_dist_food < self.catch_distance)
            num_near_food_zone = int(np.sum(is_near_food_zone))

            # 统计携粮个体中有多少进入了物理卸货判定区 (dist < catch_distance)
            is_near_nest_zone = is_carrying & (physical_dist_nest < self.catch_distance)
            num_near_nest_zone = int(np.sum(is_near_nest_zone))

            # 组装基础及事件信息
            infos[n] = {
                'Reward/Total': np.mean(total_reward),
                'Reward/Efficiency': np.mean(r_efficiency),
                'Reward/Collision': np.mean(r_collision),
                'Reward/Event': np.mean(r_event),
                
                'Count/Social_State': np.sum(mask_social), 
                'Count/Pickups': np.sum(pickup_mask),
                'Count/Deliveries': np.sum(delivery_mask),
            }

            for k in explore_p.keys():
                infos[n][f'Param_Explore/{k}'] = explore_p[k]
                infos[n][f'Param_Social/{k}'] = social_p[k]
                infos[n][f'Param_Approach/{k}'] = approach_p[k]
                infos[n][f'Param_Carry/{k}'] = carry_p[k]

            # 🚀 注入核心诊断指标：直接在控制台揭示死锁根本原因
            infos[n].update({
                'Diagnostic/Min_Dist_To_Food_M': min_food_dist_m,  # 非携粮个体与食物的最小物理距离(米)
                'Diagnostic/Num_Agents_In_Catch_Zone': num_near_food_zone,  # 成功进入拾取范围的个体数
                # 处于拾取区的个体，当前的 lambda_pick 动作值是多少？
                'Diagnostic/Lambda_Pick_In_Catch_Zone': float(np.mean(pde_params['lambda_pick'][n][is_near_food_zone])) if num_near_food_zone > 0 else 0.0,
                # 处于卸货区的个体，当前的 lambda_drop 动作值是多少？
                'Diagnostic/Lambda_Drop_In_Nest_Zone': float(np.mean(pde_params['lambda_drop'][n][is_near_nest_zone])) if num_near_nest_zone > 0 else 0.0,
            })

            self.last_pickup_masks = pickup_mask      
            self.last_delivery_masks = delivery_mask  
            self.last_collision_masks = collision_mask 

        return rew_all, infos

    def _get_episode_end_flags(self):
        self.steps += 1
        return episode_end_flags(
            self.alive_agent_buf, self.early_stop, self.steps, self.episode_length
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.cleanup()

        #self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(self.timestep//self.interval)

        self.reset_position()  
        
        for _ in range(self.interval-1):
            super(Supervisor, self).step(self.timestep//self.interval)
            self.handle_receiver(-1)

        observations = self.get_observations()
        observation = ((observations, self.obs_neighbor_num_buf.copy()), self.get_state())
        return observation, {"per_env": [{} for _ in range(self.num_envs)]}

    def get_info(self):
        return {"per_env": [{} for _ in range(self.num_envs)]}
