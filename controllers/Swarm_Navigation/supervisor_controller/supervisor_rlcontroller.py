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
        self.num_envs = self.args.n_rollout_threads
        self.timestep = self.args.timestep
        self.interval = self.args.interval

        super().__init__(timestep=self.timestep)

        # self.world_size = np.array([6.0, 2.0])
        self.world_size = np.array([3.0, 1.0])
        
        self.num_obs_agents = self.args.num_obs_agents
        self.obs_view = self.args.obs_view
        self.comm_view = self.args.comm_view
        
        self.collision_distance = self.args.collision_distance
        self.episode_length = self.args.episode_length

        self.num_observations = 16

        # 状态空间
        single_num_state = self.num_observations
        self.num_states = self.num_agents * single_num_state
        self.num_obs_targets = getattr(self.args, 'num_obs_targets', 0)
        self.obs_neighbor_dim = self.num_obs_targets + self.num_obs_agents

        self.num_actions = 4 

        # PDE框架与宏观物理场相关参数
        self.kde_bandwidth = 0.05     # KDE核函数带宽 (米)
                
        self.emitter, self.receiver = self.initialize_comms("emitter", "receiver")

        self.arena_pos = self.add_arenas()
        
        self.add_nodes(self.arena_pos)

        self.robots =[]
        for i in range(1, self.num_envs + 1):
            robot_env =[]
            for j in range(1, self.num_agents + 1):
                robot_env.append(self.getFromDef(f"epuck{i}-{j}"))
            self.robots.append(robot_env)

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
                Box(0, self.num_agents, (self.num_envs, self.num_agents, self.obs_neighbor_dim), dtype=np.int32),
            )),
            Box(-np.inf, np.inf, (self.num_envs, self.num_agents, self.num_states), dtype=np.float32),
        ))

        self.signal_strength = 80
        self.ps_sensor_mm = {'min': 50, 'max': 85}
        self.tof_sensor_mm = {'min': 0, 'max': 160}
        self.angle_mm = {'min': -np.pi, 'max': np.pi}
        self.pos_x_mm = {'min': self.world_size[0] / -2.0, 'max': self.world_size[0] / 2.0}
        self.pos_y_mm = {'min': self.world_size[1] / -2.0, 'max': self.world_size[1] / 2.0}
        self.max_speed = 0.12

        self.save_data = False
        self.episode_data_log =[]
        self.save_path = f"data/rl_data_0.csv"
        
        self.index = 0

        self.is_state_machine_baseline = False 
        
        self.cleanup()

    def step(self, action):

        control_observations = self.get_observations()
        pde_params = self._map_actions_to_params(action, control_observations)

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

        terminated, truncated = self._get_episode_end_flags()
        done = np.logical_or(terminated, truncated)

    
        if self.save_data == True:
            self.save_test_data(pde_params, observations, target_wheel_speeds, reward_components_info, done)
            if self.steps[0] == self.episode_length:
                self.index += 1
        
        observation = ((observations, self.obs_neighbor_num_buf.copy()), state)
        info = {"per_env": reward_components_info}
        return observation, reward, terminated, truncated, info

    def save_test_data(self, pde_params, observations, target_wheel_speeds, reward_infos, done):
        self.save_path = f'data/test_{self.index}.csv'
        
        for env_idx in range(self.num_envs):
            q_total, q_shape, q_uniformity = self.calculate_formation_quality(env_idx)
            
            for i in range(self.num_agents):
                corridor_width = float(observations[env_idx, i, 12]) 

                goal_x = 1.05 
                goal_y = 0.0

                curr_x = self.agent_pos[env_idx, i, 0] - self.arena_pos[env_idx, 0]
                target_hw = self.get_theoretical_half_width(curr_x)
                norm_width = target_hw / 0.5
                ideal_beta = np.clip(1.0 + (1.0 - norm_width) * 5.0, 1.0, 5.0)
                
                agent_log = {
                    'step': self.steps[env_idx],
                    'agent_id': i,
                    
                    # 1. 运动学基础数据
                    'pos_x': self.agent_pos[env_idx, i, 0],
                    'pos_y': self.agent_pos[env_idx, i, 1],
                    'vel_x': self.agent_vec[env_idx, i, 0],
                    'vel_y': self.agent_vec[env_idx, i, 1],
                    'angle': self.agent_angle[env_idx, i],
                    
                    # 2. 底层控制指令
                    'cmd_omega_l': float(target_wheel_speeds[env_idx, i, 0]),
                    'cmd_omega_r': float(target_wheel_speeds[env_idx, i, 1]),
                    
                    # 3. 物理场/网络输出参数
                    'w_flow': float(pde_params['w_flow'][env_idx, i]),
                    'w_shape': float(pde_params['w_shape'][env_idx, i]),
                    'k_diff': float(pde_params['k_diff'][env_idx, i]),
                    'alpha': float(pde_params['alpha'][env_idx, i]),
                    'beta': float(pde_params['beta'][env_idx, i]),
                    
                    # 4. 形态评估与环境状态
                    'q_total': float(q_total),
                    'q_shape': float(q_shape),
                    'q_uniformity': float(q_uniformity),
                    'corridor_width': corridor_width, 
                    
                    # 🌟 5. 新增：雷达图关键支持数据
                    'goal_x': float(goal_x),
                    'goal_y': float(goal_y),
                    'ideal_beta': float(ideal_beta),
                    
                    # 6. 其他观测与事件
                    'y_grad_variance': float(observations[env_idx, i, 8]),
                    'dist_centroid': float(observations[env_idx, i, 7]), 
                    'is_colliding': int(self.last_collision_masks[env_idx, i]) if hasattr(self, 'last_collision_masks') else 0,
                    'score' : int(self.success_metrics[env_idx, 0]) if hasattr(self, 'success_metrics') else 0,
                    'is_alive': int(self.alive_agent_buf[env_idx, i]),
                }
                self.episode_data_log.append(agent_log)
        
        if self.steps[0] >= self.episode_length:
            self.save_log_to_csv()
            print(f"Data saved to {self.save_path} with {len(self.episode_data_log)} rows.")
            self.episode_data_log =[]

    def save_log_to_csv(self):
        if len(self.episode_data_log) > 0:
            df = pd.DataFrame(self.episode_data_log)
            df.to_csv(self.save_path, index=False)

    def random_pos(self, arena_pos):

        x_grid_len = 0.1
        y_grid_len = 0.1

        spawn_x_max = -0.75

        world_width = float(self.world_size[0])
        world_height = float(self.world_size[1])

        n_cols = math.floor(world_width / x_grid_len)

        valid_x_positions = []

        for c in range(n_cols):
            px = (
                c * x_grid_len
                - world_width / 2.0
                + x_grid_len / 2.0
            )

            if px <= spawn_x_max:
                valid_x_positions.append(px)

        if len(valid_x_positions) == 0:
            raise RuntimeError(
                f"没有满足 px <= {spawn_x_max} 的生成位置。"
                f"请检查 world_size={self.world_size} 和 spawn_x_max。"
            )

        max_center_y = world_height / 2.0 - y_grid_len / 2.0

        positive_y_positions = np.arange(
            y_grid_len / 2.0,
            max_center_y + 1e-8,
            y_grid_len,
            dtype=np.float32,
        )

        pair_slots = [
            (float(px), float(py))
            for px in valid_x_positions
            for py in positive_y_positions
        ]

        num_pairs = self.num_agents // 2
        has_center_agent = (self.num_agents % 2 == 1)

        if len(pair_slots) < num_pairs:
            raise RuntimeError(
                f"可用镜像位置对不足：需要 {num_pairs} 对，"
                f"但只有 {len(pair_slots)} 对。"
            )

        chosen_pair_indices = np.random.choice(
            len(pair_slots),
            size=num_pairs,
            replace=False,
        )

        local_positions = []

        for idx in chosen_pair_indices:
            px, py = pair_slots[idx]

            local_positions.append([px, py])
            local_positions.append([px, -py])

        if has_center_agent:

            used_x = {
                round(pair_slots[idx][0], 6)
                for idx in chosen_pair_indices
            }

            center_x_candidates = [
                px for px in valid_x_positions
                if round(px, 6) not in used_x
            ]

            if len(center_x_candidates) == 0:
                center_x_candidates = valid_x_positions

            center_x = float(np.random.choice(center_x_candidates))
            local_positions.append([center_x, 0.0])

        agent_pos = np.asarray(local_positions, dtype=np.float32)

        np.random.shuffle(agent_pos)

        agent_pos[:, 0] += arena_pos[0]
        agent_pos[:, 1] += arena_pos[1]

        centroid_y = float(np.mean(agent_pos[:, 1]))

        if not np.isclose(
            centroid_y,
            float(arena_pos[1]),
            atol=1e-6,
        ):
            raise RuntimeError(
                f"初始化质心不在水平中心线上："
                f"centroid_y={centroid_y:.8f}, "
                f"target_y={float(arena_pos[1]):.8f}"
            )

        return agent_pos

    def add_nodes(self, arena_pos):
        for i in range(1, self.num_envs+1):
            agent_pos = self.random_pos(arena_pos[i-1])
            for j in range(1, self.num_agents + 1):
                self.importRobot(i, j, agent_pos[j - 1, 0], agent_pos[j - 1, 1], 0.05, random.uniform(-np.pi, np.pi))
        return None 
   
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
        self.add_funnel_obstacles(arena_pos_re)

        return arena_pos_re

    def get_theoretical_half_width(self, x_local):
        """根据局部坐标 x，解析计算当前位置环境的理论半宽"""
        # 环境几何参数
        half_width_narrow = 0.14       
        half_width_open = 0.5          # 开阔地半宽
        
        # 关键 X 轴坐标点
        start_funnel_x = -0.7          # 漏斗开始收缩
        entrance_x = -0.1              # 窄道正式开始 (最窄处)
        exit_x = 0.6                   # 窄道结束 (开始变宽)
        end_transition_x = 0.9         # 彻底恢复到开阔地

        # 1. 左侧初始开阔区
        if x_local < start_funnel_x:
            return half_width_open
        
        # 2. 漏斗入口收缩区
        elif start_funnel_x <= x_local < entrance_x:
            ratio = (x_local - start_funnel_x) / (entrance_x - start_funnel_x)
            return half_width_open + ratio * (half_width_narrow - half_width_open)
        
        # 3. 窄道内部保持区
        elif entrance_x <= x_local < exit_x:
            return half_width_narrow
        
        # 4. 窄道出口复原区 
        elif exit_x <= x_local < end_transition_x:
            ratio = (x_local - exit_x) / (end_transition_x - exit_x)
            return half_width_narrow + ratio * (half_width_open - half_width_narrow)
        
        # 5. 右侧目标开阔区
        else:
            return half_width_open

    def add_funnel_obstacles(self, arena_centers):
        self.internal_walls = [] 
        wall_high = 0.1
        thickness = 0.04 
        
        # --- 核心参数 ---
        half_width = 0.14      
        entrance_x = -0.1       # 接口 X 坐标
        start_x = -0.7          # 漏斗起点 X 坐标
        
        dy = 0.5 - half_width  
        dx = 0.6              
        smooth_angle = np.arctan(dy / dx) 
        slanted_length = np.sqrt(dx**2 + dy**2) 

        x_comp = (thickness / 2.0) * np.sin(smooth_angle) 
        y_comp = (thickness / 2.0) / np.cos(smooth_angle) 

        for i in range(self.num_envs):
            cx, cy, cz = arena_centers[i]
            env_walls = []
            
            p1_u, p2_u = (cx + start_x, cy + 0.5), (cx + entrance_x, cy + half_width)
            p3_u, p4_u = (cx + entrance_x, cy + half_width), (cx + 0.6, cy + half_width)
            p1_l, p2_l = (cx + start_x, cy - 0.5), (cx + entrance_x, cy - half_width)
            p3_l, p4_l = (cx + entrance_x, cy - half_width), (cx + 0.6, cy - half_width)
            env_walls.extend([(p1_u, p2_u), (p1_l, p2_l), (p3_u, p4_u), (p3_l, p4_l)])
            self.internal_walls.append(env_walls)


            slanted_x_center = cx - 0.4 - x_comp
            slanted_y_center_base = 0.32 + y_comp

            # 上斜墙
            self.importAngledWall(slanted_x_center, cy + slanted_y_center_base, wall_high/2, 
                                slanted_length, thickness, wall_high, -smooth_angle)
            # 下斜墙
            self.importAngledWall(slanted_x_center, cy - slanted_y_center_base, wall_high/2, 
                                slanted_length, thickness, wall_high, smooth_angle)
            
            # 窄道直墙：其 Y 坐标中心应为 half_width + thickness/2
            straight_y_center = half_width + (thickness / 2.0)
            
            # 上直墙
            self.importAngledWall(cx + 0.25, cy + straight_y_center, wall_high/2, 
                                0.7, thickness, wall_high, 0.0)
            # 下直墙
            self.importAngledWall(cx + 0.25, cy - straight_y_center, wall_high/2, 
                                0.7, thickness, wall_high, 0.0)

    def get_closest_point_on_segment(self, P, A, B):
        """计算点 P 到线段 AB 的最近点"""
        P = np.array(P)
        A = np.array(A)
        B = np.array(B)
        dir = B - A
        length_sq = np.sum(dir**2)
        if length_sq == 0: return A
        # 计算投影比例
        t = np.clip(np.dot(P - A, dir) / length_sq, 0, 1)
        return A + t * dir

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

    def importAngledWall(self, x, y, z, len_x, len_y, len_z, angle):
            """
            生成带有 Z 轴旋转角度的墙壁 (用于构建漏斗和斜面障碍物)
            angle: 旋转角度 (弧度制)
            """
            root = self.getRoot()
            chFd = root.getField("children")
            line_String = """
                            Solid {
                            translation %f %f %f
                            rotation 0 0 1 %f
                            children[
                                Shape {
                                appearance Appearance {
                                    material Material {
                                    diffuseColor 0.6 0.4 0.3  # 独特的障碍物颜色
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
                            """ % (x, y, z, angle, len_x, len_y, len_z, len_x, len_y, len_z)
            chFd.importMFNodeFromString(-1, line_String)    

    def importRobot(self, arena_id, id, x, y, z, ro):
        root = self.getRoot()
        chFd = root.getField("children")
        line_String = """      
                        DEF epuck%d-%d E-puck {
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
        
                            #         turretSlot [
                            #     Pi-puck {
                            #     }
                            # ]
    
        chFd.importMFNodeFromString(-1, line_String)

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

        TOTAL_W_BUDGET = 0.13
        MAX_K = 0.035

        BETA_MIN = 1.0
        BETA_MAX = 5.0

        CONTROL_DT = 0.032

        if torch.is_tensor(raw_action_batch):
            raw_action = (
                raw_action_batch
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        else:
            raw_action = np.asarray(
                raw_action_batch,
                dtype=np.float32,
            )

        expected_shape = raw_action.shape[:2]

        FILTER_X_MIN = -0.75
        FILTER_X_MAX = 0.75

        if hasattr(self, "agent_pos") and hasattr(self, "arena_pos"):
            alive_mask = self.alive_agent_buf.astype(bool)

            local_x = (
                self.agent_pos[:, :, 0]
                - self.arena_pos[:, 0][:, np.newaxis]
            )

            alive_count = np.maximum(
                np.sum(alive_mask, axis=1),
                1,
            )

            centroid_x = np.sum(
                local_x * alive_mask,
                axis=1,
            ) / alive_count

            env_filter_active = (
                (centroid_x >= FILTER_X_MIN)
                & (centroid_x <= FILTER_X_MAX)
                & np.any(alive_mask, axis=1)
            )

            filter_mask = np.broadcast_to(
                env_filter_active[:, np.newaxis],
                expected_shape,
            ).copy()

        else:
            alive_mask = np.ones(
                expected_shape,
                dtype=bool,
            )

            centroid_x = np.zeros(
                expected_shape[0],
                dtype=np.float32,
            )

            filter_mask = np.zeros(
                expected_shape,
                dtype=bool,
            )

        z_target = np.clip(
            raw_action[..., 0] - raw_action[..., 1],
            -30.0,
            30.0,
        )

        flow_ratio_target = 1.0 / (
            1.0 + np.exp(-z_target)
        )

        k_raw = np.clip(
            raw_action[..., 2],
            -1.0,
            1.0,
        )

        k_target = (
            (k_raw + 1.0)
            / 2.0
            * MAX_K
        )

        beta_raw = np.clip(
            raw_action[..., 3],
            -1.0,
            1.0,
        )

        beta_target = (
            BETA_MIN
            + (BETA_MAX - BETA_MIN)
            * (beta_raw + 1.0)
            / 2.0
        )

    
        need_initialize = (
            not hasattr(self, "_smooth_flow_ratio")
            or self._smooth_flow_ratio.shape != expected_shape
        )

        if need_initialize:
            self._smooth_flow_ratio = flow_ratio_target.copy()
            self._smooth_k = k_target.copy()
            self._smooth_beta = beta_target.copy()

            self._flow_ratio_velocity = np.zeros(
                expected_shape,
                dtype=np.float32,
            )
            self._k_velocity = np.zeros(
                expected_shape,
                dtype=np.float32,
            )
            self._beta_velocity = np.zeros(
                expected_shape,
                dtype=np.float32,
            )

        if hasattr(self, "steps"):
            reset_env_mask = np.asarray(self.steps) == 0

            self._smooth_flow_ratio[reset_env_mask] = (
                flow_ratio_target[reset_env_mask]
            )
            self._smooth_k[reset_env_mask] = (
                k_target[reset_env_mask]
            )
            self._smooth_beta[reset_env_mask] = (
                beta_target[reset_env_mask]
            )

            self._flow_ratio_velocity[reset_env_mask] = 0.0
            self._k_velocity[reset_env_mask] = 0.0
            self._beta_velocity[reset_env_mask] = 0.0

        def critically_damped_filter(
            current,
            velocity,
            target,
            tau,
            max_rate,
            max_acceleration,
            dt,
        ):

            omega = 2.0 / max(tau, 1e-6)

            acceleration = (
                omega**2 * (target - current)
                - 2.0 * omega * velocity
            )

            acceleration = np.clip(
                acceleration,
                -max_acceleration,
                max_acceleration,
            )

            new_velocity = velocity + acceleration * dt

            new_velocity = np.clip(
                new_velocity,
                -max_rate,
                max_rate,
            )

            new_value = current + new_velocity * dt

            return new_value, new_velocity

        filtered_flow_ratio, filtered_flow_velocity = (
            critically_damped_filter(
                current=self._smooth_flow_ratio,
                velocity=self._flow_ratio_velocity,
                target=flow_ratio_target,
                tau=1.2,
                max_rate=0.35,
                max_acceleration=0.8,
                dt=CONTROL_DT,
            )
        )

        filtered_k, filtered_k_velocity = critically_damped_filter(
            current=self._smooth_k,
            velocity=self._k_velocity,
            target=k_target,
            tau=1.0,
            max_rate=0.015,
            max_acceleration=0.04,
            dt=CONTROL_DT,
        )

        filtered_beta, filtered_beta_velocity = critically_damped_filter(
            current=self._smooth_beta,
            velocity=self._beta_velocity,
            target=beta_target,
            tau=1.5,
            max_rate=0.60,
            max_acceleration=1.20,
            dt=CONTROL_DT,
        )

        active_filter_mask = filter_mask & alive_mask

        smooth_flow_ratio = np.where(
            active_filter_mask,
            filtered_flow_ratio,
            flow_ratio_target,
        )

        smooth_k = np.where(
            active_filter_mask,
            filtered_k,
            k_target,
        )

        smooth_beta = np.where(
            active_filter_mask,
            filtered_beta,
            beta_target,
        )

        flow_ratio_velocity = np.where(
            active_filter_mask,
            filtered_flow_velocity,
            0.0,
        )

        k_velocity = np.where(
            active_filter_mask,
            filtered_k_velocity,
            0.0,
        )

        beta_velocity = np.where(
            active_filter_mask,
            filtered_beta_velocity,
            0.0,
        )

        # 合法范围裁剪
        smooth_flow_ratio = np.clip(
            smooth_flow_ratio,
            0.0,
            1.0,
        )

        smooth_k = np.clip(
            smooth_k,
            0.0,
            MAX_K,
        )

        smooth_beta = np.clip(
            smooth_beta,
            BETA_MIN,
            BETA_MAX,
        )

        # 死亡机器人保持上一时刻状态
        smooth_flow_ratio = np.where(
            alive_mask,
            smooth_flow_ratio,
            self._smooth_flow_ratio,
        )

        smooth_k = np.where(
            alive_mask,
            smooth_k,
            self._smooth_k,
        )

        smooth_beta = np.where(
            alive_mask,
            smooth_beta,
            self._smooth_beta,
        )

        flow_ratio_velocity = np.where(
            alive_mask,
            flow_ratio_velocity,
            self._flow_ratio_velocity,
        )

        k_velocity = np.where(
            alive_mask,
            k_velocity,
            self._k_velocity,
        )

        beta_velocity = np.where(
            alive_mask,
            beta_velocity,
            self._beta_velocity,
        )

        self._smooth_flow_ratio = smooth_flow_ratio
        self._smooth_k = smooth_k
        self._smooth_beta = smooth_beta

        self._flow_ratio_velocity = flow_ratio_velocity
        self._k_velocity = k_velocity
        self._beta_velocity = beta_velocity

        w_flow = smooth_flow_ratio * TOTAL_W_BUDGET
        w_shape = (
            1.0 - smooth_flow_ratio
        ) * TOTAL_W_BUDGET

        alpha = np.ones_like(
            smooth_beta,
            dtype=np.float32,
        )

        return {
            "w_flow": w_flow.astype(np.float32),
            "w_shape": w_shape.astype(np.float32),
            "k_diff": smooth_k.astype(np.float32),
            "alpha": alpha,
            "beta": smooth_beta.astype(np.float32),
        }

    # def _calculate_target_wheel_speeds(self, pde_params, observations):
    #     all_wheel_speeds = np.zeros((self.num_envs, self.num_agents, 2))
    #     wheel_radius = 0.02
    #     wheel_sep = 0.05685
    #     max_wheel_speed = 6.28
 
    #     grad_rho = observations[:, :, 3:5]
    #     vec_centroid = observations[:, :, 5:7] 
    #     dist_centroid = observations[:, :, 7]
    #     vec_to_goal = observations[:, :, 9:11]
    #     corridor_width = observations[:, :, 12] # 获取观测中的环境宽度

    #     def ensure_dim(tensor):
    #         return tensor[:, :, np.newaxis] if tensor.ndim == 2 else tensor

    #     w_flow = ensure_dim(pde_params['w_flow'])
    #     w_shape = ensure_dim(pde_params['w_shape'])
    #     k_diff = ensure_dim(pde_params['k_diff'])
    #     alpha = ensure_dim(pde_params['alpha'])
    #     beta = ensure_dim(pde_params['beta'])
  
    #     V_global_flow = vec_to_goal

    #     R0 = 0.2
    #     SHAPE_GAIN = 50.0 

    #     dynamic_R0 = R0 - (1.0 - corridor_width) * 0.05
    #     if dynamic_R0.ndim == 2:
    #         pass 

    #     x_rel_body = vec_centroid[:, :, 0] * dist_centroid
    #     y_rel_body = vec_centroid[:, :, 1] * dist_centroid
        
    #     cos_theta = np.cos(self.agent_angle)
    #     sin_theta = np.sin(self.agent_angle)
        
    #     x_rel_global = x_rel_body * cos_theta - y_rel_body * sin_theta
    #     y_rel_global = x_rel_body * sin_theta + y_rel_body * cos_theta

    #     dist_warped = np.sqrt(alpha[:,:,0] * x_rel_global**2 + beta[:,:,0] * y_rel_global**2) + 1e-6
    #     error_warped = dist_warped - dynamic_R0
        
    #     F_global_x = SHAPE_GAIN * (error_warped / dist_warped) * (alpha[:,:,0] * x_rel_global)
    #     F_global_y = SHAPE_GAIN * (error_warped / dist_warped) * (beta[:,:,0] * y_rel_global)
        
    #     V_form_shape_x = F_global_x * cos_theta + F_global_y * sin_theta
    #     V_form_shape_y = -F_global_x * sin_theta + F_global_y * cos_theta
        
    #     V_form_shape = np.stack([V_form_shape_x, V_form_shape_y], axis=-1)

    #     V_advection = w_flow * V_global_flow + w_shape * V_form_shape
        
    #     rho_norm = observations[:, :, 2]
    #     eps_rho = 0.05 
        
    #     V_interaction = - (k_diff / (rho_norm[:, :, np.newaxis] + eps_rho)) * grad_rho 

    #     final_v_body = V_advection + V_interaction
        
    #     if final_v_body.ndim == 4:
    #         final_v_body = final_v_body.squeeze(2)

    #     v_body_x = final_v_body[:, :, 0]
    #     v_body_y = final_v_body[:, :, 1]
        
    #     target_angle = np.arctan2(v_body_y, v_body_x)

    #     kp_omega, omega_max = 2.0, 4.0
    #     omega_cmd = np.clip(kp_omega * target_angle, -omega_max, omega_max)

    #     direction_alignment = np.clip(np.cos(target_angle), 0.0, 1.0)
    #     v_cmd = np.clip(np.linalg.norm(final_v_body, axis=-1), 0.0, self.max_speed)
    #     v_cmd *= 0.15 + 0.85 * direction_alignment

    #     # 机器人间安全避碰参数
    #     collision_distance = float(getattr(self, "collision_distance", 0.07))
    #     SAFE_AGENT_DISTANCE = max(0.13, 1.8 * collision_distance)
    #     EMERGENCY_DISTANCE = max(0.075, 1.05 * collision_distance)
    #     AVOID_OMEGA_MAX = 1.8
    #     MIN_FORWARD_SCALE = 0.10
    #     AVOID_FILTER_ALPHA = 0.25
    #     EMERGENCY_REVERSE_SPEED = 0.012

    #     state_shape = (self.num_envs, self.num_agents)
    #     if not hasattr(self, "avoid_omega_state") or self.avoid_omega_state.shape != state_shape:
    #         self.avoid_omega_state = np.zeros(state_shape, dtype=np.float32)

    #     if hasattr(self, "steps"):
    #         reset_env_mask = np.asarray(self.steps).reshape(-1) == 0
    #         self.avoid_omega_state[reset_env_mask] = 0.0

    #     for n in range(self.num_envs):
    #         alive_mask = self.alive_agent_buf[n].astype(bool)
    #         if not alive_mask.any():
    #             continue

    #         v, w = v_cmd[n].copy(), omega_cmd[n].copy()
    #         positions = self.agent_pos[n, :, :2]
    #         alive_ids = np.where(alive_mask)[0]
    #         target_avoid_omega = np.zeros(self.num_agents, dtype=np.float32)

    #         relative_vectors = positions[None, :, :] - positions[:, None, :]
    #         distance_matrix = np.linalg.norm(relative_vectors, axis=-1)
    #         np.fill_diagonal(distance_matrix, np.inf)
    #         distance_matrix[~alive_mask, :] = np.inf
    #         distance_matrix[:, ~alive_mask] = np.inf

    #         for i in alive_ids:
    #             nearest_id = int(np.argmin(distance_matrix[i]))
    #             nearest_distance = float(distance_matrix[i, nearest_id])

    #             if nearest_distance >= SAFE_AGENT_DISTANCE:
    #                 continue

    #             relative_vector = relative_vectors[i, nearest_id]
    #             neighbor_angle = float(np.arctan2(relative_vector[1], relative_vector[0]))
    #             heading = float(np.asarray(self.agent_angle[n, i]).squeeze())
    #             bearing_error = (neighbor_angle - heading + np.pi) % (2.0 * np.pi) - np.pi

    #             closeness = np.clip(
    #                 (SAFE_AGENT_DISTANCE - nearest_distance) /
    #                 (SAFE_AGENT_DISTANCE - EMERGENCY_DISTANCE + 1e-6),
    #                 0.0, 1.0
    #             )

    #             frontal_factor = np.clip(np.cos(bearing_error), 0.0, 1.0)
    #             risk_factor = closeness * (0.30 + 0.70 * frontal_factor)

    #             side_sign = float(np.sign(np.sin(bearing_error)))
    #             if abs(side_sign) < 1e-3:
    #                 side_sign = 1.0 if (i + nearest_id) % 2 == 0 else -1.0

    #             target_avoid_omega[i] = -side_sign * AVOID_OMEGA_MAX * risk_factor
    #             v[i] *= np.clip(1.0 - 0.85 * risk_factor, MIN_FORWARD_SCALE, 1.0)

    #             if nearest_distance < EMERGENCY_DISTANCE:
    #                 target_avoid_omega[i] *= 1.25
    #                 if frontal_factor > 0.20:
    #                     v[i] = -EMERGENCY_REVERSE_SPEED
    #                 else:
    #                     v[i] = min(v[i], 0.003)

    #         self.avoid_omega_state[n] = (
    #             (1.0 - AVOID_FILTER_ALPHA) * self.avoid_omega_state[n]
    #             + AVOID_FILTER_ALPHA * np.clip(target_avoid_omega, -2.2, 2.2)
    #         )
    #         self.avoid_omega_state[n, ~alive_mask] = 0.0
    #         w = np.clip(w + self.avoid_omega_state[n], -omega_max, omega_max)

    #         v_l = v - 0.5 * w * wheel_sep
    #         v_r = v + 0.5 * w * wheel_sep
    #         omega_l = np.clip(v_l / wheel_radius, -max_wheel_speed, max_wheel_speed)
    #         omega_r = np.clip(v_r / wheel_radius, -max_wheel_speed, max_wheel_speed)

    #         wheel_speeds_env = np.stack([omega_l, omega_r], axis=1)
    #         all_wheel_speeds[n] = np.where(alive_mask[:, None], wheel_speeds_env, 0.0)

    #     return all_wheel_speeds


    def _calculate_target_wheel_speeds(self, pde_params, observations):
        all_wheel_speeds = np.zeros((self.num_envs, self.num_agents, 2), dtype=np.float32)
        wheel_radius, wheel_sep, max_wheel_speed = 0.02, 0.05685, 6.28

        rho_norm = observations[:, :, 2]
        grad_rho = observations[:, :, 3:5]
        vec_centroid = observations[:, :, 5:7]
        dist_centroid = observations[:, :, 7]
        vec_to_goal = observations[:, :, 9:11]
        corridor_width = observations[:, :, 12]

        def ensure_dim(x):
            x = np.asarray(x, dtype=np.float32)
            return x[:, :, None] if x.ndim == 2 else x

        w_flow = ensure_dim(pde_params["w_flow"])
        w_shape = ensure_dim(pde_params["w_shape"])
        k_diff = ensure_dim(pde_params["k_diff"])
        alpha = ensure_dim(pde_params["alpha"])
        beta = ensure_dim(pde_params["beta"])

        V_global_flow = vec_to_goal

        R0, SHAPE_GAIN = 0.2, 50.0
        dynamic_R0 = R0 - (1.0 - corridor_width) * 0.05

        x_rel_body = vec_centroid[:, :, 0] * dist_centroid
        y_rel_body = vec_centroid[:, :, 1] * dist_centroid
        cos_theta, sin_theta = np.cos(self.agent_angle), np.sin(self.agent_angle)

        x_rel_global = x_rel_body * cos_theta - y_rel_body * sin_theta
        y_rel_global = x_rel_body * sin_theta + y_rel_body * cos_theta

        dist_warped = np.sqrt(alpha[:, :, 0] * x_rel_global**2 + beta[:, :, 0] * y_rel_global**2) + 1e-6
        error_warped = dist_warped - dynamic_R0

        F_global_x = SHAPE_GAIN * error_warped / dist_warped * alpha[:, :, 0] * x_rel_global
        F_global_y = SHAPE_GAIN * error_warped / dist_warped * beta[:, :, 0] * y_rel_global

        V_form_shape_x = F_global_x * cos_theta + F_global_y * sin_theta
        V_form_shape_y = -F_global_x * sin_theta + F_global_y * cos_theta
        V_form_shape = np.stack((V_form_shape_x, V_form_shape_y), axis=-1)

    
        V_advection = w_flow * V_global_flow + w_shape * V_form_shape 
        rho_norm = observations[:, :, 2] 
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
        v_cmd = np.clip(np.linalg.norm(final_v_body, axis=-1), 0.0, self.max_speed) 
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
        self.states_buf = np.zeros((self.num_envs, self.num_agents, self.num_states), dtype=np.float32)
        self.alive_agent_buf = np.ones((self.num_envs, self.num_agents), dtype=np.bool_) 
        self.message = np.zeros((self.num_envs, self.num_agents, 8), dtype=np.float32) 
        self.early_stop = np.zeros((self.num_envs, 1), dtype=np.bool_)
        self.num_collision = np.zeros((self.num_envs, 1), dtype=np.float32)

        self.agent_pos = np.zeros((self.num_envs, self.num_agents, 3), dtype=np.float32)
        self.agent_angle = np.zeros((self.num_envs, self.num_agents), dtype=np.float32) 
        self.agent_vec = np.zeros((self.num_envs, self.num_agents, 3), dtype=np.float32) 
        self.last_collision_masks = np.zeros((self.num_envs, self.num_agents), dtype=bool)

        self.obs_neighbor_num_buf = np.zeros((self.num_envs, self.num_agents, self.obs_neighbor_dim), dtype=np.int32) 

        self.steps = np.zeros((self.num_envs), dtype=np.int32)

        self.is_arrived_flag = np.zeros(self.num_envs, dtype=bool)
        self.success_metrics = np.zeros((self.num_envs, 1), dtype=np.float32)
        self.formation_reached_flag = np.zeros(self.num_envs, dtype=bool)

    def reset_position(self):
        for i in range(self.num_envs):
            agent_pos = self.random_pos(self.arena_pos[i])
            for j in range(self.num_agents):
                epuck_default_pos = self.robots[i][j].getField('translation').getSFVec3f()
                epuck_default_pos[:2] = agent_pos[j][:2]
                robot_default_rotation = [0,0,1,0]
                robot_default_rotation[3] = random.uniform(-np.pi, np.pi)
                self.robots[i][j].getField('translation').setSFVec3f(epuck_default_pos)
                self.robots[i][j].getField('rotation').setSFRotation(robot_default_rotation)
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
            
        # 1. 更新底层硬件位姿与速度
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
        
        if not hasattr(self, 'y_grad_variance_ema'):
            self.y_grad_variance_ema = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)

        TARGET_GOAL_X = 1.05  
        EMA_ALPHA = 0.1
        MAX_EXPECTED_DENSITY = 2.0

        if self.message is not None:
            for n in range(self.num_envs):
                if not self.alive_agent_buf[n].any():
                    continue

                local_pos_env = self.agent_pos[n, :, :2] - self.arena_pos[n][:2]
                global_vel_env = self.agent_vec[n, :, :2] 

                env_rho_norm = np.zeros(self.num_agents)
                env_grad_body_x = np.zeros(self.num_agents)
                env_grad_body_y = np.zeros(self.num_agents)
                env_vel_body_norm = np.zeros((self.num_agents, 2))
                env_rot_vec = np.zeros((self.num_agents, 2))
                env_forward_width = np.ones(self.num_agents) 

       
                for i in range(self.num_agents):
                    if not self.alive_agent_buf[n, i]: continue

                    my_pos = local_pos_env[i]
                    angle = self.agent_angle[n, i]
                    my_rot_vec = np.array([np.cos(angle), np.sin(angle)])
                    env_rot_vec[i] = my_rot_vec
                    
                    v_g = global_vel_env[i]
                    v_b_x = v_g[0] * my_rot_vec[0] + v_g[1] * my_rot_vec[1]
                    v_b_y = -v_g[0] * my_rot_vec[1] + v_g[1] * my_rot_vec[0]
                    env_vel_body_norm[i] = np.clip(np.array([v_b_x, v_b_y]) / self.max_speed, -1, 1)

                    curr_hw = self.get_theoretical_half_width(my_pos[0])
                    future_hw = self.get_theoretical_half_width(my_pos[0] + 0.1)
                    effective_hw = min(curr_hw, future_hw)
                    env_forward_width[i] = np.clip(effective_hw / 0.5, 0.28, 1.0)

                    h = 0.15
                    h_sq = h**2
                    wall_rho = 0.0
                    wall_grad_global = np.zeros(2)

                    h_wall = 0.05
                    h_wall_sq = h_wall**2

                    potential_walls = [
                        np.array([self.pos_x_mm['min'], my_pos[1]]), 
                        np.array([self.pos_x_mm['max'], my_pos[1]]), 
                        np.array([my_pos[0], self.pos_y_mm['min']]), 
                        np.array([my_pos[0], self.pos_y_mm['max']])
                    ]
                    
                    if hasattr(self, 'internal_walls'):
                        for seg_start, seg_end in self.internal_walls[n]:
                            cp = self.get_closest_point_on_segment(my_pos, seg_start, seg_end)
                            potential_walls.append(cp)

                    for wp in potential_walls:
                        diff = wp - my_pos 
                        dist = np.linalg.norm(diff)
                        if dist < 0.2:
                            weight = 2.0 * np.exp(-(dist**2) / (2 * h_wall_sq))
                            wall_rho += weight
                            wall_grad_global += (weight / h_wall_sq) * diff

                    all_dists = np.linalg.norm(local_pos_env - my_pos, axis=1)
                    neighbor_mask = (all_dists < self.comm_view) & self.alive_agent_buf[n]
                    neighbor_mask[i] = False
                    neighbors = local_pos_env[neighbor_mask]

                    agent_rho = 0.0
                    agent_grad_global = np.zeros(2)
                    if len(neighbors) > 0:
                        diffs = neighbors - my_pos 
                        d_sqs = np.sum(diffs**2, axis=1)
                        # 密度贡献
                        weights = np.exp(-d_sqs / (2 * h_sq))
                        agent_rho = np.sum(weights)
                        # 梯度贡献
                        agent_grad_global = np.sum((weights[:, np.newaxis] / h_sq) * diffs, axis=0)

                    t_grad_g = agent_grad_global + wall_grad_global
                    
                    # 转到机体坐标系
                    env_grad_body_x[i] = t_grad_g[0] * my_rot_vec[0] + t_grad_g[1] * my_rot_vec[1]
                    env_grad_body_y[i] = -t_grad_g[0] * my_rot_vec[1] + t_grad_g[1] * my_rot_vec[0]
                    
                    env_rho_norm[i] = np.clip((agent_rho + wall_rho) / MAX_EXPECTED_DENSITY, 0, 1)


                alive_positions = local_pos_env[self.alive_agent_buf[n]]
                if len(alive_positions) > 0:
                    true_centroid = np.mean(alive_positions, axis=0)
                    centroid_dist_to_goal = np.linalg.norm(true_centroid - np.array([TARGET_GOAL_X, 0.0]))

                    q_total_env, q_shape_env, q_uniformity_env = self.calculate_formation_quality(n, forced_beta=1.2,)
                    
                    formation_ready = (q_total_env >= 0.75) and (true_centroid[0] < -0.90)
                    
                    if formation_ready:
                        self.formation_reached_flag[n] = True
                else:
                    centroid_dist_to_goal = 10.0
                    q_total_env = 0.0

                for i in range(self.num_agents):
                    if not self.alive_agent_buf[n, i]: continue

                    my_pos = local_pos_env[i]
                    my_rot_vec = env_rot_vec[i]

                    goal_global = np.array([TARGET_GOAL_X, 0.0])
                    vec_goal_global = goal_global - my_pos
                    dist_goal = np.linalg.norm(vec_goal_global)
                    vec_goal_unit = vec_goal_global / (dist_goal + 1e-6)
                    
                    goal_body_x = vec_goal_unit[0] * my_rot_vec[0] + vec_goal_unit[1] * my_rot_vec[1]
                    goal_body_y = -vec_goal_unit[0] * my_rot_vec[1] + vec_goal_unit[1] * my_rot_vec[0]

                    all_dists = np.linalg.norm(local_pos_env - my_pos, axis=1)
                    real_neighbor_mask = (all_dists < self.comm_view) & self.alive_agent_buf[n]
                    real_neighbor_mask[i] = False
                    real_neighbor_indices = np.where(real_neighbor_mask)[0]

                    vec_centroid_body = np.zeros(2)
                    dist_centroid = 0.0 # 标量初始化
                    current_std_y = 0.0

                    if len(real_neighbor_indices) > 0:
                        centroid_global = np.mean(local_pos_env[real_neighbor_indices], axis=0)
                        vec_centroid_global = centroid_global - my_pos
                        raw_dist_centroid = np.linalg.norm(vec_centroid_global)
                        
                        if raw_dist_centroid > 1e-6:
                            dist_centroid = raw_dist_centroid # 标量赋值
                            vec_centroid_norm = vec_centroid_global / raw_dist_centroid
                            vec_centroid_body[0] = vec_centroid_norm[0] * my_rot_vec[0] + vec_centroid_norm[1] * my_rot_vec[1]
                            vec_centroid_body[1] = -vec_centroid_norm[0] * my_rot_vec[1] + vec_centroid_norm[1] * my_rot_vec[0]

                        neighbor_grad_ys = env_grad_body_y[real_neighbor_indices]
                        current_std_y = np.std(neighbor_grad_ys)
                    
                    self.y_grad_variance_ema[n, i] = (1.0 - EMA_ALPHA) * self.y_grad_variance_ema[n, i] + EMA_ALPHA * current_std_y
                      
                    self_state_part = env_vel_body_norm[i] 

                    macro_field_part = np.array([
                        env_rho_norm[i],
                        np.tanh(env_grad_body_x[i]),
                        np.tanh(env_grad_body_y[i]),
                        vec_centroid_body[0],
                        vec_centroid_body[1],
                        np.clip(dist_centroid, 0, 1.0),
                        self.y_grad_variance_ema[n, i]
                    ], dtype=np.float32)

                    goal_feature = np.array([
                        goal_body_x,
                        goal_body_y,
                        np.clip(dist_goal / 3.0, 0, 1.0)
                    ], dtype=np.float32)

                    corridor_feature = np.array([env_forward_width[i]], dtype=np.float32)

                    # --- Phase Indicator 判定逻辑 ---
                    phase_indicator = np.zeros(3, dtype=np.float32)

                    if centroid_dist_to_goal < 0.1 and self.formation_reached_flag[n]: 
                        phase_indicator[2] = 1.0
                    elif self.formation_reached_flag[n]: 
                        phase_indicator[1] = 1.0
                    else:
                        phase_indicator[0] = 1.0

                    observation[n, i] = np.concatenate([
                        self_state_part, 
                        macro_field_part, 
                        goal_feature, 
                        corridor_feature,
                        phase_indicator
                    ])

            self.states_buf = np.zeros((self.num_envs, self.num_agents, self.num_states), dtype=np.float32)
            all_agent_obs = observation.reshape(self.num_envs, -1) 
            for n in range(self.num_envs):
                for i in range(self.num_agents):
                    self.states_buf[n, i] = all_agent_obs[n]

        return observation

    def get_reward(self, pde_params, observations):

        if not hasattr(self, 'phase_stats'):
            self.phase_stats = {}
            
        for n in range(self.num_envs):
            # 仅在每局第 0 步或环境刚创建时进行重置
            if self.steps[n] == 0 or n not in self.phase_stats:
                self.phase_stats[n] = {
                    'PhaseA': {'steps': 0, 'Reward': 0.0, 'W_Flow': 0.0, 'W_Shape': 0.0, 'K_Diff': 0.0, 'Beta': 0.0},
                    'PhaseB': {'steps': 0, 'Reward': 0.0, 'W_Flow': 0.0, 'W_Shape': 0.0, 'K_Diff': 0.0, 'Beta': 0.0},
                    'PhaseC': {'steps': 0, 'Reward': 0.0, 'W_Flow': 0.0, 'W_Shape': 0.0, 'K_Diff': 0.0, 'Beta': 0.0}
                }

        rew_all = np.zeros((self.num_envs, self.num_agents, 1), dtype=np.float32)
        infos = [{} for _ in range(self.num_envs)]

        goal_pos = np.array([2.0, 0.0], dtype=np.float32)
        success_formation_threshold = 0.40
        success_goal_distance = 0.10
        success_speed_threshold = 0.02
        opening_threshold = 0.8
        robot_radius = 0.035
        clearance_margin = 0.002

        current_lambda = float(getattr(self, 'constraint_lambda', 0.0))

        def smoothstep(edge0, edge1, x):
            t = np.clip((x - edge0) / (edge1 - edge0 + 1e-6), 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)

        def masked_mean(values, mask):
            if not np.any(mask):
                return 0.0
            return float(np.mean(np.asarray(values, dtype=np.float32)[mask]))

        def zero_info(lambda_value, early_stopped=0.0):
            info = {
                'Reward/Total': 0.0,
                'Reward/Penalized': 0.0,
                'Reward/Task': 0.0,
                'Reward/Formation': 0.0,
                'Reward/Progress': 0.0,
                'Reward/r_squeeze_penalty': 0.0,
                'Reward/Arrival_Bonus': 0.0,
                'Reward/Collision': 0.0,
                'Reward/Motion_Penalty': 0.0,
                'Reward/Constraint_Penalty': 0.0,
                'Reward/PhaseA': 0.0,
                'Reward/PhaseB': 0.0,
                'Reward/PhaseC': 0.0,
                'Reward/Terminal_Circle_Penalty': 0.0,
                'Reward/Terminal_Speed_Penalty': 0.0,
                'Reward/Terminal_Drift_Penalty': 0.0,
                'Reward/Terminal_Center_Penalty': 0.0,
                'Reward/Navigation_Distance_Penalty': 0.0,
                'Constraint/Cost': 0.0,
                'Constraint/Lambda': float(lambda_value),
                'Constraint/Collision_Wall': 0.0
            }
            for phase_name in ['PhaseA', 'PhaseB', 'PhaseC']:
                info[f'Param/{phase_name}_W_Flow'] = 0.0
                info[f'Param/{phase_name}_W_Shape'] = 0.0
                info[f'Param/{phase_name}_K_Diff'] = 0.0
                info[f'Param/{phase_name}_Beta'] = 0.0
            return info

        vel_body = observations[:, :, 0:2]
        vec_goal = observations[:, :, 9:11]
        corridor_width = observations[:, :, 12]
        local_density = observations[:, :, 2]

        w_flow_batch = np.asarray(pde_params['w_flow'], dtype=np.float32)
        w_shape_batch = np.asarray(pde_params['w_shape'], dtype=np.float32)
        k_diff_batch = np.asarray(pde_params['k_diff'], dtype=np.float32)
        beta_batch = np.asarray(pde_params['beta'], dtype=np.float32)

        env_constraint_means = []

        for n in range(self.num_envs):
            if not self.alive_agent_buf[n].any():
                infos[n] = zero_info(current_lambda)
                continue

            alive_mask = self.alive_agent_buf[n].copy()
            if bool(self.early_stop[n, 0]):
                infos[n] = zero_info(current_lambda, early_stopped=1.0)
                continue

            w_flow_env = w_flow_batch[n]
            w_shape_env = w_shape_batch[n]
            k_diff_env = k_diff_batch[n]
            beta_env = beta_batch[n]

            r_phase_a = np.zeros(self.num_agents, dtype=np.float32)
            r_phase_b = np.zeros(self.num_agents, dtype=np.float32)
            r_phase_c = np.zeros(self.num_agents, dtype=np.float32)
            r_formation = np.zeros(self.num_agents, dtype=np.float32)
            r_progress = np.zeros(self.num_agents, dtype=np.float32)
            r_collision = np.zeros(self.num_agents, dtype=np.float32)
            r_squeeze_penalty = np.zeros(self.num_agents, dtype=np.float32)
            r_arrival_bonus = np.zeros(self.num_agents, dtype=np.float32)
            r_motion_penalty = np.zeros(self.num_agents, dtype=np.float32)

            q_total, q_shape, q_uniformity = self.calculate_formation_quality(n)
            formation_gate = smoothstep(0.35, 0.82, q_total)
            shape_gate = smoothstep(0.35, 0.85, q_shape)
            uniformity_gate = smoothstep(0.35, 0.72, q_uniformity)
            quality_gate = formation_gate * shape_gate * uniformity_gate
            progress_gate = 0.50 + 0.50 * quality_gate

            positions = self.agent_pos[n, :, :2] - self.arena_pos[n][:2]
            alive_positions = positions[alive_mask]
            current_centroid = np.mean(alive_positions, axis=0)
            centroid_dist_to_goal = float(np.linalg.norm(current_centroid - goal_pos))
            rel_positions = alive_positions - current_centroid

            alive_indices = np.where(alive_mask)[0]
            collision_mask = np.zeros(self.num_agents, dtype=bool)
            if len(alive_indices) > 1:
                alive_pos = self.agent_pos[n, alive_indices, :2]
                dist_matrix = squareform(pdist(alive_pos))
                np.fill_diagonal(dist_matrix, np.inf)
                collision_mask[alive_indices] = np.any(
                    dist_matrix < self.collision_distance,
                    axis=1,
                )
                self.num_collision[n] += np.sum(collision_mask[alive_indices]) // 2

            corridor_env = np.clip(corridor_width[n], 0.28, 1.0)
            agent_y = positions[:, 1]
            half_width = corridor_env * 0.5
            safe_half_width = np.maximum(
                half_width - robot_radius - clearance_margin,
                0.02,
            )
            wall_overrun = np.maximum(np.abs(agent_y) - safe_half_width, 0.0)
            wall_overrun = np.where(alive_mask, wall_overrun, 0.0)
            wall_collision = (
                np.abs(agent_y) > np.maximum(half_width - robot_radius, 0.02)
            ) & alive_mask
            total_collision_mask = collision_mask | wall_collision
            self.last_collision_masks[n] =  collision_mask

            terminal_gate = 1.0 - smoothstep(0.10, 0.35, centroid_dist_to_goal)
            progress_attenuation = smoothstep(0.12, 0.45, centroid_dist_to_goal)
            centroid_velocity = np.mean(self.agent_vec[n, alive_mask, :2], axis=0)
            centroid_speed = float(np.linalg.norm(centroid_velocity))
            centroid_goal_dir = (goal_pos - current_centroid) / (centroid_dist_to_goal + 1e-6)
            centroid_forward_speed = max(float(np.dot(centroid_velocity, centroid_goal_dir)), 0.0)
            centroid_forward_speed_norm = np.clip(
                centroid_forward_speed / (self.max_speed + 1e-6),
                0.0,
                1.0,
            )
            centroid_speed_norm = np.clip(
                centroid_speed / (self.max_speed + 1e-6),
                0.0,
                1.0,
            )

            current_x_std = np.std(rel_positions[:, 0]) if len(rel_positions) > 1 else 0.0
            current_y_std = np.std(rel_positions[:, 1]) if len(rel_positions) > 1 else 0.0
 
            mean_norm_width = float(np.mean(observations[n, alive_mask, 12]))
            target_beta_shape = float(np.clip(1.2 + (1.0 - mean_norm_width) * 5.0, 1.2, 5.0))

            target_y_std = float(0.2 / np.sqrt(2.0) / np.sqrt(target_beta_shape))
            current_aspect = current_x_std / (current_y_std + 1e-4)
            target_aspect = np.sqrt(target_beta_shape)
            aspect_log_error = float(np.log((current_aspect + 1e-4) / (target_aspect + 1e-4)))
            aspect_error = float(np.maximum(np.abs(aspect_log_error) - 0.08, 0.0))
            y_over_error = float(np.maximum(current_y_std - target_y_std, 0.0))

            q_circle_total, q_circle_shape, q_circle_uniformity = self.calculate_formation_quality(
                n,
                forced_beta=1.2,
            )
            q_circle_floor = min(q_circle_total, q_circle_shape, q_circle_uniformity)
            
            circle_formation_gate = smoothstep(0.35, 0.82, q_circle_total)
            circle_shape_gate = smoothstep(0.35, 0.85, q_circle_shape)
            circle_uniformity_gate = smoothstep(0.35, 0.72, q_circle_uniformity)
            circle_quality_gate = circle_formation_gate * circle_shape_gate * circle_uniformity_gate
            
            terminal_circle_log_error = float(np.log(current_aspect + 1e-4))
            terminal_circle_error = float(np.maximum(np.abs(terminal_circle_log_error) - 0.06, 0.0))

            phase_features = observations[n, :, 13:16]
            mask_phase_a = (phase_features[:, 0] > 0.5) & alive_mask
            mask_phase_b = (phase_features[:, 1] > 0.5) & alive_mask
            mask_phase_c = (phase_features[:, 2] > 0.5) & alive_mask
            unassigned_alive = alive_mask & ~(mask_phase_a | mask_phase_b | mask_phase_c)
            mask_phase_a = mask_phase_a | unassigned_alive

            is_in_constriction = (corridor_env <= opening_threshold) & alive_mask

            proj_speed = np.sum(vel_body[n] * vec_goal[n], axis=1)
            positive_proj_speed = np.maximum(proj_speed, 0.0)
            local_ring_error = np.maximum(np.abs(observations[n, :, 7] - 0.20) - 0.05, 0.0)

            phase_a_time_decay = np.exp(-self.steps[n] / 200.0)
            
            is_in_starting_area = (current_centroid[0] < -0.70)

            target_aspect_A = np.sqrt(1.2)
            aspect_log_error_circle = float(np.log((current_aspect + 1e-4) / (target_aspect_A + 1e-4)))
            aspect_error_circle = float(np.maximum(np.abs(aspect_log_error_circle) - 0.08, 0.0))

            target_beta_active = np.ones(self.num_agents, dtype=np.float32) * 1.2
            target_beta_active[mask_phase_b] = target_beta_shape

            r_beta_penalty = 0.8 * (beta_env - target_beta_active) ** 2

            phase_a_quality_reward = (
                4.5 * q_circle_floor
                + 3.0 * q_circle_total
                + 2.0 * q_circle_uniformity
            )
            
            if is_in_starting_area:
                r_phase_a[mask_phase_a] = phase_a_quality_reward * phase_a_time_decay
                r_phase_a[mask_phase_a] -= 2.0 * aspect_error_circle 
                r_phase_a[mask_phase_a] -= r_beta_penalty[mask_phase_a]
            else:
                r_phase_a[mask_phase_a] = -1.0 - r_beta_penalty[mask_phase_a]

            phase_b_progress = (
                8.0
                * positive_proj_speed
                * progress_gate
                * progress_attenuation
            )
            
            phase_b_shape_reward = -1.2 * np.clip(aspect_error / 0.75, 0.0, 1.0)
            phase_b_width_penalty = 5.0 * y_over_error ** 2
            
            r_phase_b[mask_phase_b] = phase_b_progress[mask_phase_b]
            r_phase_b[mask_phase_b] += phase_b_shape_reward - phase_b_width_penalty
            r_phase_b[mask_phase_b] -= r_beta_penalty[mask_phase_b]
            r_phase_b[mask_phase_b] -= 0.5 * np.clip(
                wall_overrun[mask_phase_b] / 0.08,
                0.0,
                1.0,
            )

            phase_c_speed_penalty = 2.0 * centroid_speed_norm ** 2  
            
            phase_c_goal_reward = 2.0 * (1.0 - np.clip(centroid_dist_to_goal / 0.10, 0.0, 1.0)) * circle_quality_gate
            
            phase_c_quality_reward = (
                2.5 * q_circle_floor
                + 2.0 * q_circle_total
                + 1.0 * q_circle_uniformity
            )
            
            r_phase_c[mask_phase_c] = (
                phase_c_goal_reward
                + phase_c_quality_reward     
                - phase_c_speed_penalty     
                - 2.0 * aspect_error_circle  
                - r_beta_penalty[mask_phase_c]
            )

            r_formation[:] = r_phase_a + r_phase_c
            r_progress[:] = np.where(mask_phase_b, phase_b_progress, 0.0)
            r_squeeze_penalty[mask_phase_b] -= phase_b_width_penalty
            r_squeeze_penalty[is_in_constriction] -= 0.6 * np.clip(
                wall_overrun[is_in_constriction] / 0.08,
                0.0,
                1.0,
            )

            premature_motion_penalty = np.zeros(self.num_agents, dtype=np.float32)
            
            premature_motion_penalty[mask_phase_a] = (
                4.0
                * centroid_forward_speed_norm
            )
            
            navigation_distance_penalty = np.zeros(self.num_agents, dtype=np.float32)
            
            navigation_distance_penalty[mask_phase_b] = (
                0.8
                * ((1.0 - quality_gate) ** 2)
                * (1.0 - terminal_gate)
                * np.clip(centroid_dist_to_goal / 2.0, 0.0, 1.0) ** 2
            )
            
            # 仅对活着的智能体减去两项动作惩罚
            r_motion_penalty[alive_mask] -= (
                premature_motion_penalty[alive_mask] 
                + navigation_distance_penalty[alive_mask] 
            )

            is_arrived = (
                self.formation_reached_flag[n]
                and q_circle_floor >= success_formation_threshold
                and centroid_dist_to_goal < success_goal_distance
                and centroid_speed < success_speed_threshold
            )
            is_new_success = is_arrived and not bool(self.is_arrived_flag[n])
            if is_new_success:
                self.is_arrived_flag[n] = True
                self.success_metrics[n, 0] = 1.0
                self.early_stop[n, 0] = True
                r_arrival_bonus[alive_mask] += 50.0

            terminal_circle_penalty = terminal_gate * 5.0 * terminal_circle_error ** 2
            terminal_speed_penalty = terminal_gate * (
                2.0 * centroid_speed_norm
                + 2.0 * centroid_speed_norm ** 2
            )
            terminal_drift_penalty = terminal_gate * 2.0 * centroid_forward_speed_norm ** 2
            terminal_center_penalty = terminal_gate * 3.0 * centroid_dist_to_goal
            r_arrival_bonus[mask_phase_c] += (
                terminal_gate
                * q_circle_floor
                * np.exp(-8.0 * centroid_speed_norm)
            )

            wall_overrun_norm = np.clip(wall_overrun / 0.08, 0.0, 1.0)

            collision_wall_cost = (
                0.8 * collision_mask.astype(np.float32)   # 机器人硬相撞
                + 0.5 * wall_collision.astype(np.float32) # 坚硬撞墙
            )

            constraint_cost = np.clip(
                np.where(alive_mask, collision_wall_cost, 0.0),
                0.0,
                2.0,
            )
            
            constraint_penalty = -current_lambda * constraint_cost
            r_collision[:] = -current_lambda * collision_wall_cost

            step_penalty = np.where(mask_phase_b, 0.02, 0.0) * alive_mask.astype(np.float32)

            task_reward = (
                r_phase_a
                + r_phase_b
                + r_phase_c
                + r_arrival_bonus
                + r_motion_penalty
                - step_penalty
            )

            task_reward = np.clip(task_reward, -200.0, 200.0)
            
            total_reward = np.where(
                alive_mask,
                task_reward + constraint_penalty,
                0.0,
            )

            total_reward = np.clip(total_reward, -100.0, 100.0) 
            rew_all[n, :, 0] = total_reward.astype(np.float32)

            env_constraint_means.append(float(np.mean(constraint_cost[alive_mask])))

            for phase_name, phase_mask, phase_rewards in [
                ('PhaseA', mask_phase_a, r_phase_a),
                ('PhaseB', mask_phase_b, r_phase_b),
                ('PhaseC', mask_phase_c, r_phase_c),
            ]:
                if np.any(phase_mask):
                    step_reward = float(np.mean(phase_rewards[phase_mask]))
                    step_w_flow = float(np.mean(w_flow_env[phase_mask]))
                    step_w_shape = float(np.mean(w_shape_env[phase_mask]))
                    step_k_diff = float(np.mean(k_diff_env[phase_mask]))
                    step_beta = float(np.mean(beta_env[phase_mask]))
                    
                    stats = self.phase_stats[n][phase_name]
                    stats['steps'] += 1
                    stats['Reward'] += step_reward
                    stats['W_Flow'] += step_w_flow
                    stats['W_Shape'] += step_w_shape
                    stats['K_Diff'] += step_k_diff
                    stats['Beta'] += step_beta

            infos[n] = {
                'Reward/Total': masked_mean(total_reward, alive_mask),
                'Reward/Penalized': masked_mean(total_reward, alive_mask),
                'Reward/Task': masked_mean(task_reward, alive_mask),
                'Reward/Formation': masked_mean(r_formation, alive_mask),
                'Reward/Progress': masked_mean(r_progress, alive_mask),
                'Reward/r_squeeze_penalty': masked_mean(r_squeeze_penalty, alive_mask),
                'Reward/Arrival_Bonus': masked_mean(r_arrival_bonus, alive_mask),
                'Reward/Collision': masked_mean(r_collision, alive_mask),
                'Reward/Motion_Penalty': masked_mean(r_motion_penalty, alive_mask),
                'Reward/Constraint_Penalty': masked_mean(constraint_penalty, alive_mask),
                
                'Reward/PhaseA': self.phase_stats[n]['PhaseA']['Reward'] / max(1, self.phase_stats[n]['PhaseA']['steps']),
                'Reward/PhaseB': self.phase_stats[n]['PhaseB']['Reward'] / max(1, self.phase_stats[n]['PhaseB']['steps']),
                'Reward/PhaseC': self.phase_stats[n]['PhaseC']['Reward'] / max(1, self.phase_stats[n]['PhaseC']['steps']),
                
                'Reward/Terminal_Circle_Penalty': float(terminal_circle_penalty),
                'Reward/Terminal_Speed_Penalty': float(terminal_speed_penalty),
                'Reward/Terminal_Drift_Penalty': float(terminal_drift_penalty),
                'Reward/Terminal_Center_Penalty': float(terminal_center_penalty),
                'Reward/Navigation_Distance_Penalty': masked_mean(navigation_distance_penalty, alive_mask),
                'Constraint/Cost': masked_mean(constraint_cost, alive_mask),
                'Constraint/Lambda': current_lambda,
                'Constraint/Collision_Wall': masked_mean(collision_wall_cost, alive_mask)
            }

            for phase_name in ['PhaseA', 'PhaseB', 'PhaseC']:
                stats = self.phase_stats[n][phase_name]
                steps = max(1, stats['steps'])
                infos[n][f'Param/{phase_name}_W_Flow'] = stats['W_Flow'] / steps
                infos[n][f'Param/{phase_name}_W_Shape'] = stats['W_Shape'] / steps
                infos[n][f'Param/{phase_name}_K_Diff'] = stats['K_Diff'] / steps
                infos[n][f'Param/{phase_name}_Beta'] = stats['Beta'] / steps

        return rew_all, infos

    def calculate_formation_quality(self, n_env, forced_beta=None):
        alive_mask = self.alive_agent_buf[n_env] 

        if not alive_mask.any():
            return 0.0, 0.0, 0.0

        # 1. 只获取存活机器人的位置
        all_positions = self.agent_pos[n_env, :, :2] - self.arena_pos[n_env][:2]
        alive_positions = all_positions[alive_mask] 

        # 2. 计算基于存活个体的真实质心
        centroid = np.mean(alive_positions, axis=0) 
        rel_pos = alive_positions - centroid  # 相对于存活群体质心的相对位置
        
        avg_x = centroid[0]
        target_hw = self.get_theoretical_half_width(avg_x)
        norm_width = np.clip(target_hw / 0.5, 0.28, 1.0)

        if forced_beta is not None:
            t_beta = float(np.mean(forced_beta))
            t_alpha = 1.0
        else:
            t_alpha = 1.0
            beta_gain = 5.0 
            t_beta = 1.0 + (1.0 - norm_width) * beta_gain
            t_beta = np.clip(t_beta, 1.0, 5.0) 
        
        R0 = 0.2
        adaptive_R0 = R0 - (1.0 - norm_width) * 0.05 

        # 3. 在环境自适应的 warped space 中评估形态。
        warp_scale = np.array([np.sqrt(t_alpha), np.sqrt(t_beta)], dtype=np.float32)
        warped_pos = rel_pos * warp_scale
        dist_warped = np.linalg.norm(warped_pos, axis=1)

        TOLERANCE_R = 0.01 
        raw_shape_errors = np.maximum(0.0, np.abs(dist_warped - adaptive_R0) - TOLERANCE_R)

        grad_norm = np.sqrt((t_alpha * rel_pos[:, 0])**2 + (t_beta * rel_pos[:, 1])**2) / (dist_warped + 1e-6)
        physical_shape_errors = raw_shape_errors / grad_norm
        mean_physical_error = np.mean(physical_shape_errors)
        
        q_shape = np.exp(-25.0 * mean_physical_error)

        # ==========================================
        # 4. warped space 中的间距均匀度 (仅针对存活者)
        # ==========================================
        if len(alive_positions) < 2:
            q_uniformity = 1.0 # 只有一个人时，不存在不均匀
        else:
            angles = np.arctan2(warped_pos[:, 1], warped_pos[:, 0])
            sort_idx = np.argsort(angles)
            sorted_pos = warped_pos[sort_idx]
            
            shifted_pos = np.roll(sorted_pos, shift=-1, axis=0) 
            adjacent_dists = np.linalg.norm(sorted_pos - shifted_pos, axis=1)
            
            dist_std = np.std(adjacent_dists)
            
            TOLERANCE_STD = 0.005 
            effective_std = np.maximum(0.0, dist_std - TOLERANCE_STD)
            q_uniformity = np.exp(-15.0 * effective_std)

        # 5. 综合指标
        q_total = 0.7 * q_shape + 0.3 * q_uniformity
        
        return np.clip(q_total, 0.0, 1.0), q_shape, q_uniformity
    
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
