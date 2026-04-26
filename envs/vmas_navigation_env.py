# envs/vmas_navigation_env.py
import numpy as np
import torch
import vmas
from vmas import make_env

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VmasNavigationEnv:
    """VMAS Navigation环境包装器"""
    
    def __init__(self, config, continuous_actions=True):
        self.n_agent = config.getint('n_agent')
        self.agent = config.get('agent')
        self.max_steps = config.getint('max_steps', 100)
        self.T = self.max_steps
        self.coop_gamma = config.getfloat('coop_gamma')
        self.seed = config.getint('seed')
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.continuous_actions=True
        # Navigation场景特定参数
        self.n_obstacles = config.getint('n_obstacles', 3)
        self.observe_all_goals = config.getboolean('observe_all_goals', False)
        self.shared_reward = config.getboolean('shared_reward', True)
        self.collision_penalty = config.getfloat('collision_penalty', -0.1)
        self.goal_reward = config.getfloat('goal_reward', 1.0)
        self.min_distance_between_entities = config.getfloat('min_distance_between_entities', 0.1)
        
        # 创建VMAS Navigation环境
        self.env = make_env(
            scenario="navigation",
            num_envs=1,  # 单环境训练
            device=device,
            continuous_actions=True,
            # Navigation场景参数
            n_agents=self.n_agent,
            n_obstacles=self.n_obstacles,
            observe_all_goals=self.observe_all_goals,
            shared_reward=self.shared_reward,
            collision_penalty=self.collision_penalty,
            goal_reward=self.goal_reward,
            min_distance_between_entities=self.min_distance_between_entities,
        )
        
        # 取 action/observation 维度
        self.n_actions = self.env.action_space[0].shape[0] if continuous_actions else self.env.action_space[0].n
        self.n_obs = self.env.observation_space[0].shape[0]

        self.n_a_ls = [self.n_actions for _ in range(self.n_agent)]
        self.n_s_ls = [self.n_obs for _ in range(self.n_agent)]

        # 掩码
        self.neighbor_mask = self._create_neighbor_mask(self.n_agent)
        self.distance_mask = self._create_distance_mask(self.n_agent)

        # 当前观测
        self.current_obs = None


    def reset(self):
        self.fp = np.ones((self.n_agent, self.n_actions)) / self.n_actions
        """重置环境，返回每个 agent 的观测"""
        reset_result = self.env.reset()

        # VMAS reset 返回 (obs, info) 或 obs
        if isinstance(reset_result, tuple):
            obs = reset_result[0]  # obs 是 list，长度 = n_agent
        else:
            obs = reset_result

        # obs[i] shape = [num_envs, obs_dim]，取第 0 个环境
        obs = [o[0].cpu().numpy() for o in obs]

        if len(obs) != self.n_agent:
            print(f"WARNING: Expected {self.n_agent} agents, got {len(obs)} observations")

        self.current_obs = obs
        self.step_count = 0
        self.T = 0  # 维护算法需要的 time step
        return obs


    def step(self, actions):
        """
        执行一步环境交互
        参数:
            actions: list/np.ndarray, shape = [n_agent, action_dim]
        返回:
            obs, rewards, dones, infos, episode_done
        """
        # 转 tensor，shape = [1, n_agent, act_dim]
        # actions = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device).unsqueeze(0)
        # actions = torch.as_tensor(actions, dtype=torch.float32, device=device)

        # actions = actions.unsqueeze(0)

        obs, rewards, dones, infos = self.env.step(actions)

        # VMAS 返回 batched 数据，取第 0 个环境
        obs = [o[0].cpu().numpy() for o in obs]          # list[n_agent, obs_dim]
        rewards = rewards[0].cpu().numpy().tolist()      # list[n_agent]
        dones = dones[0].cpu().numpy().astype(bool)      # list[n_agent]
        info = infos[0]                                  # dict，单环境信息
        
        # 计算全局奖励（平均）
        global_reward = float(sum(rewards) / len(rewards))
        
        # 额外信息
        collision_info = self._extract_collision_info(info)
        goal_info = self._extract_goal_info(info)
        infos = [{"collision_info": collision_info, "goal_reached": goal_info} for _ in range(self.n_agent)]

        # episode_done 取全局 done 或 max_steps
        episode_done = bool(dones.all() or self.step_count >= self.max_steps)

        self.current_obs = obs
        self.step_count += 1
        self.T += 1
        # return obs, rewards, dones, infos, episode_done
        return obs, rewards, dones, global_reward

    def get_neighbor_action(self, action):
        naction = []
        for i in range(self.n_agent):
            naction.append(action[self.neighbor_mask[i] == 1])
        return naction
    
    def update_fingerprint(self, fp):
        self.fp = fp
    
    def get_state(self):
        """拼接全局状态 (所有 agent obs 拼接)"""
        return np.concatenate(self.current_obs, axis=0)

    def get_avail_actions(self):
        """返回每个 agent 的可用动作 (连续: 全 1, 离散: one-hot)"""
        if self.continuous_actions:
            return np.ones((self.n_agent, self.n_actions))
        else:
            avail_actions = np.zeros((self.n_agent, self.n_actions))
            avail_actions[:, :] = 1
            return avail_actions

    def _create_neighbor_mask(self, n_agent):
        """邻居掩码，默认全 1（完全连通图）"""
        return np.ones((n_agent, n_agent), dtype=np.float32)

    def _create_distance_mask(self, n_agent):
        """距离掩码，默认全 1"""
        return np.ones((n_agent, n_agent), dtype=np.float32)

    def _extract_collision_info(self, info):
        """提取碰撞信息（若 VMAS info 有 'collisions' 字段）"""
        if "collisions" in info:
            return {k: v for k, v in info["collisions"].items()}
        return {}

    def _extract_goal_info(self, info):
        """提取目标达成信息（若 VMAS info 有 'goal_reached' 字段）"""
        if "goal_reached" in info:
            return {k: v for k, v in info["goal_reached"].items()}
        return {}
    
    def get_fingerprint(self):
        return self.fp

    def close(self):
        self.env.close()