import torch
import numpy as np
import vmas


class VmasNavigationEnv:
    """
    兼容旧版接口的 VMAS Navigation 封装
    - step() 返回: (obs, rewards, dones, infos, episode_done)
    - 支持 get_state(), get_avail_actions()
    - 在 info 中加入 collision / goal_reached 信息
    """

    def __init__(self, scenario_name="navigation", n_agents=3, device="cpu", continuous_actions=True):
        self.scenario_name = scenario_name
        self.n_agents = n_agents
        self.device = device
        self.continuous_actions = continuous_actions

        # 初始化 VMAS 环境
        self.env = vmas.make_env(
            scenario=scenario_name,
            num_envs=1,
            device=device,
            continuous_actions=continuous_actions
        )

        # 取 action/observation 维度
        self.n_actions = self.env.action_space[0].shape[0] if continuous_actions else self.env.action_space[0].n
        self.n_obs = self.env.observation_space[0].shape[0]

        # 🆕 补全算法需要的属性
        self.n_a_ls = [self.n_actions for _ in range(n_agents)]
        self.n_s_ls = [self.n_obs for _ in range(n_agents)]

        # 掩码
        self.neighbor_mask = self._create_neighbor_mask(n_agents)
        self.distance_mask = self._create_distance_mask(n_agents)

        # 当前观测
        self.current_obs = None

    def reset(self):
        obs, _ = self.env.reset()
        obs = obs[0]  # [n_agents, obs_dim]
        self.current_obs = [o.cpu().numpy() for o in obs]
        return self.current_obs

    def step(self, actions):
        """
        参数:
            actions: list/np.ndarray, shape = [n_agents, action_dim]
        返回:
            obs, rewards, dones, infos, episode_done
        """
        # 转成 tensor
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device).unsqueeze(0)
        obs, rewards, dones, info = self.env.step(actions)

        # VMAS 返回的是 batched 数据，需要取第 0 个环境
        obs = obs[0]          # [n_agents, obs_dim]
        rewards = rewards[0]  # [n_agents]
        dones = dones[0]      # [n_agents]
        info = info[0]

        # 转 numpy
        obs = [o.cpu().numpy() for o in obs]
        rewards = rewards.cpu().numpy()
        dones = dones.cpu().numpy().astype(bool)

        # 加入扩展统计信息
        collision_info = self._extract_collision_info(info)
        goal_info = self._extract_goal_info(info)

        infos = [{"collision_info": collision_info, "goal_reached": goal_info} for _ in range(self.n_agents)]

        # episode_done 取全局 done（VMAS 通常用 all_done 标记）
        episode_done = bool(dones.all())

        self.current_obs = obs
        return obs, rewards, dones, infos, episode_done

    def get_state(self):
        """拼接全局状态 (所有 agent obs 拼接)"""
        return np.concatenate(self.current_obs, axis=0)

    def get_avail_actions(self):
        """返回每个 agent 的可用动作 (连续: 全 1, 离散: one-hot)"""
        if self.continuous_actions:
            return np.ones((self.n_agents, self.n_actions))
        else:
            avail_actions = np.zeros((self.n_agents, self.n_actions))
            avail_actions[:, :] = 1
            return avail_actions

    def _create_neighbor_mask(self, n_agents):
        """邻居掩码，默认全 1（完全连通图）"""
        return np.ones((n_agents, n_agents), dtype=np.float32)

    def _create_distance_mask(self, n_agents):
        """距离掩码，默认全 1"""
        return np.ones((n_agents, n_agents), dtype=np.float32)

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

    def close(self):
        self.env.close()
