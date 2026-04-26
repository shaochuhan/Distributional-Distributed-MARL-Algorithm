"""
VMAS Individual Reward Environment Adapter
为D2AC_CU算法提供"奖励私有"版本的VMAS场景

关键特性：
- 状态全局可见：所有智能体可以观察彼此的状态
- 动作全局可见：所有智能体可以观察彼此的动作
- 奖励私有：每个智能体只知道自己的奖励
"""

import numpy as np
import torch
from envs.vmas_unified_env import VmasUnifiedEnv


class VmasIndividualRewardEnv(VmasUnifiedEnv):
    """
    VMAS环境的个体奖励适配器
    
    将原始VMAS的共享奖励转换为个体奖励，适配D2AC_CU算法
    """
    
    def __init__(self, config):
        """
        初始化环境
        
        Args:
            config: ConfigParser对象，包含环境配置
        """
        super().__init__(config)
        
        # 奖励转换模式
        self.reward_mode = config.get('reward_mode', 'individual')  # 'individual' or 'contribution'
        
        # 记录上一步的状态（用于计算个体贡献）
        self.last_positions = None
        self.last_goal_distances = None
        
        print(f"[VmasIndividualRewardEnv] 使用奖励模式: {self.reward_mode}")
        print(f"[VmasIndividualRewardEnv] 场景: {self.scenario_name}")
    
    def reset(self):
        """重置环境"""
        obs = super().reset()
        
        # 重置状态记录
        self._update_state_tracking()
        
        return obs
    
    def _update_state_tracking(self):
        """更新状态追踪信息"""
        # 记录智能体位置
        self.last_positions = []
        for agent in self.vmas_env.agents:
            pos = agent.state.pos[0].cpu().numpy()  # [x, y]
            self.last_positions.append(pos.copy())
        
        # 记录到目标的距离（用于计算贡献）
        self.last_goal_distances = []
        if self.scenario_name == 'dropout':
            # Dropout: 一个共享目标
            goal_pos = self.vmas_env.world.landmarks[0].state.pos[0].cpu().numpy()
            for agent_pos in self.last_positions:
                dist = np.linalg.norm(agent_pos - goal_pos)
                self.last_goal_distances.append(dist)
        elif self.scenario_name == 'dispersion':
            # Dispersion: 每个智能体有自己的最近目标
            for agent_pos in self.last_positions:
                min_dist = float('inf')
                for landmark in self.vmas_env.world.landmarks:
                    goal_pos = landmark.state.pos[0].cpu().numpy()
                    dist = np.linalg.norm(agent_pos - goal_pos)
                    if dist < min_dist:
                        min_dist = dist
                self.last_goal_distances.append(min_dist)
    
    def step(self, actions):
        """
        执行动作，并转换奖励为个体奖励
        
        Args:
            actions: list of arrays/ints, 每个智能体的动作
        
        Returns:
            obs: list of arrays, 观测
            rewards: list of floats, 个体奖励（私有）
            done: bool, 是否结束
            global_reward: float, 全局奖励（仅用于日志）
        """
        # 执行原始步骤
        obs, shared_rewards, done, global_reward = super().step(actions)
        
        # 转换为个体奖励
        individual_rewards = self._compute_individual_rewards(
            shared_rewards, actions, obs, done
        )
        
        # 更新状态追踪
        if not done:
            self._update_state_tracking()
        
        return obs, individual_rewards, done, global_reward
    
    def _compute_individual_rewards(self, shared_rewards, actions, obs, done):
        """
        将共享奖励转换为个体奖励
        
        Args:
            shared_rewards: list of floats, 原始共享奖励
            actions: list, 智能体动作
            obs: list, 当前观测
            done: bool, 是否结束
        
        Returns:
            list of floats, 个体奖励
        """
        if self.scenario_name == 'dropout':
            return self._dropout_individual_rewards(shared_rewards, actions, done)
        elif self.scenario_name == 'dispersion':
            return self._dispersion_individual_rewards(shared_rewards, actions, done)
        elif self.scenario_name == 'navigation':
            return self._navigation_individual_rewards(shared_rewards, actions, done)
        else:
            # 默认：使用原始奖励
            print(f"[警告] 场景 {self.scenario_name} 未实现个体奖励转换，使用原始奖励")
            return shared_rewards
    
    def _dropout_individual_rewards(self, shared_rewards, actions, done):
        """
        Dropout场景的个体奖励
        
        原始奖励：团队共享 +1（有人到达）- 能量惩罚
        个体奖励：只有到达的智能体获得 +1，其他智能体只有能量惩罚
        """
        individual_rewards = []
        
        # 检查哪个智能体到达了目标
        goal_pos = self.vmas_env.world.landmarks[0].state.pos[0].cpu().numpy()
        goal_reached_agents = []
        
        for i, agent in enumerate(self.vmas_env.agents):
            agent_pos = agent.state.pos[0].cpu().numpy()
            dist_to_goal = np.linalg.norm(agent_pos - goal_pos)
            
            # 判断是否到达（阈值：0.1）
            if dist_to_goal < 0.1:
                goal_reached_agents.append(i)
        
        # 计算个体奖励
        for i in range(self.n_agent):
            # 能量惩罚（私有）
            action_magnitude = np.linalg.norm(self._get_action_vector(actions[i]))
            energy_penalty = -self.energy_coeff * action_magnitude
            
            # 任务奖励（私有）
            if i in goal_reached_agents:
                task_reward = 1.0
            else:
                task_reward = 0.0
            
            individual_rewards.append(task_reward + energy_penalty)
        
        return individual_rewards
    
    def _dispersion_individual_rewards(self, shared_rewards, actions, done):
        """
        Dispersion场景的个体奖励
        
        原始奖励：团队共享（目标被到达）
        个体奖励：每个智能体根据自己是否到达目标获得奖励
        """
        individual_rewards = []
        
        # 检查每个智能体是否到达了某个目标
        for i, agent in enumerate(self.vmas_env.agents):
            agent_pos = agent.state.pos[0].cpu().numpy()
            agent_reward = 0.0
            
            # 检查所有目标
            for landmark in self.vmas_env.world.landmarks:
                goal_pos = landmark.state.pos[0].cpu().numpy()
                dist_to_goal = np.linalg.norm(agent_pos - goal_pos)
                
                # 如果到达目标（阈值：0.1）
                if dist_to_goal < 0.1:
                    agent_reward += 1.0
                    break  # 每个智能体只奖励一次
            
            # 时间惩罚（可选）
            time_penalty = -0.01  # 每步 -0.01
            agent_reward += time_penalty
            
            individual_rewards.append(agent_reward)
        
        return individual_rewards
    
    def _navigation_individual_rewards(self, shared_rewards, actions, done):
        """
        Navigation场景的个体奖励
        
        原始奖励：可能是共享的（取决于shared_rew参数）
        个体奖励：每个智能体根据自己是否到达目标获得奖励
        """
        individual_rewards = []
        
        # Navigation场景：每个智能体有自己的目标
        for i, agent in enumerate(self.vmas_env.agents):
            agent_pos = agent.state.pos[0].cpu().numpy()
            
            # 获取该智能体的目标
            goal = agent.goal
            goal_pos = goal.state.pos[0].cpu().numpy()
            
            # 计算到目标的距离
            dist_to_goal = np.linalg.norm(agent_pos - goal_pos)
            
            # 奖励：负距离（鼓励接近目标）
            reward = -dist_to_goal
            
            # 如果到达目标（阈值：0.1）
            if dist_to_goal < 0.1:
                reward += 10.0  # 到达目标的额外奖励
            
            individual_rewards.append(reward)
        
        return individual_rewards
    
    def _get_action_vector(self, action):
        """
        从动作中提取向量（用于计算能量）
        
        Args:
            action: int or array, 动作
        
        Returns:
            numpy array, 动作向量
        """
        if isinstance(action, (int, np.integer)):
            # 离散动作：映射到连续空间
            from envs.vmas_unified_env import discrete_to_continuous_action
            return discrete_to_continuous_action(action, self.vmas_action_size)
        elif isinstance(action, np.ndarray):
            return action
        elif isinstance(action, torch.Tensor):
            return action.cpu().numpy()
        else:
            return np.array([0.0, 0.0])
    
    def get_observation_with_global_info(self):
        """
        获取增强观测：包含全局状态和动作信息
        
        用于满足"状态和动作全局可见"的要求
        
        Returns:
            list of arrays, 每个智能体的增强观测
        """
        # 获取基础观测
        base_obs = self._process_obs(self.vmas_env.get_from_scenario('observation'))
        
        # 收集全局信息
        all_positions = []
        all_velocities = []
        for agent in self.vmas_env.agents:
            pos = agent.state.pos[0].cpu().numpy()
            vel = agent.state.vel[0].cpu().numpy()
            all_positions.append(pos)
            all_velocities.append(vel)
        
        # 为每个智能体添加全局信息
        enhanced_obs = []
        for i in range(self.n_agent):
            # 自己的观测
            own_obs = base_obs[i]
            
            # 其他智能体的位置和速度（全局可见）
            other_info = []
            for j in range(self.n_agent):
                if i != j:
                    other_info.append(all_positions[j])
                    other_info.append(all_velocities[j])
            
            # 合并
            if other_info:
                enhanced = np.concatenate([own_obs] + other_info)
            else:
                enhanced = own_obs
            
            enhanced_obs.append(enhanced)
        
        return enhanced_obs


def create_individual_reward_env(config):
    """
    工厂函数：创建个体奖励环境
    
    Args:
        config: ConfigParser对象
    
    Returns:
        VmasIndividualRewardEnv实例
    """
    return VmasIndividualRewardEnv(config)
