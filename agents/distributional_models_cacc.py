"""
Distributional Multi-Agent Reinforcement Learning with Consensus-based Critic Update

本模块实现了基于值分布的多智能体强化学习算法。

共识更新方法：simple_distributional_consensus
- 简单高效，适合大多数场景
- 基于预测不确定性的置信度加权
- 自动过滤不确定的预测

核心思想：
1. 使用邻居预测的不确定性（标准差sigma）作为权重
2. 不确定性低的预测获得更高的权重
3. 这样可以自然地过滤掉不确定或异常的预测
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import copy
import torch.optim.lr_scheduler as lr_scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Critic(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim=64):
        super(Critic,self).__init__()
        # 构建基础网络：先将state和action拼接作为输入
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        # 最后的输出层：一个输出均值，一个输出log(std)
        self.mu = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # 将状态和动作拼接，经过隐藏层计算特征
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 通过两个头部输出均值和log(标准差)
        mu = self.mu(x)                             # Q值分布的均值
        log_sigma = self.log_std(x)                 # Q值分布的log标准差
        sigma = torch.exp(log_sigma)          # 从log(σ)计算标准差
        return mu, sigma


class Actor(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 输出层：动作的概率分布
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # 输出层，生成动作的logits
        action_probs = self.fc3(x)  # （后续softmax）
        return action_probs  

# class Actor(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim=128):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, hidden_dim//2)
#         self.fc4 = nn.Linear(hidden_dim//2, output_dim)
#         self.dropout = nn.Dropout(0.1)
        
#     def forward(self, state):
#         x = F.relu(self.fc1(state))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc3(x))
#         action_logits = self.fc4(x)
#         return action_logits


class distributional_CACC_agent:
    def __init__(self,agent_id,n_agents,n_states,n_actions,slow_lr, fast_lr,gamma=0.95,global_state_dim=None):
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.n_states = n_states  # 智能体局部状态维度
        self.n_actions = n_actions
        self.gamma = gamma
        self.global_state_dim = global_state_dim

        critic_input_dim = global_state_dim + n_agents
        self.critic = Critic(input_dim=critic_input_dim,output_dim=1).to(device)
        self.actor = Actor(input_dim=n_states * n_agents,output_dim=n_actions).to(device)

        self.gamma = gamma
        self.n_actions = n_actions
        self.fast_lr = fast_lr
        self.slow_lr = slow_lr

        'Adam'
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=slow_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=fast_lr)

        self.critic_scheduler = lr_scheduler.StepLR(self.critic_optimizer, step_size=2000, gamma=0.95) 
        self.actor_scheduler = lr_scheduler.StepLR(self.actor_optimizer, step_size=3000, gamma=0.95) 

        self.beta = 0.9  # 手动设置的动量因子
        self.critic_momentum_buffer = {name: torch.zeros_like(param) for name, param in self.critic.named_parameters() if param.requires_grad}
        self.actor_momentum_buffer = {name: torch.zeros_like(param) for name, param in self.actor.named_parameters() if param.requires_grad}

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def critic_update(self,states,next_states,joint_actions,next_joint_actions,local_reward):
        states = states.clone().detach().to(device)

        # subarray = [torch.stack(list(temp)) for temp in next_states.flat]
        # next_states = torch.stack(subarray, dim=0).view(next_states.shape[0],next_states.shape[1], -1)
        next_states = next_states.clone().detach().to(device)

        # subarray = [torch.stack(list(temp)) for temp in joint_actions.flat]
        # joint_actions = torch.stack(subarray, dim=0).view(joint_actions.shape[0],joint_actions.shape[1], -1)
        joint_actions = joint_actions.clone().detach().to(device)
        next_joint_actions = next_joint_actions.clone().detach().to(device)
        local_reward = local_reward.clone().detach().to(device)

        # 保存更新前的 Critic 网络权重
        critic_weights_temp = self.critic.state_dict()

        # 拼接状态和动作作为输入
        state_action = torch.cat((states, joint_actions.float()), dim=1)
        next_state_action = torch.cat((next_states, next_joint_actions.float()), dim=1) #下一时间的状态动作对

        # 1. 当前Critic预测
        mu_pred, sigma_pred = self.critic(state_action)

        # *** 关键修复：确保sigma_pred的数值稳定性 ***
        sigma_pred = torch.clamp(sigma_pred, min=1e-6, max=10.0)  # 限制sigma范围

        # 2. 计算目标值分布参数（使用目标网络，不参与梯度）
        with torch.no_grad():
            mu_next, sigma_next = self.critic(next_state_action)

            # *** 关键修复：确保sigma_next的数值稳定性 ***
            sigma_next = torch.clamp(sigma_next, min=1e-4, max=5.0)

            # 计算目标分布的均值和标准差
            mu_target = local_reward + self.gamma * mu_next
            # 如果done=1，则mu_target = reward，此时希望sigma_target趋近0
            sigma_target = self.gamma * sigma_next + 1e-3
            sigma_target = torch.clamp(sigma_target, min=1e-3, max=5.0)  # 限制范围

        # 更稳定的KL散度计算
        sigma_ratio = torch.clamp(sigma_pred / sigma_target, min=0.1, max=10.0)
        log_sigma_ratio = torch.log(sigma_ratio)
        mu_diff = torch.clamp(mu_target - mu_pred, min=-5, max=5)    
        # 添加熵正则化项
        entropy_reg = -0.01 * torch.log(sigma_pred + 1e-6).mean()    
        kl_div = log_sigma_ratio + \
                (sigma_target.pow(2) + mu_diff.pow(2)) / (2 * sigma_pred.pow(2)) - 0.5    
        kl_div = torch.clamp(kl_div, min=-5, max=5)
        critic_loss = kl_div.mean() + entropy_reg


        # 计算梯度并应用优化
        self.critic_optimizer.zero_grad()  # 清除梯度缓存
        critic_loss.backward()  # 反向传播

        # '梯度裁剪'
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()  # 更新网络权重
        # 获取更新后的网络参数
        critic_weights = self.critic.state_dict()
        # 重置 Critic 网络为更新前的参数
        self.critic.load_state_dict(critic_weights_temp)
        # 更新后进行软更新
        # self.soft_update_target_critic(tau=0.001)  # 非常缓慢的更新

        return critic_weights, critic_loss.item()

    def actor_update(self,states,joint_actions,local_actions):
        states = states.clone().detach().to(device)
        joint_actions = joint_actions.clone().detach().to(device)
        local_actions = local_actions.clone().detach().to(device)

        # 拼接状态和动作作为输入
        state_action = torch.cat((states, joint_actions.float()), dim=1)

        '''critic的输出是(mu, sigma)'''
        # Q = self.critic(state_action).detach()                # Q(s, a)
        mu, std = self.critic(state_action)
        Q = mu.detach()

        with torch.no_grad():
            # *** 关键修复：添加数值检查 ***
            actor_logits = self.actor(states)        
            # 检查actor输出是否有效
            if torch.any(torch.isnan(actor_logits)) or torch.any(torch.isinf(actor_logits)):
                print("Warning: Invalid actor logits in actor_update")
                return self.actor.state_dict(), 0.0
            # 限制logits范围
            actor_logits = torch.clamp(actor_logits, min=-10, max=10)
            all_action_probs = torch.softmax(actor_logits, dim=1)
            # 检查概率是否有效
            if torch.any(torch.isnan(all_action_probs)) or torch.any(all_action_probs <= 0):
                print("Warning: Invalid action probabilities")
                return self.actor.state_dict(), 0.0
        
            # 当前策略对所有动作的概率
            # all_action_probs = torch.softmax(self.actor(states), dim=1)  # [batch_size, n_actions]
            q_values_per_action = []
            # 循环遍历所有可能的动作
            for a in range(self.n_actions):
                # 创建所有样本中第 a 个动作的 tensor
                a_tensor = torch.full((states.shape[0], 1), a, dtype=torch.long, device=device)
                # 拷贝一份 joint_action，用来替换当前 agent 的动作为 a
                joint_action_copy = joint_actions.clone()
                joint_action_copy[:, self.agent_id] = a
                # 构建 critic 输入：状态 + 替换后的 joint_action
                sa_all = torch.cat([states, joint_action_copy.float()], dim=1)
                # 得到 Q(s, a)
                # q_val = self.critic(sa_all).detach()
                q_mu, q_std = self.critic(sa_all)
                q_val = q_mu.detach()
                # q_values_per_action.append(q_val)
                # *** 关键修复：检查Q值有效性 ***
                if torch.any(torch.isnan(q_val)) or torch.any(torch.isinf(q_val)):
                    print(f"Warning: Invalid Q values for action {a}")
                    q_val = torch.zeros_like(q_val)  # 使用0作为备用            
                q_values_per_action.append(q_val)

            # 拼成 [batch_size, n_actions]，每一列是一个动作对应的 Q 值
            q_values_per_action = torch.stack(q_values_per_action, dim=1).squeeze(-1)
            # 计算 V(s) = E_{a ~ pi}[Q(s,a)]，即所有 Q * 动作概率的加权平均
            V = (all_action_probs * q_values_per_action).sum(dim=1, keepdim=True)

            # 检查V值有效性
            if torch.any(torch.isnan(V)) or torch.any(torch.isinf(V)):
                print("Warning: Invalid V values")
                V = torch.zeros_like(Q)
        
        A = Q -V 
        # *** 关键修复：限制优势函数范围 ***
        A = torch.clamp(A, min=-10, max=10)

        '''
        约束 global_TD_error 的范围： 使用 torch.clamp 限制 TD 误差的范围:
        '''
        # global_TD_error = torch.clamp(global_TD_error, min=-10, max=10)

        # 对每个状态对应的动作价值用log函数
        action_logits = self.actor(states)

        action_logits = torch.clamp(action_logits, min=-10, max=10)
        action_probs = torch.softmax(action_logits,dim=1)
        '避免梯度异常：'
        # action_probs = action_probs.clamp(min=1e-8, max=1.0) 
        action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
        temp_actions = local_actions.clone().detach().view(-1, 1)
        # log_probs = torch.log(action_probs.gather(1, temp_actions))
        '''
        对 action_probs 加小值偏移： 避免在 torch.log 中输入零或接近零的值：
        '''
        log_probs = torch.log(action_probs.gather(1, temp_actions) + 1e-8)
        
        # *** 关键修复：检查log概率有效性 ***
        if torch.any(torch.isnan(log_probs)) or torch.any(torch.isinf(log_probs)):
            print("Warning: Invalid log probabilities")
            return self.actor.state_dict(), 0.0
    
        actor_loss = torch.mean(-log_probs * A.detach())

        # *** 关键修复：检查损失有效性 ***
        if torch.isnan(actor_loss) or torch.isinf(actor_loss):
            print("Warning: Invalid actor loss, skipping update")
            return self.actor.state_dict(), 0.0
    
        # 计算梯度并应用优化
        self.actor_optimizer.zero_grad()  # 清除梯度缓存
        # actor_loss.requires_grad_(True)
        actor_loss.backward()  # 反向传播

        '梯度裁剪'
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(),max_norm=10.)  # 设置梯度的最大范数

        # *** 关键修复：严格的梯度裁剪 ***
        # total_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        total_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.1)    
        # 如果梯度过大，跳过更新
        if total_norm > 15.0:
            print(f"Warning: Large actor gradient norm {total_norm:.2f}, skipping update")
            self.actor_optimizer.zero_grad()
            return self.actor.state_dict(), actor_loss.item()    

        self.actor_optimizer.step()  # 更新网络权重
        # 获取更新后的网络参数
        actor_weights = self.actor.state_dict()
        # 重置 actor 网络为更新前的参数
        # self.critic.load_state_dict(critic_weights_temp)

        return actor_weights,actor_loss.item()


    def consensus_critic(self, critic_weights_innodes, states_batch, actions_batch):
        num_neighbors = len(critic_weights_innodes)
        if num_neighbors == 0:
            return
        
        # 如果没有评估数据，使用简单平均
        if states_batch is None or actions_batch is None:
            aggregated_weights = {k: torch.zeros_like(v) for k, v in self.critic.state_dict().items()}
            for neighbor_weights in critic_weights_innodes:
                for key, value in neighbor_weights.items():
                    aggregated_weights[key] += value.detach()
            for key in aggregated_weights:
                aggregated_weights[key] /= num_neighbors
            self.critic.load_state_dict(aggregated_weights)
            return
        
        # 临时保存当前权重
        current_weights = self.critic.state_dict()
        
        # 评估每个邻居的预测质量（基于不确定性）
        confidence_scores = []
        
        for neighbor_weights in critic_weights_innodes:
            self.critic.load_state_dict(neighbor_weights)
            with torch.no_grad():
                state_action = torch.cat((states_batch, actions_batch.float()), dim=1)
                mu, sigma = self.critic(state_action)
                
                # 计算置信度：不确定性的倒数
                confidence = 1.0 / (sigma.mean() + 1e-4)
                confidence_scores.append(confidence.item())
        
        # 归一化置信度得分
        total_confidence = sum(confidence_scores)
        weights = [c / total_confidence for c in confidence_scores]
        
        # 基于置信度加权平均网络参数
        aggregated_weights = {k: torch.zeros_like(v) for k, v in current_weights.items()}
        
        for i, neighbor_weights in enumerate(critic_weights_innodes):
            for key, value in neighbor_weights.items():
                aggregated_weights[key] += weights[i] * value.detach()
        
        # 加载加权平均后的参数
        self.critic.load_state_dict(aggregated_weights)



    # def consensus_critic(self,critic_weights_innodes):
    #     # 初始化一个存储合并参数的字典，结构与 Critic 的 state_dict 一致
    #     aggregated_weights = {k: torch.zeros_like(v) for k, v in self.critic.state_dict().items()}
    #     num_neighbors = len(critic_weights_innodes)
    
    #     # 对邻居权重逐层加和
    #     for neighbor_weights in critic_weights_innodes:
    #         for key, value in neighbor_weights.items():
    #             aggregated_weights[key] += value.detach()  
    
    #     # 求均值作为共识后的新权重
    #     for key in aggregated_weights:
    #         aggregated_weights[key] /= num_neighbors
        
    #     self.critic.load_state_dict(aggregated_weights)



    # 改进的动作选择
    def get_action(self, state, from_policy=False, mu=0.1):
        random_action = np.random.choice(self.n_actions)
        if from_policy:
            state_tensor = state.clone().detach().unsqueeze(0).reshape(1, -1)
        
            with torch.no_grad():  # 避免不必要的梯度计算
                logits = self.actor(state_tensor)
            
                # 添加温度参数来控制探索
                temperature = max(0.5, 2.0 * np.exp(-mu * 10))  # 随着mu减小，temperature也减小
                logits = logits / temperature
            
                logits = torch.clamp(logits, min=-10, max=10)
                action_probs = torch.softmax(logits, dim=1)
            
                # 添加噪声增加探索
                if mu > 0.02:  # 只在探索率较高时添加噪声
                    noise = torch.randn_like(action_probs) * 0.01
                    action_probs = torch.softmax(logits + noise, dim=1)
            
                action_probs = action_probs.squeeze().detach().cpu().numpy()
                action_probs = np.clip(action_probs, 1e-8, 1.0)
                action_probs = action_probs / action_probs.sum()  # 重新归一化
            
            action_from_policy = np.random.choice(self.n_actions, p=action_probs)
            self.action = np.random.choice([action_from_policy, random_action], p=[1 - mu, mu])
        else:
            self.action = random_action
    
        return self.action

    # def get_action(self, state, from_policy=False, mu=0.1):
    #     if from_policy:
    #         state_tensor = state.clone().detach().unsqueeze(0).reshape(1, -1)
        
    #         with torch.no_grad():
    #             logits = self.actor(state_tensor)
            
    #             # 使用温度系数和噪声注入
    #             temperature = max(0.3, 1.0 * np.exp(-mu * 5))
    #             logits = logits / temperature
            
    #             # 添加Gumbel噪声增强探索
    #             if mu > 0.01:
    #                 gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    #                 logits = logits + 0.1 * gumbel_noise
            
    #             action_probs = torch.softmax(logits, dim=1)
    #             action_probs = action_probs.squeeze().detach().cpu().numpy()
            
    #         # 动态epsilon-greedy
    #         if np.random.random() < mu:
    #             action = np.random.choice(self.n_actions)
    #         else:
    #             action = np.random.choice(self.n_actions, p=action_probs)
            
    #     else:
    #         action = np.random.choice(self.n_actions)
    
    #     return action   