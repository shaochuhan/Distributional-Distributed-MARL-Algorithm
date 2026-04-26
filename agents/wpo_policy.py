"""
Wasserstein Policy Optimization (WPO) for Multi-Agent Reinforcement Learning
Based on the paper: "Wasserstein Policy Optimization" (Pfau et al., 2025)

This implementation integrates WPO with the existing consensus-based multi-agent framework.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def one_hot(x, n_class):
    """One-hot encoding"""
    if isinstance(x, torch.Tensor):
        return F.one_hot(x.long(), num_classes=n_class).float()
    else:
        x = torch.as_tensor(x, dtype=torch.long, device=device)
        return F.one_hot(x, num_classes=n_class).float()


def init_layer(layer, layer_type):
    """Initialize layer parameters"""
    if layer_type == 'fc':
        nn.init.xavier_uniform_(layer.weight, gain=0.5)
        nn.init.constant_(layer.bias, 0)
    elif layer_type == 'lstm':
        for name, param in layer.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=0.5)
            elif 'bias' in name:
                nn.init.constant_(param, 0)


class WPOConsensusPolicy(nn.Module):
    """
    Wasserstein Policy Optimization with Consensus Update
    
    核心创新:
    1. 使用 WPO 更新替代传统策略梯度
    2. 结合共识机制进行分布式学习
    3. 支持高斯策略的均值和方差自适应调整
    """
    
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, 
                 n_fc=64, n_h=64, n_s_ls=None, n_a_ls=None, identical=True):
        super(WPOConsensusPolicy, self).__init__()
        
        self.name = 'wpo_cu'
        self.n_s = n_s
        self.n_a = n_a
        self.n_agent = n_agent
        self.n_step = n_step
        self.identical = identical
        self.neighbor_mask = torch.as_tensor(neighbor_mask, device=device)
        self.n_fc = n_fc
        self.n_h = n_h
        
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        
        # ===== WPO特定参数 (借鉴原论文配置) =====
        # 参考: acme/agents/jax/wpo/types.py L70-85
        # 原论文默认配置:
        # - epsilon_mean = 0.0025 (很小的KL约束)
        # - epsilon_stddev = 1e-6 (几乎不约束方差)
        # - dual_loss_scale = 0.0 (禁用dual优化)
        self.use_variance_rescaling = False  # 禁用手动Fisher缩放
        self.continuous_action = False  # 离散动作空间
        
        # 探索参数 (调整以提高稳定性)
        self.target_entropy_ratio = 0.4  # 从0.5降到0.4 (减少后期探索)
        self.std_target = 0.25  # 从0.3降到0.25 (更小的方差目标)
        self.entropy_reg_coef = 0.02  # 从0.01增到0.02 (更强的熵约束)
        self.std_reg_coef = 0.005  # 从0.001增到0.005 (更强的方差约束)
        
        self._init_net()
        self._reset()
    
    def _init_net(self):
        """初始化网络结构"""
        self.fc_x_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        
        # Actor heads: 输出高斯策略的均值和log标准差
        self.actor_mean_heads = nn.ModuleList()
        self.actor_logstd_heads = nn.ModuleList()
        
        # Critic heads: Q网络
        self.critic_heads = nn.ModuleList()
        
        self.na_ls_ls = []
        self.n_n_ls = []
        
        for i in range(self.n_agent):
            n_n, _, n_na, _, na_ls = self._get_neighbor_dim(i)
            n_s = self.n_s if self.identical else self.n_s_ls[i]
            n_a = self.n_a if self.identical else self.n_a_ls[i]
            
            self.na_ls_ls.append(na_ls)
            self.n_n_ls.append(n_n)
            
            # 特征提取层
            fc_x_layer = nn.Linear(n_s, self.n_fc).to(device)
            init_layer(fc_x_layer, 'fc')
            self.fc_x_layers.append(fc_x_layer)
            
            # LSTM层
            lstm_layer = nn.LSTMCell(self.n_fc, self.n_h).to(device)
            init_layer(lstm_layer, 'lstm')
            self.lstm_layers.append(lstm_layer)
            
            # Actor - 高斯策略的均值
            actor_mean = nn.Linear(self.n_h, n_a).to(device)
            nn.init.xavier_uniform_(actor_mean.weight, gain=0.01)  # Xavier初始化 - 自动调整初始值范围
            nn.init.constant_(actor_mean.bias, 0)
            self.actor_mean_heads.append(actor_mean)
            
            # Actor - 高斯策略的log标准差
            actor_logstd = nn.Linear(self.n_h, n_a).to(device)
            nn.init.constant_(actor_logstd.weight, 0.0)
            nn.init.constant_(actor_logstd.bias, -1.0)  
            self.actor_logstd_heads.append(actor_logstd)
            
            # Critic头
            critic_head = nn.Linear(self.n_h + n_na, 1).to(device)
            init_layer(critic_head, 'fc')
            self.critic_heads.append(critic_head)
    
    def _get_neighbor_dim(self, i_agent):
        """获取邻居维度信息"""
        n_n = int(self.neighbor_mask[i_agent].sum())
        if self.identical:
            return n_n, self.n_s * (n_n+1), self.n_a * n_n, [self.n_s] * n_n, [self.n_a] * n_n
        else:
            ns_ls = []
            na_ls = []
            for j in torch.where(self.neighbor_mask[i_agent])[0]:
                ns_ls.append(self.n_s_ls[j])
                na_ls.append(self.n_a_ls[j])
            return n_n, self.n_s_ls[i_agent] + sum(ns_ls), sum(na_ls), ns_ls, na_ls
    
    def _reset(self):
        """重置LSTM状态"""
        self.states_fw = []
        self.states_bw = []
        for i in range(self.n_agent):
            h = torch.zeros(1, self.n_h, device=device)
            c = torch.zeros(1, self.n_h, device=device)
            self.states_fw.append((h, c))
            self.states_bw.append((h, c))
    
    def _run_rnn(self, lstm_layer, xs, dones, state):
        """运行RNN层"""
        h, c = state
        
        # 确保dones有正确的维度并匹配xs的batch size
        xs_batch_size = xs.size(0)
        
        if isinstance(dones, torch.Tensor):
            dones_float = dones.float()
            if dones_float.dim() == 0:
                dones_float = dones_float.unsqueeze(0)
            # 确保dones的batch维度与xs匹配
            if dones_float.size(0) != xs_batch_size:
                if dones_float.size(0) == 1:
                    dones_float = dones_float.expand(xs_batch_size)
                else:
                    # 取对应的batch
                    dones_float = dones_float[:xs_batch_size]
        else:
            dones_float = torch.tensor([dones] * xs_batch_size, dtype=torch.float32, device=device)
        
        # 确保dones_float是1D张量
        if dones_float.dim() > 1:
            dones_float = dones_float.squeeze()
        if dones_float.dim() == 0:
            dones_float = dones_float.unsqueeze(0)
        
        # 确保h和c的batch维度正确(在重置之前)
        if h.dim() == 1:
            h = h.unsqueeze(0)  # [hidden_size] -> [1, hidden_size]
        if c.dim() == 1:
            c = c.unsqueeze(0)
        
        # 关键修复: 确保h的batch维度与dones_float匹配
        if h.size(0) != dones_float.size(0):
            if h.size(0) == 1 and dones_float.size(0) > 1:
                # h需要扩展到batch size
                h = h.expand(dones_float.size(0), -1)
                c = c.expand(dones_float.size(0), -1)
            elif dones_float.size(0) == 1 and h.size(0) > 1:
                # dones需要扩展
                dones_float = dones_float.expand(h.size(0))
            else:
                # 都不是1,取较小的那个作为batch size
                min_batch = min(h.size(0), dones_float.size(0))
                h = h[:min_batch]
                c = c[:min_batch]
                dones_float = dones_float[:min_batch]
        
        # 重置LSTM状态（当episode结束时）
        # 使用 (1 - dones) 来保持状态或清零
        h = h * (1 - dones_float.unsqueeze(-1))
        c = c * (1 - dones_float.unsqueeze(-1))
        
        # 准备输入到LSTMCell
        # xs的形状: [batch_size, input_size] = [1, 128]
        # h和c需要形状: [batch_size, hidden_size] = [1, 128]
        
        # 确保h和c的batch维度正确
        if h.dim() == 1:
            h = h.unsqueeze(0)  # [128] -> [1, 128]
        if c.dim() == 1:
            c = c.unsqueeze(0)  # [128] -> [1, 128]
        
        # 如果xs是[1, 128]但h是[batch, 128]且batch != 1，需要调整
        if xs.size(0) != h.size(0):
            if h.size(0) == 1:
                # h是[1, 128]，需要保持
                h = h.squeeze(0)  # [1, 128] -> [128]
                c = c.squeeze(0)  # [1, 128] -> [128]
            else:
                # xs需要扩展
                xs = xs.expand(h.size(0), -1)
        else:
            # batch size匹配，squeeze到1D
            h = h.squeeze(0)
            c = c.squeeze(0)
        
        # LSTMCell调用
        h_new, c_new = lstm_layer(xs.squeeze(0) if xs.size(0) == 1 else xs, (h, c))
        
        # 确保输出有batch维度
        if h_new.dim() == 1:
            h_new = h_new.unsqueeze(0)  # [128] -> [1, 128]
        if c_new.dim() == 1:
            c_new = c_new.unsqueeze(0)  # [128] -> [1, 128]
        
        return h_new, (h_new, c_new)
    
    def _run_comm_layers(self, obs, dones, states, agent_id=None):
        """运行通信层"""
        if agent_id is not None:
            # 单智能体模式
            obs_i = obs
            # 确保obs_i是float32类型的tensor
            if isinstance(obs_i, np.ndarray):
                obs_i = torch.as_tensor(obs_i, dtype=torch.float32, device=device)
            elif isinstance(obs_i, torch.Tensor):
                obs_i = obs_i.float().to(device)  # 确保是float32
            else:
                obs_i = torch.tensor(obs_i, dtype=torch.float32, device=device)
            
            if obs_i.dim() == 1:
                obs_i = obs_i.unsqueeze(0)
            
            if isinstance(dones, np.ndarray):
                dones = torch.as_tensor(dones, dtype=torch.float32, device=device)
            elif isinstance(dones, torch.Tensor):
                dones = dones.float().to(device)
            else:
                dones = torch.tensor(dones, dtype=torch.float32, device=device)
            
            if dones.dim() == 0:
                dones = dones.unsqueeze(0)
            
            xs_i = F.relu(self.fc_x_layers[agent_id](obs_i))
            state_i = states[agent_id] if isinstance(states, list) else states
            hs_i, new_states_i = self._run_rnn(self.lstm_layers[agent_id], xs_i, dones, state_i)
            
            h_new, c_new = new_states_i
            return hs_i, (h_new.detach(), c_new.detach())
        else:
            # 多智能体模式
            hs = []
            new_states = []
            
            for i in range(self.n_agent):
                obs_i = obs[i] if isinstance(obs, (list, tuple)) else obs
                
                # 确保obs_i是float32类型的tensor
                if isinstance(obs_i, np.ndarray):
                    obs_i = torch.as_tensor(obs_i, dtype=torch.float32, device=device)
                elif isinstance(obs_i, torch.Tensor):
                    obs_i = obs_i.float().to(device)  # 确保是float32
                else:
                    obs_i = torch.tensor(obs_i, dtype=torch.float32, device=device)
                
                if obs_i.dim() == 1:
                    obs_i = obs_i.unsqueeze(0)
                
                dones_tensor = dones
                if isinstance(dones_tensor, np.ndarray):
                    dones_tensor = torch.as_tensor(dones_tensor, dtype=torch.float32, device=device)
                elif isinstance(dones_tensor, torch.Tensor):
                    dones_tensor = dones_tensor.float().to(device)
                else:
                    dones_tensor = torch.tensor(dones_tensor, dtype=torch.float32, device=device)
                
                if dones_tensor.dim() == 0:
                    dones_tensor = dones_tensor.unsqueeze(0)
                 
                xs_i = F.relu(self.fc_x_layers[i](obs_i))
                state_i = states[i] if isinstance(states, list) else states
                hs_i, new_states_i = self._run_rnn(self.lstm_layers[i], xs_i, dones_tensor, state_i)
                
                if hs_i.dim() == 1:
                    hs_i = hs_i.unsqueeze(0)
                hs.append(hs_i)
                
                h_new, c_new = new_states_i
                if h_new.dim() == 1:
                    h_new = h_new.unsqueeze(0)
                if c_new.dim() == 1:
                    c_new = c_new.unsqueeze(0)
                new_states.append((h_new.detach(), c_new.detach()))
            
            hs_cat = torch.cat(hs, dim=0)
            return hs_cat, new_states
    
    def _run_critic_heads(self, hs, actions, detach=False):
        """运行Critic头"""
        vs = []
        
        # 确保actions的形状正确
        # actions应该是 [n_agents] 或 [batch_size, n_agents]
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)  # [n_agents] -> [1, n_agents]
        
        # 关键修复：从hs获取真实的batch size，而不是从actions
        # hs的形状: [batch_size, hidden] 或 [n_agents, hidden]
        if hs.size(0) == self.n_agent:
            # 多智能体模式，每个agent一个状态
            batch_size = 1
        else:
            # 批次训练模式
            batch_size = hs.size(0)
        
        # 确保actions的batch维度与hs匹配
        if actions.size(0) != batch_size:
            if actions.size(0) == 1 and batch_size > 1:
                # actions需要扩展
                actions = actions.expand(batch_size, -1)
            elif batch_size == 1 and actions.size(0) > 1:
                # 取第一个
                actions = actions[:1]
        
        for i in range(self.n_agent):
            n_n = self.n_n_ls[i]
            
            # 确保hs[i]有正确的维度 [batch_size, hidden_size]
            # hs可能是 [n_agents, hidden_size] 或 [batch_size, hidden_size]
            if hs.size(0) == self.n_agent:
                # 多智能体模式: hs = [n_agents, hidden_size]
                h_i_base = hs[i]
                if h_i_base.dim() == 1:
                    h_i_base = h_i_base.unsqueeze(0)
                # 如果batch_size > 1，需要扩展h_i_base
                if batch_size > 1 and h_i_base.size(0) == 1:
                    h_i_base = h_i_base.expand(batch_size, -1)
            else:
                # 单智能体训练或批次模式: hs = [batch_size, hidden_size]
                h_i_base = hs
                if h_i_base.dim() == 1:
                    h_i_base = h_i_base.unsqueeze(0)
                # 确保h_i_base的batch维度与actions匹配
                if h_i_base.size(0) != batch_size:
                    if h_i_base.size(0) == 1:
                        h_i_base = h_i_base.expand(batch_size, -1)
                    else:
                        # 如果不匹配且不是1，这可能是个错误，但我们尝试复用
                        # 这种情况发生在单智能体训练时
                        pass
            
            if n_n:
                # 获取邻居索引
                if isinstance(self.neighbor_mask, torch.Tensor):
                    neighbor_indices = np.where(self.neighbor_mask[i].cpu().numpy())[0]
                else:
                    neighbor_indices = np.where(self.neighbor_mask[i])[0]
                
                js = torch.from_numpy(neighbor_indices).long().to(device)
                
                # 从actions中选择邻居的动作
                # actions的形状: [batch_size, n_agents]
                # 我们需要在dim=1（agent维度）上选择
                na_i = torch.index_select(actions, 1, js)  # [batch_size, n_neighbors]
                
                na_i_ls = []
                for j in range(n_n):
                    # na_i[:, j] 获取第j个邻居的动作 [batch_size]
                    neighbor_action = na_i[:, j]  # 应该是 [batch_size]
                    
                    # 强制确保是1D张量
                    if neighbor_action.dim() > 1:
                        neighbor_action = neighbor_action.squeeze()
                    elif neighbor_action.dim() == 0:
                        neighbor_action = neighbor_action.unsqueeze(0)
                    
                    # one_hot编码: [batch_size] -> [batch_size, n_actions]
                    one_hot_action = one_hot(neighbor_action, self.na_ls_ls[i][j])
                    
                    # 确保one_hot_action是2D: [batch_size, n_actions]
                    if one_hot_action.dim() == 1:
                        one_hot_action = one_hot_action.unsqueeze(0)  # [n_actions] -> [1, n_actions]
                    
                    # 最终检查：必须匹配batch_size
                    if one_hot_action.size(0) != batch_size:
                        if one_hot_action.size(0) == 1 and batch_size > 1:
                            # 扩展到正确的batch size
                            one_hot_action = one_hot_action.expand(batch_size, -1)
                        elif batch_size == 1 and one_hot_action.size(0) > 1:
                            # 取第一个样本
                            one_hot_action = one_hot_action[:1]
                        else:
                            # 维度完全不匹配，强制重塑
                            # 这种情况不应该发生，但作为最后的保护
                            neighbor_action_expanded = neighbor_action.expand(batch_size) if neighbor_action.size(0) == 1 else neighbor_action
                            one_hot_action = one_hot(neighbor_action_expanded, self.na_ls_ls[i][j])
                    
                    na_i_ls.append(one_hot_action)
                
                # 最终安全检查：确保所有张量的第0维都匹配
                for idx, tensor in enumerate(na_i_ls):
                    if tensor.size(0) != h_i_base.size(0):
                        # 强制匹配
                        if tensor.size(0) == 1:
                            na_i_ls[idx] = tensor.expand(h_i_base.size(0), -1)
                        elif h_i_base.size(0) == 1:
                            na_i_ls[idx] = tensor[:1]
                        else:
                            # 都不是1，直接取batch_size的切片
                            na_i_ls[idx] = tensor[:h_i_base.size(0)]
                
                # 现在所有张量都是 [batch_size, feature_size]，可以在dim=1上concat
                h_i = torch.cat([h_i_base] + na_i_ls, dim=1)
            else:
                h_i = h_i_base.to(device)
            
            v_i = self.critic_heads[i](h_i).squeeze()
            if detach:
                vs.append(v_i.detach().cpu().numpy())
            else:
                vs.append(v_i)
        return vs
    
    def forward(self, obs, done, nactions=None, out_type='p'):
        """前向传播"""
        # 处理观测输入
        if isinstance(obs, (list, tuple)):
            # 多智能体观测列表
            obs_processed = obs
        elif isinstance(obs, np.ndarray):
            obs_processed = torch.from_numpy(obs).float().to(device)
            if obs_processed.dim() == 1:
                obs_processed = obs_processed.unsqueeze(0)
        else:
            obs_processed = obs
        
        # 处理done标志
        if isinstance(done, np.ndarray):
            done = torch.from_numpy(done).float().to(device)
        elif isinstance(done, (int, float)):
            done = torch.tensor(done, dtype=torch.float32, device=device)
        
        # 运行通信层
        h, new_states = self._run_comm_layers(obs_processed, done, self.states_fw)
        
        if out_type.startswith('p'):
            # 返回策略分布参数
            self.states_fw = new_states
            means = []
            stds = []
            for i in range(self.n_agent):
                mean = self.actor_mean_heads[i](h[i]).squeeze()
                logstd = self.actor_logstd_heads[i](h[i]).squeeze()
                std = torch.exp(logstd).clamp(min=0.01, max=2.0)
                means.append(mean.detach().cpu().numpy())
                stds.append(std.detach().cpu().numpy())
            return means, stds
        else:
            # 返回价值估计
            if nactions is None:
                nactions = [None] * self.n_agent
            action_tensor = torch.tensor([nactions], device=device).long()
            return self._run_critic_heads(h, action_tensor, detach=True)
    
    def compute_wpo_loss(self, agent_id, obs, nactions, acts, dones, Rs, Advs, 
                         e_coef, v_coef):
        """
        计算WPO损失
        
        核心WPO更新公式:
        ∇θ J = F^{-1} E[∇θ ∇a log π(a|s) ∇a Q(s,a)]
        
        使用简化的Fisher矩阵近似:
        - 均值: σ^2 ∇μ log π
        - 方差: 0.5 σ^2 ∇σ log π
        """
        obs = torch.from_numpy(obs).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)
        acts = torch.from_numpy(acts).long().to(device)
        Rs = torch.from_numpy(Rs).float().to(device)
        Advs = torch.from_numpy(Advs).float().to(device)
        
        # 前向传播
        hs, new_states = self._run_comm_layers(obs, dones, self.states_bw, agent_id=agent_id)
        self.states_bw[agent_id] = new_states
        
        # 获取策略分布参数
        mean = self.actor_mean_heads[agent_id](hs)
        logstd = self.actor_logstd_heads[agent_id](hs)
        std = torch.exp(logstd).clamp(min=0.01, max=2.0)
        
        # 创建高斯分布
        dist = torch.distributions.Normal(mean, std)
        
        # 采样动作用于WPO更新
        n_samples = 5  # 每个状态采样动作数
        sampled_actions = dist.sample((n_samples,))  # [n_samples, batch_size, n_a]
        
        # 根据动作空间类型选择不同的Q值计算方式
        if self.continuous_action:
            # ===== 连续动作空间：使用原始WPO算法 =====
            # 计算Q值梯度 ∇_a Q(s,a)
            q_grads = []
            for k in range(n_samples):
                actions_k = sampled_actions[k].detach().requires_grad_(True)  # [batch_size, n_a]
                
                # 计算Q值
                h_with_action = torch.cat([hs, actions_k], dim=-1)
                q_k = self.critic_heads[agent_id](h_with_action).squeeze()
                
                # 计算梯度 ∇_a Q(s,a)
                q_k_sum = q_k.sum()
                q_grad_k = torch.autograd.grad(
                    outputs=q_k_sum,
                    inputs=actions_k,
                    create_graph=False,
                    retain_graph=True
                )[0]
                q_grads.append(q_grad_k)
            
            q_grad = torch.stack(q_grads).mean(dim=0)  # [batch_size, n_a]
            
            # WPO梯度更新（连续动作的原始公式）
            # ∇θ J = F^{-1} E[∇θ ∇_a log π(a|s) · ∇_a Q(s,a)]
            if self.use_variance_rescaling:
                # Fisher矩阵预条件: σ^2 * 梯度
                mean_grad_scale = (std ** 2)  #Fisher逆
                mean_wpo_grad = q_grad * mean_grad_scale
            else:
                mean_wpo_grad = q_grad
            
            # 策略损失（连续WPO）
            policy_loss = -(mean * mean_wpo_grad.detach()).mean()
            
            # 方差的WPO梯度
            sampled_mean = sampled_actions.mean(dim=0)
            std_wpo_grad = ((sampled_mean - mean) / (std + 1e-8)) * q_grad
            if self.use_variance_rescaling:
                std_wpo_grad = std_wpo_grad * (0.5 * std ** 2)
            std_loss = -(logstd * std_wpo_grad.detach()).mean()
            policy_loss = policy_loss + std_loss
            
        else:
            # ===== 离散动作空间：使用策略梯度近似 =====
            # 计算log概率
            log_probs = dist.log_prob(sampled_actions)  # [n_samples, batch_size, n_a]
            
            # 计算Q值估计（用于评估采样动作的好坏）
            
            q_values = []
            batch_size = hs.size(0)  # 获取实际的batch size
            
            for k in range(n_samples):
                actions_k = sampled_actions[k]  # [batch_size, n_a]
                
                # 将连续动作映射到离散动作索引
                discrete_acts = torch.argmax(actions_k, dim=-1, keepdim=True)  # [batch_size, 1]
                
                # 为所有智能体创建动作张量
                # 注意：这里需要为每个batch样本都创建完整的n_agent维度
                all_actions = torch.zeros(batch_size, self.n_agent, dtype=torch.long, device=device)
                all_actions[:, agent_id] = discrete_acts.squeeze(-1)
                
                # 计算Q值（不需要梯度，只是用来评估）
                with torch.no_grad():
                    vs_k = self._run_critic_heads(hs, all_actions, detach=False)
                    q_k = vs_k[agent_id] if isinstance(vs_k, list) else vs_k
                    q_values.append(q_k)
            
            q_values = torch.stack(q_values)  # [n_samples, batch_size]
            
            # WPO启发的策略更新（适配离散动作空间）
            # 使用Q值加权的策略梯度
            
            # 计算优势函数（相对于平均Q值）
            q_mean = q_values.mean(dim=0, keepdim=True)  # [1, batch_size]
            advantages = q_values - q_mean  # [n_samples, batch_size]
            
            # ===== 借鉴原论文: 使用标准策略梯度而非手动Fisher缩放 =====
            # 原论文使用natural_gradient_adaptor自动处理Fisher信息矩阵
            # 参考: acme/jax/losses/wpo.py L209-218
            # 这里简化为标准REINFORCE with baseline
            weighted_log_prob = (log_probs.sum(dim=-1) * advantages.detach()).mean(dim=0)  # [batch_size]
            mean_wpo_loss = -weighted_log_prob.mean()
            
            # 策略损失(基于WPO思想的策略梯度)
            policy_loss = mean_wpo_loss
            
            # ===== 借鉴原论文: 使用熵约束而非Q方差约束 =====
            # 原论文通过epsilon_stddev控制方差变化
            # 参考: acme/agents/jax/wpo/types.py L70-85
            # 这里使用熵作为探索指标，更稳定
            # 目标熵 = log(n_actions) * 0.5 (保持适度探索)
            target_entropy = np.log(self.n_a) * 0.5
            current_entropy = dist.entropy().mean()
            
            # 方差损失: 鼓励策略保持一定的探索性
            # 当熵过低时增加损失，防止过早收敛
            entropy_reg_loss = torch.clamp(target_entropy - current_entropy, min=0.0) ** 2 * 0.01
            policy_loss = policy_loss + entropy_reg_loss
        
        # ===== 借鉴原论文: 简化KL约束 =====
        # 原论文默认dual_loss_scale=0.0，即禁用KL双重损失
        # 参考: acme/agents/jax/wpo/types.py L82
        # 这里使用标准差的L2正则化来稳定训练
        # 防止标准差过大或过小
        std_mean = std.mean()
        std_reg_loss = ((std_mean - 0.3) ** 2) * 0.001  # 目标std约为0.3
        
        # 熵损失(鼓励探索) - 原始代码保留
        entropy = dist.entropy().mean()
        entropy_loss = -entropy * e_coef
        
        # Critic损失(TD误差)
        if nactions is not None:
            # nactions是所有智能体的动作 [batch_size, n_agents] 或 [n_agents]
            if isinstance(nactions, np.ndarray):
                if nactions.ndim == 1:
                    # 单个样本 [n_agents] -> [1, n_agents]
                    nactions_tensor = torch.from_numpy(nactions).long().to(device).unsqueeze(0)
                else:
                    # 批次样本 [batch_size, n_agents]
                    nactions_tensor = torch.from_numpy(nactions).long().to(device)
            else:
                # 假设是列表
                nactions_array = np.array(nactions)
                if nactions_array.ndim == 1:
                    nactions_tensor = torch.from_numpy(nactions_array).long().to(device).unsqueeze(0)
                else:
                    nactions_tensor = torch.from_numpy(nactions_array).long().to(device)
            
            vs = self._run_critic_heads(hs, nactions_tensor, detach=False)
            v = vs[agent_id] if isinstance(vs, list) else vs
        else:
            # 使用当前智能体的动作
            discrete_acts = torch.argmax(mean, dim=-1, keepdim=True)
            all_actions = torch.zeros(discrete_acts.size(0), self.n_agent, dtype=torch.long, device=device)
            all_actions[:, agent_id] = discrete_acts.squeeze(-1)
            vs = self._run_critic_heads(hs, all_actions, detach=False)
            v = vs[agent_id] if isinstance(vs, list) else vs
        
        value_loss = ((Rs - v) ** 2).mean() * v_coef
        
        # ===== 总损失组合 =====
        # 借鉴原论文: policy_loss + entropy_loss + value_loss
        # 参考: acme/jax/losses/wpo.py L242-246
        # 移除了kl_loss（原论文默认dual_loss_scale=0）
        total_loss = policy_loss + entropy_loss + std_reg_loss + value_loss
        
        # 返回各项损失用于日志记录
        return policy_loss, value_loss, entropy_loss, std_reg_loss
    
    def consensus_update(self):
        """共识更新,平均化邻居的LSTM参数"""
        with torch.no_grad():
            for i in range(self.n_agent):
                mean_wts = self._get_critic_wts(i)
                for param, wt in zip(self.lstm_layers[i].parameters(), mean_wts):
                    param.copy_(wt)
    
    def _get_critic_wts(self, i_agent):
        """获取邻居的Critic权重用于共识更新"""
        wts = []
        
        # 收集当前agent的LSTM权重
        for wt in self.lstm_layers[i_agent].parameters():
            wts.append(wt.detach().clone())
        
        # 收集邻居的权重
        neighbors = list(np.where(self.neighbor_mask[i_agent].cpu().numpy() == 1)[0])
        for j in neighbors:
            for k, wt in enumerate(self.lstm_layers[j].parameters()):
                wts[k] += wt.detach().clone()
        
        # 平均化
        n = 1 + len(neighbors)
        for k in range(len(wts)):
            wts[k] /= n
        
        return wts
