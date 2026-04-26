"""
IA2C and MA2C algorithms
@author: Tianshu Chu
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from agents.utils import OnPolicyBuffer, MultiAgentOnPolicyBuffer, Scheduler
from agents.policies import (LstmPolicy, FPPolicy, ConsensusPolicy, NCMultiAgentPolicy,
                             CommNetMultiAgentPolicy, DIALMultiAgentPolicy,DistributionalConsensusPolicy)
from agents.wpo_models import WPO_IA2C_CU  # WPO 算法导入
import logging
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class IA2C:
class IA2C(nn.Module):##^
    """
    The basic IA2C implementation with decentralized actor and centralized critic,
    limited to neighborhood area only.
    使用去中心化的actor和中心化的critic
    """
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=True):
        super(IA2C, self).__init__()##+
        self.name = 'ia2c'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)
        
    #将状态转移添加到智能体的转移缓冲区中
    def add_transition(self, ob, naction, action, reward, value, done):
        # 支持reward为列表（每个agent单独的奖励）或单个值（所有agent共享）
        if isinstance(reward, (list, tuple, np.ndarray)):
            rewards = reward
        else:
            rewards = [reward] * self.n_agent
        
        # 应用reward normalization和clipping
        if self.reward_norm > 0:
            rewards = [r / self.reward_norm for r in rewards]
        if self.reward_clip > 0:
            rewards = [np.clip(r, -self.reward_clip, self.reward_clip) for r in rewards]
        
        for i in range(self.n_agent):
            self.trans_buffer[i].add_transition(ob[i], naction[i], action[i], rewards[i], value[i], done)

    # 更新优化器的参数，并根据学习率衰减策略更新学习率
    def backward(self, Rends, dt, summary_writer=None, global_step=None):
        self.optimizer.zero_grad()
        for i in range(self.n_agent):
            obs, nas, acts, dones, Rs, Advs = self.trans_buffer[i].sample_transition(Rends[i], dt)
            if i == 0:
                self.policy[i].backward(obs, nas, acts, dones, Rs, Advs,
                                        self.e_coef, self.v_coef,
                                        summary_writer=summary_writer, global_step=global_step)
            else:
                self.policy[i].backward(obs, nas, acts, dones, Rs, Advs,
                                        self.e_coef, self.v_coef)
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.lr_decay != 'constant':
            self._update_lr()

    #执行前向传播，生成智能体的动作输出
    def forward(self, obs, done, nactions=None, out_type='p'):
        out = []
        if nactions is None:
            nactions = [None] * self.n_agent
        for i in range(self.n_agent): 
            cur_out = self.policy[i](obs[i], done, nactions[i], out_type)
            out.append(cur_out)
        return out

    #加载模型检查点，如果存在，则恢复模型和优化器的状态
    def load(self, model_dir, global_step=None, train_mode=False):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if global_step is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        tokens = file.split('.')[0].split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = file
                            save_step = cur_step
            else:
                save_file = 'checkpoint-{:d}.pt'.format(global_step)
        if save_file is not None:
            # file_path = model_dir + save_file
            file_path = os.path.join(model_dir, save_file)
            checkpoint = torch.load(file_path)
            logging.info('Checkpoint loaded: {}'.format(file_path))
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            if train_mode:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.policy.train()
            else:
                self.policy.eval()
            return True
        logging.error('Can not find checkpoint for {}'.format(model_dir))
        return False

    def reset(self):
        for i in range(self.n_agent):
            self.policy[i]._reset()

    #保存当前模型和优化器状态到指定目录
    def save(self, model_dir, global_step):
        import os
        file_path = os.path.join(model_dir, 'checkpoint-{:d}.pt'.format(global_step))
        torch.save({'global_step': global_step,
                    'model_state_dict': self.policy.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                    file_path)
        logging.info('Checkpoint saved: {}'.format(file_path))

    #初始化算法的各种参数，包括状态和动作的维度、邻居关系、随机种子、设备选择等
    def _init_algo(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                   total_step, seed, use_gpu, model_config):
        # init params
        use_gpu = True
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.identical_agent = False
        if (max(self.n_a_ls) == min(self.n_a_ls)):
            # note for identical IA2C, n_s_ls may have varient dims
            self.identical_agent = True
            self.n_s = n_s_ls[0]
            self.n_a = n_a_ls[0]
        else:
            self.n_s = max(self.n_s_ls)
            self.n_a = max(self.n_a_ls)
        self.neighbor_mask = neighbor_mask
        self.n_agent = len(self.neighbor_mask)
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_step = model_config.getint('batch_size')
        self.n_fc = model_config.getint('num_fc')
        self.n_lstm = model_config.getint('num_lstm')
        # init torch
        if use_gpu and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            self.device = torch.device("cuda:0")
            logging.info('Use gpu for pytorch...')
        else:
            torch.manual_seed(seed)
            torch.set_num_threads(1)
            self.device = torch.device("cpu")
            logging.info('Use cpu for pytorch...x')

        self.policy = self._init_policy()
        self.policy.to(self.device)
        
        # init exp buffer and lr scheduler for training
        if total_step:
            self.total_step = total_step
            self._init_train(model_config, distance_mask, coop_gamma)

    #初始化智能体的策略，创建 LstmPolicy 实例
    def _init_policy(self):
        policy = []
        for i in range(self.n_agent):
            n_n = np.sum(self.neighbor_mask[i])
            if self.identical_agent:
                local_policy = LstmPolicy(self.n_s_ls[i], self.n_a_ls[i], n_n, self.n_step,
                                          n_fc=self.n_fc, n_lstm=self.n_lstm, name='{:d}'.format(i))
            else:
                na_dim_ls = []
                for j in np.where(self.neighbor_mask[i] == 1)[0]:
                    na_dim_ls.append(self.n_a_ls[j])
                local_policy = LstmPolicy(self.n_s_ls[i], self.n_a_ls[i], n_n, self.n_step,
                                          n_fc=self.n_fc, n_lstm=self.n_lstm, name='{:d}'.format(i),
                                          na_dim_ls=na_dim_ls, identical=False)
                # local_policy.to(self.device)
            policy.append(local_policy)
        return nn.ModuleList(policy)

    def _init_scheduler(self, model_config):
        # init lr scheduler
        self.lr_init = model_config.getfloat('lr_init')
        self.lr_decay = model_config.get('lr_decay')
        if self.lr_decay == 'constant':
            self.lr_scheduler = Scheduler(self.lr_init, decay=self.lr_decay)
        else:
            lr_min = model_config.getfloat('lr_min')
            self.lr_scheduler = Scheduler(self.lr_init, lr_min, self.total_step, decay=self.lr_decay)

    def _init_train(self, model_config, distance_mask, coop_gamma):
        # init lr scheduler
        self._init_scheduler(model_config)
        # init parameters for grad computation
        self.v_coef = model_config.getfloat('value_coef')
        self.e_coef = model_config.getfloat('entropy_coef')
        self.max_grad_norm = model_config.getfloat('max_grad_norm')
        # init optimizer
        alpha = model_config.getfloat('rmsp_alpha')
        epsilon = model_config.getfloat('rmsp_epsilon')
        self.optimizer = optim.RMSprop(self.policy.parameters(), self.lr_init, 
                                       eps=epsilon, alpha=alpha)
        # init transition buffer
        gamma = model_config.getfloat('gamma')
        self._init_trans_buffer(gamma, distance_mask, coop_gamma)

    def _init_trans_buffer(self, gamma, distance_mask, coop_gamma):
        self.trans_buffer = []
        for i in range(self.n_agent):
            # init replay buffer
            self.trans_buffer.append(OnPolicyBuffer(gamma, coop_gamma, distance_mask[i]))

    def _update_lr(self):
        # TODO: refactor this using optim.lr_scheduler
        cur_lr = self.lr_scheduler.get(self.n_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr


class IA2C_FP(IA2C):
    """
    In fingerprint IA2C, neighborhood policies (fingerprints) are also included.
    """
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=True):
        # super().__init__()  # 先调用 nn.Module 的构造函数
        super(IA2C_FP,self).__init__(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                                     total_step, model_config)  ##+
        self.name = 'ia2c_fp'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma, 
                        total_step, seed, use_gpu, model_config)

    def _init_policy(self):
        policy = []
        for i in range(self.n_agent):
            n_n = np.sum(self.neighbor_mask[i])
            # neighborhood policies are included in local state
            if self.identical_agent:
                n_s1 = int(self.n_s_ls[i] + self.n_a*n_n)
                policy.append(FPPolicy(n_s1, self.n_a, int(n_n), self.n_step, n_fc=self.n_fc,
                                       n_lstm=self.n_lstm, name='{:d}'.format(i)))
            else:
                na_dim_ls = []
                for j in np.where(self.neighbor_mask[i] == 1)[0]:
                    na_dim_ls.append(self.n_a_ls[j])
                n_s1 = int(self.n_s_ls[i] + sum(na_dim_ls))
                policy.append(FPPolicy(n_s1, self.n_a_ls[i], int(n_n), self.n_step, n_fc=self.n_fc,
                                       n_lstm=self.n_lstm, name='{:d}'.format(i),
                                       na_dim_ls=na_dim_ls, identical=False))
        return nn.ModuleList(policy)


class MA2C_NC(IA2C):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0,  use_gpu=True):
        super(MA2C_NC,self).__init__(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                                     total_step, model_config)  ##:+
        self.name = 'ma2c_nc'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)

    #用于将当前的状态转移添加到转移缓冲区
    def add_transition(self, ob, p, action, reward, value, done):
        #对奖励进行归一化
        if self.reward_norm > 0:
            # reward = torch.tensor(reward, dtype=torch.float32, device=device) / self.reward_norm
            reward = reward / self.reward_norm
        #对奖励进行截断
        if self.reward_clip > 0:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)

        # 将 reward, value, done 转移至 GPU
        reward = torch.as_tensor(reward, device=device)  ## 修改：确保 reward 在 GPU 上
        # value = torch.as_tensor(value, device=device)    # 修改：确保 value 在 GPU 上
        done = torch.as_tensor(done, device=device)      # 修改：确保 done 在 GPU 上

        #如果所有智能体的状态和动作相同（identical_agent），直接添加转移
        # if self.identical_agent:
        #     self.trans_buffer.add_transition(np.array(ob), np.array(p), action,
        #                                      reward, value, done)##
        if self.identical_agent:
            # self.trans_buffer.add_transition(
            #     torch.as_tensor(np.array(ob), device=device),  # 修改：转移 ob
            #     torch.as_tensor(np.array(p), device=device),    # 修改：转移 p
            #     torch.as_tensor(np.array(p.cpu()), device=device),
            #     action, reward, value, done
            # )
            self.trans_buffer.add_transition(
                ob,  # 修改：转移 ob
                p.to(device),    # 修改：转移 p
                action, reward, value, done
            )
        else:
            pad_ob, pad_p = self._convert_hetero_states(ob, p)
            self.trans_buffer.add_transition(pad_ob, pad_p, action,
                                             reward, value, done)

    def backward(self, Rends, dt, summary_writer=None, global_step=None):
        self.optimizer.zero_grad()
        ## Rs是累计折扣奖励，Advs是优势值
        obs, ps, acts, dones, Rs, Advs = self.trans_buffer.sample_transition(Rends, dt)
        
        self.policy.backward(obs, ps, acts, dones, Rs, Advs, self.e_coef, self.v_coef,
                             summary_writer=summary_writer, global_step=global_step)
        #如果设置了最大梯度范数，则进行梯度裁剪
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        #执行优化器的参数更新，并根据需要更新学习率
        self.optimizer.step()
        if self.lr_decay != 'constant':
            self._update_lr()

    def forward(self, obs, done, ps, actions=None, out_type='p'):

        # ps = torch.as_tensor(ps).to(device)
        if actions is not None:
            actions = torch.as_tensor(actions, device=device)  # 修改：转移 actions
            # actions = actions

        if self.identical_agent:
            if isinstance(obs, list):
                if isinstance(obs[0], torch.Tensor):
                    obs = [o.cpu().numpy() for o in obs]
            return self.policy.forward(obs, done, ps.cpu().numpy(),
                                       actions, out_type)
            # return self.policy.forward(obs, done, ps, actions, out_type)
        else:
            pad_ob, pad_p = self._convert_hetero_states(obs, ps)
            return self.policy.forward(pad_ob, done, pad_p,
                                       actions, out_type)

    def reset(self):
        self.policy._reset()

    def _convert_hetero_states(self, ob, p):
        # pad_ob = np.zeros((self.n_agent, self.n_s))
        # pad_p = np.zeros((self.n_agent, self.n_a))
        pad_ob = torch.zeros((self.n_agent, self.n_s), device=device)  ## 修改：初始化在 GPU 上
        pad_p = torch.zeros((self.n_agent, self.n_a), device=device)   # 修改：初始化在 GPU 上
        for i in range(self.n_agent):
            # pad_ob[i, :len(ob[i])] = ob[i]
            # pad_p[i, :len(p[i])] = p[i]
            pad_ob[i, :len(ob[i])] = torch.as_tensor(ob[i], device=device)  ## 修改：转移到 GPU
            pad_p[i, :len(p[i])] = torch.as_tensor(p[i], device=device)      # 修改：转移到 GPU
        return pad_ob, pad_p

    def _init_policy(self):
        if self.identical_agent:
            return NCMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                      self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm)
        else:
            return NCMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                      self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                      n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls, identical=False)

    def _init_trans_buffer(self, gamma, distance_mask, coop_gamma):
        self.trans_buffer = MultiAgentOnPolicyBuffer(gamma, coop_gamma, distance_mask)


class IA2C_CU(MA2C_NC):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=True):
        super(IA2C_CU,self).__init__(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                                     total_step, model_config)  # 调用父类 MA2C_NC 的初始化方法##:+
        
        self.name = 'ma2c_cu'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config) ##-

    def _init_policy(self):
        #如果智能体是同质的，则返回一个统一的策略
        if self.identical_agent:
            policy =  ConsensusPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                   self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm)
        #如果智能体是异质的，则创建一个 ConsensusPolicy 实例
        else:
            policy =  ConsensusPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                   self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                   n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls, identical=False)
        policy = policy.to(device)
        return policy


    def backward(self, Rends, dt, summary_writer=None, global_step=None):
        # Rends = torch.as_tensor(Rends, dtype=torch.float32).to(device)
        # dt = torch.as_tensor(dt, dtype=torch.float32).to(device)
        super(IA2C_CU, self).backward(Rends, dt, summary_writer, global_step)
        self.policy.consensus_update()


class MA2C_CNET(MA2C_NC):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=True):
        super(MA2C_CNET,self).__init__(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                                     total_step, model_config)  ##:+
        self.name = 'ma2c_ic3'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)

    def _init_policy(self):
        if self.identical_agent:
            return CommNetMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                           self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm)
        else:
            return CommNetMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                           self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                           n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls, identical=False)


class MA2C_DIAL(MA2C_NC):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=True):
        super(MA2C_DIAL,self).__init__(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                                     total_step, model_config)  ##:+
        self.name = 'ma2c_dial'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)

    def _init_policy(self):
        if self.identical_agent:
            return DIALMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                        self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm)
        else:
            return DIALMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                        self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                        n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls, identical=False)





class DistributionalIA2C_CU(IA2C):
    """
    分布式IA2C_CU实现，在IA2C_CU基础上增加分布式Critic网络
    """
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=True):
        self.name = 'distributional_ia2c_cu'
        super(DistributionalIA2C_CU, self).__init__(n_s_ls, n_a_ls, neighbor_mask, 
                                                    distance_mask, coop_gamma, total_step, 
                                                    model_config, seed, use_gpu)
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)
        
    def reset(self):
        """重置策略状态 - 修复subscriptable错误"""
        self.policy._reset()

    def forward(self, obs, done, nactions=None, out_type='p'):
        """前向传播 - 适配单一策略对象"""
        if nactions is None:
            nactions = [None] * self.n_agent
        return self.policy.forward(obs, done, nactions, out_type)


    ########################
    # def forward(self, obs, done, nactions=None, out_type='p', agent_id=None):
    #     """前向传播
    #     Args:
    #         obs: 观测数据
    #             - 训练时: 单个agent的obs [batch_size, obs_dim]
    #             - 推理时: 所有agents的obs列表 [[obs0], [obs1], ...]
    #         done: done标志
    #         nactions: 可用动作
    #         out_type: 'p' (策略) 或 'v' (价值)
    #         agent_id: 
    #             - None: 推理模式，处理所有agents
    #             - int: 训练模式，只处理指定agent
    #     """
    #     if nactions is None:
    #         nactions = [None] * self.n_agent
        
    #     if agent_id is not None:
    #         # 训练模式：单个agent
    #         return self.policy.forward(obs, done, nactions, out_type, agent_id)
    #     else:
    #         # 推理模式：所有agents
    #         out = []
    #         for i in range(self.n_agent):
    #             obs_i = obs[i] if isinstance(obs, list) else obs
    #             nactions_i = nactions[i] if isinstance(nactions, list) else nactions
    #             result = self.policy.forward(obs_i, done, nactions_i, out_type, agent_id=i)
    #             out.append(result)
    #         return out
        

    def _init_policy(self):
        """初始化分布式共识策略"""
        self.identical_agent = False
        if self.identical_agent:
            policy = DistributionalConsensusPolicy(
                self.n_s, self.n_a, self.n_agent, self.n_step,
                self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm
            )
        else:
            policy = DistributionalConsensusPolicy(
                self.n_s, self.n_a, self.n_agent, self.n_step,
                self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls, identical=False
            )
        policy = policy.to(device)
        return policy
    

    def backward(self, Rends, dt, summary_writer=None, global_step=None):
        """反向传播，包含分布式损失计算和共识更新"""
        self.optimizer.zero_grad()
        
        all_losses = []
        
        # 重要：在每次backward开始前，确保清理所有可能的状态
        self._reset_internal_states()  # 你需要实现这个方法

        for i in range(self.n_agent):
            obs, nas, acts, dones, Rs, Advs = self.trans_buffer[i].sample_transition(Rends[i], dt)
            
            # 计算分布式损失
            actor_loss, critic_loss, entropy_loss = self.policy.compute_distributional_loss(
                i, obs, nas, acts, dones, Rs, Advs, self.e_coef, self.v_coef
            )
            
            # 直接存储总损失，避免中间累加
            agent_loss = actor_loss + critic_loss + entropy_loss
            all_losses.append(agent_loss)

            if i == 0 and summary_writer is not None:
                summary_writer.add_scalar('loss/actor_loss', actor_loss.item(), global_step)
                summary_writer.add_scalar('loss/critic_loss', critic_loss.item(), global_step)
                summary_writer.add_scalar('loss/entropy_loss', entropy_loss.item(), global_step)

        total_loss = torch.stack(all_losses).sum()
        total_loss.backward(retain_graph=True)

        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        # 执行共识更新
        self.policy.consensus_update()
        
        if self.lr_decay != 'constant':
            self._update_lr()


    def _reset_internal_states(self):
        """重置内部状态，防止计算图问题"""
        # 如果有LSTM层，重置其隐藏状态
        if hasattr(self, 'lstm_hidden_states'):
            self.lstm_hidden_states = None  
        # 如果有其他状态变量，也在这里重置
        if hasattr(self.policy, 'reset_states'):
            self.policy.reset_states()  
        # 清理可能的缓存
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def add_transition(self, ob, naction, action, reward, value, done):
        """添加转移数据 - 适配分布式价值"""
        if self.reward_norm > 0:
            reward = reward / self.reward_norm
        if self.reward_clip > 0:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        for i in range(self.n_agent):
            # value现在是分布参数(mu, sigma)，取均值作为价值估计
            if isinstance(value[i], tuple):
                value_estimate = value[i][0].item()  # 取均值
            else:
                value_estimate = value[i]
            self.trans_buffer[i].add_transition(ob[i], naction[i], action[i], reward, value_estimate, done)
