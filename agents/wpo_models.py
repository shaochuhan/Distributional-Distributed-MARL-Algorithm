"""
Wasserstein Policy Optimization (WPO) Model Implementation
基于 Pfau et al. 2025 论文的 WPO 算法

集成到现有的 IA2C 框架中,与 DistributionalIA2C_CU 并行
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agents.wpo_policy import WPOConsensusPolicy
from agents.utils import OnPolicyBuffer, Scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class WPO_IA2C_CU(nn.Module):
    """
    Wasserstein Policy Optimization with Consensus Update for Multi-Agent RL
    
    主要特点:
    1. 使用 WPO 更新替代传统策略梯度
    2. 高斯策略,支持连续动作空间
    3. 共识更新机制,分布式学习
    4. 与现有 DistributionalIA2C_CU 完全独立
    """
    
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=True):
        super(WPO_IA2C_CU, self).__init__()
        
        self.name = 'wpo_ia2c_cu'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                       total_step, seed, use_gpu, model_config)
    
    def add_transition(self, ob, naction, action, reward, value, done):
        """添加转移数据"""
        if self.reward_norm > 0:
            reward = reward / self.reward_norm
        if self.reward_clip > 0:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        
        for i in range(self.n_agent):
            # value 可能是高斯分布参数 (mean, std),取均值
            if isinstance(value[i], tuple):
                value_estimate = value[i][0].item() if torch.is_tensor(value[i][0]) else value[i][0]
            else:
                value_estimate = value[i]
            
            self.trans_buffer[i].add_transition(
                ob[i], naction[i], action[i], reward, value_estimate, done
            )
    
    def backward(self, Rends, dt, summary_writer=None, global_step=None):
        """
        反向传播,使用 WPO 更新
        """
        self.optimizer.zero_grad()
        
        all_losses = []
        all_policy_losses = []
        all_value_losses = []
        all_entropy_losses = []
        all_kl_losses = []
        
        for i in range(self.n_agent):
            obs, nas, acts, dones, Rs, Advs = self.trans_buffer[i].sample_transition(
                Rends[i], dt
            )
            
            # 计算 WPO 损失
            # 注意: 现在返回std_reg_loss替代kl_loss (参考原论文简化版)
            policy_loss, value_loss, entropy_loss, std_reg_loss = self.policy.compute_wpo_loss(
                i, obs, nas, acts, dones, Rs, Advs, 
                self.e_coef, self.v_coef
            )
            
            # 总损失
            agent_loss = policy_loss + value_loss + entropy_loss + std_reg_loss
            all_losses.append(agent_loss)
            
            # 记录各部分损失
            all_policy_losses.append(policy_loss.item())
            all_value_losses.append(value_loss.item())
            all_entropy_losses.append(entropy_loss.item())
            all_kl_losses.append(std_reg_loss.item())  # 记录std正则化损失
        
        # 总损失
        total_loss = torch.stack(all_losses).sum()
        total_loss.backward()
        
        # 梯度裁剪
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        
        # 优化器步骤
        self.optimizer.step()
        
        # 共识更新
        self.policy.consensus_update()
        
        # 学习率衰减
        if self.lr_decay != 'constant':
            self._update_lr()
        
        # TensorBoard 记录
        if summary_writer is not None and global_step is not None:
            summary_writer.add_scalar('loss/wpo_policy_loss', 
                                     np.mean(all_policy_losses), global_step)
            summary_writer.add_scalar('loss/wpo_value_loss', 
                                     np.mean(all_value_losses), global_step)
            summary_writer.add_scalar('loss/wpo_entropy_loss', 
                                     np.mean(all_entropy_losses), global_step)
            summary_writer.add_scalar('loss/wpo_kl_loss', 
                                     np.mean(all_kl_losses), global_step)
            summary_writer.add_scalar('loss/wpo_total_loss', 
                                     total_loss.item(), global_step)
    
    def forward(self, obs, done, nactions=None, out_type='p'):
        """
        前向传播
        
        Returns:
            - out_type='p': (means, stds) - 高斯策略参数
            - out_type='v': values - 价值估计
        """
        if nactions is None:
            nactions = [None] * self.n_agent
        
        return self.policy.forward(obs, done, nactions, out_type)
    
    def load(self, model_dir, global_step=None, train_mode=False):
        """加载模型"""
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
            file_path = os.path.join(model_dir, save_file)
            checkpoint = torch.load(file_path)
            logging.info('Checkpoint found: {}'.format(file_path))
            
            try:
                # 尝试严格加载模型
                self.policy.load_state_dict(checkpoint['model_state_dict'], strict=True)
                logging.info('✓ Model loaded successfully (strict mode)')
                
                if train_mode:
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    except:
                        logging.warning('⚠ Failed to load optimizer state, using fresh optimizer')
                    self.policy.train()
                else:
                    self.policy.eval()
                return True
                
            except RuntimeError as e:
                error_msg = str(e)
                if 'size mismatch' in error_msg:
                    # 模型架构不匹配
                    logging.warning('=' * 70)
                    logging.warning('⚠ MODEL ARCHITECTURE MISMATCH DETECTED!')
                    logging.warning('=' * 70)
                    logging.warning('Checkpoint和当前模型的网络结构不匹配。')
                    logging.warning('这通常发生在以下情况:')
                    logging.warning('  1. 环境的智能体邻居配置发生了变化')
                    logging.warning('  2. 状态/动作空间维度改变')
                    logging.warning('  3. 使用了不同版本的配置文件')
                    logging.warning('')
                    logging.warning('错误详情:')
                    for line in error_msg.split('\n')[:5]:
                        logging.warning(f'  {line}')
                    logging.warning('')
                    logging.warning('解决方案:')
                    logging.warning('  → 将从头开始训练新模型')
                    logging.warning('  → 如需使用旧checkpoint，请确保环境配置一致')
                    logging.warning('=' * 70)
                    
                    # 从头开始
                    if train_mode:
                        self.policy.train()
                    else:
                        self.policy.eval()
                    return False
                else:
                    # 其他类型的RuntimeError，重新抛出
                    raise e
        
        logging.info('No checkpoint found in {}. Starting training from scratch.'.format(model_dir))
        return False
    
    def reset(self):
        """重置策略状态"""
        self.policy._reset()
    
    def save(self, model_dir, global_step):
        """保存模型"""
        file_path = os.path.join(model_dir, 'checkpoint-{:d}.pt'.format(global_step))
        torch.save({
            'global_step': global_step,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, file_path)
        logging.info('Checkpoint saved: {}'.format(file_path))
    
    def _init_algo(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                   total_step, seed, use_gpu, model_config):
        """初始化算法参数"""
        # 基本参数
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.identical_agent = False
        
        # 修复: 同时检查状态空间和动作空间是否相同
        # cacc_catchup环境中动作空间相同但状态空间不同(不同邻居数)
        if (max(self.n_a_ls) == min(self.n_a_ls) and 
            max(self.n_s_ls) == min(self.n_s_ls)):
            self.identical_agent = True
            self.n_s = n_s_ls[0]
            self.n_a = n_a_ls[0]
        else:
            self.identical_agent = False
            self.n_s = max(self.n_s_ls)
            self.n_a = max(self.n_a_ls)
        
        # 调试信息
        logging.info(f'WPO n_s_ls: {self.n_s_ls}')
        logging.info(f'WPO n_a_ls: {self.n_a_ls}')
        logging.info(f'WPO identical_agent: {self.identical_agent}')
        logging.info(f'WPO n_s: {self.n_s}, n_a: {self.n_a}')
        
        self.neighbor_mask = neighbor_mask
        self.n_agent = len(self.neighbor_mask)
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_step = model_config.getint('batch_size')
        self.n_fc = model_config.getint('num_fc')
        self.n_lstm = model_config.getint('num_lstm')
        
        # PyTorch 设置
        if use_gpu and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            self.device = torch.device("cuda:0")
            logging.info('WPO: Use GPU for PyTorch')
        else:
            torch.manual_seed(seed)
            torch.set_num_threads(1)
            self.device = torch.device("cpu")
            logging.info('WPO: Use CPU for PyTorch')
        
        # 初始化策略
        self.policy = self._init_policy()
        self.policy.to(self.device)
        
        # 训练参数
        if total_step:
            self.total_step = total_step
            self._init_train(model_config, distance_mask, coop_gamma)
    
    def _init_policy(self):
        """初始化 WPO 策略"""
        if self.identical_agent:
            policy = WPOConsensusPolicy(
                self.n_s, self.n_a, self.n_agent, self.n_step,
                self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm
            )
        else:
            policy = WPOConsensusPolicy(
                self.n_s, self.n_a, self.n_agent, self.n_step,
                self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls, identical=False
            )
        return policy
    
    def _init_scheduler(self, model_config):
        """初始化学习率调度器"""
        self.lr_init = model_config.getfloat('lr_init')
        self.lr_decay = model_config.get('lr_decay')
        
        if self.lr_decay == 'constant':
            self.lr_scheduler = Scheduler(self.lr_init, decay=self.lr_decay)
        else:
            lr_min = model_config.getfloat('lr_min')
            self.lr_scheduler = Scheduler(
                self.lr_init, lr_min, self.total_step, decay=self.lr_decay
            )
    
    def _init_train(self, model_config, distance_mask, coop_gamma):
        """初始化训练参数"""
        # 学习率调度器
        self._init_scheduler(model_config)
        
        # 损失系数
        self.v_coef = model_config.getfloat('value_coef')
        self.e_coef = model_config.getfloat('entropy_coef')
        self.max_grad_norm = model_config.getfloat('max_grad_norm')
        
        # 优化器 - WPO 论文推荐使用 Adam
        # 但为了与现有代码一致,这里使用 RMSprop
        alpha = model_config.getfloat('rmsp_alpha')
        epsilon = model_config.getfloat('rmsp_epsilon')
        self.optimizer = optim.RMSprop(
            self.policy.parameters(), 
            self.lr_init,
            eps=epsilon, 
            alpha=alpha
        )
        
        # Transition buffer
        gamma = model_config.getfloat('gamma')
        self.trans_buffer = []
        for i in range(self.n_agent):
            self.trans_buffer.append(
                OnPolicyBuffer(gamma, coop_gamma, distance_mask[i])
            )
    
    def _update_lr(self):
        """更新学习率"""
        lr = self.lr_scheduler.get(self.n_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
