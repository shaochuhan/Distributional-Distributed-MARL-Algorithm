import numpy as np
import logging
from typing import List, Union, Optional
import torch


class MaliciousAgentWrapper:
    """
    Args:
        base_env: 原始环境实例
        malicious_agents: 恶意智能体ID列表，如[0, 2]
        malicious_type: 恶意行为类型
        seed: 随机种子（可选）
        log_interval: 日志记录间隔，0表示不记录详细日志
    """
    # 支持的恶意行为类型
    SUPPORTED_BEHAVIORS = [
        'random',      
        'opposite',    
        'fixed_0',     
        'fixed_max',   
        'min_prob',    
    ]
    
    def __init__(self, base_env, malicious_agents: Optional[List[int]] = None, 
                 malicious_type: str = 'random', seed: Optional[int] = None,
                 log_interval: int = 1000):
        self.base_env = base_env
        self.malicious_agents = malicious_agents if malicious_agents else []
        self.malicious_type = malicious_type
        self.log_interval = log_interval
        self.step_count = 0
        self.malicious_action_count = 0
        
        # 验证恶意行为类型
        if self.malicious_type not in self.SUPPORTED_BEHAVIORS:
            raise ValueError(f"Unsupported malicious type: {self.malicious_type}. "
                           f"Supported types: {self.SUPPORTED_BEHAVIORS}")
        
        # 验证恶意智能体ID
        if self.malicious_agents:
            max_agent_id = max(self.malicious_agents)
            if hasattr(base_env, 'n_agent') and max_agent_id >= base_env.n_agent:
                raise ValueError(f"Invalid malicious agent ID: {max_agent_id}. "
                               f"Environment has {base_env.n_agent} agents (0-{base_env.n_agent-1})")
        
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
            
        # 记录初始化日志
        if self.malicious_agents:
            logging.info(f'MaliciousAgentWrapper initialized: '
                        f'agents={self.malicious_agents}, type={self.malicious_type}')
        
    def step(self, actions):
        """
        重写step函数，修改恶意智能体的动作
        Args:
            actions: 原始动作列表/数组
        Returns:
            环境step的返回值（观察、奖励、done、info）
        """
        if not self.malicious_agents:
            return self.base_env.step(actions)
            
        # 复制动作以避免修改原始数据
        if isinstance(actions, (list, tuple)):
            modified_actions = list(actions)
        elif isinstance(actions, np.ndarray):
            modified_actions = actions.copy()
        elif isinstance(actions, torch.Tensor):
            modified_actions = actions.clone().detach().cpu().numpy()  # 转成 numpy，方便修改
            modified_actions = modified_actions.tolist()  # 转成 list，保证后续赋值可行
        
        # 修改恶意智能体的动作
        for agent_id in self.malicious_agents:
            if agent_id < len(modified_actions):
                original_action = modified_actions[agent_id]
                malicious_action = self._get_malicious_action(agent_id, original_action)
                modified_actions[agent_id] = malicious_action
                self.malicious_action_count += 1
                
                # 详细日志记录（可选）
                if self.log_interval > 0 and self.step_count % self.log_interval == 0:
                    logging.debug(f'Step {self.step_count}: Agent {agent_id} '
                                f'action changed from {original_action} to {malicious_action}')
        
        self.step_count += 1

        # 定期统计日志
        if (self.log_interval > 0 and self.step_count % self.log_interval == 0 
            and self.malicious_action_count > 0):
            logging.info(f'Malicious actions taken: {self.malicious_action_count} '
                        f'out of {self.step_count * len(self.malicious_agents)} possible')
        
        return self.base_env.step(modified_actions)
    
    def _get_malicious_action(self, agent_id: int, original_action: int) -> int:
        """
        根据恶意类型生成恶意动作
        
        Args:
            agent_id: 智能体ID
            original_action: 原始动作
            
        Returns:
            恶意动作
        """
        n_actions = self.base_env.n_a_ls[agent_id]
        
        if self.malicious_type == 'random':
            return np.random.randint(0, n_actions)
            
        elif self.malicious_type == 'opposite':
            # 选择与原动作"相反"的动作
            return (original_action + n_actions // 2) % n_actions
            
        elif self.malicious_type == 'fixed_0':
            return 0
            
        elif self.malicious_type == 'fixed_max':
            return n_actions - 1
            
        elif self.malicious_type == 'min_prob':
            # 选择与原动作不同的随机动作
            if n_actions > 1:
                possible_actions = [a for a in range(n_actions) if a != original_action]
                return np.random.choice(possible_actions) if possible_actions else original_action
            return original_action
        
        return original_action
    
    def reset(self, **kwargs):
        """重置环境"""
        self.step_count = 0
        self.malicious_action_count = 0
        return self.base_env.reset(**kwargs)
    
    def get_malicious_stats(self) -> dict:
        """
        获取恶意智能体统计信息
        
        Returns:
            包含统计信息的字典
        """
        return {
            'malicious_agents': self.malicious_agents,
            'malicious_type': self.malicious_type,
            'total_steps': self.step_count,
            'malicious_actions_taken': self.malicious_action_count,
            'malicious_ratio': (self.malicious_action_count / 
                              max(1, self.step_count * len(self.malicious_agents)))
        }
    
    def __getattr__(self, name):
        """将其他属性和方法委托给原始环境"""
        return getattr(self.base_env, name)


def apply_malicious_wrapper(env, malicious_agents_str: str, malicious_type: str = 'random',
                          seed: Optional[int] = None, log_interval: int = 1000):
    """
    便捷函数：将恶意智能体包装器应用到环境
    
    Args:
        env: 原始环境
        malicious_agents_str: 恶意智能体ID字符串，用逗号分隔，如"0,2"
        malicious_type: 恶意行为类型
        seed: 随机种子
        log_interval: 日志间隔
        
    Returns:
        包装后的环境（如果有恶意智能体）或原始环境
    """
    if not malicious_agents_str or not malicious_agents_str.strip():
        return env
        
    try:
        # 解析恶意智能体ID
        malicious_agents = [int(x.strip()) for x in malicious_agents_str.split(',') if x.strip()]
        
        if malicious_agents:
            wrapped_env = MaliciousAgentWrapper(
                env, 
                malicious_agents=malicious_agents, 
                malicious_type=malicious_type,
                seed=seed,
                log_interval=log_interval
            )
            logging.info(f'Applied malicious wrapper: agents {malicious_agents}, type {malicious_type}')
            return wrapped_env
        else:
            logging.warning('No valid malicious agents specified')
            return env
            
    except ValueError as e:
        logging.error(f'Error parsing malicious agents "{malicious_agents_str}": {e}')
        return env


def create_malicious_config(malicious_agents: Union[List[int], str], 
                          malicious_type: str = 'random') -> dict:
    """
    创建恶意智能体配置字典
    
    Args:
        malicious_agents: 恶意智能体列表或逗号分隔的字符串
        malicious_type: 恶意行为类型
        
    Returns:
        配置字典
    """
    if isinstance(malicious_agents, str):
        agents_list = [int(x.strip()) for x in malicious_agents.split(',') if x.strip()]
    else:
        agents_list = malicious_agents
        
    return {
        'agents': agents_list,
        'type': malicious_type,
        'enabled': bool(agents_list)
    }


# 示例使用函数
def example_usage():
    """使用示例"""
    # 假设有一个环境实例
    # env = SomeEnvironment()
    
    # 方法1: 直接创建包装器
    # malicious_env = MaliciousAgentWrapper(
    #     env, 
    #     malicious_agents=[0, 2], 
    #     malicious_type='random'
    # )
    
    # 方法2: 使用便捷函数
    # malicious_env = apply_malicious_wrapper(env, "0,2", "random")
    
    # 方法3: 创建配置
    # config = create_malicious_config([0, 2], 'random')
    # if config['enabled']:
    #     malicious_env = MaliciousAgentWrapper(env, **config)
    
    pass
