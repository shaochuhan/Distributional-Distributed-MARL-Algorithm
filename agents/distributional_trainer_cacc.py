import numpy as np
import torch
import pandas as pd
from torch.utils.tensorboard.writer import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_CACC(env, agents, config, writer):
    """CACC环境的distributional训练函数"""
    paths = []

    # 初始化超参数
    n_agents = config.getint('ENV_CONFIG', 'n_vehicle')  # 使用n_vehicle
    gamma = config.getfloat('MODEL_CONFIG', 'gamma')
    eps = config.getfloat('MODEL_CONFIG', 'eps')
    in_nodes = eval(config.get('ENV_CONFIG', 'in_nodes'))
    max_ep_len = config.getint('MODEL_CONFIG', 'max_ep_len')
    n_episodes = config.getint('MODEL_CONFIG', 'n_episodes')
    n_ep_fixed = config.getint('MODEL_CONFIG', 'n_ep_fixed')
    n_epochs = config.getint('MODEL_CONFIG', 'n_epochs')
    # 动态获取状态和动作维度
    n_actions = env.n_a
    max_state_dim = max(env.n_s_ls)  # 最大状态维度
    
    # 初始化存储Tensors
    # states = []  # 使用列表存储，因为每个智能体的状态维度可能不同
    states = torch.zeros((n_ep_fixed, max_ep_len + 1, n_agents, max_state_dim), dtype=torch.float32).to(device)
    actions = torch.zeros((n_ep_fixed, max_ep_len, n_agents), dtype=torch.int64).to(device)
    rewards = torch.zeros((n_ep_fixed, max_ep_len, n_agents), dtype=torch.float32).to(device)

    for t in range(n_episodes):
        # 初始化变量
        j = 0
        ep_rewards = torch.zeros(n_agents).to(device)
        ep_returns = torch.zeros(n_agents).to(device)
        est_returns = 0
        n_coop = n_agents  # 在CACC中所有智能体都是合作的
        mean_true_returns = 0
        actor_loss = torch.zeros(n_agents)
        critic_loss = torch.zeros(n_agents)
        
        # 确定当前回合
        i = t % n_ep_fixed
        
        # 重置环境并获取状态
        state_list = env.reset()
        state_tensor = torch.tensor(state_list)
        states[i][j] = torch.tensor(state_tensor, dtype=torch.float32).to(device)

        # 学习率调度
        if t >= 5000 and t % 1000 == 0:
            for node in range(n_agents):
                agents[node].critic_scheduler.step()
        if t >= 8000 and t % 2000 == 0:
            for node in range(n_agents):
                agents[node].actor_scheduler.step()

        # 探索率衰减
        if t < 5000:
            eps = max(0.05, 0.3 * (1 - t / 5000))
        elif t < 10000:
            eps = max(0.02, 0.05 * (1 - (t - 5000) / 5000))
        else:
            if t % 2000 == 0:
                eps = min(0.1, eps * 2)
            else:
                eps = max(0.01, eps * 0.9995)

        # 评估期望回报
        for node in range(n_agents):
            state_tensor = states[i, j].clone().detach().unsqueeze(0).reshape(1, -1)
            action_tensor = actions[i, j].clone().detach().unsqueeze(0).reshape(1, -1).float()
            sa_tensor = torch.cat((state_tensor, action_tensor), dim=1).unsqueeze(0)            
            mu, _ = agents[node].critic(sa_tensor)

            est_returns += mu[0].item()

        mean_est_returns = est_returns / n_coop

        # 模拟episode
        done = False
        while j < max_ep_len and not done:
            action_list = []            
            # 每个智能体选择动作
            for node in range(n_agents):
                action = agents[node].get_action(states[i, j].detach().unsqueeze(0).reshape(1,-1), from_policy=True, mu=eps)
                actions[i, j, node] = torch.as_tensor(action, dtype=actions.dtype, device=actions.device)
            
            # 执行环境步进
            # next_state_list, reward_list, done, global_reward = env.step(action_list)
            next_state_list, reward_list, done, global_reward = env.step(actions[i, j])
            
            states[i, j+1] = torch.tensor(next_state_list, dtype=torch.float32).to(device)
            rewards[i, j] = torch.tensor(reward_list, dtype=torch.float32).to(device)
            ep_rewards += rewards[i, j]
            ep_returns += rewards[i, j] * (gamma ** j)
                     
            # 训练更新（每个episode结束后）
            if (i == n_ep_fixed - 1 and j == max_ep_len - 1):
                # s = torch.stack([step for episode in states[:n_ep_fixed] for step in episode[:-1]])
                s = states[:, :-1].reshape(n_ep_fixed*max_ep_len, -1) #每个时间步的当前状态
                ns = states[:, 1:].reshape(n_ep_fixed*max_ep_len, -1) #每个时间步的下一状态
                local_r = rewards.reshape(n_ep_fixed*max_ep_len, n_agents, 1)
                local_a = actions.reshape(n_ep_fixed*max_ep_len, n_agents, 1)
                joint_actions = actions.reshape(n_ep_fixed*max_ep_len, n_agents)
                
                # 生成下一时刻的联合动作
                next_actions = torch.zeros((1, n_agents), dtype=torch.int64).to(device)
                
                for node in range(n_agents):
                    a = agents[node].get_action(ns)
                    next_actions[0, node] = a

                next_joint_actions = torch.cat((joint_actions[1:], next_actions), dim=0)
                
                # 训练循环
                for n in range(n_epochs):
                    critic_weights = []                    
                    # 更新Critic
                    for node in range(n_agents):
                        y, critic_loss[node] = agents[node].critic_update(
                            s, ns, joint_actions, next_joint_actions, local_r[:, node:node+1]
                        )
                        critic_weights.append(y)                   
                    # 共识更新
                    for node in range(n_agents):
                        critic_weights_innodes = [critic_weights[i] for i in in_nodes[node]]
                        agents[node].consensus_critic(critic_weights_innodes)
                
                # 更新Actor
                for node in range(n_agents):
                    _, actor_loss[node] = agents[node].actor_update(
                        s, joint_actions, local_a[:, node:node+1]
                    )
            j += 1

        # 记录训练信息
        critic_mean_loss = torch.mean(critic_loss).item()
        actor_mean_loss = torch.mean(actor_loss).item()
        
        mean_true_returns = torch.mean(ep_returns).item()
        
        # TensorBoard记录
        writer.add_scalars(
            "Episode_Team_Average_Returns",
            {
                "Estimated": mean_est_returns,
                "True": mean_true_returns
            }, t
        )
        
        writer.add_scalars(
            "Losses",
            {
                "Critic": critic_mean_loss,
                "Actor": actor_mean_loss
            }, t
        )
        
        writer.add_scalar("Exploration_Rate", eps, t)
        writer.add_scalar("Episode_Reward", torch.mean(ep_rewards).item(), t)
        
        # 打印训练信息
        if t % 100 == 0:
            print(f'Episode: {t} | Est. returns: {mean_est_returns:.3f} | '
                  f'True returns: {mean_true_returns:.3f} | '
                  f'Critic loss: {critic_mean_loss:.3f} | '
                  f'Actor loss: {actor_mean_loss:.3f} | '
                  f'Collision: {env.collision} | Eps: {eps:.3f}')
        
        # 保存路径信息
        path = {
            "True_team_returns": mean_true_returns,
            "True_adv_returns": 0,  # CACC中没有对抗智能体
            "Estimated_team_returns": mean_est_returns,
            "Episode": t,
            "Collision": env.collision,
            "Exploration_Rate": eps
        }
        paths.append(path)
    
    sim_data = pd.DataFrame.from_dict(paths)
    return agents, sim_data