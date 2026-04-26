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
    states = []  # 使用列表存储，因为每个智能体的状态维度可能不同
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
        
        # 初始化状态存储（如果是第一次）
        if t == 0:
            # 获取最大状态维度
            max_state_dim = max(env.n_s_ls)
            for ep_idx in range(n_ep_fixed):
                episode_states = []
                for step_idx in range(max_ep_len + 1):
                    step_states = []
                    for agent_idx in range(n_agents):
                        step_states.append(torch.zeros(max_state_dim).to(device))
                        # step_states.append(torch.zeros(env.n_s_ls[agent_idx]).to(device))
                    episode_states.append(step_states)
                states.append(episode_states)
        
        # 存储初始状态
        for agent_idx in range(n_agents):
            states[i][j][agent_idx] = torch.tensor(state_list[agent_idx], dtype=torch.float32).to(device)

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
            state_tensor = states[i][j][node].clone().detach().unsqueeze(0)
            # 创建联合动作（初始为0）
            joint_action = torch.zeros(n_agents).to(device)
            # 拼接全局状态和联合动作
            global_state = torch.cat([s for s in [states[i][j][k] for k in range(n_agents)]])
            global_state = global_state.unsqueeze(0)
            sa_tensor = torch.cat((global_state, joint_action.unsqueeze(0)), dim=1)
            
            mu, _ = agents[node].critic(sa_tensor)
            est_returns += mu[0].item()

        mean_est_returns = est_returns / n_coop

        # 模拟episode
        done = False
        while j < max_ep_len and not done:
            action_list = []
            
            # 每个智能体选择动作
            for node in range(n_agents):
                action = agents[node].get_action(
                    states[i][j][node].detach(), 
                    from_policy=True, 
                    mu=eps
                )
                actions[i, j, node] = torch.as_tensor(action, dtype=torch.int64, device=device)
                action_list.append(action)
            
            # 执行环境步进
            next_state_list, reward_list, done, global_reward = env.step(action_list)
            
            # 存储下一状态和奖励
            for agent_idx in range(n_agents):
                if j + 1 < max_ep_len:
                    states[i][j + 1][agent_idx] = torch.tensor(
                        next_state_list[agent_idx], dtype=torch.float32
                    ).to(device)
                
                # 处理奖励（CACC返回的可能是全局奖励或局部奖励）
                if isinstance(reward_list, (int, float)):
                    # 如果是全局奖励，分配给所有智能体
                    rewards[i, j, agent_idx] = reward_list / n_agents
                else:
                    # 如果是局部奖励列表
                    rewards[i, j, agent_idx] = reward_list[agent_idx] if len(reward_list) > agent_idx else reward_list
                
                ep_rewards[agent_idx] += rewards[i, j, agent_idx]
                ep_returns[agent_idx] += rewards[i, j, agent_idx] * (gamma ** j)
            
            j += 1
            
            # 训练更新（每个episode结束后）
            if (i == n_ep_fixed - 1 and j == max_ep_len) or done:
                # 准备训练数据
                batch_size = n_ep_fixed * j
                
                # 全局状态拼接
                s_list = []
                ns_list = []
                local_r_list = []
                local_a_list = []
                joint_actions_list = []
                
                for ep_idx in range(n_ep_fixed):
                    for step_idx in range(min(j, max_ep_len)):
                        # 拼接所有智能体的状态作为全局状态
                        global_s = torch.cat([states[ep_idx][step_idx][k] for k in range(n_agents)])
                        # global_ns = torch.cat([states[ep_idx][step_idx + 1][k] for k in range(n_agents)]) if step_idx + 1 <= j else global_s
                        if step_idx + 1 < len(states[ep_idx]) and step_idx + 1 < max_ep_len:
                            global_ns = torch.cat([states[ep_idx][step_idx + 1][k] for k in range(n_agents)])
                        else:
                            global_ns = torch.zeros_like(global_s)
                        s_list.append(global_s)
                        ns_list.append(global_ns)
                        local_r_list.append(rewards[ep_idx, step_idx])
                        local_a_list.append(actions[ep_idx, step_idx])
                        joint_actions_list.append(actions[ep_idx, step_idx])

                s = torch.stack(s_list)
                ns = torch.stack(ns_list)
                local_r = torch.stack(local_r_list)
                local_a = torch.stack(local_a_list)
                joint_actions = torch.stack(joint_actions_list)
                
                # 生成下一时刻的联合动作
                next_actions = torch.zeros((1, n_agents), dtype=torch.int64).to(device)
                # for node in range(n_agents):
                #     if len(s_list) > 0:
                #         next_action = agents[node].get_action(ns[-1].unsqueeze(0), from_policy=True, mu=0.1)
                #         next_actions[0, node] = next_action
                
                for node in range(n_agents):
                    if len(s_list) > 0:
                        # 从全局状态中提取该智能体的局部状态
                        # 假设每个智能体状态维度都是15
                        local_state_dim = 15
                        start_idx = node * local_state_dim
                        end_idx = start_idx + local_state_dim       
                        # 提取局部状态
                        agent_local_state = ns[-1, start_idx:end_idx]  # 取最后一个时间步的局部状态        
                        next_action = agents[node].get_action(
                            agent_local_state.unsqueeze(0), 
                            from_policy=True, 
                            mu=0.1
                        )
                        next_actions[0, node] = next_action

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