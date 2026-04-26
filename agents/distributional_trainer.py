import numpy as np
import gym
from gym import spaces
import pandas as pd
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from agents.distributional_models import distributional_CAC_agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_CAC(env,agents,config,writer):
    paths = []

    # 初始化超参数
    n_agents = config.getint('ENV_CONFIG','n_agent')
    n_states = config.getint('ENV_CONFIG','n_states')
    n_actions = config.getint('ENV_CONFIG','n_actions')
    gamma = config.getfloat('MODEL_CONFIG','gamma')
    eps = config.getfloat('MODEL_CONFIG','eps')
    in_nodes = eval(config.get('ENV_CONFIG','in_nodes'))
    max_ep_len = config.getint('MODEL_CONFIG','max_ep_len')
    n_episodes = config.getint('MODEL_CONFIG','n_episodes')
    n_ep_fixed = config.getint('MODEL_CONFIG','n_ep_fixed')
    n_epochs = config.getint('MODEL_CONFIG','n_epochs')
    use_distributional = config.getint('ENV_CONFIG','use_distributional')

    # 初始化 Tensor
    states = torch.zeros((n_ep_fixed, max_ep_len + 1, n_agents, n_states), dtype=torch.float32).to(device)
    actions = torch.zeros((n_ep_fixed, max_ep_len, n_agents), dtype=torch.int64).to(device)
    rewards = torch.zeros((n_ep_fixed, max_ep_len, n_agents), dtype=torch.float32).to(device)
    # TR_errors_team = torch.zeros((n_ep_fixed, max_ep_len, n_agents), dtype=torch.float32).to(device)
    critic_errors_team = torch.zeros((n_ep_fixed, max_ep_len, n_agents), dtype=torch.float32).to(device)

    for t in range(n_episodes):
        # 初始化变量
        j, ep_rewards, ep_returns = 0, 0, 0
        est_returns, n_coop, mean_true_returns, mean_true_returns_adv = 0, 0, 0, 0
        actor_loss, critic_loss = torch.zeros(n_agents), torch.zeros(n_agents)
        # 确定当前回合
        i = t % n_ep_fixed
        # 重置环境并获取状态
        env.reset()
        states_np, rewards_np,  done, _ = env.get_data()
        # 将 NumPy 数组转换为 PyTorch FloatTensor
        states[i, j] = torch.tensor(states_np, dtype=torch.float32).to(device)
        rewards[i, j] = torch.tensor(rewards_np, dtype=torch.float32).to(device)

        #调整学习率
        # if t >= 5000:
        #     for node in range(n_agents):
        #         agents[node].critic_scheduler.step()
        #         agents[node].actor_scheduler.step()

        # 修改学习率调度
        # if t >= 8000:  # 延迟开始
        #    if t % 1000 == 0:  # 降低调整频率
        #        for node in range(n_agents):
        #            agents[node].critic_scheduler.step()
        #    if t % 2000 == 0:
        #        for node in range(n_agents):
        #            agents[node].actor_scheduler.step()
        # 2. 恢复学习率调度
        if t >= 5000 and t % 1000 == 0:
            for node in range(n_agents):
                agents[node].critic_scheduler.step()
        if t >= 8000 and t % 2000 == 0:
            for node in range(n_agents):
                agents[node].actor_scheduler.step()



        # 添加周期性探索增强
        # if t > 6000 and t % 1000 == 0:
        #     eps = min(0.15, eps * 1.5)  # 周期性增加探索
        # else:
        #     eps = max(0.01, eps * 0.999)  # 缓慢衰减
        # 1. 改进探索率衰减策略
        if t < 5000:
            eps = max(0.05, 0.3 * (1 - t / 5000))  # 前5000轮线性衰减
        elif t < 10000:
            eps = max(0.02, 0.05 * (1 - (t - 5000) / 5000))  # 5000-10000轮继续衰减
        else:
            # 10000轮后保持小的探索率，偶尔增加
            if t % 2000 == 0:
                eps = min(0.1, eps * 2)  # 每2000轮增加探索
            else:
                eps = max(0.01, eps * 0.9995)  # 更慢的衰减

        # 每2000轮进行一次目标网络硬更新
        # if t % 2000 == 0 and t > 0:
        #     for node in range(n_agents):
        #         agents[node].distributional_CAC_agent.hard_update_target_critic()
        #     print(f"Hard updated target networks at episode {t}")

        'Evaluate expected retuns at the beginning of an episode 在episode开始前评估预期奖励'
        for node in range(n_agents):
            state_tensor = states[i, j].clone().detach().unsqueeze(0).reshape(1, -1)
            action_tensor = actions[i, j].clone().detach().unsqueeze(0).reshape(1, -1).float()
            sa_tensor = torch.cat((state_tensor, action_tensor), dim=1).unsqueeze(0)
            if use_distributional == 1:
                est_returns += agents[node].critic(sa_tensor)[0][0].item()
                    # est_returns += agents[node].target_critic(sa_tensor)[0][0].item()
            else:
                est_returns += agents[node].critic(sa_tensor)[0, 0].item()  # 使用 .item() 获取标量值
                    # est_returns += agents[node].target_critic(sa_tensor)[0, 0].item()
            n_coop += 1
        # 计算均值
        if n_coop > 0:
            # 所有合作智能体的平均期望回报
            mean_est_returns = est_returns / n_coop
        else:
            mean_est_returns = 0  # 如果没有 Cooperative agent，则设为 0 或其他值

        'Simulate episode'
        while j < max_ep_len:
            for node in range(n_agents):
                # 使用 get_action 获得动作，在当前状态（states[i,j]）下根据自己的策略（Policy）选择一个动作
                action = agents[node].get_action(states[i, j].detach().unsqueeze(0).reshape(1,-1), from_policy=True, mu=eps)
                actions[i, j, node] = torch.as_tensor(action, dtype=actions.dtype, device=actions.device)
                # actions[i, j, node] = agents[node].get_action(states[i, j].detach().unsqueeze(0).reshape(1,-1), from_policy=True, mu=eps)
            # 执行环境的步进，假设 env.step 这部分不变
            env.step(actions[i, j])
            # states[i, j + 1], rewards[i, j], done, _ = env.get_data()
            states_np, rewards_np,  done, _ = env.get_data()
            # 将 NumPy 数组转换为 PyTorch FloatTensor
            states[i, j+1] = torch.tensor(states_np, dtype=torch.float32).to(device)
            rewards[i, j] = torch.tensor(rewards_np, dtype=torch.float32).to(device)
            # 累积奖励
            ep_rewards += rewards[i, j]
            # 折扣回报
            ep_returns += rewards[i, j] * (gamma ** j)
            #------------------
            'END OF SIMULATION'
            #------------------
            # 回合结束
            if i == n_ep_fixed-1 and j == max_ep_len-1:
                s = states[:, :-1].reshape(n_ep_fixed*max_ep_len, -1) #每个时间步的当前状态
                ns = states[:, 1:].reshape(n_ep_fixed*max_ep_len, -1) #每个时间步的下一状态
                local_r = rewards.reshape(n_ep_fixed*max_ep_len, n_agents, 1)
                local_a = actions.reshape(n_ep_fixed*max_ep_len, n_agents, 1)
                joint_actions = actions.reshape(n_ep_fixed*max_ep_len, n_agents)
                # 所有 agent 的 target actor 中获取next_joint_actions
                next_actions = torch.zeros((1, n_agents), dtype=torch.int64).to(device)
                for node in range(n_agents):
                    a = agents[node].get_action(ns)
                    # a = agents[node].get_action(ns[i,j].detach().unsqueeze(0).reshape(1,-1), from_policy=True, mu=eps)
                    # a_tensor = torch.tensor(a, dtype=torch.long, device=device)  # 将动作转换为张量
                    # next_joint_actions.append(a_tensor.unsqueeze(1))
                    next_actions[0, node] = a
                next_joint_actions = torch.cat((joint_actions[1:],next_actions), dim=0)

                critic_err = critic_errors_team.reshape(n_ep_fixed*max_ep_len, n_agents, 1)
                # TR_err = TR_errors_team.reshape(n_ep_fixed*max_ep_len, n_agents, 1)
                # 计算合作智能体的合作奖励（r_coop）,形状为 (n_ep_fixed * max_ep_len, 1)1
                r_coop = torch.zeros(local_r.shape[0], local_r.shape[2]) 
                for node in (x for x in range(n_agents)):
                    r_coop = r_coop.to(device)
                    r_coop += local_r[:, node] / n_coop #对每个时间步对合作型智能体的奖励取平均
                n = 0
                while n < n_epochs:
                    critic_weights = []
                    #---------------------------------------------------
                    'BATCH LOCAL CRITIC AND TEAM-AVERAGE REWARD UPDATES'
                    #---------------------------------------------------
                    for node in range(n_agents):
                        # 对于每个智能体，更新其 Critic 和 Team Reward（TR）网络
                        common_reward = False
                        r_applied = r_coop if common_reward else local_r[:, node] #common_reward默认是false，得到每个智能体局部奖励
                        # if args.agent_label[node] == 'Cooperative':
                            # Update TR and Critic using PyTorch models。如果智能体是合作型（Cooperative），则使用合作动作；否则使用本地奖励。
                            # x, TR_loss[node] = agents[node].TR_update_local(s, team_a, r_applied)
                        y, critic_loss[node] = agents[node].critic_update(s, ns,joint_actions,next_joint_actions,r_applied)
                        # TR_weights.append(x)
                        critic_weights.append(y)
                    #--------------------------------------------------------------------------------------------
                    'RESILIENT PROJECTION-BASED CONSENSUS UPDATES OF THE CRITIC AND TEAM-AVERAGE REWARD NETWORKS'
                    '暂时去掉弹性'
                    #--------------------------------------------------------------------------------------------
                    for node in (x for x in range(n_agents)):
                        # Aggregate parameters received from neighbors从邻居收到的聚合参数
                        critic_weights_innodes = [critic_weights[i] for i in in_nodes[node]]
                        agents[node].consensus_critic(critic_weights_innodes)
                    
                    n += 1

                #----------------------------------------------
                'BATCH STOCHASTIC UPDATE OF THE ACTOR NETWORKS'
                #----------------------------------------------
                # 对于每个智能体，进行 Actor 网络的更新。如果智能体是合作型，则使用团队奖励（joint_actions）；否则使用本地奖励（local_r）
                actor_weights = []
                for node in range(n_agents):
                    # if args.agent_label[node] == 'Cooperative':
                    x, actor_loss[node] = agents[node].actor_update(s, joint_actions, local_a[:, node])
                    actor_weights.append(x)
                    # else:
                    #     actor_loss[node] = agents[node].actor_update(s, ns, local_r[:, node], local_a[:, node])
                # for node in (x for x in range(n_agents) if args.agent_label[x] == 'Cooperative'):
                #         # Aggregate parameters received from neighbors从邻居收到的聚合参数
                #         actor_weights_innodes = [actor_weights[i] for i in in_nodes[node]]
                #         agents[node].consensus_actor(actor_weights_innodes)
            j += 1

        #------------------------------------
        '''SUMMARY OF THE TRAINING EPISODE'''
        #------------------------------------
        critic_mean_loss = torch.mean(critic_loss).item()  
        # TR_mean_loss = torch.mean(TR_loss).item()  
        actor_mean_loss = torch.mean(actor_loss).item()     

        for node in range(n_agents):
            mean_true_returns += ep_returns[node] / n_coop
            # else:
            #  mean_true_returns_adv += ep_returns[node] / (n_agents - n_coop)    

        # 将标量添加到 TensorBoard
        # writer.add_scalar("estimated_episode_team_average_returns", mean_est_returns, t)
        # writer.add_scalar("true_episode_team_average_returns", mean_true_returns, t)
        # writer.add_scalar("true_episode_team_average_rewards", torch.mean(ep_rewards).item(), t)
        
        writer.add_scalars(
            "Episode_Team_Average_Returns", 
            {
                # "Estimated": mean_est_returns,
                "True": mean_true_returns
            },t)

        # 输出信息
        # print('| Episode: {} | Est. returns: {} | Returns: {} | Average critic loss: {} | Average TR loss: {} | Average actor loss: {} | Target reached: {} '.format(
        #         t, mean_est_returns, ep_returns, critic_mean_loss, TR_mean_loss, actor_mean_loss, done))
        print('| Episode: {} | Est. returns: {} | Returns: {} | Target reached: {} '.format(
            t,mean_est_returns,ep_returns,done))
        # 保存路径信息
        path = {
            "True_team_returns": mean_true_returns,
            "True_adv_returns": mean_true_returns_adv,
            "Estimated_team_returns": mean_est_returns
        }
        paths.append(path)
    
    sim_data = pd.DataFrame.from_dict(paths)
    return agents,sim_data
