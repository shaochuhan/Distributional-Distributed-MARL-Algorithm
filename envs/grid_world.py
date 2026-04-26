import numpy as np
import logging
import gymnasium as gym
# import gym
# from gym import spaces

class Grid_World(gym.Env):
    """
    Multi-agent grid-world: cooperative navigation
    This is a grid-world environment designed for the cooperative navigation problem.
    Each agent seeks to navigate to the desired position without colliding with other agents.
    The rewards are individually awarded for reaching the target and collisions.
    1) The reward for approaching the target is given as the negative Manhattan distance
       between the target and the agent at the new state.
    2) The penalty for collision is 0.5 times the reward for approaching the target.
    ARGUMENTS:  nrow, ncol: grid world dimensions
                n_agent: number of agents
                desired_state: desired position of each agent
                initial_state: initial position of each agent
                randomize_state: True if the agents' initial position is randomized at the beginning of each episode
                scaling: determines if the states are scaled
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, nrow = 5, ncol=5, n_agent = 5,desired_state = None,initial_state = None,randomize_state = False,agent='ma2c_cu',coop_gamma=-1):
        self.nrow = nrow
        self.ncol = ncol
        self.n_agent = n_agent
        self.agent = agent
        self.coop_gamma = coop_gamma
        self.initial_state = initial_state
        self.desired_state = desired_state
        self.randomize_state = randomize_state
        self.total_states = self.nrow * self.ncol
        self.name = 'grid_world'
        self.n_states = 2
        self.n_actions = n_agent 
        self.actions = {0:'LEFT', 1:'DOWN', 2:'RIGHT', 3:'UP', 4:'STAY'}
        self.reward=np.zeros(self.n_agent)
        self.observation_space = gym.spaces.MultiDiscrete([self.total_states for _ in range(self.n_agent)])
        self.action_space = gym.spaces.MultiDiscrete([self.n_actions for _ in range(self.n_agent)])
        self.reset()
        self._init_space()
        self.reward, self.done = np.full_like(self.n_agent, 0.0), False
        self.T = 60  #episode_length_sec
        self.train_mode = True

        # if scaling:
        #     x,y=np.arange(nrow),np.arange(ncol)
        #     self.mean_state=np.array([np.mean(x),np.mean(y)])
        #     self.std_state=np.array([np.std(x),np.std(y)])
        # else:
        #     self.mean_state,self.std_state=0,1

    def _state_transition(self, local_state, local_action):
        '''
        Computes a new local state wrt to the current state and action
        Arguments: local state and local action
        Returns: new local state
        local action:  0 - LEFT
                       1 - DOWN
                       2 - RIGHT
                       3 - UP
                       4 - STAY
        '''
        row=local_state[0]
        col=local_state[1]
        if local_action == 0:
            col = max(col - 1, 0)
        elif local_action == 1:
            row = max(row - 1, 0)
        elif local_action == 2:
            col = min(col + 1, self.ncol - 1)
        elif local_action == 3:
            row = min(row + 1, self.nrow - 1)
        return np.array([row,col])

    def reset(self, gui=False, test_ind=-1):
        '''Resets the environment'''
        self.fp = np.ones((self.n_agent, self.n_actions)) / self.n_actions
        if self.randomize_state:
            self.state = np.random.randint([0,0],[self.nrow,self.ncol],size=self.initial_state.shape)
        else:
            self.state = np.array(self.initial_state)
        self.reward, self.done = np.zeros(self.n_agent), False

        # if (self.train_mode):
        #     seed = self.seed
        # elif (test_ind < 0):
        #     seed = self.seed-1
        # else:
        #     seed = self.test_seeds[test_ind]
        # np.random.seed(seed)
        # self.seed += 1

        ##grid
        # return self.state, {}  # 加一个空的 info dict 以适配新的API
        self.step_count = 0
        return self.state
    
    ##grid
    def _init_space(self):
        # 初始化邻居掩码和距离掩码
        self.neighbor_mask = np.zeros((self.n_agent, self.n_agent), dtype=int)
        self.distance_mask = np.zeros((self.n_agent, self.n_agent), dtype=int)

        # 初始化距离掩码，表示每个 agent 的排列顺序（循环右移）
        cur_distance = list(range(self.n_agent))
        for i in range(self.n_agent):
            self.distance_mask[i] = cur_distance
            cur_distance = [cur_distance[-1]] + cur_distance[:-1]  # 循环右移

            # 设置邻居掩码，只考虑相邻 agent（i-1 和 i+1）
            if i >= 1:
                self.neighbor_mask[i, i-1] = 1
            if i <= self.n_agent - 2:
                self.neighbor_mask[i, i+1] = 1

        # 设置动作映射（可用于高层控制或离散策略）
        # self.n_a_ls = [4] * self.n_agent  # 每个 agent 有 4 个高层动作
        # self.n_a = 4
        # self.a_map = [(0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5)]  # 用于参数化控制策略
        # logging.info('action to high-level map: %r' % self.a_map)
        self.n_a_ls = [5] * self.n_agent  # 每个 agent 有 4 个高层动作
        self.n_a = 5

        # 状态维度（只包含自身状态，因为环境返回的观测不包含邻居信息）
        # 注意：环境的reset()和step()返回的是每个agent自己的状态（2维）
        self.n_s_ls = [self.n_states] * self.n_agent  # 每个agent的观测维度都是n_states
        # 旧代码（错误）：
        # self.n_s_ls = []
        # for i in range(self.n_agent):
        #     num_n = np.sum(self.neighbor_mask[i])  
        #     self.n_s_ls.append(num_n * 2)  # 这假设观测包含邻居状态，但实际不包含 

    def step(self, global_action):
        '''
        Makes a transition to a new state and evaluates all rewards
        Arguments: global action
        '''
        new_s=np.zeros((self.n_agent,self.n_states))
        # 更新每个智能体的状态
        for node,s,a in zip(range(self.n_agent),self.state, global_action):
            new_s[node]=self._state_transition(s,a)                                                     #State transition
        
        # # 计算奖励
        # for node in range(self.n_agent):
        #     sub_s = np.delete(new_s,node,axis=0)  # 获取除当前智能体外的其他智能体状态
        #     dist_agents = np.sum(abs(sub_s-new_s[node]),axis=1)  # 计算与其他智能体的曼哈顿距离                                        #Compute Manhattan distance between agents
        #     collision = True if np.any(dist_agents==0) else False  # 判断是否发生碰撞
        #     #dist = np.sum(abs(self.state[node]-self.desired_state[node]))
        #     dist_next = np.sum(abs(new_s[node]-self.desired_state[node]))    # 计算与目标的曼哈顿距离                           #Compute Manhattan distance to the target at future state
        #     self.reward[node] = (- dist_next                                                         #Reward for reaching the target
        #                          - int(collision)                                                  #Penalty for collision
        #                         )
        
        ##v2:添加进步奖励
        for node in range(self.n_agent):
            sub_s = np.delete(new_s, node, axis=0)
            dist_agents = np.sum(abs(sub_s - new_s[node]), axis=1)
            collision = True if np.any(dist_agents == 0) else False   
            # 计算当前和下一步到目标的距离
            dist_current = np.sum(abs(self.state[node] - self.desired_state[node]))
            dist_next = np.sum(abs(new_s[node] - self.desired_state[node]))  
            # 进步奖励：朝目标移动获得正奖励
            progress_reward = dist_current - dist_next 
            # 到达目标的大奖励
            reach_goal_reward = 10.0 if dist_next < 0.5 else 0.0    
            self.reward[node] = (
                -0.05 * dist_next          # 轻微的距离惩罚
                + 2*progress_reward         # 进步奖励（可正可负）
                + reach_goal_reward       # 到达目标奖励
                - 3.0 * int(collision)    # 碰撞重惩罚
                - 0.01                    # 轻微的时间惩罚，鼓励快速完成
            )


        self.state = new_s
        
        # self.done = np.array_equal(self.state, self.desired_state)
        ##放宽done的限制
        # 添加步数计数
        if not hasattr(self, 'step_count'):     
             self.step_count = 0
        self.step_count += 1
        # 使用容差检查
        tolerance = 0.1  # 根据环境调整
        # distances = np.array([np.sum(abs(self.state[i] - self.desired_state[i])) 
        #                      for i in range(self.n_agent)])
        distances = np.sum(np.abs(self.state - self.desired_state), axis=1)  # 向量化计算
        goal_reached = np.all(distances < tolerance)
        # 防止无限循环
        max_steps = 1000
        timeout = self.step_count >= max_steps
        self.done = goal_reached or timeout

        # 可选：添加调试信息
        # if goal_reached:
        #     print(f"目标达成！用时 {self.step_count} 步")
        # elif timeout:
        #     print(f"超时结束，距离目标最近距离: {np.min(distances):.3f}")

        self.global_reward = np.sum(self.reward)
        ##grid
        return self.state, self.reward, self.done,self.global_reward
    
    def terminate(self):
        return

    def get_data(self):
        '''
        Returns scaled reward and state, and flags if the agents has reached the target
        '''
        # state_scaled = (self.state-self.mean_state)/self.std_state
        state_scaled = self.state
        reward_scaled = self.reward/10
        return state_scaled,reward_scaled, self.done, {}
    

    def get_fingerprint(self):
        return self.fp

    def update_fingerprint(self, fp):
        self.fp = fp

    def close(self):
        pass
