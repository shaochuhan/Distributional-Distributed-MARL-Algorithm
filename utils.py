import itertools
import logging
import numpy as np
import time
import os
import pandas as pd
import subprocess
import torch
import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def check_dir(cur_dir):
    if not os.path.exists(cur_dir):
        return False
    return True


def copy_file(src_dir, tar_dir):
    cmd = 'cp %s %s' % (src_dir, tar_dir)
    subprocess.check_call(cmd, shell=True)


def find_file(cur_dir, suffix='.ini'):
    for file in os.listdir(cur_dir):
        if file.endswith(suffix):
            return cur_dir + '/' + file
    logging.error('Cannot find %s file' % suffix)
    return None


# def init_dir(base_dir, pathes=['log', 'data', 'model']):
#     if not os.path.exists(base_dir):
#         os.mkdir(base_dir)
#     dirs = {}
#     for path in pathes:
#         cur_dir = base_dir + '/%s/' % path
#         if not os.path.exists(cur_dir):
#             os.mkdir(cur_dir)
#         dirs[path] = cur_dir
#     return dirs

def init_dir(base_dir, pathes=['log', 'data', 'model']):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)  # 递归创建目录
    dirs = {}
    for path in pathes:
        cur_dir = os.path.join(base_dir, path)
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir, exist_ok=True)  # 递归创建子目录
        
        # 如果是 log 目录，添加时间戳子目录
        if path == 'log':
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            log_subdir = os.path.join(cur_dir, f'run_{timestamp}')
            os.makedirs(log_subdir, exist_ok=True)
            dirs[path] = log_subdir  # 用子目录作为日志路径
        else:
            dirs[path] = cur_dir
    return dirs

def init_log(log_dir):
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler('%s/%d.log' % (log_dir, time.time())),
                            logging.StreamHandler()
                        ])


def init_test_flag(test_mode):
    if test_mode == 'no_test':
        return False, False
    if test_mode == 'in_train_test':
        return True, False
    if test_mode == 'after_train_test':
        return False, True
    if test_mode == 'all_test':
        return True, True
    return False, False

#用于跟踪模型的训练步数和控制日志记录、测试和终止条件。
class Counter:
    def __init__(self, total_step, test_step, log_step):
        self.counter = itertools.count(1) # 计数器，从 1 开始逐步递增。
        self.cur_step = 0  # 当前的训练步数。
        self.cur_test_step = 0  #上一次测试时的步数
        self.total_step = total_step # 训练的总步数，即训练终止的目标步数。
        self.test_step = test_step  #执行测试的步数间隔。
        self.log_step = log_step
        self.stop = False  #标志是否应当停止训练

    def next(self):
        #获取下一个训练步数。使用 itertools.count 自动递增 cur_step。
        self.cur_step = next(self.counter)
        return self.cur_step

    def should_test(self):
        test = False
        # 如果距离上次测试的步数超过了 test_step，触发测试
        if (self.cur_step - self.cur_test_step) >= self.test_step:
            test = True
            self.cur_test_step = self.cur_step
        return test

    def should_log(self):
        return (self.cur_step % self.log_step == 0)

    def should_stop(self):
        # 达到或超过 total_step 步数，则终止训练
        if self.cur_step >= self.total_step:
            return True
        return self.stop

# 初始化训练环境和模型。执行多个训练步骤。评估模型在环境中的表现。记录和保存训练数据。
class Trainer():
    def __init__(self, env, model, global_counter, summary_writer, output_path=None, 
                 save_interval=None, model_dir=None):
        self.cur_step = 0
        self.global_counter = global_counter
        self.env = env
        self.agent = self.env.agent
        self.model = model.to(device)##+
        self.n_step = self.model.n_step
        self.summary_writer = summary_writer
        assert self.env.T % self.n_step == 0
        self.data = []
        self.output_path = output_path
        self.env.train_mode = True
        # 定期保存checkpoint
        self.save_interval = save_interval
        self.model_dir = model_dir
        self.last_saved_step = 0

    # 将训练奖励 reward 记录到 'train_reward' 或者 将测试奖励记录到 'test_reward'
    def _add_summary(self, reward, global_step, is_train=True):
        if is_train:
            self.summary_writer.add_scalar('train_reward', reward, global_step=global_step)
        else:
            self.summary_writer.add_scalar('test_reward', reward, global_step=global_step)

    # “策略”
    def _get_policy(self, ob, done, mode='train'):
        
        if self.agent.startswith('ma2c'):
            self.ps = self.env.get_fingerprint()
            self.ps = torch.as_tensor(self.ps, device=device)  ## 将 fingerprint 移动到 GPU
            # policy = self.model.forward(ob, done, self.ps, agent_id=None)
            policy = self.model.forward(ob, done, self.ps)
        else:
            # policy = self.model.forward(ob, done, agent_id=None)
            policy = self.model.forward(ob, done)
        
        action = []
        
        # 检查是否是WPO算法（返回高斯分布参数）
        if self.agent == 'wpo_cu' and isinstance(policy, tuple) and len(policy) == 2:
            # WPO返回 (means, stds)
            means, stds = policy
            for mean, std in zip(means, stds):
                # 从高斯分布采样动作
                if mode == 'train':
                    # 训练时从高斯分布采样
                    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
                    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)
                    dist = torch.distributions.Normal(mean_t, std_t)
                    sampled_action = dist.sample()
                    # 离散动作：映射到离散动作（取最大值的索引）
                    action_idx = torch.argmax(sampled_action).item()
                    action.append(action_idx)
                else:
                    # 测试时使用均值（确定性策略）
                    mean_np = mean if isinstance(mean, np.ndarray) else mean
                    action_idx = np.argmax(mean_np)
                    action.append(action_idx)
        else:
            # 传统算法（离散概率分布）
            for pi in policy:
                # 确保pi是numpy数组
                if isinstance(pi, torch.Tensor):
                    pi_np = pi.detach().cpu().numpy()
                else:
                    pi_np = np.array(pi) if not isinstance(pi, np.ndarray) else pi
                
                if mode == 'train':
                    # 训练时用随机选择
                    # 归一化概率
                    pi_np = pi_np / pi_np.sum()
                    action.append(np.random.choice(np.arange(len(pi_np)), p=pi_np))
                else: 
                    # 测试时选择最大概率动作
                    action.append(np.argmax(pi_np))
        
        # 返回action：离散动作索引
        return policy, torch.as_tensor(action, device=device)

    # 打分
    def _get_value(self, ob, done, action):
        # 确保输入在正确的设备上
        ob = [torch.as_tensor(o, device=device) for o in ob]
        done = torch.as_tensor(done, device=device)
        action = torch.as_tensor(action, device=device)
        
        if self.agent.startswith('ma2c'): 
            # MA2C系列算法使用fingerprint
            value = self.model.forward(ob, done, self.ps, action.detach().cpu().numpy(), 'v')
        elif self.agent == 'wpo_cu':
            # WPO算法：直接传入动作，不需要邻居动作
            action_np = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action
            value = self.model.forward(ob, done, action_np, 'v')
        else:
            # 其他算法：需要邻居动作
            action_np = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action
            # 检查环境是否有get_neighbor_action方法
            if hasattr(self.env, 'get_neighbor_action'):
                self.naction = self.env.get_neighbor_action(action_np)
            else:
                # Grid World等环境不需要邻居动作，直接使用当前动作
                self.naction = action_np
            
            if self.naction is None or (isinstance(self.naction, list) and not self.naction):
                self.naction = np.nan
            
            # 计算价值
            value = self.model.forward(ob, done, self.naction, 'v')
        return value

    # 把本轮的表现记录下来，算出奖励的平均值和波动情况（标准差）
    def _log_episode(self, global_step, mean_reward, std_reward):
        log = {'agent': self.agent,
               'step': global_step,
               'test_id': -1,
               'avg_reward': mean_reward,
               'std_reward': std_reward}
        self.data.append(log)
        self._add_summary(mean_reward, global_step)
        self.summary_writer.flush()

    # 关键步骤
    # 执行了一段连贯的训练过程。按照策略“迈出一步”，
    # 然后根据新的状态“记录得分”。同时，探索过程中还会记录“轨迹”
    def explore(self, ob, done):
        # ob = prev_ob
        # done = prev_done

        # 根据策略选择动作并估算价值，执行动作并得到奖励。
        for _ in range(self.n_step):
            # pre-decision
            policy, action = self._get_policy(ob, done)
            # post-decision
            value = self._get_value(ob, done, action)
            # transition
            self.env.update_fingerprint(policy)
            next_ob, reward, done, global_reward = self.env.step(action)
            # 将全局奖励记录到 episode_rewards ，更新步数
            self.episode_rewards.append(global_reward)
            global_step = self.global_counter.next() 
            self.cur_step += 1

            # 将 next_ob 和 done 转为 GPU 张量
            # next_ob = torch.as_tensor(next_ob, device=device)
            next_ob = [torch.as_tensor(o, device=device) for o in next_ob]
            done = torch.as_tensor(done, device=device)

            # collect experience。记录当前转移 transition
            if self.agent.startswith('ma2c'):
                self.model.add_transition(ob, self.ps, action, reward, value, done)
            elif self.agent == 'wpo_cu':
                # WPO算法使用当前动作而不是邻居动作
                action_np = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action
                self.model.add_transition(ob, action_np, action, reward, value, done)
            else:
                self.model.add_transition(ob, self.naction, action, reward, value, done)
            # logging
            if self.global_counter.should_log():
                logging.info('''Training: global step %d, episode step %d,
                                   ob: %s, a: %s, pi: %s, r: %.2f, train r: %.2f, done: %r''' %
                             (global_step, self.cur_step,
                              str(ob), str(action), str(policy), global_reward, np.mean(reward), done))
                # logging.info('''Training: global step %d, episode step %d,
                #                ob: %s, a: %s, pi: %s, r: %.2f, train r: %.2f, done: %r''' % 
                #             (global_step, self.cur_step,
                #             str(ob.cpu().numpy()), str(action.cpu().numpy()), str(policy.cpu().numpy()), 
                #             global_reward, torch.mean(reward).item(), done.item()))
            # terminal check must be inside batch loop for CACC env
            if done:
                break
            ob = next_ob
        if done:
            # R = np.zeros(self.model.n_agent)
            R = torch.zeros(self.model.n_agent, device=device)  ## 初始化 R 在 GPU 上
        else:
            _, action = self._get_policy(ob, done)
            R = self._get_value(ob, done, action)
            # R = R.to(device)  ##+
        return ob, done, R

    # 在不依赖随机策略的前提下，使用其当前最优策略来完成任务，并记录其表现（如奖励分数）
    # 反复执行测试过程，记录表现，以获得平均和波动的表现分数，便于对智能体进行评估
    # 返回平均奖励 mean_reward 和标准差 std_reward
    def perform(self, test_ind, gui=False):
        ob = self.env.reset(gui=gui, test_ind=test_ind)
        ob = [torch.as_tensor(o,device=device) for o in ob]  ## 初始化 ob 到 GPU
        # ob = torch.as_tensor(self.env.reset(gui=gui, test_ind=test_ind), device=device)  ## 初始化 ob 到 GPU
        rewards = []
        # note this done is pre-decision to reset LSTM states!
        # done = True
        done = torch.as_tensor(True, device=device)  # 初始化 done 在 GPU 上
        self.model.reset()
        while True:
            if self.agent == 'greedy':
                action = self.model.forward(ob)
            else:
                # in on-policy learning, test policy has to be stochastic
                if self.env.name.startswith('atsc'):
                    policy, action = self._get_policy(ob, done)
                else:
                    # for mission-critic tasks like CACC, we need deterministic policy
                    # 对于基于 critic 的任务（如 CACC），使用确定性策略
                    policy, action = self._get_policy(ob, done, mode='test')
                self.env.update_fingerprint(policy)

            next_ob, reward, done, global_reward = self.env.step(action)
            rewards.append(global_reward)
            if done:
                break
            # ob = torch.as_tensor(next_ob, device=device)   #############
            done = torch.as_tensor(done, device=device)
        mean_reward = np.mean(np.array(rewards))
        std_reward = np.std(np.array(rewards))
        return mean_reward, std_reward

    def run(self):
        while not self.global_counter.should_stop(): ##-
        # i = 0
        # while(i<=5):
        #     i = i + 1
            # np.random.seed(self.env.seed)
            # ob = self.env.reset()
            # note this done is pre-decision to reset LSTM states!
            # done = True

            # ob = torch.as_tensor(self.env.reset(), device=device)  # 初始化 ob 到 GPU

            obs_list = self.env.reset()
            ob = [torch.as_tensor(o, dtype=torch.float32, device=device) for o in obs_list]

            done = torch.as_tensor(True, device=device)  # 初始化 done 到 GPU
            self.model.reset()
            self.cur_step = 0
            self.episode_rewards = []
            while True:
                ob, done, R = self.explore(ob, done) ###*step in
                dt = self.env.T - self.cur_step 
                global_step = self.global_counter.cur_step
                self.model.backward(R, dt, self.summary_writer, global_step) ###*step in
                # termination
                if done:
                    self.env.terminate()
                    # pytorch implementation is faster, wait SUMO for 1s
                    # time.sleep(1) ##-
                    break
            # rewards = np.array(self.episode_rewards)
            # mean_reward = np.mean(rewards)
            # std_reward = np.std(rewards)
            rewards = torch.tensor(self.episode_rewards, device=device)
            mean_reward = torch.mean(rewards).item()
            std_reward = torch.std(rewards).item() #当前回合奖励的标准差
            
            # NOTE: for CACC we have to run another testing episode after each
            # training episode since the reward and policy settings are different!
            if not self.env.name.startswith('atsc'):
                self.env.train_mode = False
                mean_reward, std_reward = self.perform(-1) ###*step in测试模型在当前环境中的性能，返回测试的平均和标准差奖励。
                self.env.train_mode = True
            
            self._log_episode(global_step, mean_reward, std_reward)
            
            # 定期保存checkpoint
            if self.save_interval and self.model_dir:
                if global_step - self.last_saved_step >= self.save_interval:
                    self.model.save(self.model_dir, global_step)
                    self.last_saved_step = global_step

        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')


class Tester(Trainer):
    def __init__(self, env, model, global_counter, summary_writer, output_path):
        super().__init__(env, model, global_counter, summary_writer)
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.data = []
        logging.info('Testing: total test num: %d' % self.test_num)

    def run_offline(self):
        # enable traffic measurments for offline test
        is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        rewards = []
        for test_ind in range(self.test_num):
            rewards.append(self.perform(test_ind))
            self.env.terminate()
            time.sleep(2)
            self.env.collect_tripinfo()
        avg_reward = np.mean(np.array(rewards))
        logging.info('Offline testing: avg R: %.2f' % avg_reward)
        self.env.output_data()

    def run_online(self, coord):
        self.env.cur_episode = 0
        while not coord.should_stop():
            time.sleep(30)
            if self.global_counter.should_test():
                rewards = []
                global_step = self.global_counter.cur_step
                for test_ind in range(self.test_num):
                    cur_reward = self.perform(test_ind)
                    self.env.terminate()
                    rewards.append(cur_reward)
                    log = {'agent': self.agent,
                           'step': global_step,
                           'test_id': test_ind,
                           'reward': cur_reward}
                    self.data.append(log)
                avg_reward = np.mean(np.array(rewards))
                self._add_summary(avg_reward, global_step)
                logging.info('Testing: global step %d, avg R: %.2f' %
                             (global_step, avg_reward))
                # self.global_counter.update_test(avg_reward)
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')


class Evaluator(Tester):
    def __init__(self, env, model, output_path, gui=False):
        self.env = env
        self.model = model
        self.agent = self.env.agent
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.gui = gui

    def run(self):
        if self.gui:
            is_record = False
        else:
            is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        time.sleep(1)
        for test_ind in range(self.test_num):
            reward, _ = self.perform(test_ind, gui=self.gui)
            self.env.terminate()
            logging.info('test %i, avg reward %.2f' % (test_ind, reward))
            time.sleep(2)
            self.env.collect_tripinfo()
        self.env.output_data()
