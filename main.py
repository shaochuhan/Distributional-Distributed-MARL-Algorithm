import argparse
import configparser
import logging
import threading
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from envs.cacc_env import CACCEnv
from envs.grid_world import Grid_World
from envs.vmas_navigation_env import VmasNavigationEnv
from envs.vmas_unified_env import VmasUnifiedEnv  # 新增：VMAS统一适配器
# from envs.large_grid_env import LargeGridEnv
# from envs.real_net_env import RealNetEnv
from agents.models import IA2C, IA2C_FP, MA2C_NC, IA2C_CU, MA2C_CNET, MA2C_DIAL, DistributionalIA2C_CU
from agents.wpo_models import WPO_IA2C_CU  # WPO 算法导入
from agents.distributional_models import distributional_CAC_agent
from agents.distributional_models_cacc import distributional_CACC_agent
from agents.MaliciousAgentWrapper import MaliciousAgentWrapper, apply_malicious_wrapper, create_malicious_config
from utils import (Counter, Trainer, Tester, Evaluator,
                   check_dir, copy_file, find_file,
                   init_dir, init_log, init_test_flag)
import agents.distributional_trainer as training
import agents.distributional_trainer_cacc as training_cacc
import os
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

def parse_args(): 
    default_base_dir = 'base'
    default_config_dir = './config/config_ia2c_cu_catchup.ini'
    
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='option', help="train or evaluate")
    # 配置 train 子命令
    sp_train = subparsers.add_parser('train', help='train a single agent under base dir')
    sp_train.add_argument('--base-dir', type=str, default=default_base_dir, help="experiment base dir")
    sp_train.add_argument('--config-dir', type=str, required=False, default=default_config_dir, help="experiment config path")
    # 配置 evaluate 子命令
    sp_evaluate = subparsers.add_parser('evaluate', help="evaluate and compare agents under base dir")
    sp_evaluate.add_argument('--base-dir', type=str, default=default_base_dir, help="experiment base dir")
    sp_evaluate.add_argument('--evaluation-seeds', type=str, required=False, default=','.join([str(i) for i in range(2000, 2500, 10)]), help="random seeds for evaluation, split by ,")
    sp_evaluate.add_argument('--demo', action='store_true', help="shows SUMO gui")
    # 添加版本号参数
    sp_train.add_argument('--version', type=str, default='v1', 
                         help="experiment version (e.g., 'v1', 'v2', 'test')")
    # sp_train.add_argument('--malicious-seed', type=int, default=None,
    #                      help="random seed for malicious behavior")
    # sp_train.add_argument('--malicious-log-interval', type=int, default=1000,
    #                      help="log interval for malicious actions (0 to disable)")
    
    
    args = parser.parse_args()
    if not args.option:
        parser.print_help()
        exit(1)
    return args


def init_env(config, port=0):
    scenario = config.get('scenario')
    ##grid
    if scenario == 'grid_world':
        # 可用配置决定初始/目标位置等
        nrow = config.getint('nrow')
        ncol = config.getint('ncol')
        n_agent = config.getint('n_agent')
        initial_state = eval(config.get('initial_state'))  
        desired_state = eval(config.get('desired_state'))  
        randomize = config.getboolean('randomize_state')
        agent = config.get('agent') 
        coop_gamma = config.getfloat('coop_gamma')
        return Grid_World(nrow, ncol, n_agent, desired_state=np.array(desired_state),
                          initial_state=np.array(initial_state), randomize_state=randomize,
                          agent=agent,coop_gamma=coop_gamma)
    elif scenario.startswith('vmas_'):
        # 新增：统一的VMAS场景支持
        # 支持所有VMAS场景：vmas_navigation, vmas_transport, vmas_balance等
        return VmasUnifiedEnv(config)
    elif scenario == 'vmas_navigation_old':
        # 保留旧的特定导航环境（重命名以避免冲突）
        return VmasNavigationEnv(config)
    else:
        return CACCEnv(config)


def init_agent(env, config, config_evn,total_step, seed):
    if env.agent == 'ia2c':
        return IA2C(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                    total_step, config, seed=seed)
    elif env.agent == 'ia2c_fp':
        return IA2C_FP(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       total_step, config, seed=seed)
    elif env.agent == 'ma2c_nc':
        return MA2C_NC(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       total_step, config, seed=seed)
    elif env.agent == 'ma2c_cnet':
        # this is actually CommNet
        return MA2C_CNET(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                         total_step, config, seed=seed)
    elif env.agent == 'ma2c_cu':
        return IA2C_CU(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       total_step, config, seed=seed)
    elif env.agent == 'ma2c_dial':
        return MA2C_DIAL(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                        total_step, config, seed=seed)
    elif env.agent == 'd2ac_cu':
        return DistributionalIA2C_CU(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                         total_step, config, seed=seed)
    elif env.agent == 'wpo_cu':
        # WPO with Consensus Update - 新算法
        return WPO_IA2C_CU(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                          total_step, config, seed=seed)
    elif env.agent == 'd2ac':
        # 获取环境参数
        if config_evn.get('scenario') == 'grid_world':   
            n_agent = config_evn.getint('n_agent')
            n_states = config_evn.getint('n_states')
            n_actions = config_evn.getint('n_actions')
        elif config_evn.get('scenario') == 'cacc_catchup':
            n_agent = config_evn.getint('n_vehicle')
            n_states = sum(env.n_s_ls) # 所有智能体的状态空间总和
            n_actions = env.n_a  # 动作空间大小
        slow_lr = config.getfloat('slow_lr')
        fast_lr = config.getfloat('fast_lr')
        gamma = config.getfloat('gamma')
        # 创建多个智能体实例
        agents = []
        max_state_dim = max(env.n_s_ls)
        for node in range(n_agent):
            # 每个智能体的状态维度是其自身状态加上邻居状态
            if config_evn.get('scenario') == 'grid_world':
                agent = distributional_CAC_agent(node,n_agent,n_states,n_actions,slow_lr,
                          fast_lr,gamma)
            elif config_evn.get('scenario') == 'cacc_catchup':
                n_states = env.n_s_ls[node]  # 当前智能体的局部状态维度
                # global_state_dim = sum(env.n_s_ls)  # 所有智能体状态维度之和
                global_state_dim = max_state_dim * n_agent
                agent = distributional_CACC_agent(node,n_agent,max_state_dim,n_actions,slow_lr,
                          fast_lr,gamma,global_state_dim)           
            agents.append(agent)
        return agents  # 返回智能体列表
        
    else:
        return None

def train(args):
    # 基本目录与日志初始化n_agent
    base_dir = args.base_dir
    # dirs = init_dir(base_dir)
    # init_log(dirs['log']) #日志存放
    # 配置文件复制与读取
    config_dir = args.config_dir
    # copy_file(config_dir, dirs['data'])
    config = configparser.ConfigParser()
    config.read(config_dir)
    # 获取算法名和场景名
    algo_name = config.get('ENV_CONFIG', 'agent')
    scenario_name = config.get('ENV_CONFIG', 'scenario')
    
    # 从命令行参数获取版本号
    version = args.version if hasattr(args, 'version') else 'v1'
    
    # experiment_name = f"{algo_name}_{scenario_name}"
    # 构建路径
    experiment_base_dir = os.path.join(base_dir, experiment_name)
    dirs = init_dir(experiment_base_dir)
    init_log(dirs['log'])
    # 拷贝配置文件
    copy_file(config_dir, dirs['data'])
    # init env
    env = init_env(config['ENV_CONFIG'])
    # 应用恶意智能体包装器
    if hasattr(args, 'malicious_agents'):
        env = apply_malicious_wrapper(env, args.malicious_agents, args.malicious_type)

    logging.info('Training: a dim %r, agent dim: %d' % (env.n_a_ls, env.n_agent))

    # init step counter，从配置文件读取对应参数
    total_step = int(config.getfloat('TRAIN_CONFIG', 'total_step'))
    test_step = int(config.getfloat('TRAIN_CONFIG', 'test_interval'))
    log_step = int(config.getfloat('TRAIN_CONFIG', 'log_interval'))
    global_counter = Counter(total_step, test_step, log_step)

    # init centralized or multi agent
    # 初始化 agent 的模型
    seed = config.getint('ENV_CONFIG', 'seed')
    model = init_agent(env, config['MODEL_CONFIG'], config['ENV_CONFIG'],total_step, seed)
    if env.agent == 'd2ac':
        pass
    else:
        model = model.to(device)

    # model.load(dirs['model'], train_mode=True) 
    # 检查 checkpoint 文件夹是否为空 ##+
    if os.listdir(dirs['model']):
        model.load(dirs['model'], train_mode=True)
    else:
        logging.info("No checkpoint found in {}. Starting training from scratch.".format(dirs['model']))

    # disable multi-threading for safe SUMO implementation
    summary_writer = SummaryWriter(dirs['log'], flush_secs=10000)
    ##分情况训练:
    if env.agent == 'd2ac':
        if config.get('ENV_CONFIG', 'scenario') =='grid_world':
            trained_agents, sim_data = training.train_CAC(env,model,config,summary_writer)
        else:
            trained_agents, sim_data = training_cacc.train_CACC(env,model,config,summary_writer)
        # sim_data.to_csv(f"{args.log_dir}/simulation_results.csv")
        # 保存模型
        for i, agent in enumerate(trained_agents):
            torch.save(agent.actor.state_dict(), f"{args.log_dir}/actor_{i}.pth")
            torch.save(agent.critic.state_dict(), f"{args.log_dir}/critic_{i}.pth")
    else:
        # 读取保存间隔
        save_interval = config.getint('TRAIN_CONFIG', 'save_interval', fallback=None)
        trainer = Trainer(env, model, global_counter, summary_writer, 
                         output_path=dirs['data'],
                         save_interval=save_interval,
                         model_dir=dirs['model'])
        trainer.run()
        # save model
        final_step = global_counter.cur_step
        model.save(dirs['model'], final_step)

    summary_writer.close()


def evaluate_fn(agent_dir, output_dir, seeds, port, demo):
    agent = agent_dir.split('/')[-1]
    if not check_dir(agent_dir):
        logging.error('Evaluation: %s does not exist!' % agent)
        return
    # load config file 
    config_dir = find_file(agent_dir + '/data/')
    if not config_dir:
        return
    config = configparser.ConfigParser()
    config.read(config_dir)

    # init env
    env = init_env(config['ENV_CONFIG'], port=port)
    env.init_test_seeds(seeds)

    # load model for agent
    model = init_agent(env, config['MODEL_CONFIG'], 0, 0).to(device)
    if model is None:
        return
    model_dir = agent_dir + '/model/'
    if not model.load(model_dir):
        return
    # collect evaluation data
    evaluator = Evaluator(env, model, output_dir, gui=demo)
    evaluator.run()


def evaluate(args):
    base_dir = args.base_dir
    if not args.demo:
        dirs = init_dir(base_dir, pathes=['eva_data', 'eva_log'])
        init_log(dirs['eva_log'])
        output_dir = dirs['eva_data']
    else:
        output_dir = None
    # enforce the same evaluation seeds across agents
    seeds = args.evaluation_seeds
    logging.info('Evaluation: random seeds: %s' % seeds)
    if not seeds:
        seeds = []
    else:
        seeds = [int(s) for s in seeds.split(',')]
    evaluate_fn(base_dir, output_dir, seeds, 1, args.demo)


if __name__ == '__main__':
    args = parse_args()
    if args.option == 'train':
        train(args)
    else:
        evaluate(args)
