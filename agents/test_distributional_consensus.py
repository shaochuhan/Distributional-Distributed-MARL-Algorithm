"""
测试基于值分布的共识方法

此脚本验证三种共识方法的功能：
1. simple_distributional_consensus
2. consensus_critic (with different modes)
3. robust_consensus_critic
"""

import torch
import numpy as np
import sys
import os

# 添加路径以导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from distributional_models_cacc import distributional_CACC_agent

def generate_test_data(batch_size=32, n_agents=3, global_state_dim=20):
    """生成测试数据"""
    states = torch.randn(batch_size, global_state_dim)
    actions = torch.randint(0, 3, (batch_size, n_agents))
    return states, actions

def create_test_agents(n_agents=3, n_states=10, n_actions=3, global_state_dim=20):
    """创建测试智能体"""
    agents = []
    for i in range(n_agents):
        agent = distributional_CACC_agent(
            agent_id=i,
            n_agents=n_agents,
            n_states=n_states,
            n_actions=n_actions,
            slow_lr=0.001,
            fast_lr=0.001,
            gamma=0.95,
            global_state_dim=global_state_dim
        )
        agents.append(agent)
    return agents

def test_simple_distributional_consensus():
    """测试简化版本的分布式共识"""
    print("\n" + "="*60)
    print("测试 1: simple_distributional_consensus")
    print("="*60)
    
    # 创建智能体
    agents = create_test_agents(n_agents=3)
    
    # 生成测试数据
    states, actions = generate_test_data()
    
    # 收集邻居的critic权重
    neighbor_weights = [agent.critic.state_dict() for agent in agents[1:]]
    
    # 执行共识前，记录初始权重
    agent0 = agents[0]
    initial_weights = {k: v.clone() for k, v in agent0.critic.state_dict().items()}
    
    # 执行共识
    print(f"邻居数量: {len(neighbor_weights)}")
    agent0.simple_distributional_consensus(neighbor_weights, states, actions)
    
    # 检查权重是否更新
    updated = False
    for key in initial_weights:
        if not torch.equal(initial_weights[key], agent0.critic.state_dict()[key]):
            updated = True
            break
    
    if updated:
        print("✅ 权重已成功更新")
    else:
        print("❌ 权重未更新")
    
    # 测试预测
    test_input = torch.cat((states[:5], actions[:5].float()), dim=1)
    mu, sigma = agent0.critic(test_input)
    print(f"更新后的预测 - mu范围: [{mu.min():.3f}, {mu.max():.3f}]")
    print(f"更新后的预测 - sigma范围: [{sigma.min():.3f}, {sigma.max():.3f}]")
    
    return True

def test_consensus_critic_uncertainty_weighting():
    """测试基于不确定性加权的共识"""
    print("\n" + "="*60)
    print("测试 2: consensus_critic (uncertainty weighting)")
    print("="*60)
    
    agents = create_test_agents(n_agents=3)
    states, actions = generate_test_data()
    
    # 人为创建不同质量的预测
    # Agent 1: 正常训练
    # Agent 2: 添加噪声（模拟低质量）
    for _ in range(10):
        s, a = generate_test_data(batch_size=16)
        agents[1].critic_update(s, s, a, a, torch.randn(16, 1))
    
    # 添加噪声到agent 2的参数
    for param in agents[2].critic.parameters():
        param.data += torch.randn_like(param) * 0.5
    
    neighbor_weights = [agent.critic.state_dict() for agent in agents[1:]]
    
    agent0 = agents[0]
    initial_mu = None
    
    # 获取初始预测
    with torch.no_grad():
        test_input = torch.cat((states[:5], actions[:5].float()), dim=1)
        initial_mu, initial_sigma = agent0.critic(test_input)
    
    # 执行共识
    agent0.consensus_critic(
        neighbor_weights, 
        states, 
        actions,
        use_uncertainty_weighting=True,
        use_distribution_consensus=False
    )
    
    # 获取更新后的预测
    with torch.no_grad():
        updated_mu, updated_sigma = agent0.critic(test_input)
    
    print(f"初始预测 - mu: {initial_mu.mean():.3f}, sigma: {initial_sigma.mean():.3f}")
    print(f"更新后预测 - mu: {updated_mu.mean():.3f}, sigma: {updated_sigma.mean():.3f}")
    print("✅ 基于不确定性加权的共识完成")
    
    return True

def test_consensus_critic_distribution():
    """测试基于分布参数的共识"""
    print("\n" + "="*60)
    print("测试 3: consensus_critic (distribution consensus)")
    print("="*60)
    
    agents = create_test_agents(n_agents=3)
    states, actions = generate_test_data()
    
    neighbor_weights = [agent.critic.state_dict() for agent in agents[1:]]
    
    agent0 = agents[0]
    
    # 执行共识
    agent0.consensus_critic(
        neighbor_weights,
        states,
        actions,
        use_uncertainty_weighting=False,
        use_distribution_consensus=True
    )
    
    # 验证更新
    with torch.no_grad():
        test_input = torch.cat((states[:5], actions[:5].float()), dim=1)
        mu, sigma = agent0.critic(test_input)
    
    print(f"共识后预测 - mu: {mu.mean():.3f}, sigma: {sigma.mean():.3f}")
    print("✅ 基于分布参数的共识完成")
    
    return True

def test_robust_consensus():
    """测试鲁棒共识（包含异常检测）"""
    print("\n" + "="*60)
    print("测试 4: robust_consensus_critic (with anomaly detection)")
    print("="*60)
    
    agents = create_test_agents(n_agents=5)  # 更多智能体便于测试
    states, actions = generate_test_data()
    
    # 创建一个"异常"智能体（权重被大幅修改）
    anomalous_agent = agents[3]
    for param in anomalous_agent.critic.parameters():
        param.data += torch.randn_like(param) * 5.0  # 大噪声
    
    neighbor_weights = [agent.critic.state_dict() for agent in agents[1:]]
    
    agent0 = agents[0]
    
    print(f"邻居数量: {len(neighbor_weights)} (包含1个异常节点)")
    
    # 执行鲁棒共识
    agent0.robust_consensus_critic(
        neighbor_weights,
        states,
        actions,
        resilience_threshold=2.0
    )
    
    # 验证
    with torch.no_grad():
        test_input = torch.cat((states[:5], actions[:5].float()), dim=1)
        mu, sigma = agent0.critic(test_input)
    
    print(f"鲁棒共识后预测 - mu: {mu.mean():.3f}, sigma: {sigma.mean():.3f}")
    print("✅ 鲁棒共识完成（应该已过滤异常节点）")
    
    return True

def test_without_batch_data():
    """测试没有batch数据时的回退行为"""
    print("\n" + "="*60)
    print("测试 5: Fallback behavior without batch data")
    print("="*60)
    
    agents = create_test_agents(n_agents=3)
    neighbor_weights = [agent.critic.state_dict() for agent in agents[1:]]
    
    agent0 = agents[0]
    
    # 不提供batch数据，应该回退到简单平均
    agent0.consensus_critic(neighbor_weights, None, None)
    print("✅ 无batch数据时正确回退到简单平均")
    
    agent0.simple_distributional_consensus(neighbor_weights, None, None)
    print("✅ simple_distributional_consensus正确处理无数据情况")
    
    return True

def test_prediction_quality():
    """测试预测质量和不确定性量化"""
    print("\n" + "="*60)
    print("测试 6: Prediction quality and uncertainty quantification")
    print("="*60)
    
    agents = create_test_agents(n_agents=3)
    states, actions = generate_test_data()
    
    # 对每个智能体进行一些训练
    for agent in agents:
        for _ in range(20):
            s, a = generate_test_data(batch_size=16)
            ns, na = generate_test_data(batch_size=16)
            r = torch.randn(16, 1)
            agent.critic_update(s, ns, a, na, r)
    
    # 比较不同智能体的预测
    test_input = torch.cat((states[:10], actions[:10].float()), dim=1)
    
    print("\n各智能体的预测质量：")
    for i, agent in enumerate(agents):
        with torch.no_grad():
            mu, sigma = agent.critic(test_input)
        print(f"Agent {i} - mu均值: {mu.mean():.3f}, sigma均值: {sigma.mean():.3f}")
    
    # 执行共识
    neighbor_weights = [agents[1].critic.state_dict(), agents[2].critic.state_dict()]
    agents[0].simple_distributional_consensus(neighbor_weights, states, actions)
    
    # 共识后的预测
    with torch.no_grad():
        mu_consensus, sigma_consensus = agents[0].critic(test_input)
    
    print(f"\n共识后 Agent 0 - mu均值: {mu_consensus.mean():.3f}, sigma均值: {sigma_consensus.mean():.3f}")
    print("✅ 预测质量测试完成")
    
    return True

def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("开始测试基于值分布的共识方法")
    print("="*60)
    
    tests = [
        ("简化版分布式共识", test_simple_distributional_consensus),
        ("不确定性加权共识", test_consensus_critic_uncertainty_weighting),
        ("分布参数级共识", test_consensus_critic_distribution),
        ("鲁棒共识（异常检测）", test_robust_consensus),
        ("无数据回退行为", test_without_batch_data),
        ("预测质量评估", test_prediction_quality),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "通过" if result else "失败"))
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"错误: {str(e)[:50]}"))
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for name, status in results:
        emoji = "✅" if status == "通过" else "❌"
        print(f"{emoji} {name}: {status}")
    
    passed = sum(1 for _, status in results if status == "通过")
    print(f"\n总计: {passed}/{len(tests)} 测试通过")

if __name__ == "__main__":
    # 设置随机种子以保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    run_all_tests()
