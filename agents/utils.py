import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
initializers
"""
def init_layer(layer, layer_type):
    if layer_type == 'fc':
        nn.init.orthogonal_(layer.weight.data)
        nn.init.constant_(layer.bias.data, 0)
    elif layer_type == 'lstm':
        nn.init.orthogonal_(layer.weight_ih.data)
        nn.init.orthogonal_(layer.weight_hh.data)
        nn.init.constant_(layer.bias_ih.data, 0)
        nn.init.constant_(layer.bias_hh.data, 0)

"""
layer helpers
"""
def batch_to_seq(x):
    n_step = x.shape[0]
    if len(x.shape) == 1:
        x = torch.unsqueeze(x, -1)
    return torch.chunk(x, n_step)


def run_rnn(layer, xs, dones, s):
    xs = xs.to(device)
    dones = dones.to(device)
    s = s.to(device) 

    xs = batch_to_seq(xs)
    # need dones to reset states
    dones = batch_to_seq(dones)
    n_in = int(xs[0].shape[1])
    n_out = int(s.shape[0]) // 2
    s = torch.unsqueeze(s, 0)
    h, c = torch.chunk(s, 2, dim=1)
    outputs = []
    for ind, (x, done) in enumerate(zip(xs, dones)):
        if done.dtype == torch.bool:
            done = done.float()
        c = c * (1-done)
        h = h * (1-done)
        h, c = layer(x, (h, c))
        outputs.append(h)
    s = torch.cat([h, c], dim=1)
    return torch.cat(outputs), torch.squeeze(s)


# def one_hot(x, oh_dim, dim=-1):
#     # 如果是浮点数 -> 连续动作，直接返回 vmas
#     # if x.dtype in [torch.float32, torch.float64]:
#     #     if len(x.shape) == 1:
#     #         x = x.unsqueeze(0)
#     #     elif len(x.shape) >= 2:
#     #         batch_size = x.size(0)
#     #         x = x.view(batch_size, -1)
#     #     return x.to(device)
#     # else:
#     oh_shape = list(x.shape)
#     if dim == -1:
#         oh_shape.append(oh_dim)
#     else:
#         oh_shape = oh_shape[:dim+1] + [oh_dim] + oh_shape[dim+1:]
#     # x_oh = torch.zeros(oh_shape)
#     x_oh = torch.zeros(oh_shape, device=device) ##
#     x = torch.unsqueeze(x, -1)
#     if dim == -1:
#         x_oh = x_oh.scatter(dim, x, 1)
#     else:
#         x_oh = x_oh.scatter(dim+1, x, 1)
#     return x_oh



def one_hot(x, n_class):
    """使用PyTorch内置one_hot函数"""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    
    x = x.long().to(device)
    
    # PyTorch的F.one_hot会自动处理维度
    return F.one_hot(x, num_classes=n_class).float()


"""
buffers
"""
class TransBuffer:
    def reset(self):
        self.buffer = []

    @property
    def size(self):
        return len(self.buffer)

    def add_transition(self, ob, a, r, *_args, **_kwargs):
        raise NotImplementedError()

    def sample_transition(self, *_args, **_kwargs):
        raise NotImplementedError()


class OnPolicyBuffer(TransBuffer):
    def __init__(self, gamma, alpha, distance_mask):
        self.gamma = gamma
        self.alpha = alpha
        if alpha > 0:
            self.distance_mask = distance_mask
            self.max_distance = np.max(distance_mask, axis=-1)
        self.reset()

    def reset(self, done=False):
        # the done before each step is required
        self.obs = []
        self.acts = []
        self.rs = []
        self.vs = []
        self.adds = []
        self.dones = [done]

    def add_transition(self, ob, na, a, r, v, done):
        self.obs.append(ob)
        self.adds.append(na)
        self.acts.append(a)
        self.rs.append(r)
        self.vs.append(v)
        self.dones.append(done)

    def sample_transition(self, R, dt=0):
        if self.alpha < 0:
            self._add_R_Adv(R)
        else:
            self._add_s_R_Adv(R)

        # obs = np.array(self.obs, dtype=np.float32)

        # nas = np.array(self.adds, dtype=np.int32)
        # acts = np.array(self.acts, dtype=np.int32)
        # Rs = np.array(self.Rs, dtype=np.float32)
        # Advs = np.array(self.Advs, dtype=np.float32)
        # dones = np.array(self.dones[:-1], dtype=np.bool_)
        ##
        # 转换为tensor（如果还不是的话）
        obs_tensors = [torch.as_tensor(o, dtype=torch.float32) if not isinstance(o, torch.Tensor) else o for o in self.obs]
        nas_tensors = [torch.as_tensor(n, dtype=torch.int32) if not isinstance(n, torch.Tensor) else n for n in self.adds]
        acts_tensors = [torch.as_tensor(a, dtype=torch.int32) if not isinstance(a, torch.Tensor) else a for a in self.acts]
        Rs_tensors = [torch.as_tensor(r, dtype=torch.float32) if not isinstance(r, torch.Tensor) else r for r in self.Rs]
        Advs_tensors = [torch.as_tensor(a, dtype=torch.float32) if not isinstance(a, torch.Tensor) else a for a in self.Advs]
        
        obs = torch.stack(obs_tensors).detach().cpu().numpy().astype(np.int32)
        nas = torch.stack(nas_tensors).detach().cpu().numpy().astype(np.int32)
        acts = torch.stack(acts_tensors).detach().cpu().numpy().astype(np.int32)
        Rs = torch.stack(Rs_tensors).detach().cpu().numpy().astype(np.float32)
        Advs = torch.stack(Advs_tensors).detach().cpu().numpy().astype(np.float32)
        dones_list = self.dones[:-1]
        dones = torch.tensor(dones_list).detach().cpu().numpy().astype(bool)
        # use pre-step dones here
        self.reset(self.dones[-1])
        return obs, nas, acts, dones, Rs, Advs

    def _add_R_Adv(self, R):
        Rs = []
        Advs = []

        # use post-step dones here
        # for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
        #     # 修复版本
        #     if done==False and len(R) == 2:
        #         # R是分布式价值 (μ, σ)
        #         R_mu, R_sigma = R
        #         ### 分别更新均值和标准差
        #         R_mu = r + self.gamma * R_mu * (1.0 - done.float())
        #         R_sigma = self.gamma * R_sigma * (1.0 - done.float()) + 0.01  # 添加最小不确定性
        #         R_sigma = torch.clamp(R_sigma, min=0.001, max=10.0)  # 确保标准差合理
        #         Adv = R_mu - v
        #         # R = (R_mu, R_sigma)
        #         # R = R_mu + self.gamma * R_sigma
        #         R = torch.stack([R_mu, R_sigma])
        #     else:
        #         R = r + self.gamma * R * (1.-done.float())
        #         Adv = R - v
        #     Rs.append(R)
        #     Advs.append(Adv)
        for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
            try:
                # 尝试当作分布式价值处理
                if len(R) == 2:
                    # R是分布式价值 (μ, σ)
                    R_mu, R_sigma = R
                    ### 分别更新均值和标准差
                    R_mu = r + self.gamma * R_mu * (1.0 - done.float())
                    R_sigma = self.gamma * R_sigma * (1.0 - done.float()) + 0.01  # 添加最小不确定性
                    R_sigma = torch.clamp(R_sigma, min=0.001, max=10.0)  # 确保标准差合理
                    Adv = R_mu - v
                    # R = (R_mu, R_sigma)
                    # R = R_mu + self.gamma * R_sigma
                    R = torch.stack([R_mu, R_sigma])
                else:
                    R = r + self.gamma * R * (1.-done.float())
                    Adv = R - v
            except (TypeError, AttributeError):
                # 如果len()失败，说明R是0维tensor，按标量处理
                R = r + self.gamma * R * (1.0 - done.float())
                Adv = R.cpu() - v
            Rs.append(R)
            Advs.append(Adv)
        Rs.reverse()
        Advs.reverse()
        self.Rs = Rs
        self.Advs = Advs

    def _add_st_R_Adv(self, R, dt):
        Rs = []
        Advs = []
        # use post-step dones here
        tdiff = dt
        for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
            R = self.gamma * R * (1.-done)
            if done:
                tdiff = 0
            # additional spatial rewards
            tmax = min(tdiff, self.max_distance)
            for t in range(tmax + 1):
                rt = np.sum(r[self.distance_mask == t])
                R += (self.gamma * self.alpha) ** t * rt
            Adv = R - v
            tdiff += 1
            Rs.append(R)
            Advs.append(Adv)
        Rs.reverse()
        Advs.reverse()
        self.Rs = Rs
        self.Advs = Advs

    def _add_s_R_Adv(self, R):
        Rs = []
        Advs = []
        # use post-step dones here
        for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
            # 确保R和done都是tensor且在同一设备上
            if not isinstance(R, torch.Tensor):
                R = torch.tensor(R, dtype=torch.float32)
            
            if not isinstance(done, torch.Tensor):
                done = torch.tensor(done, dtype=torch.float32, device=R.device)
            else:
                done = done.float().to(R.device)
            
            R = self.gamma * R * (1. - done)
            
            # additional spatial rewards
            # 支持r为标量（单个奖励值）或数组（空间奖励）
            if isinstance(r, (int, float, np.floating)):
                # 标量奖励：直接使用
                R += r
            else:
                # 数组奖励：按距离加权
                for t in range(self.max_distance + 1):
                    rt = np.sum(r[self.distance_mask == t])
                    R += (self.alpha ** t) * rt
            
            # 计算advantage
            # 如果R是分布式（有多个值，如μ和σ），只使用第一个值（均值）
            if isinstance(R, torch.Tensor) and R.numel() > 1:
                R_for_adv = R[0] if R.dim() > 0 else R
            else:
                R_for_adv = R
                
            Adv = R_for_adv.cpu() - v
            Rs.append(R)
            Advs.append(Adv)
        Rs.reverse()
        Advs.reverse()
        self.Rs = Rs
        self.Advs = Advs


class MultiAgentOnPolicyBuffer(OnPolicyBuffer):
    def __init__(self, gamma, alpha, distance_mask):
        super().__init__(gamma, alpha, distance_mask)
    
    ##grid
    def _convert_to_numpy(self, data):
        """通用的张量到numpy转换函数"""
        result_list = []
        for item in data:
            if isinstance(item, (list, np.ndarray)):
                # 如果是列表或数组，转换其中的每个张量
                converted_item = []
                for sub_item in item:
                    if isinstance(sub_item, torch.Tensor):
                        converted_item.append(sub_item.detach().cpu().numpy())
                    else:
                        converted_item.append(sub_item)
                result_list.append(np.array(converted_item))
            elif isinstance(item, torch.Tensor):
                result_list.append(item.detach().cpu().numpy())
            else:
                result_list.append(item)
        return np.stack(result_list, axis=0)
    
    def _convert_to_numpy_simple(self, data):
        """简化版本的张量到numpy转换，直接处理张量列表"""
        if isinstance(data, list):
            result_list = []
            for item in data:
                if isinstance(item, torch.Tensor):
                    result_list.append(item.detach().cpu().numpy())
                elif isinstance(item, np.ndarray):
                    result_list.append(item)
                else:
                    result_list.append(np.array(item))
            return np.array(result_list)
        else:
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()
            else:
                return np.array(data)

    def sample_transition(self, R, dt=0):
        # alpha是空间折扣因子，小于0时不考虑空间维度
        if self.alpha < 0:
            self._add_R_Adv(R)
        else:
            self._add_s_R_Adv(R)
        # obs = np.transpose(np.array(self.obs, dtype=np.float32), (1, 0, 2))
        # policies = np.transpose(np.array(self.adds, dtype=np.float32), (1, 0, 2))
        # acts = np.transpose(np.array(self.acts, dtype=np.int32))
        # 将 Tensor 类型转换为 NumPy 数组 ##
        # obs_np = [tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in self.obs]
        obs_np = [[t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in time_step]
                    for time_step in self.obs]
        adds_np = [tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in self.adds]
        acts_np = [tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in self.acts]
        dones_np = [tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in self.dones[:-1]]

        # 转换为 NumPy 数组并执行转置
        obs = np.transpose(np.array(obs_np, dtype=np.float32), (1, 0, 2))
        policies = np.transpose(np.array(adds_np, dtype=np.float32), (1, 0, 2))
        # acts = np.transpose(np.array(acts_np, dtype=np.int32))
        acts_np_cpu = [
            [a.cpu().numpy() if isinstance(a, torch.Tensor) else np.array(a) for a in step]
            for step in acts_np
        ]
        # 转成 numpy array
        acts = np.array(acts_np_cpu, dtype=np.float32) 

        dones = np.array(dones_np, dtype=np.bool_)

        # Rs = self.Rs
        # np_list = [t.detach().cpu().numpy() for t in self.Rs]
        # Rs = np.array(np_list, dtype=np.float32)
        # Rs = np.array(self.Rs, dtype=np.float32)

        # Rs_np = [r.detach().cpu().numpy() if isinstance(r, torch.Tensor) else r for r in self.Rs]
        # Rs = np.stack(Rs_np, axis=0)

         # 使用通用转换函数处理Rs和Advs
        # Rs = self._convert_to_numpy(self.Rs)
        # Advs = self._convert_to_numpy(self.Advs)
        # Rs = self._convert_to_numpy_simple(self.Rs)
        # Advs = self._convert_to_numpy_simple(self.Advs)

        # 处理 Rs - self.Rs 已经是 numpy 数组
        if isinstance(self.Rs, np.ndarray):
            Rs = torch.from_numpy(self.Rs).float()
        else:
            Rs_data = []
            for r in self.Rs.flat:
                if isinstance(r, torch.Tensor):
                    Rs_data.append(r.detach().cpu())
                else:
                    Rs_data.append(torch.tensor(r, dtype=torch.float32))
            Rs = torch.stack(Rs_data).view(self.Rs.shape + (-1,))

        # 处理 Advs - self.Advs 已经是 numpy 数组
        if isinstance(self.Advs, np.ndarray):
            Advs = torch.from_numpy(self.Advs).float()
        else:
            Advs_data = []
            for r in self.Advs.flat:
                if isinstance(r, torch.Tensor):
                    Advs_data.append(r.detach().cpu())
                else:
                    Advs_data.append(torch.tensor(r, dtype=torch.float32))
            Advs = torch.stack(Advs_data).view(self.Advs.shape + (-1,))

        # Advs = self.Advs
        # dones = np.array(self.dones[:-1].cpu().numpy(), dtype=np.bool_) ##
        self.reset(self.dones[-1])
        # return obs, policies, acts, dones, Rs, Advs ##
        return (
            torch.as_tensor(obs, device=device),
            torch.as_tensor(policies, device=device),
            torch.as_tensor(acts, device=device),
            torch.as_tensor(dones, device=device),
            # torch.as_tensor(Rs.cpu().detach(), device=device),
            torch.as_tensor(Rs, device=device),
            torch.as_tensor(Advs, device=device),
        )

    # 根据时间折扣因子 γ 和策略值 v 计算每个智能体的奖励和优势值序列
    def _add_R_Adv(self, R):
        Rs = []  #每一时间步的即时奖励序列
        Advs = []  
        vs = np.array(self.vs)
        for i in range(vs.shape[1]):
            cur_Rs = []
            cur_Advs = []
            cur_R = R[i]
            for r, v, done in zip(self.rs[::-1], vs[::-1,i], self.dones[:0:-1]):
                # done = done.cpu().numpy() if isinstance(done, torch.Tensor) else done
                cur_R = r + self.gamma * cur_R * (1.-done.float()) ##当前累积奖励 = 即时奖励 + 折扣因子乘后续奖励
                cur_Adv = cur_R - v #优势值
                cur_Rs.append(cur_R)
                cur_Advs.append(cur_Adv)
            cur_Rs.reverse()
            cur_Advs.reverse()
            Rs.append(cur_Rs)
            Advs.append(cur_Advs)
        # self.Rs = np.array(Rs)
        # self.Advs = np.array(Advs)
        self.Rs = np.array([[r.detach().cpu().numpy() if isinstance(r, torch.Tensor) else r for r in row] for row in Rs])
        self.Advs = np.array([[r.detach().cpu().numpy() if isinstance(r, torch.Tensor) else r for r in row] for row in Advs])
        # self.Rs = np.array([[r.item() for r in row] for row in Rs])
        # self.Advs = np.array([[r.item() for r in row] for row in Advs])


    def _add_st_R_Adv(self, R, dt):
        Rs = []
        Advs = []
        vs = np.array(self.vs)
        for i in range(vs.shape[1]):
            cur_Rs = []
            cur_Advs = []
            cur_R = R[i]
            tdiff = dt
            distance_mask = self.distance_mask[i]
            max_distance = self.max_distance[i]
            for r, v, done in zip(self.rs[::-1], vs[::-1,i], self.dones[:0:-1]):
                cur_R = self.gamma * cur_R * (1.-done)
                if done:
                    tdiff = 0
                # additional spatial rewards
                tmax = min(tdiff, max_distance)
                for t in range(tmax + 1):
                    rt = np.sum(r[distance_mask==t])
                    cur_R += (self.gamma * self.alpha) ** t * rt
                cur_Adv = cur_R - v
                tdiff += 1
                cur_Rs.append(cur_R)
                cur_Advs.append(cur_Adv)
            cur_Rs.reverse()
            cur_Advs.reverse()
            Rs.append(cur_Rs)
            Advs.append(cur_Advs)
        self.Rs = np.array(Rs)
        self.Advs = np.array(Advs)

    # 引入空间上的折扣和奖励计算 
    def _add_s_R_Adv(self, R):
        Rs = []
        Advs = []
        vs = np.array(self.vs)
        for i in range(vs.shape[1]):
            cur_Rs = []
            cur_Advs = []
            cur_R = R[i]
            distance_mask = self.distance_mask[i]
            max_distance = self.max_distance[i]
            for r, v, done in zip(self.rs[::-1], vs[::-1,i], self.dones[:0:-1]):
                cur_R = self.gamma * cur_R * (1.-done.float())
                # additional spatial rewards
                for t in range(max_distance.astype(int) + 1):
                    # rt = np.sum(r[distance_mask==t])
                    mask = (distance_mask == t)
                    # rt = r[mask].sum() ###加入vmas
                    rt = r.sum() 
                    cur_R += (self.alpha ** t) * rt
                cur_Adv = cur_R - v
                cur_Rs.append(cur_R)
                cur_Advs.append(cur_Adv)
            cur_Rs.reverse()
            cur_Advs.reverse()
            Rs.append(cur_Rs)
            Advs.append(cur_Advs)
        # self.Rs = np.array(Rs)
        # self.Advs = np.array(Advs)
        self.Rs = np.array([[r.item() if isinstance(r, torch.Tensor) else r for r in agent_rs] for agent_rs in Rs], dtype=np.float32)
        self.Advs = np.array([[adv.item() if isinstance(adv, torch.Tensor) else adv for adv in agent_advs] for agent_advs in Advs], dtype=np.float32)

"""
util functions
"""
class Scheduler:
    def __init__(self, val_init, val_min=0, total_step=0, decay='linear'):
        self.val = val_init
        self.N = float(total_step)
        self.val_min = val_min
        self.decay = decay
        self.n = 0

    def get(self, n_step):
        self.n += n_step
        if self.decay == 'linear':
            return max(self.val_min, self.val * (1 - self.n / self.N))
        elif self.decay == 'exponential':
            # 指数衰减: lr = lr_init * exp(-decay_rate * step)
            import math
            decay_rate = -math.log(self.val_min / self.val) / self.N
            return max(self.val_min, self.val * math.exp(-decay_rate * self.n))
        elif self.decay == 'cosine':
            # 余弦退火: lr = lr_min + 0.5 * (lr_init - lr_min) * (1 + cos(pi * step / total_step))
            import math
            return self.val_min + 0.5 * (self.val - self.val_min) * (1 + math.cos(math.pi * self.n / self.N))
        else:
            # constant
            return self.val


