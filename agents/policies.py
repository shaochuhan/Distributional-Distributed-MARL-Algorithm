import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.utils import batch_to_seq, init_layer, one_hot, run_rnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") ##+

class Policy(nn.Module):
    def __init__(self, n_a, n_s, n_step, policy_name, agent_name, identical):
        super(Policy, self).__init__()
        self.name = policy_name
        if agent_name is not None:
            # for multi-agent system
            self.name += '_' + str(agent_name)
        self.n_a = n_a
        self.n_s = n_s
        self.n_step = n_step
        self.identical = identical
        ##+ 初始化 actor 和 critic head 并移动到指定的设备
        # self._init_actor_head(n_s).to(device)
        # self._init_critic_head(n_s).to(device)


    def forward(self, ob, *_args, **_kwargs):
        raise NotImplementedError()

    def _init_actor_head(self, n_h, n_a=None):
        if n_a is None:
            n_a = self.n_a
        # only discrete control is supported for now
        self.actor_head = nn.Linear(n_h, n_a).to(device) ##
        init_layer(self.actor_head, 'fc')

    def _init_critic_head(self, n_h, n_n=None):
        if n_n is None:
            n_n = int(self.n_n)
        if n_n:
            if self.identical:
                n_na_sparse = self.n_a*n_n
            else:
                n_na_sparse = sum(self.na_dim_ls)
            n_h += n_na_sparse
        self.critic_head = nn.Linear(n_h, 1).to(device) ##
        init_layer(self.critic_head, 'fc')

    def _run_critic_head(self, h, na, n_n=None):
        h = h.to(device) ##+
        if n_n is None:
            n_n = int(self.n_n)
        if n_n:
            na = torch.from_numpy(na).long().to(device) ##
            if self.identical:
                na_sparse = one_hot(na, self.n_a).to(device) ##
                na_sparse = na_sparse.view(-1, self.n_a*n_n)
            else:
                na_sparse = []
                na_ls = torch.chunk(na, n_n, dim=1)
                for na_val, na_dim in zip(na_ls, self.na_dim_ls):
                    na_sparse.append(torch.squeeze(one_hot(na_val, na_dim), dim=1).to(device)) ##
                na_sparse = torch.cat(na_sparse, dim=1)
            h = torch.cat([h, na_sparse], dim=1)
        return self.critic_head(h).squeeze()

    def _run_loss(self, actor_dist, e_coef, v_coef, vs, As, Rs, Advs):
        ##
        As = As.to(device)
        vs = vs.to(device) 
        Rs = Rs.to(device)
        Advs = Advs.to(device)

        log_probs = actor_dist.log_prob(As) 
        policy_loss = -(log_probs * Advs).mean()
        entropy_loss = -(actor_dist.entropy()).mean() * e_coef
        value_loss = (Rs - vs).pow(2).mean() * v_coef
        return policy_loss, value_loss, entropy_loss

    def _update_tensorboard(self, summary_writer, global_step):
        # monitor training
        summary_writer.add_scalar('loss/{}_entropy_loss'.format(self.name), self.entropy_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_policy_loss'.format(self.name), self.policy_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_value_loss'.format(self.name), self.value_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_total_loss'.format(self.name), self.loss,
                                  global_step=global_step)


class LstmPolicy(Policy):
    def __init__(self, n_s, n_a, n_n, n_step, n_fc=64, n_lstm=64, name=None,
                 na_dim_ls=None, identical=True):
        super(LstmPolicy, self).__init__(n_a, n_s, n_step, 'lstm', name, identical)
        if not self.identical:
            self.na_dim_ls = na_dim_ls
        self.n_lstm = n_lstm
        self.n_fc = n_fc
        self.n_n = n_n
        self._init_net()
        self._reset()

    def backward(self, obs, nactions, acts, dones, Rs, Advs,
                 e_coef, v_coef, summary_writer=None, global_step=None):
        obs = torch.from_numpy(obs).float()
        dones = torch.from_numpy(dones).float()
        xs = self._encode_ob(obs)
        hs, new_states = run_rnn(self.lstm_layer, xs, dones, self.states_bw)
        # backward grad is limited to the minibatch
        self.states_bw = new_states.detach()
        actor_dist = torch.distributions.categorical.Categorical(logits=F.log_softmax(self.actor_head(hs), dim=1))
        vs = self._run_critic_head(hs, nactions)
        self.policy_loss, self.value_loss, self.entropy_loss = \
            self._run_loss(actor_dist, e_coef, v_coef, vs,
                           torch.from_numpy(acts).long(),
                           torch.from_numpy(Rs).float(),
                           torch.from_numpy(Advs).float())
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()
        if summary_writer is not None:
            self._update_tensorboard(summary_writer, global_step)

    def forward(self, ob, done, naction=None, out_type='p'):
        # ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float().to(device)
        # done = torch.from_numpy(np.expand_dims(done, axis=0)).float().to(device)
        ob = ob.unsqueeze(0) 
        done = done.unsqueeze(0) 
        x = self._encode_ob(ob)
        h, new_states = run_rnn(self.lstm_layer, x, done, self.states_fw)
        if out_type.startswith('p'):
            self.states_fw = new_states.detach()
            return F.softmax(self.actor_head(h), dim=1).squeeze().detach().cpu().numpy()
        else:
            # 将naction转换为numpy数组，先移到CPU
            if isinstance(naction, torch.Tensor):
                naction_np = naction.detach().cpu().numpy()
            else:
                naction_np = np.array(naction) 
            return self._run_critic_head(h, naction_np).detach().cpu().numpy()
    
            ##return self._run_critic_head(h, np.array([naction])).detach().cpu().numpy()

    def _encode_ob(self, ob):
        ob = ob.to(device).float()
        return F.relu(self.fc_layer(ob))

    def _init_net(self):
        self.fc_layer = nn.Linear(self.n_s, self.n_fc)
        init_layer(self.fc_layer, 'fc')
        self.lstm_layer = nn.LSTMCell(self.n_fc, self.n_lstm)
        init_layer(self.lstm_layer, 'lstm')
        self._init_actor_head(self.n_lstm)
        self._init_critic_head(self.n_lstm)

    def _reset(self):
        # forget the cumulative states every cum_step
        self.states_fw = torch.zeros(self.n_lstm * 2)
        self.states_bw = torch.zeros(self.n_lstm * 2)


class FPPolicy(LstmPolicy):
    def __init__(self, n_s, n_a, n_n, n_step, n_fc=64, n_lstm=64, name=None,
                 na_dim_ls=None, identical=True):
        super(FPPolicy, self).__init__(n_s, n_a, n_n, n_step, n_fc, n_lstm, name,
                         na_dim_ls, identical)

    def _init_net(self):
        if self.identical:
            self.n_x = self.n_s - self.n_n * self.n_a
        else:
            self.n_x = int(self.n_s - sum(self.na_dim_ls))
        self.fc_x_layer = nn.Linear(self.n_x, self.n_fc)
        init_layer(self.fc_x_layer, 'fc')
        n_h = self.n_fc
        if self.n_n:
            self.fc_p_layer = nn.Linear(self.n_n * self.n_a, self.n_fc)
            init_layer(self.fc_p_layer, 'fc')
            n_h += self.n_fc
        self.lstm_layer = nn.LSTMCell(n_h, self.n_lstm)
        init_layer(self.lstm_layer, 'lstm')
        self._init_actor_head(self.n_lstm)
        self._init_critic_head(self.n_lstm)

    def _encode_ob(self, ob):
        ob = ob.to(device)
        if ob.dtype != torch.float32:
            ob = ob.float()
        x = F.relu(self.fc_x_layer(ob[:, :self.n_x]))
        if self.n_n:
            p = F.relu(self.fc_p_layer(ob[:, self.n_x:]))
            x = torch.cat([x, p], dim=1)
        return x


class NCMultiAgentPolicy(Policy):
    """ Inplemented as a centralized meta-DNN. To simplify the implementation, all input
    and output dimensions are identical among all agents, and invalid values are casted as
    zeros during runtime."""
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        super(NCMultiAgentPolicy, self).__init__(n_a, n_s, n_step, 'nc', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        # self.neighbor_mask = neighbor_mask.to(device) ##
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()

    def backward(self, obs, fps, acts, dones, Rs, Advs,
                 e_coef, v_coef, summary_writer=None, global_step=None):
        # obs = torch.from_numpy(obs).float().transpose(0, 1).to(device) ##
        # dones = torch.from_numpy(dones).float().to(device) ##
        # fps = torch.from_numpy(fps).float().transpose(0, 1).to(device) ##
        # acts = torch.from_numpy(acts).long().to(device) ##
        obs = obs.float().transpose(0, 1).to(device) ##(n_steps, n_agents, obs_dim)转为 (n_agents, n_steps, *)？
        dones = dones.float().to(device) ##
        fps = fps.float().transpose(0, 1).to(device) ##策略指纹，用于差异化策略，形状为 (n_steps, n_agents, fps_dim)？
        acts = acts.long().to(device) ##
        
        # 通信层的前向传播：计算通信层输出 hs 和新隐藏状态 new_states
        hs, new_states = self._run_comm_layers(obs, dones, fps, self.states_bw)
        # backward grad is limited to the minibatch，限制反向传播到当前批次
        self.states_bw = new_states.detach().to(device) ##

        # 策略和价值网络的前向传播：基于通信层输出计算策略的 logits（未归一化概率分布），输出 ps 是每个智能体的动作概率分布
        ps = self._run_actor_heads(hs)
        # 基于通信层输出计算价值函数 𝑉(𝑠)，即对状态的预测回报
        # vs = self._run_critic_heads(hs, acts)

        ################## 修改：提取每个时间步的平均hidden state和action
        # 或者只用第一个时间步
        acts_first = acts[0:1, :].transpose(0, 1)  # [n_agents, 1]
        vs = self._run_critic_heads(hs[:, 0:1, :], acts_first)

        self.policy_loss = 0
        self.value_loss = 0
        self.entropy_loss = 0
        # Rs = torch.from_numpy(Rs).float().to(device) ##
        # Advs = torch.from_numpy(Advs).float().to(device) ##
        Rs = Rs.float().to(device) ##
        Advs = Advs.float().to(device) ##

        for i in range(self.n_agent):
            actor_dist_i = torch.distributions.categorical.Categorical(logits=ps[i])
            ##grid
            # acts shape: (n_steps, n_agents), Rs/Advs shape: (n_ensemble, n_steps, n_agents)
            # For agent i, we need Rs[i, :, i] and Advs[i, :, i] to get shape [n_steps]
            # Transpose acts to (n_agents, n_steps) for indexing
            policy_loss_i, value_loss_i, entropy_loss_i = \
                self._run_loss(actor_dist_i, e_coef, v_coef, vs[i],
                    acts.transpose(0,1)[i], Rs[i,:], Advs[i,:])     
                    # acts.transpose(0,1)[i], Rs[i,:,i], Advs[i,:,i])
                
                # self._run_loss(actor_dist_i, e_coef, v_coef, vs[i],
                #     acts[i], Rs[i], Advs[i])
            self.policy_loss += policy_loss_i
            self.value_loss += value_loss_i
            self.entropy_loss += entropy_loss_i
        # 总损失是策略损失、价值损失和熵损失的加权和
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()
        if summary_writer is not None:
            self._update_tensorboard(summary_writer, global_step)

    def forward(self, ob, done, fp, action=None, out_type='p'):
        # TxNxm
        ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float().to(device) ##
        done = torch.from_numpy(np.expand_dims(done.cpu().numpy(), axis=0)).float().to(device) ##
        fp = torch.from_numpy(np.expand_dims(fp, axis=0)).float().to(device) ##
        # ob = torch.as_tensor(ob).unsqueeze(0).float().to(device)
        # done = torch.as_tensor(done).unsqueeze(0).float().to(device)
        # fp = torch.as_tensor(fp).unsqueeze(0).float().to(device)

        # h dim: NxTxm
        h, new_states = self._run_comm_layers(ob, done, fp, self.states_fw)
        if out_type.startswith('p'):
            self.states_fw = new_states.detach().to(device) ##
            return self._run_actor_heads(h, detach=True)
        else:
            # action = torch.from_numpy(np.expand_dims(action, axis=1)).long().to(device) ##
            # action = torch.as_tensor(action).unsqueeze(1).long().to(device)
            action = action.unsqueeze(1).long().to(device)
            
            # 处理tensor列表 加入vmas后
            # if isinstance(action, list) and isinstance(action[0], torch.Tensor):
            #     action = torch.stack([act.squeeze() for act in action], dim=0)
            # action = action.unsqueeze(1).float().to(device)  # 改为float

            return self._run_critic_heads(h, action, detach=True)

    def _get_comm_s(self, i, n_n, x, h, p):
        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long().to(device)
        m_i = torch.index_select(h, 0, js).view(1, self.n_h * n_n).to(device)
        p_i = torch.index_select(p, 0, js).to(device)
        nx_i = torch.index_select(x, 0, js).to(device)
        if self.identical:
            p_i = p_i.view(1, self.n_a * n_n)
            nx_i = nx_i.view(1, self.n_s * n_n)
            x_i = x[i].unsqueeze(0).to(device)
        else:
            p_i_ls = []
            nx_i_ls = []
            for j in range(n_n):
                p_i_ls.append(p_i[j].narrow(0, 0, self.na_ls_ls[i][j]))
                nx_i_ls.append(nx_i[j].narrow(0, 0, self.ns_ls_ls[i][j]))
            p_i = torch.cat(p_i_ls).unsqueeze(0).to(device)
            nx_i = torch.cat(nx_i_ls).unsqueeze(0).to(device)
            x_i = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0).to(device)
        s_i = [F.relu(self.fc_x_layers[i](torch.cat([x_i, nx_i], dim=1))),
               F.relu(self.fc_p_layers[i](p_i)),
               F.relu(self.fc_m_layers[i](m_i))]
        return torch.cat(s_i, dim=1)

    def _get_neighbor_dim(self, i_agent):
        # n_n = int(np.sum(self.neighbor_mask[i_agent]))
        n_n = int(self.neighbor_mask[i_agent].sum()) ## PyTorch 的求和操作
        if self.identical:
            return n_n, self.n_s * (n_n+1), self.n_a * n_n, [self.n_s] * n_n, [self.n_a] * n_n
        else:
            ns_ls = []
            na_ls = []
            # for j in np.where(self.neighbor_mask[i_agent])[0]:
            for j in torch.where(self.neighbor_mask[i_agent])[0]: ##
                ns_ls.append(self.n_s_ls[j].to(device))
                na_ls.append(self.n_a_ls[j].to(device))
            # return n_n, self.n_s_ls[i_agent] + sum(ns_ls), sum(na_ls), ns_ls, na_ls
            return n_n, self.n_s_ls[i_agent] + torch.stack(ns_ls).sum(), torch.stack(na_ls).sum(), ns_ls, na_ls

    def _init_actor_head(self, n_a):
        # only discrete control is supported for now
        self.n_h = 64 ##+
        actor_head = nn.Linear(self.n_h, n_a)
        init_layer(actor_head, 'fc')
        self.actor_heads.append(actor_head)

    def _init_comm_layer(self, n_n, n_ns, n_na):
        n_lstm_in = 3 * self.n_fc
        fc_x_layer = nn.Linear(n_ns, self.n_fc)
        init_layer(fc_x_layer, 'fc')
        self.fc_x_layers.append(fc_x_layer)
        if n_n:
            fc_p_layer = nn.Linear(n_na, self.n_fc)
            init_layer(fc_p_layer, 'fc')
            fc_m_layer = nn.Linear(self.n_h * n_n, self.n_fc)
            init_layer(fc_m_layer, 'fc')
            self.fc_m_layers.append(fc_m_layer)
            self.fc_p_layers.append(fc_p_layer)
            lstm_layer = nn.LSTMCell(n_lstm_in, self.n_h)
        else:
            self.fc_m_layers.append(None)
            self.fc_p_layers.append(None)
            lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)

    def _init_critic_head(self, n_na):
        critic_head = nn.Linear(self.n_h + n_na, 1)
        init_layer(critic_head, 'fc')
        self.critic_heads.append(critic_head)

    def _init_net(self):
        self.fc_x_layers = nn.ModuleList()
        self.fc_p_layers = nn.ModuleList()
        self.fc_m_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        self.actor_heads = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        self.ns_ls_ls = []
        self.na_ls_ls = []
        self.n_n_ls = []
        for i in range(self.n_agent):
            n_n, n_ns, n_na, ns_ls, na_ls = self._get_neighbor_dim(i)
            self.ns_ls_ls.append(ns_ls)
            self.na_ls_ls.append(na_ls)
            self.n_n_ls.append(n_n)
            self._init_comm_layer(n_n, n_ns, n_na)
            n_a = self.n_a if self.identical else self.n_a_ls[i]
            self._init_actor_head(n_a)
            self._init_critic_head(n_na)

    def _reset(self):
        self.states_fw = torch.zeros(self.n_agent, self.n_h * 2).to(device)
        self.states_bw = torch.zeros(self.n_agent, self.n_h * 2).to(device)

    def _run_actor_heads(self, hs, detach=False):
        ps = []
        for i in range(self.n_agent):
            if detach:
                # p_i = F.softmax(self.actor_heads[i](hs[i]).to(device), dim=1).squeeze().detach().numpy()
                p_i = F.softmax(self.actor_heads[i](hs[i]).to(device), dim=1).squeeze().detach().cpu().numpy() ##
            else:
                p_i = F.log_softmax(self.actor_heads[i](hs[i]).to(device), dim=1)
            ps.append(p_i)
        return ps

    def _run_comm_layers(self, obs, dones, fps, states):
        obs = batch_to_seq(obs)
        dones = batch_to_seq(dones)
        fps = batch_to_seq(fps)
        h, c = torch.chunk(states.to(device), 2, dim=1)
        outputs = []
        for t, (x, p, done) in enumerate(zip(obs, fps, dones)):
            next_h = []
            next_c = []
            x = x.squeeze(0).to(device)
            p = p.squeeze(0).to(device)
            for i in range(self.n_agent):
                n_n = self.n_n_ls[i]
                if n_n:
                    s_i = self._get_comm_s(i, n_n, x, h, p)
                else:
                    if self.identical:
                        x_i = x[i].unsqueeze(0).to(device)
                    else:
                        x_i = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0).to(device)
                    s_i = F.relu(self.fc_x_layers[i](x_i))
                h_i, c_i = h[i].unsqueeze(0) * (1-done), c[i].unsqueeze(0) * (1-done)
                next_h_i, next_c_i = self.lstm_layers[i](s_i, (h_i, c_i))
                next_h.append(next_h_i) # next_h_i (1, hidden_dim) in vmas
                next_c.append(next_c_i)
            h, c = torch.cat(next_h), torch.cat(next_c)
            outputs.append(h.unsqueeze(0))
        outputs = torch.cat(outputs)
        return outputs.transpose(0, 1), torch.cat([h, c], dim=1)

    
    def _run_critic_heads(self, hs, actions, detach=False):
        vs = []
        for i in range(self.n_agent):
            n_n = self.n_n_ls[i]
            if n_n:
                # js = torch.from_numpy(np.where(self.neighbor_mask[i].cpu().numpy())[0]).long().to(device)
                if isinstance(self.neighbor_mask, torch.Tensor):
                    js = torch.from_numpy(np.where(self.neighbor_mask[i].cpu().numpy())[0]).long()
                else:
                    js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long()
                na_i = torch.index_select(actions, 0, js.to(device))
                na_i_ls = []
                for j in range(n_n):
                    # na_i_ls.append(one_hot(na_i[j], self.na_ls_ls[i][j])).to(device) ##
                    na_i_ls.append(one_hot(na_i[j], self.na_ls_ls[i][j]))
                h_i = torch.cat([hs[i]] + na_i_ls, dim=1)
            else:
                h_i = hs[i].to(device)
            v_i = self.critic_heads[i](h_i).squeeze()
            if detach:
                vs.append(v_i.detach().cpu().numpy()) ##
            else:
                vs.append(v_i)
        return vs

    

class ConsensusPolicy(NCMultiAgentPolicy):
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        Policy.__init__(self, n_a, n_s, n_step, 'cu', None, identical)
        self.n_h = n_h
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = torch.as_tensor(neighbor_mask,device=device)
        self.n_fc = n_fc
        # self.n_h = n_h
        self._init_net()
        self._reset()

    def consensus_update(self):
        consensus_update = []
        with torch.no_grad():
            for i in range(self.n_agent):
                mean_wts = self._get_critic_wts(i)
                for param, wt in zip(self.lstm_layers[i].parameters(), mean_wts):
                    param.copy_(wt)

    def _init_net(self):
        self.fc_x_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        self.actor_heads = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        self.na_ls_ls = []
        self.n_n_ls = []
        for i in range(self.n_agent):
            n_n, _, n_na, _, na_ls = self._get_neighbor_dim(i)
            n_s = self.n_s if self.identical else self.n_s_ls[i]
            self.na_ls_ls.append(na_ls)
            self.n_n_ls.append(n_n)
            fc_x_layer = nn.Linear(n_s, self.n_fc).to(device)
            init_layer(fc_x_layer, 'fc')
            self.fc_x_layers.append(fc_x_layer)
            lstm_layer = nn.LSTMCell(self.n_fc, self.n_h).to(device)
            init_layer(lstm_layer, 'lstm')
            self.lstm_layers.append(lstm_layer)
            n_a = self.n_a if self.identical else self.n_a_ls[i]
            self._init_actor_head(n_a)
            self._init_critic_head(n_na)

    def _get_critic_wts(self, i_agent):
        wts = []
        for wt in self.lstm_layers[i_agent].parameters():
            wts.append(wt.detach().to(device))
        neighbors = list(np.where(self.neighbor_mask[i_agent].cpu().numpy() == 1)[0]) ##
        for j in neighbors:
            for k, wt in enumerate(self.lstm_layers[j].parameters()):
                wts[k] += wt.detach().to(device)
        n = 1 + len(neighbors)
        for k in range(len(wts)):
            wts[k] /= n
        return wts

    def _run_comm_layers(self, obs, dones, fps, states):
        # NxTxm
        obs = obs.transpose(0, 1).to(device)
        hs = []
        new_states = []
        for i in range(self.n_agent):
            xs_i = F.relu(self.fc_x_layers[i](obs[i])).to(device)
            hs_i, new_states_i = run_rnn(self.lstm_layers[i], xs_i, dones, states[i])
            hs.append(hs_i.unsqueeze(0))
            new_states.append(new_states_i.unsqueeze(0))
        return torch.cat(hs), torch.cat(new_states)


class CommNetMultiAgentPolicy(NCMultiAgentPolicy):
    """Reference code: https://github.com/IC3Net/IC3Net/blob/master/comm.py.
       Note in CommNet, the message is generated from hidden state only, so current state
       and neigbor policies are not included in the inputs."""
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        Policy.__init__(self, n_a, n_s, n_step, 'cnet', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()

    def _init_comm_layer(self, n_n, n_ns, n_na):
        fc_x_layer = nn.Linear(n_ns, self.n_fc)
        init_layer(fc_x_layer, 'fc')
        self.fc_x_layers.append(fc_x_layer)
        if n_n:
            fc_m_layer = nn.Linear(self.n_h, self.n_fc)
            init_layer(fc_m_layer, 'fc')
            self.fc_m_layers.append(fc_m_layer)
        else:
            self.fc_m_layers.append(None)
        lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)

    def _get_comm_s(self, i, n_n, x, h, p):
        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long()
        js = js.to(device) 
        m_i = torch.index_select(h, 0, js).mean(dim=0, keepdim=True)
        nx_i = torch.index_select(x, 0, js)
        if self.identical:
            nx_i = nx_i.view(1, self.n_s * n_n)
            x_i = x[i].unsqueeze(0)
        else:
            nx_i_ls = []
            for j in range(n_n):
                nx_i_ls.append(nx_i[j].narrow(0, 0, self.ns_ls_ls[i][j]))
            nx_i = torch.cat(nx_i_ls).unsqueeze(0)
            x_i = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)
        return F.relu(self.fc_x_layers[i](torch.cat([x_i, nx_i], dim=1))) + \
               self.fc_m_layers[i](m_i)


class DIALMultiAgentPolicy(NCMultiAgentPolicy):
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        Policy.__init__(self, n_a, n_s, n_step, 'dial', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()

    def _init_comm_layer(self, n_n, n_ns, n_na):
        fc_x_layer = nn.Linear(n_ns, self.n_fc)
        init_layer(fc_x_layer, 'fc')
        self.fc_x_layers.append(fc_x_layer)
        if n_n:
            fc_m_layer = nn.Linear(self.n_h*n_n, self.n_fc)
            init_layer(fc_m_layer, 'fc')
            self.fc_m_layers.append(fc_m_layer)
        else:
            self.fc_m_layers.append(None)
        lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)

    def _get_comm_s(self, i, n_n, x, h, p):
        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long().to(device)
        m_i = torch.index_select(h, 0, js).view(1, self.n_h * n_n)
        nx_i = torch.index_select(x, 0, js)
        if self.identical:
            nx_i = nx_i.view(1, self.n_s * n_n)
        else:
            nx_i_ls = []
            for j in range(n_n):
                nx_i_ls.append(nx_i[j].narrow(0, 0, self.ns_ls_ls[i][j]))
            nx_i = torch.cat(nx_i_ls).unsqueeze(0)
        a_i = one_hot(p[i].argmax().unsqueeze(0), self.n_fc)
        return F.relu(self.fc_x_layers[i](torch.cat([x[i].unsqueeze(0), nx_i], dim=1))) + \
               F.relu(self.fc_m_layers[i](m_i)) + a_i







class DistributionalConsensusPolicy(NCMultiAgentPolicy):
    """
    分布式共识策略，Critic输出分布参数
    """
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        Policy.__init__(self, n_a, n_s, n_step, 'distributional_cu', None, identical)
        
        self.n_h = n_h
        self.n_agent = n_agent
        self.neighbor_mask = torch.as_tensor(neighbor_mask, device=device)
        self.n_fc = n_fc
        self.gamma = 0.99  # 折扣因子，可以从model_config获取

        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls   

        self._init_net()
        self._reset()


    def _init_net(self):
        """初始化网络结构"""
        self.fc_x_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        self.actor_heads = nn.ModuleList()
        self.critic_mu_heads = nn.ModuleList()      # Critic均值头
        self.critic_sigma_heads = nn.ModuleList()   # Critic标准差头
        
        self.na_ls_ls = []
        self.n_n_ls = []
        
        for i in range(self.n_agent):
            n_n, _, n_na, _, na_ls = self._get_neighbor_dim(i)

            n_s = self.n_s if self.identical else self.n_s_ls[i]
            n_a = self.n_a if self.identical else self.n_a_ls[i]
            # n_s = self.n_s_ls[i]  # 使用具体智能体的观测维度
            # n_a = self.n_a_ls[i]  # 使用具体智能体的动作维度

            # 智能体 {i}: 观测维度={n_s}, 动作维度={n_a}, 邻居数量={n_n}

            self.na_ls_ls.append(na_ls)
            self.n_n_ls.append(n_n)
            
            # 特征提取层
            fc_x_layer = nn.Linear(n_s, self.n_fc).to(device)
            self._init_layer(fc_x_layer, 'fc')
            self.fc_x_layers.append(fc_x_layer)
            
            # LSTM层
            lstm_layer = nn.LSTMCell(self.n_fc, self.n_h).to(device)
            self._init_layer(lstm_layer, 'lstm')
            self.lstm_layers.append(lstm_layer)
            
            # Actor头
            # n_a = self.n_a if self.identical else self.n_a_ls[i]
            # self._init_actor_head(n_a)
            # Actor头 - 使用正确的动作维度
            actor_head = nn.Linear(self.n_h, n_a).to(device)
            nn.init.xavier_uniform_(actor_head.weight, gain=0.1)
            nn.init.constant_(actor_head.bias, 0)
            self.actor_heads.append(actor_head)
            
            # 分布式Critic头
            self._init_distributional_critic_head(n_na)

    def _init_layer(self, layer, layer_type):
        """初始化层参数"""
        if layer_type == 'fc':
            nn.init.xavier_uniform_(layer.weight, gain=0.5)
            nn.init.constant_(layer.bias, 0)
        elif layer_type == 'lstm':
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param, gain=0.5)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

    def _init_actor_head(self, n_a):
        """初始化Actor头"""
        actor_head = nn.Linear(self.n_h, n_a).to(device)
        nn.init.xavier_uniform_(actor_head.weight, gain=0.1)
        nn.init.constant_(actor_head.bias, 0)
        self.actor_heads.append(actor_head)

    def _init_distributional_critic_head(self, n_na):
        """初始化分布式Critic头"""
        # 均值头
        critic_mu_head = nn.Linear(self.n_h + n_na, 1).to(device)
        nn.init.xavier_uniform_(critic_mu_head.weight, gain=0.5)
        nn.init.constant_(critic_mu_head.bias, 0)
        self.critic_mu_heads.append(critic_mu_head)
        
        # 标准差头（输出log_sigma）
        critic_sigma_head = nn.Linear(self.n_h + n_na, 1).to(device)
        nn.init.xavier_uniform_(critic_sigma_head.weight, gain=0.1)
        nn.init.constant_(critic_sigma_head.bias, -1.0)  # 初始化为小的标准差
        self.critic_sigma_heads.append(critic_sigma_head)

    def _get_critic_wts(self, i_agent):
        """获取邻居的Critic权重用于共识更新"""
        wts = []
        
        # 收集当前agent的LSTM权重
        for wt in self.lstm_layers[i_agent].parameters():
            wts.append(wt.detach().clone())
            
        # 收集邻居的权重
        neighbors = list(np.where(self.neighbor_mask[i_agent].cpu().numpy() == 1)[0])
        for j in neighbors:
            for k, wt in enumerate(self.lstm_layers[j].parameters()):
                wts[k] += wt.detach().clone()
                
        # 平均化
        n = 1 + len(neighbors)
        for k in range(len(wts)):
            wts[k] /= n
            
        return wts

    def consensus_update(self):
        """共识更新，平均化邻居的网络参数"""
        with torch.no_grad():
            for i in range(self.n_agent):
                mean_wts = self._get_critic_wts(i)
                for param, wt in zip(self.lstm_layers[i].parameters(), mean_wts):
                    param.copy_(wt)

    #######只对一个智能体，没有循环
    def _run_single_agent_comm(self, agent_id, obs, dones, fps, state):
        """为单个agent运行通信层
        
        Args:
            agent_id: 当前agent的ID
            obs: 当前agent的观测 [batch_size, obs_dim]
            dones: done标志 [batch_size]
            fps: 指纹特征
            state: 当前agent的LSTM状态 (h, c)
        
        Returns:
            h: 隐藏状态 [batch_size, hidden_dim]
            new_state: 新的LSTM状态 (h, c)
        """
        expected_dim = self.fc_x_layers[agent_id].in_features
        
        # 确保obs是tensor
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        
        actual_dim = obs.size(-1)
        
        # 维度检查和修复
        if actual_dim != expected_dim:
            print(f"智能体 {agent_id}: 观测维度不匹配 - 期望 {expected_dim}, 实际 {actual_dim}")
            obs = self._fix_obs_dimension(obs, expected_dim, agent_id)
        
        # 前向传播
        xs = F.relu(self.fc_x_layers[agent_id](obs))
        h, new_state = self._run_rnn(self.lstm_layers[agent_id], xs, dones, state)
        
        # 确保输出维度正确
        if h.dim() == 1:
            h = h.unsqueeze(0)
        
        # 分离状态的梯度
        h_new, c_new = new_state
        if h_new.dim() == 1:
            h_new = h_new.unsqueeze(0)
        if c_new.dim() == 1:
            c_new = c_new.unsqueeze(0)
        
        return h, (h_new.detach(), c_new.detach())



    #################
    def _run_comm_layers(self, obs, dones, fps, states, agent_id=None):
        if agent_id is not None:
            # ========== 单智能体模式 ==========
            obs_i = obs
        
            # 确保是2D tensor: [batch_size, obs_dim]
            if isinstance(obs_i, np.ndarray):
                obs_i = torch.as_tensor(obs_i, dtype=torch.float32, device=device)
            if obs_i.dim() == 1:
                obs_i = obs_i.unsqueeze(0)
        
            # 检查和修复维度
            expected_dim = self.fc_x_layers[agent_id].in_features
            actual_dim = obs_i.shape[-1]
        
            if actual_dim != expected_dim:
                print(f"警告: 智能体 {agent_id} 观测维度不匹配 - 期望 {expected_dim}, 实际 {actual_dim}")
                obs_i = self._fix_obs_dimension(obs_i, expected_dim, agent_id)
        
            # 确保dones是正确的形状和类型
            if isinstance(dones, np.ndarray):
                dones = torch.as_tensor(dones, dtype=torch.float32, device=device)
            if dones.dim() == 0:
                dones = dones.unsqueeze(0)
        
            # 编码观察
            xs_i = F.relu(self.fc_x_layers[agent_id](obs_i))
        
            # 通过LSTM
            state_i = states[agent_id] if isinstance(states, list) else states
            hs_i, new_states_i = self._run_rnn(self.lstm_layers[agent_id], xs_i, dones, state_i)
        
            # 确保输出维度正确
            if hs_i.dim() == 1:
                hs_i = hs_i.unsqueeze(0)

            # 处理新状态
            h_new, c_new = new_states_i
            if h_new.dim() == 1:
                h_new = h_new.unsqueeze(0)
            if c_new.dim() == 1:
                c_new = c_new.unsqueeze(0)

            return hs_i, (h_new.detach(), c_new.detach())
    
        else:
            # ========== 多智能体模式 ==========
            hs = []
            new_states = []

            for i in range(self.n_agent):
                # 获取该智能体的观察
                if isinstance(obs, (list, tuple)):
                    obs_i = obs[i]
                else:
                    obs_i = obs  # 假设所有智能体共享观察

                # 确保是2D tensor
                if isinstance(obs_i, np.ndarray):
                    obs_i = torch.as_tensor(obs_i, dtype=torch.float32, device=device)
                if obs_i.dim() == 1:
                    obs_i = obs_i.unsqueeze(0)
            
                # 检查和修复维度
                expected_dim = self.fc_x_layers[i].in_features
                actual_dim = obs_i.shape[-1]
            
                if actual_dim != expected_dim:
                    print(f"警告: 智能体 {i} 观测维度不匹配 - 期望 {expected_dim}, 实际 {actual_dim}")
                    obs_i = self._fix_obs_dimension(obs_i, expected_dim, i)

                # 确保dones是正确的形状和类型
                dones_tensor = dones
                if isinstance(dones_tensor, np.ndarray):
                    dones_tensor = torch.as_tensor(dones_tensor, dtype=torch.float32, device=device)
                if dones_tensor.dim() == 0:
                    dones_tensor = dones_tensor.unsqueeze(0)

                obs_i = torch.as_tensor(obs_i, dtype=torch.float32, device=device)
                # 编码观察
                xs_i = F.relu(self.fc_x_layers[i](obs_i))

                # 通过LSTM
                state_i = states[i] if isinstance(states, list) else states
                hs_i, new_states_i = self._run_rnn(self.lstm_layers[i], xs_i, dones_tensor, state_i)
            
                # 确保输出维度正确
                if hs_i.dim() == 1:
                    hs_i = hs_i.unsqueeze(0)
                hs.append(hs_i)
            
                # 处理新状态
                h_new, c_new = new_states_i
                if h_new.dim() == 1:
                    h_new = h_new.unsqueeze(0)
                if c_new.dim() == 1:
                    c_new = c_new.unsqueeze(0)
                new_states.append((h_new.detach(), c_new.detach()))
        
            # 拼接所有智能体的hidden states
            hs_cat = torch.cat(hs, dim=0)
        
            return hs_cat, new_states


    # def _run_comm_layers(self, obs, dones, fps, states):
        
    #     hs = []
    #     new_states = []
        
    #     for i in range(self.n_agent):
    #         obs_i = obs[i]
    #         expected_dim = self.fc_x_layers[i].in_features
    #         actual_dim = obs_i.cpu().numpy().size
            
    #         # 维度检查和修复
    #         if actual_dim != expected_dim:
    #             print(f"智能体 {i}: 观测维度不匹配 - 期望 {expected_dim}, 实际 {actual_dim}")
    #             obs_i = self._fix_obs_dimension(obs_i, expected_dim, i)
                
    #         try:
    #             obs_i = torch.as_tensor(obs_i, dtype=torch.float32, device=device)
    #             xs_i = F.relu(self.fc_x_layers[i](obs_i))
    #             hs_i, new_states_i = self._run_rnn(self.lstm_layers[i], xs_i, dones, states[i])
                
    #             # 确保输出维度正确
    #             if hs_i.dim() == 1:
    #                 hs_i = hs_i.unsqueeze(0)
    #             hs.append(hs_i)
                
    #             # 处理新状态
    #             h_new, c_new = new_states_i
    #             if h_new.dim() == 1:
    #                 h_new = h_new.unsqueeze(0)
    #             if c_new.dim() == 1:
    #                 c_new = c_new.unsqueeze(0)
    #             new_states.append((h_new, c_new))
                
    #         except RuntimeError as e:
    #             print(f"智能体 {i} 处理错误: {e}")
    #             print(f"obs_i 形状: {obs_i.shape}")
    #             raise
            
    #     detached_states = []
    #     for (h_new, c_new) in new_states:
    #         detached_states.append((h_new.detach(), c_new.detach()))
    #     return torch.cat(hs, dim=0), detached_states
        # 这里继续原来的逻辑，例如 LSTM、通信更新等


    # def _run_comm_layers(self, obs, dones, fps, states):
    #     """运行通信层，处理LSTM"""
    #      # 改进的观测处理逻辑
    #     if isinstance(obs, list):
    #         processed_obs = self._process_obs_list(obs)
    #     else:
    #         processed_obs = self._process_obs_tensor(obs)
        
    #     # 处理dones
    #     # dones = self._process_dones(dones)
        
    #     hs = []
    #     new_states = []
        
    #     for i in range(self.n_agent):
    #         obs_i = processed_obs[i]
    #         expected_dim = self.fc_x_layers[i].in_features
    #         actual_dim = obs_i.size(-1)
            
    #         # 维度检查和修复
    #         if actual_dim != expected_dim:
    #             print(f"智能体 {i}: 观测维度不匹配 - 期望 {expected_dim}, 实际 {actual_dim}")
    #             obs_i = self._fix_obs_dimension(obs_i, expected_dim, i)
                
    #         try:
    #             xs_i = F.relu(self.fc_x_layers[i](obs_i))
    #             hs_i, new_states_i = self._run_rnn(self.lstm_layers[i], xs_i, dones, states[i])
                
    #             # 确保输出维度正确
    #             if hs_i.dim() == 1:
    #                 hs_i = hs_i.unsqueeze(0)
    #             hs.append(hs_i)
                
    #             # 处理新状态
    #             h_new, c_new = new_states_i
    #             if h_new.dim() == 1:
    #                 h_new = h_new.unsqueeze(0)
    #             if c_new.dim() == 1:
    #                 c_new = c_new.unsqueeze(0)
    #             new_states.append((h_new, c_new))
                
    #         except RuntimeError as e:
    #             print(f"智能体 {i} 处理错误: {e}")
    #             print(f"obs_i 形状: {obs_i.shape}")
    #             raise
            
    #     detached_states = []
    #     for (h_new, c_new) in new_states:
    #         detached_states.append((h_new.detach(), c_new.detach()))
    #     return torch.cat(hs, dim=0), detached_states
    
    def _process_obs_list(self, obs):
        """处理观测，添加邻居信息"""
        processed = []
    
        for i in range(self.n_agent):
            # 自身观测
            agent_obs = [obs[i]]
        
            # 添加邻居观测
            for j in range(self.n_agent):
                if self.neighbor_mask[i][j] == 1:
                    agent_obs.append(obs[j])
        
            # 合并并转换为tensor
            full_obs = np.concatenate(agent_obs)
            full_obs_tensor = torch.tensor(full_obs, dtype=torch.float32, device=device)
            processed.append(full_obs_tensor)
    
        return processed


    def _process_obs_tensor(self, obs):
        """处理观测张量,添加邻居信息"""    
        # 转换为tensor
        if isinstance(obs, torch.Tensor):
            obs_tensor = obs.to(device)
        else:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
    
        # 分析输入维度
        # print(f"DEBUG: 输入obs张量形状: {obs_tensor.shape}")
    
        # 处理不同的输入格式
        if obs_tensor.dim() == 2:
            batch_size, last_dim = obs_tensor.shape
        
            # 情况1: [batch_size, n_agent * 5] - 所有智能体基础观测拼接
            expected_base_dim = self.n_agent * 5
            if last_dim == expected_base_dim:
                print(f"DEBUG: 检测到格式1 - 基础观测拼接 [{batch_size}, {last_dim}]")
                base_obs_list = []
                for i in range(self.n_agent):
                    start_idx = i * 5
                    end_idx = (i + 1) * 5
                    base_obs_i = obs_tensor[:, start_idx:end_idx]  # [batch, 5]
                    base_obs_list.append(base_obs_i)
        
            # 情况2: [batch_size, sum(n_s_ls)] - 已包含邻居信息
            elif last_dim == sum(self.n_s_ls):
                print(f"DEBUG: 检测到格式2 - 完整观测拼接 [{batch_size}, {last_dim}]")
                processed_obs = []
                start_idx = 0

                for i in range(self.n_agent):
                    expected_dim = self.n_s_ls[i]
                    end_idx = start_idx + expected_dim
                    obs_i = obs_tensor[:, start_idx:end_idx]
                    processed_obs.append(obs_i)
                    start_idx = end_idx
            
                return processed_obs
        
            # 情况3: [batch_size, 5] - 单一观测需要分配给所有智能体
            elif last_dim == 5:
                # print(f"DEBUG: 检测到格式3 - 单一观测分配 [batch_size={batch_size}, obs_dim=5]")
                # print("INFO: 将5维观测复制给所有智能体作为基础观测")
            
                # 为每个智能体复制相同的基础观测
                base_obs_list = []
                for i in range(self.n_agent):
                    base_obs_list.append(obs_tensor.clone())  # [batch_size, 5]
        
            # 情况4: 其他维度
            else:
                print(f"DEBUG: 尝试均匀分割观测维度 [{batch_size}, {last_dim}]")
                if last_dim % self.n_agent == 0:
                    base_dim_per_agent = last_dim // self.n_agent
                    base_obs_list = []
                    for i in range(self.n_agent):
                        start_idx = i * base_dim_per_agent
                        end_idx = (i + 1) * base_dim_per_agent
                        base_obs_i = obs_tensor[:, start_idx:end_idx]
                        base_obs_list.append(base_obs_i)
                else:
                    raise ValueError(f"无法解析观测张量维度: {obs_tensor.shape}, "
                                   f"期望基础维度: {expected_base_dim} 或完整维度: {sum(self.n_s_ls)}")
    
        elif obs_tensor.dim() == 1:
            # print(f"DEBUG: 1维张量，添加batch维度")
            obs_tensor = obs_tensor.unsqueeze(0)
            return self._process_obs_tensor(obs_tensor)  # 递归处理
    
        else:
            raise ValueError(f"不支持的张量维度: {obs_tensor.shape}")
    
        # 添加邻居信息
        # print(f"DEBUG: 开始添加邻居信息，基础观测列表长度: {len(base_obs_list)}")
        processed_obs = []
    
        for i in range(self.n_agent):
            # 自身观测
            agent_obs_list = [base_obs_list[i]]
            neighbor_count = 0
        
            # 添加邻居观测
            for j in range(self.n_agent):
                if self.neighbor_mask[i][j] == 1:
                    agent_obs_list.append(base_obs_list[j])
                    neighbor_count += 1
        
            # 拼接所有观测
            full_obs = torch.cat(agent_obs_list, dim=-1)  # [batch, total_dim]
            processed_obs.append(full_obs)
        
            # print(f"DEBUG: 智能体{i} - 邻居数:{neighbor_count}, 输出维度:{full_obs.shape[-1]}, 期望维度:{self.n_s_ls[i]}")
    
        return processed_obs

    

    def _fix_obs_dimension(self, obs, expected_dim, agent_id):
        """修复观测维度不匹配问题"""
        actual_dim = obs.size(-1)
        
        if actual_dim > expected_dim:
            # 截断多余的维度
            print(f"智能体 {agent_id}: 截断观测从 {actual_dim} 到 {expected_dim}")
            return obs[..., :expected_dim]
        elif actual_dim < expected_dim:
            # 零填充不足的维度
            print(f"智能体 {agent_id}: 零填充观测从 {actual_dim} 到 {expected_dim}")
            padding_size = expected_dim - actual_dim
            padding = torch.zeros(*obs.shape[:-1], padding_size, device=device)
            return torch.cat([obs, padding], dim=-1)
        else:
            return obs

    # def _run_rnn(self, rnn_layer, xs, dones, states):
    #     """运行RNN层"""
    #     outputs = []
    #     h, c = states
        
    #     for i, x in enumerate(xs):
    #         if dones[i]:
    #             h = torch.zeros_like(h)
    #             c = torch.zeros_like(c)
    #         h, c = rnn_layer(x, (h, c))
    #         outputs.append(h)
            
    #     return torch.stack(outputs), (h, c)

    def _run_rnn(self, rnn_layer, xs, dones, states):
        """运行RNN层"""
        outputs = []
        h, c = states
        
        # 确保输入是正确的格式
        if xs.dim() == 1:
            xs = xs.unsqueeze(0)  # 添加batch维度
            
        # 处理dones格式
        if isinstance(dones, torch.Tensor):
            if dones.dim() == 0:
                dones = dones.unsqueeze(0)
            dones_list = dones.cpu().numpy() if dones.numel() > 1 else [dones.item()]
        else:
            dones_list = [dones] if not isinstance(dones, (list, np.ndarray)) else dones
        
        for i, x in enumerate(xs):
            # 检查是否需要重置隐状态
            done_flag = dones_list[min(i, len(dones_list)-1)]
            if done_flag:
                h = torch.zeros_like(h)
                c = torch.zeros_like(c)
            
            h, c = rnn_layer(x.unsqueeze(0) if x.dim() == 1 else x, (h, c))
            outputs.append(h)
            
        return torch.stack(outputs) if len(outputs) > 1 else outputs[0], (h, c)
    

    # def forward(self, obs, done, nactions=None, out_type='p', agent_id=None):
    #     """前向传播
        
    #     Args:
    #         obs: 观测数据
    #         done: done标志
    #         nactions: 可用动作
    #         out_type: 输出类型 'p'(策略) 或 'v'(价值)
    #         agent_id: 指定agent ID。如果为None，返回所有agents
    #     """
    #     if nactions is None:
    #         nactions = [None] * self.n_agent
        
    #     if agent_id is not None:
    #         # 单agent模式
    #         if out_type == 'p':
    #             return self._compute_policy(obs, done, nactions, agent_id)
    #         elif out_typel == 'v':
    #             return self._compute_value_distribution(obs, done, nactions, agent_id)
    #     else:
    #         # 多agent模式（兼容旧代码）
    #         if out_type == 'p':
    #             return self._compute_all_policies(obs, done, nactions)
    #         elif out_type == 'v':
    #             return self._compute_all_values(obs, done, nactions)


    def forward(self, obs, done, nactions=None, out_type='p'):
        """前向传播"""
        if nactions is None:
            nactions = [None] * self.n_agent
            
        if out_type == 'p':  # 策略输出
            return self._compute_policy(obs, done, nactions,agent_id=None)
        elif out_type == 'v':  # 价值输出（分布参数）
            return self._compute_value_distribution(obs, done, nactions, agent_id=None)
        else:
            raise ValueError(f"Unknown output type: {out_type}")





    # def _compute_policy(self, obs, done, nactions):
    #     """计算策略输出"""
    #     fps = self._get_fps(obs, nactions)
    #     hs, self.states = self._run_comm_layers(obs, done, fps, self.states)
        
    #     policies = []
    #     for i in range(self.n_agent):
    #         policy_logits = self.actor_heads[i](hs[i])
    #         policy_logits = torch.clamp(policy_logits, min=-5, max=5)
    #         policies.append(F.softmax(policy_logits, dim=-1))
            
    #     return policies





    ############只对一个智能体，没有循环
    # def _compute_policy(self, obs, done, nactions, agent_id):
    #     """计算单个agent的策略输出"""
    #     fps = self._get_fps(obs, nactions)
        
    #     # 只处理当前agent
    #     h, new_state = self._run_single_agent_comm(agent_id, obs, done, fps, self.states[agent_id])
        
    #     # 更新状态
    #     self.states[agent_id] = new_state
        
    #     # 只返回当前agent的policy
    #     policy_logits = self.actor_heads[agent_id](h)
    #     policy_logits = torch.clamp(policy_logits, min=-5, max=5)
    #     policy = F.softmax(policy_logits, dim=-1)
        
    #     return policy
    
    ############################
    def _compute_policy(self, obs, done, nactions, agent_id=None):
        # 计算策略输出
    
        # Args:
        #     obs: 观察，单智能体模式下为 [batch_size, obs_dim]，多智能体模式下为列表
        #     done: 完成标志 [batch_size]
        #     nactions: 邻居动作
        #     agent_id: 如果不为None，只计算单个智能体的策略；否则计算所有智能体
    
        # Returns:
        #     单智能体模式: [batch_size, n_actions]
        #     多智能体模式: 列表，每个元素为 [batch_size, n_actions]

        # 获取策略指纹
        fps = self._get_fps(obs, nactions, agent_id)
    
        if agent_id is not None:
            # 单智能体模式
            hs, new_state = self._run_comm_layers(obs, done, fps, self.states, agent_id)

            # 更新该智能体的状态
            if isinstance(self.states, list):
                self.states[agent_id] = new_state
        
            # 计算策略
            policy_logits = self.actor_heads[agent_id](hs)
            policy_logits = torch.clamp(policy_logits, min=-5, max=5)
            policy = F.softmax(policy_logits, dim=-1)
        
            return policy
        else:
            # 多智能体模式
            hs, self.states = self._run_comm_layers(obs, done, fps, self.states, agent_id=None)

            policies = []
            for i in range(self.n_agent):
                policy_logits = self.actor_heads[i](hs[i])
                policy_logits = torch.clamp(policy_logits, min=-5, max=5)
                policies.append(F.softmax(policy_logits, dim=-1))
        
            return policies


    
    # def _compute_policy(self, obs, done, nactions):
    #     """计算策略输出"""
    #     # print(f"_compute_policy called, obs.requires_grad: {obs.requires_grad if hasattr(obs, 'requires_grad') else 'N/A'}")
    #     fps = self._get_fps(obs, nactions)
    #     hs, self.states = self._run_comm_layers(obs, done, fps, self.states)
        
    #     policies = []
    #     for i in range(self.n_agent):
    #         policy_logits = self.actor_heads[i](hs[i])
    #         policy_logits = torch.clamp(policy_logits, min=-5, max=5)
    #         policies.append(F.softmax(policy_logits, dim=-1))
            
    #     return policies


    ###################
    def _compute_value_distribution(self, obs, done, nactions, agent_id=None):
        """计算价值分布参数"""
        # print(f"_compute_value_distribution called, obs.requires_grad: {obs.requires_grad if hasattr(obs, 'requires_grad') else 'N/A'}")
        fps = self._get_fps(obs, nactions, agent_id)

        if agent_id is not None:
            hs, _ = self._run_comm_layers(obs, done, fps, self.states, agent_id)
            i = agent_id
            if fps[agent_id] is not None:
                # h_input = torch.cat([hs[i], fps[i]], dim=-1)
                # 方法1（推荐）：将hs[i]降维成1维，与fps[i]保持一致
                hs_flat = hs[agent_id].squeeze(0)  # [1, 64] -> [64]
                h_input = torch.cat([hs_flat, fps[i]], dim=-1)  # [64] + [4] -> [68]

                ### 动态调整到网络期望的维度
                # expected_dim = self.critic_mu_heads[i].in_features
                # actual_dim = h_input.numel()           
                # if actual_dim != expected_dim:
                #     if actual_dim < expected_dim:
                #         # 用零填充
                #         padding = torch.zeros(expected_dim - actual_dim, device=h_input.device)
                #         h_input = torch.cat([h_input, padding], dim=-1)
                #     else:
                #         # 截断
                #         h_input = h_input[:expected_dim]

                #     # 如果后续网络需要2维输入，再扩展回去
                #     # h_input = h_input.unsqueeze(0)  # [68] -> [1, 68]
                # else:
                #     h_input = hs[i]

            mu = self.critic_mu_heads[agent_id](h_input)
            log_sigma = self.critic_sigma_heads[agent_id](h_input)
            log_sigma = torch.clamp(log_sigma, min=-10, max=2)
            sigma = torch.exp(log_sigma)
            
            return (mu, sigma)

        else:
            hs, _ = self._run_comm_layers(obs, done, fps, self.states, agent_id=None)
        
            values = []
            for i in range(self.n_agent):
                # 拼接隐藏状态和邻居动作
                if fps[i] is not None:
                    # h_input = torch.cat([hs[i], fps[i]], dim=-1)
                    # 方法1（推荐）：将hs[i]降维成1维，与fps[i]保持一致
                    hs_flat = hs[i].squeeze(0)  # [1, 64] -> [64]
                    h_input = torch.cat([hs_flat, fps[i]], dim=-1)  # [64] + [4] -> [68]

                    ### 动态调整到网络期望的维度
                    expected_dim = self.critic_mu_heads[i].in_features
                    actual_dim = h_input.numel()           
                    if actual_dim != expected_dim:
                        if actual_dim < expected_dim:
                            # 用零填充
                            padding = torch.zeros(expected_dim - actual_dim, device=h_input.device)
                            h_input = torch.cat([h_input, padding], dim=-1)
                        else:
                            # 截断
                            h_input = h_input[:expected_dim]

                    # 如果后续网络需要2维输入，再扩展回去
                    # h_input = h_input.unsqueeze(0)  # [68] -> [1, 68]
                else:
                    h_input = hs[i]
                
                # 计算分布参数
                mu = self.critic_mu_heads[i](h_input)
                log_sigma = self.critic_sigma_heads[i](h_input)
                log_sigma = torch.clamp(log_sigma, min=-10, max=2)
                sigma = torch.exp(log_sigma)

                values.append((mu, sigma))
            
            return values




    # def _compute_value_distribution(self, obs, done, nactions):
    #     """计算价值分布参数"""
    #     # print(f"_compute_value_distribution called, obs.requires_grad: {obs.requires_grad if hasattr(obs, 'requires_grad') else 'N/A'}")
    #     fps = self._get_fps(obs, nactions)
    #     hs, _ = self._run_comm_layers(obs, done, fps, self.states)
        
    #     values = []
    #     for i in range(self.n_agent):
    #         # 拼接隐藏状态和邻居动作
    #         if fps[i] is not None:
    #             # h_input = torch.cat([hs[i], fps[i]], dim=-1)
    #             # 方法1（推荐）：将hs[i]降维成1维，与fps[i]保持一致
    #             hs_flat = hs[i].squeeze(0)  # [1, 64] -> [64]
    #             h_input = torch.cat([hs_flat, fps[i]], dim=-1)  # [64] + [4] -> [68]

    #             ### 动态调整到网络期望的维度
    #             expected_dim = self.critic_mu_heads[i].in_features
    #             actual_dim = h_input.numel()           
    #             if actual_dim != expected_dim:
    #                 if actual_dim < expected_dim:
    #                     # 用零填充
    #                     padding = torch.zeros(expected_dim - actual_dim, device=h_input.device)
    #                     h_input = torch.cat([h_input, padding], dim=-1)
    #                 else:
    #                     # 截断
    #                     h_input = h_input[:expected_dim]

    #             # 如果后续网络需要2维输入，再扩展回去
    #             # h_input = h_input.unsqueeze(0)  # [68] -> [1, 68]
    #         else:
    #             h_input = hs[i]
                
    #         # 计算分布参数
    #         mu = self.critic_mu_heads[i](h_input)
    #         log_sigma = self.critic_sigma_heads[i](h_input)
    #         log_sigma = torch.clamp(log_sigma, min=-10, max=2)
    #         sigma = torch.exp(log_sigma)
            
    #         values.append((mu, sigma))
            
    #     return values
    



    # def compute_distributional_loss(self, agent_id, obs, nas, acts, dones, Rs, Advs, e_coef, v_coef):
    #     """计算单个agent的分布式损失"""
        
    #     # 转换为tensor
    #     obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
    #     nas_tensor = torch.as_tensor(nas, dtype=torch.float32, device=device)
    #     dones_tensor = torch.as_tensor(dones, dtype=torch.bool, device=device)
    #     acts_tensor = torch.as_tensor(acts, dtype=torch.long, device=device)
    #     Rs_tensor = torch.as_tensor(Rs, dtype=torch.float32, device=device)
    #     Advs_tensor = torch.as_tensor(Advs, dtype=torch.float32, device=device)
        
    #     ################ 前向传播 - 传入agent_id
    #     policy = self._compute_policy(obs_tensor, dones_tensor, nas_tensor, agent_id)
    #     # policy = self._compute_policy(obs_tensor, dones_tensor, nas_tensor)
    #     mu_pred, sigma_pred = self._compute_value_distribution(obs_tensor, dones_tensor, nas_tensor, agent_id)
        
    #     # 确保数值稳定性
    #     sigma_pred = torch.clamp(sigma_pred, min=1e-6, max=10.0)
        
    #     # Actor损失
    #     if acts_tensor.dim() == 1:
    #         acts_tensor = acts_tensor.unsqueeze(-1)
        
    #     # 确保维度匹配
    #     if policy.size(0) != acts_tensor.size(0):
    #         policy = policy.expand(acts_tensor.size(0), -1)
        
    #     log_probs = torch.log(policy.gather(-1, acts_tensor) + 1e-8).squeeze(-1)
    #     actor_loss = -(log_probs * Advs_tensor.detach()).mean()
        
    #     # 熵损失
    #     entropy = -(policy * torch.log(policy + 1e-8)).sum(dim=-1)
    #     entropy_loss = -e_coef * entropy.mean()
        
    #     # Critic损失 - 使用MSE
    #     value_pred = mu_pred.squeeze(-1)
    #     critic_loss = v_coef * F.mse_loss(value_pred, Rs_tensor.detach())
        
    #     return actor_loss, critic_loss, entropy_loss



    ##############
    def compute_distributional_loss(self, agent_id, obs, nas, acts, dones, Rs, Advs, e_coef, v_coef):
        # 确保输入是numpy数组
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if isinstance(nas, torch.Tensor):
            nas = nas.cpu().numpy()
        if isinstance(acts, torch.Tensor):
            acts = acts.cpu().numpy()
        if isinstance(dones, torch.Tensor):
            dones = dones.cpu().numpy()
        if isinstance(Rs, torch.Tensor):
            Rs = Rs.cpu().numpy()
        if isinstance(Advs, torch.Tensor):
            Advs = Advs.cpu().numpy()
    
        # 转换为tensor
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        nas_tensor = torch.as_tensor(nas, dtype=torch.float32, device=device)
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=device)  # 使用float而非bool
        acts_tensor = torch.as_tensor(acts, dtype=torch.long, device=device)
        Rs_tensor = torch.as_tensor(Rs, dtype=torch.float32, device=device)
        
        # Advs可能是list of tensors/scalars，需要正确处理维度
        if isinstance(Advs, list):
            # 先检查列表中的元素类型和维度
            if len(Advs) > 0:
                first_elem = Advs[0]
                if isinstance(first_elem, (torch.Tensor, np.ndarray)):
                    # 如果元素是tensor/array且是标量(0维)，直接stack
                    if (isinstance(first_elem, torch.Tensor) and first_elem.dim() == 0) or \
                       (isinstance(first_elem, np.ndarray) and first_elem.ndim == 0):
                        Advs_tensor = torch.tensor([float(a) for a in Advs], dtype=torch.float32, device=device)
                    else:
                        # 元素是多维的，需要flatten
                        Advs_flat = []
                        for adv in Advs:
                            if isinstance(adv, torch.Tensor):
                                Advs_flat.extend(adv.flatten().tolist())
                            elif isinstance(adv, np.ndarray):
                                Advs_flat.extend(adv.flatten().tolist())
                            else:
                                Advs_flat.append(float(adv))
                        Advs_tensor = torch.tensor(Advs_flat, dtype=torch.float32, device=device)
                else:
                    # 列表中是普通数值
                    Advs_tensor = torch.tensor(Advs, dtype=torch.float32, device=device)
            else:
                Advs_tensor = torch.tensor([], dtype=torch.float32, device=device)
        else:
            Advs_tensor = torch.as_tensor(Advs, dtype=torch.float32, device=device)
            if Advs_tensor.dim() > 1:
                Advs_tensor = Advs_tensor.flatten()
    
        # 确保维度正确
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        if dones_tensor.dim() == 0:
            dones_tensor = dones_tensor.unsqueeze(0)
        if acts_tensor.dim() == 0:
            acts_tensor = acts_tensor.unsqueeze(0)
    
        # 前向传播 - 只计算单个智能体的策略和价值
        policy = self._compute_policy(obs_tensor, dones_tensor, nas_tensor, agent_id)
        mu_pred, sigma_pred = self._compute_value_distribution(obs_tensor, dones_tensor, nas_tensor, agent_id)
    
        # 确保policy是2D: [batch_size, n_actions]
        if policy.dim() == 3:
            policy = policy.squeeze(1)  # [60, 1, 4] -> [60, 4]
        elif policy.dim() == 1:
            policy = policy.unsqueeze(0)  # [4] -> [1, 4]
    
        # 确保mu_pred和sigma_pred是正确的形状
        # 保留batch维度，只压缩最后一维
        if mu_pred.dim() > 1 and mu_pred.size(-1) == 1:
            mu_pred = mu_pred.squeeze(-1)  # 只压缩最后一维
        if mu_pred.dim() == 0:
            mu_pred = mu_pred.unsqueeze(0)  # 标量->1维
            
        if sigma_pred.dim() > 1 and sigma_pred.size(-1) == 1:
            sigma_pred = sigma_pred.squeeze(-1)
        if sigma_pred.dim() == 0:
            sigma_pred = sigma_pred.unsqueeze(0)
    
        # 确保数值稳定性
        sigma_pred = torch.clamp(sigma_pred, min=1e-6, max=10.0)
    
        # Actor损失
        if acts_tensor.dim() == 1:
            acts_tensor = acts_tensor.unsqueeze(-1)  # [60] -> [60, 1]
    
        # 确保policy和acts的batch维度匹配
        if policy.size(0) != acts_tensor.size(0):
            if policy.size(0) == 1 and acts_tensor.size(0) > 1:
                policy = policy.expand(acts_tensor.size(0), -1)
            elif acts_tensor.size(0) == 1 and policy.size(0) > 1:
                acts_tensor = acts_tensor.expand(policy.size(0), -1)
    
        # 计算log概率
        # policy: [60, 4], acts_tensor: [60, 1] -> gather -> [60, 1] -> squeeze -> [60]
        gathered = policy.gather(-1, acts_tensor)  # [batch, 1]
        if gathered.dim() > 1 and gathered.size(-1) == 1:
            log_probs = torch.log(gathered.squeeze(-1) + 1e-8)
        else:
            log_probs = torch.log(gathered + 1e-8)
        
        # 确保log_probs至少是1维
        if log_probs.dim() == 0:
            log_probs = log_probs.unsqueeze(0)
    
        # 确保Advs维度匹配
        if Advs_tensor.dim() == 0:
            Advs_tensor = Advs_tensor.unsqueeze(0)
        if log_probs.dim() > 0 and Advs_tensor.dim() > 0:
            if Advs_tensor.size(0) != log_probs.size(0):
                if Advs_tensor.size(0) == 1:
                    Advs_tensor = Advs_tensor.expand(log_probs.size(0))
    
        actor_loss = -(log_probs * Advs_tensor.detach()).mean()
    
        # 熵损失
        entropy = -(policy * torch.log(policy + 1e-8)).sum(dim=-1)
        entropy_loss = -e_coef * entropy.mean()
    
        # Critic损失 - 使用MSE
        # 安全的squeeze：只在有维度时才squeeze
        if mu_pred.dim() > 1 and mu_pred.size(-1) == 1:
            value_pred = mu_pred.squeeze(-1)
        else:
            value_pred = mu_pred
        
        # 确保value_pred至少是1维
        if value_pred.dim() == 0:
            value_pred = value_pred.unsqueeze(0)
    
        # 确保Rs维度匹配
        if Rs_tensor.dim() == 0:
            Rs_tensor = Rs_tensor.unsqueeze(0)
        if value_pred.dim() > 0 and Rs_tensor.dim() > 0:
            if Rs_tensor.size(0) != value_pred.size(0):
                if Rs_tensor.size(0) == 1:
                    Rs_tensor = Rs_tensor.expand(value_pred.size(0))
    
        critic_loss = v_coef * F.mse_loss(value_pred, Rs_tensor.detach())
    
        return actor_loss, critic_loss, entropy_loss


    def _compute_policy(self, obs, done, nactions, agent_id=None):
        """
        计算策略输出

        Args:
            obs: 观察，单智能体模式下为 [batch_size, obs_dim]，多智能体模式下为列表
            done: 完成标志 [batch_size]
            nactions: 邻居动作
            agent_id: 如果不为None，只计算单个智能体的策略；否则计算所有智能体

        Returns:
            单智能体模式: [batch_size, n_actions]
            多智能体模式: 列表，每个元素为 [batch_size, n_actions]
        """
        # 获取策略指纹
        fps = self._get_fps(obs, nactions, agent_id)
    
        if agent_id is not None:
            # 单智能体模式
            hs, new_state = self._run_comm_layers(obs, done, fps, self.states, agent_id)
        
            # 更新该智能体的状态
            if isinstance(self.states, list):
                self.states[agent_id] = new_state
        
            # 确保hs是2D: [batch_size, hidden_dim]
            if hs.dim() == 1:
                hs = hs.unsqueeze(0)
        
            # 计算策略
            policy_logits = self.actor_heads[agent_id](hs)
            policy_logits = torch.clamp(policy_logits, min=-5, max=5)
            policy = F.softmax(policy_logits, dim=-1)
        
            # 确保输出是2D: [batch_size, n_actions]
            if policy.dim() == 1:
                policy = policy.unsqueeze(0)

            return policy
        else:
            # 多智能体模式
            hs, self.states = self._run_comm_layers(obs, done, fps, self.states, agent_id=None)
        
            policies = []
            for i in range(self.n_agent):
                policy_logits = self.actor_heads[i](hs[i])
                policy_logits = torch.clamp(policy_logits, min=-5, max=5)
                policies.append(F.softmax(policy_logits, dim=-1))
        
            return policies


    def backward(self, obs, nas, acts, dones, Rs, Advs, e_coef, v_coef, 
                 summary_writer=None, global_step=None):
        """反向传播（兼容原有接口）"""
        total_actor_loss = 0
        total_critic_loss = 0  
        total_entropy_loss = 0
        
        for i in range(self.n_agent):
            actor_loss, critic_loss, entropy_loss = self.compute_distributional_loss(
                i, obs, nas, acts, dones, Rs, Advs, e_coef, v_coef
            )
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
            total_entropy_loss += entropy_loss
            
        total_loss = total_actor_loss + total_critic_loss + total_entropy_loss
        
        if summary_writer is not None and global_step is not None:
            summary_writer.add_scalar('loss/total_actor_loss', total_actor_loss.item(), global_step)
            summary_writer.add_scalar('loss/total_critic_loss', total_critic_loss.item(), global_step)
            summary_writer.add_scalar('loss/total_entropy_loss', total_entropy_loss.item(), global_step)
            
        return total_loss

    def _get_fps(self, obs, nactions,agent_id=None):
        """获取邻居动作特征"""
        fps = []
        for i in range(self.n_agent):
            if nactions[i] is not None:
                fp_i = self._get_fp(nactions[i], self.na_ls_ls[i])
                fps.append(fp_i.to(device))
            else:
                fps.append(None)
        return fps
        

    def _get_fp(self, nactions, na_ls):
        """将邻居动作转换为one-hot特征 - 修复版本"""
        fps = []    
        # 使用na_ls[0]作为所有智能体的动作维度（假设动作空间相同）
        na_dim = na_ls[0] if na_ls else 4    
        for i, na in enumerate(nactions):
            if na is not None:
                fp = torch.zeros(na_dim, device=device)
                # 确保动作索引在有效范围内
                na = max(0, min(int(na), na_dim - 1))
                fp[na] = 1.0
                fps.append(fp)   
        if fps:
            return torch.cat(fps)
        return None
    
    def _get_neighbor_dim(self, i_agent):
        # n_n = int(np.sum(self.neighbor_mask[i_agent]))
        n_n = int(self.neighbor_mask[i_agent].sum()) ## PyTorch 的求和操作
        if self.identical:
            return n_n, self.n_s * (n_n+1), self.n_a * n_n, [self.n_s] * n_n, [self.n_a] * n_n
        else:
            ns_ls = []
            na_ls = []
            # for j in np.where(self.neighbor_mask[i_agent])[0]:
            for j in torch.where(self.neighbor_mask[i_agent])[0]: ##
                ns_ls.append(self.n_s_ls[j])
                na_ls.append(self.n_a_ls[j])
            return n_n, self.n_s_ls[i_agent] + sum(ns_ls), sum(na_ls), ns_ls, na_ls
            # return n_n, self.n_s_ls[i_agent] + torch.stack(ns_ls).sum(), torch.stack(na_ls).sum(), ns_ls, na_ls

    def _reset(self):
        """重置网络状态"""
        self.states = []
        for i in range(self.n_agent):
            h = torch.zeros(1, self.n_h, device=device)
            c = torch.zeros(1, self.n_h, device=device)
            self.states.append((h, c))