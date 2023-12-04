import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from collections import OrderedDict

class DGCRM(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(DGCRM, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.DGCRM_cells = nn.ModuleList()
        self.DGCRM_cells.append(DDGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.DGCRM_cells.append(DDGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        # print(x.shape, self.node_num, self.input_dim)
        # exit()
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]     #x=[batch,steps,nodes,input_dim]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]   #state=[batch,steps,nodes,input_dim]
            inner_states = []
            for t in range(seq_length):   #如果有两层GRU，则第二层的GGRU的输入是前一层的隐藏状态
                state = self.DGCRM_cells[i](current_inputs[:, t, :, :], state, [node_embeddings[0][:, t, :, :], node_embeddings[1]])#state=[batch,steps,nodes,input_dim]
                # state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state,[node_embeddings[0], node_embeddings[1]])
                inner_states.append(state)   #一个list，里面是每一步的GRU的hidden状态
            output_hidden.append(state)  #每层最后一个GRU单元的hidden状态
            current_inputs = torch.stack(inner_states, dim=1)
            #拼接成完整的上一层GRU的hidden状态，作为下一层GRRU的输入[batch,steps,nodes,hiddensize]
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.DGCRM_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class DDGCRN(nn.Module):
    def __init__(self, args):
        super(DDGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = 1
        self.hidden_dim = args.rnn_unit
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layer
        self.use_D = args.use_day
        self.use_W = args.use_week
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.default_graph = args.default_graph
        self.node_embeddings1 = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        self.node_embeddings2 = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        self.T_i_D_emb = nn.Parameter(torch.empty(288, args.embed_dim))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, args.embed_dim))

        self.encoder1 = DGCRM(args.num_nodes, self.input_dim, args.rnn_unit, args.cheb_k,
                              args.embed_dim, args.num_layer)
        self.encoder2 = DGCRM(args.num_nodes, self.input_dim, args.rnn_unit, args.cheb_k,
                              args.embed_dim, args.num_layer)
        #predictor
        self.end_conv1 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv2 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv3 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
    
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])
    
    def forward(self, source, label=None, ddg_i=2):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        # supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        # init_state = self.encoder.init_hidden(source.shape[0])   #[2,64,307,64] 前面是2是因为有两层GRU
        # output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        # # output = output[:, -1:, :, :]                                   #B, 1, N, hidden
        # output = self.dropout1(output[:, -1:, :, :])
        #
        # #CNN based predictor
        # output = self.end_conv((output))                         #B, T*C, N, 1
        #
        # return output
        node_embedding1 = self.node_embeddings1
        if self.use_D:
            t_i_d_data   = source[..., 1]

            # T_i_D_emb = self.T_i_D_emb[(t_i_d_data[:, -1, :] * 288).type(torch.LongTensor)]
            T_i_D_emb = self.T_i_D_emb[(t_i_d_data * 288).type(torch.LongTensor)]
            node_embedding1 = torch.mul(node_embedding1, T_i_D_emb)

        if self.use_W:
            d_i_w_data   = source[..., 2]
            # D_i_W_emb = self.D_i_W_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)]
            D_i_W_emb = self.D_i_W_emb[(d_i_w_data).type(torch.LongTensor)]
            node_embedding1 = torch.mul(node_embedding1, D_i_W_emb)

        # time_embeddings = T_i_D_emb
        # time_embeddings = D_i_W_emb

        node_embeddings=[node_embedding1,self.node_embeddings1]

        source = source[..., 0].unsqueeze(-1)

        if ddg_i == 1:
            init_state1 = self.encoder1.init_hidden(source.shape[0])  # [2,64,307,64] 前面是2是因为有两层GRU
            output, _ = self.encoder1(source, init_state1, node_embeddings)  # B, T, N, hidden
            # output = output[:, -1:, :, :]                                   #B, 1, N, hidden
            output = self.dropout1(output[:, -1:, :, :])

            # CNN based predictor
            output1 = self.end_conv1(output)  # B, T*C, N, 1

            return output1

        else:
            init_state1 = self.encoder1.init_hidden(source.shape[0])   #[2,64,307,64] 前面是2是因为有两层GRU
            output, _ = self.encoder1(source, init_state1, node_embeddings)      #B, T, N, hidden
            # output = output[:, -1:, :, :]                                   #B, 1, N, hidden
            output = self.dropout1(output[:, -1:, :, :])

            #CNN based predictor
            output1 = self.end_conv1(output)                         #B, T*C, N, 1

            source1 = self.end_conv2(output)

            source2 = source -source1

            init_state2 = self.encoder2.init_hidden(source2.shape[0])   #[2,64,307,64] 前面是2是因为有两层GRU
            output2, _ = self.encoder2(source2, init_state2, node_embeddings)      #B, T, N, hidden
            # output2 = output2[:, -1:, :, :]                                   #B, 1, N, hidden
            output2 = self.dropout2(output2[:, -1:, :, :])

            # source2 = self.end_conv4(output2)

            output2 = self.end_conv3(output2)

            return output1 + output2

class DDGCRNCell(nn.Module):  #这个模块只进行GRU内部的更新，所以需要修改的是AGCN里面的东西
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(DDGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = GFS(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim)  # DGCN-----------------------------------------------------------
        self.update = GFS(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim)   # -----------------------------------------------------------

    def forward(self, x, state, node_embeddings):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)

class DGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(DGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.weights = nn.Parameter(torch.FloatTensor(cheb_k,dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        self.hyperGNN_dim = 16
        self.middle_dim = 2
        self.embed_dim = embed_dim
        self.fc=nn.Sequential( #疑问，这里为什么要用三层linear来做，为什么激活函数是sigmoid
                OrderedDict([('fc1', nn.Linear(dim_in, self.hyperGNN_dim)),
                             #('sigmoid1', nn.ReLU()),
                             ('sigmoid1', nn.Sigmoid()),
                             ('fc2', nn.Linear(self.hyperGNN_dim, self.middle_dim)),
                             #('sigmoid1', nn.ReLU()),
                             ('sigmoid2', nn.Sigmoid()),
                             ('fc3', nn.Linear(self.middle_dim, self.embed_dim))]))
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        node_num = node_embeddings[0].shape[1]
        supports1 = torch.eye(node_num).to(node_embeddings[0].device)
        filter = self.fc(x)
        nodevec = torch.tanh(torch.mul(node_embeddings[0], filter))  #[B,N,dim_in]
        supports2 = DGCN.get_laplacian(F.relu(torch.matmul(nodevec, nodevec.transpose(2, 1))), supports1)

        #supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)


        #default cheb_k = 3
        # for k in range(2, self.cheb_k):
        #     support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        #supports3 = torch.matmul(2 * supports2, supports2) - supports1
        x_g1 = torch.einsum("nm,bmc->bnc", supports1, x)
        x_g2 = torch.einsum("bnm,bmc->bnc", supports2, x)
        #x_g3 = torch.einsum("bnm,bmc->bnc", supports3, x)
        x_g = torch.stack([x_g1,x_g2],dim=1)

        # supports = torch.stack(support_set, dim=0)   #[2,nodes,nodes]  也就是这里把单位矩阵和自适应矩阵拼在一起了
        # x_g = torch.einsum("knm,bmc->bknc", supports, x)

        # weights = torch.einsum('bnd,dkio->bnkio', nodevec, self.weights_pool)

        weights = torch.einsum('nd,dkio->nkio', node_embeddings[1], self.weights_pool)    #[B,N,embed_dim]*[embed_dim,chen_k,dim_in,dim_out] =[B,N,cheb_k,dim_in,dim_out]
                                                                                  #[N, cheb_k, dim_in, dim_out]=[nodes,cheb_k,hidden_size,output_dim]
        bias = torch.matmul(node_embeddings[1], self.bias_pool) #N, dim_out                 #[che_k,nodes,nodes]* [batch,nodes,dim_in]=[B, cheb_k, N, dim_in]

        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        # x_gconv = torch.einsum('bnki,bnkio->bno', x_g, weights) + bias  #b, N, dim_out
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  #b, N, dim_out
        # x_gconv = torch.einsum('bnki,kio->bno', x_g, self.weights) + self.bias    #[B,N,cheb_k,dim_in] *[N,cheb_k,dim_in,dim_out] =[B,N,dim_out]

        return x_gconv

    @staticmethod
    def get_laplacian(graph, I, normalize=True):
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            #L = I - torch.matmul(torch.matmul(D, graph), D)
            L = torch.matmul(torch.matmul(D, graph), D)
        else:
            graph = graph + I
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.matmul(torch.matmul(D, graph), D)
        return L


def func(x):
    return x

class MLPGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(MLPGCN, self).__init__()
        self.in_dim = dim_in
        self.out_dim = dim_out 
        self.act = torch.relu
        self.align = nn.Linear(dim_in, dim_out) if dim_in!=dim_out else func
        self.ln = nn.LayerNorm([8600, dim_out])
        self.w = nn.Parameter(torch.randn(self.in_dim, self.out_dim), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(self.out_dim), requires_grad=True)

    def forward(self, x, node_embeddings):
        # x.shape : B,N,d
        res = self.align(x)
        x = torch.einsum('acd,de->ace', x, self.w)
        x = x + self.b
        return res + self.ln(self.act(x))
    
class GFS(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k=None, embed_dim=None, affine=True):
        super(GFS, self).__init__()
        self.in_dim = dim_in
        self.out_dim = dim_out 
        self.act = torch.relu
        self.affine = affine
        self.node_num = 8600
        self.align = nn.Linear(dim_in, dim_out) if dim_in!=dim_out else func
        
        self.w = nn.Parameter(torch.randn(self.in_dim, self.out_dim), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(self.out_dim), requires_grad=True)

        self.node_weight = nn.Parameter(torch.ones(1, self.node_num)/self.node_num, requires_grad=True)
        self.add_weight = nn.Parameter(torch.ones(self.node_num, 1)/self.node_num, requires_grad=True)
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.node_num, self.out_dim), requires_grad=True)
            self.affine_bias = nn.Parameter(torch.zeros(self.node_num, self.out_dim), requires_grad=True)

    def forward(self, x, node_embeddings=None):
        res = self.align(x)
        x = torch.einsum('acd,de->ace', x, self.w)
        x = self.act(x + self.b)
        x = torch.einsum('ef,fe,acd->aed', self.add_weight, self.node_weight, x)
        if self.affine:
            x = self.affine_weight*x + self.affine_bias
        return res + x