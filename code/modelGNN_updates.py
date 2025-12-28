import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F
from torch.nn.modules.module import Module

# NOTE: DGL imports removed for Kaggle compatibility
# The TypedLinear and fn from DGL are only used in RelGraphConv class
# which is not used in the main training pipeline (ReGCN is used instead)
# If you need RelGraphConv, install compatible DGL version:
# !pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html

import scipy.sparse as sp

att_op_dict = {
    'sum': 'sum',
    'mul': 'mul',
    'concat': 'concat'
}

"""GatedGNN with residual connection"""
class ReGGNN(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_GNN_layers, dropout, act=nn.functional.relu,
                 residual=True, att_op='mul', alpha_weight=1.0):
        super(ReGGNN, self).__init__()
        self.num_GNN_layers = num_GNN_layers
        self.residual = residual
        self.att_op = att_op
        self.alpha_weight = alpha_weight
        self.out_dim = hidden_size
        if self.att_op == att_op_dict['concat']:
            self.out_dim = hidden_size * 2

        self.emb_encode = nn.Linear(feature_dim_size, hidden_size).float()
        self.dropout_encode = nn.Dropout(dropout)
        self.z0 = nn.Linear(hidden_size, hidden_size).float()
        self.z1 = nn.Linear(hidden_size, hidden_size).float()
        self.r0 = nn.Linear(hidden_size, hidden_size).float()
        self.r1 = nn.Linear(hidden_size, hidden_size).float()
        self.h0 = nn.Linear(hidden_size, hidden_size).float()
        self.h1 = nn.Linear(hidden_size, hidden_size).float()
        self.soft_att = nn.Linear(hidden_size, 1).float()
        self.ln = nn.Linear(hidden_size, hidden_size).float()
        self.act = act

    def gatedGNN(self, x, adj):
        a = torch.matmul(adj, x)
        # update gate
        z0 = self.z0(a.float())
        z1 = self.z1(x.float())
        z = torch.sigmoid(z0 + z1)
        # reset gate
        r = torch.sigmoid(self.r0(a.float()) + self.r1(x.float()))
        # update embeddings
        h = self.act(self.h0(a.float()) + self.h1(r.float() * x.float()))

        return h * z + x * (1 - z)

    def forward(self, inputs, adj, mask):
        x = inputs
        x = self.dropout_encode(x)
        x = self.emb_encode(x.float())
        x = x * mask
        for idx_layer in range(self.num_GNN_layers):
            if self.residual:
                x = x + self.gatedGNN(x.float(), adj.float()) * mask.float()  # add residual connection, can use a weighted sum
            else:
                x = self.gatedGNN(x.float(), adj.float()) * mask.float()
        # soft attention
        soft_att = torch.sigmoid(self.soft_att(x))
        x = self.act(self.ln(x))
        x = soft_att * x * mask
        # sum/mean and max pooling

        # sum and max pooling
        if self.att_op == att_op_dict['sum']:
            graph_embeddings = torch.sum(x, 1) + torch.amax(x, 1)
        elif self.att_op == att_op_dict['concat']:
            graph_embeddings = torch.cat((torch.sum(x, 1), torch.amax(x, 1)), 1)
        else:
            graph_embeddings = torch.sum(x, 1) * torch.amax(x, 1)

        return graph_embeddings  

"""GCNs with residual connection"""
class ReGCN(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_GNN_layers, dropout, num_relations=3, num_bases=3, act=nn.functional.relu,
                 residual=True, att_op="mul", alpha_weight=1.0):
        super(ReGCN, self).__init__()
        self.num_GNN_layers = num_GNN_layers
        self.residual = residual
        self.att_op = att_op
        self.alpha_weight = alpha_weight
        self.hidden_size = hidden_size
        self.out_dim = hidden_size
        if self.att_op == att_op_dict['concat']:
            self.out_dim = hidden_size * 2

        self.gnnlayers = torch.nn.ModuleList()
        for layer in range(self.num_GNN_layers):
            if layer == 0:
                self.gnnlayers.append(RelationalGraphConvLayer(feature_dim_size, hidden_size, num_bases, num_relations, dropout, bias=True))  # bias=False
            else:
                self.gnnlayers.append(RelationalGraphConvLayer(hidden_size, hidden_size, num_bases, num_relations, dropout, bias=True))
        self.soft_att0 = nn.Linear(32, 1).float()
        self.ln0 = nn.Linear(1, 32).float()

        self.q = nn.Linear(hidden_size, hidden_size).float()
        self.k = nn.Linear(hidden_size, hidden_size).float()
        self.v = nn.Linear(hidden_size, hidden_size).float()
        self.soft_att = nn.Linear(hidden_size, 1).float()
        self.ln = nn.Linear(hidden_size, hidden_size).float()
        self.act = act
       
    def forward(self, inputs, adj, mask):
        x = inputs
        x = x.permute(0, 1, 3, 2)
        attn_token = torch.tanh(self.soft_att0(x.float()).float())
        attn_token = torch.nn.functional.softmax(self.ln0(attn_token), dim=3)
        x = attn_token * x
        x = torch.sum(x, 3)

        for idx_layer in range(self.num_GNN_layers):
            if idx_layer == 0:
                x = self.gnnlayers[idx_layer](adj, x) * mask
            else:
                if self.residual:
                    x = 0.4 * x + 0.6 * self.gnnlayers[idx_layer](adj, x) * mask  # Residual Connection, can use a weighted sum
                else:
                    x = self.gnnlayers[idx_layer](adj, x) * mask
        # soft attention
        query = self.q(x.float()).float()
        key = self.k(x.float()).float()
        value = self.v(x.float()).float()
        attn_node = torch.matmul(query, key.permute(0, 2, 1))
        attn_node = attn_node / math.sqrt(self.hidden_size)
        attn_probs = torch.nn.functional.softmax(attn_node, dim=-1)
        context = torch.matmul(attn_probs, value)
        graph_embeddings = torch.sum(context, 1) * torch.amax(context, 1)

        return graph_embeddings


"""GatedGNN"""
class GGGNN(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_GNN_layers, dropout, act=nn.functional.relu):
        super(GGGNN, self).__init__()
        self.num_GNN_layers = num_GNN_layers
        self.emb_encode = nn.Linear(feature_dim_size, hidden_size).float()
        self.dropout_encode = nn.Dropout(dropout)
        self.z0 = nn.Linear(hidden_size, hidden_size).float()
        self.z1 = nn.Linear(hidden_size, hidden_size).float()
        self.r0 = nn.Linear(hidden_size, hidden_size).float()
        self.r1 = nn.Linear(hidden_size, hidden_size).float()
        self.h0 = nn.Linear(hidden_size, hidden_size).float()
        self.h1 = nn.Linear(hidden_size, hidden_size).float()
        self.soft_att = nn.Linear(hidden_size, 1).float()
        self.ln = nn.Linear(hidden_size, hidden_size).float()
        self.act = act

    def gatedGNN(self, x, adj):
        a = torch.matmul(adj, x)
        # update gate
        z0 = self.z0(a.float())
        z1 = self.z1(x.float())
        z = torch.sigmoid(z0 + z1)
        # reset gate
        r = torch.sigmoid(self.r0(a.float()) + self.r1(x.float()))
        # update embeddings
        h = self.act(self.h0(a.float()) + self.h1(r.float() * x.float()))

        return h * z + x * (1 - z)

    def forward(self, inputs, adj, mask):
        x = inputs
        x = self.dropout_encode(x)
        x = self.emb_encode(x.float())
        x = x * mask
        for idx_layer in range(self.num_GNN_layers):
            x = self.gatedGNN(x.float(), adj.float()) * mask.float()
        return x


""" Simple GCN layer, similar to https://arxiv.org/abs/1609.02907 """
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout, act=torch.relu, bias=False):
        super(GraphConvolution, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.act = act
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        x = self.dropout(input)
        support = torch.matmul(x.float(), self.weight.float())
        output = torch.matmul(adj.float(), support.float())
        if self.bias is not None:
            output = output + self.bias
        return self.act(output)


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, n_layers=2, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.relu = torch.relu
        self.dropout = dropout
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations))
        for i in range(n_layers - 2):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
        self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations))

    def forward(self, x, edge_index, edge_type):
        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x, edge_index, edge_type))
            x = self.relu(x)
            x = self.dropout(x, p=self.dropout, training=self.training)
        return x

class RelationalGraphConvLayer(Module):
    def __init__(
        self, input_size, output_size, num_bases, num_rel, dropout, bias=False):
        super(RelationalGraphConvLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_bases = num_bases
        self.num_rel = num_rel
        self.act = torch.relu
        self.dropout = nn.Dropout(dropout)
        # R-GCN weights
        if num_bases > 0:
            self.w_bases = Parameter(torch.FloatTensor(self.num_bases, self.input_size, self.output_size))
            self.w_rel = Parameter(torch.FloatTensor(self.num_rel, self.num_bases))
        else:
            self.w = Parameter(torch.FloatTensor(self.num_rel, self.input_size, self.output_size))
        # R-GCN bias
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.output_size))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.num_bases > 0:
            nn.init.xavier_uniform_(self.w_bases.data)
            nn.init.xavier_uniform_(self.w_rel.data)
        else:
            nn.init.xavier_uniform_(self.w.data)
        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias.data.unsqueeze(0))


    def forward(self, A, X):
        X = self.dropout(X)
        self.w = (
            torch.einsum("rb, bio -> rio", (self.w_rel, self.w_bases))
            if self.num_bases > 0
            else self.w
        )
        weights = self.w.view(self.w.shape[0] * self.w.shape[1], self.w.shape[2])
        # Each relations * Weight
        supports = []
        A = A.permute(1, 0, 2, 3)
        for i in range(self.num_rel):
            supports.append(torch.matmul(A[i].float(), X.float()))

        tmp = torch.cat(supports, dim=2)
        out = torch.matmul(tmp.float(), weights.float())  # shape(#node, output_size)

        if self.bias is not None:
            out += self.bias.unsqueeze(0)
        return self.act(out)


def to_sparse(x):
    """converts dense tensor x to sparse format"""
    x_typename = torch.typename(x).split(".")[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def csr2tensor(A, cuda):
    coo = A.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    if cuda:
        out = torch.sparse.FloatTensor(i, v, torch.Size(shape)).cuda()
    else:
        out = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return out

# ============================================
# RelGraphConv class commented out for Kaggle compatibility
# This class requires DGL's TypedLinear and function modules
# which have compatibility issues with PyTorch 2.x on Kaggle
# The main training uses ReGCN/ReGGNN which don't require DGL
# ============================================
# class RelGraphConv(torch.nn.Module):
#     ... (class definition removed for Kaggle compatibility)

weighted_graph = False
print('using default unweighted graph')
