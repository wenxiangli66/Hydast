import torch.nn as nn
import torch
import torch.nn.functional as F
# Logistic Regression
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn


import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, HypergraphConv
from torch_geometric.nn import global_add_pool as add_p
from torch_geometric.nn import global_max_pool as max_p
from torch_geometric.nn import global_mean_pool as mean_p
from baseline_Seqmodels import *

class MultiLayerTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, vac_emb):
        super(MultiLayerTransformer, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.fin = nn.Linear(input_size, hidden_size)
        self.Transformer = TransformerLayer(heads=2,feature_size=64, dropout=0.5, num_layers=2)
        self.TCN=TCN(input_size=64, hidden_size=64, output_size=64)
        #self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*2, output_size*2)
        self.graph_embedding = nn.Parameter(vac_emb)
        self.conv1= HypergraphConv(in_channels=64, out_channels=64)
        self.conv2 = HypergraphConv(in_channels=64, out_channels=64)
        # self.conv2 = GATConv(in_channels=64, out_channels=64)
        # self.conv1= GCNConv(in_channels=64, out_channels=64)
        self.global_tensor_dim = 64
        self.fhi = nn.Linear(768, hidden_size)

        # self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, graphs):
        # 使用clone()创建输入的副本，避免原地修改
        x = self.fin(x.clone())

        for i in range(100):
            for j in range(len(graphs[i])):
                try:
                    graph = graphs[i][j]
                    if graph.edge_index.size(1) > 0:
                        # 1. 节点特征初始化
                        node_features = self.fhi(self.graph_embedding[graph.x])
                        edge_index = graph.edge_index
                        num_nodes = node_features.size(0)
                        
                        # 2. 构建多尺度超图结构
                        hyperedge_dict = {}
                        hedge_id = 0
                        
                        # 2.1 保留所有原始边
                        for idx in range(edge_index.size(1)):
                            src, dst = edge_index[:, idx]
                            hyperedge_dict[hedge_id] = [int(src), int(dst)]
                            hedge_id += 1
                        
                        # 2.2 构建基于路径的超边
                        for k in range(num_nodes):
                            path_nodes = set()
                            stack = [(k, [k])]
                            while stack:
                                node, path = stack.pop()
                                if len(path) <= 3:
                                    neighbors = edge_index[1][edge_index[0] == node]
                                    for next_node in neighbors:
                                        next_node = int(next_node)
                                        if next_node not in path:
                                            new_path = path + [next_node]
                                            stack.append((next_node, new_path))
                                            if len(new_path) > 2:
                                                path_nodes.update(new_path)
                        
                            if len(path_nodes) > 2:
                                hyperedge_dict[hedge_id] = list(path_nodes)
                                hedge_id += 1
                        
                        # 3. 构建超图边索引
                        node_to_hedge = []
                        hedge_to_node = []
                        
                        for hedge_id, nodes in hyperedge_dict.items():
                            for node in nodes:
                                node_to_hedge.append(int(node))
                                hedge_to_node.append(hedge_id)
                        
                        # 4. 转换为张量并进行超图卷积
                        if len(node_to_hedge) > 0:
                            hyperedge_index = torch.tensor([node_to_hedge, hedge_to_node], 
                                                         dtype=torch.long,
                                                         device=edge_index.device)
                            
                            # 5. 多层超图卷积 - 避免原地操作
                            h1 = F.relu(self.conv1(node_features, hyperedge_index))
                            h1 = F.dropout(h1, p=0.3, training=self.training)
                            h1 = h1 + node_features  # 使用加法而不是原地操作
                            
                            h2 = F.relu(self.conv2(h1, hyperedge_index))
                            h2 = F.dropout(h2, p=0.3, training=self.training)
                            graph_x = h2 + node_features  # 使用加法而不是原地操作
                            
                            # 6. 特征聚合
                            if graph.batch is None:
                                graph.batch = torch.zeros(node_features.size(0), 
                                                        dtype=torch.long, 
                                                        device=node_features.device)
                            
                            # 使用多种池化方式
                            graph_mean = mean_p(graph_x, graph.batch)
                            graph_max = max_p(graph_x, graph.batch)
                            graph_emb = torch.add(graph_mean, graph_max).div(2.0)  # 避免原地操作
                            
                            if graph_emb.dim() > 1:
                                graph_emb = graph_emb.squeeze(dim=1)
                            
                            # 特征融合 - 使用新变量
                            x_new = x[i, j] + 0.3 * graph_emb
                            x[i, j] = x_new
                        
                except Exception as e:
                    print(f"Error processing graph {i},{j}: {e}")
                    print(f"Graph info - nodes: {num_nodes}, edges: {edge_index.size(1)}")
                    continue

        # Transformer和TCN处理
        mask = torch.any(x != 0, dim=2)
        _, out1 = self.Transformer(x,mask)  # 移除mask参数，如果TransformerLayer不需要的话
        out2 = self.TCN(x)

        # 使用cat而不是concat，避免潜在的原地操作
        output = torch.cat([out1, out2], dim=-1)
        final_output = self.fc(output)

        return final_output,output


class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers):
        super(TransformerEncoderModel, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, src):
        src = self.transformer_encoder(src)
        src = self.fc(src)
        return src


class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size=3, num_layers=3):
        super(TCN, self).__init__()

        self.layers = nn.ModuleList()
        in_channels = input_size

        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, hidden_size, kernel_size, padding=(kernel_size - 1) // 2),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size),
                    nn.Dropout(0.2)
                )
            )
            in_channels = hidden_size

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 输入数据 x 的形状: (batch_size, sequence_length, feature_size)
        x = x.transpose(1, 2).contiguous()  # 转换为 (batch_size, feature_size, sequence_length)

        # 使用新变量存储中间结果，避免原地操作
        out = x
        for layer in self.layers:
            # 每一层使用新的变量存储结果
            out = layer(out)
            
        # 使用新变量进行池化操作
        pooled = torch.mean(out, dim=2)  # 对时间维度取平均
        
        # 最后的线性层
        output = self.fc(pooled)  # (batch_size, output_size)

        return output

# class TCNWithResidual(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, kernel_size=3, num_layers=3, dilation_rate=2):
#         super(TCNWithResidual, self).__init__()
#
#         self.layers = nn.ModuleList()
#         in_channels = input_size
#
#         for i in range(num_layers):
#             dilation = dilation_rate ** i  # 空洞卷积的扩张率
#             padding = (kernel_size - 1) * dilation // 2
#             self.layers.append(
#                 nn.Sequential(
#                     nn.Conv1d(in_channels, hidden_size, kernel_size, padding=padding, dilation=dilation),
#                     nn.ReLU(),
#                     nn.LayerNorm(hidden_size)  # 添加层归一化
#                 )
#             )
#             in_channels = hidden_size
#
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         x = x.transpose(1, 2)  # 转换为 (batch_size, feature_size, sequence_length)
#
#         for layer in self.layers:
#             residual = x
#             x = layer(x)
#             x = x + residual  # 残差连接
#
#         x = x.mean(dim=2)  # 对时间维度取平均
#         x = self.fc(x)  # (batch_size, output_size)
#
#         return x