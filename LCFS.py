from supar import Parser
import nltk
import heapq
import torch
import dgl
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from nltk.parse.stanford import StanfordDependencyParser

#NLTK for tokenization
text = nltk.word_tokenize('The gourmet food is delicious but the service is poor')
print(text)

#BERT Model
model_name = 'bert-base-uncased'  # 您可以选择其他预训练模型
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 标记化句子
marked_text1 = ["[CLS]"] + text + ["[SEP]"]
# 将分词转化为词向量
input_ids = torch.tensor(tokenizer.encode(marked_text1, add_special_tokens=True)).unsqueeze(0)  # 添加批次维度
outputs = model(input_ids)
# 获取词向量
word_embeddings = outputs.last_hidden_state
# 提取单词对应的词向量（去掉特殊标记的部分）
word_embeddings = word_embeddings[:, 1:-1, :]  # 去掉[CLS]和[SEP]标记
# 使用切片操作去除第一个和最后一个元素
word_embedding = word_embeddings[0][1:-1, :]  # 节点特征
print(word_embedding.shape) #torch.Size([6, 768])


#BiAffine model compute for obtaining arcs、rels、probs
parser = Parser.load('biaffine-dep-en')   #'biaffine-dep-roberta-en'
dataset = parser.predict([text], prob=True, verbose=True)
print(dataset.sentences[0])
print(f"arcs:  {dataset.arcs[0]}\n"
      f"rels:  {dataset.rels[0]}\n"
      f"probs: {dataset.probs[0].gather(1,torch.tensor(dataset.arcs[0]).unsqueeze(1)).squeeze(-1)}")


#Construct Graph of stence, arcs--> node
arcs = dataset.arcs[0]  # node information
edges = [i + 1 for i in range(len(arcs))]
for i in range(len(arcs)):
      if arcs[i] == 0:
            arcs[i] = edges[i]

#将节点的序号减一，以便适应DGL graph从0序号开始
arcs = [arc - 1 for arc in arcs]
edges = [edge - 1 for edge in edges]
graph = (arcs,edges)
graph_line = '({}, {})\n'.format(graph[0], graph[1])  # Graph information transform String
print("graph:", graph)
print(graph_line)

# #Create a DGL graph
# g_weight = dgl.graph((arcs,edges))
# g_weight.edata['weight'] = torch.tensor([1.0] * g_weight.number_of_edges())  # 设置边的权重为1
# nx.draw(g_weight.to_networkx(),with_labels=True)
# plt.show()
# print(g_weight.edata['weight'])

# 创建一个有权图
G = nx.Graph()
for i,j in zip(arcs, edges):
        G.add_edge(i, j, weight=1)
# print(G)
# nx.draw(G,with_labels=True)
# plt.show()

# 节点的数量
num_nodes = G.number_of_nodes()

#Calculate the SRD values by node distance matrix(Phan et al.)
SRD_path_matrix = torch.zeros((num_nodes, num_nodes))
# Calculate the shortest path length between two nodes
for i in range(num_nodes):
    for j in range(num_nodes):
        paths = nx.shortest_path_length(G, source=i, target=j, weight='weight')
        SRD_path_matrix[i, j] = paths
print(SRD_path_matrix)


#Calculate the SRD values(Our Method. Self-Attention/Multi-Head Attention)
word_embedding_feature = word_embedding.unsqueeze(0)
embedding_dim = word_embedding_feature.shape[-1]
num_heads = 4  #当num_heads = 1则，Multi-Head Attention ===>Self Attention
#Using Multi-headAttention
attention = torch.nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)
word_embeddings_transposed = word_embedding_feature.transpose(0, 1)
attn_output, attn_weights = attention(word_embeddings_transposed, word_embeddings_transposed, word_embeddings_transposed)
attn_weights = attn_weights.squeeze()
#SRD_weight_matrix
SRD_weight_matrix = SRD_path_matrix * attn_weights
print(SRD_weight_matrix)


# # 设置阈值
# threshold_a = 2
#
# # 创建一个新的矩阵，用于存储经过阈值处理后的结果
# binary_matrix = np.zeros_like(SRD_path_matrix)
#
# # 根据阈值设置矩阵的值
# binary_matrix[SRD_path_matrix <= threshold_a] = 1
#
# print("SRD_matrix_CMD Phan et al. Method")
# print(binary_matrix)




#thresholds a*平均注意力权重
average_weights = torch.mean(attn_weights, dim=1)
threshold_a = 2


#LCFS-CMD
# 创建一个新的0矩阵，用于存储经过阈值处理后的结果
threshold_a = average_weights[1] * threshold_a
binary_matrix = np.zeros_like(SRD_path_matrix)

# 根据阈值设置矩阵的值
binary_matrix[SRD_weight_matrix <= threshold_a] = 1
print("SRD_matrix_CMD Our Method")
print(binary_matrix)
# 矩阵转换为张量
matrix_tensor = torch.tensor(binary_matrix)
print(matrix_tensor.shape)
V_CMD = torch.mm(matrix_tensor, word_embedding)
print(V_CMD)



#LCFS-CDW
for i in range(num_nodes):
    threshold_a = average_weights[i] * threshold_a
    for j in range(num_nodes):
        # paths = nx.shortest_path_length(G, source=i, target=j, weight='weight')
        distance = SRD_weight_matrix[i][j]
        # 应用阈值并进行线性衰减
        if distance > threshold_a:
            SRD_path_matrix[i, j] = 1.0 - (distance - threshold_a) / (num_nodes)
        else:
            SRD_path_matrix[i, j] = 1.0
print(SRD_path_matrix)
V_CMW = torch.mm(SRD_path_matrix, word_embedding)
print(V_CMW)