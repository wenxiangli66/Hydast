import pandas as pd
import torch
from torch_geometric.data import DataLoader
import pickle
import numpy as np
from tqdm import tqdm
import os
from util import read_pkl
from build_tree import Voc
from torch_geometric.data import Data

class EHRTokenizer(object):
    def __init__(self, vocab_file):
        self.vocab = Voc()  # fused all codes such as diag_codes, proce_codes and atc_codes
        self.diag_voc, self.proce_voc, self.atc_voc = self.add_vocab(
            vocab_file)  # get for build ontology EHR Model

    def add_vocab(self, vocab_file):
        voc1, voc2, voc3 = Voc(), Voc(), Voc()# 为每种代码类型创建独立的词汇表
        all_codes_dic = pickle.load(open(vocab_file, 'rb'))# 加载词汇文件
        diag_codes, proce_codes, atc_codes = ['d_'+d for d in all_codes_dic['diag_codes']], [
            'p_'+p for p in all_codes_dic['proce_codes']], ['a_'+a for a in all_codes_dic['atc_codes']]
        # 每种代码都添加前缀（如 d_、p_、a_）以区分不同类型。
        # 将各类型代码添加到相应的词汇表中
        voc1.add_sentence(diag_codes)
        self.vocab.add_sentence(diag_codes)

        voc2.add_sentence(proce_codes)
        self.vocab.add_sentence(proce_codes)

        voc3.add_sentence(atc_codes)
        self.vocab.add_sentence(atc_codes)

        return voc1, voc2, voc3

    # for each single graph transform the code to index
    def build_single_graph(self, diag_codes, proce_codes, atc_codes):
        single_voc = Voc() # 创建一个单独的词汇表
        # 将诊断代码、程序代码和药物代码分别添加到single_voc中。每种代码前面加上对应的前缀（d_、p_、a_），以区分不同类型的代码。
        single_voc.add_sentence(['d_'+d for d in diag_codes])
        single_voc.add_sentence(['p_'+p for p in proce_codes])
        single_voc.add_sentence(['a_'+a for a in atc_codes])

        sorted_idx = sorted(single_voc.idx2word.keys())# 对索引进行排序
        # 生成排序后的代码列表
        sorted_codes = []
        for idx in sorted_idx:
            code = single_voc.idx2word[idx]
            sorted_codes.append(code)

        return single_voc, self.convert_codes_to_ids(sorted_codes, '')
    # 返回创建的single_voc 和通过 convert_codes_to_ids方法转换的排序代码的索引列表。
    # convert_codes_to_ids 方法将代码转换为对应的索引，便于后续使用。

    # construct the graph with ontology code index
    def build_onto_single_graph(self, diag_codes, proce_codes, atc_codes):
        single_voc = Voc()
        single_voc.add_sentence(['d_'+d for d in diag_codes])# 添加诊断代码
        single_voc.add_sentence(['p_'+p for p in proce_codes])# 添加程序代码
        single_voc.add_sentence(['a_'+a for a in atc_codes])# 添加药物代码

        code_ids = []# 用于存储代码索引

        for code in diag_codes:
            code_ids.append(self.vocab.word2idx['d_'+code]) # 将诊断代码转换为索引

        for code in proce_codes:
            code_ids.append(self.vocab.word2idx['p_'+code])  # 将程序代码转换为索引

        for code in atc_codes:
            if code != 'nan':
                code_ids.append(self.vocab.word2idx['a_'+code]) # 将药物代码转换为索引

        return single_voc, code_ids# 返回词汇表和代码索引
#build_onto_single_graph 方法在构图中的贡献是为每个病人构建一个特定的词汇表（Voc）和代码索引列表（code_ids）
    # 在 # build_patient_graph方法中，生成的code_ids会被用作图中的节点，表示诊断、程序和药物的节点。
    # single_voc 提供了字典映射（word2idx）来查找每个诊断、程序和药物的索引。

    def convert_codes_to_ids(self, codes, c_type):
        ids = []
        for code in codes:
            ids.append(self.vocab.word2idx[c_type+code])
        return ids

class UndirectPatientOntoGraphEx(object):
    def __init__(self, diag_codes, proce_codes, atc_codes, rel_infos, tokenizer):
        self.diag_codes = diag_codes
        self.proce_codes = proce_codes
        self.atc_codes = atc_codes
        self.tokenizer = tokenizer #用于处理代码的 EHRTokenizer 实例

        self.diag_proce_rel, self.diag_atc_rel, self.proce_atc_rel = rel_infos

        self.x, self.edge_index, self.edge_type = \
            self.build_patient_graph(
                self.tokenizer, diag_codes, proce_codes, atc_codes)

    # construct patient graph onto_code_ids for ontology index mapping
    def build_patient_graph(self, tokenizer: EHRTokenizer, diag_codes, proce_codes, atc_codes):

#如果是疾病预测任务，诊断代码将被忽略。

        single_voc, onto_code_ids = tokenizer.build_onto_single_graph(
            diag_codes, proce_codes, atc_codes)

        edge_idx = []

        '''
        for heterogeneous graph neural network model: RGCN, RGAT...
        0,1,2,3,4,5,6,7,8 represent the edge type for diag-diag, proce-proce, atc-atc, 
        diag-proce, proce-diag, diag-atc, atc-diag, proce-atc, atc-proce
        '''
        edge_type = []
        edge_len = len(edge_idx)
        code_set = set()
        # 初始化边和边类型：

        # construct the edge for diagnosis and procedure
        all_diag_proce_pairs = [(d, p)
                                for d in diag_codes for p in proce_codes]
        # 对于每个诊断代码d，与每个程序代码p，组合形成一对(d, p)，包括了诊断和程序节点之间所有可能的配对。
        # 形成一个列表 all_diag_proce_pairs。

        # if there has the relations, construct the edge in this graph
        valid_diag_proce_pairs = [edge_idx.extend([(single_voc.word2idx['d_'+d_p[0]], single_voc.word2idx['p_'+d_p[1]]),
                                                   (single_voc.word2idx['p_'+d_p[1]], single_voc.word2idx['d_'+d_p[0]])])
                                  for d_p in all_diag_proce_pairs if d_p[0]+'-'+d_p[1] in self.diag_proce_rel]
        # 检查 d_p[0]+'-'+d_p[1] 是否存在于 self.diag_proce_rel 中。如果存在，则表示 d_p 是一个有效的诊断-程序配对。


        # update edge type for each edge
        edge_type.extend([3, 4]*int((len(edge_idx)-edge_len)/2))
        # 使用 extend 方法将类型 3 和 4 添加到 edge_type 列表中。这里的 3 代表从诊断到程序的边，4 代表从程序到诊断的边。
        # 计算新添加边的数量，通过 len(edge_idx) - edge_len 来获取新增边的数量，并用 int(... / 2) 确定添加多少个边类型（因为每对边都有两个类型）。

        for d_p in all_diag_proce_pairs:
            if d_p[0]+'-'+d_p[1] in self.diag_proce_rel:
                if d_p[0] not in code_set:
                    edge_idx.extend(
                        [(single_voc.word2idx['d_'+d_p[0]], single_voc.word2idx['d_'+d_p[0]])])
                    code_set.add(d_p[0])
                    edge_type.append(0)

                if d_p[1] not in code_set:
                    edge_idx.extend(
                        [(single_voc.word2idx['p_'+d_p[1]], single_voc.word2idx['p_'+d_p[1]])])
                    code_set.add(d_p[1])
                    edge_type.append(1)
        edge_len = len(edge_idx)
        # 更新 edge_len 变量，以便后续计算新增边的数量。


        # construct the edge for diagnosis and atc
        all_diag_atc_pairs = [(d, a) for d in diag_codes for a in atc_codes]

        valid_diag_atc_pairs = [edge_idx.extend([(single_voc.word2idx['d_'+d_a[0]], single_voc.word2idx['a_'+d_a[1]]),
                                                 (single_voc.word2idx['a_'+d_a[1]], single_voc.word2idx['d_'+d_a[0]])])
                                for d_a in all_diag_atc_pairs if d_a[0]+'-'+d_a[1] in self.diag_atc_rel]

        edge_type.extend([5, 6]*int((len(edge_idx)-edge_len)/2))

        for d_a in all_diag_atc_pairs:
            if d_a[0]+'-'+d_a[1] in self.diag_atc_rel:
                if d_a[0] not in code_set:
                    edge_idx.extend(
                        [(single_voc.word2idx['d_'+d_a[0]], single_voc.word2idx['d_'+d_a[0]])])
                    code_set.add(d_a[0])
                    edge_type.append(0)
                if d_a[1] not in code_set:
                    edge_idx.extend(
                        [(single_voc.word2idx['a_'+d_a[1]], single_voc.word2idx['a_'+d_a[1]])])
                    code_set.add(d_a[1])
                    edge_type.append(2)
        edge_len = len(edge_idx)

        # construct the edge for procedure and atc
        all_proce_atc_pairs = [(p, a) for p in proce_codes for a in atc_codes]
        valid_proce_atc_pairs = [edge_idx.extend([(single_voc.word2idx['p_'+p_a[0]], single_voc.word2idx['a_'+p_a[1]]), (single_voc.word2idx['a_'+p_a[1]],
                                                 single_voc.word2idx['p_'+p_a[0]])]) for p_a in all_proce_atc_pairs if p_a[0]+'-'+p_a[1] in self.proce_atc_rel]

        edge_type.extend([7, 8]*int((len(edge_idx)-edge_len)/2))

        for p_a in all_proce_atc_pairs:
            if p_a[0]+'-'+p_a[1] in self.proce_atc_rel:
                if p_a[0] not in code_set:
                    edge_idx.extend(
                        [(single_voc.word2idx['p_'+p_a[0]], single_voc.word2idx['p_'+p_a[0]])])
                    code_set.add(p_a[0])
                    edge_type.append(1)
                if p_a[1] not in code_set:
                    edge_idx.extend(
                        [(single_voc.word2idx['a_'+p_a[1]], single_voc.word2idx['a_'+p_a[1]])])
                    code_set.add(p_a[1])
                    edge_type.append(2)

        row = list(map(lambda x: x[0], edge_idx))
        col = list(map(lambda x: x[1], edge_idx))
        # 构建边的索引，形成邻接矩阵。
        assert len(row) == len(edge_type)
        # 检查 row 列表的长度是否与 edge_type 列表的长度相等。
        # 目的：确保每条边都有一个对应的类型。这是一个重要的检查，因为如果数量不匹配，可能会导致后续处理出现错误。

        # 返回构建的代码索引、边的索引和边的类型。
        return onto_code_ids, [row, col], edge_type


#全局的EHR配对
def load_rel(rel_dir):
    # set the threshold for pmi
    pmi_threshold = 1  #设置一个阈值 pmi_threshold，用于过滤与关系相关的点互信息（PMI）值。
    # read the relations from the file
    diag_proce_df = pd.read_csv(rel_dir+'\diag_proce_rel.csv', sep='\t')
    diag_proce_df = diag_proce_df[diag_proce_df['pmi'] > pmi_threshold]
    # 从指定目录读取诊断与程序之间的关系数据，使用制表符（\t）作为分隔符。
    # 过滤出 PMIs 大于 pmi_threshold 的行。

    diag_pres_df = pd.read_csv(
        rel_dir+'/diag_pres_rel.csv', sep='\t', dtype={'tail ent': str})
    diag_pres_df = diag_pres_df[diag_pres_df['pmi'] > pmi_threshold]
    proce_pres_df = pd.read_csv(
        rel_dir+'/proce_pres_rel.csv', sep='\t', dtype={'tail ent': str})
    proce_pres_df = proce_pres_df[proce_pres_df['pmi'] > pmi_threshold]
    # 读取与诊断和处方、程序和处方相关的关系数据，并过滤出 PMIs 大于 pmi_threshold 的行。
    ndc2rxnorm_file = 'mimic4/ndc2rxnorm_mapping.txt'
    with open(ndc2rxnorm_file, 'r') as f:
        ndc2rxnorm = eval(f.read())
    # 从指定文件中读取 NDC（国家药品编码）到 RXNORM（药物命名标准）之间的映射关系。

    # 生成关系对
    diag_proce_pairs = []
    diag_proce_df.apply(lambda row: diag_proce_pairs.append(
        str(row['head ent'])+'-'+str(row['tail ent'])), axis=1)
    # 对于每一行数据，将诊断和程序的关系对（以 head ent 和 tail ent 连接）添加到 diag_proce_pairs 列表中。


    diag_pres_pairs = []
    diag_pres_df.apply(lambda row: diag_pres_pairs.append(str(
        row['head ent'])+'-'+ndc2rxnorm[row['tail ent']]) if row['tail ent'] in ndc2rxnorm else None, axis=1)
    # 如果 tail ent 存在于 NDC 到 RXNORM 的映射中，则将诊断和处方的关系对添加到 diag_pres_pairs 列表中。


    proce_pres_pairs = []
    proce_pres_df.apply(lambda row: proce_pres_pairs.append(str(
        row['head ent'])+'-'+ndc2rxnorm[row['tail ent']]) if row['tail ent'] in ndc2rxnorm else None, axis=1)
    # 类似地，生成程序与处方之间的关系对。
    diag_pres_pairs = set(diag_pres_pairs)
    diag_proce_pairs = set(diag_proce_pairs)
    proce_pres_pairs = set(proce_pres_pairs)

    return diag_proce_pairs, diag_pres_pairs, proce_pres_pairs

REL_INFOS = load_rel('mimic4')


#：该函数从 EHR 数据文件中提取诊断、程序和药物代码，并将其转换为图结构数据（UndirectPatientOntoGraphEx 对象）
# ，
#输出sub_Graph是一个嵌套列表，按患者 ID 分组，每个患者对应一个图数据列表
# sub_Graph = [
#     [ehr_graph_1, ehr_graph_2],  # 1号病人的所有病历
#     [ehr_graph_3, ehr_graph_4],  # 2号病人的所有病历
#     [ehr_graph_5]                # 3号病人的所有病历
# ]
# ehr_graph_1 = UndirectPatientOntoGraphEx(['D1', 'D2'], ['P1', 'P2'], ['A1'], REL_INFOS, tokenizer)
def load_ehr_infos(ehr_file, tokenizer):
    # Initialize dictionary to store EHR information
    ehr_infos = {}#初始化一个空字典，用于存储按患者 ID 索引的 EHR 信息
    ehr_df = pd.read_csv(ehr_file)#从指定的 CSV 文件读取 EHR 数据
    columns_to_convert = ['ATC', 'ICD_DIAG', 'ICD_PROCE']
    ehr_df[columns_to_convert] = ehr_df[columns_to_convert].astype(str)
#确保指定的列被视为字符串，这对处理编码非常重要。
    pre_sub = ehr_df['SUBJECT_ID'].iloc[0]
    sub_Graph = []#所有病人的所有病历单列表
    sub_list = []#一个病人的病历单列表
    for _, row in tqdm(ehr_df.iterrows()):
#循环遍历 DataFrame 的每一行，使用 tqdm 显示进度条。
        subject_id = row['SUBJECT_ID']
        diag_codes = list(set(row['ICD_DIAG'].split(',')))
        proce_codes = list(set(row['ICD_PROCE'].split(',')))
        atc_codes = list(set(row['ATC'].split(',')))
#提取每位患者的 HADM_ID，并将相关的医学编码拆分成列表。使用 set 确保每个编码唯一。

        # Create patient graph object using extracted codes and tokenizer
        ehr_graph = UndirectPatientOntoGraphEx(diag_codes, proce_codes, atc_codes, REL_INFOS, tokenizer)#构造EHR全局图，MKG

        if subject_id == pre_sub:
            sub_list.append(ehr_graph)
        else:
            sub_Graph.append(sub_list)
            sub_list = []
            sub_list.append(ehr_graph)
            pre_sub = subject_id
    # 将创建的患者图对象以 HADM_ID 为键存入 ehr_infos 字典。
    sub_Graph.append(sub_list)
    return sub_Graph
#返回包含所有按患者 ID 索引的 EHR 图的字典。

#rxnorm_to_atc_mapping 函数的目的是处理 NDC（国家药品代码）到 ATC（解剖治疗化学分类）映射的数据。
# Function to process NDC to ATC mappings

def rxnorm_to_atc_mapping(ndc2atc_file_path):#ndc2atc_file_path：包含 NDC 到 ATC 映射的 CSV 文件路径。
    ndc2atc_file = open(ndc2atc_file_path, 'r')
    rxnorm2atc = pd.read_csv(ndc2atc_file)
    #使用 open 打开指定的 CSV 文件，并利用 Pandas 的 read_csv 方法读取数据，存储为 DataFrame rxnorm2atc。
    rxnorm2atc = rxnorm2atc.drop(columns=['YEAR', 'MONTH', 'NDC'])#删除不需要的列 YEAR、MONTH 和 NDC。
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)#基于 RXCUI 列去重，以确保每个 RXCUI 只出现一次。
    rxnorm2atc['RXCUI'] = rxnorm2atc['RXCUI'].map(lambda x: str(x))#将 RXCUI 列中的值转换为字符串类型，确保在后续映射中没有数据类型的问题。
    # Create dictionary mapping 'RXCUI' to 'ATC4'
    rxnorm2atc_mapping = rxnorm2atc.set_index('RXCUI')['ATC4'].to_dict()
    #将 DataFrame 中的 RXCUI 列设为索引，并提取 ATC4 列，最终转换为字典 rxnorm2atc_mapping，其格式为 {'RXCUI': 'ATC4'}。
    return rxnorm2atc_mapping
#返回从 RXCUI 到 ATC4 的映射字典。

#load_text_embeddings 函数的目的是从指定目录加载诊断、程序和 ATC 代码的文本嵌入（embeddings），并返回相应的字典。
# Load text embedding from .pkl file
def load_text_embeddings(base_dir):
    # read diagnosis code embeddings
    embs = read_pkl(base_dir + 'diag_embs.pkl')#使用 read_pkl 函数读取诊断-处方嵌入文件。
    diag_embs = {}
    for code, emb in embs.items():
        #print(code[2:],emb)
        diag_embs[code[2:].replace('.', '')] = emb#创建一个字典 diag_embs，将每个代码（去掉前缀并替换点）与其对应的嵌入关联。

    # read procedure code embeddings
    embs = read_pkl(base_dir + 'pro_embs.pkl')
    proce_embs = {}
    for code, emb in embs.items():
        proce_embs[code[2:].replace('.', '')] = emb
#同样地，读取手术-处方嵌入文件并创建字典 proce_embs，将程序代码与嵌入关联。


    rxnorm2atc = rxnorm_to_atc_mapping(base_dir + 'ndc2atc_level4.csv')
#调用之前定义的 rxnorm_to_atc_mapping 函数，读取 NDC 到 ATC 的映射数据，返回一个字典 rxnorm2atc。

    # read atc code embeddings
    embs = read_pkl(base_dir + 'atc_embs.pkl')
    atc_embs = {}#读取 ATC-处方嵌入文件，创建字典 atc_embs。实质是RXNORM_embeding
    for code, emb in embs.items():
        code = code[2:]
        keys = [key for key, value in rxnorm2atc.items() if value == code]
        #对于每个嵌入，去掉前缀，然后查找 rxnorm2atc 字典中所有与此 ATC 代码对应的 NDC（RXCUI）
        for key in keys:
            atc_embs[key] = emb
            #将所有匹配的 NDC 代码与相应的嵌入关联。
    return diag_embs, proce_embs, atc_embs



def gen_Graph(dataset):
    dataset = dataset
    base_dir = dataset+'/'
    vocab_file = base_dir+'vocab.pkl'
    #从配置中获取数据集名称，并构建基础目录路径和词汇文件路径。

    tokenizer = EHRTokenizer(vocab_file)
    #使用指定的词汇文件初始化 EHRTokenizer。

    ehr_file = base_dir+'hame_graph.csv'
    ehr_infos = load_ehr_infos(ehr_file, tokenizer)

    diag_embeddings, proce_embeddings, atc_embeddings = load_text_embeddings('mimic4/')

    vocab_emb = np.random.randn(len(tokenizer.vocab.word2idx), 768)
    # 初始化一个随机的词汇嵌入矩阵 vocab_emb，大小为 (词汇表大小, 768)。768 是常用的嵌入维度（例如，BERT 的维度）

    #pickle.dump(tokenizer.vocab.idx2word, open('mimic4/id2word.pkl', 'wb'))

    for idx, word in tokenizer.vocab.idx2word.items():
        # 使用 tokenizer.vocab.idx2word 遍历词汇表，获取每个词及其对应的索引。
        w_type = word[0]  # w_type：提取词的类型，通常第一个字符用于区分不同的代码类型（如 d 表示诊断代码，p 表示程序代码，a 表示 ATC 代码）
        if w_type == 'd' and word[2:] in diag_embeddings:
            vocab_emb[idx] = diag_embeddings[word[2:]]
            # 如果 word[2:]（去掉前两个字符后的部分）在 diag_embeddings 中，则将对应的嵌入赋值给 vocab_emb[idx]。
        elif w_type == 'p' and word[2:] in proce_embeddings:
            vocab_emb[idx] = proce_embeddings[word[2:]]
        elif w_type == 'a' and word[2:] in atc_embeddings:
            vocab_emb[idx] = atc_embeddings[word[2:]]
    vocab_emb = torch.tensor(vocab_emb, dtype=torch.float)

    print(len(ehr_infos))
    graph_list = []
    for i in range(len(ehr_infos)):
        data_list = []
        for j in range(len(ehr_infos[i])):
            ehr = ehr_infos[i][j]
            ehr_x = torch.tensor(
                ehr.x, dtype=torch.long)
            left_ehr_edge_index = torch.tensor(
                ehr.edge_index, dtype=torch.long)
            left_edge_type = torch.tensor(ehr.edge_type, dtype=torch.long)
            # print(ehr_x.shape)
            # 获取当前 EHR 的信息，并将其转换为 PyTorch 张量。
            cur_idx_data = Data(x=ehr_x, edge_index=left_ehr_edge_index,
                                edge_attr=left_edge_type)#封装成图卷积的数据类型，以便后续直接使用
            data_list.append(cur_idx_data)
        graph_list.append(data_list)
    print(graph_list[1][0].x,graph_list[1][0])
    # for i, data_list in enumerate(graph_list):
    #     for j, graph_data in enumerate(data_list):
    #         if hasattr(graph_data, 'x') and graph_data.x is not None:
    #             # 检查节点特征矩阵的形状
    #             if graph_data.x.dim() == 2:  # 确保是二维张量 [num_nodes, feature_dim]
    #                 num_nodes = graph_data.x.shape[0]
    #                 feature_dim = graph_data.x.shape[1]
    #                 print(f"Graph List[{i}][{j}]: num_nodes = {num_nodes}, feature_dim = {feature_dim}")
    #             elif graph_data.x.dim() == 1:  # 如果是一维张量 [num_nodes]
    #                 num_nodes = graph_data.x.shape[0]
    #                 print(f"Graph List[{i}][{j}]: num_nodes = {num_nodes}, feature_dim = 1 (scalar features)")
    #             else:
    #                 print(f"Graph List[{i}][{j}]: Unexpected node feature shape: {graph_data.x.shape}")
    #         else:
    #             print(f"Graph List[{i}][{j}]: Node features are missing or empty.")

    # print("第一个病人的第一个病历单的节点特征：")
    # print( graph_list.x.shape )
    # print("第一个病人的第一个病历单的边索引：")
    # print(graph_list[0][0].edge_index)
    print('*********************')
    print(vocab_emb)
    return graph_list, vocab_emb

