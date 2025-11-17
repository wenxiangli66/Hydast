import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import pickle
import numpy as np
from tqdm import tqdm
import os
from util import read_pkl
from build_tree import Voc

# 英文到数字编码的转换
#词汇表构建：通过 add_vocab 方法，将所有诊断、过程和药物代码添加到融合词汇表中，同时分别创建三个独立的词汇表。
# 单一图构建：使用 build_single_graph 或 build_onto_single_graph 方法构建一个单一图（单患者图），这些图结构会作为图神经网络的输入。
# 代码索引转换：通过 convert_codes_to_ids 方法，将代码转换为图中的节点索引，用于模型训练和推理


# 这段代码定义了一个名为 EHRTokenizer 的类，
# 用于处理电子健康记录（EHR）中的各种代码（例如诊断代码、程序代码和药物代码）
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

    def convert_codes_to_ids(self, codes, c_type):
        ids = []
        for code in codes:
            ids.append(self.vocab.word2idx[c_type+code])
        return ids


#这个 UndirectPatientOntoGraphEx 类用于构建无向的患者电子健康记录（EHR）图，
# 主要是通过诊断（diagnosis）、过程（procedure）、药物（ATC）的关系信息来构建图结构，
# 并将这些信息编码为图的节点、边和边的类型。
class UndirectPatientOntoGraphEx(object):
    def __init__(self, diag_codes, proce_codes, atc_codes, rel_infos, tokenizer, disease_prediction=False):
        self.diag_codes = diag_codes
        self.proce_codes = proce_codes
        self.atc_codes = atc_codes
        self.tokenizer = tokenizer #用于处理代码的 EHRTokenizer 实例

        self.diag_proce_rel, self.diag_atc_rel, self.proce_atc_rel = rel_infos

        self.x, self.edge_index, self.edge_type = \
            self.build_patient_graph(
                self.tokenizer, diag_codes, proce_codes, atc_codes, disease_prediction=disease_prediction)

    # construct patient graph onto_code_ids for ontology index mapping
    def build_patient_graph(self, tokenizer: EHRTokenizer, diag_codes, proce_codes, atc_codes, disease_prediction=False):

        # disease prediciton task only have two type of nodes, proce and atc nodes
        if disease_prediction:
            diag_codes = []
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
#


# EHRPairData 是一个用于表示电子健康记录（EHR）图对的自定义数据类，继承自 torch_geometric.data.Data。
# 这个类主要用于将两个 EHR 图的节点、边信息以及它们之间的相似性标签存储在一起。
class EHRPairData(Data):
    """
    The data for pairs of EHR graphs
    """

    def __init__(self, x_left, edge_index_left, x_right, edge_index_right, y, left_edge_type,
                 right_edge_type):
        """
        Args:
        x_left (Tensor): Nodes in the left EHR graph.
        x_right (Tensor): Nodes in the right EHR graph.
        edge_index_left (LongTensor): Edge indices of the left EHR graph.
        edge_index_right (LongTensor): Edge indices of the right EHR graph.
        left_edge_type (Tensor): Edge type for edges in the left EHR graph.
        right_edge_type (Tensor): Edge type for edges in the right EHR graph.
        y (Tensor): Label for the EHR pair. 1 for similar, 0 for dissimilar.
        """
        super(EHRPairData, self).__init__()
        self.x_left = x_left
        self.x_right = x_right
        self.edge_index_left = edge_index_left
        self.edge_index_right = edge_index_right
        self.left_edge_type = left_edge_type
        self.right_edge_type = right_edge_type
        self.y = y

    def __inc__(self, key, value):
        if key == 'edge_index_left':
            return self.x_left.size(0)
        if key == 'edge_index_right':
            return self.x_right.size(0)
        else:
            return super(EHRPairData, self).__inc__(key, value)


# construct all the kg relations from the kg relations file
#这个函数 load_rel 的作用是加载、过滤并处理不同类型的医疗关系数据，
# 具体包括诊断（diag）、手术（proce）和处方（pres）之间的关系对。函数最终返回经过筛选和处理的三组关系对。
def load_rel(rel_dir):
    # set the threshold for pmi
    pmi_threshold = 1  #设置一个阈值 pmi_threshold，用于过滤与关系相关的点互信息（PMI）值。
    # read the relations from the file
    diag_proce_df = pd.read_csv(rel_dir+'diag_proce_rel.csv', sep='\t')
    diag_proce_df = diag_proce_df[diag_proce_df['pmi'] > pmi_threshold]
    # 从指定目录读取诊断与程序之间的关系数据，使用制表符（\t）作为分隔符。
    # 过滤出 PMIs 大于 pmi_threshold 的行。

    diag_pres_df = pd.read_csv(
        rel_dir+'diag_pres_rel.csv', sep='\t', dtype={'tail ent': str})
    diag_pres_df = diag_pres_df[diag_pres_df['pmi'] > pmi_threshold]
    proce_pres_df = pd.read_csv(
        rel_dir+'proce_pres_rel.csv', sep='\t', dtype={'tail ent': str})
    proce_pres_df = proce_pres_df[proce_pres_df['pmi'] > pmi_threshold]
    # 读取与诊断和处方、程序和处方相关的关系数据，并过滤出 PMIs 大于 pmi_threshold 的行。
    ndc2rxnorm_file = '../data/ndc_atc/ndc2rxnorm_mapping.txt'
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
# 将列表转换为集合，以去除重复的关系对，并返回三个集合：

# loading the relationship from our knowledge graph
REL_INFOS = load_rel('../data/mimic3/')


#这个函数 construct_EHR_pairs_dataloader 用于构建一个 EHR（电子健康记录）对的数据加载器，
# 目的是在图神经网络（如 GCN、GAT 等）任务中，将患者的 EHR 数据对及其相似性标签加载到模型中进行训练或推理
def construct_EHR_pairs_dataloader(processed_file, tokenizer, ehr_file, label_file, shuffle=True, batch_size=1, disease_prediction=False):
    """
    Construct EHR pairs data loader

    Args:
    processed_file (str): The file to save the processed data
    tokenizer: The tokenizer to convert the codes to index
    ehr_file (str): The file of EHR data
    label_file (str): The file of label data
    shuffle (bool): Whether to shuffle the data, default is True
    batch_size (int): The batch size, default is 1
    disease_prediction (bool): Whether to predict the disease, True for disease prediction task, False for EHR clustering task

    Returns:
    DataLoader: DataLoader object for loading the data
    """
    if not os.path.exists(processed_file):
        print("ehr infos construct start")
        # Load EHR pairs
        ehr_pairs = load_ehr_pairs(label_file)
        print("ehr pair construct complete")
        ehr_infos = load_ehr_infos(
            ehr_file, tokenizer, disease_prediction=disease_prediction)
        print("ehr infos construct complete")

        data_list = []
        #初始化一个空列表 data_list，
        for index in tqdm(range(len(ehr_pairs)),  desc='construct ehr pairs'):#并使用 tqdm 进行进度条显示。
            left_hadm_id, right_hadm_id, label = ehr_pairs[index][
                0], ehr_pairs[index][1], ehr_pairs[index][2]
            #获取当前 EHR 对的 ID 和标签。

            # Skip if EHR edge index is empty
            if len(ehr_infos[left_hadm_id].edge_index[0]) == 0:
                continue
            if len(ehr_infos[right_hadm_id].edge_index[0]) == 0:
                continue
                #如果 EHR 对的边索引为空，则跳过该对，跳过空边索引。

            # Convert data to tensor
            left_ehr_x = torch.tensor(
                ehr_infos[left_hadm_id].x, dtype=torch.long).unsqueeze(1)
            right_ehr_x = torch.tensor(
                ehr_infos[right_hadm_id].x, dtype=torch.long).unsqueeze(1)

            left_ehr_edge_index = torch.tensor(
                ehr_infos[left_hadm_id].edge_index, dtype=torch.long)
            right_ehr_edge_index = torch.tensor(
                ehr_infos[right_hadm_id].edge_index, dtype=torch.long)

            left_edge_type = torch.tensor(
                ehr_infos[left_hadm_id].edge_type, dtype=torch.long)
            right_edge_type = torch.tensor(
                ehr_infos[right_hadm_id].edge_type, dtype=torch.long)
            # 将 EHR 信息转换为 PyTorch 张量，以便输入到模型中。

            cur_idx_data = EHRPairData(left_ehr_x, left_ehr_edge_index, right_ehr_x, right_ehr_edge_index,
                                       torch.tensor(
                                           label, dtype=torch.float), left_edge_type,
                                       right_edge_type)  # for mse loss
            data_list.append(cur_idx_data)
        # 创建一个 EHRPairData 对象，包含左右 EHR 的节点、边索引、边类型和标签，并将其添加到 data_list 中。

        if not os.path.exists(os.path.dirname(processed_file)):
            os.makedirs(os.path.dirname(processed_file))
        torch.save(data_list, processed_file)
    #     如果处理文件的目录不存在，则创建该目录，并将 data_list 保存到 processed_file 中。

    else:
        data_list = torch.load(processed_file)
    #     如果处理文件已存在，直接加载该文件。

    loader = DataLoader(data_list, batch_size=batch_size,
                        shuffle=shuffle, follow_batch=['x_left', 'x_right'])
    # 使用 DataLoader 创建一个数据加载器，支持批处理和数据随机打乱。
    return loader


def construct_dataloder(tokenizer, processed_file, ehr_file, batch_size=1,
                        disease_prediction=False):
    """
    Construct query dataloader for EHR data.

    Args:
    tokenizer: Tokenizer to convert the codes to index.
    processed_file (str): Path to the processed file.
    ehr_file (str): Path to the EHR file.
    batch_size (int): Batch size for dataloader, default is 1.
    disease_prediction (bool): Whether to predict disease, True for disease prediction task, False for EHR clustering task.

    Returns:
    DataLoader: DataLoader object for loading the data.
    List: List of cohorts.
    """

    # Construct query pairs and corresponding diseases
    hadm_ids, diseases = generate_cohort_data(
        ehr_file)
    # 使用 generate_cohort_data 函数加载 EHR 数据，生成患者入院 ID（hadm_ids）和对应的疾病标签（diseases）
    if not os.path.exists(processed_file):
        ehr_infos = load_ehr_infos(
            ehr_file, tokenizer, disease_prediction=disease_prediction)
        # 使用 load_ehr_infos 函数加载 EHR 信息，包括节点特征和边索引
        data_list,  cohorts = [], []
        for i in tqdm(range(len(hadm_ids))):
            # 初始化 data_list 和 cohorts，并使用 tqdm 显示进度条。
            left_ehr = ehr_infos[hadm_ids[i]]
            left_ehr_x = torch.tensor(
                left_ehr.x, dtype=torch.long).unsqueeze(1)
            left_ehr_edge_index = torch.tensor(
                left_ehr.edge_index, dtype=torch.long)
            left_edge_type = torch.tensor(left_ehr.edge_type, dtype=torch.long)
            # 获取当前 EHR 的信息，并将其转换为 PyTorch 张量。
            cur_idx_data = Data(x=left_ehr_x, edge_index=left_ehr_edge_index,
                                edge_type=left_edge_type)

            data_list.append(cur_idx_data)
            cohorts.append(diseases[i])
        # 创建一个 Data 对象，包含 EHR 的节点特征、边索引和边类型，并将其添加到 data_list 中，同时将对应的疾病标签添加到 cohorts 中。
        if not os.path.exists(os.path.dirname(processed_file)):
            os.makedirs(os.path.dirname(processed_file))
        torch.save((cohorts, data_list), processed_file)
    else:
        cohorts, data_list = torch.load(processed_file)
    loader = DataLoader(data_list, batch_size=batch_size,
                        shuffle=False, follow_batch=['x_left', 'x_right'])
    return loader, cohorts


# load_ehr_infos 函数用于从 CSV 文件中加载电子健康记录（EHR）信息，处理相关的医学编码，
# 并为每条记录构建患者图对象。
def load_ehr_infos(ehr_file, tokenizer, disease_prediction=False):
    # Initialize dictionary to store EHR information
    ehr_infos = {}#初始化一个空字典，用于存储按患者 ID 索引的 EHR 信息
    ehr_df = pd.read_csv(ehr_file)#从指定的 CSV 文件读取 EHR 数据
    columns_to_convert = ['ATC', 'ICD9_DIAG', 'ICD9_PROCE']
    ehr_df[columns_to_convert] = ehr_df[columns_to_convert].astype(str)
#确保指定的列被视为字符串，这对处理编码非常重要。
    for _, row in tqdm(ehr_df.iterrows()):
#循环遍历 DataFrame 的每一行，使用 tqdm 显示进度条。

        hadm_id = row['HADM_ID']
        diag_codes = list(set(row['ICD9_DIAG'].split(',')))
        proce_codes = list(set(row['ICD9_PROCE'].split(',')))
        atc_codes = list(set(row['ATC'].split(',')))
#提取每位患者的 HADM_ID，并将相关的医学编码拆分成列表。使用 set 确保每个编码唯一。

        # Create patient graph object using extracted codes and tokenizer
        ehr_graph = UndirectPatientOntoGraphEx(diag_codes, proce_codes, atc_codes, REL_INFOS, tokenizer,
                                               disease_prediction=disease_prediction)
#使用提取的诊断、程序和 ATC 代码，以及关系（REL_INFOS）和分词器，实例化一个 UndirectPatientOntoGraphEx 对象。
        ehr_infos[hadm_id] = ehr_graph
    # 将创建的患者图对象以 HADM_ID 为键存入 ehr_infos 字典。
    return ehr_infos
#返回包含所有按患者 ID 索引的 EHR 图的字典。


#load_ehr_pairs 函数用于从标签文件中加载电子健康记录（EHR）对及其相似性标签。
def load_ehr_pairs(label_file):#label_file：包含 EHR 对及其标签的 CSV 文件路径。
    ehr_pairs = []#初始化一个空列表，用于存储 EHR 对及其标签。
    label_df = pd.read_csv(label_file, sep='\t')
    for _, row in label_df.iterrows():#循环遍历 DataFrame 的每一行。
        ehr_pairs.append([row[0], row[1], int(row[2])])#将每行的前两个元素（EHR 对的标识符）和第三个元素（标签，转换为整数）
        # 作为一个列表添加到 ehr_pairs 中。
    return ehr_pairs
# 返回包含所有 EHR 对及其相似性标签的列表。

#construct_query_pairs 函数用于从给定的电子健康记录（EHR）文件中构造查询对。
def construct_query_pairs(ehr_file):
    ehr_df = pd.read_csv(ehr_file)
    left_ehr_id = ehr_df.loc[0, 'HADM_ID']#从 DataFrame 中提取第一行的 HADM_ID，作为左侧 EHR ID。
    right_ehr_ids = list(ehr_df.loc[1:, 'HADM_ID'].values)
#获取 DataFrame 中除了第一行以外的所有 HADM_ID，将其存储为一个列表，作为右侧 EHR IDs。
    return [left_ehr_id]*len(right_ehr_ids), right_ehr_ids
#返回一个元组，其中第一个元素是由左侧 EHR ID 重复构成的列表（其长度与右侧 EHR IDs 相同），
# 第二个元素是右侧 EHR IDs 列表。

#get_cohorts 函数的主要功能是将测试、验证和训练 EHR 数据中的疾病标签转换为相应的索引。
def get_cohorts(test_ehr_file, valid_ehr_file, train_ehr_file):
    #test_ehr_file：测试数据的 EHR 文件路径。
# valid_ehr_file：验证数据的 EHR 文件路径。
# train_ehr_file：训练数据的 EHR 文件路径
    train_df, test_df, valid_df = pd.read_csv(train_ehr_file), pd.read_csv(
        test_ehr_file), pd.read_csv(valid_ehr_file)
    #使用 Pandas 从指定的文件路径读取训练、测试和验证数据，并将其存储为 DataFrame。
    diseases = list(set(test_df['disease'].values))#从测试数据中提取所有独特的疾病标签，创建一个列表 diseases。
    test_cohorts, valid_cohorts, train_cohorts = [], [], []#初始化三个空列表，分别用于存储测试、验证和训练数据中疾病的索引。
    for idx, row in train_df.iterrows():
        train_cohorts.append(diseases.index(row['disease']))
        #遍历训练数据中的每一行，通过 diseases.index(row['disease']) 获取疾病的索引，并将其添加到 train_cohorts 列表中。
    for idx, row in test_df.iterrows():
        test_cohorts.append(diseases.index(row['disease']))
    for idx, row in valid_df.iterrows():
        valid_cohorts.append(diseases.index(row['disease']))

    return test_cohorts, valid_cohorts, train_cohorts
#返回包含测试、验证和训练数据中疾病索引的三个列表。


#generate_cohort_data 函数的主要功能是从 EHR 文件中提取住院就诊 ID 和疾病信息，并生成一个疾病索引列表。
def generate_cohort_data(ehr_file):
    """
    Generate cohort data from EHR file.

    Args:
        ehr_file (str): Path to the EHR file in CSV format.

    Returns:
        tuple: A tuple containing two lists:
            - hadm_ids (list): List of Hospital Admission IDs (HADM_ID).
            - diseases (list): List of unique diseases extracted from the EHR.

    This function reads the EHR file specified by 'ehr_file' and extracts HADM_IDs
    and diseases associated with each admission. It then generates a list of unique
    diseases and creates a corresponding cohort list based on the index of each
    disease in the unique disease list.

    """

    admission_df = pd.read_csv(ehr_file)
    hadm_ids = admission_df['HADM_ID'].values#从 DataFrame 中提取所有的住院就诊 ID（HADM_ID），并将其存储为 NumPy 数组。
    diseases = list(set(admission_df['disease'].values))
    #从 DataFrame 中提取所有独特的疾病标签，并创建一个列表 diseases。
    cohorts = []
    for _, row in admission_df.iterrows():
        cohorts.append(diseases.index(row['disease']))
#初始化一个空列表 cohorts，遍历每一行，获取疾病在独特疾病列表中的索引，并将其添加到 cohorts 列表中。
    return hadm_ids, cohorts
#返回包含 HADM_IDs 和对应疾病索引的两个列表。


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
    embs = read_pkl(base_dir + 'diag_desc_emb.pkl')#使用 read_pkl 函数读取诊断-处方嵌入文件。
    diag_embs = {}
    for code, emb in embs.items():
        diag_embs[code[2:].replace('.', '')] = emb#创建一个字典 diag_embs，将每个代码（去掉前缀并替换点）与其对应的嵌入关联。

    # read procedure code embeddings
    embs = read_pkl(base_dir + 'proce_desc_emb.pkl')
    proce_embs = {}
    for code, emb in embs.items():
        proce_embs[code[2:].replace('.', '')] = emb
#同样地，读取手术-处方嵌入文件并创建字典 proce_embs，将程序代码与嵌入关联。


    rxnorm2atc = rxnorm_to_atc_mapping(base_dir + 'ndc2atc_level4.csv')
#调用之前定义的 rxnorm_to_atc_mapping 函数，读取 NDC 到 ATC 的映射数据，返回一个字典 rxnorm2atc。

    # read atc code embeddings
    embs = read_pkl(base_dir + 'atc_desc_emb.pkl')
    atc_embs = {}#读取 ATC-处方嵌入文件，创建字典 atc_embs。
    for code, emb in embs.items():
        code = code[2:]
        keys = [key for key, value in rxnorm2atc.items() if value == code]
        #对于每个嵌入，去掉前缀，然后查找 rxnorm2atc 字典中所有与此 ATC 代码对应的 NDC（RXCUI）
        for key in keys:
            atc_embs[key] = emb
            #将所有匹配的 NDC 代码与相应的嵌入关联。
    return diag_embs, proce_embs, atc_embs
#返回三个字典：diag_embs（诊断嵌入）、proce_embs（手术嵌入）和 atc_embs（ATC处方 嵌入）。


#load_dataset 函数负责加载数据集和嵌入，并根据配置设置任务类型。这一过程涉及从指定目录读取数据、加载嵌入、初始化词汇嵌入矩阵，
# 以及确定任务目标（如疾病预测或聚类）。最终，它会返回准备好的数据集和其他相关信息，以供后续的模型训练和评估使用。
def load_dataset(config, train_processed_file, test_processed_file, valid_cluster_processed_file,
                 test_knn_processed_file, valid_knn_processed_file,
                 train_input_file, batch_size):
    
    # mimic3 or mimic4
    dataset = config.dataset
    base_dir = dataset+'/'
    vocab_file = base_dir+'vocab.pkl'
#从配置中获取数据集名称，并构建基础目录路径和词汇文件路径。

    tokenizer = EHRTokenizer(vocab_file)
#使用指定的词汇文件初始化 EHRTokenizer。

    ehr_file = base_dir+'common_admission_df.csv'
    ehr_infos = load_ehr_infos(ehr_file, tokenizer, disease_prediction=True)

    # read text embeddings for medical codes
    diag_embeddings, proce_embeddings, atc_embeddings = load_text_embeddings(
        'chatgpt_desc/')
    #加载诊断、手术和 ATC 代码的文本嵌入。

    vocab_emb = np.random.randn(len(tokenizer.vocab.word2idx), 768)
    #初始化一个随机的词汇嵌入矩阵 vocab_emb，大小为 (词汇表大小, 768)。768 是常用的嵌入维度（例如，BERT 的维度）
    for idx, word in tokenizer.vocab.idx2word.items():
        #使用 tokenizer.vocab.idx2word 遍历词汇表，获取每个词及其对应的索引。
        w_type = word[0]#w_type：提取词的类型，通常第一个字符用于区分不同的代码类型（如 d 表示诊断代码，p 表示程序代码，a 表示 ATC 代码）
        if w_type == 'd' and word[2:] in diag_embeddings:
            vocab_emb[idx] = diag_embeddings[word[2:]]
            #如果 word[2:]（去掉前两个字符后的部分）在 diag_embeddings 中，则将对应的嵌入赋值给 vocab_emb[idx]。
        elif w_type == 'p' and word[2:] in proce_embeddings:
            vocab_emb[idx] = proce_embeddings[word[2:]]
        elif w_type == 'a' and word[2:] in atc_embeddings:
            vocab_emb[idx] = atc_embeddings[word[2:]]
    vocab_emb = torch.tensor(vocab_emb, dtype=torch.float)
#：将 NumPy 数组 vocab_emb 转换为 PyTorch 的浮点型张量，

    train_ehr_file, train_label_file = base_dir + \
        'train_admissions.csv', base_dir + 'train_label.csv'
    valid_ehr_file, valid_label_file = base_dir + \
        'valid_admissions.csv', base_dir + 'valid_label.csv'
    test_ehr_file = base_dir + 'test_admissions.csv'
#设置训练、验证和测试数据的文件路径。

    # True for disease prediction task, False for EHR clustering task
    disease_prediction = False
    if config.task == 'knn':
        disease_prediction = True
#默认将 disease_prediction 设置为 False。如果任务类型为 KNN，则将其设置为 True。


#看以下这个构造数据，把它看懂
    # construct dataloader for EHR pairs
    train_dataloader = construct_EHR_pairs_dataloader(train_processed_file, tokenizer, train_ehr_file, train_label_file,
                                                      batch_size=batch_size, disease_prediction=disease_prediction)

    test_dataloader, _ = construct_dataloder(tokenizer, test_processed_file, test_ehr_file, batch_size=batch_size,
                                             disease_prediction=False)
    valid_cluster_dataloader, _ = construct_dataloder(tokenizer, valid_cluster_processed_file, valid_ehr_file, batch_size=batch_size,
                                                      disease_prediction=False)

    valid_knn_dataloader, _ = construct_dataloder(tokenizer, valid_knn_processed_file, valid_ehr_file, batch_size=batch_size,
                                                  disease_prediction=True)
    test_knn_dataloader, _ = construct_dataloder(tokenizer, test_knn_processed_file, test_ehr_file, batch_size=batch_size,
                                                 disease_prediction=True)
    train_input_dataloader, _ = construct_dataloder(tokenizer, train_input_file, train_ehr_file, batch_size=batch_size,
                                                    disease_prediction=True)

    test_cohorts, valid_cohorts, train_cohorts = get_cohorts(
        test_ehr_file, valid_ehr_file, train_ehr_file)

    return vocab_emb, tokenizer, train_dataloader, test_dataloader, valid_cluster_dataloader, test_knn_dataloader, valid_knn_dataloader, \
        test_cohorts, valid_cohorts, train_input_dataloader, train_cohorts
