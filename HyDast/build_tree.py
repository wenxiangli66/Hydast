#https://dexur.com/icd9/ code website


class Voc(object):
    def __init__(self):
        self.idx2word = {}#用于存储索引到单词的映射。
        self.word2idx = {}#用于存储单词到索引的映射。

    #接收一个句子（可以是单词的列表），并将其中的每个单词添加到词汇表中。
    def add_sentence(self,sentence):
        for word in sentence:#遍历句子中的每个单词 word。
            if word not in self.word2idx:#如果不在 word2idx 中，将单词添加到 idx2word 和 word2idx 中
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)#使用当前的 word2idx 长度作为新单词的索引。


def _remove_duplicate(input):
    return list(set(input))
#这段代码定义了一个名为 _remove_duplicate 的函数，用于去除输入列表中的重复元素
#
# _remove_duplicate 函数的作用是去除输入列表中的重复元素。使用 set 数据结构来实现去重时，
# 所有元素都会被视为无序的，因此边的方向信息可能会丢失。




#这段代码定义了一个名为 build_stage_one_edges 的函数，用于构建图的边索引。子节点-父节点
def build_stage_one_edges(res, graph_voc):
    """
    :param res:由多个样本组成的列表，每个样本都是一个节点序列
    :param graph_voc: 包含节点到索引的映射的词汇对象，通常是一个包含 word2idx 字典的实例。
    :return: edge_idx [[1,2,3],[0,1,0]]表示边的起始和结束节点
    """
    edge_idx = []#edge_idx：初始化一个空列表，用于存储图的边。
    for sample in res:
        sample_idx = list(map(lambda x: graph_voc.word2idx[x], sample))
        #对于每个样本，使用 map 函数将节点（字符串）转换为对应的索引，形成 sample_idx 列表
        for i in range(len(sample_idx) - 1):#通过嵌套循环，遍历每个样本中的节点
            # only direct children -> ancestor
            # 仅添加直接子节点到父节点的边
            edge_idx.append((sample_idx[i+1], sample_idx[i]))
            #将相邻节点之间的边添加到 edge_idx 中，表示从子节点到父节点的关系。
            #
            # # self-loop except leaf node
            # if i != 0:
            #     edge_idx.append((sample_idx[i], sample_idx[i]))

    edge_idx = _remove_duplicate(edge_idx)#去重
    row = list(map(lambda x: x[0], edge_idx)) # 提取边的起始节点
    col = list(map(lambda x: x[1], edge_idx))# 提取边的结束节点
    return [row, col] # 返回边的起始和结束节点列表

#这段代码用于构建图的边索引。与前一个函数类似，它的目标是为给定的样本构建边，但这次是从父节点-子节点
def build_stage_two_edges(res, graph_voc):
    """
    :param res:
    :param graph_voc:
    :return: edge_idx [[1,2,3],[0,1,0]]
    """
    edge_idx = []
    for sample in res:
        sample_idx = list(map(lambda x: graph_voc.word2idx[x], sample))
        # only ancestors -> leaf node
        edge_idx.extend([(sample_idx[0], sample_idx[i])#。这里假设 sample_idx[0] 是祖先节点，其余节点是叶子节点。
                         for i in range(1, len(sample_idx))])

    edge_idx = _remove_duplicate(edge_idx)
    row = list(map(lambda x: x[0], edge_idx))
    col = list(map(lambda x: x[1], edge_idx))
    return [row, col]


#在你的 build_ancestors 函数中，目的是构建每个样本的祖先节点和叶节点列表。
def build_ancestors(res, graph_voc):
    """
    :param res:
    :param graph_voc:
    :return: ancestor_nodes: the ancestor for leave nodes;  left_nodes: the leave nodes
    """
    ancestor_nodes,leave_nodes = [],[]#用于存储每个样本的所有节点（即所有祖先节点）。用于存储每个样本的根节点（即叶节点）。
    for sample in res:
        sample_idx = list(map(lambda x: graph_voc.word2idx[x], sample))#对于每个样本，首先将节点转换为它们在词汇中的索引。
        # only ancestors -> leaf node
        ancestor_nodes.append([sample_idx[i] for i in range(0, len(sample_idx))])# 存储样本中的所有节点索引。
        leave_nodes.append([sample_idx[0] for i in range(0, len(sample_idx))])# 存储样本中的根节点（即第一个节点），对于每个样本，根节点重复存储。
    return ancestor_nodes,leave_nodes

#函数的目的是构建图的边索引，包括祖先与子节点之间的关系以及根节点与所有子节点之间的关系。
def build_cominbed_edges(res, graph_voc):
    """
    :param res:
    :param graph_voc:
    :return: edge_idx [[1,2,3],[0,1,0]]
    """
    edge_idx = []#用于存储边的索引，形式为元组 (父节点, 子节点)。
    for sample in res:
        sample_idx = list(map(lambda x: graph_voc.word2idx[x], sample))#对于每个样本，首先将节点转换为它们在词汇中的索引。
        for i in range(len(sample_idx) - 1):
            # ancestor <- direct children
            edge_idx.append((sample_idx[i+1], sample_idx[i]))#记录从子节点到祖先节点的边（ancestor <- direct children）。

            # ancestors -> leaf node
            edge_idx.extend([(sample_idx[0], sample_idx[i])#记录根节点到所有子节点的边（ancestors -> leaf node）。
                             for i in range(1, len(sample_idx))])
            #
            #
            # # self-loop except leaf node
            # if i != 0:
            #     edge_idx.append((sample_idx[i], sample_idx[i]))

    edge_idx = _remove_duplicate(edge_idx)#使用 _remove_duplicate 函数去除重复的边。
    row = list(map(lambda x: x[0], edge_idx))
    col = list(map(lambda x: x[1], edge_idx))
    return [row, col]


# tree order


"""
ICD-9
"""


def expand_level2_diag():
    level2 = ['A00-A09', 'A15-A19', 'A20-A28', 'A30-A49', 'A50-A64', 'A65-A69', 'A70-A74', 'A75-A79', 'A80-A89', 'A92-A99',
              'B00-B09', 'B15-B19', 'B20-B24', 'B25-B34', 'B35-B49', 'B50-B64', 'B65-B83', 'B85-B89', 'B90-B94',
              'B95-B98',
              'B99-B99', 'C00-C97', 'D00-D09', 'D10-D36', 'D37-D48', 'D50-D53', 'D55-D59', 'D60-D64', 'D65-D69', 'D70-D77',
              'D80-D89',
              'E00-E07', 'E10-E14', 'E15-E16', 'E20-E35', 'E40-E46', 'E50-E64', 'E65-E68', 'E70-E90', 'F00-F09', 'F10-F19',
              'F20-F29', 'F30-F39', 'F40-F48', 'F50-F59', 'F60-F69', 'F70-F79', 'F80-F89', 'F90-F98', 'F99-F99', 'G00-G09',
              'G10-G14', 'G20-G26', 'G30-G32', 'G35-G37', 'G40-G47', 'G50-G59', 'G60-G64', 'G70-G73', 'G80-G83',
              'G90-G99',
              'H00-H06', 'H10-H13', 'H15-H22', 'H25-H28', 'H30-H36', 'H40-H42', 'H43-H45', 'H46-H48', 'H49-H52',
              'H53-H54',
              'H55-H59', 'H60-H62', 'H65-H75', 'H80-H83', 'H90-H95', 'I00-I02', 'I05-I09', 'I10-I15', 'I20-I25',
              'I26-I28',
              'I30-I52', 'I60-I69', 'I70-I79', 'I80-I89', 'I95-I99', 'J00-J06', 'J09-J18', 'J20-J22', 'J30-J39',
              'J40-J47',
              'J60-J70', 'J80-J84', 'J85-J86', 'J90-J94', 'J95-J99', 'K00-K14', 'K20-K31', 'K35-K38', 'K40-K46',
              'K50-K52',
              'K55-K64', 'K65-K67', 'K70-K77', 'K80-K87', 'K90-K93', 'L00-L08', 'L10-L14', 'L20-L30', 'L40-L45',
              'L50-L54',
              'L55-L59', 'L60-L75', 'L80-L99', 'M00-M25', 'M30-M36', 'M40-M54', 'M60-M79', 'M80-M94', 'M95-M99',
              'N00-N08',
              'N10-N16', 'N17-N19', 'N20-N23', 'N25-N29', 'N30-N39', 'N40-N51', 'N60-N64', 'N70-N77', 'N80-N98', 'N99-N99', 'O00-O08',
              'O10-O16',
              'O20-O29', 'O30-O48', 'O60-O75', 'O80-O84', 'O85-O92', 'O94-O99', 'P00-P04', 'P05-P08', 'P10-P15',
              'P20-P29', 'P35-P39', 'P50-P61', 'P70-P74', 'P75-P78', 'P80-P83', 'P90-P96', 'Q00-Q07',
              'Q10-Q18', 'Q20-Q28', 'Q30-Q34', 'Q35-Q37', 'Q38-Q45', 'Q50-Q56', 'Q60-Q64', 'Q65-Q79',
              'Q80-Q89', 'Q90-Q99','R00-R09','R10-R19','R20-R23','R25-R29','R30-R39','R40-R46','R47-R49','R50-R69',
              'R70-R79','R80-R82','R83-R89','R90-R94','R95-R99','S00-S09','S10-S19','S20-S29','S30-S39','S40-S49',
              'S50-S59','S60-S69','S70-S79','S80-S89','S90-S99','T00-T07','T08-T14','T15-T19','T20-T32','T33-T35',
              'T36-T50','T51-T65','T66-T78','T79-T79','T80-T88','T90-T98','V01-X59','X60-X84','X85-Y09','Y10-Y34','Y35-Y36',
              'Y40-Y84','Y85-Y89','Y90-Y98','Z00-Z13','Z20-Z29','Z30-Z39','Z40-Z54','Z55-Z65','Z70-Z76','Z80-Z99']

    level2_expand = {}
    for i in level2:
        tokens = i.split('-')
        if i[0] == 'V':
            if len(tokens) == 1:
                level2_expand[i] = i
            else:
                for j in range(int(tokens[0][1:]), int(tokens[1][1:]) + 1):
                    level2_expand["V%02d" % j] = i
        elif i[0] == 'E':
            if len(tokens) == 1:
                level2_expand[i] = i
            else:
                for j in range(int(tokens[0][1:]), int(tokens[1][1:]) + 1):
                    level2_expand["E%03d" % j] = i
        else:
            if len(tokens) == 1:
                level2_expand[i] = i
            else:
                for j in range(int(tokens[0]), int(tokens[1]) + 1):
                    level2_expand["%03d" % j] = i
    return level2_expand

# ontology raw source这段代码的目的是扩展给定的 ICD-9 级别 2 过程编码（level2），
# 将范围内的编码转换为具体的编码，并将每个编码映射到其对应的级别 2 编码。
# http://icd9.chrisendres.com/index.php?action=procslist
def expand_level2_proce():
    level2 = ['00','01-05','06-07','08-16','17','18-20','21-29','30-34','35-39','40-41','42-54','55-59','60-64','65-71','72-75','76-84','85-86','87-99']#为什么要这么分类
    level2_expand = {}#这是一个字典，用于存储扩展后的编码映射。
    for i in level2:#遍历 level2 列表中的每个编码：
        tokens = i.split('-')
        if len(tokens)==1:#如果编码是单一的（例如 '00'），直接将其添加到 level2_expand 字典中。
            level2_expand[i] = i#直接将其添加到 level2_expand 字典中。
        else:
            for j in range(int(tokens[0]),int(tokens[1])+1):
                level2_expand["%02d"%j] = i
                #使用 range 函数生成从起始值到结束值的所有编码，并将它们映射到对应的级别 2 编码。
    return level2_expand

#目的是构建一个医疗诊断的层次树（树结构），并将不同级别的诊断编码进行组织和映射。
def build_diag_tree(unique_codes):
    res = []# 用于存储每个样本的结果
    graph_voc = Voc()  # 创建一个词汇表对象
    unique_codes = [code[2:] for code in unique_codes]# 从每个代码中去除前两个字符

    root_node = 'diag_root'# 定义树的根节点
    level3_dict = expand_level2_diag() # 获取级别 3 的映射
    # print("build diagtree........")
    for code in unique_codes:# 遍历每个唯一的诊断编码
        level1 = code # 级别 1 编码
        level2 = level1[:4] if level1[0] == 'E' else level1[:3]#根据编码的首字符判断并生成 level2。
        level3 = level3_dict[level2] # 从级别 2 映射到级别 3
        level4 = root_node # 级别 4 是根节点
        
        sample = [level1, level2, level3, level4]# 创建样本，包含所有级别的编码
        
        # print(sample)
        # print('\n')
        graph_voc.add_sentence(sample) # 将样本添加到词汇表中
        res.append(sample)# 将样本添加到结果列表中
    return res, graph_voc

def build_proce_tree(unique_codes):
    res = []
    graph_voc = Voc()
    unique_codes = [code[2:] for code in unique_codes] # 从每个代码中去除前两个字符

    root_node = 'proce_root'
    level3_dict = expand_level2_proce()
    # print("build proceTree..........")
    for code in unique_codes:
        level1 = code
        level2 = level1[:2]
        level3 = level3_dict[level2]
        level4 = root_node

        sample = [level1, level2, level3, level4]
        # print(sample)
        # print('\n')
        graph_voc.add_sentence(sample)
        res.append(sample)

    return res, graph_voc


"""
ATC
"""

def build_atc_tree(unique_codes):
    res = []
    graph_voc = Voc()
    unique_codes = [code[2:] for code in unique_codes]
    # print("build ATC...........")
    root_node = 'atc_root'
    for code in unique_codes:
        sample = [code] + [code[:i] for i in [4, 3, 1]] + [root_node]
        # 从编码中提取的子字符串，构成不同级别的编码：
        # code[:4]：表示 ATC 编码的前 4 个字符（通常对应于更具体的类别）。
        # code[:3]：表示前 3 个字符。
        # code[:1]：表示前 1 个字符（通常是大类别）。
        # 最后加上根节点 atc_root。


        # print(sample)
        # print('\n')
        graph_voc.add_sentence(sample)
        res.append(sample)

    return res, graph_voc
