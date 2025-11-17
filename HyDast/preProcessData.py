import torch
torch.cuda.is_available()

import pandas as pd
import numpy as np

#directory for MIMIC-III data
base_dir = 'mimic4_origin/'
# D:\python project\GNN-fourth paper\MHGRL-main\data\mimic4_ori\mimic-iv v2.0\hosp

admission_file = base_dir + 'admissions1.csv'
procedure_file = base_dir + 'procedures_icd.csv'#手术ICD记录表
prescriptions_file = base_dir + 'prescriptions.csv'#处方信息
diagnoses_file = base_dir + 'diagnoses_icd1.csv'
patients_file = base_dir + 'patients1.csv'

admission_df = pd.read_csv(admission_file)

admission_df.columns = admission_df.columns.str.upper()
admission_df.ADMITTIME = pd.to_datetime(admission_df.ADMITTIME,format='%Y-%m-%d %H:%M:%S',errors='coerce')
admission_df = admission_df.sort_values(['SUBJECT_ID','ADMITTIME'])

proce_df = pd.read_csv(procedure_file)
pres_df = pd.read_csv(prescriptions_file,dtype={'ndc':'category'})
#读取处方数据文件，并将 NDC 列的数据类型设为 category。
diag_df = pd.read_csv(diagnoses_file)

MIN_CODE_THRESHOLD = 50#定义最小阈值，用于筛选频繁出现的代码。
MEDIUM_CODE_THRESHOLD = 100
LARGE_CODE_THEESHOLD = 500

#函数根据指定的代码阈值（threshold）筛选出频率高于阈值的代码，并返回相应的数据。
def construct_valid_subset(raw_df,column='icd_code',threshold=MIN_CODE_THRESHOLD,desc='filter desc:'):
    base_df = raw_df[column].value_counts()
#这行代码计算指定列（column）中每个唯一代码的出现次数，
#并返回一个包含代码及其频率的数据框 base_df。
    valid_code = base_df[base_df>=threshold].index.values
    #从 base_df 中筛选出频率大于等于 threshold 的代码，
    #并将这些代码的值存储在 valid_code 数组中。
    filtered_df = raw_df[raw_df[column].isin(valid_code)]
    #只保留 raw_df 中列（column）值在 valid_code 列表中的行，
    #形成新的数据框 filtered_df。
    filtered_admission_ids = set(filtered_df['hadm_id'].tolist())
#从 filtered_df 中提取 HADM_ID（入院编号）列的所有值，并将其转换为一个集合 filtered_admission_ids。
    return valid_code,filtered_admission_ids,filtered_df

diag_codes,diag_admission_ids,diag_df = construct_valid_subset(diag_df,desc='valid diagnoses code base/num: ')
proce_codes,proce_admission_ids,proce_df = construct_valid_subset(proce_df,desc='valid procedure code base/num: ')
pres_codes,pres_admission_ids,pres_df = construct_valid_subset(pres_df,column='ndc',desc='valid prescription code base/num: ')
#函数分别对诊断、手术和处方数据进行处理。
common_admission_ids = diag_admission_ids & proce_admission_ids & pres_admission_ids

print(11111111111111111111111111111111111111111111111)
print(len(common_admission_ids))

diag_df = diag_df[diag_df['icd_version'] == 10]
common_diag_df = diag_df[diag_df['hadm_id'].isin(common_admission_ids)]
proce_df = proce_df[proce_df['icd_version'] == 10]
common_proce_df = proce_df[proce_df['hadm_id'].isin(common_admission_ids)]
common_pres_df = pres_df[pres_df['hadm_id'].isin(common_admission_ids)]
common_admission_df = admission_df[admission_df['HADM_ID'].isin(common_admission_ids)]

common_diag_df.to_csv('mimic4/common_diag_df.csv', index=False)
common_proce_df.to_csv('mimic4/common_proce_df.csv', index=False)
common_pres_df.to_csv('mimic4/common_pres_df.csv', index=False)

print(len(common_admission_ids))


print('donedonedone')


# In[66]:

#common_admission_ids 计算三个数据框中共有的入院编号，
#即这些编号在诊断、手术和处方数据中都有出现。

diag_df = diag_df.groupby(['subject_id','hadm_id']).agg({'icd_code':lambda x:','.join(x)}).reset_index().rename(columns={'icd_code':'ICD_DIAG'})
common_diag_df = diag_df[diag_df['hadm_id'].isin(common_admission_ids)]
proce_df.icd_code = proce_df.icd_code.astype(str)
proce_df = proce_df.groupby(['subject_id','hadm_id']).agg({'icd_code':lambda x:','.join(x)}).reset_index().rename(columns={'icd_code':'ICD_PROCE'})

ndc2rxnorm_file = 'mimic4_origin/ndc2rxnorm_mapping.txt'
with open(ndc2rxnorm_file, 'r') as f:
    ndc2rxnorm = eval(f.read())

pres_df = pres_df[pres_df['ndc'].isin(ndc2rxnorm)].reset_index()

pres_df = pres_df.groupby(['subject_id','hadm_id']).agg({'ndc':lambda x:','.join(x)}).reset_index()

#对 proce_df ，proce_df，pres_df按 SUBJECT_ID 和 HADM_ID 分组。


common_df = pd.merge(common_diag_df,proce_df,on=['subject_id','hadm_id'])
#将 common_diag_df 和 proce_df 按 SUBJECT_ID 和 HADM_ID 合并。
common_df = pd.merge(common_df,pres_df,on=['subject_id','hadm_id'])

common_df = pd.merge(common_df,admission_df,left_on=['subject_id','hadm_id'], right_on=['SUBJECT_ID','HADM_ID'])
print(common_df['ADMITTIME'])

#将合并后的数据框与 pres_df 按 SUBJECT_ID 和 HADM_ID 再次合并，
#形成最终的 common_df，包含了诊断、程序和处方信息。
print('done')


'''
    for each admission, category it to only single visit or the visit can be formulated in a visit sequence
'''
print('patient statics: ')
print(len(common_df.subject_id.unique()))

visit_num_df = common_df[['subject_id','hadm_id']].groupby('subject_id').hadm_id.unique().reset_index()

# 选择 common_df 中的 SUBJECT_ID 和 HADM_ID 列。
# 按 SUBJECT_ID 分组，并使用 unique() 函数获取每个患者的所有入院编号。
# 重置索引并创建一个新的数据框 visit_num_df。

visit_num_df['HADM_ID_LEN'] = visit_num_df['hadm_id'].apply(lambda x:len(x))
#添加新列 HADM_ID_LEN，计算每个患者的入院编号数组的长度，即每个患者的就诊次数。
multi_subjects = visit_num_df[visit_num_df['HADM_ID_LEN']>1].subject_id.unique()
#从 visit_num_df 中筛选出 HADM_ID_LEN 大于 1 的患者，即有多次就诊记录的患者。
# 获取这些患者的 SUBJECT_ID 并计算其数量
print("筛选的病人总数：", len(multi_subjects))
common_multi_df = common_df[common_df['subject_id'].isin(multi_subjects)]

multi_hadms = set(common_multi_df['hadm_id'].tolist())

common_admission_df = common_admission_df[common_admission_df['HADM_ID'].isin(multi_hadms)]

common_admission_df= common_admission_df.sort_values(by=['SUBJECT_ID', 'ADMITTIME'], ascending=[True, True])

common_admission_df.to_csv('mimic4/common_admission_df.csv', index=False)

print(len(common_admission_df))


def ndc2atc(pres_df):
    with open(ndc2rxnorm_file,'r') as f:
        ndc2rxnorm = eval(f.read())
    #pres_df['ATC'] = pres_df['ndc'].map(lambda x:','.join([ndc2rxnorm[ndc] for ndc in x.split(',')]))
    pres_df['ATC'] = pres_df['ndc']
    return pres_df
common_multi_df = ndc2atc(common_multi_df)
common_multi_df= common_multi_df.sort_values(by=['subject_id', 'ADMITTIME'], ascending=[True, True])
common_multi_df.to_csv('mimic4/hame_graph.csv', index=False)

print(len(common_multi_df))


import pickle
all_diag_codes = []
common_multi_df['ICD_DIAG'].apply(lambda x:all_diag_codes.extend(x.split(',')))
all_diag_codes = list(set(all_diag_codes))
print(len(all_diag_codes))
all_proce_codes = []
common_multi_df['ICD_PROCE'].apply(lambda x:all_proce_codes.extend(x.split(',')))
all_proce_codes = list(set(all_proce_codes))
print(len(all_proce_codes))
all_atc_codes = []
common_multi_df['ATC'].apply(lambda x:all_atc_codes.extend(x.split(',')))
all_atc_codes = list(set(all_atc_codes))
print(len(all_atc_codes))

pickle.dump({'diag_codes':all_diag_codes,'proce_codes':all_proce_codes,'atc_codes':all_atc_codes},open('mimic4/vocab.pkl','wb'))

'''
    construct all the knowledge graph with PMI value
    the entity type: diagnose, procedure, prescription
    the relation type: diagnose-procedure, diagnose-prescription, procedure-presciption
'''
from math import log

def construct_ent_pairs(x,head_col,tail_col,all_pairs):
    for head_ent in x[head_col].split(','):
        for tail_ent in x[tail_col].split(','):
            all_pairs.append(head_ent+','+tail_ent)
#从DataFrame的每一行中提取头实体（head_col列）和尾实体（tail_col列）
#，并将它们组合成关系对，添加到 all_pairs 列表中。

'''
    based on the valid pmi value, construct the relation
'''
def construct_relation(common_df,head_col,tail_col):
    all_pairs = []
    common_df.apply(construct_ent_pairs,axis=1,args=(head_col,tail_col,all_pairs))
#     print(len(all_pairs))
    entity_freq = {}
    rel_pair_count = {}
    for rel_pair in all_pairs:
        head_ent,tail_ent = rel_pair.split(',')
        if rel_pair not in rel_pair_count:
            rel_pair_count[rel_pair] = 1
        else:
            rel_pair_count[rel_pair]+=1
        if head_ent not in entity_freq:
            entity_freq[head_ent] = 1
        else:
            entity_freq[head_ent]+=1
        if tail_ent not in entity_freq:
            entity_freq[tail_ent] = 1
        else:
            entity_freq[tail_ent]+=1

    num_windows = len(all_pairs)
    pmi_result = []
    for rel_pair in rel_pair_count:
        entities = rel_pair.split(',')
        pmi = log((1.0*rel_pair_count[rel_pair]/num_windows)/(1.0*entity_freq[entities[0]]*entity_freq[entities[1]]/(num_windows*num_windows)))
        if pmi<0:continue#仅保留PMI值大于0的实体对。
        pmi_result.append([entities[0],entities[1],pmi])
    return pmi_result
#计算不同类型实体对之间的PMI值
#PMI用于衡量两个实体共同出现的概率与它们各自出现的概率之间的关系。


def write_relation(pmi_result,output_file):
    with open(output_file,'w',encoding='utf-8') as writer:
        writer.write('head ent'+'\t'+'tail ent'+'\t'+'pmi\n')
        for key in pmi_result:
            writer.write(key[0]+'\t'+key[1]+'\t'+str(key[2])+'\n')
    print('relation file writing done...')
#将PMI结果写入文件

diag_proce_rel = construct_relation(common_multi_df,'ICD_DIAG','ICD_PROCE')
print('diagnose and procedure relation num: ',len(diag_proce_rel))
#计算诊断与手术之间的关系
diag_pres_rel = construct_relation(common_multi_df,'ICD_DIAG','ndc')
print('diagnose and prescription relation num: ',len(diag_pres_rel))
#计算诊断与处方之间的关系
proce_pres_rel = construct_relation(common_multi_df,'ICD_PROCE','ndc')
print('procedure and presciption relation num: ',len(proce_pres_rel))
#计算程序与处方之间的关系。
write_relation(diag_proce_rel,'mimic4/diag_proce_rel.csv')
write_relation(diag_pres_rel,'mimic4/diag_pres_rel.csv')
write_relation(proce_pres_rel,'mimic4/proce_pres_rel.csv')
#将这些关系及其PMI值写入到CSV文件中

print("数据处理完毕")

