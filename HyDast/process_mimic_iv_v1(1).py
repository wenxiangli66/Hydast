# This script processes MIMIC-III dataset and builds longitudinal diagnosis records for patients with at least two visits.
# The output data are cPickled, and suitable for training Doctor AI or RETAIN
# Written by Edward Choi (mp2893@gatech.edu)
# Usage: Put this script to the foler where MIMIC-III CSV files are located. Then execute the below command.
# python process_mimic.py ADMISSIONS.csv DIAGNOSES_ICD.csv PATIENTS.csv <output file>

# Output files
# <output file>.pids: List of unique Patient IDs. Used for intermediate processing
# <output file>.morts: List of binary values indicating the mortality of each patient
# <output file>.dates: List of List of Python datetime objects. The outer List is for each patient. The inner List is for each visit made by each patient
# <output file>.seqs: List of List of List of integer diagnosis codes. The outer List is for each patient. The middle List contains visits made by each patient. The inner List contains the integer diagnosis codes that occurred in each visit
# <output file>.types: Python dictionary that maps string diagnosis codes to integer diagnosis codes.

import sys
# import cPickle as pickle
import pickle
import os
import sys
from datetime import datetime

def convert_to_icd10(dxstr):
    """
    Put a period in the right place because the MIMIC-3 data files exclude them.
    Generally, procedure codes have dots after the first two digits,
    while diagnosis codes have dots after the first three digits.
    """

    """
    Format ICD-10 code to insert a period after the third character.
    """
    return dxstr[:3] + "." + dxstr[3:]


def convert_to_icd9(dxStr):  # EABCDEEFG   EABCD.EEFG    EABCD
    if dxStr.startswith('E'):
        if len(dxStr) > 4:
            return dxStr[:4] + '.' + dxStr[4:]
        else:
            return dxStr
    else:
        if len(dxStr) > 3:
            return dxStr[:3] + '.' + dxStr[3:]
        else:
            return dxStr


def convert_to_3digit_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4:
            return dxStr[:4]

        else:
            return dxStr
    else:
        if len(dxStr) > 3:
            return dxStr[:3]
        else:
            return dxStr


def convert_to_3digit_icd10(dxStr):
    return dxStr[:3]


if __name__ == '__main__':
    admissionFile = 'mimic4/common_admission_df.csv'
    diagnosisFile = 'mimic4/common_diag_df.csv'
    patientsFile = 'mimic4_origin/patients1.csv'
    project_path = os.getcwd()

    # 指定输出文件的名称（不含文件扩展名）
    output_file_name = "output_filename"
    output_folder = "mimic4"
    # 构建输出文件的完整路径
    output_directory = os.path.join(project_path, output_folder)
    os.makedirs(output_directory, exist_ok=True)  # 确保目录存在或创建目录

    outFile = os.path.join(output_directory, output_file_name)  # 输出文件路径

    print('Collecting mortality information')
    pidDodMap = {}
    infd = open(patientsFile, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[0])
        # patient ID
        dod_hosp = tokens[5]
        # dead time
        if len(dod_hosp) > 0:
            pidDodMap[pid] = 1
        else:
            pidDodMap[pid] = 0
    infd.close()

    print('Building pid-admission mapping, admission-date mapping')
    pidAdmMap = {}
    admDateMap = {}
    infd = open(admissionFile, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[0])
        # patient ID
        admId = int(tokens[1])
        # bing an ID
        admTime = datetime.strptime(tokens[2], '%Y-%m-%d %H:%M:%S')
        # ru yuan time
        admDateMap[admId] = admTime
        if pid in pidAdmMap:
            pidAdmMap[pid].append(admId)
        else:
            pidAdmMap[pid] = [admId]
    infd.close()

    print('Building admission-dxList mapping')

    admDxMap = {}
    admDxMap_3digit = {}
    infd = open(diagnosisFile, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        admId = int(tokens[1])
        icd_version = tokens[4]
        # bing an ID
        if icd_version == '10':
            dxStr = convert_to_icd10(
                tokens[3])  # Uncomment this line and comment the line below, if you want to use the entire ICD9 digits.
            dxStr_3digit = convert_to_3digit_icd10(tokens[3])
            # shi ji bian ma IDC-9
            if admId in admDxMap:
                admDxMap[admId].append(dxStr)

            else:
                admDxMap[admId] = [dxStr]

            if admId in admDxMap_3digit:
                admDxMap_3digit[admId].append(dxStr_3digit)
            else:
                admDxMap_3digit[admId] = [dxStr_3digit]
    infd.close()
    print('adm长度:',len(admDateMap))

    print('Building pid-sortedVisits mapping')
    pidSeqMap = {}
    pidSeqMap_3digit = {}
    for pid, admIdList in pidAdmMap.items():
        if len(admIdList) < 2:
            continue  # jump,Ensure that only patients with multiple admissions records are included.

        sortedList = sorted([(admDateMap[admId], admDxMap[admId]) for admId in admIdList if admId in admDxMap])
        pidSeqMap[pid] = sortedList

        sortedList_3digit = sorted(
            [(admDateMap[admId], admDxMap_3digit[admId]) for admId in admIdList if admId in admDxMap_3digit])
        pidSeqMap_3digit[pid] = sortedList_3digit

    print('Building pids, dates, mortality_labels, strSeqs')
    pids = []
    dates = []
    seqs = []
    morts = []
    for pid, visits in pidSeqMap.items():
        pids.append(pid)
        morts.append(pidDodMap[pid])
        seq = []
        date = []
        for visit in visits:
            date.append(visit[0])
            seq.append(visit[1])
        dates.append(date)
        seqs.append(seq)

    print('Building pids, dates, strSeqs for 3digit ICD9 code')
    seqs_3digit = []
    for pid, visits in pidSeqMap_3digit.items():
        seq = []
        for visit in visits:
            seq.append(visit[1])
        seqs_3digit.append(seq)

    print('Converting strSeqs to intSeqs, and making types')
    types = {}
    newSeqs = []
    for patient in seqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                if code in types:
                    newVisit.append(types[code])
                else:
                    types[code] = len(types)
                    newVisit.append(types[code])
            newPatient.append(newVisit)
        newSeqs.append(newPatient)

    print('Converting strSeqs to intSeqs, and making types for 3digit ICD9 code')
    types_3digit = {}
    newSeqs_3digit = []

    remove_id = []
    for id, patient in enumerate(seqs_3digit):
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in set(visit):
                if code in types_3digit:
                    newVisit.append(types_3digit[code])
                else:
                    types_3digit[code] = len(types_3digit)
                    newVisit.append(types_3digit[code])
            newPatient.append(newVisit)
        if len(newPatient) < 2:
            remove_id.append(id)
        newSeqs_3digit.append(newPatient)
    # {“A":[..., ..., ], "B":2....}

    # all_dis_info = {}
    # for id, patient_2 in enumerate(seqs_3digit):
    #     if len(patient_2) > 0:
    #         last_info = patient_2[-1]
    #         added_key = set()
    #         for last_info_i in last_info:
    #             if last_info_i[0] not in added_key:
    #                 if last_info_i[0] not in all_dis_info  :
    #                     all_dis_info[last_info_i[0]] = [newSeqs_3digit[id]]
    #                     added_key.add(last_info_i[0])
    #                 else:
    #                     all_dis_info[last_info_i[0]].append(newSeqs_3digit[id])
    #
    # all_dis_info2 = dict(sorted(all_dis_info.items()))
    # all_dis_info2_label = {}
    # for i, (key, value)  in enumerate(all_dis_info2.items()):
    #     all_dis_info2_label[key] = [i for _ in range(len(all_dis_info2[key]))]

    pids_indices_to_remove = []
    new_label = []
    label_info = {
        'A': 0,
        'B':0,
        'C': 1,
        'D': 2,
        'E': 3,
        'F': 4,
        'G': 5,
        'H': 7,
        'I': 8,
        'J': 9,
        'K': 10,
        'L': 11,
        'M': 12,
        'N': 13,
        'O': 14,
        'P': 15,
        'Q': 16,
        'R': 17,
        'S': 18,
        'U': 21,
        'Z': 20,
         'T':18,
        'V':19,
         'W':19,
          'X':19,
          'Y':19}


    for id, patient_2 in enumerate(seqs_3digit):
        if len(patient_2) > 0:
            last_info = patient_2[-1]
            added_key = []
            for last_info_i in last_info:
                # print(last_info_i)
                # if last_info_i[0] == "B":
                #     added_key.append(0)
                if "D00" <= last_info_i < "D49":
                    added_key.append(1)
                if "H00" <= last_info_i < "H60":
                    added_key.append(6)
                # if "T00" <= last_info_i < "T99":
                #     added_key.append(18)
                # if "V00" < last_info_i < "Y99":
                #     added_key.append(19)
                else:
                    added_key.append(label_info[last_info_i[0]])
            new_label.append(list(set(added_key)))
        else:
            new_label.append([])
        #     pids_indices_to_remove.append(id)

    # pids = [pids[i] for i in range(len(pids)) if i not in pids_indices_to_remove]
    print("----")
    for id in sorted(remove_id, reverse=True):
        del morts[id]
        del newSeqs_3digit[id]
        del new_label[id]
        del pids[id]

    non_empty_counts = [sum(1 for sublist in sublist_list if sublist) for sublist_list in newSeqs_3digit]

    # 计算总的非空子列表数量
    total_non_empty_count = sum(non_empty_counts)

    print("非空子列表的数量:", non_empty_counts)
    print("总的非空子列表数量:", total_non_empty_count)


    # 要查找的数字范围
    start_num = 0
    end_num = 21

    # 初始化存储结果的字典
    counts = {number: 0 for number in range(start_num, end_num + 1)}

    # 循环遍历要查找的数字范围
    for number in range(start_num, end_num + 1):
        # 计算包含当前数字的子列表个数
        count = sum(1 for sublist in new_label if number in sublist)
        # 将结果存储到字典中
        counts[number] = count

    # 打印结果
    for number, count in counts.items():
        print(f"包含{number}的子列表个数:", count)





    pickle.dump(pids, open(outFile + '.pids', 'wb'), -1)#病人病号
    pickle.dump(dates, open(outFile + '.dates', 'wb'), -1)
    pickle.dump(morts, open(outFile + '.morts', 'wb'), -1) #
    pickle.dump(newSeqs, open(outFile + '.seqs', 'wb'), -1)
    pickle.dump(types, open(outFile + '.types', 'wb'), -1)
    pickle.dump(newSeqs_3digit, open(outFile + '.3digitICD9.seqs', 'wb'), -1) #
    pickle.dump(types_3digit, open(outFile + '.3digitICD9.types', 'wb'), -1)
    # pickle.dump(all_dis_info2, open(outFile + '.4digit_info.seqs', 'wb'), -1)
    pickle.dump(new_label, open(outFile + '.4digit_label.seqs', 'wb'), -1)#每个病人所患疾病

    print(len(new_label))

#     label_info = {chr(i + 65): i for i in range(26)}
# disease_categories = {
#     "A": 0,
#     "B": 1,
#     "C": 2,
#     # ... 添加剩余的疾病类别
# }
# new_label = []
# for id, patient_2 in enumerate(seqs_3digit):
#     if len(patient_2) > 0:
#         last_info = patient_2[-1]
#         added_key = [0] * len(disease_categories)
#         for last_info_i in last_info:
#             disease_code = last_info_i[0]
#             if disease_code in disease_categories:
#                 added_key[disease_categories[disease_code]] = 1
#         new_label.append(added_key)
# import pandas as pd
#
# # 将数据转换为 Pandas DataFrame
# df = pd.DataFrame(data=new_label, index=pids, columns=list(disease_categories.keys()))
#
# # 保存 DataFrame 到 CSV 文件
# df.to_csv('disease_information_table.csv')
