
#basic process_mimic for retain in mimicIII
import sys
#import cPickle as pickle
import os
from datetime import datetime
import pickle
import pandas as pd
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


if __name__ == '__main__':
    admissionFile = 'mimic4/common_admission_df.csv'
    diagnosisFile = 'mimic4/common_diag_df.csv'
    patientsFile = 'mimic4/patients1.csv'
    project_path = os.getcwd()

    # 指定输出文件的名称（不含文件扩展名）
    output_file_name = "output_filename"
    output_folder = "output"
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
        if icd_version == '9':
            dxStr = convert_to_icd9(
                tokens[3])  # Uncomment this line and comment the line below, if you want to use the entire ICD9 digits.
            dxStr_3digit = convert_to_3digit_icd9(tokens[3])
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
        newSeqs_3digit.append(newPatient)
    new_label = []
    # label_info = {
    #     'A': 0,
    #     'B': 0,
    #     'C': 1,
    #     'D': 2,
    #     'E': 3,
    #     'F': 4,
    #     'G': 5,
    #     'H': 7,
    #     'I': 8,
    #     'J': 9,
    #     'K': 10,
    #     'L': 11,
    #     'M': 12,
    #     'N': 13,
    #     'O': 14,
    #     'P': 15,
    #     'Q': 16,
    #     'R': 17,
    #     'S': 18,
    #     'U': 21,
    #     'Z': 20,
    #     'T': 18,
    #     'V': 19,
    #     'W': 19,
    #     'X': 19,
    #     'Y': 19}

    for id, patient_2 in enumerate(seqs_3digit):
        if len(patient_2) > 0:
            last_info = patient_2[-1]
            added_key = []
            for last_info_i in last_info:
                # print(last_info_i)
                # if last_info_i[0] == "B":
                #     added_key.append(0)
                if "001" <= last_info_i < "139":
                    added_key.append(1)
                if "140" <= last_info_i < "239":
                    added_key.append(2)
                if "240" <= last_info_i < "279":
                    added_key.append(3)
                if "280" < last_info_i < "289":
                    added_key.append(4)
                if "290" <= last_info_i < "319":
                    added_key.append(5)
                if "320" <= last_info_i < "389":
                    added_key.append(6)
                if "390" <= last_info_i < "459":
                    added_key.append(7)
                if "460" < last_info_i < "519":
                    added_key.append(8)
                if "520" <= last_info_i < "579":
                    added_key.append(9)
                if "580" <= last_info_i < "629":
                    added_key.append(10)
                if "630" <= last_info_i < "679":
                    added_key.append(11)
                if "680" < last_info_i < "709":
                    added_key.append(12)
                if "710" <= last_info_i < "739":
                    added_key.append(13)
                if "740" <= last_info_i < "759":
                    added_key.append(14)
                if "760" <= last_info_i < "779":
                    added_key.append(15)
                if "780" < last_info_i < "799":
                    added_key.append(16)
                if "800" <= last_info_i < "999":
                    added_key.append(17)
                if "v01" <= last_info_i < "v91":
                    added_key.append(18)
                if "E000" <= last_info_i < "E999":
                    added_key.append(19)
                # if "280" < last_info_i < "289":
                #     added_key.append(4)
            new_label.append(list(set(added_key)))
        else:
            new_label.append([])

    pickle.dump(pids, open(outFile + '.pids', 'wb'), -1)  # 病人病号
    pickle.dump(dates, open(outFile + '.dates', 'wb'), -1)
    pickle.dump(morts, open(outFile + '.morts', 'wb'), -1)  #
    pickle.dump(newSeqs, open(outFile + '.seqs', 'wb'), -1)
    pickle.dump(types, open(outFile + '.types', 'wb'), -1)

    pickle.dump(newSeqs_3digit, open(outFile + '.3digitICD9.seqs', 'wb'), -1)  #
    pickle.dump(types_3digit, open(outFile + '.3digitICD9.types', 'wb'), -1)
    # pickle.dump(all_dis_info2, open(outFile + '.4digit_info.seqs', 'wb'), -1)
    pickle.dump(new_label, open(outFile + '.4digit_label.seqs', 'wb'), -1)  # 每个病人所患疾病
    for i in range(len(new_label)):
      print(new_label[i])
