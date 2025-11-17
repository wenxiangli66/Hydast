from openai import OpenAI
import pickle
import numpy as np
import pandas as pd
import dill

OpenAI.api_timeout = 60

client = OpenAI(
  base_url="https://api.feidaapi.com/v1",
  api_key="sk-mcQlh0tHGOl9dMIxOV3IAuDnDcHNBKjYflABbddblwulgh1K"
)

'''
atc_df = pd.read_csv('chatgpt_desc/ndc2atc_level4.csv')
#使用 open 打开指定的 CSV 文件，并利用 Pandas 的 read_csv 方法读取数据，存储为 DataFrame rxnorm2atc。
atc_df = atc_df['ATC4']#删除不需要的列 YEAR、MONTH 和 NDC。
atc_df = atc_df.unique()#基于 RXCUI 列去重，以确保每个 RXCUI 只出现一次。
print(len(atc_df))

atc_sentence = {}
for i in range(len(atc_df)):
  print(i)
  content = "Please define briefly in one paragraph ATC:" + str(atc_df[i])
  completion = client.chat.completions.create(
    model="gpt-4o",
    store=True,
    messages=[
      {"role": "user", "content": content}
    ]
  )
  name = 'a_' + str(atc_df[i])
  atc_sentence[name] = completion.choices[0].message.content
pickle.dump(atc_sentence, open('mimic4/atc_sentence.pkl', 'wb'))
print(atc_sentence)
'''


def read_pkl(path):
  with open(path, 'rb') as f:
    data = dill.load(f)
  return data

embs = read_pkl('mimic4/id2word.pkl')
print(len(embs))

diag_sentence = {}
pro_sentence = {}
for idx, word in embs.items():
  if word[0] == 'd':
    content = "Please define briefly in one paragraph ICD_10:" + str(word[2:])
    completion = client.chat.completions.create(
      model="gpt-4o",
      store=True,
      messages=[
        {"role": "user", "content": content}
      ]
    )
    name = 'd_' + str(word[2:])
    diag_sentence[name] = completion.choices[0].message.content
    print(diag_sentence[name])
  if word[0] == 'p':
    content = "Please define briefly in one paragraph ICD_10:" + str(word[2:])
    completion = client.chat.completions.create(
      model="gpt-4o",
      store=True,
      messages=[
        {"role": "user", "content": content}
      ]
    )
    name = 'p_' + str(word[2:])
    pro_sentence[name] = completion.choices[0].message.content
    print(pro_sentence[name])

pickle.dump(diag_sentence, open('mimic4/diag_sentence.pkl', 'wb'))
pickle.dump(pro_sentence, open('mimic4/pro_sentence.pkl', 'wb'))


'''
diag_df = pd.read_csv('chatgpt_desc/diagnoses_icd1.csv')
diag_df = diag_df[diag_df['icd_version'] == 10]
diag_df = diag_df['icd_code']#删除不需要的列 YEAR、MONTH 和 NDC。
diag_df = diag_df.unique()#基于 RXCUI 列去重，以确保每个 RXCUI 只出现一次。
print(len(diag_df))


for i in range(len(diag_df)):
  print(i)
  
pickle.dump(diag_sentence, open('mimic4/diag_sentence.pkl', 'wb'))
print(diag_sentence)

pro_df = pd.read_csv('chatgpt_desc/procedures_icd.csv')
pro_df = pro_df[pro_df['icd_version'] == 10]

pro_df = pro_df['icd_code']#删除不需要的列 YEAR、MONTH 和 NDC。
pro_df = pro_df.unique()#基于 RXCUI 列去重，以确保每个 RXCUI 只出现一次。
print(len(pro_df))

pro_sentence = {}
for i in range(len(pro_df)):
  print(i)
  content = "Please define briefly in one paragraph ICD_10:" + str(pro_df[i])
  completion = client.chat.completions.create(
    model="gpt-4o",
    store=True,
    messages=[
      {"role": "user", "content": content}
    ]
  )
  name = 'p_' + str(pro_df[i])
  pro_sentence[name] = completion.choices[0].message.content
pickle.dump(pro_sentence, open('mimic4/pro_sentence.pkl', 'wb'))
print(pro_sentence)
'''





