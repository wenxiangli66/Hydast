import numpy as np
import pandas as pd
from process_mimic_iv_v1 import *
from sklearn.metrics.pairwise import cosine_similarity
# ... （之前的代码不变）
pids = pickle.load(open("mimic4/output_filename.pids", "rb"))
new_label = pickle.load(open("mimic4/output_filename.4digit_label.seqs", "rb"))
# 将 pids 和 new_label 转换为 DataFrame
# df = pd.DataFrame(data=new_label, index=pids, columns=['Disease_' + str(i) for i in range(19)])
columns = list(range(22))
df = pd.DataFrame(columns=columns)
_TEST_RATIO=0.3

# 添加偏移值（例如，偏移值为2）
for i, row in enumerate(new_label):
    df.loc[i, row] = 1
    # df[i,row]=1
# 显示DataFrame

# print(new_label.shape())


df['patient id'] = pids
df.set_index('patient id', inplace=True)
# 显示结果
# print(df_result)

# 保存 DataFrame 到 CSV 文件
nTest = int(_TEST_RATIO * len(new_label))
cos_test_rows = df.iloc[:nTest]
cos_train_rows=df.iloc[nTest:]

df_train_result = cos_train_rows.fillna(0).astype(int)
df_test_result = cos_test_rows.fillna(0).astype(int)
cosine_train_sim = cosine_similarity(df_train_result)
cosine_test_sim = cosine_similarity(df_test_result)

np.save("mimic4/cos_train.npy", cosine_train_sim)
cosine_sim_argsort = np.argsort(cosine_train_sim, axis=1)
np.save("mimic4/cos_train_max_thress_index.npy", cosine_sim_argsort[:, -3:])

np.save("mimic4/cos_test.npy", cosine_test_sim)
cosine_sim_argsort = np.argsort(cosine_test_sim, axis=1)
np.save("mimic4/cos_test_max_thress_index.npy", cosine_sim_argsort[:, -3:])