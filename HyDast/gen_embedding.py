from sentence_transformers import SentenceTransformer
import time
import dill
import pickle

def read_pkl(path):
  with open(path, 'rb') as f:
    data = dill.load(f)
  return data

start = time.time()
# 加载预训练的Sentence-BERT模型
model = SentenceTransformer('all-mpnet-base-v2')

def get_embedding(text):
    return model.encode(text)

atc_sentence = read_pkl('mimic4/atc_sentence.pkl')
atc_embs = {}
for word, sentence in atc_sentence.items():
    text = sentence
    embedding = get_embedding(text)
    atc_embs[word] = embedding
print(len(atc_embs))

pickle.dump(atc_embs, open('mimic4/atc_embs.pkl', 'wb'))


pro_sentence = read_pkl('mimic4/pro_sentence.pkl')
pro_embs = {}
for word, sentence in pro_sentence.items():
    text = sentence
    embedding = get_embedding(text)
    pro_embs[word] = embedding
print(len(pro_embs))
pickle.dump(pro_embs, open('mimic4/pro_embs.pkl', 'wb'))
num_words = len(pro_embs)
embedding_dim = pro_embs[list(pro_embs.keys())[0]].shape[0]  # 假设嵌入是1维数组或向量
print(f"单词的数量: {num_words}")
print(f"嵌入的维度: {embedding_dim}")



diag_sentence = read_pkl('mimic4/diag_sentence.pkl')
diag_embs = {}
for word, sentence in diag_sentence.items():
    text = sentence
    embedding = get_embedding(text)
    diag_embs[word] = embedding
print(len(diag_embs))
pickle.dump(diag_embs, open('mimic4/diag_embs.pkl', 'wb'))