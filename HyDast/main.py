#读入数据
import numpy
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
import random
from sklearn.metrics import accuracy_score
manualSeed = 1
random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import  argparse
from genGraph import gen_Graph
from torch_geometric.data import Data
# from bas import *
from sklearn.decomposition import PCA
#import umap.umap_ as umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from EHR_HyDST_model import *

# 这么设置使用确定性算法，如果代码中有算法cuda没有确定性实现，则会报错，可以验证代码中有没有cuda没有确定性实现的代码
# torch.use_deterministic_algorithms(True)
# 这么设置使用确定性算法，如果代码中有算法cuda没有确定性实现，也不会报错
torch.use_deterministic_algorithms(True, warn_only=True)

# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
_TEST_RATIO = 0.3
# _VALIDATION_RATIO = 0.1

def one_hot(samples):
    labels = list(set(label for sample in samples for label in sample))
    # Create a dictionary to map labels to indices
    label_to_index = {label: i for i, label in enumerate(labels)}
    # Convert samples to one-hot encoding
    one_hot_samples = []
    for sample in samples:
        one_hot_sample = [0] * len(labels)
        for label in sample:
            one_hot_sample[label_to_index[label]] = 1
        one_hot_samples.append(one_hot_sample)
    # Convert the one-hot samples to a PyTorch tensor
    tensor_samples = torch.tensor(one_hot_samples, dtype=torch.float)
    return tensor_samples

def delete_null(data,lable,graph):
    valid_indexes = [i for i, sublist in enumerate(data) if len(sublist) > 0]
    data = [data[i] for i in valid_indexes]
    lable = [lable[i] for i in valid_indexes]
    graph = [graph[i] for i in valid_indexes]

    return np.array(data), np.array(lable), np.array(graph)
def load_data_simple(seqFile, labelFile, timeFile=''):
    sequences = np.array(pickle.load(open(seqFile, 'rb')))  # 加载序列数据
    labels = np.array(pickle.load(open(labelFile, 'rb')))  # 加载标签数据
    seqGraph, vac_emb = gen_Graph('mimic4')
    print(seqGraph[0])
    labels2 = np.array(pickle.load(open("mimic4/output_filename.4digit_label.seqs", "rb")))
    # labels2 = np.array(pickle.load(open("./output/output_filename.4digit_label.seqs", "rb")))

    labels2 = one_hot(labels2)
    sequences, labels, seqGraph = delete_null(sequences, labels, seqGraph)

    if len(timeFile) > 0:
        times = np.array(pickle.load(open(timeFile, 'rb')))

    dataSize = len(labels)
    np.random.seed(0)
    nid = np.random.permutation(dataSize)  # 对数据进行随机排列的索引数组，以便后续划分数据集
    nTest = int(_TEST_RATIO * dataSize)  # 计算测试集和验证集的样本数量。
    # nValid = int(_VALIDATION_RATIO * dataSize)

    test_indices = nid[:nTest]  # 使用索引数组划分测试集、验证集和训练集。
    # valid_indices = nid[nTest:nTest + nValid]
    train_indices = nid[nTest:]

    train_set_x = sequences[train_indices]  # 通过索引获取对应数据集的序列和标签。
    train_set_y = labels[train_indices]
    train_graph = seqGraph[train_indices]
    test_set_x = sequences[test_indices]
    test_set_y = labels[test_indices]
    test_graph = seqGraph[test_indices]
    # valid_set_x = sequences[valid_indices]
    # valid_set_y = labels[valid_indices]
    train_set_t = None
    valid_set_t = None
    test_set_t = None


    train_set_y_2 = labels2[train_indices]
    test_set_y_2= labels2[test_indices]
    # valid_set_y_2 = labels2[valid_indices]


    if len(timeFile) > 0:
        train_set_t = pickle.load(open(timeFile + '.train', 'rb'))
        valid_set_t = pickle.load(open(timeFile + '.valid', 'rb'))
        test_set_t = pickle.load(open(timeFile + '.test', 'rb'))

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    # 使用 len_argsort 函数对训练集、验证集和测试集中的序列数据进行排序
    train_sorted_index = len_argsort(train_set_x)
    train_set_x = [train_set_x[i] for i in train_sorted_index]
    train_graph = [train_graph[i] for i in train_sorted_index]
    train_set_y = [train_set_y[i] for i in train_sorted_index]
    # 使用 train_sorted_index 对训练集的输入数据 train_set_x 进行重新排序
    # train_set_x,train_set_y=delete_null(train_set_x, train_set_y)
    train_set_y_2 = [train_set_y_2[i] for i in train_sorted_index]

    # valid_sorted_index = len_argsort(valid_set_x)
    # valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    # valid_set_y = [valid_set_y[i] for i in valid_sorted_index]
    # valid_set_y_2 = [valid_set_y_2[i] for i in valid_sorted_index]


    test_sorted_index = len_argsort(test_set_x)
    test_set_x = [test_set_x[i] for i in test_sorted_index]
    test_graph = [test_graph[i] for i in test_sorted_index]
    test_set_y = [test_set_y[i] for i in test_sorted_index]
    test_set_y_2 = [test_set_y_2[i] for i in test_sorted_index]

    # test_set_x, test_set_y = delete_null(test_set_x, test_set_y)
    if len(timeFile) > 0:
        train_set_t = [train_set_t[i] for i in train_sorted_index]
        # valid_set_t = [valid_set_t[i] for i in valid_sorted_index]
        test_set_t = [test_set_t[i] for i in test_sorted_index]

    train_set = (train_set_x, train_graph, train_set_y, train_set_y_2,train_set_t)
    # valid_set = (valid_set_x, valid_set_y, valid_set_y_2,valid_set_t)
    test_set = (test_set_x, test_graph, test_set_y, test_set_y_2,test_set_t)

    return train_set,  test_set, vac_emb


def padMatrixWithoutTime(seqs, inputDimSize=1613):  # inputdimsize这儿设的值是错的

    length = np.array([len(seq) for seq in seqs]).astype('int32')
    maxlen = np.max(length)

    # 计算每个序列的长度，并将长度转换为整数类型，存储在 lengths 中。
    n_samples = len(seqs)

    x = np.zeros((maxlen, n_samples, inputDimSize))  # 函数创建一个全零张量，用于存储填充后的序列数据
    for idx, seq in enumerate(seqs):
        for xvec, subseq in zip(x[:, idx, :], seq):
            xvec[subseq] = 1.
    # [20 30 40 ]  [0 00 0 1 0 0 00 -1 ----00 010]
    return x, length


# 通过遍历序列数据 seqs，将每个子序列中的整数值用 one-hot 编码的形式在 x 中标记为 1。

class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.seq = self.dataset[0]
        self.graphs = self.dataset[1]
        self.label = self.dataset[2]
        self.label2=self.dataset[3]

        # self.cos_index = np.load(cos_index_path)
    # __getitem__方法也是一个特殊方法，用于按索引获取数据
    def __getitem__(self, item):
        return self.seq[item], self.label[item], self.label2[item], self.graphs[item]

    # __len__ 方法同样是一个特殊方法，在需要确定数据集大小时被调用
    def __len__(self):
        return len(self.seq)


def my_collate_fun1(batch):
    texts, labels, label2, graph= zip(*batch)
    new_texts, length = padMatrixWithoutTime(texts)
    new_texts = torch.tensor(new_texts).float().permute(1, 0, 2)
    label2 = torch.stack(label2)
    # new_cos_index = torch.tensor(cos_index)
    return torch.tensor(new_texts).float(), torch.tensor(labels).float(), torch.tensor(label2).float(), torch.tensor(length), graph

trainSet,testSet, vac_emb = load_data_simple("mimic4/output_filename.3digitICD9.seqs", "mimic4/output_filename.morts", timeFile="")
# trainSet, testSet = load_data_simple("./output/output_filename.3digitICD9.seqs", "./output/output_filename.morts", timeFile="")

train_dataset = MyDataset(trainSet)
test_dataset = MyDataset(testSet)
print(len(train_dataset))

x = torch.zeros((len(train_dataset), 768))
for i in range(len(train_dataset)):
    graphs = train_dataset.graphs[i]
    graph_feature = torch.zeros((1, 768))
    for j in range(len(graphs)):
        graph = graphs[j]
        graph_x = vac_emb[graph.x]
        graph_emb = mean_p(graph_x, graph.batch).squeeze(dim=1)
        graph_feature = graph_feature + graph_emb
    x[i] = graph_feature / len(graphs)

print(x)


#plot_data, length = padMatrixWithoutTime(train_dataset.seq)
averaged_features = x
labels = train_dataset.label
print(len(labels))

print(len(averaged_features))
# 数据标准化
scaler = StandardScaler()
normalized_features = scaler.fit_transform(averaged_features)
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(normalized_features)
colors = ['#F94144', '#277DA1']
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=[colors[label] for label in labels], cmap='viridis', alpha=0.7)
# plt.xlabel("t-SNE Component 1")
# plt.ylabel("t-SNE Component 2")
# plt.title("t-SNE Visualization of High-Dimensional Data")
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.box(False)  # 去掉边框
plt.tight_layout(pad=0)
plt.savefig('original.png', dpi=600)

plt.show()
plt.close()

# # UMAP降维至2维
# reducer = PCA(n_components=2, random_state=42)
# embedding = reducer.fit_transform(normalized_features)
#
# # 可视化并根据标签染色
# colors = ['#F94144', '#277DA1']
#
# plt.figure(figsize=(8, 6),dpi=300)
# plt.scatter(embedding[:, 0], embedding[:, 1], color=[colors[label] for label in labels], s=5)
#
# plt.title("PCA Visualization of Original Features")
# plt.xlabel("PCA Dimension 1")
# plt.ylabel("PCA Dimension 2")
# plt.savefig('original.png', dpi=600)
# plt.show()
# plt.close()

a = train_dataset[0][0]
b = train_dataset[1]
c = train_dataset[2]
train_dataloader = DataLoader(train_dataset, batch_size=100, collate_fn=my_collate_fun1, shuffle=False, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=100, collate_fn=my_collate_fun1, shuffle=False, drop_last=True)
#
# for i, (data, label, label2, _) in enumerate(train_dataloader):
    # print(label2)
# i是批次序号，而 data 则是从 train_dataloader 中获取的一个批次数据，包含了填充后的文本数据、对应的标签数据以及文本长度信息

#统计数据集中每个病人，最后一个病历数据中出现的疾病数量以及对应的1-level CCS Code数量分布
    #确定最终的Expert个数 N
    #确定要分类的疾病数量 K


#读入3-level CCS Code -- 1-level CCS Code映射，构建Dict

#遍历数据集中的每个病人
    #按照病例序列中的最后一个病例数据 查询Dict 生成最后一个病例数据对应的 1-level标签
    #将病人最后一个病例数据中的疾病，映射到0~K之间，作为待预测的label

#将每个病人的标签序列加入到数据集中

#模型代码
    #单个Expert
        #输入为病人病例序列
        #输出为1-level对应的3-level疾病

    #MOE
        #设置N个Expert，每个对应一个1-level标签
        #将病人序列输入到每个Expert
        #按照Expert与 3-level CCS code的映射关系，将专家的预测结果合并为一个K维的向量，这个K维的向量作为用户下一次患病情况的预测输出，O1
        #设置一个Gate net，输入为Expert输出concat，输出为一个值，对应是否死亡，O2
    #Loss
        #将O1与用户最后一个病例中的预测Label计算 Loss1
        #将O2与用户是否死亡计算 Loss2
        #采用加权和的方式和并Loss1 与 Loss2
import numpy as np


def precision_at_k(true_labels, predicted_probs, k):
    """
    计算 Precision@k

    参数:
    - true_labels: np.array, 真实标签 (0/1)
    - predicted_probs: np.array, 预测概率 (0~1之间的分数)
    - k: int, 计算 Precision@k 的前 k 个样本

    返回:
    - precision_k: float, Precision@k 指标
    """
    # 根据预测概率对样本排序（降序）
    sorted_indices = np.argsort(predicted_probs)[::-1]

    # 选取前 k 个样本
    top_k_indices = sorted_indices[:k]

    # 计算 Precision@k：前 k 个预测中，真实标签为 1（死亡）的占比
    precision_k = np.sum(true_labels[top_k_indices]) / k

    return precision_k
#训练模型
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 1613
    hidden_size = 64
    num_layers = 2
    output_size = 1
    #
    # # Create model
    # model = MultiLayerLSTM(input_size, hidden_size, num_layers, output_size).to(device)

    # #损失函数、优化器
    # loss_fn = nn.CrossEntropyLoss()   # 100 100  [100,2] [100]
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 3# 设定训练轮数
    # best_auroc=0.02
    runs = 3

    best_auroc_list = []
    best_auprc_list = []

    accuracy_values = []
    recall_values = []
    precision_values = []
    f1_values = []
    auroc_values = []
    auprc_values = []
    for run in range(runs):
        best_auroc = 0
        # args = ModelArgs(
        #     d_model=64,
        #     n_layer=2,
        #     vocab_size=64
        # )
        # mamba_model = Mamba(args)
        trans_model = TransformerEncoderModel(input_dim=64, output_dim=64,num_heads=4,num_layers=2)
        model = MultiLayerTransformer(input_size, hidden_size, num_layers, output_size, vac_emb).to(device)
        loss_fn = nn.CrossEntropyLoss()  # 100 100  [100,2] [100]
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(epochs):
            running_loss = 0.0
            model.train()

            train_label = []
            train_pre_label = []
            trained_features = []
            output_labels11 = []
            for j, data in enumerate(train_dataloader):
                inputs, labels,labels2, lengths, graphs = data
                output_labels11.append(labels.numpy())
                labels = labels.type(torch.FloatTensor)

                # 获取输入数据和标签
                # print(inputs.shape)
                inputs, labels, labels2 = inputs.to(device), labels.to(device), labels2.to(device)
                optimizer.zero_grad()  # 梯度清零
                for i in range(100):
                    for j in range(len(graphs[i])):
                        graphs[i][j].to(device)
                # label_all = torch.cat((labels.unsqueeze(1), labels2), dim=1)
                # 前向传播
                outputs, features= model(inputs, graphs)
                # print("LSTM outputs shape:", outputs.shape)
                # print("LSTM inputs shape:", inputs.shape)
                # outputs1 = outputs[:, :]  # outputs1 是从模型输出中提取出了在最后一个时间步（序列的末尾）的第一个特征或维度的值。
                # outputs1 = outputs.reshape(-1)
                labels=labels.long()
                loss = loss_fn(outputs.squeeze(), labels)
                with open("train-lstm.txt", "a") as f:
                    f.write(str(loss.item()) + "\n")
                loss.backward()
                optimizer.step()
                #
                #         # 统计损失
                running_loss += loss.item()
                if j % 100 == 10:  # 每10个batch打印一次损失
                    print(
                        f'Epoch [{epoch + 1}/{epochs}], Step [{j + 1}/{len(train_dataloader)}], Loss: {running_loss / 10}')
                    running_loss = 0.0
                trained_features.append(features.cpu().tolist())
            all_trained_features = np.array(trained_features)


            # 在每个epoch结束后，用测试集评估模型并计算AUROC
            model.eval()  # 切换到评估模式
            all_predictions = []
            all_labels = []

            with torch.no_grad():
                for id,test_data in enumerate(test_dataloader):
                    test_inputs, test_labels,test_labels2,test_lengths, graphs= test_data
                    test_labels = test_labels.type(torch.FloatTensor)

                    test_inputs, test_labels, test_labels2 = test_inputs.to(device), test_labels.to(device), test_labels2.to(device)

                    for i in range(100):
                        for j in range(len(graphs[i])):
                            graphs[i][j].to(device)

                    test_outputs, features = model(test_inputs, graphs)
                    # test_outputs1 = test_outputs[:, :]
                    test_labels = test_labels.long()
                    loss=loss_fn(test_outputs.squeeze(), test_labels)
                    with open("test-lstm.txt", "a") as f:
                        f.write(str(loss.item()) + "\n")

                    # 死亡的标签
                    all_predictions.append(torch.softmax(test_outputs,dim=1).detach().cpu().numpy())
                    # all_predictions.append(test_outputs1.detach().cpu().numpy())
                    all_labels.append(test_labels.detach().cpu().numpy())

                # 将预测和标签转换为numpy数组
                # all_predictions = np.concatenate(all_predictions)
                all_predict = np.concatenate([output[:, 1] for output in all_predictions])
                all_labels = np.concatenate(all_labels)
                precision, recall, _ = precision_recall_curve(all_labels, all_predict)
                print(f'Precision@50: {precision_at_k(all_labels, all_predict, 50):.4f}')
                print(f'Precision@100: {precision_at_k(all_labels, all_predict, 100):.4f}')
                print(f'Precision@150: {precision_at_k(all_labels, all_predict, 150):.4f}')
                # threshold = 0.5
                # all_predictions = (all_predictions > threshold).astype(int)  # 转换为0或1
                # all_labels = all_labels.astype(int)
                # 计算 Accuracy
                all_predictions = [1 if i>0.7 else 0  for i in all_predict ]
                accuracy = accuracy_score(all_labels, all_predictions)
                # 计算 Recall
                recall = recall_score(all_labels, all_predictions,zero_division=1)
                # 计算 Precision
                precision = precision_score(all_labels, all_predictions,zero_division=1)
                # 计算 F1-Score
                f1 = f1_score(all_labels, all_predictions,zero_division=1)
                # 计算AUPRC
                #auprc = auc(recall, precision)

                precision1, recall1, _ = precision_recall_curve(all_labels, all_predict, pos_label=1)
                auprc = auc(recall1, precision1)

                # 计算AUROC
                auroc = roc_auc_score(all_labels, all_predict)

                print(f'Epoch [{epoch + 1}/{epochs}], Test Accuracy: {accuracy}')
                print(f'Epoch [{epoch + 1}/{epochs}], Test Recall: {recall}')
                print(f'Epoch [{epoch + 1}/{epochs}], Test Precision: {precision}')
                print(f'Epoch [{epoch + 1}/{epochs}], Test F1 Score: {f1}')
                print(f'Epoch [{epoch + 1}/{epochs}], Test AUROC: {auroc}')
                print(f'Epoch [{epoch + 1}/{epochs}], Test AUPRC: {auprc}')

                if auroc > best_auroc:
                    if run == 0:
                        best_predict = all_predict
                        best_labels = all_labels
                    best_auroc = auroc
                    # best_auroc = auroc
                    best_recall = recall
                    best_precision = precision
                    best_f1 = f1
                    best_auprc = auprc
                    best_accuracy = accuracy
                    output_features = all_trained_features
                    output_labels = output_labels11
                    best_precision_50 = precision_at_k(all_labels, all_predict, 50)
                    best_precision_100 = precision_at_k(all_labels, all_predict, 100)
                    best_precision_150 = precision_at_k(all_labels, all_predict, 150)



        best_auroc_list.append(best_auroc)
        best_auprc_list.append(best_auprc)

        accuracy_values.append(best_accuracy)
        recall_values.append(best_recall)
        precision_values.append(best_precision)
        f1_values.append(best_f1)
        auroc_values.append(best_auroc)
        auprc_values.append(best_auprc)
        print(f'Run {run + 1} Best AUROC: {best_auroc}')
        print(f'Precision@50: {best_precision_50:.4f}')
        print(f'Precision@100: {best_precision_100:.4f}')
        print(f'Precision@150: {best_precision_150:.4f}')
        # torch.save(model.state_dict(), "base_model1.pth")
        #     if auroc > best_auroc:
        #         best_auroc = auroc
        # print(f'Best AUROC: {best_auroc}')  # 打印训练过程中的最佳 AUROC 值

    # 计算所有epoch的平均值
    mean_accuracy = sum(accuracy_values) / len(accuracy_values)
    std_acc = np.std(accuracy_values, ddof=1)

    mean_recall = sum(recall_values) / len(recall_values)
    std_recall = np.std(recall_values, ddof=1)

    mean_precision = sum(precision_values) / len(precision_values)
    std_precision = np.std(precision_values, ddof=1)

    mean_f1 = sum(f1_values) / len(f1_values)
    std_f1 = np.std(f1_values, ddof=1)

    mean_auroc = sum(auroc_values) / len(auroc_values)
    std_auroc = np.std(auroc_values, ddof=1)

    mean_auprc = sum(auprc_values) / len(auprc_values)
    std_auprc = np.std(auprc_values, ddof=1)

    # 输出最终的平均值
    print(f'Average Accuracy: {mean_accuracy}')
    print(f'Average Recall: {mean_recall}')
    print(f'Average Precision: {mean_precision}')
    print(f'Average F1 Score: {mean_f1}')
    print(f'Average AUROC: {mean_auroc}')
    print(f'Average AUPRC: {mean_auprc}')
    # np.savetxt('../mimic3/retain-mimicIII-pre.txt', best_predict)
    # np.savetxt('../mimic3/retain-mimicIII-label.txt', best_labels)
    print(f'Accuracy std: : {std_acc}')
    print(f'recall std: : {std_recall}')
    print(f'precision std: : {std_precision}')
    print(f' f1 std: : {std_f1}')
    print(f'AUROC std: : {std_auroc}')
    print(f'AUPRC std: : {std_auprc}')
    np.savetxt('lstm-mimicIII-pre.txt', best_predict)
    np.savetxt('lstm-mimicIII-label.txt', best_labels)
    print(best_auroc_list)
    print(best_auprc_list)


    averaged_features = output_features.reshape(-1, 128)
    labels = np.array(output_labels).reshape(-1)

    # 数据标准化
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(averaged_features)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(normalized_features)
    colors = ['#F94144', '#277DA1']
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=[colors[int(label)] for label in labels], cmap='viridis', alpha=0.7)
    # plt.xlabel("t-SNE Component 1")
    # plt.ylabel("t-SNE Component 2")
    # plt.title("t-SNE Visualization of High-Dimensional Data")
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.box(False)  # 去掉边框
    plt.tight_layout(pad=0)
    plt.savefig('process.png', dpi=600)
    plt.show()
    plt.close()
    # # UMAP降维至2维
    # reducer = PCA(n_components=2, random_state=42)
    # embedding = reducer.fit_transform(normalized_features)
    #
    # # 可视化并根据标签染色
    # colors = ['#F94144', '#277DA1']
    #
    # plt.figure(figsize=(8, 6), dpi=300)
    # plt.scatter(embedding[:, 0], embedding[:, 1], color=[colors[int(label)] for label in labels], s=5)
    #
    # plt.title("PCA Visualization of Original Features")
    # plt.xlabel("PCA Dimension 1")
    # plt.ylabel("PCA Dimension 2")
    # plt.savefig('process.png', dpi=600)
    # plt.show()
    # plt.close()
