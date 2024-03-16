import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt # 导入 Matplotlib 工具包
import networkx as nx  # 导入 NetworkX 工具包
from networkx.algorithms.flow import edmonds_karp  # 导入 edmonds_karp 算法函数
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import copy
import RFGNNSY_Reliability


##Application 1
Input = np.array([0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 18, 18, 18, 18])
Output = np.array([5,  6,  7,  7,  8,  9,  9, 10, 11, 12, 13, 14, 15, 15, 15, 16, 16, 16, 19, 17, 17, 17, 19, 19, 19,  0,  1,  2,  3,  4])
Capacity = np.array([8, 4, 9, 13, 7, 6, 5, 12, 7, 8, 15, 8, 10, 4, 7, 10, 11, 13, 13, 13, 4, 9, 15, 11, 14, 11, 8, 11, 13, 6])
Cost = np.array([4, 4, 3, 5, 4, 5, 5, 5, 5, 3, 2, 6, 3, 4, 2, 3, 3, 2, 6, 3, 4, 4, 8, 6, 7, 8, 9, 7, 10, 12])
IR = np.full((1, len(Input)), 0.9)[0]
Num_Node = 20
MF0 = 20
s = 18
t = 19

G = nx.DiGraph()
for i in range(len(Input)):
    G.add_edge(Input[i], Output[i], capacity=Capacity[i])
maxFlowValue, maxFlowDict = nx.maximum_flow(G, s, t, flow_func=edmonds_karp)

SAMPLE = 50000
N_train = 20
N_train_min = 10
N_train_max = 20
Delt = 5
U = 100
lamd = IR

def gg(Capacityxx):
    Gxx = nx.DiGraph()
    for j in range(len(Input)):
        Gxx.add_edge(Input[j], Output[j], capacity=Capacityxx[j])
    maxFlowValue1, maxFlowDict1 = nx.maximum_flow(Gxx, s, t, flow_func=edmonds_karp)
    return maxFlowValue1

def RR(XX, SAMPLE):
    Capacityxx = np.zeros((SAMPLE, len(Input)), dtype=float)
    maxFlowValuexx = np.zeros((SAMPLE, 1), dtype=float)
    for h in range(SAMPLE):
        indices = np.where(XX[h, :] == 0)
        Capacityxx[h, :] = Capacity.copy()
        Capacityxx[h, indices[0]] = 0
        maxFlowValuexx[h] = gg(Capacityxx[h, :])
    return Capacityxx, maxFlowValuexx

def Surrogate(Capacity_train, maxFlowValue_train, hidden_channels):
    bili = 1
    batch_size1 = 128
    Epoch1 = 500
    model = RFGNNSY_Reliability.PM(Input, Output, s, t, Num_Node, Capacity_train, maxFlowValue_train, bili, batch_size1, Epoch1, hidden_channels)
    out1 = torch.empty((0,))
    model.eval()
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.edge_weight, data.batch)
            out1 = torch.cat([out1, out], dim=0)
    return out1

XX = np.zeros((SAMPLE, len(Input)))
for j in range(len(Input)):
    XX[:, j] = np.random.binomial(n=1, p=lamd[j], size=SAMPLE)
unique_rows, inverse, counts = np.unique(XX, axis=0, return_inverse=True, return_counts=True)
XX_RR = unique_rows
Capacity_RR, maxFlowValue_RR = RR(XX_RR, XX_RR.shape[0])
Capacity_XX = Capacity_RR[inverse, :]
maxFlowValue_XX = maxFlowValue_RR[inverse]
Rxx_test = np.count_nonzero(maxFlowValue_XX >= MF0) / SAMPLE

Capacity_Bian = Capacity_RR.copy()
Capacity_test = Capacity_RR.copy()
dataset = RFGNNSY_Reliability.CL_test(Input, Output, s, t, Num_Node, Capacity_test)
loader = DataLoader(dataset, batch_size=128, shuffle=False)

NN = 30
NN0 = 2 * Num_Node
Hidden_channels = torch.tensor(np.arange(NN0, NN0 + NN * 2, 2).reshape(NN, 1))
Rxx_predict = np.zeros((U, NN), dtype=float)
Error_abs_train = np.zeros((U, NN), dtype=float)
Error_abs_test = np.zeros((U, NN), dtype=float)
Error = np.zeros((U, NN), dtype=float)
N_train_ADD = np.zeros((U, 1), dtype=int)
Rxx_predict_min = np.zeros((U, 1), dtype=float)
Rxx_predict_max = np.zeros((U, 1), dtype=float)
Rxx_predict_mean = np.zeros((U, 1), dtype=float)
Stop = np.zeros((U, 1), dtype=float)
Stop0 = 0.05

Index1 = np.ceil(np.linspace(0, XX_RR.shape[0] - 1, N_train)).reshape(-1, 1)
Index = np.array([[int(x) for x in row] for row in Index1])
XX_RR_train = XX_RR[Index.ravel(), :]
Index_hidden = 0

for i in range(U):
    if i == 0:
        Capacityxx, maxFlowValuexx = RR(XX_RR_train, N_train)

    if i == 0:
        Capacity_train = Capacityxx
        maxFlowValue_train = maxFlowValuexx
    else:
        Capacity_train = np.concatenate([Capacity_train, Capacityxx], axis=0)
        maxFlowValue_train = np.concatenate([maxFlowValue_train, maxFlowValuexx], axis=0)

    maxFlowValue_predict_unique_list = []
    for h in range(NN):
        out_h = Surrogate(Capacity_train, maxFlowValue_train, Hidden_channels[h])
        maxFlowValue_predict_unique_list.append(out_h.detach().numpy())
        print(h)
    maxFlowValue_predict_unique = np.array(maxFlowValue_predict_unique_list).reshape(NN, maxFlowValue_predict_unique_list[0].shape[0]).T

    maxFlowValue_predict_unique[maxFlowValue_predict_unique < 0] = 0
    Error_abs = np.abs(maxFlowValue_predict_unique[Index.ravel(), :] - maxFlowValue_train)
    Error_abs_train[i, :] = np.mean(Error_abs, axis=0).reshape(1, -1)
    maxFlowValue_predict_unique[Index.ravel(), :] = maxFlowValue_train
    maxFlowValue_predict = maxFlowValue_predict_unique[inverse, :]

    indicator = np.where(maxFlowValue_predict >= MF0, 1, 0)
    Rxx_predict[i, :] = np.mean(indicator, axis=0)
    Error[i, :] = np.abs((Rxx_predict[i, :] - Rxx_test) / Rxx_test)
    Rxx_predict_min[i] = np.min(Rxx_predict[i, :])
    Rxx_predict_max[i] = np.max(Rxx_predict[i, :])
    Rxx_predict_mean[i] = np.mean(Rxx_predict[i, :])

    Stop[i] = (Rxx_predict_max[i] - Rxx_predict_min[i]) / (1 - Rxx_predict_mean[i])
    if Stop[i] <= Stop0:
        break
    print("Number of Iteration:", i)
    print("Real Reliability:", Rxx_test)
    print("Predict Reliability:", Rxx_predict_mean[i])
    print("Stopping Criterion:", Stop[i])

    indicator_unique = np.where(maxFlowValue_predict_unique > MF0, 1, 0)
    ones_count = np.sum(indicator_unique, axis=1).reshape(-1, 1)
    zeros_count = indicator_unique.shape[1] - ones_count
    L = np.abs((ones_count - zeros_count) / NN)
    cc = np.argsort(L, axis=0)
    LL = L[cc, 0]

    indicator_unique_LL = np.where(LL <= 0.5, 1, 0)
    counts_LL = np.sum(indicator_unique_LL)
    N_train_suan = int(np.ceil(counts_LL / Delt))
    if N_train_suan < N_train_min:
        N_train_add = N_train_min
    elif N_train_suan > N_train_max:
        N_train_add = N_train_max
    else:
        N_train_add = N_train_suan
    N_train_ADD[i] = N_train_add

    selected_rows = []
    ii = 0
    while len(selected_rows) < N_train_add and ii < len(LL):
        row = cc[ii]
        if not any(np.array_equal(row, index) for index in Index):
            selected_rows.append(row)
        ii += 1
    ccc = np.array(selected_rows)
    Index = np.concatenate([Index, ccc], axis=0)

    Capacityxx = Capacity_Bian[ccc, :].reshape(-1, Capacity_Bian.shape[1])
    maxFlowValuexx = np.zeros((Capacityxx.shape[0], 1), dtype=float)
    for h in range(len(Capacityxx)):
        maxFlowValuexx[h] = gg(Capacityxx[h, :])

    Error_abs1 = np.abs(maxFlowValue_predict_unique[ccc.ravel(), :] - maxFlowValuexx)
    Error_abs_test[i, :] = np.mean(Error_abs1, axis=0).reshape(1, -1)
    Error_abs_mean = 0.8 * Error_abs_train[i, :] + 0.2 * Error_abs_test[i, :]
    max_index = np.argmax(Error_abs_mean)
    min_index = np.argmin(Error_abs_mean)
    Hidden_channels[max_index] = Hidden_channels[min_index]
    Index_hidden += 1

    unique_rows_train, inverse_train, counts_train = np.unique(Capacity_train, axis=0, return_inverse=True, return_counts=True)

print("Estimated Reliability:", Rxx_predict_mean[i])
print("Real Reliability:", Rxx_test)

plt.figure(1)
plt.axhline(y=Rxx_test, linestyle='--', color='r', label='Real Results')
plt.plot(Rxx_predict_mean[0:i + 1], marker='o', label='Predict Results')
plt.plot(Rxx_predict_max[0:i + 1], linestyle=':', label='Upper Bound')
plt.plot(Rxx_predict_min[0:i + 1], linestyle=':', label='Lower Bound')
plt.xlabel("Number of Iteration")
plt.ylabel("Network Reliability")
plt.legend()
plt.show(block=True)

