import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool
from torch_geometric.utils import to_networkx
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import copy


class Qianchuli(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = torch.nn.Linear(1, 1)
    def forward(self, x, edge_index, edge_weight1):
        norm = edge_weight1
        return self.propagate(edge_index, x=x, norm=norm)
    def message(self, x_j, norm):
        return norm * x_j
    def update(self, aggr_out):
        return aggr_out

class RFGNNConvSY(MessagePassing):
    def __init__(self, input_dimension, hidden_channels):
        super().__init__(aggr='add')
        self.lin = torch.nn.Linear(input_dimension, hidden_channels)
    def forward(self, x, edge_index, edge_weight):
        x = self.lin(x)
        norm = edge_weight
        return self.propagate(edge_index, x=x, norm=norm)
    def message(self, x_j, norm):
        return norm * x_j
    def update(self, aggr_out):
        return aggr_out

class RFGNN(torch.nn.Module):
    def __init__(self, input_dimension, output_dimension, hidden_channels):
        super(RFGNN, self).__init__()
        # torch.manual_seed(12345)
        self.conv = RFGNNConvSY(input_dimension, hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, output_dimension)
    def forward(self, x, edge_index, edge_weight, batch):
        x = self.conv(x, edge_index, edge_weight)
        x = x.relu()
        x = self.lin1(x)
        x = global_max_pool(x, batch)
        x = self.lin2(x)
        return x

def CL(Input, Output, s, t, Num_Node, Capacity1, maxFlowValue1):
    dataset = []
    edge_index = torch.tensor(np.vstack((Input, Output)), dtype=torch.long)
    edge_index_fan = torch.tensor(np.vstack((Output, Input)), dtype=torch.long)
    SAMPLE = len(maxFlowValue1)
    for i in range(SAMPLE):
        Capacity2 = Capacity1[i, :]
        edge_weight1 = torch.tensor(Capacity2.reshape(-1, 1), dtype=torch.float)
        QCL = Qianchuli(1, 1)
        xx = torch.ones((Num_Node, 1), dtype=torch.float)
        node_list = []
        node_list.append(QCL(xx, edge_index, edge_weight1))
        while True:
            indices_node = np.where(node_list[-1] == 0)
            xx[indices_node[0]] = 0
            xx[s] = 1
            node_new = QCL(xx, edge_index, edge_weight1)
            if np.array_equal(node_new, node_list[-1]):
                break
            else:
                node_list.append(node_new)
        m1 = xx[Input] * edge_weight1
        xx = torch.ones((Num_Node, 1), dtype=torch.float)
        node_list_fan = []
        node_list_fan.append(QCL(xx, edge_index_fan, edge_weight1))
        while True:
            indices_node_fan = np.where(node_list_fan[-1] == 0)
            xx[indices_node_fan[0]] = 0
            xx[t] = 1
            node_new_fan = QCL(xx, edge_index_fan, edge_weight1)
            if np.array_equal(node_new_fan, node_list_fan[-1]):
                break
            else:
                node_list_fan.append(node_new_fan)
        m2 = xx[Output] * edge_weight1
        indices_m2 = np.where(m2 == 0)
        m1[indices_m2[0]] = 0
        xx = torch.eye(Num_Node, dtype=torch.float)
        edge_weight = m1.reshape(-1, 1).clone().detach()
        yy = torch.tensor(maxFlowValue1[i], dtype=torch.float)
        dataset.append(Data(x=xx, edge_index=edge_index, edge_weight=edge_weight, y=yy))
    return dataset

def CL_test(Input, Output, s, t, Num_Node, Capacity1):
    dataset = []
    edge_index = torch.tensor(np.vstack((Input, Output)), dtype=torch.long)
    edge_index_fan = torch.tensor(np.vstack((Output, Input)), dtype=torch.long)
    SAMPLE = len(Capacity1)
    for i in range(SAMPLE):
        Capacity2 = Capacity1[i, :]
        edge_weight1 = torch.tensor(Capacity2.reshape(-1, 1), dtype=torch.float)
        QCL = Qianchuli(1, 1)
        xx = torch.ones((Num_Node, 1), dtype=torch.float)
        node_list = []
        node_list.append(QCL(xx, edge_index, edge_weight1))
        while True:
            indices_node = np.where(node_list[-1] == 0)
            xx[indices_node[0]] = 0
            xx[s] = 1
            node_new = QCL(xx, edge_index, edge_weight1)
            if np.array_equal(node_new, node_list[-1]):
                break
            else:
                node_list.append(node_new)
        m1 = xx[Input] * edge_weight1
        xx = torch.ones((Num_Node, 1), dtype=torch.float)
        node_list_fan = []
        node_list_fan.append(QCL(xx, edge_index_fan, edge_weight1))
        while True:
            indices_node_fan = np.where(node_list_fan[-1] == 0)
            xx[indices_node_fan[0]] = 0
            xx[t] = 1
            node_new_fan = QCL(xx, edge_index_fan, edge_weight1)
            if np.array_equal(node_new_fan, node_list_fan[-1]):
                break
            else:
                node_list_fan.append(node_new_fan)
        m2 = xx[Output] * edge_weight1
        indices_m2 = np.where(m2 == 0)
        m1[indices_m2[0]] = 0
        xx = torch.eye(Num_Node, dtype=torch.float)
        edge_weight = m1.reshape(-1, 1).clone().detach()
        dataset.append(Data(x=xx, edge_index=edge_index, edge_weight=edge_weight))
    return dataset


def PM(Input, Output, s, t, Num_Node, Capacity1, maxFlowValue1, bili, batch_size1, Epoch1, hidden_channels1):
    dataset = CL(Input, Output, s, t, Num_Node, Capacity1, maxFlowValue1)
    SAMPLE = len(maxFlowValue1)
    train_num = round(SAMPLE * bili)
    train_dataset = dataset[:train_num]
    test_dataset = dataset[train_num:]
    batch_size = batch_size1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dimension = Num_Node
    hidden_channels11 = hidden_channels1
    num_classes = 1
    output_dimension = num_classes
    model = RFGNN(input_dimension=input_dimension, output_dimension=output_dimension, hidden_channels=hidden_channels11)
    # print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()

    def train():
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_weight, data.batch)
            loss = criterion(out, data.y.view(-1, 1))
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_weight, data.batch)
            loss = criterion(out, data.y.view(-1, 1))
            # print('Epoch: {}, Loss: {:.4f}'.format(epoch + 1, loss.item()))
        return loss

    Epoch = Epoch1
    loss1 = torch.zeros(Epoch)
    k = 0
    for epoch in range(1, Epoch+1):
        loss = train()
        # loss1[k] = loss
        # k = k + 1
        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    # plt.plot(range(1, Epoch+1), loss1.detach().numpy())
    # plt.title('Loss Function')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.show()

    return model





