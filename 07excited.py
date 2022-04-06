import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader  # 这里的Dataset是抽象类


##1. 准备数据集
class DiabetesDataset(Dataset):# 继承抽象类DataSet

    ##初始化读数据
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]      ##shape : [N* M]矩阵，则shape[0]=N； 这里的shape[0]就是样本数目
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    ## 取一条数据
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    ## 得到数据的数量
    def __len__(self):
        return self.len


dataset = DiabetesDataset('diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset,      #数据集
                          batch_size=32,        #batch_size
                          shuffle=True)        #读数据时，并行进程的数量（和CPU核心进程有关）


##2. 实例化模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

## 3.损失函数和优化器
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

##4. 训练
for epoch in range(100):
    for i, data in enumerate(train_loader, 0):
        # 1. Prepare data
        inputs, labels = data           ## 这里的data 是一个（x,y)的向量
        # 2. Forward
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.item())
        # 3. Backward
        optimizer.zero_grad()
        loss.backward()
        # 4. Update
        optimizer.step()
