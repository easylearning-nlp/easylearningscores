'''
1.prepare dataset
2.Design model using Class
3.Construct loss and optimizer
4.Trainging cycle
'''
import torch
x_data = torch.tensor([[1.0],[2.0],[3.0]])
y_data = torch.tensor([[2.0],[4.0],[6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1) #Class nn.linear contain two member Tensors:weights and bias.y =wx+b

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
'''
训练流程如下：
（1）求y_pred预测值
（2）求解损失值;进行梯度清零
（3）进行反向传播backward
（4）进行权重更新update
'''
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#output weight and bias
print("w = ", model.linear.weight.item())
print("b = ", model.linear.bias.item())
#Test Model
x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print("y_pred = ",y_test.data)