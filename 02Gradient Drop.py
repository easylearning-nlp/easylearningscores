import numpy as np
import matplotlib.pyplot as plt;

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0

def  forward(x):
    return x * w

def cost(xs, ys):
    cost = 0
    for x,y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred- y) ** 2
    return cost / len(xs)

def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w-y)
    return grad / len(xs)
epoch_list=[]
cost_val_list=[]

for epoch in np.arange(1000):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.001*grad_val
    epoch_list.append(epoch)
    cost_val_list.append(cost_val)
    print("Epoch:",epoch, "w=",w, "Loss=",cost_val)

    ##画图
plt.plot(epoch_list, cost_val_list)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
