import torch
import random

def sgd(params, lr, batch_size):
    with torch.no_grad():  
        for param in params:  
            param -= lr * param.grad / batch_size  
            param.grad.zero_()

def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X,w) + b
    y += torch.normal(0,1,y.shape)
    return X,y.reshape((-1,1))

def data_iter(batch_size, features, labels):
    num_example = len(features)
    indice = [i for i in range(num_example)]
    random.shuffle(indice)
    for i in range(0,num_example,batch_size):
        batch_indice = torch.tensor(indice[i:min(i + batch_size,num_example)])
        yield features[batch_indice],labels[batch_indice]

def linrg(X,w,b):
    return torch.matmul(X,w) + b

def MSE (y,y_hat):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(parms,lr,batch_size):
    with torch.no_grad():
        for parm in parms:
            parm -= lr * parm.grad / batch_size
            parm.grad.zero_()

def train(lr=0.03,num_epoch=3,batch_size=10):
    lr = lr
    num_epochs = num_epoch
    net = linrg
    loss = MSE 

    w = torch.normal(0,1,size=(2,1),requires_grad=True)
    b = torch.zeros(1,requires_grad=True)

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
            l.sum().backward()
            sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

train()