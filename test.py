import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
 
# 模型代码编写，forward 就是直接调用 model(x) 时执行的计算流程
class CNN(nn.Module):
  def __init__(self, in_channels=1, num_classes=10):
    super(CNN,self).__init__()
    self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
    self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
    self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
    self.fc1 = nn.Linear(16*7*7, num_classes)
 
  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool(x)
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = x.reshape(x.shape[0], -1)
    x = self.fc1(x)
    return x
 
# 测试能否使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
# 超参数设定
input_size = 784
in_channel = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5
 
# 读取数据集
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
 
# 实例化模型
model = CNN().to(device)
 
# 设定损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
 
# 下面部分是训练，有时可以单独拿出来写函数
num_epochs = 4
for epoch in range(num_epochs):
  for batch_idex, (data, targets) in enumerate(train_loader):
    # 如果模型在 GPU 上，数据也要读入 GPU
    data = data.to(device=device)
    targets = targets.to(device=device)
    # print(data.shape)   # [64,1,28,28] Batch 大小 64 , 1 channel, 28*28 像素
    # forward 前向模型计算输出，然后根据输出算损失
    scores = model(data)
    loss = criterion(scores, targets)
    # backward 反向传播计算梯度
    optimizer.zero_grad()
    loss.backward()
    # 梯度下降，优化参数
    optimizer.step()
    if(batch_idex+1)%10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idex * len(data), len(train_loader.dataset),
                100. * batch_idex / len(train_loader), loss.item()))
 
# 评估准确度的函数
def check_accuracy(loader, model):
  if loader.dataset.train:
    print("Checking acc on training data")
  else:
    print("Checking acc on testing data")
  num_correct = 0
  num_samples = 0
  model.eval()  # 将模型调整为 eval 模式，具体可以搜一下区别
  with torch.no_grad():
    for x, y in loader:
      x = x.to(device=device)
      y = y.to(device=device)
      scores = model(x)
      # 64*10
      _, predictions = scores.max(1)
      num_correct += (predictions == y).sum()
      num_samples += predictions.size(0)
    acc = float(num_correct)/float(num_samples)*100
    print(f'Got {num_correct} / {num_samples} with accuracy {acc:.2f}')
  
  model.train()
  return acc
  
 
check_accuracy(train_loader, model)
check_accuracy(test_loader, model) 
        
