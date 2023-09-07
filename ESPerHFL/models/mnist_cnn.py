import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class mnist_lenet(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(mnist_lenet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 10, kernel_size=5)

        self.conv2 = nn.Conv2d(10, 20, kernel_size= 5)

        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50, output_channels)

    def forward(self, x):
        x = self.conv1(x)

        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)

        x = self.conv2_drop(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return  x

class mnist_lenet1(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(mnist_lenet1, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 10, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(10,momentum=0.1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size= 5)
        self.bn2 = nn.BatchNorm2d(20,momentum=0.1)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50, output_channels)


        # self.moving_mean = Variable(torch.zeros(1,10,24,24))
        # self.moving_var = Variable(torch.zeros(1,10,24,24))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

#def batch_norm_1d(x, gamma, beta, is_training, moving_mean, moving_var, moving_momentum=0.1):
def batch_norm_1d(x, gamma, beta):
    eps = 1e-5
    x_mean = torch.mean(x, dim=0, keepdim=True) # 保留维度进行 broadcast
    x_var = torch.mean((x - x_mean) ** 2, dim=0, keepdim=True)
    x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
    # if is_training:
    #     x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
    #     moving_mean[:] = moving_momentum * moving_mean + (1. - moving_momentum) * x_mean
    #     moving_var[:] = moving_momentum * moving_var + (1. - moving_momentum) * x_var
    # else:
    #     x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
    #return gamma.view_as(x_mean) * x_hat + beta.view_as(x_mean)
    return torch.mul(gamma,x_hat)  + beta