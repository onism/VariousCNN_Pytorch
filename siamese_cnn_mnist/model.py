import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn_layer = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 28, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(28,56, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(56*7*7, 512),
            nn.ReLU(),
            nn.Linear(512,2)
        )
    def _forward(self, x):
        output = self.cnn_layer(x)
        # size : batch * 56 * 7 * 7
        output = output.view(output.size()[0],-1)
        output = self.fc(output)
        return output
    
    def forward(self, input_1, input_2):
        output1 = self._forward(input_1)
        output2 = self._forward(input_2)
        return output1, output2


class ContrastiveLoss(nn.Module):
    '''
    The objective of the siamese architecture is not to classify input images, 
    but to differentiate between them. So, a classification loss function 
    (such as cross entropy) would not be the best fit. Instead, 
    this architecture is better suited to use a contrastive function. 
    Intuitively, this function just evaluates how well the network is distinguishing 
    a given pair of images
    ref: https://hackernoon.com/one-shot-learning-with-siamese-networks-in-pytorch-8ddaab10340e
    '''
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin 
    
    def forward(self, output1, output2, label):
        dist = F.pairwise_distance(output1, output2, keepdim=True)
        loss = torch.mean((1-label) * torch.pow(dist,2) + 
                            label*torch.pow(torch.clamp(self.margin -dist, min=0.0),2))
        return loss

model = SiameseNetwork()
x = np.random.randn(16,1,28,28).astype(np.float32)
label = np.random.randint(1, size=(16,1))
x = torch.from_numpy(x)
o1, o2 = model(x,x)

criterion = ContrastiveLoss()
loss = criterion(o1,o2, torch.from_numpy(label))
print(loss.item())

