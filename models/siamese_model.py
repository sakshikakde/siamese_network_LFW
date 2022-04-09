import torch
import torch.nn.functional as F
from torch import nn 

class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 5, padding=(2,2))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=(2,2))
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=(1,1))
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=(1,1))
        self.bn4 = nn.BatchNorm2d(512)
        self.fc5 = nn.Linear(131072, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 2)


    def feature_extractor(self,x):
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x = torch.flatten(x, 1) 
        x = self.bn5(F.relu(self.fc5(x)))
        return x
    
    def similarity(self, fv1, fv2):
        fv = torch.cat((fv1, fv2), 1)
        fv = F.relu(self.fc1(fv))
        sim = self.fc2(fv)
        return sim

    def forward(self, x):
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        fv1 = self.feature_extractor(x1)
        fv2 = self.feature_extractor(x2)
        sim = self.similarity(fv1, fv2)
        return sim