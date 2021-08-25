import torch.nn as nn

class MyConvNet(nn.Module):
    def __init__(self):
        super(MyConvNet, self).__init__()
        self.conv1 = nn.Sequential(
             nn.Conv2d(
                     in_channels=1
                    ,out_channels=16
                    ,kernel_size=7
                    ,stride=1
             )
#            ,nn.ReLU()
            ,nn.MaxPool2d(
                kernel_size=3
                ,stride=3
            )
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                     in_channels=16
                    ,out_channels=64
                    ,kernel_size=4
                    ,stride=1
#                    ,padding=1
             )
            ,nn.MaxPool2d(
                kernel_size=3
                ,stride=2
            )
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                     in_channels=64
                    ,out_channels=256
                    ,kernel_size=2
                    ,stride=1
#                    ,padding=1
             )
#            ,nn.ReLU()
            ,nn.MaxPool2d(
                 kernel_size=2
                ,stride=2
            )
        )
        self.classifier=nn.Sequential(
            nn.Linear(256*2*2,256)
            ,nn.ReLU()
            ,nn.Dropout(p=0.2)
            ,nn.Linear(256,32)
            ,nn.ReLU()
#            ,nn.Dropout(p=0.5)
            ,nn.Linear(32,7)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x=x.view(x.size(0),-1)
        output=self.classifier(x)
        return output