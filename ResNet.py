import torch
import torch.nn as nn

class AdvancedCNN(nn.Module):
    def block1(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
        )

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv12 = self._make_block(in_channels=3, out_channels=64, num_conv=2)
        self.conv22 = self._make_block(in_channels=64, out_channels=128, num_conv=2)
        self.conv32 = self._make_block(in_channels=128, out_channels=256, num_conv=2)
        self.conv13 = self._make_block(in_channels=512, out_channels=512, num_conv=3)
        self.conv23 = self._make_block(in_channels=512, out_channels=512, num_conv=3)
        self.linear_1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(7*7*512, 4096),
            nn.LeakyReLU()
        )
        self.linear_2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(),
        )
        self.linear_3 = nn.Linear(4096, 1000)

    def forward(self, x):
        x = self.conv12(x)
        x = self.conv22(x)
        x = self.conv33(x)
        x = self.conv13(x)
        x = self.conv23(x)
        x = x.view(x.shape[0], -1)  # x = self.flatten(x)
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        return x