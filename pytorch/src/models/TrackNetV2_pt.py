import torch
import torch.nn as nn

class TrackNetV2(nn.Module):
    def __init__(self, input_height, input_width):
        super(TrackNetV2, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.layer6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.layer10 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer11 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.layer14 = nn.Upsample(scale_factor=2)
        self.layer15 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.layer16 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.layer17 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.layer18 = nn.Upsample(scale_factor=2)
        self.layer19 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.layer20 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.layer21 = nn.Upsample(scale_factor=2)
        self.layer22 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer23 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer24 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.layer2(self.layer1(x))
        x2 = self.layer5(self.layer4(self.layer3(x1)))
        x3 = self.layer9(self.layer8(self.layer7(self.layer6(x2))))
        x = self.layer13(self.layer12(self.layer11(self.layer10(x3))))
        x = torch.cat([self.layer14(x), x3], dim=1)
        x = self.layer17(self.layer16(self.layer15(x)))
        x = torch.cat([self.layer18(x), x2], dim=1)
        x = self.layer20(self.layer19(x))
        x = torch.cat([self.layer21(x), x1], dim=1)
        x = self.layer23(self.layer22(x))
        x = self.layer24(x)
        return x
