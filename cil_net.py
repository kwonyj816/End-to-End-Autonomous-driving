import torch
import torch.nn as nn

class CIL_skip_net(nn.Module):
    def __init__(self):
        super(CIL_skip_net, self).__init__()

        # 이미지 처리 네트워크
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
        
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
        )

        self.conv_block6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
        )

        self.conv_block7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
        )

        self.conv_block8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
        )

        self.img_fc = nn.Sequential(
            nn.Linear(8192, 512),  # in_feature는 256x51x91
            nn.Dropout(p=0.3),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.Dropout(p=0.5),
            nn.ReLU(),
        )

        self.speed_fc = nn.Sequential(
            nn.Linear(1, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
        )


        self.emb_fc = nn.Sequential(
            nn.Linear(512+128, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

        self.control_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256,256),
            #nn.Dropout(0.5)
            nn.ReLU(),
            nn.Linear(256, 2),
        )


    def forward(self, img, speed):

        x1 = self.conv_block1(img)
        x2 = self.conv_block2(x1)
        x2 = x1 + x2

        x3 = self.conv_block3(x2)
        x4 = self.conv_block4(x3)
        x4 = x3 + x4

        x5 = self.conv_block5(x4)
        x6 = self.conv_block6(x5)
        x6 = x5 + x6

        x7 = self.conv_block7(x6)
        x8 = self.conv_block8(x7)
        x8 = x7 + x8
        img = x8.view(-1, 8192) #8192
        img = self.img_fc(img)

        speed = speed.view(-1, 1)
        speed = speed.to(torch.float32)
        speed = self.speed_fc(speed)


        emb = torch.cat([img, speed], dim=1)
        emb = self.emb_fc(emb)

        pred_control = self.control_fc(emb)

        return  pred_control, img  
