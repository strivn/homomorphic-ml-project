# %%
# Backup code for previous resnet model

# %%
import torch
import torch.nn as nn

class CustomBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomBlock, self).__init__()
        # Define conv1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # Define bn1 (GroupNorm with 2 groups for in_channels channels)
       # self.gn1 = nn.GroupNorm(2, in_channels, eps=1e-05, affine=True)
        # Define ReLU activation
        self.relu = nn.ReLU(inplace=True)
        # Define conv2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # Define bn2 (GroupNorm with 2 groups for out_channels channels)
        self.gn2 = nn.GroupNorm(2, out_channels, eps=1e-05, affine=True)
        # Define downsample
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.GroupNorm(32, out_channels, eps=1e-05, affine=True)
        )

    def forward(self, x):
        # Forward pass through conv1, bn1, and relu
        out = self.conv1(x)
       # out = self.gn1(out)
        out = self.relu(out)
        # Forward pass through conv2 and bn2
        out = self.conv2(out)
        out = self.gn2(out)

        # Downsampling path
        residual = self.downsample(x)

        # Add residual to the output
        out += residual

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)


        return out

# %%
import torch
import torch.nn as nn

class ResNet_Part1(nn.Module):
    def __init__(self):
        super(ResNet_Part1, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.GroupNorm(32, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.GroupNorm(2, 64, eps=1e-5, affine=True),
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.GroupNorm(2, 64, eps=1e-5, affine=True),

            nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.GroupNorm(2, 64, eps=1e-5, affine=True),
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.GroupNorm(2, 64, eps=1e-5, affine=True))


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        return x



# %%
import torch
import torch.nn as nn

class ResNet_Part2(nn.Module):
    def __init__(self):
        super(ResNet_Part2, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 9)

    def forward(self, x):

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# %%
class FullResNet(nn.Module):
    def __init__(self, model_part1, model_layer2, model_layer3, model_layer4, model_part2):
        super(FullResNet, self).__init__()
        # Store each part as a submodule
        self.model_part1 = model_part1
        self.model_layer2 = model_layer2
        self.model_layer3 = model_layer3
        self.model_layer4 = model_layer4
        self.model_part2 = model_part2

    def forward(self, x):
        # Pass the input through each part sequentially
        x = self.model_part1(x)    # Initial ResNet layers
        x = self.model_layer2(x)   # Custom block layer 2
        x = self.model_layer3(x)   # Custom block layer 3
        x = self.model_layer4(x)   # Custom block layer 4
        x = self.model_part2(x)    # Final ResNet layers
        return x

# %%
model_part1 = ResNet_Part1()
model_layer2 = CustomBlock(64, 128)
model_layer3 = CustomBlock(128, 256)
model_layer4 = CustomBlock(256, 512)
model_part2 = ResNet_Part2()

model = FullResNet(model_part1, model_layer2, model_layer3, model_layer4, model_part2)


