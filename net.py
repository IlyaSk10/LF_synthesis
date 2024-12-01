import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        # Encoder
        self.enc1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.enc2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.enc4 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.enc5 = nn.Conv1d(128, 256, kernel_size=3, padding=1)

        # Decoder
        self.dec1 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.dec2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.dec3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.dec4 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.dec5 = nn.Conv1d(16, 1, kernel_size=3, padding=1)

        # Pooling and Upsampling
        self.pool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.enc1(x))
        x2 = self.pool(F.relu(self.enc2(x1)))
        x3 = self.pool(F.relu(self.enc3(x2)))
        x4 = self.pool(F.relu(self.enc4(x3)))
        x5 = self.pool(F.relu(self.enc5(x4)))

        # Decoder
        x6 = self.upsample(F.relu(self.dec1(x5)))

        # Ensure the size matches for skip connection
        if x6.size(2) != x4.size(2):
            x6 = F.pad(x6, (0, x4.size(2) - x6.size(2)))

        x7 = self.upsample(F.relu(self.dec2(x6 + x4)))

        if x7.size(2) != x3.size(2):
            x7 = F.pad(x7, (0, x3.size(2) - x7.size(2)))

        x8 = self.upsample(F.relu(self.dec3(x7 + x3)))

        if x8.size(2) != x2.size(2):
            x8 = F.pad(x8, (0, x2.size(2) - x8.size(2)))

        x9 = self.upsample(F.relu(self.dec4(x8 + x2)))

        if x9.size(2) != x1.size(2):
            x9 = F.pad(x9, (0, x1.size(2) - x9.size(2)))

        x10 = self.dec5(x9 + x1)  # Final skip connection

        return x10