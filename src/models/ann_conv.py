from torch import nn
import torch

#device = 'cuda' if torch.cuda_is_available() else 'cpu'

class MFCC_CNN(torch.nn.Module):
    """
    ANN-based model for classifying using Mel Frequency Cepstral Coefficients extracted from the spectrograms

    TODO: consider adding dropout, residuals, other time channel pooling techniques, pooling layers?
    """

    def __init__(self, channels=[16,32,64]):
        super().__init__()

        self.init_channels = 1 # mfcc matrix
        channels.insert(0, self.init_channels)

        # setup our convolutions
        self.convs = nn.ModuleList()
        for i in range(len(channels)-1):
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=3),
                nn.BatchNorm2d(channels[i+1]),
                nn.SiLU(),
                nn.MaxPool2d(kernel_size=2)
            ))

        # FCN for classification
        self.fcn = nn.Sequential(
            nn.Linear(160, 256), # 1728 should be adjusted based on shape of flattened output after convolutions
            nn.SiLU(),
            nn.Linear(256, 5),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
        - x: tensor of shape (batch_size, channels, freq, time)
        """
        for conv in self.convs:
            x = conv(x)
            print(x.shape)
        
        # for now, we'll handle varying time dimension by taking global average across time dimension
        x = x.mean(dim=-1)

        x = torch.flatten(x, start_dim=1)

        x = self.fcn(x)
        
        return x


if __name__ == '__main__':
    cnn = MFCC_CNN([8, 16, 32])
    #cnn = cnn.to(device)

    y = cnn(torch.randn(10, 1, 60, 100))
    print(y)