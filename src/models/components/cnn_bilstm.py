from torch import nn

class CNN_BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(5,9), stride=1, padding=1,filters=8),
            nn.Conv2d(8, 8, kernel_size=(5,7), stride=1, padding=1,filters=8),
            nn.LSTM(8, 32, bidirectional=True),
            nn.LSTM(32*2, 32, bidirectional=True),
            nn.LSTM(32*2, 32, bidirectional=True),
            nn.LSTM(32*2, 32, bidirectional=True),
            nn.Linear(32*2, 10),
            nn.Linear(10, 10),
        )
    
    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.cnn(x)
