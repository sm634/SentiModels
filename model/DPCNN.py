import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule


class DPCNN(BasicModule):
    """
    DPCNN for sentences classification.
    """

    def __init__(self, config):
        super(DPCNN, self).__init__()
        self.config = config
        self.channel_size = 250
        self.vocab_size = self.config.vocab_size
        self.embedding_dim = self.config.word_embedding_dimension
        self.embed = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)

        self.conv_region_embedding = nn.Conv1d(in_channels=self.config.sequence_length, out_channels=self.channel_size,
                                               kernel_size=3, stride=1)
        self.conv3 = nn.Conv1d(in_channels=self.channel_size, out_channels=self.channel_size, kernel_size=3, stride=1)
        self.pooling = nn.MaxPool1d(kernel_size=3, stride=2)
        self.padding_conv = nn.ZeroPad2d((1, 1, 0, 0))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.act_fun = nn.ReLU()
        self.linear_out = nn.Linear(self.channel_size, config.linear_out)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch = self.config.batch_size

        x = self.embed(x)

        x = self.conv_region_embedding(x)  # [batch_size, channel_size, length, 1]

        x = self.act_fun(x)
        x = self.conv3(x)

        x = self.act_fun(x)
        x = self.conv3(x)

        while x.size()[-1] > 2:
            x = self._block(x)

        # flattening the output of the final pooling layer to feed the fully connected/linear layer.
        x = x.view(batch, self.channel_size)

        # dropout to counter over-fitting.
        x = self.dropout(x)

        x = self.linear_out(x)
        x = self.sigmoid(x)

        return x

    def _block(self, x):
        # Pooling - downsample for equal factor in linear combination.
        px = self.pooling(x)

        # Convolution
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)

        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)

        # Short Cut
        x = x + px

        return x

    def predict(self, x):
        self.eval()
        out = self.forward(x)
        predict_labels = torch.max(out, 1)[1]
        self.train(mode=True)
        return predict_labels
