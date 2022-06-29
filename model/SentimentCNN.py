import torch.nn as nn
import torch.nn.functional as F


class BaseSentimentCNN(nn.Module):
    """
    The base CNN prototype model for sentiment classification
    """

    def __init__(self, config):
        super(BaseSentimentCNN, self).__init__()
        """
        :vocab_size: [int] number of words to embed to index from lookup table.
        :embedding_dim: [int] the vector dimension size representing a word in the embedding matrix.
        :batch_size: [int] the input batch sample size.
        :seq_length: [int] the fixed sentence length of words (padded or trimmed) to feed the conv layer.
        :dropout: the probability of dropout
        """
        self.config = config
        # embedding layer
        self.embedding_dim = config.word_embedding_dimension
        self.vocab_size = config.vocab_size
        self.batch_size = config.batch_size
        self.seq_length = config.sequence_length
        self.dropout = config.dropout

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)

        # convolution layer 1
        self.conv1 = nn.Conv1d(in_channels=self.seq_length, out_channels=64, kernel_size=3)
        # convolution layer 2
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        # max pooling layer
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        # convolution layer 3
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3)
        # convolution layer 4
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3)
        # global averaging pool.
        self.avgpool = nn.AvgPool1d(94)

        # drop out
        self.dropout = nn.Dropout(self.dropout)
        # output to fully connected layer.
        self.fc = nn.Linear(8, 1)
        # sigmoid transform of output
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # add sequence of convolution and max pooling layers.
        x = self.embedding(x)
        x = x.reshape(self.batch_size, self.seq_length,
                      self.embedding_dim)  # reshape it to [batch, seq_length, 300 embedding size]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.avgpool(x)

        # flattening the output of the final pooling layer to feed the fully connected layer.
        x = x.view(x.shape[0] * x.shape[2], -1)
        # dropout
        x = self.dropout(x)
        # fully connected layer
        x = self.fc(x)

        x = self.sig(x)

        return x
