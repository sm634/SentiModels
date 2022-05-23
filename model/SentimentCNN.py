import torch.nn as nn
import torch.nn.functional as F


class SentimentCNN(nn.Module):
    """
    The CNN prototype model used for sentiment analysis
    """

    def __init__(self, vocab_size, output_size, embedding_dim, batch_size, seq_length):
        super(SentimentCNN, self).__init__()
        """
        :vocab_size: [int] number of words to embed to index from lookup table.
        :output_size: [int] this will be 1 as only a single sigmoid transformation output will be provided per input.
        :embedding_dim: [int] the vector dimension size representing a word in the embedding matrix.
        :batch_size: [int] the input batch sample size.
        :seq_length: [int] the fixed sentence length of words (padded or trimmed) to feed the conv layer.
        """

        # embedding layer
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # convolution layer 1
        self.conv1 = nn.Conv1d(seq_length, 64, 3)
        # convolution layer 2
        self.conv2 = nn.Conv1d(64, 32, 3)
        # max pooling layer
        self.pool1 = nn.MaxPool1d(3, 3)
        # convolution layer 3
        self.conv3 = nn.Conv1d(32, 16, 3)
        # convolution layer 4
        self.conv4 = nn.Conv1d(16, 8, 3)
        # global averaging pool.
        self.avgpool = nn.AvgPool1d(94)

        # drop out
        self.dropout = nn.Dropout(0.2)
        # output to fully connected layer.
        self.fc = nn.Linear(8, output_size)
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

        #         # flattening the output of the final pooling layer to feed the fully connected layer.
        x = x.view(x.shape[0] * x.shape[2], -1)
        # dropout
        x = self.dropout(x)
        # fully connected layer
        x = self.fc(x)

        x = self.sig(x)

        return x
