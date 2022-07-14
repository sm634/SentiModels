

class Config(object):
    def __init__(self, word_embedding_dimension=300, vocab_size=20000, epoch=5, sequence_length=40, learning_rate=0.01,
                 batch_size=64, dropout=0.5, preprocess=False, linear_out=1):
        self.word_embedding_dimension = word_embedding_dimension
        self.vocab_size = vocab_size
        self.epoch = epoch
        self.sequence_length = sequence_length
        self.lr = learning_rate
        self.batch_size = batch_size
        self.dropout = dropout
        self.preprocess = preprocess
        self.linear_out = linear_out


