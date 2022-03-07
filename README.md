# DSRWG-SentiModel
Repository for scripts used for the Sentiment Analysis model with deep learning prototype for the Chatbot Analytics workstream in IA working group.
Current prototype is a CNN based on: https://arxiv.org/ftp/arxiv/papers/2006/2006.03541.pdf

![image](https://user-images.githubusercontent.com/50050912/157023128-fa016fbe-f2c5-463e-b900-2120301690f2.png)

### Dataset
- The dataset used for prototype is combined from two separate movie review datasets (50k records and 25k records) which gives a total of 75k records for training, validation and testing. The datasets were used in the orginal paper and taken from:
  - https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
  - https://www.kaggle.com/c/word2vec-nlp-tutorial/data

### Result
- The current initial prototype version has an accuracy on the test set of 0.843.

### Next Steps
- Word2Vec generated embedding matrix has not yet been used to train the model. As shown in the paper referenced above, this should improve the model accuracy.
- Trying out different optimizers could also be tested.
- Trying out a different model (e.g. RNN with LSTM) could also be attempted.  
