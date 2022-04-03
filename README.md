# DSRWG-SentiModel
Repository for scripts used for the Sentiment Analysis model with deep learning prototype for the Chatbot Analytics workstream in IA working group.
Current prototype is a CNN based on this [paper](https://arxiv.org/ftp/arxiv/papers/2006/2006.03541.pdf)

![image](https://user-images.githubusercontent.com/50050912/157023128-fa016fbe-f2c5-463e-b900-2120301690f2.png)

### Dataset
- The datasets used to train, validate and test the model(s) are listed below:
    
    <u>IMDB reviews<u>
    - https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
    - https://www.kaggle.com/c/word2vec-nlp-tutorial/data
  
    <u>Amazon reviews polarity and Yelp reviews polarity datasets<u>
    - https://course.fast.ai/datasets#nlp
  
    <u>Sentiment140 tweets data<u>
    - http://help.sentiment140.com/for-students
      
        
### Result
The following is the best scores achieved by the model when trained on individual datasets with the sentence sequence length hyperparameter (changes to other hyperparameters did not have as much of an impact):
     
![image](https://user-images.githubusercontent.com/50050912/161446073-a0860c33-2aea-410d-8ec7-666d7ed32a33.png)


Other constant (unchanged) hyperparameters:
        
        <u>learning rate --0.001<u>
        <u>Optimizer --AdamOptimizer<u>
        <u>conv layers (provided in image and paper above)<u>
        <u>criterion --BinaryCrossEntropy<u>

            
### Next Steps
See: [project tracker](https://capgemini.sharepoint.com/:x:/r/sites/DataScienceResearchDSRWG/Shared%20Documents/General/project_tracker.xlsx?d=w0033b3549f974dbc89cb0a711a0c8e73&csf=1&web=1&e=x7JwJK)
