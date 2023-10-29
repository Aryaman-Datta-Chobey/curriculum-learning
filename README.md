## Curriculum Learning Experiments

Setting up:

1. Clone this repository
2. Unzip the files in data/preprocessed

### Scoring Difficulty 
![image](https://github.com/Aryaman-Datta-Chobey/curriculum-learning/assets/97906694/41c52e08-30e8-4fcf-81fb-ad01141fcf07)
#### LSTM reviewer model creation
![image](https://github.com/Aryaman-Datta-Chobey/curriculum-learning/assets/97906694/53db9f00-6956-44f2-a147-e6a9ee7c2cad)
 Data for LSTM training loop is generate by  Rev_lstm_creation.py and save to data/lstm_reviewers  : 
1. Reads data from unzipped folders in data/preprocessed 
2.  Trains from scratch a BPE tokenizer on the babyLM train set  and locally saves it as a tokenizer object and outputs the tokenizer’s vocab file (saved as BPEvocab).
3.  Tokenizes test and validation sets and splits the  train set into 5 tokenized metasets which are all saved for  training LSTM reviewer models.
4.  Each training sentence is labelled by a sentID for tracking and debugging


