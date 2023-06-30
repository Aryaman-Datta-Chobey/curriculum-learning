import random
import math
import glob
import csv
import os
import torch
from tokenizers import (
    normalizers, #1
    Tokenizer,
    pre_tokenizers, #2
    models, #3
    trainers,#3
    processors, #4
    decoders, #5
)

## INITIALIZE VARIABLES
random.seed(5) # set random seeed
## Specify file names

csv_file = "./data/preprocessed/babylm_sent_train.csv"

csv_dir = "./data/preprocessed/"
output_dir = "./data/lstm_reviewers/"



## DEFINE FUNCTIONS

def loadFromCSV(csvfname):
  """
  Input: csv fname sentid, sentence, domain, sentlen ...
  Output:
    domain_sent_dict: {domain1: {sentid1: sent1, sentid2: sent2 ... },
               domain2: {sentid1: sent1, sentid2: sent2 ... }
      }}
    domain_sentfreq_dict: {domain1: sent count of domain1, domain2: sent count of domain1}
    domain_wordfreq_dict: {domain1: word count of domain1, domain2: word count of domain1}

  """
  with open(csvfname, 'r') as file:
    domain_sent_dict = {}
    domain_sentfreq_dict = {}
    domain_wordfreq_dict = {}

    reader = csv.DictReader(file, delimiter="\037")
    for row in reader:

      if row["domain"] not in domain_sent_dict:
        domain_sentfreq_dict[row["domain"]] = 1
        domain_wordfreq_dict[row["domain"]] = int(row["length"])

        new_sent_dict = {}
        new_sent_dict[row["sentid"]] = row["sentence"]
        domain_sent_dict[row["domain"]] = new_sent_dict

      else:
        domain_sent_dict[row["domain"]][row["sentid"]] = row["sentence"]
        domain_sentfreq_dict[row["domain"]] += 1
        domain_wordfreq_dict[row["domain"]] += int(row["length"])

  return domain_sent_dict, domain_sentfreq_dict, domain_wordfreq_dict



def sampleFromCSV(num_samples, domain_dict):
  """
  Input:
    num_samples: (int) number of sample groups

    dat_dict: dictionary in the form of domain_sent_dict output from
      loadFromCSV

  Output:
    sample_sent_list: list of dictionaries in style of input dictionary
      (domain_sent_dict). number of dictionaries equivalent to num_samples

    sample_sentfreq_list: list of dictionaries in style of
      domain_sentfreq_list output from loadFromCSV. number of dictionaries
      equivalent to num_samples, and the dictionary at each index corresponds
      to the dictionary at the same index of the sample_sent_list output

    sample_wordfreq_list: list of dictionaries in style of
      domain_wordfreq_list output from loadFromCSV. number of dictionaries
      equivalent to num_samples, and the dictionary at each index corresponds
      to the dictionary at the same index of the sample_sent_list output
  """
  sample_sent_list = []
  sample_sentfreq_list = []
  sample_wordfreq_list = []

  # add dict for each sample to sample sent_list
  for i in range(num_samples):
    sample_sent_list.append({})
    sample_sentfreq_list.append({})
    sample_wordfreq_list.append({})

  for domain in list(domain_dict.keys()):

    # add domain dict to sample dicts
    for i in range(num_samples):
      sample_sent_list[i][domain] = {}
      sample_sentfreq_list[i][domain] = 0
      sample_wordfreq_list[i][domain] = 0

    # make and shuffle sentids from dat_dict
    sentid_list = list(domain_dict[domain].keys())
    random.shuffle(sentid_list)

    # split shuffled sentids into the samples
    current_sample = 0
    while len(sentid_list) != 0:

      # get sentid and sentence
      sentid = sentid_list.pop(0)
      sentence_from_id = domain_dict[domain][sentid]

      # fill in sample dict info
      sample_sent_list[current_sample][domain][sentid] = sentence_from_id
      sample_sentfreq_list[current_sample][domain] += 1
      sample_wordfreq_list[current_sample][domain] += len(sentence_from_id)

      # reset current sample
      current_sample += 1
      if current_sample == num_samples:
        current_sample = 0

  return sample_sent_list, sample_sentfreq_list, sample_wordfreq_list



def corpusCreation(dat_dict):
  """
  Input:
    dat_dict: dictionary in the form of domain_sent_dict output from
      loadFromCSV

  Output:
    shuffled_sent_dict: {sentid1: sent1, sentid2: sent2, ... sentidN: sentN}
      (order of sentids will however be shuffled, and very unlikely to be
      sequential)

    shuffled_sent_list: list of sentences shuffled in the same order as
      shuffled_sent_dict above
  """
  shuffled_sent_dict = {}
  shuffled_sent_list = []

  # make unshuffled dict
  domainless_dat_dict = {}
  for domain in dat_dict:
    for sentid in dat_dict[domain]:
      domainless_dat_dict[sentid] = dat_dict[domain][sentid]

  # make and shuffle list
  sentid_list = list(domainless_dat_dict.keys())
  random.shuffle(sentid_list)

  # fill shuffled_sent_list and shuffled_sent_dict
  for sentid in sentid_list:
    sent = domainless_dat_dict[sentid]
    shuffled_sent_dict[sentid] = sent
    shuffled_sent_list.append(sent)

  return shuffled_sent_list, shuffled_sent_dict

def createFiles(tokenizer, csv_dir, output_dir, num_samples):
    """
    NOTE: csv_dir must contain one .csv file ending in _train and one
      .csv file ending in _dev. It may also contain a .csv file ending in _test

    """
    print(type(num_samples))
    # make and save vocab
    vocab = tokenizer.get_vocab().keys()
    file = open(output_dir + "BPEVocab.txt","w")
    file.write('\n'.join(vocab))
    file.close


    for filename in os.listdir(csv_dir):
      if filename.split(".")[-1] != "csv":
        continue

      fileType = filename.split("_")[-1][:-4]
      print(fileType)
      if fileType != "train" and fileType != "dev" and fileType != "test":
        continue

      # training data files for reviewers
      if fileType == "train":
        train_dat_dict, tSentFreqDict, tWordFreqDict = loadFromCSV(csv_dir + filename)
        print(csv_dir + filename)
        sampleSentDictsList, sentFreqList, wordFreqList = sampleFromCSV(num_samples, train_dat_dict)
        for i in range(len(sampleSentDictsList)):

          # make txt file for each reviewer model
          sent_list, sent_dict = corpusCreation(sampleSentDictsList[i])
          tokenized_filename = "rev" + str(i) + "_tokenized.txt"
          createSentsFile(tokenized_filename, sent_list, output_dir)

          # make csv file for each reviewer model
          revCSV_filename = "rev" + str(i) + "_sentids.csv"
          makeRevCSV(revCSV_filename, sent_dict, output_dir)

      # dev and test data files
      else: # (dev or test)
        dat_dict, sentFreqDict, wordFreqDict = loadFromCSV(csv_dir + filename)
        sent_list, sent_dict = corpusCreation(dat_dict)
        tokenized_filename = "tokenized_" + fileType
        createSentsFile(tokenized_filename, sent_list, output_dir)

    return

def createSentsFile(fname, sents, output_dir):
    print(fname)
    encodedSents = []
    for sent in sents:
        encodedSent = " ".join((tokenizer.encode(sent)).tokens)
        encodedSents.append(encodedSent)
    print(len(sents))
    # print(sents[0])
    # print(encodedSents[0])
    # print(sents[len(sents) -1])
    # print(encodedSents[len(sents) -1])
    with open(output_dir + fname, "w") as f:
        f.write('\n'.join(encodedSents))
    return



def makeRevCSV(fname, sent_dict, output_dir):
    """
    NOTE: takes filename as a str containing the model name followed by
      "_sentids.csv". No underscores may be present in the model name.
    """
    with open(output_dir + fname, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "sentid", "sentnum", "sentence"])
        for i,sentid in enumerate(sent_dict.keys()):
          model = fname.split("_")[0]
          row = [model, sentid, i, sent_dict[sentid]]
          writer.writerow(row)
    return






## TRAIN TOKENIZER


tokenizer = Tokenizer(models.BPE()) # tokenizer objects are intialized with a model argument which specifies the algorithm they will use for tokenization
# GPT-2 does not use a normalizer, so we go directly pre-tokenization
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False) #prevents default addition of space char at the beginning of a sentence

#3 passing input to model and training tokenizer
domain_sent_dict, domain_sentfreq_dict, domain_wordfreq_dict= loadFromCSV(csv_file)
train = corpusCreation(domain_sent_dict)

trainer = trainers.BpeTrainer(vocab_size=50272, special_tokens=["<|endoftext|>", '<image>', '</c>', '<PERSON>']) # EOS is apparently the only special token for BPE (rest based on baseline github) , set vocab size according to babyLM baseline model config val
tokenizer.train_from_iterator(train, trainer=trainer)

#4
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

#5
tokenizer.decoder = decoders.ByteLevel()

#possibly not working
tokenizer_dir = 'BabyLM_10M_Tokenizer.pt'
torch.save(tokenizer, tokenizer_dir)



## CREATE FILES

createFiles(tokenizer, csv_dir, output_dir, 5)


