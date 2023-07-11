import random
import csv
import math


## FUNCTIONS TO LOAD DATA

def loadFromCSV(csvfname, separator = "\037"):
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

    reader = csv.DictReader(file, delimiter=separator)
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


## FUNCTIONS 

## FUNCTIONS TO CREATE CURRICULA

def root_p(time_step, c0, T,p=2):
    """
    Input:
    - time_step (int) current time step/ batch; t in eqn 8
    - T (int) max time steps/ batches; T in eqn 8
    - c0 (int) # tokens in data domain at the first time step
    - p (int/ float): exp/ root val. Default to 2.

    Output:
    proportion of data to be sampled

    """
    val = (time_step * (1-c0**p)/T) + c0**p
    sqrt_val = val**(1/p)
    prop = min(1, sqrt_val)
    return prop

def sample_sents(curr_dat, batch_size):
    """
    Input:
    - curr_dat (list of sentence, sentid pairs to be sampled at current step)
    - bsize (int; # sentences at each time step, which is the same as batch size)
    """
    if batch_size > len(curr_dat):
        print(batch_size, curr_dat)

    assert batch_size <= len(curr_dat) # Cannot sample more than the number of items

    return(random.sample(curr_dat, batch_size))

def read_file(fname, separator="\037"):
    with open(fname, 'r') as f:
        dat = f.readlines()

    output = []
    header = dat[0].split(separator)
    sent_ind = header.index('sentence') #check this
    id_ind = header.index('sentid')

    for line in dat[1:]:
        curr_dat = line.split(separator)
        curr_sent = curr_dat[sent_ind]
        curr_id = curr_dat[id_ind]
        output.append([curr_id, curr_sent])

    return(output)

def create_curriculum(input_fname, output_fname, T, c0, p, batch_size, separator="\037"):
    dat = read_file(input_fname, separator)
    num_total = len(dat)
    print('Num total sents', len(dat))
    csv_fname = open(f'{output_fname}.csv', 'w')
    txt_fname = open(f'{output_fname}.txt', 'w')
    writer = csv.writer(csv_fname, delimiter=separator)
    writer.writerow(['sentid', 'sentence']) #add header

    for i in range(T):
        curr_prop = root_p(i, c0, T)
        curr_ind = int(num_total*curr_prop)
        curr_dat = dat[:curr_ind]
        curr_sents = sample_sents(curr_dat, batch_size)
        for item in curr_sents:
            writer.writerow(item)
            txt_fname.write(item[1]+'\n')

    csv_fname.close()
    txt_fname.close()

if __name__ == '__main__':
    T = int(input('T (num of training steps): ').strip())
    p = float(input('p (base of the root): ').strip())
    random.seed(17)
    create_curriculum(input_fname='./data/train_by_sum_surp.csv',
                      output_fname=f'./data/curriculum_lstmrev_sumsurp_root{p}_T{T}',
                      T=T,
                      c0=0.01,
                      p = p,
                      batch_size=32,
                      separator="\037")




