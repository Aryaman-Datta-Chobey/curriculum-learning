import random
import csv
import math

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

def create_curriculum(input_fname, output_fname, T, c0, batch_size, separator="\037"):
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


