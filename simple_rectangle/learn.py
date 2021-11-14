from rect_world import *
from agents import *
import random

# fix the random seed
random.seed(1337)

# make all programs
# a program is simply a tuple of (T,B,L,R), 
# we will simply use rect on it to turn it into a callable func
def make_all_progs():
    progs = []
    for T in range(MAX_LEN):
        for B in range(MAX_LEN):
            for L in range(MAX_LEN):
                for R in range(MAX_LEN):
                    # make sure the rect is valid
                    if T+1 < B and L+1 < R:
                        progs.append((T,B,L,R))
    return progs

# split the progs into train and test sets
def split_progs(progs):
    train_progs = []
    test_progs = []
    for prog in progs:
        if random.random() < 0.5:
            train_progs.append(prog)
        else:
            test_progs.append(prog)
    return train_progs, test_progs

# a simple batch sampler
def batch_sample(progs, batch_size):
    batch = []
    for _ in range(batch_size):
        prog = random.choice(progs)
        batch.append(prog)
    return batch

def test(speaker, listener):
    # print ("iteration:", train_iter, "==============")
    # n_correct = 0
    # total_attempts = 0    
    # total_utt_len = 0        
    # for j, prog in enumerate(test):
    #     utterance_len = random.randint(2, 4)
    #     utts = speaker(prog, utterance_len)
    #     total_utt_len += len(utts)
    #     rect_inferred, num_attempts = listener(utts)
    #     if rect_inferred == prog:
    #         n_correct += 1
    #     total_attempts += num_attempts

    #     print ("num_correct ", n_correct, "total ", j, "prog ", prog, "utts ", total_utt_len, 
    #     " ", utts)

    # print some stats
    # print ("batch training loss ", loss)
    # print ("test accuracy (prog == recovered_prog):", n_correct / len(test))
    # # print the average utt length
    # print ("avg. utt length:", total_utt_len / len(test))
    # print ("avg. num. attempts:", total_attempts / len(test))
    pass

# train the listener against the uninformative speaker
def train_S_L(speaker, listener):
    all_progs = make_all_progs()
    train, test = split_progs(all_progs)

    # hackity hack
    s_ce = S_CE(l_nn)

    for train_iter in range(0, 100000000):
        
        # visualize a few instances from the test set to get a feel for how well the model is doing
        if train_iter % 10000 == 0:
            print ("iteration:", train_iter, "==============")
            for t_idd in range(10):
                random_test_prog = random.choice(test)
                utts = s_ce(random_test_prog, 6)
                Rect(*random_test_prog).draw(f'tmp/{t_idd}_rect_orig.png', utts)
                inferred_params = listener(utts)
                if len(inferred_params) > 0:
                    Rect(*inferred_params[0]).draw(f'tmp/{t_idd}_rect_inferred_nn.png', utts)

        # actually do the training ! 
        batch = batch_sample(train, 32)
        # create a training batch consisting of utterance-program pairs
        batch_utterances = []
        batch_programs = []
        for prog_id, prog in enumerate(batch):
            # print (f"\niter: {train_iter} prog_num: {prog_id} cur_prog: {prog} ")
            # get the utterance
            # get a random utterance length from 0 to 10
            utterance_len = random.randint(0,10)
            
            # randomly try both speakers
            # utts = speaker(prog, utterance_len) if random.random() < 0.5 else s_ce(prog, utterance_len)
            # that turned out to be a bad idea, so we'll just use the normal speaker
            utts = speaker(prog, utterance_len)
            # print (f"utt_len : {utterance_len} utt : {utts}")
            # print ("rec prog ", listener(utts))
            batch_utterances.append(utts)
            batch_programs.append(prog)
        # take the gradients
        loss = listener.train(batch_utterances, batch_programs)
        if train_iter % 100 == 0:
            print (f"batch {train_iter} training loss ", loss)
        

if __name__ == '__main__':
    # train_S_L(S0(), L_NN_F())

    # initialize a neural listener
    l_nn = L_NN_F()
    # make a cegis speaker
    s_ce = S_CE(l_nn)
    s0 = S0()
    train_S_L(s0, l_nn)