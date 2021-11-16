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
                    if rect_is_valid(T,B,L,R):
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

# train l0_nn on top of s0
# build s_ce on top of l0_nn to generate informative specs
# train l1_nn on top of the informative specs from s_ce
def train_S_L(s0, l0_nn, s_ce, l1_nn):
    all_progs = make_all_progs()
    train, test = split_progs(all_progs)

    # basically until forever
    for train_iter in range(0, 100000000):
        
        # visualize a few instances from the test set to get a feel for how well the model is doing
        if train_iter % 1000 == 0:
            print ("iteration:", train_iter, "==============")
            for t_idd in range(10):
                random_test_prog = random.choice(test)
                utts = s_ce(random_test_prog, 6, diagnose=True)
                Rect(*random_test_prog).draw(f'tmp/{t_idd}_rect_orig.png', utts)
                inferred_params_l0 = l0_nn(utts)
                inferred_params_l1 = l1_nn(utts)
                for k, inferred_params in enumerate([inferred_params_l0]):
                    if len(inferred_params) > 0:
                        Rect(*inferred_params[0]).draw(f'tmp/{t_idd}_rect_inferred_l{k}.png', utts)
            print ("saving models")
            l0_nn.save(f'tmp/l0_nn.pth')
            l1_nn.save(f'tmp/l1_nn.pth')

        # actually do the training ! 
        batch = batch_sample(train, 64)
        # create a training batch consisting of utterance-program pairs
        s0_batch_utterances = []
        sce_batch_utterances = []
        batch_programs = []
        for prog_id, prog in enumerate(batch):
            batch_programs.append(prog)            
            # get a random utterance length from 0 to 10
            utterance_len = random.randint(0,10)
            s0_batch_utterances.append(s0(prog, utterance_len))
            sce_batch_utterances.append(s_ce(prog, utterance_len))

        # train the literal listener and pragmatic listener
        loss_l0 = l0_nn.train(s0_batch_utterances, batch_programs)
        loss_l1 = l1_nn.train(sce_batch_utterances, batch_programs)

        # some stats
        if train_iter % 100 == 0:
            print (f"batch {train_iter} l0_loss {loss_l0} l1_loss {loss_l1}")
        

if __name__ == '__main__':
    # train_S_L(S0(), L_NN_F())
    
    # the literal speaker
    s0 = S0()
    # make the neural literal listener l0_nn
    l0_nn = L_NN_F()
    # make a cegis s0
    s_ce = S_CE(l0_nn)
    # make the neural pragmatic listener l1_nn
    l1_nn = L_NN_F()

    train_S_L(s0, l0_nn, s_ce, l1_nn)