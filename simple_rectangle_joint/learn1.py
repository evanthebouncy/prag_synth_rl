from math import inf
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

# visualize some rectangles from the test set
def visualize_test(speaker, n_utter, listener, test, n_test_rects):
    global_proposed_set = set()
    individual_proposed_count = 0
    for t_idd in range(n_test_rects):
        random_test_prog = random.choice(test)
        utts = speaker(random_test_prog, n_utter)
        Rect(*random_test_prog).draw(f'tmp/{t_idd}_rect_orig.png', utts)
        rec_rect_params = listener(utts)

        for ijk, params in enumerate(rec_rect_params):
            Rect(*params).draw(f'tmp/{t_idd}_inferred_sat_{ijk}.png', utts)  

# train neural listener
def train_L(s_ce, l1_nn, train, test, n_train_iter, save_path):

    # basically until forever
    for train_iter in range(0, n_train_iter):

        # actually do the training ! 
        batch = batch_sample(train, 64)
        # create a training batch consisting of utterance-program pairs
        sce_batch_utterances = []
        batch_programs = []
        for prog_id, prog in enumerate(batch):
            batch_programs.append(prog)            
            # get a random utterance length from 0 to 10
            utterance_len = 10
            sce_batch_utterances.append(s_ce.utter(prog, utterance_len))

        # train the literal listener and pragmatic listener
        loss_l1 = l1_nn.train(sce_batch_utterances, batch_programs)

        # some stats
        if train_iter % 500 == 0:
            print (f"batch {train_iter} {save_path}_loss {loss_l1}")
            visualize_test(s_ce, 10, l1_nn, test, 10)
            l1_nn.save('tmp/'+save_path)

    return l1_nn

def evaluate_on_test(s0, l0_listeners_dicts, test):
    ret = dict()
    for key in l0_listeners_dicts:
        ret[key] = []

    for test_prog in test:
        utts = s0(test_prog, 3)
        for l0_hyper_param in l0_listeners_dicts:
            inferred_params_l0 = l0_listeners_dicts[l0_hyper_param](utts)
            uniques = len(set([str(x) for x in inferred_params_l0]))
            ret[l0_hyper_param].append(uniques)
    
    return ret


if __name__ == '__main__':
    print ("attempting to make all programs")
    all_progs = make_all_progs()
    train, test = split_progs(all_progs)
    print ("finished making all programs")

    # train a listener on spec generated by s0
    l_s0 = L_NN_F(budget=10)
    train_L(S0(), l_s0, train, test, 5000, 'L(S0)_5000.pth')

    # train a l1 on spec generated by s_ce(l0)
    l_s_l0 = L_NN_F(budget=10)
    train_L(S_CE(L_SAMPLE(10)), l_s_l0, train, test, 5000, 'L(S(L0))_5000.pth')

    # train a l1 on spec generated by s_ce(l1)
    l_s_l = L_NN_F(budget=10)
    train_L(S_CE(l_s_l), l_s_l, train, test, 5000, 'L(S(L))_5000.pth')

    # train a l1 on spec generated by s_ce(l(s0))
    l_s_l_s0 = L_NN_F(budget=10)
    train_L(S_CE(l_s0), l_s_l_s0, train, test, 5000, 'L(S(L(S0)))_5000.pth')

    # train a l1 on spec generated by s_ce(l1, l0)
    l_s_ll0= L_NN_F(budget=5)
    # this is important, we need to run l1_nn first to have it propose solutions before l0
    train_L(S_CE(L_Ensemble([l_s_ll0, L_SAMPLE(5)])), l_s_ll0, train, test, 5000, 'L(S(L,L0))_5000.pth')

