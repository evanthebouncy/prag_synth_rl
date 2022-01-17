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
        valid_rect_params = listener.enumerate_valid_rects(utts)

        # add all the valid params to global_proposed_set
        for rect_params in valid_rect_params:
            global_proposed_set.add(tuple(rect_params))
            individual_proposed_count += 1

        for ijk, params in enumerate(valid_rect_params):
            if Rect(*params).consistent(utts):
                Rect(*params).draw(f'tmp/{t_idd}_inferred_sat_{ijk}.png', utts)
            else:
                Rect(*params).draw(f'tmp/{t_idd}_inferred_unsat_{ijk}.png', utts)

    print (f"global_proposed_set_count: {len(global_proposed_set)}")
    print (f"individual_proposed_count: {individual_proposed_count}")   

# train l0_nn on top of s0
def train_L0_joint(s0, l0_nn, train, test, n_train_iter):

    # basically until forever
    for train_iter in range(0, n_train_iter):
        
        # if train_iter % (n_train_iter-1) == 0:
        #     print ("iteration:", train_iter, "==============")
        #     print ("saving models")
        #     l0_nn.save(f'tmp/l0_nn_f.pth')

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
            # fixing it to 3 and see if it helps
            utterance_len = 3
            s0_batch_utterances.append(s0(prog, utterance_len))

        # train the literal listener and pragmatic listener
        loss_l0 = l0_nn.train(s0_batch_utterances, batch_programs)

        # some stats
        if train_iter % 1000 == 0:
            print (f"batch {train_iter} l0_loss {loss_l0}")

    return l0_nn

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
    
    # the literal speaker
    s0 = S0()

    # good_budget = 1000
    # good_alpha = 0.01
    # l0_nn = L_NN_J(good_budget, good_alpha)
    # l0_nn = train_L0_joint(s0, l0_nn, train, test, 1000000)
    
    # visualize_test(s0, 3, l0_nn, test, 10)

    listener_dict = {'10' : L_SAMPLE(10), '100' : L_SAMPLE(100), '1000' : L_SAMPLE(1000)}
    eval_dict = evaluate_on_test(s0, listener_dict, test)
    for key in eval_dict:
        print (f"{key} : {np.mean(eval_dict[key])}")

    assert 0

    # build an ensemble of listeners
    ensembles = []
    for _ in range(1):
        print ("current on ", _)
        l0_nn = L_NN_J(good_budget, good_alpha)
        l0_nn = train_L0_joint(s0, l0_nn, train, test)        
        ensembles.append(l0_nn)
    
    l_ensemble = L_Ensemble(ensembles)
    
    listener_dict = {'one' : ensembles[0], 'ten' : l_ensemble, 'all' : L_FULL()}


    eval_dict = evaluate_on_test(s0, listener_dict, test)

    for key in eval_dict:
        print (f"{key} : {np.mean(eval_dict[key])}")


    budget_search = [10, 100, 1000, 10000]
    alpha_search = [0.1, 0.01, 0.001, 0.0001]

    budget_search = [1000]
    alpha_search = [0.1, 0.01]

    agg_results = dict()
    for budget in budget_search:
        for alpha in alpha_search:
            agg_results[(budget, alpha)] = []

    for i in range(40):
        l0_listeners_dicts = dict()
        for budget in budget_search:
            for alpha in alpha_search:
                l0_nn = L_NN_J(budget, alpha)
                l0_nn = train_L0_joint(s0, l0_nn, train, test)
                l0_nn.save(f'tmp/l0_nn_f_{budget}_{alpha}.pth')
                l0_listeners_dicts[(budget, alpha)] = l0_nn  

        eval_dict = evaluate_on_test(s0, l0_listeners_dicts, test)
        for key in eval_dict:
            avg = sum(eval_dict[key])/len(eval_dict[key])
            agg_results[key].append(avg)

        print (agg_results)