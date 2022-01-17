from math import inf
from rect_world import *
from agents import *
import random

# fix the random seed
random.seed(1337)

def gen_specs(train_set, listener_budget):
    # the literal listener
    l0 = L_SAMPLE(listener_budget)
    s_ce = S_CE(l0)

    test1 = test[875]
    print (test1)
    utts = s_ce.utter(test1, 10)

if __name__ == '__main__':
    print ("attempting to make all programs")
    all_progs = make_all_progs()
    train, test = split_progs(all_progs)
    print ("finished making all programs")

    # the literal listener
    l0 = L_SAMPLE(10)
    s_ce = S_CE(l0)

    test1 = test[875]
    print (test1)
    utts = s_ce.utter(test1, 10)
    print (utts)
    print (l0(utts))
    assert 0

    # make the speaker and listener    
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