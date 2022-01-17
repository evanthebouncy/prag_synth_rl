from rect_world import *
from agents import *
from learn import *
import random

import os, sys

def run_until_sat(listener, utterance):
    budget = 1
    result = []
    while len(result) == 0:
        budget *= 2
        result = listener.enumerate(utterance, budget)
    return result, budget


# do the communciation efficiency test
def test_efficiency(speaker, list_of_listeners, test_progs):
    
    listener_utters_required = [[] for _ in range(len(list_of_listeners))]

    def get_num_utters_required(speaker, listener, prog):
        for utter_len in range(10):
            utterance = speaker(prog, utter_len)
            result, budget = run_until_sat(listener, utterance)
            if result[0] == prog:
                return utter_len
        return 10

    for i, prog in enumerate(test_progs):
        print ("testing on ", i)
        # for all the listeners
        for i, listener in enumerate(list_of_listeners):
            listener_utters_required[i].append(get_num_utters_required(speaker, listener, prog))

    return listener_utters_required


if __name__ == '__main__':
    random.seed(1337)
    all_progs = make_all_progs()
    train, test = split_progs(all_progs)

    # open the directory of saved listener models
    model_dir = 'tmp/models/'
    # list of all the listener models
    model_files = [x for x in os.listdir(model_dir) if 'pth' in x]
    print (model_files)
    
    # load all of them
    listeners_dict = {}
    for model_file in model_files:
        nn_listener = L_NN_F(budget = 10)
        path = model_dir + model_file
        nn_listener.load(path)
        listeners_dict[model_file.split(".")[0]] = nn_listener

    # break the seed used to split train/test and get a new one
    random.seed()
    random.shuffle(test)
    prog = test[0]
    prog = (4,6,0,3)
    print (prog)
    
    utterance = []
    while True:
        # visualize the prog, and the two guesses
        Rect(*prog).draw(f'tmp/interact_orig.png', utterance)
        for listener_name in listeners_dict:
            param_guess, budget = run_until_sat(listeners_dict[listener_name], utterance)
            print (f"budgets spent {listener_name}:{budget}")
            Rect(*param_guess[0]).draw(f'tmp/interact_{listener_name}.png', utterance)
        # get the next utterance
        user_utt = input('next utterance: ')
        if user_utt == 'restart':
            utterance = []
            continue
        else:
            x,y = eval(user_utt)
            b = Rect(*prog)(x,y)
            utterance.append(((x,y), b))
