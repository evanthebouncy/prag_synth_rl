from rect_world import *
from agents import *
from interact import run_until_sat, make_all_progs, split_progs
import random
import os, sys
import pickle
import numpy as np 

# a few hard-coded strategies that are plausible for humans
class CropSpeaker:

    def __init__(self, listener):
        self.listener = listener

    def utter_once(self, prog, past_utterances):
        rect = Rect(*prog)
        if len(past_utterances) == 0:
            x,y = prog[2], prog[0]
            return ((x,y), rect(x,y))
        if len(past_utterances) == 1:
            x,y = prog[3], prog[1]
            return ((x,y), rect(x,y))
        else:
            result, budget = run_until_sat(self.listener, past_utterances)
            proposed_rect = Rect(*result[0])

            old_coords = set([x[0] for x in past_utterances])
            all_coords = []
            for x in range(MAX_LEN):
                for y in range(MAX_LEN):
                    # check if the coordinate is new
                    if (x,y) not in old_coords:
                        all_coords.append((x,y))
            # shufflt the coordinates to a random order
            random.shuffle(all_coords)
            for coord in all_coords:
                if proposed_rect(*coord) != rect(*coord):
                    return coord, rect(*coord)

        return None

def test_interactive_efficiency(speaker, listener, prog):
    
    total_guess = 0
    utterances = []
    for i in range(10):
        # try to recover the hypothesis
        result, budget = run_until_sat(listener, utterances)
        total_guess += budget
        # stop if we got the right answer
        if result[0] == prog:
            break
        # extend the utterance
        utterance = speaker.utter_once(prog, utterances)
        utterances.append(utterance)

    return len(utterances), total_guess

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

    results = {}
    for l_name in listeners_dict:
        if l_name == 'L(S(L))_5000':
            continue

        results[l_name] = {
            'num_utter' : [],
            'total_guess' : []
        }
        listener = listeners_dict[l_name]
        speaker = CropSpeaker(listener)
        for prog in test:
            num_utter, total_guess = test_interactive_efficiency(speaker, listener, prog)
            results[l_name]['num_utter'].append(num_utter)
            results[l_name]['total_guess'].append(total_guess)
    
    # save the results
    with open('tmp/results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # print the results
    for l_name in results:
        print (l_name)
        num_utter = results[l_name]['num_utter']
        total_guess = results[l_name]['total_guess']
        # get the mean and standard error
        num_utter_mean = np.mean(num_utter)
        num_utter_se = np.std(num_utter) / np.sqrt(len(num_utter))
        total_guess_mean = np.mean(total_guess)
        total_guess_se = np.std(total_guess) / np.sqrt(len(total_guess))
        print ('num_utter:', num_utter_mean, '+/-', num_utter_se)
        print ('total_guess:', total_guess_mean, '+/-', total_guess_se)


