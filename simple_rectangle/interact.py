from rect_world import *
from agents import *
from learn import *
import random

def run_until_sat(listener, utterance):
    budget = 1
    result = []
    while len(result) == 0:
        budget *= 2
        result = listener.enumerate(utterance, budget)
    return result, budget

if __name__ == '__main__':
    # make the two agents
    l0_nn = L_NN_F()
    l0_nn.load('tmp/l0_nn_saved.pth')
    # make the neural pragmatic listener l1_nn
    l1_nn = L_NN_F()
    l1_nn.load('tmp/l1_nn_saved.pth')

    # make the test set and get a rectangle from it
    all_progs = make_all_progs()
    train, test = split_progs(all_progs)
    
    # break the seed used to split train/test and get a new one
    random.seed()
    prog = random.choice(test)

    utterance = []
    while True:
        # visualize the prog, and the two guesses
        Rect(*prog).draw(f'tmp/interact_orig.png', utterance)
        param_l0_guess, l0_budget = run_until_sat(l0_nn, utterance)
        param_l1_guess, l1_budget = run_until_sat(l1_nn, utterance)
        print (f"budgets spent l0:{l0_budget}  l1:{l1_budget}")
        for i, param in enumerate([param_l0_guess[0], param_l1_guess[0]]):
            Rect(*param).draw(f'tmp/interact_l{i}.png', utterance)
        # get the next utterance
        next_utterance = input('next utterance: ')
        x,y = eval(next_utterance)
        b = Rect(*prog)(x,y)
        utterance.append(((x,y), b))
