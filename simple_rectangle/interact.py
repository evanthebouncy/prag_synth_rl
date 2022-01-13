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
    # note, if the listener is "big_sample_saved" make sure to use a budget of 100!

    # make the two agents
    l0_nn = L_NN_F(100)
    l0_nn.load('tmp/l0_nn_big_sample_saved.pth')
    # make the neural pragmatic listener l1_nn
    l1_nn = L_NN_F(100)
    l1_nn.load('tmp/l1_nn_big_sample_saved.pth')

    # # the counter example speaker
    s_ce = S_CE(l0_nn)

    # make the test set and get a rectangle from it
    all_progs = make_all_progs()
    train, test = split_progs(all_progs)
    
    listeners = [l0_nn, l1_nn]

    if False:
        # # test the efficiency of the two agents
        listener_utters_required = test_efficiency(s_ce, listeners, test)
        print ("against S_CE(l0_nn)")
        # for each listener print the number of average utters required
        for i, listener in enumerate(listener_utters_required):
            print ("listener", i, "required", sum(listener) / len(listener))

    # break the seed used to split train/test and get a new one
    random.seed()
    prog = random.choice(test)

    utterance = s_ce(prog, 6)
    print ("preferred utterance ", utterance)
    
    utterance = []
    while True:
        # visualize the prog, and the two guesses
        Rect(*prog).draw(f'tmp/interact_orig.png', utterance)
        for i, listener in enumerate([l0_nn,l1_nn]):
            param_guess, budget = run_until_sat(listener, utterance)
            print (f"budgets spent l{i}:{budget}")
            Rect(*param_guess[0]).draw(f'tmp/interact_l{i}.png', utterance)
        # get the next utterance
        next_utterance = input('next utterance: ')
        x,y = eval(next_utterance)
        b = Rect(*prog)(x,y)
        utterance.append(((x,y), b))
