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
                    if T < B and L < R:
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

if __name__ == '__main__':
    all_progs = make_all_progs()
    train, test = split_progs(all_progs)
    print (len(train), len(test))