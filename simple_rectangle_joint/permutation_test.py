import pickle
import numpy as np
import matplotlib.pyplot as plt
import random

# read the pickle at tmp/results.pkl
with open('tmp/results.pkl', 'rb') as f:
    results = pickle.load(f)

to_viz = {
    'L(S0)_5000' : results['L(S0)_5000']['num_utter'],
    'L(S(L0))_5000' : results['L(S(L0))_5000']['num_utter'],
    'L(S(L(S0)))_5000' : results['L(S(L(S0)))_5000']['num_utter'],
    'L(S(L,L0))_5000' : results['L(S(L,L0))_5000']['num_utter']
}

def split_and_take_mean(joint_set):
    size = len(joint_set) // 2
    random.shuffle(joint_set)
    set1 = joint_set[:size]
    set2 = joint_set[size:]
    # compute difference of the means
    return np.mean(set1) - np.mean(set2)

A = to_viz['L(S(L,L0))_5000']
B = to_viz['L(S0)_5000']

diff_mean = np.mean(A) - np.mean(B)
AB = [x for x in A+B]
sampled_diff_of_means = [split_and_take_mean(AB) for _ in range(10000)]

plt.hist(sampled_diff_of_means, bins=100)
plt.show()

print (f"the original diff of means is {diff_mean}")
print (f"number of sampled means {len(sampled_diff_of_means)}")
print (f"fraction of those more extreme than diff_mean {len([x for x in sampled_diff_of_means if abs(x) > abs(diff_mean)])/len(sampled_diff_of_means)}")
