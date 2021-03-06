result = {(10, 0.1): [0.4634146341463415, 0.34146341463414637, 0.24390243902439024, 0.2926829268292683, 0.2682926829268293, 0.6341463414634146, 0.2926829268292683, 0.3170731707317073, 0.34146341463414637, 0.3170731707317073], (10, 0.01): [2.7804878048780486, 2.6341463414634148, 2.1707317073170733, 3.2195121951219514, 2.682926829268293, 3.1951219512195124, 2.7560975609756095, 2.8536585365853657, 2.682926829268293, 3.073170731707317], (10, 0.001): [2.707317073170732, 2.317073170731707, 2.707317073170732, 2.7560975609756095, 2.7560975609756095, 3.1463414634146343, 2.4878048780487805, 2.658536585365854, 2.8780487804878048, 2.8536585365853657], (10, 0.0001): [2.1219512195121952, 1.9268292682926829, 1.8780487804878048, 1.7560975609756098, 1.7560975609756098, 1.951219512195122, 1.853658536585366, 1.8780487804878048, 1.829268292682927, 2.317073170731707], (100, 0.1): [0.2682926829268293, 0.5609756097560976, 0.1951219512195122, 0.5365853658536586, 0.6097560975609756, 0.8292682926829268, 0.17073170731707318, 0.3170731707317073, 0.34146341463414637, 0.3170731707317073], (100, 0.01): [4.7560975609756095, 4.585365853658536, 4.439024390243903, 5.780487804878049, 4.878048780487805, 5.512195121951219, 5.2682926829268295, 5.024390243902439, 4.7073170731707314, 5.609756097560975], (100, 0.001): [4.585365853658536, 4.804878048780488, 4.414634146341464, 4.7560975609756095, 4.609756097560975, 6.121951219512195, 4.975609756097561, 4.390243902439025, 4.512195121951219, 5.2926829268292686], (100, 0.0001): [3.4146341463414633, 3.1219512195121952, 2.4878048780487805, 3.5609756097560976, 3.268292682926829, 3.926829268292683, 3.341463414634146, 3.4146341463414633, 3.268292682926829, 3.5609756097560976], (1000, 0.1): [0.4878048780487805, 0.2682926829268293, 0.1951219512195122, 0.5365853658536586, 0.2682926829268293, 0.8048780487804879, 0.14634146341463414, 0.5853658536585366, 0.3170731707317073, 0.7317073170731707], (1000, 0.01): [7.7317073170731705, 6.2682926829268295, 6.317073170731708, 6.682926829268292, 6.585365853658536, 7.439024390243903, 6.585365853658536, 7.0, 6.024390243902439, 7.2926829268292686], (1000, 0.001): [6.951219512195122, 7.317073170731708, 6.170731707317073, 7.219512195121951, 5.853658536585366, 7.487804878048781, 5.975609756097561, 7.146341463414634, 5.975609756097561, 7.878048780487805], (1000, 0.0001): [4.609756097560975, 3.5853658536585367, 3.975609756097561, 4.2439024390243905, 3.5365853658536586, 5.902439024390244, 3.7804878048780486, 4.7073170731707314, 4.219512195121951, 4.560975609756097], (10000, 0.1): [0.5853658536585366, 0.2682926829268293, 0.43902439024390244, 0.5365853658536586, 0.7073170731707317, 0.8048780487804879, 0.36585365853658536, 0.6829268292682927, 0.3170731707317073, 0.7317073170731707], (10000, 0.01): [8.560975609756097, 7.682926829268292, 7.2682926829268295, 8.0, 8.341463414634147, 10.951219512195122, 7.634146341463414, 8.878048780487806, 8.024390243902438, 8.804878048780488], (10000, 0.001): [8.21951219512195, 7.829268292682927, 8.048780487804878, 8.512195121951219, 7.951219512195122, 10.78048780487805, 7.195121951219512, 8.585365853658537, 8.121951219512194, 8.902439024390244], (10000, 0.0001): [5.414634146341464, 4.414634146341464, 5.073170731707317, 5.512195121951219, 5.536585365853658, 5.585365853658536, 4.902439024390244, 5.682926829268292, 5.073170731707317, 6.365853658536586]}

import numpy as np

for key in result:
    print (key)
    val = result[key]
    # get the mean and std of the values
    mean = np.mean(val)
    std = np.std(val)
    # make the confidence interval
    conf_int = 1.96 * std
    # print the mean and confidence interval
    print (mean - conf_int, mean, mean + conf_int)
import matplotlib.pyplot as plt

some_data1 = [0.3963963963963964, 0.1891891891891892, 0.1981981981981982, 1.063063063063063, 0.018018018018018018, 0.22522522522522523, 0.7027027027027027, 0.4864864864864865, 0.44144144144144143, 0.4864864864864865, 0.07207207207207207, 0.44144144144144143, 1.1981981981981982, 0.6306306306306306, 0.3333333333333333, 0.9099099099099099, 0.3783783783783784, 0.5405405405405406, 1.045045045045045, 0.0, 0.7297297297297297, 0.018018018018018018, 0.13513513513513514, 0.5765765765765766, 0.22522522522522523, 0.26126126126126126, 0.8108108108108109, 0.7477477477477478, 0.21621621621621623, 0.2972972972972973, 0.5315315315315315, 0.24324324324324326, 0.0, 0.6306306306306306, 0.2702702702702703, 0.22522522522522523, 0.45045045045045046, 0.6846846846846847, 0.3333333333333333, 0.21621621621621623]
some_data2 = [11.396396396396396, 9.09009009009009, 12.396396396396396, 9.621621621621621, 13.576576576576576, 8.837837837837839, 9.08108108108108, 8.26126126126126, 13.198198198198199, 10.162162162162161, 11.405405405405405, 12.801801801801801, 10.333333333333334, 10.747747747747749, 8.63963963963964, 11.27927927927928, 10.378378378378379, 9.954954954954955, 9.414414414414415, 11.522522522522523, 9.64864864864865, 11.09009009009009, 10.513513513513514, 13.072072072072071, 9.765765765765765, 12.243243243243244, 12.774774774774775, 10.99099099099099, 9.288288288288289, 12.468468468468469, 11.243243243243244, 9.315315315315315, 11.891891891891891, 13.018018018018019, 9.774774774774775, 13.9009009009009, 9.82882882882883, 13.35135135135135, 9.333333333333334, 10.963963963963964]
# using bootstrap to estimate the confidence interval
n = len(some_data1)
# bootstrap a bunch of means
bootstrap_means = [np.mean(np.random.choice(some_data2, size=n)) for i in range(1000)]
print (np.mean(some_data2), np.mean(bootstrap_means))
print (np.std(some_data2) / np.sqrt(len(some_data1)), np.std(bootstrap_means))
plt.hist(bootstrap_means, bins=10)
plt.show()


# batch 0 L(S0)_5000.pth_loss 9.237767219543457
# batch 500 L(S0)_5000.pth_loss 6.805760860443115
# batch 1000 L(S0)_5000.pth_loss 6.193828582763672
# batch 1500 L(S0)_5000.pth_loss 6.255030632019043
# batch 2000 L(S0)_5000.pth_loss 6.182240962982178
# batch 2500 L(S0)_5000.pth_loss 6.230778217315674
# batch 3000 L(S0)_5000.pth_loss 6.064933776855469
# batch 3500 L(S0)_5000.pth_loss 5.936612129211426
# batch 4000 L(S0)_5000.pth_loss 5.884978294372559
# batch 4500 L(S0)_5000.pth_loss 5.762722015380859
# batch 0 L(S(L0))_5000.pth_loss 9.292414665222168
# batch 500 L(S(L0))_5000.pth_loss 6.41703987121582
# batch 1000 L(S(L0))_5000.pth_loss 5.805128574371338
# batch 1500 L(S(L0))_5000.pth_loss 5.556344509124756
# batch 2000 L(S(L0))_5000.pth_loss 5.927003383636475
# batch 2500 L(S(L0))_5000.pth_loss 5.7043962478637695
# batch 3000 L(S(L0))_5000.pth_loss 5.675553798675537
# batch 3500 L(S(L0))_5000.pth_loss 5.489285469055176
# batch 4000 L(S(L0))_5000.pth_loss 5.362655162811279
# batch 4500 L(S(L0))_5000.pth_loss 5.207956790924072
# batch 0 L(S(L))_5000.pth_loss 9.24477481842041
# batch 500 L(S(L))_5000.pth_loss 8.334355354309082
# batch 1000 L(S(L))_5000.pth_loss 8.440486907958984
# batch 1500 L(S(L))_5000.pth_loss 8.368941307067871
# batch 2000 L(S(L))_5000.pth_loss 8.382320404052734
# batch 2500 L(S(L))_5000.pth_loss 8.4812593460083
# batch 3000 L(S(L))_5000.pth_loss 8.303346633911133
# batch 3500 L(S(L))_5000.pth_loss 8.208292007446289
# batch 4000 L(S(L))_5000.pth_loss 8.376419067382812
# batch 4500 L(S(L))_5000.pth_loss 8.436613082885742
# batch 0 L(S(L(S0)))_5000.pth_loss 9.217643737792969
# batch 500 L(S(L(S0)))_5000.pth_loss 5.2278571128845215
# batch 1000 L(S(L(S0)))_5000.pth_loss 4.197971343994141
# batch 1500 L(S(L(S0)))_5000.pth_loss 3.38120174407959
# batch 2000 L(S(L(S0)))_5000.pth_loss 2.7169408798217773
# batch 2500 L(S(L(S0)))_5000.pth_loss 2.7501461505889893
# batch 3000 L(S(L(S0)))_5000.pth_loss 2.647569179534912
# batch 3500 L(S(L(S0)))_5000.pth_loss 2.193345546722412
# batch 4000 L(S(L(S0)))_5000.pth_loss 2.094135284423828
# batch 4500 L(S(L(S0)))_5000.pth_loss 1.9479639530181885
# batch 0 L(S(L,L0))_5000.pth_loss 9.201677322387695
# batch 500 L(S(L,L0))_5000.pth_loss 5.826408386230469
# batch 1000 L(S(L,L0))_5000.pth_loss 5.057812213897705
# batch 1500 L(S(L,L0))_5000.pth_loss 3.741082191467285
# batch 2000 L(S(L,L0))_5000.pth_loss 2.977325677871704
# batch 2500 L(S(L,L0))_5000.pth_loss 3.2311105728149414
# batch 3000 L(S(L,L0))_5000.pth_loss 2.1455652713775635
# batch 3500 L(S(L,L0))_5000.pth_loss 1.7518888711929321
# batch 4000 L(S(L,L0))_5000.pth_loss 1.39530611038208
# batch 4500 L(S(L,L0))_5000.pth_loss 1.0959255695343018

batch_numbs = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
L_S0 = [9.237767219543457, 6.805760860443115, 6.193828582763672, 6.255030632019043, 6.182240962982178, 6.230778217315674, 6.064933776855469, 5.936612129211426, 5.884978294372559, 5.762722015380859]
L_S_L0 = [9.292414665222168, 6.41703987121582, 5.805128574371338, 5.556344509124756, 5.927003383636475, 5.7043962478637695, 5.675553798675537, 5.489285469055176, 5.362655162811279, 5.207956790924072]
L_S_L = [9.24477481842041, 8.334355354309082, 8.440486907958984, 8.368941307067871, 8.382320404052734, 8.4812593460083, 8.303346633911133, 8.208292007446289, 8.376419067382812, 8.436613082885742]
L_S_L_S0 = [9.217643737792969, 5.2278571128845215, 4.197971343994141, 3.38120174407959, 2.7169408798217773, 2.7501461505889893, 2.647569179534912, 2.193345546722412, 2.094135284423828, 1.9479639530181885]
L_S_LL0 = [9.201677322387695, 5.826408386230469, 5.057812213897705, 3.741082191467285, 2.977325677871704, 3.2311105728149414, 2.1455652713775635, 1.7518888711929321, 1.39530611038208, 1.0959255695343018]

# plot the results
plt.plot(batch_numbs, L_S0, label='L(S0)')
plt.plot(batch_numbs, L_S_L0, label='L(S(L0))')
plt.plot(batch_numbs, L_S_L, label='L(S(L))')
plt.plot(batch_numbs, L_S_L_S0, label='L(S(L(S0)))')
plt.plot(batch_numbs, L_S_LL0, label='L(S(L,L0))')
# set y axis label to "cross entropy loss"
plt.ylabel('cross entropy loss')
# set x axis label to "batch number"
plt.xlabel('batch number')
# add legend
plt.legend()
# show the plot
plt.show()
# clear the plot
plt.clf()

import pickle
# import the scipy statistics module
import scipy.stats as stats
    
# read the pickle at tmp/results.pkl
with open('tmp/results.pkl', 'rb') as f:
    results = pickle.load(f)

to_viz = {
    'L(S0)_5000' : results['L(S0)_5000']['num_utter'],
    'L(S(L0))_5000' : results['L(S(L0))_5000']['num_utter'],
    'L(S(L(S0)))_5000' : results['L(S(L(S0)))_5000']['num_utter'],
    'L(S(L,L0))_5000' : results['L(S(L,L0))_5000']['num_utter']
}

print ("number of trials")
print (len(to_viz['L(S(L(S0)))_5000']))


number_of_bins = 10

# change the print of a numpy number to 3 decimal places
np.set_printoptions(precision=3)

# visualize the results as multiple vertical histograms
label_names = ['L(S0)', 'L(S(L0))', 'L(S(L(S0)))', 'L(S(L,L0))']
label_means =   [f"{np.mean(to_viz[key])}"[:5] for key in to_viz]
# make sure the number in label_means is only 3 decimal places
label_std_err = [f"{stats.sem(to_viz[key])}"[:5] for key in to_viz]
# zip the above 3 lists together
label_data = list(zip(label_names, label_means, label_std_err))
# merge the triplet into a single string
labels = [f"{x[0]} \nmean:{x[1]} \nse:{x[2]}" for x in label_data]
data_sets = [to_viz['L(S0)_5000'], to_viz['L(S(L0))_5000'], to_viz['L(S(L(S0)))_5000'], to_viz['L(S(L,L0))_5000']]

# Computed quantities to aid plotting
hist_range = (np.min(data_sets), np.max(data_sets))
binned_data_sets = [
    np.histogram(d, range=hist_range, bins=number_of_bins)[0]
    for d in data_sets
]

binned_maximums = np.max(binned_data_sets, axis=1)

# compute all prefix sum of binned_maximums and store as x_locations
x_locations = [1.75 * sum(binned_maximums[:i]) for i in range(len(binned_maximums))]

# The bin_edges are the same for all of the histograms
bin_edges = np.linspace(hist_range[0], hist_range[1], number_of_bins + 1)
centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
heights = np.diff(bin_edges)

# Cycle through and plot each histogram
fig, ax = plt.subplots()
for x_loc, binned_data in zip(x_locations, binned_data_sets):
    lefts = x_loc - 0.5 * binned_data
    ax.barh(centers, binned_data, height=heights, left=lefts)
    

ax.set_xticks(x_locations)
ax.set_xticklabels(labels)

ax.set_ylabel("number of utterances required", fontsize=14)
ax.set_xlabel("types of listeners", fontsize=14)


# add a title
ax.set_title(f"number of utterances required for each type of listener, number of trials = {len(to_viz['L(S(L(S0)))_5000'])}", fontsize=14)
plt.show()

# perform t-test between L(S(L,L0)) and L(S0)
from scipy.stats import ttest_ind
for other in ['L(S0)_5000', 'L(S(L0))_5000', 'L(S(L(S0)))_5000']:
    result = ttest_ind(to_viz['L(S(L,L0))_5000'], to_viz[other], axis=0, equal_var=False)
    print (f"t-test comparing L(S(L,L0)) and {other}")
    print ('pvalue : ', result.pvalue, ' < 0.0001 : ', result.pvalue < 0.0001, ' significant : ', result.pvalue < 0.0001)




# take a look at search efficiency measured as number of guesses

to_viz = {
    'L(S0)_5000' :  np.log(np.array(results['L(S0)_5000']['total_guess'])),
    'L(S(L0))_5000' :  np.log(np.array(results['L(S(L0))_5000']['total_guess'])),
    'L(S(L(S0)))_5000' :  np.log(np.array(results['L(S(L(S0)))_5000']['total_guess'])),
    'L(S(L,L0))_5000' :  np.log(np.array(results['L(S(L,L0))_5000']['total_guess']))
}

number_of_bins = 20

# visualize the results as multiple vertical histograms
data_sets = [to_viz['L(S0)_5000'], to_viz['L(S(L0))_5000'], to_viz['L(S(L(S0)))_5000'], to_viz['L(S(L,L0))_5000']]

# Computed quantities to aid plotting
hist_range = (np.min(data_sets), np.max(data_sets))
binned_data_sets = [
    np.histogram(d, range=hist_range, bins=number_of_bins)[0]
    for d in data_sets
]

binned_maximums = np.max(binned_data_sets, axis=1)

# compute all prefix sum of binned_maximums and store as x_locations
x_locations = [1.75 * sum(binned_maximums[:i]) for i in range(len(binned_maximums))]

# The bin_edges are the same for all of the histograms
bin_edges = np.linspace(hist_range[0], hist_range[1], number_of_bins + 1)
centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
heights = np.diff(bin_edges)

# Cycle through and plot each histogram
fig, ax = plt.subplots()
for x_loc, binned_data in zip(x_locations, binned_data_sets):
    print (x_loc)
    lefts = x_loc - 0.5 * binned_data
    ax.barh(centers, binned_data, height=heights, left=lefts)

ax.set_xticks(x_locations)
ax.set_xticklabels(labels)

ax.set_ylabel("search budget required (log scale)")
ax.set_xlabel("types of listeners")

plt.show()

for other in ['L(S0)_5000', 'L(S(L0))_5000', 'L(S(L(S0)))_5000']:
    result = ttest_ind(to_viz['L(S(L,L0))_5000'], to_viz[other], axis=0, equal_var=False)
    print (f"t-test comparing L(S(L,L0)) and {other}")
    print (result)
    print ('pvalue : ', result.pvalue, ' < 0.0001 : ', result.pvalue < 0.0001, ' significant : ', result.pvalue < 0.0001)

