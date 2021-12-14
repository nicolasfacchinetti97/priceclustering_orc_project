from instance import instance
from item import item

from progress_bar import progress

import configparser


# read configuration file
config = configparser.ConfigParser()
config.read("config.ini")
instance_path = config['inputfile']['fileName']
num_clusters = int(config['inputfile']['numClusters'])
profit_margin = float(config['inputfile']['profitMargin'])
verbose = int(config['inputfile']['verbose'])

printv = print if verbose else lambda *a, **k: None

items = instance(instance_path)


# dict of states (i,k)
"""
i in N is the last considered item
k is the number of cluster used to group items in [1, i]
"""
pairs = {}

# base case computation
print('Base cases creation...\n')
for h in range(0, items.N):
    pairs[h+1,1] = {}
    pairs[h+1, 1]['z'] = (items.get_item(h).price - items.get_item(0).price)/2
    d_h = sum(map(lambda item: item.demand, items.items[0: h+1]))
    pairs[h+1, 1]['v'] = (items.get_item(h).price + items.get_item(0).price)/2 * d_h

# ============================================================================================================
# extension of the states
print("Extension of the labes...")
"""
scans set N with index i ranging from 0 to N-1
index used to retrive items (starting from 0), when used for states remember to +1
"""
for i in range(0, items.N):
    if not verbose:
        progress(i, items.N, status='Computing the labels')
    printv(f'The value of i is {i+1}')

    """
    scans set K with index k ranging from 1 to min i,K
    i+1 because refer to a label state, final +1 becasue range in python not consider the last element
    """
    for k in range(2, min(i+1,num_clusters)+1):
        printv(f"\tComputing label ({i+1},{k})\n\tj in [{k-1}, {i}]")
        candidate_labels = {}

        """
        compare all the labels when range [1..i] is partitioned into [1..j] and [j + 1..i]
        with j in [k − 1..i − 1] and only the best one is retained
        """
        # compute the labels
        for j in range(k-1, i+1):
            # get all the values for computation
            p_i = items.get_item(i).price
            p_jp1 = items.get_item(j).price             # no need of +1 because refer to scale of paper (start 1)
            summ = (p_i + p_jp1)/2
            diff = (p_i - p_jp1)/2
            z_j_km1 = pairs[j,k-1]['z']
            sum_d1 = sum(map(lambda item: item.demand, items.items[j+1: i+1]))
            sum_d2 = sum(map(lambda item: item.demand, items.items[0: j+1]))
            # compute candidate values of z and v when range [1..i] is partitioned into [1..j] and [j + 1..i].
            z_j = max(z_j_km1, diff)
            v_j = pairs[j,k-1]['v'] + sum_d1*summ + max((z_j_km1 - diff)*sum_d1, (diff - z_j_km1)*sum_d2)
            candidate_labels[j] = ([z_j, v_j])
            printv(f"\t\t with j={j}, v:{v_j} z:{z_j}")

        # dominance check
        sum_d3 = sum(map(lambda item: item.demand, items.items[0: i+1]))
        min_j = -1
        best_diff = float('inf')
        for j in candidate_labels:
            z_j, v_j = candidate_labels[j]
            diff_j = z_j - v_j/sum_d3
            printv(f'\tFor j={j} the diff is {diff_j}')
            if diff_j < best_diff:
                best_diff = diff_j
                min_j = j
        printv(f"\tBest values are obtained with j:{min_j}\n")

        # add new labels to state set
        top_vals = candidate_labels[min_j]
        pairs[i+1,k] = {}
        pairs[i+1,k]['z'] = top_vals[0]
        pairs[i+1,k]['v'] = top_vals[1]

print('Computed all the states labels:')
pairs = {key:values for (key, values) in sorted(pairs.items())}
print(*(f'\tState {x[0]} -> {x[1]}' for x in pairs.items()), sep='\n')

# ============================================================================================================
# terminantion check
original_profit = sum(map(lambda item: item.demand*item.price, items.items[0: items.N]))
desired_profit = original_profit*profit_margin
print(f'\nTerminantion check for optimal values.\n\tThe original profit is {original_profit}' +
    f'\n\tThe desired profit with a margin of {profit_margin} is {desired_profit}')
for key in pairs:
    z = pairs[key]['z']
    v = pairs[key]['v']
    if v >= desired_profit:
        print(f'{key} is an optimal state -> {pairs[key]}')
