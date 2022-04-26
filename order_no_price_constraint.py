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
    d_h = items.items[h].cumdemand
    pairs[h+1, 1]['v'] = (items.get_item(h).price + items.get_item(0).price)/2 * d_h


printv(*(f'\tState {x[0]} -> {x[1]}' for x in pairs.items()), sep='\n')

# ============================================================================================================
# extension of the states
print("Extension of the labes...")
"""
scans set N with index i ranging from 0 to N-1
index used to retrive items (starting from 0), when used for states remember to +1
"""
for i in range(1, items.N+1):
    if not verbose:
        progress(i-1, items.N, status='Computing the labels')
    printv(f'The value of i is {i}')

    """
    scans set K with index k ranging from 1 to min i,K
    i+1 because refer to a label state, final +1 becasue range in python not consider the last element
    """
    for k in range(2, min(i,num_clusters)+1):
        printv(f"\tComputing label ({i},{k})\n\tj in [{k-1}, {i-1}]")
        candidate_labels = []
        
        total_demand = items.items[i-1].cumdemand
        min_j = -1
        best_diff = float('inf')
        """
        compare all the labels when range [1..i] is partitioned into [1..j] and [j + 1..i]
        with j in [k − 1..i − 1] and only the best one is retained
        """
        # compute the labels
        for j in range(k-1, i):
            # get all the values for computation
            p_last = items.get_item(i-1).price
            p_first = items.get_item(j).price             # no need of +1 because refer to scale of paper (start 1)
            v_new = (p_last + p_first)/2
            z_new = (p_last - p_first)/2
            z_old = pairs[j,k-1]['z']
            v_old = pairs[j,k-1]['v']

            old_demand =  items.items[j-1].cumdemand
            new_demand = total_demand - old_demand

            # compute candidate values of z and v when range [1..i] is partitioned into [1..j] and [j + 1..i].
            z_j = max(z_old, z_new)
            v_j = v_old + new_demand*v_new + max((z_old - z_new)*new_demand, (z_new - z_old)*old_demand)
            # dominance check
            q = z_j - v_j/total_demand
            if q < best_diff:
                best_diff = q
                min_j = j
                candidate_labels = [z_j, v_j]
            printv(f"\t\twith j={j}, v:{v_j} z:{z_j} q: {q}")

        printv(f"\tNon dominated solution with j:{min_j}\n")

        # add new labels to state set
        pairs[i,k] = {}
        pairs[i,k]['z'] = candidate_labels[0]
        pairs[i,k]['v'] = candidate_labels[1]

print('Computed all the state labels:')
pairs = {key:values for (key, values) in sorted(pairs.items())}
print(*(f'\tState {x[0]} -> {x[1]}' for x in pairs.items()), sep='\n')

# ============================================================================================================
# terminantion check
print(f'\nTerminantion check for optimal values.')

optimal_pairs = {}
for key in pairs:
    n = key[0]
    original_profit = sum(map(lambda item: item.demand*item.price, items.items[0: n]))
    desired_profit = original_profit*profit_margin
    printv(f'\nFor {key} the original profit is {original_profit} and the desired profit is {desired_profit}')

    z = pairs[key]['z']
    v = pairs[key]['v']
    if v >= desired_profit:
        printv(f'{key} -> {pairs[key]} satisfy profit margin.')
    else:
        total_demand = items.items[n-1].cumdemand
        delta = (desired_profit - v)/total_demand
        z += delta
        v += delta*total_demand
        printv(f'{key} -> {pairs[key]} not satisfy profit margin.\n\tIncrease z to {z} and v to {v}')
    optimal_pairs[key] = {}
    optimal_pairs[key]['z'] = z
    optimal_pairs[key]['v'] = v

print("\nThe optimal labels are:")
print(*(f'\tState {x[0]} -> {x[1]}' for x in optimal_pairs.items()), sep='\n')