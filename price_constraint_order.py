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
    pairs[h+1, 1]['s'] = [1]
    pairs[h+1, 1]['e'] = [h+1]
    pairs[h+1, 1]['q'] = [pairs[h+1, 1]['v']/d_h]

printv(*(f'\tState {x[0]} -> {x[1]}' for x in pairs.items()), sep='\n')

# ============================================================================================================
# extension of the states
print("\nExtension of the labes...")
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
        candidate_states = {}
        """
        compare all the labels when range [1..i] is partitioned into [1..j] and [j + 1..i]
        with j in [k − 1..i − 1] and only the best one is retained
        """
        # compute the labels
        for j in range(k-1, i):
            print(f'----------------{j}---------------------')
            # get all the values for computation
            p_last = items.get_item(i-1).price
            p_first = items.get_item(j).price             # no need of +1 because refer to scale of paper (start 1)
            v_new = (p_last + p_first)/2
            v_old = pairs[j,k-1]['v']
            z_new = (p_last - p_first)/2
            z_old = pairs[j,k-1]['z']
            z_max = abs(z_new-z_old)
            new_demand = sum(map(lambda item: item.demand, items.items[j: i]))
            
            demand_old_clusters = []
            print('Calculus demand old clusters')
            for kk in range(0, k-1):
                old_start = pairs[j, k-1]['s'][kk]
                old_end = pairs[j, k-1]['e'][kk]
                print(f'\tk: {kk}, start: {old_start} end: {old_end}')
                demand_old_clusters.append(sum(map(lambda i: i.demand, items.items[old_start-1:old_end])))
            print(demand_old_clusters)
            # compute candidate values of z and v when range [1..i] is partitioned into [1..j] and [j + 1..i].
            z_j = max(z_old, z_new)
            print(f'zmax is {z_max}')
            v_j = v_old + new_demand*v_new 
            print(f'z in state {(j,k-1)}={z_old} new={z_new}')

            new_increase = new_demand * min(z_max, z_new)
            old_increase = 0
            for cluster, d in enumerate(demand_old_clusters):
                price_end_cluster = items.items[pairs[j, k-1]['e'][cluster]-1].price
                print(f'\tmax price in cluster {cluster}: {price_end_cluster}')
                max_price_cluster = price_end_cluster - pairs[j, k-1]['q'][cluster]
                print(f'\tmax possibile price increase from cluster {cluster}: {max_price_cluster}')
                old_increase += d*min(z_max, max_price_cluster)   
            
            print(f'possible increase from old {old_increase} new {new_increase}')     
            if old_increase >= new_increase:
                print("\tincrease new")
                v_j += new_increase
            else:
                print("\tincrease old")
                v_j += old_increase
            
            candidate_states[j] = {}
            candidate_states[j]['z'] = z_j
            candidate_states[j]['v'] = v_j
            candidate_states[j]['s'] = pairs[j, k-1]['s'].copy() + [j+1]
            candidate_states[j]['e'] = pairs[j, k-1]['e'].copy() + [i]
            candidate_states[j]['q'] = pairs[j, k-1]['q'].copy() + [v_new]
            if new_increase <= old_increase:
                candidate_states[j]['q'][k-1] += min(z_max, z_new)
            else:
                for kk in range(0, k-1):
                    price_end_cluster = items.items[pairs[j, k-1]['e'][cluster]-1].price
                    print(f'\tprice last item {price_end_cluster}')
                    max_price_cluster = price_end_cluster - pairs[j, k-1]['q'][cluster]
                    print(f'\tmax possibile price increase from cluster {cluster}: {max_price_cluster}')
                    candidate_states[j]['q'][kk] += demand_old_clusters[kk]*min(z_max, max_price_cluster)  
            
            printv(*(f'\tj {x[0]} -> {x[1]}' for x in candidate_states.items()), sep='\n')


        # dominance check   
        printv("\nDominance check")
        printv(*(f'\tj: {x[0]} -> {x[1]}' for x in candidate_states.items()), sep='\n')

        original_profit = sum(map(lambda item: item.demand*item.price, items.items[0: i]))
        desired_profit = original_profit*profit_margin
        printv(f'\nFor {i} items the original profit is {original_profit} and the desired profit is {desired_profit}')
        # find stationary points
        points = {}
        for jj in candidate_states.keys():
            v_j = candidate_states[jj]['v']
            z_j = candidate_states[jj]['z']
            seq_data = [candidate_states[jj][key] for key in ['s', 'e', 'q']]
            cluster_info = [[data[c_idx] for data in seq_data] for c_idx in range(0, len(seq_data[0]))]

            print(f'With j:{jj} the cluster are {cluster_info} with profit {v_j}')
            points[jj] = [(z_j, v_j)]
            
            if v_j < desired_profit:
                while v_j < desired_profit:
                    # find minimum cluster price increase among the possible
                    # --------------- metti list comprension
                    possible_increase = []
                    for d in cluster_info:
                        cluster_price = d[2]
                        cluster_end = d[1]
                        possible_increase.append(items.items[cluster_end-1].price - cluster_price)
                    print(f'\tThe possible increases in each cluster are: {possible_increase}')
                    value = 0 if all(e == 0 for e in possible_increase) else min([n for n in possible_increase if n != 0])
                    if value is 0:
                        # no possible further increase, can't reach v for price costraint
                        print(f"\tCan't further increase v for price costraint, remove state {jj}")
                        del points[jj]
                        break
                    print(f'\tThe min val is {value}')
                    increasable_cluster = [i for i,n in enumerate(possible_increase) if n != 0]
                    print(f'\tThe increasable clusters are {increasable_cluster}')
                    demand = 0
                    for c in increasable_cluster:
                        s = cluster_info[c][0]
                        e = cluster_info[c][1]
                        demand += sum(map(lambda item: item.demand, items.items[s-1: e]))
                        cluster_info[c][2] += value
                    print(f'\tTotal demand of increased cluster prices: {demand}')
                    
                    v_j += value * demand
                    z_j += value
                    print(F'\tNew cluster info {cluster_info} with profit {v_j}\n')
                    points[jj].append((z_j, v_j))
            else:
                while v_j >= desired_profit:
                    # find minimum cluster price decrease among the possible
                    # --------------- metti list comprension
                    possible_decrease = []
                    for d in cluster_info:
                        cluster_price = d[2]
                        cluster_start = d[0]
                        possible_decrease.append(cluster_price-items.items[cluster_start-1].price)
                    print(f'\tThe possible decreases in each cluster are: {possible_decrease}')
                    value = 0 if all(e == 0 for e in possible_decrease) else min([n for n in possible_decrease if n != 0 ])
                    if value is 0:
                        break
                    print(f'\tThe min val is {value}')
                    decreasable_cluster = [i for i,n in enumerate(possible_decrease) if n != 0]
                    print(f'\tThe increasable clusters are {decreasable_cluster}')
                    demand = 0
                    for c in decreasable_cluster:
                        s = cluster_info[c][0]
                        e = cluster_info[c][1]
                        demand += sum(map(lambda item: item.demand, items.items[s-1: e]))
                        cluster_info[c][2] -= value
                    print(f'\tTotal demand of increased cluster prices: {demand}')
                    #   ----------------------- CONSIDERARE COSA FARE SE NON CI SONO CLUSTER ESPANDIBILI ----------------
                
                    
                    v_j -= value * demand
                    z_j -= value
                    print(f'\tNew cluster info {cluster_info} with profit {v_j}\n')
                    points[jj].insert(0,(z_j, v_j))
                
            print("::::::::::::::::::::::")
        print("List of points")
        print(points)
        # find non dominated solution
        bestj = min(points.keys())
        for jj, poly in points.items():         # si può partire da points [1:]

            def check_poly(poly, best):
                for p1, p2 in zip(best, best[1:]):
                    for point in poly:
                        # find segment of best that contains point
                        if p1[0] <= point[0] < p2[0]:
                            if point[1] >= p1[1]:               # maggiore o magg uguale
                                return False
                            else:
                                if point[1] >= p2[1]:
                                    return False
                                elif point[1] >= p1[1] + (p2[1]-p1[1])*(point[0]-p1[0])/(p2[0]-p1[0]):
                                    return False    
                return True

            if check_poly(poly, points[bestj]) is True:
                bestj = jj 
        printv(f"\tNon dominated solution with j:{bestj}\n")
        # add new labels to state set
        pairs[i,k] = {}
        pairs[i,k]['z'] = candidate_states[bestj]['z']
        pairs[i,k]['v'] = candidate_states[bestj]['v']
        pairs[i,k]['s'] = candidate_states[bestj]['s']
        pairs[i,k]['e'] = candidate_states[bestj]['e']
        pairs[i,k]['q'] = candidate_states[bestj]['q']
        print(*(f'\tState {x[0]} -> {x[1]}' for x in pairs.items()), sep='\n')
        print("_____________________________________________________________________")

print('Computed all the state labels:')
pairs = {key:values for (key, values) in sorted(pairs.items())}
print(*(f'\tState {x[0]} -> {x[1]}' for x in pairs.items()), sep='\n')

# ============================================================================================================
# terminantion check
print(f'\nTerminantion check for optimal values.')
"""
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
        total_demand = sum(map(lambda item: item.demand, items.items[0: n]))
        delta = (desired_profit - v)/total_demand
        z += delta
        v += delta*total_demand
        printv(f'{key} -> {pairs[key]} not satisfy profit margin.\n\tIncrease z to {z} and v to {v}')
    optimal_pairs[key] = {}
    optimal_pairs[key]['z'] = z
    optimal_pairs[key]['v'] = v

print("\nThe optimal labels are:")
print(*(f'\tState {x[0]} -> {x[1]}' for x in optimal_pairs.items()), sep='\n')
"""