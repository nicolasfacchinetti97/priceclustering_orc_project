from instance import instance
from item import item

from progress_bar import progress

import configparser

class NoPoints(Exception):
    pass

def calc_demand(items, first, last):
    return sum(map(lambda i: i.demand, items.items[first-1: last]))

def value_in_segmentp1p2(p1, p2, val):
    return p1[1] + (p2[1] - p1[1])*(val - p1[0])/(p2[0] - p1[0])

def check_poly(poly, best):
    """check if best is dominated by poly"""
    for point in poly:
        found = False
        for p1, p2 in zip(best, best[1:]):
            # find segment of best that contains point
            if p1[0] < point[0] <= p2[0]:
                best_poly_value_in_point = value_in_segmentp1p2(p1, p2, point[0])
                print(f"\t\t\tSegment of best: {p1}, {p2}. Point of new poly to check {point}.\n" + 
                        f"\t\t\tValue in {point[0]} of best is {best_poly_value_in_point}")
                if point[1] > best_poly_value_in_point:
                    print("\t\t\tpoint of poly above the best segment")
                    return False
    return True

def find_next_point(cluster_info, v_j, z_j):
    # find minimum cluster price increase among the possible
    possible_increase = [items.items[d[1]-1].price - d[2] for d in cluster_info]

    print(f'\t\t\tThe possible increases in each cluster are: {possible_increase}')
    value = 0 if all(e == 0 for e in possible_increase) else min([n for n in possible_increase if n != 0])
    if value is 0:
        # no possible further increase, can't reach v for price costraint
        raise NoPoints
    increasable_cluster = [i for i,n in enumerate(possible_increase) if n != 0]
    
    demand = 0
    for c in increasable_cluster:
        s = cluster_info[c][0]
        e = cluster_info[c][1]
        demand += sum(map(lambda item: item.demand, items.items[s-1: e]))
        cluster_info[c][2] += value
    print(f'\t\t\tThe min increase is {value} for the clusters {increasable_cluster}, with demand {demand}')
    
    v_j += value * demand
    z_j += value
    print(F'\t\t\tNew cluster info {cluster_info} with profit {v_j}\n')
    return cluster_info, v_j, z_j

def truncate_poly(x_p, points):
    a = points[0]
    b = points[1]
    y_p = value_in_segmentp1p2(a, b, x_p)
    return (x_p, y_p)

def extend_state(items, pairs, i, j, k):
    # get all the values for computation
    p_last = items.get_item(i-1).price
    p_first = items.get_item(j).price             # no need of +1 because refer to scale of paper (start 1)
    v_new = (p_last + p_first)/2
    v_old = pairs[j,k-1]['v']
    z_new = (p_last - p_first)/2
    z_old = pairs[j,k-1]['z']
    z_max = abs(z_new-z_old)
    new_demand = calc_demand(items, j+1, i)
    
    # compute candidate values of z and v when range [1..i] is partitioned into [1..j] and [j + 1..i].
    z_j = max(z_old, z_new)
    v_j = v_old + new_demand*v_new 
    print(f'\tIn state {(j,k-1)} z old={z_old}, z new={z_new}, z max={z_max}')

    demand_old_clusters = [calc_demand(items, pairs[j, k-1]['s'][kk], pairs[j, k-1]['e'][kk]) for kk in range(0, k-1)]
    max_price_cluster = {kk: items.items[pairs[j, k-1]['e'][kk]-1].price - pairs[j, k-1]['q'][kk] for kk in range(k-1)}
    print(f'\tdemand old clusters {demand_old_clusters}, max price increase {max_price_cluster}')

    new_increase = new_demand * min(z_max, z_new)
    old_increase = sum(map(lambda kk: demand_old_clusters[kk]*min(z_max, max_price_cluster[kk]), range(k-1)))
    print(f'\tpossible tot price increase in old: {old_increase} new: {new_increase}') 

    state_j = {}
    state_j['q'] = pairs[j, k-1]['q'].copy() + [v_new]
    # maximize cluster prices according to zmax and price costraint 
    if old_increase >= new_increase:
        v_j += new_increase
        z_increase = min(z_max, z_new)
        state_j['q'][k-1] += z_increase
        print(f"\tcan increase new cluster of {z_increase}")
    else:
        v_j += old_increase
        for kk in range(0, k-1):
            z_increase = min(z_max, max_price_cluster[kk]) 
            state_j['q'][kk] += z_increase
            print(f"\tcan increase cluster {kk} of {z_increase}")
    state_j['v'] = v_j
    state_j['z'] = z_j
    state_j['s'] = pairs[j, k-1]['s'].copy() + [j+1]
    state_j['e'] = pairs[j, k-1]['e'].copy() + [i]
    return state_j


def find_stationary_points(candidates, desired_profit):
    """
    for each candidate state found with the extension procedure search the sationary points
    """
    points = {}
    for jj in candidates.keys():
        v_j = candidates[jj]['v']
        z_j = candidates[jj]['z']
        seq_data = [candidates[jj][key] for key in ['s', 'e', 'q']]
        cluster_info = [[data[c_idx] for data in seq_data] for c_idx in range(0, len(seq_data[0]))]
        # cluster_info = [start cluster, end cluster, price]

        printv(f'\t\tWith j:{jj} the cluster are {cluster_info} with profit {v_j}')
        
        points[jj] = [(v_j, z_j)]
        try:
            while v_j < desired_profit:
                cluster_info, v_j, z_j = find_next_point(cluster_info, v_j, z_j)
                points[jj].append((v_j, z_j))
                printv(f"\t\t\tNew point is: {(v_j, z_j)}")
        except NoPoints:
            printv(f"\t\t\tCan't further increase v for price costraint, remove state {jj}")
            del points[jj]
        
        printv(f"\t\tPoints of the poly: {points[jj]}")
        # truncate poly to desiderided profit if v_j of last point > desired_profit
        if points[jj][-1][0] > desired_profit and len(points[jj])>1:
            points[jj][-1] = truncate_poly(desired_profit, points[jj][-2:])
            printv(f"\t\tChange last point to {points[jj][-1]}")
            
        printv("\t\t::::::::::::::::::::::")
    return points

def find_non_dominated_solution(points):
    # divide polygonal chains and solution's points thats satisfy the desidered profit
    polygonal_chains = {x[0]: x[1] for x in points.items() if len(x[1])>1}
    satisfy_profit = [(x[0], x[1]) for x in points.items() if len(x[1])==1]
    # find the non dominated polygonal chain
    best_polygon = []
    if len(polygonal_chains) > 0:
        best_polygon = min(polygonal_chains.keys())
        for jj, poly in list(polygonal_chains.items())[1:]:
            printv(f'\t\t...checking soluton with j={jj}')
            if check_poly(poly, points[best_polygon]) is True:
                best_polygon = jj
                printv(f"\t\tNew best polygon {jj}")
        best_polygon = [(best_polygon, [points[best_polygon][-1]])]
    satisfy_profit += best_polygon
    # compare the eventual best polygonal chain with the points that exceed desired profit and choose the one with lower z value
    minValue = min(satisfy_profit, key = lambda t: t[1][0][1])[1][0][1]
    # two or more points may have the same value of z, choose the one with max v
    best = max([x for x in satisfy_profit if x[1][0][1] == minValue], key = lambda d: d[1][0][0])
    return best[0]
#----------------------------------------------------------------------------------------------------------

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
    d_h = calc_demand(items, 1, h+1)
    pairs[h+1, 1]['v'] = (items.get_item(h).price + items.get_item(0).price)/2 * d_h
    pairs[h+1, 1]['s'] = [1]
    pairs[h+1, 1]['e'] = [h+1]
    pairs[h+1, 1]['q'] = [pairs[h+1, 1]['v']/d_h]

printv(*(f'\tState {x[0]} -> {x[1]}' for x in pairs.items()), sep='\n')

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
            print(f'--------------------- j={j} ---------------------')
            candidate_states[j] = extend_state(items, pairs, i, j, k)
            printv(f'\tState {(i,k)} -> {list(candidate_states[j].items())}')

        # dominance check   
        printv("\nDominance check")
        printv(*(f'\tj: {x[0]} -> {x[1]}' for x in candidate_states.items()), sep='\n')

        original_profit = sum(map(lambda item: item.demand*item.price, items.items[0: i]))
        desired_profit = original_profit*profit_margin
        printv(f'\n\tFor {i} items the original profit is {original_profit} and the desired profit is {desired_profit}\n' +
                '\tStationary points calculus...')
        
        # find stationary points
        points = find_stationary_points(candidate_states, desired_profit)
        print("\tList of points\n" + f"\t{points}")

        # find non dominated solution
        bestj = find_non_dominated_solution(points)
        printv(f"\tNon dominated solution with j:{bestj}\n")
        # add new labels to state set
        pairs[i,k] = {}
        pairs[i,k]['z'] = candidate_states[bestj]['z']
        pairs[i,k]['v'] = candidate_states[bestj]['v']
        pairs[i,k]['s'] = candidate_states[bestj]['s']
        pairs[i,k]['e'] = candidate_states[bestj]['e']
        pairs[i,k]['q'] = candidate_states[bestj]['q']
        print(f'\tState {(i,k)} -> {list(pairs[i,k].items())}')
        print("______________________________________________________________________________________________________________________")

print('Computed all the state labels:')
pairs = {key:values for (key, values) in sorted(pairs.items())}
print(*(f'\tState {x[0]} -> {x[1]}' for x in pairs.items()), sep='\n')

# ============================================================================================================
# terminantion check
print(f'\nTerminantion check for optimal values.')

optimal_pairs = {}
for key in pairs:
    n_items = key[0]
    original_profit = sum(map(lambda item: item.demand*item.price, items.items[0: n_items]))
    desired_profit = original_profit*profit_margin
    printv(f'\nFor {key} the original profit is {original_profit} and the desired profit is {desired_profit}')

    z = pairs[key]['z']
    v = pairs[key]['v']
    if v >= desired_profit:
        printv(f'{key} -> {pairs[key]} satisfy profit margin.')
    else:
        printv(f'{key} -> {pairs[key]} not satisfy profit margin, extend the state...')
        # find the stationary points and take only the last one, which correponds to the desired profit
        v_z_last_point = find_stationary_points({key : pairs[key]}, desired_profit)[key][-1]
        v = v_z_last_point[0]
        z = v_z_last_point[1]
        printv(f'Increase z to {z} and v to {v}')
    optimal_pairs[key] = {}
    optimal_pairs[key]['z'] = z
    optimal_pairs[key]['v'] = v

print("\nThe optimal labels are:")
print(*(f'\tState {x[0]} -> {x[1]}' for x in optimal_pairs.items()), sep='\n')