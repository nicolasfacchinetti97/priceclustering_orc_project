from more_itertools import first, last
from numpy import True_
from instance import instance
from item import item

import itertools

from progress_bar import progress

import configparser

def extension1(last_state, point):
    C = last_state["C"].copy()
    O = last_state["O"].copy()
    candidates = []
    for k in O:
        O.remove(k)
        O.append(k + [point])
        candidates.append({'C': C, 'O':O})
    return candidates

def extension2(last_state, point):
    C = last_state["C"].copy()
    O = last_state["O"].copy()
    candidates = []
    for k in O:
        O.remove(k)
        C.append(k + [point])
        candidates.append({'C': C, 'O':O})
    return candidates

def extension3(last_state, point):
    C = last_state["C"].copy()
    O = last_state["O"].copy()
    C.append([point])
    return {'C': C, 'O':O}

def extension4(last_state, point):
    C = last_state["C"].copy()
    O = last_state["O"].copy()
    O.append([point])
    return {'C': C, 'O':O}


class NoPoints(Exception):
    pass

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
                printv(f"\t\t\tSegment of best: {p1}, {p2}. Point of new poly to check {point}.\n" + 
                        f"\t\t\tValue in {point[0]} of best is {best_poly_value_in_point}")
                if point[1] > best_poly_value_in_point:
                    printv("\t\t\tpoint of poly above the best segment")
                    return False
    return True

def truncate_poly(x_p, points):
    a = points[0]
    b = points[1]
    y_p = value_in_segmentp1p2(a, b, x_p)
    return (x_p, y_p)

def find_next_point(cluster_info, v_j, z_j):
    # find minimum cluster price increase among the possible
    possible_increase_close = [items.items[d[1]-1].price - d[2] for d in cluster_info]

    printv(f'\t\t\tThe possible increases in each cluster are: {possible_increase_close}')
    value = 0 if all(e == 0 for e in possible_increase_close) else min([n for n in possible_increase_close if n != 0])
    if value is 0:
        # no possible further increase, can't reach v for price costraint
        raise NoPoints
    increasable_cluster = [i for i,n in enumerate(possible_increase_close) if n != 0]
    
    demand = 0
    for c in increasable_cluster:
        demand += cluster_info[c][3]
        cluster_info[c][2] += value
    printv(f'\t\t\tThe min increase is {value} for the clusters {increasable_cluster}, with demand {demand}')
    
    v_j += value * demand
    z_j += value
    printv(F'\t\t\tNew cluster info {cluster_info} with profit {v_j}\n')
    return cluster_info, v_j, z_j

def find_stationary_points(candidates, v_oo, desired_profit):
    """
    for each candidate state found with the extension procedure search the sationary points
    """
    v_j = candidates['v']
    z_j = candidates['z']
    seq_data = [candidates[key] for key in ['s', 'e', 'q', 'd']]
    cluster_info = [[data[c_idx] for data in seq_data] for c_idx in range(0, len(seq_data[0]))]
    # cluster_info = [start cluster, end cluster, price, demand]

    printv(f'\t\tThe cluster are {cluster_info} with profit {v_j}')
    
    points = [(v_j, z_j)]
    try:
        while v_j < desired_profit:
            cluster_info, v_j, z_j = find_next_point(cluster_info, v_j, z_j)
            points.append((v_j, z_j))
            printv(f"\t\t\tNew point is: {(v_j, z_j)}")
    except NoPoints:
        printv(f"\t\t\tCan't further increase v for price costraint")
    
    printv(f"\t\tPoints of the poly: {points}")
    # truncate poly to desiderided profit if v_j of last point > desired_profit
    if points[-1][0] > desired_profit and len(points)>1:
        points[-1] = truncate_poly(desired_profit, points[-2:])
        printv(f"\t\tChange last point to {points[-1]}")
        
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

# dict of states (i)
"""
i in N is the last considered item
"""
states = {}

# base case computation
print('Base cases creation...\n')
states[0] = [{"C":[], "O":[], "z":0, "v":0}]

printv(*(f'\tState {x[0]} -> {x[1]}' for x in states.items()), sep='\n')

# extension of the states
print("\nExtension of the labes...")
"""
scans set N with index i ranging from 1 to N
index used to manange states and items (starting from 1), when used for retrive elements remeber -1 (starting from 0)
"""
for i in range(1, items.N+1):
    if not verbose:
        progress(i-1, items.N, status='Computing the labels')

    last_state = states[i-1]
    point_i = items.get_item(i-1)
    printv(f'Computing state {i}.\n\tThe state {i-1} has {last_state}.\n\tNew point has {point_i}')
    # four types of extension
    candidate_states = []
    for s in last_state:    
        candidate_states.extend(extension1(s, i))
        candidate_states.extend(extension2(s, i))
        candidate_states.append(extension3(s, i))
        candidate_states.append(extension4(s, i))
    printv(candidate_states)
    
    limit = items.N - i
    oldl = len(candidate_states)
    candidate_states = [c for c in candidate_states if len(c["O"]) <= limit]
    printv(f"Removal of states that violate |O| <= n-i = {limit}, for a total of {oldl - len(candidate_states)} states")
    printv(candidate_states)
    
    # sort the open clusters in ascending order of their first point

    z_c = []
    z_o = []
    for c in candidate_states:
        # compute z for closed cluster
        if len(c["C"]) > 0:
            z_cc = []
            for cc in c["C"]:
                z_cc.append((items.get_item(cc[-1]-1).price - items.get_item(cc[0]-1).price)/2)
            z_c.append(z_cc)
        else:
            z_c.append([0])

        # estimate lower bound on z for open cluster
        if len(c["O"]) > 0:
            z_oc = []
            # find p'(K) for each open cluster
            for j, cc in enumerate(c["O"]):
                candiate_lastp = i+j+1
                # printv(f'for {cc} the candidate next element is {candiate_lastp}')
                z_oc.append((items.get_item(candiate_lastp-1).price - items.get_item(cc[0]-1).price)/2)
            z_o.append(z_oc)
        else:
            z_o.append([0])
    # printv(f'\nz values for each closed cluster {z_c}')
    # take the max value of z for each state
    z_c = [max(c) for c in z_c]
    # printv(f'max z value for each closed cluster {z_c}')
    # printv(f'z values estimate for each open cluster {z_o}')
    # take the max value of z for each state
    z_o = [max(c) for c in z_o]
    # printv(f'max z value estimate for each open cluster {z_o}')
    z = [max(z_c[j], z_o[j]) for j in range(len(z_o))]
    print(f"final z value: {z}")

    for count, cluster in enumerate(candidate_states):
        cluster['z'] = z[count]

    # v calculus 
    v_c = []
    v_o = []

    points_list = []
    original_profit = sum(map(lambda item: item.demand*item.price, items.items[0: i]))
    desired_profit = original_profit*profit_margin
    print(f"For itemset to {i} with a margin of {profit_margin} the desired profit is {desired_profit}")
    for c in candidate_states:
        v_cc = 0
        z = c["z"]
        q = []
        d = []
        # calculus of v for closed clusters
        for cc in c["C"]:
            p_min = items.get_item(cc[0]-1).price
            p_max = items.get_item(cc[-1]-1).price
            cluster_price = min(p_min + z, p_max)
            q.append(cluster_price)
            sum_demand = 0
            for j in cc:
                # contribution item j = dj * min{p−(K) + z, p+(K)}
                demand = items.get_item(j-1).demand
                v_cc += demand * cluster_price
                sum_demand += demand
            d.append(sum_demand)
        v_c.append(v_cc)

        # estimates bound v on open clusters
        v_oo = [0, 0]
        for cc in c["O"]:
            p_i_next = items.get_item(i).price
            p_min = items.get_item(cc[0]-1).price
            for j in cc:
                # dj min{p−(K) + z, pi+1} ≤ contribution item j ≤ dj (p−(K) + z).
                lower_bound = items.get_item(j-1).demand * min(p_min + z, p_i_next)
                high_bound = items.get_item(j-1).demand * (p_min + z)
                v_oo[0] += lower_bound
                v_oo[1] += high_bound
        v_o.append(v_oo)

        print(v_oo)
        mod_s1 = {'v': v_cc+v_oo[0], 'z': z, 'q': q, 'd': d, 's': [s[0] for s in c['C']], 'e': [s[-1] for s in c['C']]}
        print(c)
        points = find_stationary_points(mod_s1, v_oo[1]-v_oo[0], desired_profit)
        print(points)
        points_list.append(points)
        print()
        

    printv(f'v values for each closed cluster {v_c}')
    printv(f'v estimates for each closed open cluster {v_o}')

    for count, cluster in enumerate(candidate_states):
        cluster['v'] = [v_o[count][0] + v_c[count], v_o[count][1] + v_c[count]]

    print(candidate_states)
    print(points_list)
    
    # dominance check
    print("\nDominance check...")
    permut = list(itertools.permutations(range(len(candidate_states)),2))
    dominated = []
    for i1, i2 in permut:
        if i1 not in dominated and i2 not in dominated:
            s1 = candidate_states[i1]
            s2 = candidate_states[i2]
            if len(s1["C"]) <= len(s2["C"]) and s1['z']<=s2['z'] and len(s1["O"]) == len(s2["O"]):
                # s1 dominate s2
                # check profit on closed cluster
                # find stationary points
                print(f'\nChecking {i1} and {i2}\n{i1}: {s1}\n{i2}: {s2}')
                a = points_list[i1]
                b = points_list[i2]
                p = {i1:a, i2:b}
                printv("\tList of points\n" + f"\t{p}")
                # find non dominated solution
                bestj = find_non_dominated_solution(p)
                if bestj == i2:
                    # if 2 is not dominated by 1 skip to next permutation
                    printv(f"\tNon domintate:{bestj}, skip...\n")
                    continue
                
                # find a valid permutation of a state that satisfy the two inequalities in all open cluster's pairs
                s1_O = s1['O']
                found = True
                if len(s1_O) > 0:
                    permutations_of_items = itertools.permutations(range(len(s1_O)), len(s1_O))
                    found = False
                    for perm in permutations_of_items:
                        s2_reorder = [s2['O'][idx] for idx in perm]
                        for c1, c2 in zip(s1_O, s2_reorder):
                            p_minus_1 = items.get_item(c1[0]-1).price
                            p_minus_2 = items.get_item(c2[0]-1).price
                            D_1 = sum(map(lambda i: i.demand, [items.items[idx-1] for idx in c1]))
                            D_2 = sum(map(lambda i: i.demand, [items.items[idx-1] for idx in c2]))
                            if not((p_minus_1 >= p_minus_2) and (D_1 >= D_2)):
                                print("Dont satisfy constranint on open cluster\n")
                                break
                            found = True
                        if found:
                            print(f"Itemsets in {s1_O} and {s2_reorder} satisfy constraint on open clusters.")
                            break      
                if found:
                    printv(f'Found that state {i2} is dominated by {i1}\n')
                    dominated.append(i2)
    candidate_states = [c for count, c in enumerate(candidate_states) if count not in dominated]
    printv(f'Removed {len(dominated)} states: {dominated}')

    # assign non-dominated solution
    states[i] = candidate_states
    
    printv("______________________________________________________________________________________________________________________")


for s in states:
    print(s)
    print(states[s])