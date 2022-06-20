from os import remove
import sys
from instance import instance
from item import item

import itertools

from progress_bar import progress

import configparser

def extension1(last_state, point):
    C = last_state["C"].copy()
    candidates = []
    for k in last_state["O"]:
        o_mod = last_state["O"].copy()
        o_mod.remove(k)
        o_mod.extend([k + [point]])
        candidates.append({'C': C, 'O':o_mod})
    return candidates

def extension2(last_state, point):
    O = last_state["O"].copy()
    candidates = []
    for k in O:
        o_mod = O.copy()
        o_mod.remove(k)
        c_mod = last_state["C"].copy()
        c_mod.extend([k + [point]])
        candidates.append({'C': c_mod, 'O':o_mod})
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
    printv(poly)
    found = False
    for point in poly:
        
        for p1, p2 in zip(best, best[1:]):
            # find segment of best that contains point
            if p1[0] < point[0] <= p2[0]:
                found = True
                best_poly_value_in_point = value_in_segmentp1p2(p1, p2, point[0])
                printv(f"\t\tSegment of best: {p1}, {p2}.\n\t\tNew poly :{point}, " + 
                        f"value in best {point[0], best_poly_value_in_point}")
                if point[1] > best_poly_value_in_point:
                    printv("\t\t\tpoint of poly above the best segment")
                    return False
    if not found:
        printv("No point in common, no dominance relation...")
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

def find_stationary_points(candidates, desired_profit):
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
    max_profit = points[-1][0]
    if max_profit > desired_profit and len(points)>1:
        points[-1] = truncate_poly(desired_profit, points[-2:])
        printv(f"\t\tChange last point to {points[-1]}")
        
    printv("\t\t::::::::::::::::::::::")
    return points, max_profit

def find_non_dominated_solution(points):
    # divide polygonal chains and solution's points thats satisfy the desidered profit
    polygonal_chains = {x[0]: x[1] for x in points.items() if len(x[1])>1}
    satisfy_profit = [(x[0], x[1]) for x in points.items() if len(x[1])==1]
    # find the non dominated polygonal chain
    best_polygon = []
    if len(polygonal_chains) > 0:
        best_polygon = next(iter(polygonal_chains))
        for jj, poly in list(polygonal_chains.items())[1:]:
            # put in poly all the points of best
            best = points[best_polygon]
            mod_poly = []
            
            for p1, p2 in zip(poly, poly[1:]):
                mod_poly.append(p1)
                for p_best in best:
                    # find segment of poly where find the points of best
                    if p1[0] < p_best[0] <= p2[0]:
                        new_point_poly = value_in_segmentp1p2(p1, p2, p_best[0])
                        mod_poly.append((p_best[0], new_point_poly))
            mod_poly.append(poly[-1])

            if check_poly(mod_poly, best) is True:
                best_polygon = jj
        best_polygon = [(best_polygon, [points[best_polygon][-1]])]
    satisfy_profit += best_polygon
    # compare the eventual best polygonal chain with the points that exceed desired profit and choose the one with lower z value
    minValue = min(satisfy_profit, key = lambda t: t[1][0][1])[1][0][1]
    # two or more points may have the same value of z, choose the one with max v
    best = max([x for x in satisfy_profit if x[1][0][1] == minValue], key = lambda d: d[1][0][0])
    return best[0]

def calc_z(state, items, i):
    z_c = []
    z_o = []
    # compute z for closed cluster
    if len(state["C"]) > 0:
        for cc in state["C"]:
            z_c.append((items.get_item(cc[-1]-1).price - items.get_item(cc[0]-1).price)/2)
    else:
        z_c.append(0)

    # estimate lower bound on z for open cluster
    if len(state["O"]) > 0:
        # find p'(K) for each open cluster
        for j, cc in enumerate(state["O"]):
            candiate_lastp = i+j+1
            # printv(f'for {cc} the candidate next element is {candiate_lastp}')
            z_o.append((items.get_item(candiate_lastp-1).price - items.get_item(cc[0]-1).price)/2)
    else:
        z_o.append(0)
    # take the max z in open/closed clusters z call
    return  max(max(z_c), max(z_o))


def calc_v_closed_clusters(state, items, z):
    q = []
    d = []
    v_c = 0
    # calculus of v for closed clusters
    for c in state["C"]:
        p_min = items.get_item(c[0]-1).price
        p_max = items.get_item(c[-1]-1).price
        cluster_price = min(p_min + z, p_max)
        q.append(cluster_price)
        sum_demand = 0
        for j in c:
            # contribution item j = dj * min{p−(K) + z, p+(K)}
            demand = items.get_item(j-1).demand
            v_c += demand * cluster_price
            sum_demand += demand
        d.append(sum_demand)
    return q, d, v_c

def calc_v_open_clusters(state, items, i, z):
            # estimates bound v on open clusters
            v_o = [0, 0]
            for c in state["O"]:
                p_i_next = items.get_item(i).price
                p_min = items.get_item(c[0]-1).price
                for j in c:
                    # dj min{p−(K) + z, pi+1} ≤ contribution item j ≤ dj (p−(K) + z).
                    lower_bound = items.get_item(j-1).demand * min(p_min + z, p_i_next)
                    high_bound = items.get_item(j-1).demand * (p_min + z)
                    v_o[0] += lower_bound
                    v_o[1] += high_bound

            return v_o

#----------------------------------------------------------------------------------------------------------

# read configuration file
fconfig = sys.argv[1]
config = configparser.ConfigParser()
config.read(fconfig)
instance_path = config['inputfile']['fileName']
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
printv('Base cases creation...\n')
states[0] = [{"C":[], "O":[], "z":0, "v":0}]

printv(*(f'\tState {x[0]} -> {x[1]}' for x in states.items()), sep='\n')

# extension of the states
printv("\nExtension of the labes...")
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

    # sort the open clusters in ascending order of their first point
    for c in candidate_states:
        c["O"].sort()
    printv(candidate_states)
    
    limit = items.N - i
    oldl = len(candidate_states)
    candidate_states = [c for c in candidate_states if len(c["O"]) <= limit]
    printv(f"Removal of states that violate |O| <= n-i = {limit}, for a total of {oldl - len(candidate_states)} states")
    printv(candidate_states)

    points_list = []
    original_profit = sum(map(lambda item: item.demand*item.price, items.items[0: i]))
    desired_profit = original_profit*profit_margin
    printv(f"For itemset to {i} with a margin of {profit_margin} the desired profit is {desired_profit}")

    # find lower bound profit on item after i
    lb_profit_after_i = sum([i.price*i.demand for i in items.items[i:]])

    # compute v, z and the points for dominance checking
    to_keep = []
    for count, c in enumerate(candidate_states):
        print(f'\n{c}')
        # calc z for the candidate state
        z = calc_z(c, items, i)
        c["z"] = z
        # calc v for the candidate state
        q, d, v_c = calc_v_closed_clusters(c, items, z)
        v_o = calc_v_open_clusters(c, items, i, z)
        print(v_c)
        print(v_o)
        v = [v_o[0] + v_c, v_o[1] + v_c]
        c["v"] = v
        # bound on desired profit
        bound_profit = desired_profit - v_o[0]
        printv(f'The desired profit without the bound {bound_profit}')
        mod_s1 = {'v': v_c, 'z': z, 'q': q, 'd': d, 's': [s[0] for s in c['C']], 'e': [s[-1] for s in c['C']]}
        printv(c)
        points, max_profit = find_stationary_points(mod_s1, bound_profit)
        # find the max profit of the solution considering closed and open clusters + lower bound on item after i
        profit_solution = max_profit + v_o[0] + lb_profit_after_i
        printv(f'Max profit of the solution with closed/open clusters + lower bound item after i: {profit_solution}')
        if profit_solution >= desired_profit:
            printv("Solution can reach desired profit with open cluster")
            points_list.append(points)
            to_keep.append(count)
        else:
            printv("Solution cant reach with open cluster, discarded")

    removed = len(candidate_states) - len(to_keep)
    candidate_states = [candidate_states[i] for i in to_keep]

    dominated = []    
    # dominance check
    printv("\nDominance check...")
    permut = list(itertools.permutations(range(len(candidate_states)),2))
    for i1, i2 in permut:
        if i1 not in dominated and i2 not in dominated:
            s1 = candidate_states[i1]
            s2 = candidate_states[i2]
            if len(s1["C"]) <= len(s2["C"]) and s1['z']<=s2['z'] and len(s1["O"]) == len(s2["O"]):
                # s1 dominate s2 if...
                # ... check profit clusters
                printv(f'\nChecking {i1} and {i2}\n{i1}: {s1}\n{i2}: {s2}')
                if len(s1["C"]) > 0:
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
                
                # ... find a valid permutation of a state that satisfy the two inequalities in all open cluster's pairs
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
                                printv(f"Permutation {perm} don't satisfy constranint on open cluster")
                                break
                            found = True
                        if found:
                            printv(f"Itemsets in {s1_O} and {s2_reorder} satisfy constraint on open clusters.")
                            break      
                if found:
                    printv(f'Found that state {i2} is dominated by {i1}\n')
                    dominated.append(i2)
                else:
                    printv('No permutation of open clusters satisfy constraints, cant establish dominance\n')
    candidate_states = [c for count, c in enumerate(candidate_states) if count not in dominated]
    printv(f'Removed {len(dominated) + removed} states, dominated: {dominated}')

    # assign non-dominated solution
    states[i] = candidate_states
    
    printv("______________________________________________________________________________________________________________________")

printv('Computed all the state labels:')
printv(*(f'\tItem {x[0]}\n {x[1]}' for x in states.items()), sep='\n')

# ============================================================================================================
# terminantion check
printv(f'\nTerminantion check for optimal values.')

last_states = states[items.N]
original_profit = sum(map(lambda item: item.demand*item.price, items.items[0: items.N]))
desired_profit = original_profit*profit_margin
printv(f'\nFor {items.N} items the original profit is {original_profit} and the desired profit is {desired_profit}')
optimal_pairs = {}
for count, state in enumerate(last_states):
    z = state['z']
    v = state['v'][0]
    if v >= desired_profit:
        printv(f'{state} satisfy profit margin.')
    else:
        printv(f'{state} don\'t satisfy profit margin, extend the state...')
        # find the stationary points and take only the last one, which correponds to the desired profit
        q, d, _ = calc_v_closed_clusters(state, items, z)
        mod_s = {'v': v, 'z': z, 'q': q, 'd': d, 's': [s[0] for s in state['C']], 'e': [s[-1] for s in state['C']]}
        points, _ = find_stationary_points(mod_s, desired_profit)
        v_z_last_point = points[-1]
        v = v_z_last_point[0]
        z = v_z_last_point[1]
    if v >= desired_profit:
        printv("State reach the desired profit!\n")
        optimal_pairs[count] = (v,z)
    else:
        printv("State don't reach the desired profit, removed.\n")

printv("\nThe optimal labels are:")
printv(*(f'\tWith {len(last_states[x[0]]["C"])} state {last_states[x[0]]}\n\tv: {x[1][0]} -> z: {x[1][1]}' for x in optimal_pairs.items()), sep='\n')