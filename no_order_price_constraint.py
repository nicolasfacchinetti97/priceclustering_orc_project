from more_itertools import first, last
from instance import instance
from item import item

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
states[0] = {"C":[], "O":[], "z":0, "v":0}

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
    candidate_states.extend(extension1(last_state, i))
    candidate_states.extend(extension2(last_state, i))
    candidate_states.append(extension3(last_state, i))
    candidate_states.append(extension4(last_state, i))
    printv(candidate_states)
    
    limit = items.N - i
    printv(f"Removal of states that violate |O| <= n-i = {limit}")
    candidate_states = [c for c in candidate_states if len(c["O"]) <= limit]
    printv(candidate_states)
    
    # sort the open clusters in ascending order of their first point

    # compute z for closed cluster
    z_c = []
    for c in candidate_states:
        if len(c["C"]) > 0:
            z_cc = []
            for cc in c["C"]:
                z_cc.append((items.get_item(cc[-1]-1).price - items.get_item(cc[0]-1).price)/2)
            z_c.append(z_cc)
        else:
            z_c.append([0])
    printv(f'\nz values for each closed cluster {z_c}')
    # take the max value of z for each state
    z_c = [max(c) for c in z_c]
    printv(f'max z value for each closed cluster {z_c}')
    
    # estimate lower bound on z for open cluster
    z_o = []
    for c in candidate_states:
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
    printv(f'z values estimate for each open cluster {z_o}')
    # take the max value of z for each state
    z_o = [max(c) for c in z_o]
    printv(f'max z value estimate for each open cluster {z_o}')
    # ZZZZZZZZZZZZZZZZZZZ METTI IL CALCOLO DELLE DUE Z IN UN SOLO CILO FOR -----------------------------------------------
    z = [max(z_c[j], z_o[j]) for j in range(len(z_o))]
    print(f"final z value: {z}")

    for count, cluster in enumerate(candidate_states):
        cluster['z'] = z[count]
    print(candidate_states)

    # v calculus 
    v_c = []
    for c in candidate_states:
        v_cc = 0
        z = c["z"]
        for cc in c["C"]:
            p_min = items.get_item(cc[0]-1).price
            p_max = items.get_item(cc[-1]-1).price
            for j in cc:
                # contribution item j = dj * min{p−(K) + z, p+(K)}
                contribution_j = items.get_item(j-1).demand * min(p_min + z, p_max)
                v_cc += contribution_j
        v_c.append(v_cc)
    printv(f'v values for each closed cluster {v_c}')
    # estimates bound v on open clusters
    v_o = []
    for c in candidate_states:
        v_oo = [0, 0]
        z = c["z"]
        p_i_next = items.get_item(i).price
        for cc in c["O"]:
            p_min = items.get_item(cc[0]-1).price
            for j in cc:
                # dj min{p−(K) + z, pi+1} ≤ contribution item j ≤ dj (p−(K) + z).
                lower_bound = items.get_item(j-1).demand * min(p_min + z, p_i_next)
                high_bound = items.get_item(j-1).demand * (p_min + z)
                v_oo[0] += lower_bound
                v_oo[1] += high_bound
        v_o.append(v_oo)
    printv(f'v estimates for each closed open cluster {v_o}')
    for count, cluster in enumerate(candidate_states):
        cluster['v'] = [v_o[count][0] + v_c[count], v_o[count][1] + v_c[count]]
    print(candidate_states)
    # dominance check

    # assign non-dominated solution
    states[i] = candidate_states[1]
    print()
