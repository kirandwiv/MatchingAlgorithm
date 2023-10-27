import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parallel_pandas import ParallelPandas
from funcs import *
import multiprocessing
import pebble
import json  
import ipywidgets as widgets
from concurrent.futures import ThreadPoolExecutor
from tarjan_alg import *  
import networkx as nx
import time

## Unpackers (these just unpack arguments for pre-existing functions so that they can be used with par.pool.map)
def mdf_np_unpack(item):
    ''' Unpacks the tuple of arguments for the mdf_np function. '''
    return mdf_np(*item)

def mdf_yp_unpack(item):
    ''' Unpacks the tuple of arguments for the mdf_yp function. '''
    return mdf_yp(*item)

def EADAM_unpack(item):
    ''' Unpacks the tuple of arguments for the EADAM function. '''
    return EADAM(*item)


### WANT TO HAVE TO DIVIDE UP TASKS ONLY ONCE. 

def s_simulate(item):
    n, k = item
    df = mdf_np(n, k)
    df1 = df.copy()
    result = EADAM(df, k)
    return df1, result


def f_simulate(nsims, n, k):
    input_ls = [(n,k)]*int(nsims)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = pool.map(s_simulate, input_ls) # parallelize
    return results    

def gs_simulate(item):
    n, k = item
    df = mdf_np(n, k)
    print('preference matrix created')
    preferences = df.copy()
    matches, _ = run_gale_shapley(df, k)
    print('gale_shapley done')
    _,n, cycles = find_cycles(preferences, matches, k)
    print('cycles found')
    n_in_cycles = 0
    for item in cycles:
        n_in_cycles += len(item)
    return n, n_in_cycles, len(matches), cycles

def gs_simulate_nx(item):
    start = time.time()
    n, k = item
    #df = mdf_np(n, k, c = 200)
    df = create_array(n,k)
    end1 = time.time()
    print(f"Time to make preferences: {end1-start}")
    preferences = df.copy()
    matches, _ = run_gale_shapley(df, k)
    end2 = time.time()
    print(f"Time to run GS: {end2-end1}")
    edgelist = to_edgelist(preferences, matches, k)
    end3 = time.time()
    print(f"Time to make edgelist: {end3-end2}")
    G= nx.from_pandas_edgelist(edgelist, source = 'new_id_x', target = 'new_id_y', create_using=nx.DiGraph())
    n, n_in_cycles, cycles = get_strongly_connected_components(G)
    end4 = time.time()
    print(f"Time to find SCC: {end4-end3}")
    return n, n_in_cycles, len(matches), cycles

def gs_simulate_nx_max_lengths(item, length_of = True):
    start = time.time()
    n, k = item
    #df = mdf_np(n, k, c = 200)
    df = create_array(n,k)
    end1 = time.time()
    print(f"Time to make preferences: {end1-start}")
    preferences = df.copy()
    matches, _ = run_gale_shapley(df, k)
    end2 = time.time()
    print(f"Time to run GS: {end2-end1}")
    n_changed, n_matches, x1, x2 = get_max_weight_matching(preferences, matches, n, k)
    cycle_lengths, _ = len_cycles(x1, x2)
    percent_lengths = [item/n_matches for item in cycle_lengths]
    return cycle_lengths, percent_lengths

def gs_simulate_nx_max(item, length_of = True):
    start = time.time()
    n, k = item
    #df = mdf_np(n, k, c = 200)
    df = create_array(n,k)
    end1 = time.time()
    print(f"Time to make preferences: {end1-start}")
    preferences = df.copy()
    matches, _ = run_gale_shapley(df, k)
    end2 = time.time()
    print(f"Time to run GS: {end2-end1}")
    n_changed, n_matches, x1, x2 = get_max_weight_matching(preferences, matches, n, k)
    return n_changed, n_matches

def gs_f_simulate(nsims, n, k):
    input_ls = [(n,k)]*int(nsims)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = pool.map(gs_simulate, input_ls) # parallelize
    return results

def gs_f_simulate_nx(nsims, n, k):
    input_ls = [(n,k)]*int(nsims)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = pool.map(gs_simulate_nx, input_ls) # parallelize
    return results

def gs_f_simulate_nx_max(nsims, n, k):
    input_ls = [(n,k)]*int(nsims)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = pool.map(gs_simulate_nx_max, input_ls) # parallelize
    return results

def gs_f_simulate_nx_max_lengths(nsims, n, k):
    input_ls = [(n,k)]*int(nsims)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = pool.map(gs_simulate_nx_max_lengths, input_ls) # parallelize
    return results

def s_simulate_MM_EA_GS(item):
    n, k = item
    df = mdf_np(n, k)
    preferences = df.copy()
    _, GS_result, _, _, eadam_results = EADAM(df, k)
    n_changes, n_matches, x1, x2 = get_max_weight_matching(preferences, GS_result, n, k, EADAM = False)
    n_changes, n_matches, x_eadam_1, x_eadam_2 = get_max_weight_matching(preferences, eadam_results, n, k, EADAM = True)
    cycle_lengths, _ = len_cycles(x1, x2)
    percent_lengths = [item/n_matches for item in cycle_lengths]
    cycle_lengths1, _ = len_cycles(x_eadam_1, x_eadam_2)
    percent_lengths1 = [item/n_matches for item in cycle_lengths1]
    return cycle_lengths, percent_lengths, cycle_lengths1, percent_lengths1

def f_simulate_MM_EA_GS(nsims, n, k):
    input_ls = [(n,k)]*int(nsims)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = pool.map(s_simulate_MM_EA_GS, input_ls) # parallelize
    return results   

## Simulations 

def simulate(nsims, n, k):
    ''' Simulates nsims runs of EADAM where n is the number of agents/institutions and k is the size of the preference list.
    Should return a list of results, one for each simulation. We'd like to have returned to us. Fraction of people who change
    matching between a one-shot GS and EADAM. And the number of rounds of EADAM we went through.
    _______
    Parameters:
    nsims: int, number of simulations to run
    n: int, number of agents/institutions
    k: int, size of preference list (k = 3, 5, 10)
    '''
    ### Create a list of preference dataframes. One dataframe per simulation. 
    input_ls = [(n,k)]*int(nsims)
    par = pebble.ProcessPool(multiprocessing.cpu_count()).map(mdf_np_unpack, input_ls) # parallelize the creation of the preference dataframes
    dfs_list = list(par.result()) # creates a list of df preferences. 
    dfs_list = [(item, k) for item in dfs_list]

    ### Run EADAM on each preference dataframe.

    par2 = pebble.ProcessPool(64).map(EADAM_unpack, dfs_list) # parallelize the running of EADAM
    results = list(par2.result()) # creates a list of results.
    ## Note that the outputs from EADAM are:
    # 1. sp_f, 
    # 2. gs_result, 
    # 3. iter_list, 
    # 4. j, 
    # 5. EADAM_result

    return results

## Analysis and Plotting Functions

def obtain_n_diffs(results):
    '''
    Takes the results from a batch of simulations
    and returns a list of tuples. Where first term is the difference, second term is the length of initial matches. 
    '''
    relevant = [[item[1], item[4]] for item in results]
    differences = [(len(find_diff(item[0], item[1])), len(item[0])) for item in relevant]
    percent_diff = [item[0]/item[1] for item in differences]
    return differences, percent_diff

def make_df(n, k, differences, percent_diff, save = False):
    df = pd.DataFrame({'n': [n]*len(differences), 'k': [k]*len(differences),
                   'differences': [item[0] for item in differences], 'percent_diff': percent_diff,
                   'matches': [item[1] for item in differences]})
    if save == True:
        df.to_csv(f'data/simulations/n_{n}_k_{k}.csv') ## if wanted saved, saves with parameters in name.
    return df

def concat(list_of_files, path):
    '''
    Takes a list of files and concatenates them into a single dataframe
    '''
    li = []
    for filename in list_of_files:
        df = pd.read_csv(path + filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame

def find_cycles(preferences, matches, k):
    '''
    Function that takes as an input the results of a simulation and returns the number
    of cycles in the GS result.
    _______
    Inputs:
    preference: the initial preference dataframe
    matches: the GS matches dataframe
    '''    
    ## Step 1: drop all unmatched students from the preference dataframe.
    preferences = preferences[preferences['student_id'].isin(matches['student_id'])]
    preferences.reset_index(inplace = True, drop = True) 
    
    ## Step 2: Remove students who got preferred choice
    preferences['rejections'] = matches.applications
    relevant = preferences[preferences['rejections'] != 0] ## drop all who were never rejected. They will necessarily not point to anyone else. 
    
    ## Step 3: For others, keep only preferences above match. 
    for i in range(1,k):
        relevant.iloc[:, i] = np.where(relevant['rejections']<i+1, -100, relevant.iloc[:, i])
    relevant.set_index('student_id', inplace = True)
    pointing = pd.DataFrame(relevant.iloc[:, :k].stack(level = 0)).reset_index()
    pointing = pointing[pointing[0] != -100]
    
    ## Step 4: Prepare for Tarjan's Algorithm
    to_merge = matches.loc[:,[0, 'student_id']]
    pointing = pointing.merge(to_merge, on = 0, how = 'left')
    pointing = pointing[pointing['student_id_y'].isin(pointing['student_id_x'])]
    normalizer = pd.DataFrame(pd.concat([pointing['student_id_x'], pointing['student_id_y']], axis = 0).unique())
    normalizer['new_id'] = normalizer.index
    pointing = pointing.merge(normalizer, left_on = 'student_id_x', right_on = 0, how = 'left')
    pointing = pointing.merge(normalizer, left_on = 'student_id_y', right_on = 0, how = 'left')
    pairs = pointing[['new_id_x', 'new_id_y']]
    
    ## Step 5: Run Tarjan's Algorithm
    g = Graph(len(normalizer))
    for i in range(len(pairs)):
        g.addEdge(pairs.iloc[i, 0], pairs.iloc[i, 1])
    g.SCC()
    return pairs, g.Cycle, g.cycles

def to_edgelist(preferences, matches, k):
    '''
    Function that takes as an input the results of a simulation and returns the number
    of cycles in the GS result.
    _______
    Inputs:
    preference: the initial preference dataframe
    matches: the GS matches dataframe
    '''    
    ## Step 1: drop all unmatched students from the preference dataframe.
    preferences = preferences[preferences['student_id'].isin(matches['student_id'])]
    preferences.reset_index(inplace = True, drop = True) 
    
    ## Step 2: Remove students who got preferred choice
    preferences['rejections'] = matches.applications
    relevant = preferences[preferences['rejections'] != 0] ## drop all who were never rejected. They will necessarily not point to anyone else. 
    
    ## Step 3: For others, keep only preferences above match. 
    for i in range(1,k):
        relevant.iloc[:, i] = np.where(relevant['rejections']<i+1, -100, relevant.iloc[:, i])
    relevant.set_index('student_id', inplace = True)
    pointing = pd.DataFrame(relevant.iloc[:, :k].stack(level = 0)).reset_index()
    pointing = pointing[pointing[0] != -100]
    
    ## Step 4: Prepare for Tarjan's Algorithm
    to_merge = matches.loc[:,[0, 'student_id']]
    pointing = pointing.merge(to_merge, on = 0, how = 'left')
    pointing = pointing[pointing['student_id_y'].isin(pointing['student_id_x'])]
    normalizer = pd.DataFrame(pd.concat([pointing['student_id_x'], pointing['student_id_y']], axis = 0).unique())
    normalizer['new_id'] = normalizer.index
    pointing = pointing.merge(normalizer, left_on = 'student_id_x', right_on = 0, how = 'left')
    pointing = pointing.merge(normalizer, left_on = 'student_id_y', right_on = 0, how = 'left')
    pairs = pointing[['new_id_x', 'new_id_y']]
    return pairs

def get_strongly_connected_components(G):
    scc = []
    n_in_scc = 0
    n_scc = 0
    for cc in nx.kosaraju_strongly_connected_components(G):
        if len(cc)>1:
           scc.append(list(cc))
           n_in_scc += len(cc)
           n_scc += 1
    return n_scc, n_in_scc, scc

def make_df_cycles(n, k, results, save = False, path = 'data/simulations/cycles/'):
    n_cycles = [item[0] for item in results]
    n_agents_in_cycles = [item[1] for item in results]
    n_matches = [item[2] for item in results]
    percent_in_cycles = [item[1]/item[2] for item in results]
    df = pd.DataFrame({'n': [n]*len(n_cycles), 'k': [k]*len(n_cycles),
                   'n_cycles': n_cycles, 'n_agents_in_cycles': n_agents_in_cycles,
                   'n_matches': n_matches, 'percent_in_cycles': percent_in_cycles})
    if save == True:
        df.to_csv(path +f'n_{n}_k_{k}_cycles.csv')
    return df

def get_max_weight_matching(preferences, matches, n, k, EADAM = False):
    '''
    This algorithm identifies the max_weight_matching among agents
    and determines the number of changes between the GS algorithm and
    the max_weight_matching.
    __________
    Inputs:
    preferences: the initial preference dataframe
    matches: the GS matches dataframe 
    '''
    if EADAM == False:
        # Step 1: Remove Unmatched Students
        preferences = preferences[preferences['student_id'].isin(matches['student_id'])]
        preferences.reset_index(inplace = True, drop = True) 
        # Step 2: Remove students who got their top choice (they'll never be improved)
        preferences['rejections'] = matches.applications
        relevant = preferences[preferences['rejections'] != 0] 
        # Step 3: For others, keep only preferences above match AND their Match.
        for i in range(1,k):
            relevant.iloc[:, i] = np.where(relevant['rejections']<i, -100, relevant.iloc[:, i])
        relevant.set_index('student_id', inplace = True)
        n_matches = len(matches)
        # Stack the DataFrame to create an edgelist
        pointing = pd.DataFrame(relevant.iloc[:, :k].stack(level = 0)).reset_index()
        # Drop all irrelevant matches 
        pointing = pointing[pointing[0] != -100]
        pointing = pointing.reset_index(drop=True)
        # Set Appropriate Weigths for the edgelist (3n on match, 3n+1 on all preferred)
        l1 = [([k*n+1]*(l) + [k*n]) for l in relevant.rejections]
        l2 = [item for sublist in l1 for item in sublist]
        pointing['weight'] = l2
        # Remove schools matched to a first-choice student. No-one should bother pointing to those
        first_match_schools = matches[matches['applications'] == 0][0]
        pointing = pointing[~(pointing[0].isin(first_match_schools))]
        # Add S to denote "school". We want to make sure they're not being confused
        pointing[0] = pointing[0].astype(str)+'S' 
        pointing.drop('level_1', axis = 1, inplace = True)
        pointing.columns = ['source', 'target', 'weight']
        
        # Create Graph
        G= nx.from_pandas_edgelist(pointing, edge_attr = True)
        # Solve for Max Weight Matching
        max_weight_matching = nx.max_weight_matching(G)
        max_matches = pd.DataFrame(max_weight_matching)
        matches = matches[matches['applications']!=0]
        matches = matches[[0, 'student_id']]
        matches[0]= matches[0].astype(str)+'S'
        mask = max_matches[0].isin(matches[0])
        max_matches['school_id'] = np.where(max_matches[0].isin(matches[0]), max_matches[0], max_matches[1])
        max_matches['student_id'] = np.where(max_matches[0].isin(matches[0]), max_matches[1], max_matches[0])
        x2 = set(zip(max_matches['school_id'], max_matches['student_id']))
        x1 = set(zip(matches[0], matches['student_id']))
        n_diff = len(x1.difference(x2))
        return n_diff, n_matches, x1, x2
    else:
        eadam_results = matches.copy()
        n_matches = len(matches)
        schools_to_remove = eadam_results[eadam_results['applications'] == 0][0]
        eadam_results = eadam_results[eadam_results['applications'] != 0] ## Here I dropped the rows where only a single application was made. 
        preferences = preferences[preferences['student_id'].isin(eadam_results['student_id'])]
        preferences.reset_index(inplace = True, drop = True) 
        preferences.sort_values(by = 'student_id', inplace = True)
        eadam_results.sort_values(by = 'student_id', inplace = True)
        eadam_results.reset_index(inplace = True, drop = True)
        has_checked = np.zeros(len(preferences))
        for i in range(0,k):
            preferences.iloc[:,i] = np.where(has_checked == False, preferences.iloc[:, i], -100)
            has_checked = np.where(preferences.iloc[:, i] == eadam_results.iloc[:, 0], True, has_checked)  
        relevant = pd.DataFrame(preferences.iloc[:, :k].stack(level = 0)).reset_index()
        pointing = relevant[relevant[0] != -100]
        pointing = pointing.reset_index(drop=True)
        # Set Appropriate Weigths for the edgelist (3n on match, 3n+1 on all preferred)
        sizes = pointing.groupby('level_0').count()[0].to_list()
        l1 = [([k*n+1]*(l-1) + [k*n]) for l in sizes]
        l2 = [item for sublist in l1 for item in sublist]
        pointing['weight'] = l2
        pointing = pointing[~pointing[0].isin(schools_to_remove)]
        pointing[0] = pointing[0].astype(str)+'S' 
        pointing.drop('level_1', axis = 1, inplace = True)
        pointing.columns = ['source', 'target', 'weight']
        
        # Create Graph
        original = pointing[pointing.weight == k*n]
        original = original[['source', 'target']]
        G= nx.from_pandas_edgelist(pointing, edge_attr = True)
        # Solve for Max Weight Matching
        max_weight_matching = nx.max_weight_matching(G)
        max_matches = pd.DataFrame(max_weight_matching)
        max_matches['school_id'] = np.where(max_matches[0].isin(original['target']), max_matches[0], max_matches[1])
        max_matches['student_id'] = np.where(max_matches[0].isin(original['target']), max_matches[1], max_matches[0])
        x2 = set(zip(max_matches['school_id'], max_matches['student_id']))
        x1 = set(zip(original['target'], original['source']))
        n_diff = len(x1.difference(x2))
        return n_diff, n_matches, x1, x2

def make_df_max_match_length(n, k, results, save = False, path = 'data/simulations/max_length_matches_w_eadam/'):
    results1 = [item[0] for item in results]
    results2 = [item[1] for item in results]
    results3 = [item[2] for item in results]
    results4 = [item[3] for item in results]
    cycle_lengths = [item for sublist in results1 for item in sublist]
    as_percent_of_matches = [item for sublist in results2 for item in sublist]
    cycle_lengths_eadam = [item for sublist in results3 for item in sublist]
    as_percent_of_matches_eadam = [item for sublist in results4 for item in sublist]
    df = pd.DataFrame({'n': [n]*len(cycle_lengths), 'k': [k]*len(cycle_lengths), 'cycle_lengths': cycle_lengths,
                       'as_percent_of_matches': as_percent_of_matches, 'cycle_lengths_eadam': cycle_lengths_eadam, 'as_percent_of_matches_eadam': as_percent_of_matches_eadam})
    if save == True:
        df.to_csv(path +f'n_{n}_k_{k}_max_length_diff.csv')
    return df

def make_df_max_match(n, k, results, save = False, path = 'data/simulations/max_matches_1000_4_6_8/'):
    n_changes = [item[0] for item in results]
    n_matches = [item[1] for item in results]
    percent_changed = [item[0]/item[1] for item in results]
    df = pd.DataFrame({'n': [n]*len(n_matches), 'k': [k]*len(n_changes),
                   'n_changes': n_changes,
                   'n_matches': n_matches, 'percent_changed': percent_changed})
    if save == True:
        df.to_csv(path +f'n_{n}_k_{k}_max_diff.csv')
    return df

def len_cycles(x1, x2):
    '''
    This function takes as inputs the results from the Gale-Shapley and the Max Matching and returns the length of the implied cycles,
    as well as the cycles themselves. 
    '''
    original_match = dict(x1)
    difference = dict(x2.difference(x1))
    difference = {difference[i]:i for i in difference} # reverse the dictionary for ease of use
    
    explored = [] # initiate list of explored nodes
    unexplored = list(difference.keys()) # initiate list of unexplored nodes
    cycle_lengths = [] # initiate list of cycle lengths
    cycles = [] # initiate list of cycles

    while len(unexplored) != 0:
        current_node = unexplored[0] # start with first unexplored node 
        current_cycle = [] # initiate list of nodes in current cycle
        while current_node not in current_cycle: # condition is that we've not already visited this node
            current_cycle.append(current_node) # add the current node to our current cycle
            explored.append(current_node) # add the current node to our list of explored nodes
            unexplored.remove(current_node) # remove the current node from our list of unexplored nodes
            matched_school = difference[current_node] # find the school that the current node is matched to
            current_node = original_match[matched_school] # find the student that the school was matched to originally. Then repeat with that node.
        cycle_lengths.append(len(current_cycle))  # once we've found a cycle, add the length of the cycle to our list of cycle lengths
        cycles.append(current_cycle) # once we've found a cycle, add the cycle to our list of cycles

    return cycle_lengths, cycles
