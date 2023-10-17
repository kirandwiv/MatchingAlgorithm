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

def get_max_weight_matching(preferences, matches, n, k):
    '''
    This algorithm identifies the max_weight_matching among agents
    and determines the number of changes between the GS algorithm and
    the max_weight_matching.
    __________
    Inputs:
    preferences: the initial preference dataframe
    matches: the GS matches dataframe 
    '''
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
    
    # Stack the DataFrame to create an edgelist
    pointing = pd.DataFrame(relevant.iloc[:, :3].stack(level = 0)).reset_index()
    # Drop all irrelevant matches 
    pointing = pointing[pointing[0] != -100]
    # Set Appropriate Weigths for the edgelist (3n on match, 3n+1 on all preferred)
    l1 = [([3*len(preferences)+1]*(k) + [3*len(preferences)]) for k in relevant.rejections]
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
    
    