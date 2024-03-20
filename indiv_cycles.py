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
from multi_funcs import *

#%% From GS to Graph

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
    preferences['rejections'] = matches.applications

    relevant = preferences.copy()
    
    ## Step 2: Remove students who got preferred choice
    #relevant = preferences[preferences['rejections'] != 0] ## drop all who were never rejected. They will necessarily not point to anyone else. 
    
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
    pairs = pairs[pairs['new_id_x'] != pairs['new_id_y']]
    return pairs


#%% Find Min Length of Individual Cycles

import networkx as nx

def all_pairs_shortest_cycle_length(G):
    # Compute all pairs shortest path lengths
    shortest_path_lengths = nx.all_pairs_shortest_path_length(G)
    
    shortest_path_lengths = dict(shortest_path_lengths)

    # Dictionary to store shortest cycle lengths
    shortest_cycle_lengths = {}

    # Iterate over each node in the graph
    for i in G.nodes():
        shortest_cycle = float('inf')
        
        # Check one-hop edges from node i
        for j in G.successors(i):
            # Check if there is a shortest path from j back to i
            if i in shortest_path_lengths[j]:
                cycle_length = shortest_path_lengths[j][i] + 1
                shortest_cycle = min(shortest_cycle, cycle_length)
        
        # If shortest_cycle is still infinity, no cycle found
        if shortest_cycle == float('inf'):
            shortest_cycle_lengths[i] = -1
        else:
            shortest_cycle_lengths[i] = shortest_cycle
            
    return shortest_cycle_lengths

#%% Putting all together in a function

def ic_simulate(item):
    n, k = item
    df = create_array(n,k)
    preferences = df.copy()
    preferences2 = df.copy()
    matches, _ = run_gale_shapley(df, k)
    edgelist = to_edgelist(preferences, matches, k)
    G = nx.from_pandas_edgelist(edgelist, source = 'new_id_x', target = 'new_id_y', create_using=nx.DiGraph())
    shortest_cycle_lengths = all_pairs_shortest_cycle_length(G)
    shortest_cycle_lengths = list(shortest_cycle_lengths.values())
    return shortest_cycle_lengths


def ic_simulate_par(n, k, nsims):
    input_ls = [(n,k)]*int(nsims)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = pool.map(ic_simulate, input_ls) # parallelize
    return results

#%% Organizing the results
def make_df_ic(results, n, k):    
    results = [(item, id) for id, sublist in enumerate(results) for item in sublist]
    results = pd.DataFrame(results)
    results = results[results[0]!=-1]
    results.columns = ['cycle_length', 'sim_id']
    results.to_csv(f'data/simulations/ic_sims/results_{n}_{k}.csv')
    return results