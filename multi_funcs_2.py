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



#%% Function that Returns SCC, EADAM AND MM for n,k. 

def s_simulate_all(item):
    n, k = item
    df = create_array(n,k)
    preferences = df.copy()
    preferences2 = df.copy()
    
    ## Run Gale-Shapley and EADAM
    _, GS_result, _, j, eadam_results = EADAM(df, k)
    n_changes_eadam = len(find_diff(GS_result, eadam_results))
    GS_result2 = GS_result.copy()
    
    ## Get MM results
    n_changes, n_matches, x1, x2 = get_max_weight_matching(preferences, GS_result, n, k, EADAM = False)
    cycle_lengths, _ = len_cycles(x1, x2)
    percent_lengths = [item/n_matches for item in cycle_lengths]
    eadam_results[0] = eadam_results[0].astype(str)+'S'
    x3 = set(zip(eadam_results[0], eadam_results['student_id']))
    GS_result[0] = GS_result[0].astype(str)+'S'
    x4 = set(zip(GS_result[0], GS_result['student_id']))
    cycle_lengths1, _ = len_cycles(x4, x3)
    percent_lengths1 = [item/n_matches for item in cycle_lengths1]
    
    ## Get SCC results
    edgelist = to_edgelist(preferences2, GS_result2, k)
    G= nx.from_pandas_edgelist(edgelist, source = 'new_id_x', target = 'new_id_y', create_using=nx.DiGraph())
    n, n_in_cycles, cycles = get_strongly_connected_components(G)
    
    return cycle_lengths, percent_lengths, cycle_lengths1, percent_lengths1, n_changes, n_matches, n_changes_eadam, j, n, n_in_cycles

def f_simulate_all(nsims, n, k):
    input_ls = [(n,k)]*int(nsims)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = pool.map(s_simulate_all, input_ls) # parallelize
    return results  


#%% Make Dataframes

def make_df_all(n, k, results, save = False, path = 'data/simulations/new_scale/'):
    results1 = [item[0] for item in results]
    results2 = [item[1] for item in results]
    results3 = [item[2] for item in results]
    results4 = [item[3] for item in results]
    results5 = [item[4] for item in results]
    results6 = [item[5] for item in results]
    results7 = [item[6] for item in results]
    results8 = [item[7] for item in results]
    results9 = [item[8] for item in results]
    results10 = [item[9] for item in results]
    cycle_lengths = [item for sublist in results1 for item in sublist]
    as_percent_of_matches = [item for sublist in results2 for item in sublist]
    cycle_lengths_eadam = [item for sublist in results3 for item in sublist]
    as_percent_of_matches_eadam = [item for sublist in results4 for item in sublist]
    df = pd.DataFrame({'n': [n]*len(cycle_lengths), 'k': [k]*len(cycle_lengths), 'cycle_lengths': cycle_lengths, 'as_percent_of_matches': as_percent_of_matches})
    df2 = pd.DataFrame({'n': [n]*len(cycle_lengths_eadam), 'k': [k]*len(cycle_lengths_eadam), 'cycle_lengths_eadam': cycle_lengths_eadam, 'as_percent_of_matches_eadam': as_percent_of_matches_eadam})
    df3 = pd.DataFrame({'n': [n]*len(results5), 'k': [k]*len(results5), 'n_changes': results5, 'n_matches': results6})
    df4 = pd.DataFrame({'n': [n]*len(results7), 'k': [k]*len(results7), 'n_changes_eadam': results7, 'n_matches': results6,'n_iterations': results8})
    df5 = pd.DataFrame({'n': [n]*len(results9), 'k': [k]*len(results9), 'n': results9, 'n_in_cycles': results10})
    if save == True:
        df.to_csv(path +f'n_{n}_k_{k}_max_length_diff.csv')
        df2.to_csv(path +f'n_{n}_k_{k}_max_length_diff_eadam.csv')
        df3.to_csv(path +f'n_{n}_k_{k}_max_diff.csv')
        df4.to_csv(path +f'n_{n}_k_{k}_max_diff_eadam.csv')
        df5.to_csv(path+ f'n_{n}_k_{k}_n_scc.csv')
    return df