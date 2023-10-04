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

def concat(list_of_files):
    '''
    Takes a list of files and concatenates them into a single dataframe
    '''
    li = []
    for filename in list_of_files:
        df = pd.read_csv('data/simulations/' + filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame