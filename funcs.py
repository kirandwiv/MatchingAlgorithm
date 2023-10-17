import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parallel_pandas import ParallelPandas
pd.options.mode.chained_assignment = None  # default='warn'
from random import sample  

ParallelPandas.initialize(n_cpu=16, split_factor=4, disable_pr_bar=True)

def create_array(n, k):  
    """  
    Creates an n x k array where the observations are drawn from (1, n) and there is no repetition of values across rows.  
  
    :param n: Number of rows  
    :param k: Number of columns  
    :return: n x k array  
    """  
    if k > n:  
        raise ValueError("k should be less than or equal to n.")  
  
    array = []  
    for row in range(n):  
        array.append(sample(range(1, n + 1), k))
        
    x = pd.DataFrame(array)
    x = x.iloc[:n].reset_index(drop=True)
    x['student_id'] = x.index
    x['applications'] = 0
    #x['N'] = 0
    x['k'] = 0
    x['matched'] = False
    #x[['rank1', 'rank2', 'rank3']] = np.random.uniform(size=(n,3)) We need to change this so that it applies to all k
    for i in np.arange(k):
        var = 'rank' + str(i+1)
        x[var] = np.random.uniform(size=(n,1))
    x['underdemanded'] = True
  
    return x  

## Functions for generating data
def mdf_np(n, k = 3, c=10):
    '''
    This function generates a dataframe of student preferences and school scores over those students.
    This function does this without parallelization. 
    _____
    n: number of students/schools
    k: depth of student preferences 
    '''
    x = pd.DataFrame(np.random.randint(0, n, (int(n*c),k)))
    x = x[~(x.apply(lambda row: len(row) != len(set(row)), axis=1))]
    if len(x)<n:
        return mdf_np(n, k, c*1.5)
    else:
        x = x.iloc[:n].reset_index(drop=True)
        x['student_id'] = x.index
        x['applications'] = 0
        #x['N'] = 0
        x['k'] = 0
        x['matched'] = False
        #x[['rank1', 'rank2', 'rank3']] = np.random.uniform(size=(n,3)) We need to change this so that it applies to all k
        for i in np.arange(k):
            var = 'rank' + str(i+1)
            x[var] = np.random.uniform(size=(n,1))
        x['underdemanded'] = True
    return x

def mdf_yp(n, k = 3):
    '''
    This function generates a dataframe of student preferences and school scores over those students.
    This function does this with parallelization.
    _____
    n: number of students/schools
    k: depth of student preferences
    '''
    x = pd.DataFrame(np.random.randint(0, n, (int(n*1.2),k)))
    x = x[~(x.p_apply(lambda row: len(row) != len(set(row)), axis=1))]
    if len(x)<n:
        return "Oops"
    else:
        x = x.iloc[:n].reset_index(drop=True)
        x['student_id'] = x.index
        x['applications'] = 0
        #x['N'] = 0
        x['k'] = 0
        x['matched'] = False
        for i in np.arange(k):
            var = 'rank' + str(i+1)
            x[var] = np.random.uniform(size=(n,1))
        x['underdemanded'] = True
    return x

## Gale-Shapley Algorithm Functions

def find_matches_3(sp):
    '''
    This function takes a dataframe of student preferences and returns a dataframe of matches.
    It does this by first sorting the dataframe by rank1 (the schools score for a student) and then 
    dropping duplicates from the 0 column (which gives the ID each student has applied to).
    _____
    sp: dataframe of student preferences and school scores over those students.
    '''
    x = sp.sort_values(['rank1'], ascending = False).drop_duplicates(subset = 0)
    return x[[0,'student_id', 'k']] 

def create_mask(sp3, sp, k=3):
    '''
    This function takes a dataframe of matches and a dataframe of student preferences and returns a mask of students who have not been assigned to a school.
    '''
    mask = ~(np.logical_or(sp['student_id'].isin(sp3['student_id']), sp['applications'] == k)) ## creates a mask of all students who have not been assigned to a school. 
    return mask ## The mask gives all values that should be shifted over.

def shift(sp, mask, k = 3):
    '''
    This function takes a dataframe of student preferences, a mask of students who have not been assigned to a school, and the number of schools to apply to. 
    It returns an updated dataframe of student preferences. In this dataframe we have shifted over the values for students who have not been assigned to a school.
    We also update the number of failed applications each student has made.
    '''
    for i in np.arange(k-1):
        var1 = 'rank' + str(i+1)
        var2 = 'rank' + str(i+2)
        sp.loc[mask, i] = sp.loc[mask, i+1] # Shifts over all the values for relevant columns.
        sp.loc[mask, var1] = sp.loc[mask, var2]
    sp.loc[mask, 'applications'] += 1 ## Updates to new values.
    #sp.loc[mask, 'N'] += sp.loc[mask, 'k'] ## Update to show how many students have applied to each school.
    return sp ## Returns the updated dataframe.

def find_k(x):
    x['k'] = x.groupby(0)['student_id'].transform('count')
    return x

## Gale-Shapley Algorithm

def run_gale_shapley(sp, k = 3):
    '''Runs the Gale-Shapley Algorithm for a particular dataframe.
    ______
    sp: dataframe of student preferences and school scores over those students.    
    '''
    r_apps = 0
    i=0
    mask = [1]
    while r_apps < k and sum(mask) > 0: ## note that there are two ways to end the cycle. 1 all unmatched students have applied to 3 schools. 2. There are no unmatched students.
        ## Step 1: Find k
        sp = find_k(sp)
        #sp['underdemanded'] = np.where(sp['k']>1, False, sp['underdemanded'])

        ## Step 2: Find matches
        matches = find_matches_3(sp) ## finds first round of matches.

        ## Step 3: Create mask and apply to matched column
        mask = create_mask(matches, sp, k) ## Creates a mask of students who have not been assigned to a school.
        sp.loc[:,'matched'] = mask

        ## Step 4: Update the underdemanded column
        condition = np.logical_or(sp['k'] > 1, sp['underdemanded'] == False)
        sp['underdemanded'] = np.where(np.logical_and(condition,sp['matched'] == False), False, sp['underdemanded'])
        sp['underdemanded'] = np.where(sp['matched'] == True, True, sp['underdemanded'])


        ## Step 5: Shift Values Over for unmatched
        sp = shift(sp, mask, k)

        ## Step 6: Update N, to reflect total number of matches for each school. 
        #sp['N'] += matches.set_index(0)['k'].reindex(sp[0], fill_value=0).reset_index(drop=True)

        ## Step 7: Remove all rejects who have applied to k schools:
        sp = sp[sp['applications'] < k] # If you've been rejected from three schools, you're out. We also don't want you to stay in the dataset. 
        sp.reset_index(drop=True, inplace=True) # Reset the index
        i+=1
    return sp, i

## EADAM Algorithm Functions

# Comparison Functions:

def comparison(sp_f, sp):
    '''
    This functions looks at matches produced across iterations of EADAM and determines whether 
    there has been any change in matches. If not, then we can stop the algorithm.
    ______
    sp_f: round n+1 dataframe.
    sp: round n matchings.
    '''
    X = pd.testing.assert_frame_equal(sp_f[[0, 'student_id']].reset_index(drop = True), sp[[0, 'student_id']].reset_index(drop=True), check_dtype=False)
    if X  is None:
        X = False
    else:
        X = True
    return X

def compare(sp_f, sp):
    '''
    This function takes two sets of matchings and compares them. This is used
    to determine whether we should "end" the EADAM algorithm or not.
    ______
    sp_f: round n+1 dataframe. 
    sp: prior round matchings.  
    '''
    X = True ## initialize boolean to say that they are the same
    if len(sp_f) != len(sp): ## if the number of matchings is different, then they are not the same
        X = True
    else:
        X = comparison(sp_f, sp) ## if the number of matchings is the same, then we need to compare them
    return X

# Removing Underdemanded Schools Function:

def remove_ud(sp_f, sp):
    '''
    A function that takes as input a final set of matchings and determines which schools have been underdemanded!
    The underdemanded schools are all the schools that have either (1) gone unmatched (as they were never demanded)
    or (2) have been demanded by only a single student. 
    ______
    sp_f: updated set of matchings
    sp: original set of student preferences
    '''
    ## All underdemanded schools underdemanded in the sense of never matched have
    # already deleted self. Now delete others.
    sp_f = sp_f[sp_f['underdemanded'] == False] ## delete all agents that wound up at underdemanded schools
    ## Now we want to back to the original school list and grab only those agents that wound up at demanded schools.
    sp_new = sp[sp['student_id'].isin(sp_f['student_id'])] ## Holds on only to those students that ended up matched to a demanded school!
    return sp_new

# EADAM Algorithm:
def EADAM(sp, k = 3):
    '''
    This function runs the EADAM algorithm.
    ______
    sp: dataframe of student preferences and school scores over those students. 
    '''
    change = True ## initialize boolean to say that there has been a change
    j = 0 ## initialize counter, counts number of rounds in EADAM (number of times we run GS)
    iter_list = [] ## initialize list to hold the number of iterations in each GS run.
    undermatched_matches = [] ## initialize list to hold the number of undermatched students in each round
    while change == True:
        sp_f, i = run_gale_shapley(sp, k)
        if j == 0:
            gs_result = sp_f
        undermatched_matches.append(sp_f[sp_f['underdemanded'] == True]) ## save the deleted undermatched results.
        sp_f = remove_ud(sp_f, sp)
        change = compare(sp_f, sp)
        sp = sp_f
        iter_list.append(i)
        j += 1
    EADAM_result = pd.concat([sp_f, *undermatched_matches], ignore_index=True) ## add the final results in too.
    return sp_f, gs_result, iter_list, j, EADAM_result

## Analysis Functions

def find_diff(gs_result, EADAM_result):
    '''
    This function finds rows that have changed between the GS result and the EADAM result.
    _______
    gs_result: dataframe containing Gale-Shapley produced matches. 
    EADAM_result: dataframe containing the matches produced by EADAM.
    '''
    comb = pd.concat([gs_result.loc[:, [0, 'student_id']], EADAM_result.loc[:,[0, 'student_id']]])
    comb.drop_duplicates(keep=False, inplace=True)
    return comb