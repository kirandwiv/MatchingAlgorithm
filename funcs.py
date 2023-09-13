import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parallel_pandas import ParallelPandas

ParallelPandas.initialize(n_cpu=16, split_factor=4, disable_pr_bar=True)

## Functions for generating data
def mdf_np(n, k = 3):
    x = pd.DataFrame(np.random.randint(0, n, (int(n*1.2),k)))
    x = x[~(x.apply(lambda row: len(row) != len(set(row)), axis=1))]
    if len(x)<n:
        return "Oops"
    else:
        x = x.iloc[:n].reset_index(drop=True)
        x['student_id'] = x.index
        x['applications'] = 0
    return x

def mdf_yp(n, k = 3):
    x = pd.DataFrame(np.random.randint(0, n, (int(n*1.2),k)))
    x = x[~(x.p_apply(lambda row: len(row) != len(set(row)), axis=1))]
    if len(x)<n:
        return "Oops"
    else:
        x = x.iloc[:n].reset_index(drop=True)
        x['student_id'] = x.index
        x['applications'] = 0
    return x

## Gale-Shapley Algorithm Functions

def find_matches(sp):
    '''
    This function takes a dataframe of student preferences and returns a dataframe of matches.
    '''
    sp2 = sp[[0, 'student_id']] ## take the two relevant columns, chosen schools and student IDs
    matches = sp2.sample(frac=1).drop_duplicates(subset=0) ## Will have to reflect the Bayesian nature of this. 
    ## Save the number of applications for each school and then weigh students probabilities by the number of applications. 
    return matches ## Returns our first round of matches. 

def create_mask(sp3, sp):
    '''
    This function takes a dataframe of matches and a dataframe of student preferences and returns a mask of students who have not been assigned to a school.
    '''
    mask = ~(np.logical_or(sp['student_id'].isin(sp3['student_id']), sp['applications'] == 3)) ## creates a mask of all students who have not been assigned to a school. 
    return mask ## The mask gives all values that should be shifted over.

def shift(sp, mask, k = 3):
    '''
    This function takes a dataframe of student preferences, a mask of students who have not been assigned to a school, and the number of schools to apply to. 
    It returns an updated dataframe of student preferences. In this dataframe we have shifted over the values for students who have not been assigned to a school.
    We also update the number of failed applications each student has made.
    '''
    for i in np.arange(k-1):
        sp.loc[mask, i] = sp.loc[mask, i+1] # Shifts over all the values for relevant columns. 
    sp.loc[mask, 'applications'] += 1 ## Updates to new values.
    return sp ## Returns the updated dataframe.

def run_gale_shapley(sp):
    '''Runs the Gale-Shapley Algorithm for a particular dataframe'''
    r_apps = 0
    i=0
    mask = [1]
    while r_apps < 3 and sum(mask) > 0: ## note that there are two ways to end the cycle. 1 all unmatched students have applied to 3 schools. 2. There are no unmatched students.
        matches = find_matches(sp) ## Finds the first round of matches.
        mask = create_mask(matches, sp) ## Creates a mask of students who have not been assigned to a school.
        r_apps = sp.loc[mask, 'applications'].min() < 3 ## Check, if all rejects have already applied to three schools, we terminate. 
        sp = shift(sp, mask) ## Shifts over the values for students who have not been assigned to a school.
        i+=1
    return sp, i