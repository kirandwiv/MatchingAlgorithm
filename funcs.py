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