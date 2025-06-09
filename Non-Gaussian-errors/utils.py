import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm


def flatten_array(A):
    tmp_out = A.flatten()
    return tmp_out[~np.isnan(tmp_out)]

def cloud_impact_av(cloudiness_threshold, B, O):
    
    Cy = np.where(O - cloudiness_threshold > 0, O - cloudiness_threshold, 0)
    Cx = np.where(B - cloudiness_threshold > 0, B - cloudiness_threshold, 0)
    Ca = (Cy + Cx) / 2
    return Ca, Cx, Cy

def retrieve_stds_from_binned_Ca(X: np.array, df: pd.DataFrame, namelist: list, zero_Ca:np.nan):
    """
    Efficiently maps Ca values in X to corresponding standard deviations in df using pd.IntervalIndex.
    
    Parameters:
    X : np.array : 1D array of Ca values
    df : pd.DataFrame : DataFrame containing bin intervals and corresponding standard deviation values
    namelist : list : List containing two column names, 
                      namelist[0] = column with bins
                      namelist[1] = column with standard deviation values

    Returns:
    np.array : Array of standard deviation values corresponding to X
    """
    
    # Create an IntervalIndex for fast lookups
    interval_index = pd.IntervalIndex(df[namelist[0]])

    # Use pandas searchsorted-like approach to find the bin index for each X value
    bin_indices = interval_index.get_indexer(X)

    # Use bin_indices to map to the std values, setting out-of-bin values to NaN
    Y = np.where(bin_indices >= 0, df[namelist[1]].values[bin_indices], zero_Ca)

    return np.array(Y, dtype=np.float64)