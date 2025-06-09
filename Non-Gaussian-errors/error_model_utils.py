import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Callable, Dict, List, Tuple, Optional

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


# Type aliases
Array = np.ndarray
CloudImpact = Tuple[Array, Array, Array]


def derive_departures(
    obs: Array,
    synth: Array,
    mask: Array
) -> Array:
    """
    Compute departures (observed - synthetic) under a given mask.
    """
    return obs[mask] - synth[mask]


def compute_summary_stats(
    departures: Array
) -> Tuple[float, float]:
    """
    Compute mean and standard deviation of a departure array.
    """
    return float(departures.mean()), float(departures.std())


def compute_cloud_impacts(
    threshold: float,
    synth: Array,
    obs: Array,
    mask: Array,
    cloud_impact_fn: Callable[[float, Array, Array], CloudImpact]
) -> Tuple[Array, Array, Array]:
    """
    Apply cloud impact function to masked data.
    """
    return cloud_impact_fn(threshold, synth[mask], obs[mask])


def build_error_dataframe(
    ref: Dict[str, Array],
    operators: Dict[str, str],
    flatten_fn: Callable[[Array], Array],
    cloud_impact_fn: Callable[[float, Array, Array], CloudImpact],
    obs_key: str = 'O',
    sza_key: str = 'SZA',
    obs_min: float = 0.0,
    sza_max: float = 70.0,
    threshold: float = 0.0
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Create a DataFrame of departures and cloud-impact metrics for each operator,
    plus summary statistics (mean, std) per operator.

    Returns
    -------
    df : pd.DataFrame
        Contains columns V_<label>, C_o, C_b_<label>, C_a_<label>.
    stats : dict
        Mapping from operator label to (mean, std) of departures.
    """
    # Flatten inputs
    obs = flatten_fn(ref[obs_key])
    sza = flatten_fn(ref[sza_key])

    # Quality mask
    mask = (obs > obs_min) & (sza < sza_max)

    data: Dict[str, Array] = {}
    stats: Dict[str, Tuple[float, float]] = {}
    C_o: Optional[Array] = None

    # Process each operator
    for label, key in operators.items():
        synth = flatten_fn(ref[key])
        dep = derive_departures(obs, synth, mask)
        data[f'V_{label}'] = dep
        mean, std = compute_summary_stats(dep)
        stats[label] = (mean, std)

        C_a, C_b, co = compute_cloud_impacts(threshold, synth, obs, mask, cloud_impact_fn)
        data[f'C_a_{label}'] = C_a
        data[f'C_b_{label}'] = C_b
        C_o = co

    # Add common clear-sky term
    data['C_o'] = C_o

    return pd.DataFrame(data), stats


def compute_binned_stats(
    df: pd.DataFrame,
    predictor: str,
    target: str,
    n_bins: int = 25
) -> pd.DataFrame:
    """
    Bin a predictor column into n_bins and compute mean & std of a target column per bin.
    Returns a DataFrame with columns [predictor+'_mid', 'mean', 'std'].
    """
    bins = np.linspace(df[predictor].min(), df[predictor].max(), n_bins)
    grouped = df.groupby(pd.cut(df[predictor], bins=bins))[target]
    mids = [(i.left + i.right) / 2 for i in grouped.mean().index]
    return pd.DataFrame({
        predictor + '_mid': mids,
        'mean': grouped.mean().values,
        'std': grouped.std().values
    })


def plot_binned_comparison(
    binned: pd.DataFrame,
    predictor_mid: str,
    mean_col: str,
    std_col: str,
    label: str,
    ax: plt.Axes
):
    """
    Plot mean and std versus predictor_mid on given Axes.
    """
    ax.plot(binned[predictor_mid], binned[mean_col], marker='.', label=f'{label} mean')
    ax.plot(binned[predictor_mid], binned[std_col], marker='.', linestyle='--', label=f'{label} std')
    ax.legend()


def retrieve_dynamic_error(
    df: pd.DataFrame,
    binned_std: pd.DataFrame,
    predictor: str,
    std_col: str,
    zero_Ca: float
) -> Array:
    """
    Map each Ca in df to the corresponding std via binned lookup
    (e.g. using interpolation)."""
    from numpy import interp

    Ca = df[predictor]
    return interp(Ca, binned_std[predictor + '_mid'], binned_std[std_col], left=zero_Ca)


def normalize_departures(
    df: pd.DataFrame,
    stats: Dict[str, Tuple[float, float]],
    dynamic_cols: Dict[str, str]
) -> pd.DataFrame:
    """
    Add columns 'norm-<label>' and replace V_<label> with z-scores.
    dynamic_cols maps label to dynamic std column in df.
    """
    for label, (mean, std) in stats.items():
        dyn_std = retrieve_dynamic_error(
            df,
            df[[f'C_a_{label}_mid', 'std']],  # pass appropriate binned DataFrame
            f'C_a_{label}',
            'std',
            zero_Ca=0.0
        )
        df[f'norm-{label}'] = (df[f'V_{label}'] - mean) / dyn_std
        df[f'V_{label}'] = (df[f'V_{label}'] - mean) / std
    return df


# Example pipeline usage:
# operators = {'rttov12': 'B-rttov12', 'rttov13': 'B-rttov13'}
# df, stats = build_error_dataframe(ref, operators, flatten_array, cloud_impact_av)
# binned_Co = compute_binned_stats(df, 'C_o', 'V_rttov12')
# plot_binned_comparison(binned_Co, 'C_o_mid', 'mean', 'std', 'R_o', ax)
# ... similarly for other predictors
# df = normalize_departures(df, stats, dynamic_cols)
