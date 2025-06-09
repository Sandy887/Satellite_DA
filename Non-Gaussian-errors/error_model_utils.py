import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Callable, Dict, List, Tuple, Optional


def flatten_array(A: np.ndarray) -> np.ndarray:
    tmp_out = A.flatten()
    return tmp_out[~np.isnan(tmp_out)]


def cloud_impact_av(
    cloudiness_threshold: float,
    B: np.ndarray,
    O: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the average cloud impact (Ca), synthetic impact (Cx), and observed impact (Cy).
    """
    Cy = np.where(O - cloudiness_threshold > 0, O - cloudiness_threshold, 0)
    Cx = np.where(B - cloudiness_threshold > 0, B - cloudiness_threshold, 0)
    Ca = (Cy + Cx) / 2
    return Ca, Cx, Cy


def retrieve_stds_from_binned_Ca(
    X: np.ndarray,
    df: pd.DataFrame,
    namelist: List[str],
    zero_Ca: float
) -> np.ndarray:
    """
    Efficiently map Ca values in X to corresponding standard deviations in df using pd.IntervalIndex.
    """
    interval_index = pd.IntervalIndex(df[namelist[0]])
    bin_indices = interval_index.get_indexer(X)
    Y = np.where(bin_indices >= 0, df[namelist[1]].values[bin_indices], zero_Ca)
    return Y.astype(np.float64)


# Type aliases
Array = np.ndarray
CloudImpact = Tuple[Array, Array, Array]


def derive_departures(
    obs: Array,
    synth: Array,
    mask: Array
) -> Array:
    return obs[mask] - synth[mask]


def compute_summary_stats(
    departures: Array
) -> Tuple[float, float]:
    return float(departures.mean()), float(departures.std())


def compute_cloud_impacts(
    threshold: float,
    synth: Array,
    obs: Array,
    mask: Array,
    cloud_impact_fn: Callable[[float, Array, Array], CloudImpact]
) -> CloudImpact:
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
    obs = flatten_fn(ref[obs_key])
    sza = flatten_fn(ref[sza_key])
    mask = (obs > obs_min) & (sza < sza_max)

    data: Dict[str, Array] = {}
    stats: Dict[str, Tuple[float, float]] = {}
    Cy: Optional[Array] = None

    for label, key in operators.items():
        synth = flatten_fn(ref[key])
        dep = derive_departures(obs, synth, mask)
        data[f'OmB_{label}'] = dep
        mean, std = compute_summary_stats(dep)
        stats[label] = (mean, std)

        C_av, Cx, cy = compute_cloud_impacts(threshold, synth, obs, mask, cloud_impact_fn)
        data[f'C_av_{label}'] = C_av
        data[f'Cx_{label}'] = Cx
        Cy = cy

    data['Cy'] = Cy
    df = pd.DataFrame(data)
    return df, stats


def compute_binned_stats(
    df: pd.DataFrame,
    predictor: str,
    target: str,
    n_bins: int = 25
) -> pd.DataFrame:
    bins = np.linspace(df[predictor].min(), df[predictor].max(), n_bins)
    grouped = df.groupby(pd.cut(df[predictor], bins=bins))[target]
    mids = [(interval.left + interval.right) / 2 for interval in grouped.mean().index]
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
    ax.plot(binned[predictor_mid], binned[mean_col], marker='.', label=f'{label} mean')
    ax.plot(binned[predictor_mid], binned[std_col], marker='.', linestyle='--', label=f'{label} std')
    ax.legend()


def retrieve_dynamic_error(
    df: pd.DataFrame,
    binned_std: pd.DataFrame,
    predictor: str,
    predictor_mid: str,
    std_col: str,
    zero_Ca: float
) -> Array:
    from numpy import interp
    Ca = df[predictor]
    x = binned_std[predictor_mid]
    y = binned_std[std_col]
    return interp(Ca, x, y, left=zero_Ca)


def normalize_departures(
    df: pd.DataFrame,
    stats: Dict[str, Tuple[float, float]],
    dynamic_bins: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Normalize departures both globally and dynamically by predictor bins.

    dynamic_bins maps operator label -> binned DataFrame with columns
    ['C_av_<label>_mid', 'mean', 'std'].
    """
    for label, (mean, std) in stats.items():
        binned = dynamic_bins[label]
        predictor = f'C_av_{label}'
        predictor_mid = predictor + '_mid'
        dyn_std = retrieve_dynamic_error(
            df,
            binned,
            predictor,
            predictor_mid,
            'std',
            zero_Ca=0.0
        )

        df[f'norm-{label}'] = (df[f'OmB_{label}'] - mean) / dyn_std
        df[f'OmB_{label}']   = (df[f'OmB_{label}'] - mean) / std

    return df
