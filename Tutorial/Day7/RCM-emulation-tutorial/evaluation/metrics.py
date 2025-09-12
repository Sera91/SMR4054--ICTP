import xarray as xr
import numpy as np
from numpy import fft

def _filter_by_season(data : xr.Dataset, season : str) -> xr.Dataset:

    if season is None:
        pass
    elif season == 'winter':
        data = data.where(data['time.season'] == 'DJF', drop=True)
    elif season == 'summer':
        data = data.where(data['time.season'] == 'JJA', drop=True)
    elif season == 'spring':
        data = data.where(data['time.season'] == 'MAM', drop=True)
    elif season == 'autumn':
        data = data.where(data['time.season'] == 'SON', drop=True)

    return data

def rmse(target: xr.Dataset, pred: xr.Dataset,
         var_target: str, season: str=None) -> xr.Dataset:
    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    metric = ((pred - target) ** 2).mean('time') ** (1/2)
    return metric

def bias_mean(target: xr.Dataset, pred: xr.Dataset,
              var_target: str, season: str=None) -> xr.Dataset:
    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    metric = (pred.mean('time') - target.mean('time'))
    return metric

def bias_quantile(target: xr.Dataset, pred: xr.Dataset, quantile: float,
                  var_target: str, season: str=None) -> xr.Dataset:
    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    metric = (pred.quantile(quantile, 'time') - target.quantile(quantile, 'time'))
    return metric

def ratio_std(target: xr.Dataset, pred: xr.Dataset,
              var_target: str, season: str=None) -> xr.Dataset:
    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    target_std = target.std('time')
    pred_std = pred.std('time')

    metric = pred_std / target_std
    return metric

def _radial_average(power_2d):
    y, x = np.indices(power_2d.shape)
    center = np.array([(x.max() - x.min())/2.0, (y.max() - y.min())/2.0])
    r = np.hypot(x - center[0], y - center[1])
    r = r.astype(np.int32)

    tbin = np.bincount(r.ravel(), power_2d.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / np.maximum(nr, 1)  # avoid division by zero
    return radialprofile

def psd(target: xr.Dataset, pred: xr.Dataset, var_target: str, season: str=None):
    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)

    target_da = target[var_target]
    pred_da = pred[var_target]

    # Convert to numpy arrays and handle NaNs
    target_np = np.nan_to_num(target_da.values, nan=0.0)
    pred_np = np.nan_to_num(pred_da.values, nan=0.0)

    # Compute 2D FFT along spatial axes 
    fft_target = fft.fftshift(fft.fft2(target_np, axes=(-2, -1)), axes=(-2, -1))
    fft_pred = fft.fftshift(fft.fft2(pred_np, axes=(-2, -1)), axes=(-2, -1))

    # Compute power spectrum
    power_target = np.abs(fft_target) ** 2
    power_pred = np.abs(fft_pred) ** 2

    # Radial averaging over each time slice
    psd_target_list = [_radial_average(p) for p in power_target]
    psd_pred_list = [_radial_average(p) for p in power_pred]

    # Average over time dimension
    avg_psd_target = np.mean(psd_target_list, axis=0)
    avg_psd_pred = np.mean(psd_pred_list, axis=0)

    # Return as xarray.DataArrays
    psd_target_da = xr.DataArray(avg_psd_target, dims=["wavenumber"], name="PSD_target")
    psd_pred_da = xr.DataArray(avg_psd_pred, dims=["wavenumber"], name="PSD_pred")

    return psd_target_da, psd_pred_da