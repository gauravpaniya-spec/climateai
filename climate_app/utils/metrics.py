"""utils/metrics.py — ClimateAI full metrics suite (Parts 1–4)"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, List

VARIABLE_NAMES = ["temp_2m","precip","wind_u10","wind_v10","pressure_msl","geopotential_500"]

# ── basic ─────────────────────────────────────────────────────────────────────
def rmse(pred,truth): return float(np.sqrt(np.mean((pred-truth)**2)))
def mae(pred,truth):  return float(np.mean(np.abs(pred-truth)))
def mbe(pred,truth):  return float(np.mean(pred-truth))
def r2_score(pred,truth):
    ss_res=np.sum((truth-pred)**2); ss_tot=np.sum((truth-truth.mean())**2)
    return 1.0 if ss_res==0 else float(1-ss_res/max(ss_tot,1e-12))

# ── lat weighting ─────────────────────────────────────────────────────────────
def lat_weights(lats):
    w=np.cos(np.deg2rad(lats)); w=w/w.mean(); return w[:,np.newaxis]

def lat_weighted_rmse(pred,truth,lats):
    w=lat_weights(lats); sq=(pred-truth)**2
    return float(np.sqrt(np.average(sq,weights=np.broadcast_to(w,sq.shape))))

def lat_weighted_mae(pred,truth,lats):
    w=lat_weights(lats); ab=np.abs(pred-truth)
    return float(np.average(ab,weights=np.broadcast_to(w,ab.shape)))

# ── anomaly correlation ───────────────────────────────────────────────────────
def anomaly_correlation(pred,truth,clim,lats=None):
    fa=pred-clim; ta=truth-clim
    if lats is not None:
        w=np.broadcast_to(lat_weights(lats),fa.shape)
        num=np.sum(w*fa*ta); den=np.sqrt(np.sum(w*fa**2)*np.sum(w*ta**2))
    else:
        num=np.sum(fa*ta); den=np.sqrt(np.sum(fa**2)*np.sum(ta**2))
    return 0.0 if den==0 else float(num/den)

# keep alias
acc = anomaly_correlation

# ── skill score ───────────────────────────────────────────────────────────────
def skill_score(pred,truth,reference,metric_fn=rmse):
    s_m=metric_fn(pred,truth); s_r=metric_fn(reference,truth)
    if s_r==0: return 1.0 if s_m==0 else float("-inf")
    return float(1-s_m/s_r)

# ── forecast confidence score ─────────────────────────────────────────────────
def forecast_confidence_score(rmse_val: float, rmse_clim: float,
                               acc_val: float) -> float:
    """
    Composite confidence ∈ [0,1]:
      0.5 × (1 − rmse/rmse_clim)  +  0.5 × ACC
    """
    skill = max(0.0, 1.0 - rmse_val / max(rmse_clim, 1e-8))
    return float(np.clip(0.5*skill + 0.5*acc_val, 0, 1))

# ── rain classification accuracy ──────────────────────────────────────────────
def rain_classification_accuracy(pred_rain: np.ndarray,
                                  truth_rain: np.ndarray,
                                  threshold: float = 0.1) -> float:
    """
    Binary rain/no-rain classification accuracy.
    threshold: mm/hr above which we call it 'rain'.
    """
    pred_cat  = (pred_rain  >= threshold).astype(int)
    truth_cat = (truth_rain >= threshold).astype(int)
    return float((pred_cat == truth_cat).mean())

# ── baselines ─────────────────────────────────────────────────────────────────
def persistence_forecast(X: np.ndarray, lead_steps: int = 1) -> np.ndarray:
    """Repeat t as forecast for t+lead."""
    return X[:-lead_steps]

def climatology_forecast(X: np.ndarray) -> np.ndarray:
    """Return temporal mean as constant forecast (same shape as X)."""
    clim = X.mean(axis=0, keepdims=True)
    return np.broadcast_to(clim, X.shape).copy()

# ── weather summary ───────────────────────────────────────────────────────────
def generate_weather_summary(pred: np.ndarray, lats: np.ndarray,
                              variable_names: Optional[List[str]] = None) -> str:
    """
    Generate a plain-English weather summary from a prediction array.
    pred: (T, Lat, Lon, Vars)
    """
    if variable_names is None: variable_names = VARIABLE_NAMES
    lines = ["🌍 Weather Summary"]
    for i,vname in enumerate(variable_names[:len(pred[0,0,0])]):
        arr = pred[...,i]
        w   = lat_weights(lats)
        mean_v = float(np.average(arr.mean(axis=0), weights=np.broadcast_to(w,arr.mean(axis=0).shape)))
        trend  = "↑ rising" if arr[-1].mean()>arr[0].mean() else "↓ falling"
        lines.append(f"  • {vname}: mean={mean_v:.2f}, trend={trend}")
    return "\n".join(lines)

# ── aggregate report ──────────────────────────────────────────────────────────
def compute_all_metrics(pred,truth,clim,lats,variable_names=None):
    if variable_names is None: variable_names=VARIABLE_NAMES
    n_vars=pred.shape[-1]; records=[]
    for v in range(n_vars):
        p=pred[...,v]; t=truth[...,v]; c=clim[...,v]
        records.append({
            "variable": variable_names[v] if v<len(variable_names) else f"var_{v}",
            "RMSE":     rmse(p,t), "MAE": mae(p,t), "MBE": mbe(p,t),
            "R2":       r2_score(p.ravel(),t.ravel()),
            "RMSE_latw":lat_weighted_rmse(p,t,lats),
            "MAE_latw": lat_weighted_mae(p,t,lats),
            "ACC":      anomaly_correlation(p,t,c,lats),
        })
    return pd.DataFrame(records).set_index("variable").round(5)
