"""
data/data_engine.py
-------------------
ClimateAI — Data pipeline for climate / NWP data.

Implements:
  - ClimateDataEngine.generate_synthetic_era5()  → xr.Dataset
  - ClimateDataEngine.preprocess()               → train/val/test splits + stats
  - ClimateDataEngine.get_tensor_dataset()       → torch Dataset

The synthetic ERA5-like data includes:
  • Seasonal + diurnal sin/cos cycles
  • Latitude-dependent temperature gradient
  • Realistic band-limited Gaussian noise
  • Six atmospheric variables at configurable spatial resolution
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import xarray as xr


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARIABLES = [
    "temp_2m",          # 2 m temperature            [K]
    "precip",           # precipitation rate          [mm/hr]
    "wind_u10",         # 10 m zonal wind             [m/s]
    "wind_v10",         # 10 m meridional wind        [m/s]
    "pressure_msl",     # mean sea-level pressure     [Pa]
    "geopotential_500", # 500 hPa geopotential height [m²/s²]
]

LEVELS = [500, 700, 850, 925, 1000]   # pressure levels [hPa]

# Default synthetic grid
DEFAULT_LAT_SIZE = 32    # ~5.6° resolution
DEFAULT_LON_SIZE = 64    # ~5.6° resolution
HOURS_PER_STEP   = 6     # 6-hourly data (ERA5-like)
STEPS_PER_DAY    = 24 // HOURS_PER_STEP  # 4

# Positional encoding channels: sin/cos × (lat, lon, doy, hod) = 8
N_POS_ENC = 8


# ---------------------------------------------------------------------------
# Helper: band-limited noise via FFT smoothing
# ---------------------------------------------------------------------------

def _smooth_noise(shape: tuple, sigma: float = 3.0, rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate spatially correlated Gaussian noise via Gaussian smoothing in
    the Fourier domain (fast and memory-friendly).

    Parameters
    ----------
    shape : (H, W) spatial dimensions
    sigma : smoothing scale in grid cells
    rng   : numpy random Generator (for reproducibility)
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.standard_normal(shape)
    H, W  = shape
    fy    = np.fft.fftfreq(H)
    fx    = np.fft.fftfreq(W)
    FX, FY = np.meshgrid(fx, fy)
    kernel  = np.exp(-2 * math.pi ** 2 * sigma ** 2 * (FX ** 2 + FY ** 2))
    smoothed = np.real(np.fft.ifft2(np.fft.fft2(noise) * kernel))
    # re-standardise
    smoothed -= smoothed.mean()
    std = smoothed.std()
    if std > 0:
        smoothed /= std
    return smoothed


# ---------------------------------------------------------------------------
# Main engine class
# ---------------------------------------------------------------------------

class ClimateDataEngine:
    """
    End-to-end data pipeline for ClimateAI experiments.

    Parameters
    ----------
    lat_size  : number of latitude grid points
    lon_size  : number of longitude grid points
    seed      : random seed for reproducibility
    """

    def __init__(
        self,
        lat_size: int  = DEFAULT_LAT_SIZE,
        lon_size: int  = DEFAULT_LON_SIZE,
        seed: int      = 42,
    ):
        self.lat_size  = lat_size
        self.lon_size  = lon_size
        self.seed      = seed
        self._rng      = np.random.default_rng(seed)

        # Grid coordinates
        self.lats = np.linspace(-90,  90,  lat_size)   # (L,)
        self.lons = np.linspace(-180, 180, lon_size)    # (Lo,)

    # ------------------------------------------------------------------
    # 1. Synthetic ERA5 generation
    # ------------------------------------------------------------------

    def generate_synthetic_era5(self, years: int = 5) -> xr.Dataset:
        """
        Generate a realistic synthetic ERA5-like dataset.

        Physics embedded in the synthetic signal
        -----------------------------------------
        temp_2m
          Base profile: 300 K at equator → 230 K at poles (cos-lat gradient)
          Seasonal cycle: ±15 K amplitude, sin(2π·doy/365)
          Diurnal cycle:  ±5 K amplitude,  sin(2π·hod/24 − π/4)
          Spatial structure: smoothed noise (σ=4)

        precip
          Log-normal marginal; enhanced in tropics (cos-lat²); seasonal
          amplitude; always non-negative.

        wind_u10 / wind_v10
          Zero-mean; seasonal amplitude (±8 m/s); spatial structure.

        pressure_msl
          101 325 Pa base; ±1 200 Pa seasonal; ±800 Pa diurnal; noise.

        geopotential_500
          ~57 000 m²/s² base; ±3 000 m²/s² seasonal; ±1 000 m²/s²
          lat gradient; noise.

        Parameters
        ----------
        years : number of years to generate (6-hourly steps)

        Returns
        -------
        xr.Dataset with coords (time, lat, lon) and one DataArray per variable.
        """
        n_steps = years * 365 * STEPS_PER_DAY
        H, W    = self.lat_size, self.lon_size
        n_vars  = len(VARIABLES)

        # ---- Build time index ------------------------------------------------
        start = pd.Timestamp("2000-01-01 00:00")
        times = pd.date_range(start, periods=n_steps, freq=f"{HOURS_PER_STEP}h")

        doy = np.array(times.day_of_year, dtype=float)   # 1..365
        hod = np.array(times.hour,        dtype=float)   # 0,6,12,18

        # ---- Spatial grid factors -------------------------------------------
        lat_rad  = np.deg2rad(self.lats)                 # (L,)
        cos_lat  = np.cos(lat_rad)                       # (L,)
        cos2_lat = cos_lat ** 2                          # tropical weight

        # Broadcast helpers: (T,1,1) × (1,L,1) × (1,1,W)
        def _T(arr): return arr[:, np.newaxis, np.newaxis]      # time axis
        def _L(arr): return arr[np.newaxis, :, np.newaxis]      # lat  axis

        # Pre-allocate output array  (T, L, W, Vars)
        data = np.zeros((n_steps, H, W, n_vars), dtype=np.float32)

        # ------------------------------------------------------------------ #
        # Variable 0 — temp_2m  [K]
        # ------------------------------------------------------------------ #
        base_temp   = 300.0 + _L(40.0 * (cos_lat - 1))      # equator=300, poles≈260
        seasonal_T  = 15.0 * np.sin(2 * math.pi * _T(doy) / 365.0)
        diurnal_T   = 5.0  * np.sin(2 * math.pi * _T(hod) / 24.0 - math.pi / 4)
        noise_T     = np.stack([
            _smooth_noise((H, W), sigma=4.0, rng=self._rng) for _ in range(n_steps)
        ]) * 3.0   # 3 K std noise
        data[..., 0] = (base_temp + seasonal_T + diurnal_T + noise_T).astype(np.float32)

        # ------------------------------------------------------------------ #
        # Variable 1 — precip  [mm/hr]  → log-normal, non-negative
        # ------------------------------------------------------------------ #
        base_prec   = 0.5 * _L(cos2_lat)
        seasonal_P  = 0.3 * np.sin(2 * math.pi * _T(doy) / 365.0) * _L(cos2_lat)
        noise_P     = np.stack([
            _smooth_noise((H, W), sigma=5.0, rng=self._rng) for _ in range(n_steps)
        ]) * 0.4
        raw_P = base_prec + seasonal_P + noise_P
        data[..., 1] = np.maximum(0.0, raw_P).astype(np.float32)

        # ------------------------------------------------------------------ #
        # Variable 2 — wind_u10  [m/s]
        # ------------------------------------------------------------------ #
        seasonal_U  = 8.0 * np.sin(2 * math.pi * _T(doy) / 365.0)
        noise_U     = np.stack([
            _smooth_noise((H, W), sigma=6.0, rng=self._rng) for _ in range(n_steps)
        ]) * 5.0
        data[..., 2] = (seasonal_U + noise_U).astype(np.float32)

        # ------------------------------------------------------------------ #
        # Variable 3 — wind_v10  [m/s]
        # ------------------------------------------------------------------ #
        seasonal_V  = 4.0 * np.cos(2 * math.pi * _T(doy) / 365.0)
        noise_V     = np.stack([
            _smooth_noise((H, W), sigma=6.0, rng=self._rng) for _ in range(n_steps)
        ]) * 5.0
        data[..., 3] = (seasonal_V + noise_V).astype(np.float32)

        # ------------------------------------------------------------------ #
        # Variable 4 — pressure_msl  [Pa]
        # ------------------------------------------------------------------ #
        base_pres   = 101_325.0
        seasonal_PS = 1_200.0 * np.sin(2 * math.pi * _T(doy) / 365.0)
        diurnal_PS  =   800.0 * np.sin(2 * math.pi * _T(hod) / 24.0)
        noise_PS    = np.stack([
            _smooth_noise((H, W), sigma=4.0, rng=self._rng) for _ in range(n_steps)
        ]) * 500.0
        data[..., 4] = (base_pres + seasonal_PS + diurnal_PS + noise_PS).astype(np.float32)

        # ------------------------------------------------------------------ #
        # Variable 5 — geopotential_500  [m²/s²]
        # ------------------------------------------------------------------ #
        base_z500   = 57_000.0 + _L(1_000.0 * (cos_lat - 0.5))
        seasonal_Z  = 3_000.0 * np.sin(2 * math.pi * _T(doy) / 365.0)
        noise_Z     = np.stack([
            _smooth_noise((H, W), sigma=5.0, rng=self._rng) for _ in range(n_steps)
        ]) * 800.0
        data[..., 5] = (base_z500 + seasonal_Z + noise_Z).astype(np.float32)

        # ------------------------------------------------------------------ #
        # Wrap as xarray Dataset
        # ------------------------------------------------------------------ #
        data_vars = {}
        for i, vname in enumerate(VARIABLES):
            data_vars[vname] = xr.DataArray(
                data[..., i],
                dims=["time", "lat", "lon"],
                attrs={"long_name": vname, "source": "ClimateAI synthetic ERA5"},
            )

        ds = xr.Dataset(
            data_vars,
            coords={
                "time": times,
                "lat":  self.lats,
                "lon":  self.lons,
            },
            attrs={
                "title":       "ClimateAI Synthetic ERA5",
                "years":       years,
                "variables":   VARIABLES,
                "levels":      LEVELS,
                "lat_size":    H,
                "lon_size":    W,
                "time_step_h": HOURS_PER_STEP,
                "seed":        self.seed,
            },
        )
        return ds

    # ------------------------------------------------------------------
    # 2. Preprocessing
    # ------------------------------------------------------------------

    def preprocess(
        self,
        ds: xr.Dataset,
        train_frac: float = 0.70,
        val_frac:   float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray,
               np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline:

        1. Stack variables into array (T, H, W, Vars)
        2. Z-score normalise using **train-split statistics only**
        3. Compute and append positional encodings (8 channels):
              sin(2π·lat/180), cos(2π·lat/180),
              sin(2π·lon/360), cos(2π·lon/360),
              sin(2π·doy/365), cos(2π·doy/365),
              sin(2π·hod/24),  cos(2π·hod/24)
        4. Split chronologically  70 / 15 / 15  → train / val / test

        Parameters
        ----------
        ds         : xr.Dataset from generate_synthetic_era5()
        train_frac : fraction of time steps for training
        val_frac   : fraction for validation (rest → test)

        Returns
        -------
        X_train, X_val, X_test : np.ndarray (T_split, H, W, Vars+8)
        mean, std              : np.ndarray (Vars,) — train-set statistics
        lats, lons             : np.ndarray 1-D coordinate arrays
        """
        T = len(ds.time)
        H = len(ds.lat)
        W = len(ds.lon)
        n_vars = len(VARIABLES)

        # ---- Stack variables ------------------------------------------------
        raw = np.stack([ds[v].values for v in VARIABLES], axis=-1).astype(np.float32)
        # shape: (T, H, W, n_vars)

        # ---- Chronological split indices -----------------------------------
        t_train_end = int(T * train_frac)
        t_val_end   = int(T * (train_frac + val_frac))

        # ---- Z-score from training set only ---------------------------------
        train_raw = raw[:t_train_end]
        mean = train_raw.reshape(-1, n_vars).mean(axis=0)   # (Vars,)
        std  = train_raw.reshape(-1, n_vars).std(axis=0)    # (Vars,)
        std  = np.where(std == 0, 1.0, std)                 # guard / 0

        norm = (raw - mean[np.newaxis, np.newaxis, np.newaxis, :]) / \
               std [np.newaxis, np.newaxis, np.newaxis, :]
        # shape: (T, H, W, n_vars)

        # ---- Positional encodings ------------------------------------------
        times  = pd.DatetimeIndex(ds.time.values)
        doy    = np.array(times.day_of_year, dtype=np.float32)   # (T,)
        hod    = np.array(times.hour,        dtype=np.float32)   # (T,)
        lats_r = np.deg2rad(ds.lat.values).astype(np.float32)    # (H,)
        lons_r = np.deg2rad(ds.lon.values).astype(np.float32)    # (W,)

        # Spatial encodings — broadcast to (T, H, W)
        sin_lat = np.broadcast_to(np.sin(lats_r)[np.newaxis, :, np.newaxis], (T, H, W))
        cos_lat = np.broadcast_to(np.cos(lats_r)[np.newaxis, :, np.newaxis], (T, H, W))
        sin_lon = np.broadcast_to(np.sin(lons_r)[np.newaxis, np.newaxis, :], (T, H, W))
        cos_lon = np.broadcast_to(np.cos(lons_r)[np.newaxis, np.newaxis, :], (T, H, W))

        # Temporal encodings — broadcast to (T, H, W)
        sin_doy = np.broadcast_to(
            np.sin(2 * math.pi * doy / 365.0)[:, np.newaxis, np.newaxis], (T, H, W))
        cos_doy = np.broadcast_to(
            np.cos(2 * math.pi * doy / 365.0)[:, np.newaxis, np.newaxis], (T, H, W))
        sin_hod = np.broadcast_to(
            np.sin(2 * math.pi * hod / 24.0)[:, np.newaxis, np.newaxis],  (T, H, W))
        cos_hod = np.broadcast_to(
            np.cos(2 * math.pi * hod / 24.0)[:, np.newaxis, np.newaxis],  (T, H, W))

        # Stack encodings as extra channels → (T, H, W, 8)
        pos_enc = np.stack(
            [sin_lat, cos_lat, sin_lon, cos_lon,
             sin_doy, cos_doy, sin_hod, cos_hod],
            axis=-1,
        ).astype(np.float32)

        # Concatenate with normalised variables  → (T, H, W, n_vars + 8)
        X = np.concatenate([norm, pos_enc], axis=-1)

        # ---- Final splits ---------------------------------------------------
        X_train = X[:t_train_end]
        X_val   = X[t_train_end:t_val_end]
        X_test  = X[t_val_end:]

        return (
            X_train, X_val, X_test,
            mean, std,
            ds.lat.values.astype(np.float32),
            ds.lon.values.astype(np.float32),
        )

    # ------------------------------------------------------------------
    # 3. PyTorch Dataset
    # ------------------------------------------------------------------

    def get_tensor_dataset(
        self,
        X: np.ndarray,
        lead_steps: int = 1,
    ) -> "ClimateSequenceDataset":
        """
        Wrap a preprocessed array as a torch Dataset.

        Parameters
        ----------
        X          : (T, H, W, C)  preprocessed array (from preprocess())
        lead_steps : forecast lead time in steps (1 → next 6-hr step)

        Returns
        -------
        ClimateSequenceDataset yielding:
            input  : torch.Tensor (C, H, W)  — channel-first for Conv2d
            target : torch.Tensor (C, H, W)  — t + lead_steps
        """
        return ClimateSequenceDataset(X, lead_steps=lead_steps)


# ---------------------------------------------------------------------------
# PyTorch Dataset implementation
# ---------------------------------------------------------------------------

class ClimateSequenceDataset(Dataset):
    """
    Torch Dataset for autoregressive weather forecasting.

    Each item is:
        input  : X[t]           → float32 tensor (C, H, W)
        target : X[t+lead]      → float32 tensor (C, H, W)

    where C = n_vars + n_pos_enc, H = lat_size, W = lon_size.
    """

    def __init__(self, X: np.ndarray, lead_steps: int = 1):
        """
        Parameters
        ----------
        X          : (T, H, W, C)  numpy array
        lead_steps : steps into the future for the target
        """
        if lead_steps < 1:
            raise ValueError(f"lead_steps must be ≥ 1, got {lead_steps}")
        if len(X) <= lead_steps:
            raise ValueError(
                f"Array length {len(X)} ≤ lead_steps {lead_steps}; "
                "cannot form any (input, target) pairs."
            )
        # Store as float32 and convert to channel-first (T, C, H, W)
        self._X = torch.from_numpy(
            X.transpose(0, 3, 1, 2).astype(np.float32)  # (T, C, H, W)
        )
        self.lead_steps = lead_steps
        self.n_samples  = len(X) - lead_steps

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._X[idx], self._X[idx + self.lead_steps]

    @property
    def input_channels(self) -> int:
        return self._X.shape[1]   # C

    @property
    def spatial_shape(self) -> Tuple[int, int]:
        return int(self._X.shape[2]), int(self._X.shape[3])  # H, W

    def __repr__(self) -> str:
        H, W = self.spatial_shape
        return (
            f"ClimateSequenceDataset("
            f"n={self.n_samples}, lead={self.lead_steps}, "
            f"C={self.input_channels}, H={H}, W={W})"
        )
