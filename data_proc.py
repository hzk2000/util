import h5py
import numpy as np
from numpy.fft import rfft
from joblib import Parallel, delayed
from scipy.signal import fftconvolve
from lmfit.models import LorentzianModel, ExponentialModel, LinearModel, PolynomialModel, Model

# --- 1. Models (High Performance) ---
# Built-in models are faster and more stable than ExpressionModel
lorentz_mod = LorentzianModel() + LinearModel(prefix='cst_')
exp_mod = ExponentialModel() + LinearModel(prefix='cst_')
parabola_mod = PolynomialModel(degree=2)

def ramsey_func(x, A, T, Df, phi, cst):
    return cst + A * np.exp(-x/T) * np.cos(2 * np.pi * Df * x + phi)
ramsey_mod = Model(ramsey_func)

def sine_func(x, A, x0, cst):
    return cst - A * np.cos(np.pi * x / x0)
sine_mod = Model(sine_func)


# --- 2. IO Utilities ---
def save_parameters(filepath, parameters):
    with h5py.File(filepath, 'a', libver='latest') as f:
        grp = f.require_group('parameters')
        for key, value in parameters.items():
            # Store metadata as HDF5 attributes for performance
            target = grp.require_group(key) if isinstance(value, dict) else grp
            target.attrs.update(value if isinstance(value, dict) else {key: value})

def load_parameters(filepath):
    with h5py.File(filepath, 'r', libver='latest', swmr=True) as f:
        grp = f['parameters']
        params = dict(grp.attrs)
        for key in grp.keys():
            params[key] = dict(grp[key].attrs)
        return params

def get_data(params):
    # SWMR mode prevents read/write conflicts
    with h5py.File(params["tmp_file_path"], "r", libver='latest', swmr=True) as f:
        return f["time"][()], f["demod2"][()]


# --- 3. Core Processing Engine ---
def calu_data(files, params):
    sr = params.get('sample_rate', 4.8e9)
    freq = params['demod_freq']
    st = params.get('st', 0)
    w_type = params['window_type']
    
    # --- Pre-computation ---
    # 1. Define read slice and window length
    if params['demod_type'] == "partial":
        i_st, i_ed = int(st * sr), int(params['ed'] * sr)
        read_slice = slice(i_st, i_ed)
        trace_len = i_ed - i_st
        win_len = int(sr * params['demod_len'])
    else:
        with h5py.File(files[0], 'r') as f:
            trace_len = f['DutChannel_1_Acquisition_0/trace'].shape[0]
        read_slice = slice(None)
        win_len = trace_len 

    # 2. Pre-calculate rotation vector (Phasor) and Kernel
    # Moves trigonometric ops outside the loop
    t_rel = np.arange(trace_len) / sr
    phasor = np.exp(1j * (2 * np.pi * freq * (t_rel + st) + params['demod_phase']))

    kernel = None
    if w_type == "gaussian":
        kernel = np.exp(-0.5 * np.linspace(-1.5, 1.5, win_len) ** 2)
        kernel /= kernel.sum()

    # 3. Pre-allocate shared memory for zero-copy results
    res_len = trace_len - win_len + 1
    results = np.empty((len(files), res_len), dtype=np.complex128)

    # --- Parallel Execution ---
    def worker(i, file_path):
        with h5py.File(file_path, 'r', libver='latest', rdcc_nbytes=4*1024**2) as f:
            raw = f['DutChannel_1_Acquisition_0/trace'][read_slice]
        
        # Vectorized mixing
        inter = raw * phasor 
        
        if w_type == "uniform":
            cs = np.cumsum(inter)
            # Vectorized moving average
            results[i] = (cs[win_len-1:] - np.concatenate(([0], cs[:-win_len]))) / win_len
            results[i, 0] = cs[win_len-1] / win_len # Boundary fix
        else:
            results[i] = fftconvolve(inter, kernel, mode="valid")

    # Use threads for shared memory access (no pickle overhead)
    Parallel(n_jobs=-1, require='sharedmem', prefer='threads')(
        delayed(worker)(i, f) for i, f in enumerate(files)
    )

    t_out = (np.arange(res_len) / sr) + st + (win_len-1)/sr
    return t_out, results


# --- 4. Analysis Utilities ---
def find_oscillation_period(signal, sample_rate):
    # Use real-FFT (2x faster than standard FFT)
    n = len(signal)
    if n == 0: return np.inf
    
    fft_mag = np.abs(rfft(signal - np.mean(signal)))
    peak_idx = np.argmax(fft_mag[1:]) + 1 
    
    return (peak_idx * sample_rate / n) if peak_idx > 0 else np.inf

def find_fwhm(signal, time):
    # Linear interpolation for sub-sample precision
    y = np.abs(signal)
    half_max = np.max(y) / 2.0
    
    # Find indices where signal crosses half_max
    diff = y - half_max
    indices = np.where(np.diff(np.sign(diff)))[0]
    
    if len(indices) < 2: return 0.0
    
    # Interpolate roots: t = t1 + (t2-t1) * (0-y1)/(y2-y1)
    roots = [time[i] - diff[i] * (time[i+1]-time[i]) / (diff[i+1]-diff[i]) for i in indices]
    
    return roots[-1] - roots[0]