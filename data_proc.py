import h5py
import numpy as np
try:
    import keysight.qcs as qcs
except:
    print("No Keysight package")
    pass
from joblib import Parallel, delayed, parallel_backend
from lmfit.models import ExpressionModel
from scipy.signal import fftconvolve

lorentz_mod = ExpressionModel('cst + A/(1+((x-x0)/g)**2)')
sine_mod = ExpressionModel('cst - A * cos(3.14159265359 *x/x0)')
exp_mod = ExpressionModel('cst + A * exp(- x/T)')
ramsey_mod = ExpressionModel('cst + A * exp(- x/T) * cos(6.28318530718 * Df * x + phi)')
parabola_mod = ExpressionModel('a*(x-x0)**2+cst')

def save_parameters(filepath, parameters):
    with h5py.File(filepath, 'a') as f:
        param_group = f.require_group('parameters')
        for key, value in parameters.items():
            if isinstance(value, dict):  
                step_group = param_group.require_group(key)
                for sub_key, sub_value in value.items():
                    step_group[sub_key] = sub_value
            else:
                param_group[key] = value

def load_parameters(filepath):
    with h5py.File(filepath, 'r') as f:
        return {key: dict(val) if isinstance(val, h5py.Group) else val[()]
                for key, val in f['parameters'].items()}

# def demodulate(signal, freq, window, t0, phase, window_type="uniform"):   
#     f_samp = 4.8e9
#     omega = 2*np.pi*freq    
#     t = np.arange(0, len(signal)/f_samp, 1/f_samp)
#     inter = signal * np.exp( 1j*( omega * ( t+t0 ) + phase ) ) 
#     if window_type=="uniform":
#         cumsum = np.cumsum(np.insert(inter,0,0))
#         return t[window-1:], ((cumsum[window:]-cumsum[:-window])/window)
#     elif window_type == "gaussian":
#         weights = np.exp(-0.5 * (np.linspace(-1.5, 1.5, window) ** 2))
#         weights /= weights.sum()

#         weighted_sum = np.convolve(inter, weights, mode="valid")
#         # weighted_sum = np.zeros(len(signal) - window + 1, dtype=np.complex128)
#         # for i in range(len(weighted_sum)):
#         #     weighted_sum[i] = np.sum(signal[i:i + window] * weights)

#         return t[window-1:], weighted_sum
#     else:
#         raise ValueError("Invalid window_type. Choose 'gaussian' or 'uniform'.")

def demodulate(signal, freq, window, t0, phase, window_type="uniform"):
    f_samp = 4.8e9
    n = len(signal)
    
    # 构造时间轴和混频信号
    t = np.arange(n) / f_samp  # 生成时间数组
    # 提取常数相位项，减少数组运算量
    phase_const = np.exp(1j * (2 * np.pi * freq * t0 + phase)) 
    inter = signal * np.exp(1j * 2 * np.pi * freq * t) * phase_const # 混频

    # 预先切分出结果的时间轴，避免最后计算
    t_out = t[window-1:] 

    if window_type == "uniform":
        # 使用累加和(cumsum)计算移动平均，复杂度为O(N)
        cs = np.cumsum(inter)
        
        # 逻辑优化：避免使用 np.insert 产生的内存拷贝
        # 结果 = (当前累加值 - window之前的累加值) / window
        res = cs[window-1:].copy() # 获取包含足够数据点的部分
        res[1:] -= cs[:-window]    # 向量化减去滑出窗口的值
        return t_out, res / window
        
    elif window_type == "gaussian":
        # 构造高斯核
        kernel = np.exp(-0.5 * np.linspace(-1.5, 1.5, window) ** 2)
        kernel /= kernel.sum() # 归一化
        
        # 使用基于FFT的卷积，复杂度O(N log N)，远快于普通卷积 O(N*window)
        res = fftconvolve(inter, kernel, mode="valid")
        return t_out, res
        
    else:
        raise ValueError("Invalid window_type")


def readfile(file, **kwargs):
    if kwargs['demod_type']=="full":
        # signal = qcs.load(file).get_trace().values.ravel()#careful!
        with h5py.File(file, 'r') as f:
            signal = f['DutChannel_1_Acquisition_0/trace'][()]
        window = signal.shape[0]
    elif kwargs['demod_type']=="partial":
        with h5py.File(file, 'r') as f:
            signal = f['DutChannel_1_Acquisition_0/trace'][()][int(kwargs['st'] * kwargs['sample_rate']):int(kwargs['ed'] * kwargs['sample_rate'])]
        # signal = qcs.load(file).get_trace().values.ravel()[int(kwargs['st'] * kwargs['sample_rate']):int(kwargs['ed'] * kwargs['sample_rate'])]
        window = int( kwargs['sample_rate'] * kwargs['demod_len'] )
    else:
        raise ValueError("Invalid demod type. Choose 'full' or 'partial'.")
    return demodulate(signal, kwargs['demod_freq'], window, kwargs['st'], kwargs['demod_phase'], kwargs['window_type'])

def calu_data(file, params):
    with parallel_backend('loky', inner_max_num_threads=1):
        demod_list = Parallel(n_jobs=-1)(
            delayed(readfile)(f, **params) for f in file
        )

    time = demod_list[0][0]
    demod2_data = np.stack( [d[1] for d in demod_list],axis=0 )
    return time,demod2_data


def get_data(params):
    with h5py.File(params["tmp_file_path"], "r") as h5f:
        print("Datasets in file:", list(h5f.keys()))
        time_data = h5f["time"][:]  
        demod2_data = h5f["demod2"][:]  
    return time_data, demod2_data

def find_oscillation_period(signal, sample_rate):
    N = len(signal)
    fft_values = fft(signal)
    fft_frequencies = fftfreq(N, d=1/sample_rate)

    fft_magnitudes = np.abs(fft_values[:N // 2])
    fft_frequencies = fft_frequencies[:N // 2]

    dominant_frequency = fft_frequencies[np.argmax(fft_magnitudes)]

    if dominant_frequency != 0:
        return dominant_frequency
    else:
        return np.inf

def find_fwhm(signal,time,threshold=0.01):
    Y=np.abs (signal)*1e3
    index=np.where( np.abs(Y-np.max(Y)/2)<threshold )
    return np.ptp(time[index])