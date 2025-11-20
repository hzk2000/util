# import h5py
# import numpy as np
# try:
#     import keysight.qcs as qcs
# except:
#     print("No Keysight package")
#     pass
# from joblib import Parallel, delayed, parallel_backend
# from lmfit.models import ExpressionModel
# from scipy.signal import fftconvolve

# lorentz_mod = ExpressionModel('cst + A/(1+((x-x0)/g)**2)')
# sine_mod = ExpressionModel('cst - A * cos(3.14159265359 *x/x0)')
# exp_mod = ExpressionModel('cst + A * exp(- x/T)')
# ramsey_mod = ExpressionModel('cst + A * exp(- x/T) * cos(6.28318530718 * Df * x + phi)')
# parabola_mod = ExpressionModel('a*(x-x0)**2+cst')

# def save_parameters(filepath, parameters):
#     with h5py.File(filepath, 'a') as f:
#         param_group = f.require_group('parameters')
#         for key, value in parameters.items():
#             if isinstance(value, dict):  
#                 step_group = param_group.require_group(key)
#                 for sub_key, sub_value in value.items():
#                     step_group[sub_key] = sub_value
#             else:
#                 param_group[key] = value

# def load_parameters(filepath):
#     with h5py.File(filepath, 'r') as f:
#         return {key: dict(val) if isinstance(val, h5py.Group) else val[()]
#                 for key, val in f['parameters'].items()}

import h5py
import numpy as np
from numpy.fft import rfft, rfftfreq
from lmfit.models import LorentzianModel, ExponentialModel, LinearModel, PolynomialModel

# --- 1. 模型优化: 使用内置高性能模型，避免字符串解析 ---
# 组合模型比手写字符串快且稳定
lorentz_mod = LorentzianModel() + LinearModel(prefix='cst_') # A/(...) + cst
exp_mod = ExponentialModel() + LinearModel(prefix='cst_')    # A*exp(-x/t) + cst
parabola_mod = PolynomialModel(degree=2)

# Ramsey 和 Sine 这种特殊组合保留 ExpressionModel 或自定义函数
# 为了速度，建议用 python 函数定义而非字符串
def ramsey_func(x, A, T, Df, phi, cst):
    return cst + A * np.exp(-x/T) * np.cos(2 * np.pi * Df * x + phi)
ramsey_mod = Model(ramsey_func)

def sine_func(x, A, x0, cst):
    return cst - A * np.cos(np.pi * x / x0)
sine_mod = Model(sine_func)


def load_parameters(filepath):
    with h5py.File(filepath, 'r', libver='latest', swmr=True) as f:
        grp = f['parameters']
        params = dict(grp.attrs) # 加载顶层属性
        
        # 加载子组属性
        for key in grp.keys():
            params[key] = dict(grp[key].attrs)
        return params

def get_data(params):
    # 移除 print 阻塞，使用 swmr (Single Writer Multiple Reader) 模式防止读取冲突
    with h5py.File(params["tmp_file_path"], "r", libver='latest', swmr=True) as h5f:
        # 直接读取，利用切片或全读
        return h5f["time"][()], h5f["demod2"][()]


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

# def demodulate(signal, freq, window, t0, phase, window_type="uniform"):
#     f_samp = 4.8e9
#     n = len(signal)
    
#     # 构造时间轴和混频信号
#     t = np.arange(n) / f_samp  # 生成时间数组
#     # 提取常数相位项，减少数组运算量
#     phase_const = np.exp(1j * (2 * np.pi * freq * t0 + phase)) 
#     inter = signal * np.exp(1j * 2 * np.pi * freq * t) * phase_const # 混频

#     # 预先切分出结果的时间轴，避免最后计算
#     t_out = t[window-1:] 

#     if window_type == "uniform":
#         # 使用累加和(cumsum)计算移动平均，复杂度为O(N)
#         cs = np.cumsum(inter)
        
#         # 逻辑优化：避免使用 np.insert 产生的内存拷贝
#         # 结果 = (当前累加值 - window之前的累加值) / window
#         res = cs[window-1:].copy() # 获取包含足够数据点的部分
#         res[1:] -= cs[:-window]    # 向量化减去滑出窗口的值
#         return t_out, res / window
        
#     elif window_type == "gaussian":
#         # 构造高斯核
#         kernel = np.exp(-0.5 * np.linspace(-1.5, 1.5, window) ** 2)
#         kernel /= kernel.sum() # 归一化
        
#         # 使用基于FFT的卷积，复杂度O(N log N)，远快于普通卷积 O(N*window)
#         res = fftconvolve(inter, kernel, mode="valid")
#         return t_out, res
        
#     else:
#         raise ValueError("Invalid window_type")


# def readfile(file, **kwargs):
#     if kwargs['demod_type']=="full":
#         # signal = qcs.load(file).get_trace().values.ravel()#careful!
#         with h5py.File(file, 'r') as f:
#             signal = f['DutChannel_1_Acquisition_0/trace'][()]
#         window = signal.shape[0]
#     elif kwargs['demod_type']=="partial":
#         with h5py.File(file, 'r') as f:
#             signal = f['DutChannel_1_Acquisition_0/trace'][()][int(kwargs['st'] * kwargs['sample_rate']):int(kwargs['ed'] * kwargs['sample_rate'])]
#         # signal = qcs.load(file).get_trace().values.ravel()[int(kwargs['st'] * kwargs['sample_rate']):int(kwargs['ed'] * kwargs['sample_rate'])]
#         window = int( kwargs['sample_rate'] * kwargs['demod_len'] )
#     else:
#         raise ValueError("Invalid demod type. Choose 'full' or 'partial'.")
#     return demodulate(signal, kwargs['demod_freq'], window, kwargs['st'], kwargs['demod_phase'], kwargs['window_type'])

# def readfile(file, **kwargs):
#     sr = kwargs.get('sample_rate', 4.8e9)
#     st = kwargs.get('st', 0)
    
#     # 提前计算索引，减少文件打开期间的CPU耗时
#     if kwargs['demod_type'] == "partial":
#         i_start = int(st * sr)
#         i_end = int(kwargs['ed'] * sr)
#         window = int(sr * kwargs['demod_len'])
    
#     # libver='latest': 使用最新内核加速元数据解析
#     # rdcc_nbytes: 增大块缓存到4MB(默认1MB)，大幅提升非连续数据的读取速度
#     with h5py.File(file, 'r', libver='latest', rdcc_nbytes=4*1024**2) as f:
#         dset = f['DutChannel_1_Acquisition_0/trace']
        
#         if kwargs['demod_type'] == "full":
#             signal = dset[()] # 读取全部
#             window = signal.size
#         elif kwargs['demod_type'] == "partial":
#             signal = dset[i_start:i_end] # 利用HDF5切片直读，不读入无关数据
#         else:
#             raise ValueError("Invalid demod type")

#     return demodulate(signal, kwargs['demod_freq'], window, st, kwargs['demod_phase'], kwargs['window_type'])

# def calu_data(file, params):
#     with parallel_backend('loky', inner_max_num_threads=1):
#         demod_list = Parallel(n_jobs=-1)(
#             delayed(readfile)(f, **params) for f in file
#         )

#     time = demod_list[0][0]
#     demod2_data = np.stack( [d[1] for d in demod_list],axis=0 )
#     return time,demod2_data

# def calu_data(files, params):
#     # 1. 先处理第一个文件，获取时间轴和数据形状
#     time, first_data = readfile(files[0], **params)
    
#     # 2. 预分配结果大数组，避免 np.stack 的巨大内存开销和拷贝时间
#     n_files = len(files)
#     results = np.empty((n_files, *first_data.shape), dtype=first_data.dtype)
#     results[0] = first_data # 填入第一个数据

#     # 定义写入函数，利用闭包直接写入共享内存 results
#     def fill_data(i):
#         _, data = readfile(files[i], **params)
#         results[i] = data # 线程直接写内存，无序列化开销

#     # 3. 使用多线程并行 (prefer='threads')
#     # HDF5 I/O 和 NumPy 计算释放 GIL，线程效率极高且无数据传输成本
#     Parallel(n_jobs=-1, require='sharedmem')(
#         delayed(fill_data)(i) for i in range(1, n_files)
#     )

#     return time, results



# 核心解调逻辑：纯数学运算，无多余对象
def core_process(signal, freq, window, t0, phase, window_type):
    n = len(signal)
    f_samp = 4.8e9
    
    # 1. 混频 (Mixing)
    # 利用广播机制和复数旋转，避免生成巨大的时间数组 t
    # 相对时间 t_rel 用于生成旋转因子
    t_rel = np.arange(n) / f_samp
    # 初始相位包含 t0 和 phase，一次性算好
    phase_init = np.exp(1j * (2 * np.pi * freq * t0 + phase))
    inter = signal * np.exp(1j * 2 * np.pi * freq * t_rel) * phase_init

    # 2. 滤波 (Filtering)
    if window_type == "uniform":
        # 利用 cumsum 实现 O(N) 复杂度的 Boxcar 滤波
        cs = np.cumsum(inter)
        # 向量化切片相减，避免 insert/loop
        res = (cs[window-1:] - np.concatenate(([0], cs[:-window]))) / window
        # 注意：上面的写法是为了处理边界，比 insert 快且省内存
        # 第一点单独修正：cs[window-1] - 0
        res[0] = cs[window-1] / window 
        return res
        
    elif window_type == "gaussian":
        # 预计算高斯核 (假设 window 大小固定，这里每次算也很快，瓶颈在卷积)
        # 如果 window 很大，fftconvolve (O(N logN)) 比 convolve (O(N*W)) 快几十倍
        kernel = np.exp(-0.5 * np.linspace(-1.5, 1.5, window) ** 2)
        kernel /= kernel.sum()
        return fftconvolve(inter, kernel, mode="valid")

def calu_data(files, params):
    # 1. 参数预处理 (避免在循环中重复查字典)
    sr = params.get('sample_rate', 4.8e9)
    freq = params['demod_freq']
    phase = params['demod_phase']
    w_type = params['window_type']
    d_type = params['demod_type']
    st_time = params.get('st', 0)
    
    # 提前计算切片索引
    if d_type == "partial":
        i_st, i_ed = int(st_time * sr), int(params['ed'] * sr)
        win_len = int(sr * params['demod_len'])
    else:
        win_len = None # Full 模式下需读取文件后确定

    # 2. 预读取第一个文件以分配内存 (Memory Pre-allocation)
    # 这是防止内存碎片的关键
    with h5py.File(files[0], 'r') as f:
        dset = f['DutChannel_1_Acquisition_0/trace']
        if d_type == "partial":
            sig_0 = dset[i_st:i_ed]
        else:
            sig_0 = dset[()]
            win_len = sig_0.size # Full模式下更新 window
    
    # 计算输出的时间轴 (只需计算一次)
    t_total = np.arange(len(sig_0)) / sr + st_time
    t_out = t_total[win_len-1:]
    
    # 预分配结果大数组 (Shared Memory)
    n_files = len(files)
    res_len = len(sig_0) - win_len + 1
    results = np.empty((n_files, res_len), dtype=np.complex128)

    # 3. 定义工作函数 (闭包)
    # 直接写入 results，无返回值，无序列化开销
    def worker(i, file_path):
        with h5py.File(file_path, 'r', libver='latest', rdcc_nbytes=4*1024**2) as f:
            dset = f['DutChannel_1_Acquisition_0/trace']
            # 只读硬盘上需要的那一段
            raw = dset[i_st:i_ed] if d_type == "partial" else dset[()]
            
        # 计算并直接写入共享内存
        results[i] = core_process(raw, freq, win_len, st_time, phase, w_type)

    # 4. 多线程并行执行
    # require='sharedmem' 确保线程共享 results 数组
    Parallel(n_jobs=-1, require='sharedmem', prefer='threads')(
        delayed(worker)(i, f) for i, f in enumerate(files)
    )

    return t_out, results


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