import numpy as np
try:
    import keysight.qcs as qcs
except:
    print("No Keysight package")
    pass
import time
import os
import h5py
import json


# def set_mapper(paras):
#     awg_addresses = [
#         qcs.Address(chassis=1, slot=2, channel=2),
#         qcs.Address(chassis=1, slot=2, channel=3)
#     ]
#     slow_awg_addresses = [
#         qcs.Address(chassis=1, slot=13, channel=1),
#         qcs.Address(chassis=1, slot=13, channel=2)
#     ]
#     digitizer_addresses = [
#         qcs.Address(chassis=1, slot=7, channel=1),
#         qcs.Address(chassis=1, slot=7, channel=4)
#     ]
#     downconverter_addresses = [
#         qcs.Address(chassis=1, slot=6, channel=1),
#         qcs.Address(chassis=1, slot=6, channel=4)
#     ]

#     qubit_awg   = qcs.Channels(range(1), "qubit_awg", absolute_phase=True)
#     readout_awg = qcs.Channels(range(1), "readout_awg", absolute_phase=True)
#     marker_awg  = qcs.Channels(range(1), "marker_awg", absolute_phase=True)
#     osc_trigger = qcs.Channels(range(1), "osc_trigger", absolute_phase=True)
#     dig         = qcs.Channels(range(1), "dig", absolute_phase=True)

#     mapper = qcs.ChannelMapper()
#     mapper.add_channel_mapping(qubit_awg, awg_addresses[0], qcs.InstrumentEnum.M5300AWG)
#     mapper.add_channel_mapping(readout_awg, awg_addresses[1], qcs.InstrumentEnum.M5300AWG)
#     mapper.add_channel_mapping(marker_awg, slow_awg_addresses[0], qcs.InstrumentEnum.M5301AWG)
#     mapper.add_channel_mapping(osc_trigger, slow_awg_addresses[1], qcs.InstrumentEnum.M5301AWG)
#     mapper.add_channel_mapping(dig, digitizer_addresses[0], qcs.InstrumentEnum.M5200Digitizer)
#     mapper.add_downconverters(digitizer_addresses[0], downconverter_addresses[0])
#     mapper.add_downconverters(digitizer_addresses[1], downconverter_addresses[1])
#     mapper.set_lo_frequencies(
#         lo_frequency=paras["Res_freq"] - 100e6,
#         addresses=[awg_addresses[1], downconverter_addresses[0], downconverter_addresses[1]]
#     )
#     mapper.set_lo_frequencies(
#         lo_frequency=7e9,
#         addresses=[awg_addresses[0]]
#     )
#     return mapper,qubit_awg,readout_awg,marker_awg,osc_trigger,dig

# def wurst_waveform(tau, bw, N, phase_skew):
#     """"
#     Creates envelope for WURST pulse
#     :tau: duration of pulse
#     :n_steps: number of items in array
#     :bw: bandwidth
#     :f0: central frequency
#     :N: controls "squareness" of amplitude shape 
#     :phase_skew: phase added to orthogonal quadrature
#     """
#     sample_rate=4.8e9
#     times = np.linspace(0,tau,int(tau*sample_rate) )
#     # times = np.linspace(0,tau,n_steps )
#     # amplitude = np.array([1-np.abs(np.cos(np.pi*(t-(tau/2)/tau))**N for t in times])
#     amplitude = np.array([1-np.abs(np.cos(np.pi*t/tau))**N for t in times])
#     phi = np.array([(-bw/2)*t + (bw/(2*tau))*(t**2)- np.pi/2 for t in times])
#     # frequency = np.array([-bw/2 + (bw/tau)*t for t in times])

#     I = amplitude*np.sin(phi)
#     Q = amplitude*np.cos(phi + phase_skew)
#     # I = amplitude*np.cos(2*np.pi*frequency*times)
#     # Q = amplitude*np.sin(2*np.pi*frequency*times+phase_skew)
#     signal = I+Q*1j

#     return [times, signal]

def wurst_waveform(tau, bw, N, phase_skew):
    """"
    Creates envelope for WURST pulse
    :tau: duration of pulse
    :n_steps: number of items in array
    :bw: bandwidth need to add 2*np.pi!!!
    :f0: central frequency
    :N: controls "squareness" of amplitude shape 
    :phase_skew: phase added to orthogonal quadrature
    """
    sample_rate=4.8e9
    times = np.linspace(0,tau,int(tau*sample_rate) )

    amplitude = np.array([1-np.abs(np.cos(np.pi*t/tau))**N for t in times])
    phi = np.array([(-bw/2)*t + (bw/(2*tau))*(t**2) for t in times])
    signal = amplitude*np.exp(1j*phi)

    return [times, signal]

# def wideband_noise(duration, bw=2e6, amplitude=1.0, sample_rate=4.8e9):
#     """
#     Generate wideband noise between 0–bw (Hz).
    
#     Parameters
#     ----------
#     duration : float
#         Duration of the noise signal in seconds.
#     sample_rate : float, optional
#         Sampling rate in Hz (default = 4.8 GHz).
#     bw : float, optional
#         Bandwidth of the noise (default = 2 MHz).
#     amplitude : float, optional
#         Amplitude scaling factor for noise.
    
#     Returns
#     -------
#     times : ndarray
#         Time array.
#     noise_signal : ndarray (complex)
#         Complex-valued band-limited noise signal.
#     """
#     n_samples = int(duration * sample_rate)
#     times = np.linspace(0, duration, n_samples, endpoint=False)
    
#     # Generate white noise in frequency domain
#     freqs = np.fft.fftfreq(n_samples, 1/sample_rate)
#     spectrum = np.zeros(n_samples, dtype=complex)
    
#     # Select band [0, bw]
#     mask = (freqs >= 0) & (freqs <= bw)
#     spectrum[mask] = (np.random.randn(np.sum(mask)) +
#                       1j*np.random.randn(np.sum(mask)))
    
#     # Mirror negative frequencies for a real-valued time-domain signal
#     spectrum[mask[::-1]] = np.conj(spectrum[mask])
    
#     # Convert back to time domain
#     noise_signal = np.fft.ifft(spectrum)
    
#     # Scale amplitude
#     noise_signal *= amplitude / np.max(np.abs(noise_signal))
    
#     return times, noise_signal
def wideband_noise(duration, bw=2e6, noise_amp=1.0, offset=0.0, sample_rate=4.8e9):
    """
    Generate wideband noise between 0–bw (Hz) with separate tunable noise amplitude and DC offset.
    
    Parameters
    ----------
    duration : float
        Duration of the signal (seconds)
    bw : float
        Noise bandwidth (Hz)
    noise_amp : float
        RMS amplitude (scale) of the noise fluctuations
    offset : float
        DC offset added to the signal
    sample_rate : float
        Sampling rate (Hz)
    """
    n_samples = int(duration * sample_rate)
    times = np.linspace(0, duration, n_samples, endpoint=False)

    freqs = np.fft.fftfreq(n_samples, 1/sample_rate)
    spectrum = np.zeros(n_samples, dtype=complex)

    # Two-sided flat noise spectrum up to ±bw/2
    mask = np.abs(freqs) <= bw/2
    spectrum[mask] = (np.random.randn(np.sum(mask)) +
                      1j*np.random.randn(np.sum(mask)))

    # Hermitian symmetry to ensure real-valued time-domain noise
    spectrum[mask[::-1]] = np.conj(spectrum[mask])

    # Convert back to time domain
    noise_signal = np.fft.ifft(spectrum)
    noise_signal = np.real(noise_signal)
    
    # Normalize to unit RMS, then scale
    noise_signal = noise_signal / np.std(noise_signal)
    noise_signal = noise_amp * noise_signal + offset

    return times, noise_signal


def generate_pulse(envelope_type, duration, amplitude, frequency, delay, awg, phase=0, rise=0):
    if envelope_type == 'gaus':
        envelope = qcs.GaussianEnvelope(2)
        pulse = qcs.RFWaveform(duration=duration, envelope=envelope,
                           amplitude=amplitude, rf_frequency=frequency,
                           instantaneous_phase=phase)
    elif envelope_type == 'square':
        envelope = qcs.ConstantEnvelope()
        pulse = qcs.RFWaveform(duration=duration, envelope=envelope,
                           amplitude=amplitude, rf_frequency=frequency,
                           instantaneous_phase=phase)
    elif envelope_type == 'flat top gaus':
        envelope = qcs.GaussianEnvelope()
        pulse = qcs.DCWaveform(duration=rise*2, envelope=envelope,
                           amplitude=amplitude)
    else:
        raise ValueError(f"unknown envelope: {envelope_type}")
    
    return pulse, awg, delay

def add_pulses_to_program(program, pulse_params_list):
    for params in pulse_params_list:
        pulse_type = params[0]
        if pulse_type == 'waveform':
            if params[1]=="square":
                envelope_type = params[1]
                duration = params[2]
                amplitude = params[3]
                frequency = params[4]
                phase = params[5]
                awg = params[6]
                delay = params[7]
                pulse, awg_channel, pre_delay = generate_pulse(
                    envelope_type, duration, amplitude, frequency, delay, awg, phase
                )
                program.add_waveform(pulse, awg_channel, pre_delay=pre_delay)
            elif params[1]=="gaus":
                envelope_type = params[1]
                duration = params[2]
                amplitude = params[3]
                frequency = params[4]
                phase = params[5]
                awg = params[6]
                delay = params[7]
                pulse, awg_channel, pre_delay = generate_pulse(
                    envelope_type, duration, amplitude, frequency, delay, awg, phase
                )
                program.add_waveform(pulse, awg_channel, pre_delay=pre_delay)
            elif params[1]=="flat top gaus":
                envelope_type = params[1]
                duration = params[2]
                amplitude = params[3]
                frequency = params[4]
                phase = params[5]
                awg = params[6]
                delay = params[7]
                rise = params[8]
                pulse, awg_channel, pre_delay = generate_pulse(
                    envelope_type, duration, amplitude, frequency, delay, awg, phase, rise
                )
                program.add_waveform(pulse.to_flattop(duration), awg_channel, pre_delay=pre_delay)
            elif params[1]=="wurst":
                envelope_type = params[1]
                duration = params[2]
                amplitude = params[3]
                frequency = params[4]
                phase = params[5]
                awg_channel = params[6]
                pre_delay = params[7]
                bdw = params[8]
                N = params[9]
                phase_skew = params[10]

                time, envelope = zip(wurst_waveform(duration, bdw*2*np.pi, N, phase_skew))
                wurst_env = qcs.ArbitraryEnvelope(time[0], envelope[0])
                pulse = qcs.RFWaveform(duration=duration, envelope=wurst_env,
                           amplitude=amplitude, rf_frequency=frequency,
                           instantaneous_phase=phase)
                program.add_waveform(pulse, awg_channel, pre_delay=pre_delay)
            elif params[1] == "wideband_noise":
                envelope_type = params[1]
                duration = params[2]
                noise_amp = params[3]       # rename for clarity
                frequency = params[4]
                phase = params[5]
                awg_channel = params[6]
                pre_delay = params[7]
                bdw = params[8]
                offset = params[9]

                time, envelope = wideband_noise(duration, bw=bdw, noise_amp=noise_amp, offset=offset)

                max_val = np.max(envelope)
                envelope_scaled = envelope / max_val

                noise_env = qcs.ArbitraryEnvelope(time, envelope_scaled)
                # amplitude=1.0 ensures AWG doesn’t re-scale the envelope
                pulse = qcs.RFWaveform(duration=duration, envelope=noise_env,
                                    amplitude=max_val,
                                    rf_frequency=frequency,
                                    instantaneous_phase=phase)
                program.add_waveform(pulse, awg_channel, pre_delay=pre_delay)
            # elif params[1] == "bir4":
            #     # 参数顺序与 wurst 分支尽量保持一致的风格：
            #     # ['waveform','bir4', duration, amplitude, frequency, phase, awg, pre_delay,
            #     #   df_hz, beta, n, dphi1, dphi2]
            #     envelope_type = params[1]
            #     duration     = params[2]
            #     amplitude    = params[3]
            #     frequency    = params[4]
            #     phase        = params[5]
            #     awg_channel  = params[6]
            #     pre_delay    = params[7]
            #     df_hz        = params[8]
            #     beta         = params[9]
            #     n            = params[10]
            #     dphi1        = params[11] if len(params) > 11 else 1.5*np.pi
            #     dphi2        = params[12] if len(params) > 12 else -0.5*np.pi

            #     time, envelope = zip(bir4_waveform(duration, df_hz, beta, n, dphi1, dphi2))
            #     bir4_env = qcs.ArbitraryEnvelope(time[0], envelope[0])
            #     pulse = qcs.RFWaveform(duration=duration, envelope=bir4_env,
            #                         amplitude=amplitude, rf_frequency=frequency,
            #                         instantaneous_phase=phase)
            #     program.add_waveform(pulse, awg_channel, pre_delay=pre_delay)
            elif params[1] == "bir4":
                envelope_type = params[1]
                duration      = params[2]
                amplitude     = params[3]
                frequency     = params[4]
                phase         = params[5]
                awg_channel   = params[6]
                pre_delay     = params[7]
                df_hz         = params[8]     # 半扫频宽（Hz）
                beta          = params[9]
                n_sech        = params[10]
                # 两次相位跳变，缺省为 π 旋转设置
                dphi1 = params[11] if len(params) > 11 else 1.5*np.pi
                dphi2 = params[12] if len(params) > 12 else -0.5*np.pi

                time, envelope = zip(bir4_waveform(duration, df_hz, beta, n_sech, dphi1, dphi2))
                bir4_env = qcs.ArbitraryEnvelope(time[0], envelope[0])
                pulse = qcs.RFWaveform(duration=duration, envelope=bir4_env,
                                    amplitude=amplitude, rf_frequency=frequency,
                                    instantaneous_phase=phase)
                program.add_waveform(pulse, awg_channel, pre_delay=pre_delay)
            else:
                print("Pulse shape is unknown.")
            
        elif pulse_type == 'acquisition':
            envelope_type = params[1]
            duration = params[2]
            amplitude = params[3]
            frequency = params[4]
            phase = params[5]
            dig = params[6]
            delay = params[7]
            if envelope_type == 'square':
                envelope = qcs.ConstantEnvelope()
            elif envelope_type == 'gaus':
                envelope = qcs.GaussianEnvelope(2)
            else:
                raise ValueError(f"Unknown envelope: {envelope_type}")
            integrationEnvelope = qcs.RFWaveform(
                duration=duration,
                envelope=envelope,
                amplitude=amplitude,
                rf_frequency=frequency
            )
            integrationFilter = qcs.IntegrationFilter(integrationEnvelope)
            program.add_acquisition(integrationFilter, dig, pre_delay=delay)


def set_digitizer_range(mapper, dig_name, paras):
    digiChannel = mapper.get_physical_channels(dig_name)
    digiChannel[0].settings.range.value = paras["Digi_Range"]

# def save_parameters(filepath, parameters):
#     with h5py.File(filepath, 'a') as f:
#         param_group = f.create_group('parameters')
#         for key, value in parameters.items():
#             param_group.attrs[key] = value

# --- 2. HDF5 I/O 优化: 利用 attrs 和 latest内核 ---
def save_parameters(filepath, parameters):
    # 使用 libver='latest' 加速文件结构写入
    with h5py.File(filepath, 'a', libver='latest') as f:
        # 优先检查 parameters 是否存在
        grp = f.require_group('parameters')
        
        for key, value in parameters.items():
            if isinstance(value, dict):
                sub_grp = grp.require_group(key)
                # 字典通常是元数据，存为 attrs 最快且最符合 HDF5 语义
                sub_grp.attrs.update(value) 
            else:
                grp.attrs[key] = value

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_data(retValue, filepath, paras, folder_path):
    retValue.to_hdf5(filepath)
    save_parameters(filepath, paras)
    with open(filepath.replace("hdf5","json"), 'w') as json_file:
        json.dump(paras, json_file, indent=4)

def generate_filename(date_str, filename):
    folder_path = os.path.join("C:\\MW_Data", date_str[:6], f"Data_{date_str[6:]}")
    create_directory(folder_path)
    filepath = os.path.join(folder_path, filename)
    return filepath, folder_path

def generate_filename_zdrive(date_str, filename):
    folder_path = os.path.join("Z:\\MW_Data", date_str[:6], f"Data_{date_str[6:]}")
    create_directory(folder_path)
    filepath = os.path.join(folder_path, filename)
    return filepath, folder_path

# def run_exp(paras,pulse_params_list,mapper,render=False):
#     program = qcs.Program()
#     add_pulses_to_program(program, pulse_params_list)
#     set_digitizer_range(mapper, 'dig', paras)
#     program.n_shots(paras["shot_num"])
#     execute_sequence = qcs.HclBackend(mapper, hw_demod=False, init_time=paras["init_time"])
#     if not execute_sequence.is_system_ready():
#         raise RuntimeError("System not ready.")

#     if render==True:
#         program.render()
#     else:
#         retValue = execute_sequence.apply(program)
#         filepath, folder_path = generate_filename(paras, pulse_duration, re)
#         save_data(retValue, filepath, paras, pulse_duration, folder_path)
#         time.sleep(paras["sleep_time"])
#         return retValue


def vlin(st,ed,num_points):
    return np.linspace(st,ed,num_points)

def vspn(center,spn,num_points):
    return np.linspace(center-spn/2,center+spn/2,num_points)

def dB2V(dBm):
    return np.sqrt(10**( (dBm-30)/10 ) * 50)  / np.sqrt(10**( (0-30)/10 ) * 50)

def V2dB(V):
    Vref=np.sqrt(10**( (0-30)/10 ) * 50)
    return 10*np.log10( (V*Vref) **2/50/0.001)


# import numpy as np

# def bir4_waveform(tau, df_hz, beta, n=1.0,
#                   dphi1=1.5*np.pi, dphi2=-0.5*np.pi,  # π 旋转的两次相位跳变
#                   sample_rate=4.8e9):
#     """
#     生成 BIR-4-π 复包络（I+jQ），与 wurst_waveform 返回格式一致：
#     返回 [times, signal]，其中 signal 为复数组（幅度 * e^{i*相位}）

#     参数
#     ----
#     tau : float
#         总时长 (s)
#     df_hz : float
#         半扫频宽（Hz），瞬时失谐：Δ(t) = ± 2π*df_hz * tanh(beta * u)
#     beta : float
#         绝热参数（控制时间-带宽积），建议满足 beta*(tau/4)/2 ≳ 3
#     n : float
#         sech^n 的 n（幅度包络：Ω(t) ∝ sech^n(beta*u)）
#     dphi1, dphi2 : float
#         两次相位跳变（单位：弧度），π 旋转默认 (3π/2, -π/2)
#     sample_rate : float
#         采样率（与你脚本一致）

#     说明
#     ----
#     BIR-4 = AHP(FWD) → [Beff翻转 + 相位跳变1] → AHP(REV) →
#             AHP(FWD) → [Beff翻转 + 相位跳变2] → AHP(REV)
#     四段等长，各占 tau/4；第二、四段为时间反向镜像。
#     """
#     sr = sample_rate
#     N  = int(np.round(tau * sr))
#     if N < 8:
#         raise ValueError("tau 太短或采样率设置不当，导致采样点过少。")

#     # 每段长度（最后一段吃掉余数，确保总长度=N）
#     Nq = N // 4
#     Ns = [Nq, Nq, Nq, N - 3*Nq]
#     Tq = [tau/4.0]*4
#     dt = 1.0 / sr

#     def ahp_forward(ns, tq):
#         # 局部时间从 -tq/2 到 +tq/2（不含右端点，避免边界重叠）
#         u = np.linspace(-tq/2, tq/2, ns, endpoint=False)
#         amp = 1.0 / (np.cosh(beta*u)**n)          # Ω 归一化包络（≤1）
#         delta = (2*np.pi*df_hz) * np.tanh(beta*u) # 瞬时角频率偏移 Δ(t) (rad/s)
#         return amp, delta

#     # 四段：1 FWD, 2 REV, 3 FWD, 4 REV
#     amp1, d1 = ahp_forward(Ns[0], Tq[0])
#     amp2, d2 = ahp_forward(Ns[1], Tq[1]); amp2 = amp2[::-1]; d2 = -d2[::-1]
#     amp3, d3 = ahp_forward(Ns[2], Tq[2])
#     amp4, d4 = ahp_forward(Ns[3], Tq[3]); amp4 = amp4[::-1]; d4 = -d4[::-1]

#     amp = np.concatenate([amp1, amp2, amp3, amp4]).astype(np.float64)
#     omg = np.concatenate([d1,   d2,   d3,   d4  ]).astype(np.float64)

#     # 积分得到相位（相对载频的基带相位），并在段边界加入两次相位跳变
#     phi = np.cumsum(omg) * dt  # 纯由 Δ(t) 积分得到的相位
#     # 段边界索引（累计）
#     b1 = Ns[0]
#     b2 = Ns[0] + Ns[1]
#     b3 = Ns[0] + Ns[1] + Ns[2]

#     # 相位跳变：在进入段2时加 dphi1；进入段4时再加 dphi2
#     phi[b1:] += dphi1
#     phi[b3:] += dphi2

#     times  = np.arange(N) * dt
#     signal = amp * np.exp(1j * phi)  # 复包络（不含外层 amplitude 参数）

#     return [times, signal]


def bir4_waveform(tau, df_hz, beta, n=1.0,
                  dphi1=1.5*np.pi, dphi2=-0.5*np.pi,
                  sample_rate=4.8e9):
    """
    BIR-4-π 复包络（与 wurst_waveform 返回格式一致）:
    :tau: 总时长 [s]
    :df_hz: 半扫频宽 [Hz]（总扫频≈ 2*df_hz）
    :beta: 绝热参数（经验: beta*(tau/4) ≳ 3）
    :n: 幅度包络 sech^n
    :dphi1, dphi2: 两次相位跳变（π 旋转默认 3π/2, -π/2）
    :sample_rate: 采样率 [Sa/s]
    返回: [times(s), signal=amp*exp(1j*phi)]
    """
    sr = float(sample_rate)
    N  = int(np.round(tau * sr))
    if N < 8:
        raise ValueError("tau 与 sample_rate 组合太短，采样点不足。")

    # 四段等长（最后一段吃掉取整误差），每段时长 Tq=tau/4
    Nq = N // 4
    Ns = [Nq, Nq, Nq, N - 3*Nq]
    Tq = tau / 4.0
    dt = 1.0 / sr

    def ahp_forward(ns, tq):
        # 局部时间 u ∈ [-tq/2, tq/2)
        u = np.linspace(-tq/2, tq/2, ns, endpoint=False)
        amp   = 1.0 / (np.cosh(beta * u) ** n)                 # Ω(t) ≤ 1
        delta = (2.0*np.pi*df_hz) * np.tanh(beta * u)          # Δ(t) [rad/s]
        return amp.astype(np.float64), delta.astype(np.float64)

    # 1:FWD, 2:REV, 3:FWD, 4:REV（REV=时间反向且Δ取负）
    amp1, d1 = ahp_forward(Ns[0], Tq)
    amp2, d2 = ahp_forward(Ns[1], Tq); amp2 = amp2[::-1]; d2 = -d2[::-1]
    amp3, d3 = ahp_forward(Ns[2], Tq)
    amp4, d4 = ahp_forward(Ns[3], Tq); amp4 = amp4[::-1]; d4 = -d4[::-1]

    amp = np.concatenate([amp1, amp2, amp3, amp4])
    omg = np.concatenate([d1,   d2,   d3,   d4  ])            # Δ(t) [rad/s]
    phi = np.cumsum(omg) * dt                                  # 相位 = ∫Δ dt

    # 段边界相位跳变（进入段2、进入段4）
    b1 = Ns[0]
    b3 = Ns[0] + Ns[1] + Ns[2]
    phi[b1:] += dphi1
    phi[b3:] += dphi2

    times  = np.linspace(0.0, tau, N, endpoint=False)          # 与你 WURST 一致
    signal = amp * np.exp(1j * phi)

    return [times, signal]





