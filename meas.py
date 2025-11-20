import numpy as np
try:
    import keysight.qcs as qcs
except:
    print("No Keysight package")
    pass
import os
import h5py
import json

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
            
            # elif params[1] == "bir4":
            #     envelope_type = params[1]
            #     duration      = params[2]
            #     amplitude     = params[3]
            #     frequency     = params[4]
            #     phase         = params[5]
            #     awg_channel   = params[6]
            #     pre_delay     = params[7]
            #     df_hz         = params[8]     # 半扫频宽（Hz）
            #     beta          = params[9]
            #     n_sech        = params[10]
            #     # 两次相位跳变，缺省为 π 旋转设置
            #     dphi1 = params[11] if len(params) > 11 else 1.5*np.pi
            #     dphi2 = params[12] if len(params) > 12 else -0.5*np.pi

            #     time, envelope = zip(bir4_waveform(duration, df_hz, beta, n_sech, dphi1, dphi2))
            #     bir4_env = qcs.ArbitraryEnvelope(time[0], envelope[0])
            #     pulse = qcs.RFWaveform(duration=duration, envelope=bir4_env,
            #                         amplitude=amplitude, rf_frequency=frequency,
            #                         instantaneous_phase=phase)
            #     program.add_waveform(pulse, awg_channel, pre_delay=pre_delay)
            elif params[1] == "bir4":
                # Expected params list structure:
                # [type, 'bir4', duration, amplitude, frequency, phase, awg, delay, 
                #  dw0, beta, kappa, theta]
                
                envelope_type = params[1]
                duration      = params[2]
                amplitude     = params[3]
                frequency     = params[4]
                phase         = params[5]
                awg_channel   = params[6]
                pre_delay     = params[7]
                
                # Custom BIR4 params
                # dw0: FM scaling (rad/s), matches params['dw0'] in simulation
                dw0   = params[8]  
                beta  = params[9]
                kappa = params[10]
                # Theta (flip angle), default to pi if not provided
                theta = params[11] if len(params) > 11 else np.pi

                # Generate Waveform
                # Note: bir4_waveform returns [times, signal]
                time_arr, signal_arr = bir4_waveform(duration, dw0, beta, kappa, theta)
                
                # Create QCS Envelope
                bir4_env = qcs.ArbitraryEnvelope(time_arr, signal_arr)
                
                # Create Pulse
                # amplitude param scales the normalized signal
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

def vlin(st,ed,num_points):
    return np.linspace(st,ed,num_points)

def vspn(center,spn,num_points):
    return np.linspace(center-spn/2,center+spn/2,num_points)

def dB2V(dBm):
    return np.sqrt(10**( (dBm-30)/10 ) * 50)  / np.sqrt(10**( (0-30)/10 ) * 50)

def V2dB(V):
    Vref=np.sqrt(10**( (0-30)/10 ) * 50)
    return 10*np.log10( (V*Vref) **2/50/0.001)


# def bir4_waveform(tau, df_hz, beta, n=1.0,
#                   dphi1=1.5*np.pi, dphi2=-0.5*np.pi,
#                   sample_rate=4.8e9):
#     """
#     BIR-4-π 复包络（与 wurst_waveform 返回格式一致）:
#     :tau: 总时长 [s]
#     :df_hz: 半扫频宽 [Hz]（总扫频≈ 2*df_hz）
#     :beta: 绝热参数（经验: beta*(tau/4) ≳ 3）
#     :n: 幅度包络 sech^n
#     :dphi1, dphi2: 两次相位跳变（π 旋转默认 3π/2, -π/2）
#     :sample_rate: 采样率 [Sa/s]
#     返回: [times(s), signal=amp*exp(1j*phi)]
#     """
#     sr = float(sample_rate)
#     N  = int(np.round(tau * sr))
#     if N < 8:
#         raise ValueError("tau 与 sample_rate 组合太短，采样点不足。")

#     # 四段等长（最后一段吃掉取整误差），每段时长 Tq=tau/4
#     Nq = N // 4
#     Ns = [Nq, Nq, Nq, N - 3*Nq]
#     Tq = tau / 4.0
#     dt = 1.0 / sr

#     def ahp_forward(ns, tq):
#         # 局部时间 u ∈ [-tq/2, tq/2)
#         u = np.linspace(-tq/2, tq/2, ns, endpoint=False)
#         amp   = 1.0 / (np.cosh(beta * u) ** n)                 # Ω(t) ≤ 1
#         delta = (2.0*np.pi*df_hz) * np.tanh(beta * u)          # Δ(t) [rad/s]
#         return amp.astype(np.float64), delta.astype(np.float64)

#     # 1:FWD, 2:REV, 3:FWD, 4:REV（REV=时间反向且Δ取负）
#     amp1, d1 = ahp_forward(Ns[0], Tq)
#     amp2, d2 = ahp_forward(Ns[1], Tq); amp2 = amp2[::-1]; d2 = -d2[::-1]
#     amp3, d3 = ahp_forward(Ns[2], Tq)
#     amp4, d4 = ahp_forward(Ns[3], Tq); amp4 = amp4[::-1]; d4 = -d4[::-1]

#     amp = np.concatenate([amp1, amp2, amp3, amp4])
#     omg = np.concatenate([d1,   d2,   d3,   d4  ])            # Δ(t) [rad/s]
#     phi = np.cumsum(omg) * dt                                  # 相位 = ∫Δ dt

#     # 段边界相位跳变（进入段2、进入段4）
#     b1 = Ns[0]
#     b3 = Ns[0] + Ns[1] + Ns[2]
#     phi[b1:] += dphi1
#     phi[b3:] += dphi2

#     times  = np.linspace(0.0, tau, N, endpoint=False)          # 与你 WURST 一致
#     signal = amp * np.exp(1j * phi)

#     return [times, signal]

def bir4_waveform(tau, dw0, beta, kappa, theta=np.pi, sample_rate=4.8e9):
    """
    BIR-4 Waveform generation matching the 'bir4_user' PyTorch implementation.
    
    Parameters
    ----------
    tau : float
        Pulse duration [s]
    dw0 : float
        Frequency modulation scaling factor [rad/s] (approx bandwidth)
    beta : float
        AM shape parameter (tanh slope)
    kappa : float
        FM shape parameter (tanh slope)
    theta : float
        Target flip angle [rad], default is pi.
    sample_rate : float
        Sampling rate [Hz]
    """
    # 1. Time Grid
    n_samples = int(tau * sample_rate)
    times = np.linspace(0, tau, n_samples)
    dt = times[1] - times[0] if n_samples > 1 else 0
    
    # Normalized time t in [0, 1]
    t = times / tau
    
    # 2. Define Segments
    # Seg 1: [0, 0.25), Seg 2: [0.25, 0.5), Seg 3: [0.5, 0.75), Seg 4: [0.75, 1.0]
    m1 = t < 0.25
    m2 = (t >= 0.25) & (t < 0.5)
    m3 = (t >= 0.5) & (t < 0.75)
    m4 = t >= 0.75
    
    # 3. AM Envelope (a(t)) - using tanh as per bir4_user
    # Logic: 
    # a1 = tanh(beta * (1 - 4t))
    # a2 = tanh(beta * (4t - 1))
    # a3 = tanh(beta * (3 - 4t))
    # a4 = tanh(beta * (4t - 3))
    amp = np.zeros(n_samples)
    amp[m1] = np.tanh(beta * (1 - 4 * t[m1]))
    amp[m2] = np.tanh(beta * (4 * t[m2] - 1))
    amp[m3] = np.tanh(beta * (3 - 4 * t[m3]))
    amp[m4] = np.tanh(beta * (4 * t[m4] - 3))
    
    # 4. Phase Jump (dphi)
    # Applied to the middle half (Seg 2 and Seg 3)
    # dphi = pi + theta/2
    phase_step = np.zeros(n_samples)
    dphi = np.pi + theta / 2.0
    phase_step[m2 | m3] = dphi

    # 5. FM Envelope (omega(t)) -> Phase Integration
    # Logic from bir4_user (using tanh, NOT tan):
    # om1 = dw0 * tanh(kappa * 4t)
    # om2 = dw0 * tanh(kappa * (4t - 2))
    # om3 = dw0 * tanh(kappa * (4t - 2))
    # om4 = dw0 * tanh(kappa * (4t - 4))
    omega = np.zeros(n_samples)
    omega[m1] = dw0 * np.tanh(kappa * (4 * t[m1]))
    omega[m2] = dw0 * np.tanh(kappa * (4 * t[m2] - 2))
    omega[m3] = dw0 * np.tanh(kappa * (4 * t[m3] - 2))
    omega[m4] = dw0 * np.tanh(kappa * (4 * t[m4] - 4))
    
    # Force the last point to 0 to match PyTorch logic strictly (optional but safe)
    if n_samples > 0:
        omega[-1] = 0.0

    # Integrate frequency to get FM phase
    phi_fm = np.cumsum(omega) * dt
    
    # 6. Combine to Complex Signal
    # signal = amp * exp(1j * (phase_step + phi_fm))
    total_phase = phase_step + phi_fm
    signal = amp * np.exp(1j * total_phase)
    
    # Normalize amplitude to 1.0 (QCS handles scaling via the 'amplitude' parameter)
    max_val = np.max(np.abs(signal))
    if max_val > 1e-12:
        signal /= max_val
        
    return [times, signal]