import os
import h5py
import json
import numpy as np
try:
    import keysight.qcs as qcs
except ImportError:
    print("No Keysight package")
    pass

# --- 1. Waveform Generators ---

def tan_tanh_waveform(tau, bw, xi, kappa):
    """
    Tan/Tanh Adiabatic Pulse.
    Amp: Tanh envelope (rises fast, flat top)
    Freq: Tan sweep (diverges at ends, very broadband)
    """
    sample_rate = 4.8e9
    n_samples = int(tau * sample_rate)
    times = np.linspace(0, tau, n_samples)
    t = times / tau # 0 to 1

    # Amplitude: Tanh rise and fall
    # Using piece-wise or symmetric formula |tanh|
    # Standard form: tanh(xi * (1 - |2t - 1|))
    amp = np.tanh(xi * (1 - np.abs(2 * t - 1)))
    
    # Frequency: Tan sweep
    # freq(t) ~ tan(kappa * (2t - 1))
    # phase = integral(tan) = -ln(cos)
    t_centered = 2 * t - 1 # -1 to 1
    
    # Normalization factor for bandwidth
    # f_max = tan(kappa). We want f to go from -bw/2 to bw/2
    scale = (bw / 2) / np.tan(kappa)
    
    # Integral of tan(u) is -ln(cos(u))
    # Chain rule factors: d/dt (t_centered) = 2/tau
    phase_factor = scale * (tau / (2 * kappa))
    phase = -phase_factor * np.log(np.cos(kappa * t_centered))
    
    signal = amp * np.exp(1j * phase)
    return times, signal

def c_shape_waveform(tau, bw, base_shape='sech', beta=10, N=20):
    """
    C-Shape / Constant Adiabaticity (GOIA) Pulse.
    Forces constant adiabaticity K by deriving frequency sweep from amplitude.
    freq_sweep(t) propto integral(A(t)^2)
    
    base_shape: 'sech' (C-Shape Sech) or 'wurst' (C-Shape WURST)
    """
    sample_rate = 4.8e9
    n_samples = int(tau * sample_rate)
    times = np.linspace(0, tau, n_samples)
    t_norm = 2 * (times / tau) - 1 # -1 to 1

    # 1. Generate Base Amplitude
    if base_shape == 'sech':
        amplitude = 1.0 / np.cosh(beta * t_norm)
    elif base_shape == 'wurst':
        amplitude = 1 - np.abs(np.cos(np.pi * times / tau)) ** N
    else:
        amplitude = np.ones_like(times)

    # 2. Derive Phase for Constant Adiabaticity
    # d(freq)/dt = const * A^2  => freq = const * integral(A^2)
    # This ensures |A|^2 / |freq_dot| is constant
    cum_power = np.cumsum(amplitude**2)
    total_power = cum_power[-1]
    
    # Normalized frequency sweep from -1 to 1
    freq_norm = 2 * (cum_power / total_power) - 1
    
    # Scale to target bandwidth (-bw/2 to bw/2)
    freq_hz = freq_norm * (bw / 2)
    
    # Integrate frequency to get phase
    dt = times[1] - times[0]
    phase = np.cumsum(freq_hz) * dt
    
    signal = amplitude * np.exp(1j * phase)
    return times, signal

import numpy as np

def bir4_envelope_from_sim(
    tau,
    dw0,
    beta,
    kappa,
    theta=np.pi,
    sample_rate=4.8e9,
    normalize=True,
    omega_end_zero=False,
):
    """
    BIR-4 complex envelope that matches your simulation branch: type == 'bir4_user'
      - AM: piecewise tanh
      - phase step on middle half: exp(1j * dphi), dphi = pi + theta/2
      - FM: piecewise tanh -> omega(t)
      - total phase = angle(a) + cumsum(omega)*dt
      - output complex envelope s(t) = a(t) * exp(1j * cumsum(omega)*dt)

    Parameters
    ----------
    tau : float [s]
    dw0 : float [rad/s]      (same as sim)
    beta : float             (AM tanh slope)
    kappa: float             (FM tanh slope)
    theta: float [rad]       flip angle (pi for inversion)
    sample_rate : float [Hz] (AWG SR)
    normalize : bool         normalize max |s| to 1
    omega_end_zero : bool    if True, force omega[-1]=0 (sim bir4_user used False)

    Returns
    -------
    times : np.ndarray [s]
    env   : np.ndarray complex, same length as times
    """
    sr = float(sample_rate)
    n = int(np.round(tau * sr))
    if n < 8:
        raise ValueError("tau too short or sample_rate too low: not enough samples.")

    # Use endpoint=False so dt == 1/sample_rate and exactly matches AWG sampling
    times = np.arange(n, dtype=np.float64) / sr
    dt = 1.0 / sr
    t = times / tau  # normalized time in [0, 1)

    # Segment masks (same boundaries as sim)
    m1 = t < 0.25
    m2 = (t >= 0.25) & (t < 0.5)
    m3 = (t >= 0.5) & (t < 0.75)
    m4 = t >= 0.75

    # AM: piecewise tanh (same formulas as sim)
    amp = np.zeros(n, dtype=np.float64)
    amp[m1] = np.tanh(beta * (1 - 4 * t[m1]))
    amp[m2] = np.tanh(beta * (4 * t[m2] - 1))
    amp[m3] = np.tanh(beta * (3 - 4 * t[m3]))
    amp[m4] = np.tanh(beta * (4 * t[m4] - 3))

    # Complex a(t) with middle-half phase step (same as sim)
    dphi = np.pi + theta / 2.0
    a = amp.astype(np.complex64)
    a[m2 | m3] *= np.exp(1j * dphi).astype(np.complex64)

    # FM omega(t): piecewise tanh (same as sim bir4_user)
    omega = np.zeros(n, dtype=np.float64)
    omega[m1] = dw0 * np.tanh(kappa * (4 * t[m1]))
    omega[m2] = dw0 * np.tanh(kappa * (4 * t[m2] - 2))
    omega[m3] = dw0 * np.tanh(kappa * (4 * t[m3] - 2))
    omega[m4] = dw0 * np.tanh(kappa * (4 * t[m4] - 4))

    if omega_end_zero:
        omega[-1] = 0.0

    # Integrate omega to get extra FM phase (same discrete rule as sim: cumsum * dt)
    phi_fm = np.cumsum(omega) * dt

    # Total complex envelope: a(t) * exp(i * phi_fm)
    env = a * np.exp(1j * phi_fm).astype(np.complex64)

    if normalize:
        mx = np.max(np.abs(env))
        if mx > 1e-12:
            env = env / mx

    return times, env


def wurst_waveform(tau, bw, N, phase_skew):
    """
    WURST adiabatic pulse.
    Envelope: 1 - |cos|^N
    Phase: Quadratic chirp
    """
    sample_rate = 4.8e9
    times = np.linspace(0, tau, int(tau * sample_rate))
    
    # Amplitude: 1 - |cos(pi*t/tau)|^N
    amplitude = 1 - np.abs(np.cos(np.pi * times / tau)) ** N
    
    # Phase: Integral of linear frequency sweep (-bw/2 to bw/2)
    # phi(t) = -bw/2 * t + (bw/(2*tau)) * t^2
    phi = (-bw / 2) * times + (bw / (2 * tau)) * (times ** 2)
    
    signal = amplitude * np.exp(1j * phi)
    return times, signal

def bir4_waveform(tau, dw0, beta, kappa, theta=np.pi):
    """
    BIR-4 adiabatic pulse using tanh/tanh modulation (User custom version).
    Segmented into 4 parts with phase jumps for arbitrary flip angle.
    """
    sample_rate = 4.8e9
    n_samples = int(tau * sample_rate)
    times = np.linspace(0, tau, n_samples)
    dt = times[1] - times[0] if n_samples > 1 else 0
    t = times / tau # Normalized time [0, 1]

    # Masks for 4 segments
    m1 = t < 0.25
    m2 = (t >= 0.25) & (t < 0.5)
    m3 = (t >= 0.5) & (t < 0.75)
    m4 = t >= 0.75

    # 1. Amplitude (tanh envelope)
    amp = np.zeros(n_samples)
    amp[m1] = np.tanh(beta * (1 - 4 * t[m1]))
    amp[m2] = np.tanh(beta * (4 * t[m2] - 1))
    amp[m3] = np.tanh(beta * (3 - 4 * t[m3]))
    amp[m4] = np.tanh(beta * (4 * t[m4] - 3))

    # 2. Phase jumps (middle sections)
    phase_step = np.zeros(n_samples)
    phase_step[m2 | m3] = np.pi + theta / 2.0

    # 3. Frequency Modulation (tanh sweep)
    omega = np.zeros(n_samples)
    omega[m1] = dw0 * np.tanh(kappa * (4 * t[m1]))
    omega[m2] = dw0 * np.tanh(kappa * (4 * t[m2] - 2))
    omega[m3] = dw0 * np.tanh(kappa * (4 * t[m3] - 2))
    omega[m4] = dw0 * np.tanh(kappa * (4 * t[m4] - 4))
    if n_samples > 0: omega[-1] = 0.0

    # Integrate FM to get phase
    phi_fm = np.cumsum(omega) * dt
    
    signal = amp * np.exp(1j * (phase_step + phi_fm))
    
    # Normalize max amplitude to 1
    if np.max(np.abs(signal)) > 1e-12:
        signal /= np.max(np.abs(signal))
        
    return times, signal

def hs_waveform(tau, bw, beta):
    """
    Hyperbolic Secant (HS) Adiabatic Pulse.
    Amplitude: sech(beta * t')
    Frequency: tanh(beta * t')
    Excellent bandwidth-limited inversion.
    """
    sample_rate = 4.8e9
    n_samples = int(tau * sample_rate)
    times = np.linspace(0, tau, n_samples)
    
    # Centered normalized time from -1 to 1
    t_norm = 2 * (times / tau) - 1 
    
    # Amplitude: sech(beta * t)
    # Using trignometric identity sech(x) = 1/cosh(x)
    amplitude = 1.0 / np.cosh(beta * t_norm)
    
    # Frequency: bw/2 * tanh(beta * t)
    # Phase is integral of freq: (bw * tau / (2 * beta)) * ln(cosh(beta * t))
    # Note: Factor 2 in denominator comes from dt/dt_norm = tau/2
    prefactor = (bw * np.pi) * (tau / (2 * beta)) # bw is in Hz here, need rad/s? Assuming bw input is rad/s
    # If bw input is Hz: prefactor = (bw * 2 * np.pi) * ...
    # Let's assume bw is passed as angular frequency (rad/s) like WURST
    
    phase = (bw / 2) * (tau / (2 * beta)) * np.log(np.cosh(beta * t_norm))
    
    signal = amplitude * np.exp(1j * phase)
    return times, signal

def wideband_noise(duration, bw, offset=0.0):
    """
    Generates band-limited white noise.
    """
    sample_rate = 4.8e9
    n_samples = int(duration * sample_rate)
    times = np.linspace(0, duration, n_samples, endpoint=False)

    freqs = np.fft.fftfreq(n_samples, 1/sample_rate)
    spectrum = np.zeros(n_samples, dtype=complex)

    # Flat spectrum within bandwidth
    mask = np.abs(freqs) <= bw/2
    spectrum[mask] = (np.random.randn(np.sum(mask)) + 1j*np.random.randn(np.sum(mask)))
    spectrum[mask[::-1]] = np.conj(spectrum[mask]) # Hermitian symmetry for real signal

    noise_signal = np.real(np.fft.ifft(spectrum))
    noise_signal = noise_signal / np.std(noise_signal) # Unit RMS
    noise_signal = noise_signal + offset

    return times, noise_signal

# --- 2. QCS Envelope Factory ---

def create_envelope(env_type, duration, extras):
    """ Factory to generate QCS Envelopes """
    
    if env_type == 'wurst':
        # [bw, N, skew]
        times, sig = wurst_waveform(duration, extras[0]*2*np.pi, extras[1], extras[2])
        return qcs.ArbitraryEnvelope(times, sig)
    
    elif env_type == 'hs':
        # [bw, beta]
        # Example: bw=200MHz, beta=10
        times, sig = hs_waveform(duration, extras[0]*2*np.pi, extras[1])
        return qcs.ArbitraryEnvelope(times, sig)
        
    elif env_type == 'tan_tanh':
        # [bw, xi, kappa]
        # xi: amp slope (e.g. 10), kappa: freq curvature (e.g. 1.5, must be < pi/2)
        times, sig = tan_tanh_waveform(duration, extras[0]*2*np.pi, extras[1], extras[2])
        return qcs.ArbitraryEnvelope(times, sig)

    elif env_type == 'c_shape':
        # [bw, base_type, param1, param2]
        # base_type: 'sech' or 'wurst'
        # param1: beta (for sech) or N (for wurst)
        base = 'sech' if len(extras) < 2 else extras[1]
        p1 = 10 if len(extras) < 3 else extras[2]
        p2 = 20 if len(extras) < 4 else extras[3]
        times, sig = c_shape_waveform(duration, extras[0]*2*np.pi, base, p1, p2)
        return qcs.ArbitraryEnvelope(times, sig)

    elif env_type == 'bir4':
        # [dw0, beta, kappa, theta]
        times, sig = bir4_waveform(duration, extras[0], extras[1], extras[2], extras[3] if len(extras)>3 else np.pi)
        return qcs.ArbitraryEnvelope(times, sig)

    elif env_type == 'wideband_noise':
        times, sig = wideband_noise(duration, extras[0], extras[1])
        return qcs.ArbitraryEnvelope(times, sig / np.max(np.abs(sig))), np.max(np.abs(sig))
        
    elif env_type in ['square', 'gaus', 'flat top gaus']:
        # Handled by QCS built-ins
        return None 
        
    else:
        raise ValueError(f"Unknown envelope: {env_type}")

# --- 3. Main Program Builder ---

def add_pulses_to_program(program, pulse_params_list):
    """
    Iterates through pulse parameters and adds them to the QCS program.
    Supports composite pulses (BB1) via expansion.
    """
    for params in pulse_params_list:
        category = params[0]
        env_type = params[1]
        
        # BB1 Macro Expansion
        if env_type == 'bb1':
            # Usage: ['waveform', 'bb1', dur_pi, amp, freq, phase, ch, pre_del]
            # Expands to 4 pulses: Pi(x), Pi(phi), 2Pi(3phi), Pi(phi)
            dur, amp, freq, ph, ch, pre_del = params[2:8]
            
<<<<<<< Updated upstream
            # BB1 Phase: arccos(-1/4) approx 104.5 deg
            phi_bb1 = np.arccos(-0.25)
=======
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
            # elif params[1] == "bir4":
            #     # Expected params list structure:
            #     # [type, 'bir4', duration, amplitude, frequency, phase, awg, delay, 
            #     #  dw0, beta, kappa, theta]
                
            #     envelope_type = params[1]
            #     duration      = params[2]
            #     amplitude     = params[3]
            #     frequency     = params[4]
            #     phase         = params[5]
            #     awg_channel   = params[6]
            #     pre_delay     = params[7]
                
            #     # Custom BIR4 params
            #     # dw0: FM scaling (rad/s), matches params['dw0'] in simulation
            #     dw0   = params[8]  
            #     beta  = params[9]
            #     kappa = params[10]
            #     # Theta (flip angle), default to pi if not provided
            #     theta = params[11] if len(params) > 11 else np.pi

            #     # Generate Waveform
            #     # Note: bir4_waveform returns [times, signal]
            #     time_arr, signal_arr = bir4_waveform(duration, dw0, beta, kappa, theta)
                
            #     # Create QCS Envelope
            #     bir4_env = qcs.ArbitraryEnvelope(time_arr, signal_arr)
                
            #     # Create Pulse
            #     # amplitude param scales the normalized signal
            #     pulse = qcs.RFWaveform(duration=duration, envelope=bir4_env,
            #                         amplitude=amplitude, rf_frequency=frequency,
            #                         instantaneous_phase=phase)
                                    
            #     program.add_waveform(pulse, awg_channel, pre_delay=pre_delay)
            elif params[1] == "bir4":
                # Params:
                # ['waveform','bir4', duration, amplitude, frequency, phase, awg_channel, pre_delay,
                #   dw0(rad/s), beta, kappa, theta(optional), omega_end_zero(optional)]
                duration     = params[2]
                amplitude    = params[3]
                frequency    = params[4]
                phase        = params[5]
                awg_channel  = params[6]
                pre_delay    = params[7]

                dw0          = params[8]   # rad/s (same as sim)
                beta         = params[9]
                kappa        = params[10]
                theta        = params[11] if len(params) > 11 else np.pi
                omega_end_zero = params[12] if len(params) > 12 else False

                time_arr, env_arr = bir4_envelope_from_sim(
                    tau=duration,
                    dw0=dw0,
                    beta=beta,
                    kappa=kappa,
                    theta=theta,
                    sample_rate=4.8e9,
                    normalize=True,          # keep max |env| = 1, use amplitude=amplitude to scale
                    omega_end_zero=omega_end_zero
                )

                bir4_env = qcs.ArbitraryEnvelope(time_arr, env_arr)
                pulse = qcs.RFWaveform(
                    duration=duration,
                    envelope=bir4_env,
                    amplitude=amplitude,
                    rf_frequency=frequency,
                    instantaneous_phase=phase
                )
                program.add_waveform(pulse, awg_channel, pre_delay=pre_delay)


            else:
                print("Pulse shape is unknown.")
>>>>>>> Stashed changes
            
            # Sequence definitions (Duration relative to pi-pulse)
            seq = [
                (dur,   0),             # Target Pi
                (dur,   phi_bb1),       # Pi(phi)
                (2*dur, 3*phi_bb1),     # 2Pi(3phi)
                (dur,   phi_bb1)        # Pi(phi)
            ]
            
            # Recursively add waveforms
            # First pulse gets the original pre_delay, others 0
            is_first = True
            for d, p_offset in seq:
                # Use 'square' or user preferred shape? BB1 usually square/hard pulses
                # We construct a param list for a square pulse
                sub_params = ['waveform', 'square', d, amp, freq, ph + p_offset, ch, pre_del if is_first else 0]
                add_pulses_to_program(program, [sub_params])
                is_first = False
            continue

        # Standard Processing
        duration, amplitude, frequency, phase, channel, pre_delay = params[2:8]
        extras = params[8:]

        if category == 'waveform':
            if env_type == 'square':
                pulse = qcs.RFWaveform(duration, qcs.ConstantEnvelope(), amplitude, frequency, phase)
            elif env_type == 'gaus':
                pulse = qcs.RFWaveform(duration, qcs.GaussianEnvelope(2), amplitude, frequency, phase)
            elif env_type == 'flat top gaus':
                rise = extras[0]
                base = qcs.GaussianEnvelope()
                dc_pulse = qcs.DCWaveform(rise*2, base, amplitude)
                pulse = dc_pulse.to_flattop(duration) # Note: DC pulse has no freq/phase
            else:
                # Arbitrary waveforms
                res = create_envelope(env_type, duration, extras)
                if isinstance(res, tuple): env, amp_scale = res
                else: env, amp_scale = res, 1.0
                
                pulse = qcs.RFWaveform(duration, env, amplitude * amp_scale, frequency, phase)
            
            program.add_waveform(pulse, channel, pre_delay=pre_delay)

        elif category == 'acquisition':
            if env_type == 'square': env = qcs.ConstantEnvelope()
            elif env_type == 'gaus': env = qcs.GaussianEnvelope(2)
            
            pulse = qcs.RFWaveform(duration, env, amplitude, frequency)
            filt = qcs.IntegrationFilter(pulse)
            program.add_acquisition(filt, channel, pre_delay=pre_delay)

def set_digitizer_range(mapper, dig_name, paras):
    digiChannel = mapper.get_physical_channels(dig_name)
    digiChannel[0].settings.range.value = paras["Digi_Range"]

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

def save_data(retValue, filepath, paras):
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

def dB2n(P_dBm, kappa_Hz, freq_Hz):
    """
    Convert Input Power (dBm) to Intracavity Photon Number (n).
    Assumes resonance condition and input through kappa port.
    Formula: n = 4 * P_in / (hbar * omega * kappa_ang)
    """
    hbar = 1.0545718e-34
    omega = 2 * np.pi * freq_Hz
    kappa_ang = 2 * np.pi * kappa_Hz
    
    P_watts = 10 ** ((P_dBm - 30) / 10)
    
    # n = |a|^2. For single port driving: n = 4 * P_in / (hbar * w * k)
    # Note: k in denominator must be angular rate (1/s)
    n = 4 * P_watts / (hbar * omega * kappa_ang)
    return n

def n2dB(n, kappa_Hz, freq_Hz):
    """
    Convert Intracavity Photon Number (n) to Input Power (dBm).
    """
    hbar = 1.0545718e-34
    omega = 2 * np.pi * freq_Hz
    kappa_ang = 2 * np.pi * kappa_Hz
    
    P_watts = n * (hbar * omega * kappa_ang) / 4
    
    if P_watts <= 0: return -np.inf
    P_dBm = 10 * np.log10(P_watts) + 30
    return P_dBm

def plot_spectrum(signal, sample_rate):
    """
    Plots the positive frequency spectrum of a signal.
    Handles 1D (time) or 2D (traces, time) arrays.
    """
    signal = np.array(signal)
    
    # Handle 1D case by expanding to 2D
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
        
    n_traces, n_samples = signal.shape
    freqs = np.fft.fftfreq(n_samples, 1/sample_rate)
    
    # Mask for positive frequencies
    pos_mask = freqs >= 0
    freqs_pos = freqs[pos_mask]
    
    plt.figure(figsize=(10, 6))
    
    for i in range(n_traces):
        fft_vals = np.fft.fft(signal[i])
        mag = np.abs(fft_vals[pos_mask])
        plt.plot(freqs_pos, mag, label=f'Trace {i}' if n_traces < 10 else None)
        
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (a.u.)")
    plt.title("Signal Spectrum (Positive Frequencies)")
    plt.grid(True, alpha=0.3)
    if n_traces < 10: plt.legend()
    plt.tight_layout()
    plt.show()