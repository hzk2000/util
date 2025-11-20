import numpy as np
import pyvisa
import time

def call_mag():
    rm = pyvisa.ResourceManager()
    mag_z = rm.open_resource('TCPIP0::192.168.112.131::7180::SOCKET', read_termination = "\n", write_termination = "\n")
    mag_y = rm.open_resource('TCPIP0::192.168.112.130::7180::SOCKET', read_termination = "\n", write_termination = "\n")
    mag_x = rm.open_resource('TCPIP0::192.168.112.129::7180::SOCKET', read_termination = "\n", write_termination = "\n")
    return [mag_x,mag_y,mag_z]

# def wake_mag(mag_x,mag_y,mag_z):
#     for i in range(3):
#         mag_x.query("PS?")
#         mag_y.query("PS?")
#         mag_z.query("PS?")
#     for i in range(3):
#         mag_x.query("FIELD:MAG?")
#         mag_y.query("FIELD:MAG?")
#         mag_z.query("FIELD:MAG?")

def to_new_axis(v, phi = 0, psi = 0):# computes the resontaor X'',Y'',Z'' coordinates given the X,Y,Z magnet coordinates which is determined by phi and psi
    v = np.array(v)
    rot1 = np.array([[np.cos(phi),0,np.sin(phi)],[0,1,0],[-np.sin(phi),0,np.cos(phi)]])
    vp = rot1@v     # this is the X',Y',Z' coordinates
    rot2 = np.array([[1,0,0],[0,np.cos(psi),-np.sin(psi)],[0,np.sin(psi),np.cos(psi)]])
    vpp = rot2@vp
    return vpp

def from_new_axis(vpp, phi = 0, psi = 0):   # computes the magnet X,Y,Z coordinates given the X'',Y'',Z'' coordinates in the shifted frame corresponding to the aligned resonator
    vpp = np.array(vpp)                        # which is determined by phi and psi
    rot2 = np.array([[1,0,0],[0,np.cos(psi),np.sin(psi)],[0,-np.sin(psi),np.cos(psi)]])
    vp = rot2@vpp       # this is the X',Y',Z' coordinates
    rot1 = np.array([[np.cos(phi),0,-np.sin(phi)],[0,1,0],[np.sin(phi),0,np.cos(phi)]])
    v = rot1@vp
    return v

def ramp_X(B0, Mag, rate=0.0002, xi=0, phi=0, psi=0, step=0, Max_B=[0.5,0.5,0.1], heating_time = 30, epsilon=1e-6):
    import time as tm
    mag_x,mag_y,mag_z = Mag
    u = from_new_axis([np.cos(xi), np.sin(xi), 0], phi=phi, psi=psi)
    u /= np.linalg.norm(u)  

    curr_Bx = float(mag_x.query("FIELD:MAG?")[:-2])
    curr_By = float(mag_y.query("FIELD:MAG?")[:-2])
    curr_Bz = float(mag_z.query("FIELD:MAG?")[:-2])
    curr_B_vector = np.array([curr_Bx, curr_By, curr_Bz])
    curr_B_scalar = np.dot(curr_B_vector, u)
    if abs(curr_B_scalar) < epsilon:
        curr_B_scalar=0

    delta_B = B0 - curr_B_scalar

    if step <= 0 or delta_B == 0:
        steps = [B0]
    else:
        num_steps = int(np.ceil(abs(delta_B) / step))
        steps = np.linspace(curr_B_scalar, B0, num_steps + 1)[1:] 

    for B_target in steps:
        Bx_target, By_target, Bz_target = B_target * u

        if abs(Bx_target) > Max_B[0] or abs(By_target) > Max_B[1] or abs(Bz_target) > Max_B[2]:
            raise Exception('Field too high')

        delta_Bx = Bx_target - curr_Bx
        delta_By = By_target - curr_By
        delta_Bz = Bz_target - curr_Bz
        delta_B_total = np.sqrt(delta_Bx**2 + delta_By**2 + delta_Bz**2)

        if delta_B_total == 0:
            continue

        delta_u = [delta_Bx / delta_B_total, delta_By / delta_B_total, delta_Bz / delta_B_total]

        rate_max_x = rate_max_y = rate_max_z = rate 

        rate_x = rate * delta_u[0]
        rate_y = rate * delta_u[1]
        rate_z = rate * delta_u[2]

        scales = []
        for rate_i, rate_max_i in zip([rate_x, rate_y, rate_z], [rate_max_x, rate_max_y, rate_max_z]):
            if rate_i != 0:
                scale = min(1, rate_max_i / abs(rate_i))
                scales.append(scale)
            else:
                scales.append(1)

        overall_scale = min(scales)
        rate_x *= overall_scale
        rate_y *= overall_scale
        rate_z *= overall_scale

        rate_used = rate * overall_scale

        if not (mag_x.query("PS?") == '1\r' and mag_y.query("PS?") == '1\r' and mag_z.query("PS?") == '1\r'):
            mag_x.write("PS 1")
            mag_y.write("PS 1")
            mag_z.write("PS 1")
            tm.sleep(heating_time + 1)

        mag_x.write("CONF:RAMP:RATE:UNITS 0")
        mag_y.write("CONF:RAMP:RATE:UNITS 0")
        mag_z.write("CONF:RAMP:RATE:UNITS 0")

        mag_x.write(f"CONF:FIELD:TARG {Bx_target:.10f}")
        mag_x.write(f"CONF:RAMP:RATE:FIELD 1, {abs(rate_x):.10f}, {Bx_target:.10f}")
        mag_x.write("RAMP")

        mag_y.write(f"CONF:FIELD:TARG {By_target:.10f}")
        mag_y.write(f"CONF:RAMP:RATE:FIELD 1, {abs(rate_y):.10f}, {By_target:.10f}")
        mag_y.write("RAMP")

        mag_z.write(f"CONF:FIELD:TARG {Bz_target:.10f}")
        mag_z.write(f"CONF:RAMP:RATE:FIELD 1, {abs(rate_z):.10f}, {Bz_target:.10f}")
        mag_z.write("RAMP")

        time_x = abs(delta_Bx) / abs(rate_x) if rate_x != 0 else 0
        time_y = abs(delta_By) / abs(rate_y) if rate_y != 0 else 0
        time_z = abs(delta_Bz) / abs(rate_z) if rate_z != 0 else 0

        sleep_time = max(time_x, time_y, time_z)

        print(f'Ramping to {B_target*1e3:.3f} mT along the specified direction.')
        print(f'Sleeping for : {sleep_time:.2f} seconds')

        tm.sleep(sleep_time + 1)  

        # mag_x.write("PAUSE")
        # mag_y.write("PAUSE")
        # mag_z.write("PAUSE")
        # tm.sleep(10)  

        curr_Bx = Bx_target
        curr_By = By_target
        curr_Bz = Bz_target

# two yokogawa 1. USB0::0x0B21::0x0039::90YC41697::0::INSTR 2. USB0::0x0B21::0x0039::90YC41697::0::INSTR
def set_dc_voltage(target_voltage,address= 'USB0::0x0B21::0x0039::90YC41697::0::INSTR', step=0.0001e-3):
    min_step = 0.000001
    if abs(step) < min_step:
        step = min_step if step > 0 else -min_step
    
    rm = pyvisa.ResourceManager()
    dc = rm.open_resource(address)
    
    try:
        idn = dc.query("*IDN?")
        print(f"Connected to: {idn}")
    except pyvisa.VisaIOError:
        print("Failed to connect to the device.")
        return

    current_voltage = float(dc.query(":SOUR:LEV?"))
    print(f"Current current: {current_voltage} V")
    
    step = abs(step) if target_voltage > current_voltage else -abs(step)
    
    while (step > 0 and current_voltage < target_voltage) or (step < 0 and current_voltage > target_voltage):
        dc.write(f":SOUR:LEV {current_voltage}")
        # print(f"Voltage set to {current_voltage} V")
        
        current_voltage += step
        
        if (step > 0 and current_voltage > target_voltage) or (step < 0 and current_voltage < target_voltage):
            current_voltage = target_voltage
        
        time.sleep(0.2)

    dc.write(f":SOUR:LEV {target_voltage}")
    print(f"Final current set to {target_voltage} A")
    
    dc.close()
    rm.close()


def ramp_X_new(B0, Mag, rate=0.0002, xi=0, phi=0, psi=0, step=0, Max_B=[0.5,0.5,0.1], heating_time = 30, epsilon=1e-6):
    import time as tm
    mag_x,mag_y,mag_z = Mag
    u = from_new_axis([np.cos(xi), np.sin(xi), 0], phi=phi, psi=psi)
    u /= np.linalg.norm(u)  

    # print("opps")
    # tm.sleep(1)
    mag_x.stdin.write(b'FIELD:MAG?\n'); mag_x.stdin.flush(); curr_Bx = eval(mag_x.stdout.readline().decode())
    mag_y.stdin.write(b'FIELD:MAG?\n'); mag_y.stdin.flush(); curr_By = eval(mag_y.stdout.readline().decode())
    mag_z.stdin.write(b'FIELD:MAG?\n'); mag_z.stdin.flush(); curr_Bz = eval(mag_z.stdout.readline().decode())
    tm.sleep(1)

    curr_B_vector = np.array([curr_Bx, curr_By, curr_Bz])
    print(curr_B_vector)
    curr_B_scalar = np.dot(curr_B_vector, u)
    if abs(curr_B_scalar) < epsilon:
        curr_B_scalar=0

    delta_B = B0 - curr_B_scalar

    if step <= 0 or delta_B == 0:
        steps = [B0]
    else:
        num_steps = int(np.ceil(abs(delta_B) / step))
        steps = np.linspace(curr_B_scalar, B0, num_steps + 1)[1:] 

    for B_target in steps:
        print(steps)
        Bx_target, By_target, Bz_target = B_target * u

        if abs(Bx_target) > Max_B[0] or abs(By_target) > Max_B[1] or abs(Bz_target) > Max_B[2]:
            raise Exception('Field too high')

        delta_Bx = Bx_target - curr_Bx
        delta_By = By_target - curr_By
        delta_Bz = Bz_target - curr_Bz
        delta_B_total = np.sqrt(delta_Bx**2 + delta_By**2 + delta_Bz**2)

        if delta_B_total == 0:
            continue

        delta_u = [delta_Bx / delta_B_total, delta_By / delta_B_total, delta_Bz / delta_B_total]

        rate_max_x = rate_max_y = rate_max_z = rate 

        rate_x = rate * delta_u[0]
        rate_y = rate * delta_u[1]
        rate_z = rate * delta_u[2]

        scales = []
        for rate_i, rate_max_i in zip([rate_x, rate_y, rate_z], [rate_max_x, rate_max_y, rate_max_z]):
            if rate_i != 0:
                scale = min(1, rate_max_i / abs(rate_i))
                scales.append(scale)
            else:
                scales.append(1)

        overall_scale = min(scales)
        rate_x *= overall_scale
        rate_y *= overall_scale
        rate_z *= overall_scale

        rate_used = rate * overall_scale

        
        # mag_x.stdin.write(b'PS?\n'); mag_x.stdin.flush(); PS_Bx = eval(mag_x.stdout.readline().decode())
        # mag_y.stdin.write(b'PS?\n'); mag_y.stdin.flush(); PS_By = eval(mag_y.stdout.readline().decode())
        # mag_z.stdin.write(b'PS?\n'); mag_z.stdin.flush(); PS_Bz = eval(mag_z.stdout.readline().decode())
        # print("here?")

        # if not (PS_Bx == 1 and PS_By == 1 and PS_Bz == 1):
        #     mag_x.stdin.write(b'PS 1\n'); mag_x.stdin.flush()
        #     mag_y.stdin.write(b'PS 1\n'); mag_y.stdin.flush()
        #     mag_z.stdin.write(b'PS 1\n'); mag_z.stdin.flush()
        #     tm.sleep(heating_time + 1)

        mag_x.stdin.write(b'CONF:RAMP:RATE:UNITS 0\n'); mag_x.stdin.flush()
        mag_y.stdin.write(b'CONF:RAMP:RATE:UNITS 0\n'); mag_y.stdin.flush()
        mag_z.stdin.write(b'CONF:RAMP:RATE:UNITS 0\n'); mag_z.stdin.flush()

        mag_x.stdin.write(f'CONF:FIELD:TARG {Bx_target:.10f}\n'.encode('utf-8') ); mag_x.stdin.flush()
        mag_x.stdin.write(f"CONF:RAMP:RATE:FIELD 1, {abs(rate_x):.10f}, {Bx_target:.10f}".encode('utf-8') ); mag_x.stdin.flush()
        mag_x.stdin.write("RAMP".encode('utf-8') ); mag_x.stdin.flush()


        mag_y.stdin.write(f'CONF:FIELD:TARG {By_target:.10f}\n'.encode('utf-8') ); mag_y.stdin.flush()
        mag_y.stdin.write(f"CONF:RAMP:RATE:FIELD 1, {abs(rate_y):.10f}, {By_target:.10f}".encode('utf-8') ); mag_y.stdin.flush()
        mag_y.stdin.write("RAMP".encode('utf-8') ); mag_y.stdin.flush()
        
        mag_z.stdin.write(f'CONF:FIELD:TARG {Bz_target:.10f}\n'.encode('utf-8') ); mag_z.stdin.flush()
        mag_z.stdin.write(f"CONF:RAMP:RATE:FIELD 1, {abs(rate_z):.10f}, {Bz_target:.10f}".encode('utf-8') ); mag_z.stdin.flush()
        mag_z.stdin.write("RAMP".encode('utf-8') ); mag_z.stdin.flush()

        time_x = abs(delta_Bx) / abs(rate_x) if rate_x != 0 else 0
        time_y = abs(delta_By) / abs(rate_y) if rate_y != 0 else 0
        time_z = abs(delta_Bz) / abs(rate_z) if rate_z != 0 else 0

        sleep_time = max(time_x, time_y, time_z)

        print(f'Ramping to {B_target*1e3:.3f} mT along the specified direction.')
        print(f'Sleeping for : {sleep_time:.2f} seconds')

        tm.sleep(sleep_time + 1)  

        curr_Bx, curr_By, curr_Bz = Bx_target, By_target, Bz_target

def call_mag_new():
    import subprocess

    mag_x = subprocess.Popen(
        ['C:\\Program Files\\American Magnetics, Inc\\Magnet-DAQ\\Magnet-DAQ.exe', '-h', '-p', '-a=192.168.112.129'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    time.sleep(1)
    mag_y = subprocess.Popen(
        ['C:\\Program Files\\American Magnetics, Inc\\Magnet-DAQ\\Magnet-DAQ.exe', '-h', '-p', '-a=192.168.112.130'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    time.sleep(1)
    mag_z = subprocess.Popen(
        ['C:\\Program Files\\American Magnetics, Inc\\Magnet-DAQ\\Magnet-DAQ.exe', '-h', '-p', '-a=192.168.112.131'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    time.sleep(10)
    return [mag_x,mag_y,mag_z]

def exit_mag(mag_x,mag_y,mag_z):
    mag_x.stdin.write(b'*CLS\n'); mag_x.stdin.flush()
    mag_y.stdin.write(b'*CLS\n'); mag_y.stdin.flush()
    mag_z.stdin.write(b'*CLS\n'); mag_z.stdin.flush()

    mag_x.stdin.write(b'EXIT\n'); mag_x.stdin.flush()
    mag_y.stdin.write(b'EXIT\n'); mag_y.stdin.flush()
    mag_z.stdin.write(b'EXIT\n'); mag_z.stdin.flush()
    import time as tm
    tm.sleep(2)