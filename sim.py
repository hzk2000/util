import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la


def spin_matrix(s):
    upp_diag = np.array([np.sqrt((s+1)*(2*a-2)-(a-1)*a) for a in range(2,int(2*s)+2)])
    ss_x = (np.diag(upp_diag, k=1) + np.diag(upp_diag, k=-1))/2
    ss_y = (-1j*np.diag(upp_diag, k=1) + 1j*np.diag(upp_diag, k=-1))/2
    ss_z = np.diag(np.array([s+1-a for a in range(1,int(2*s)+2)]))
    id_s = np.eye(int(2*s)+1)
    return ss_x,ss_y,ss_z,id_s

def compute_tensor_spins(s1,s2):
    s_x, s_y, s_z, id_2 = spin_matrix(s1)
    ss_x, ss_y, ss_z, id_s = spin_matrix(s2)

    S_x = la.kron(s_x,id_s)
    S_y = la.kron(s_y,id_s)
    S_z = la.kron(s_z,id_s)

    I_x = la.kron(id_2,ss_x)
    I_y = la.kron(id_2,ss_y)
    I_z = la.kron(id_2,ss_z)

    return S_x, S_y, S_z, I_x, I_y, I_z

def get_ev(B_arr, s2 ,g ,A ,xi=0, s1=1/2):
    g_x,g_y,g_z = g

    S_x, S_y, S_z, I_x, I_y, I_z = compute_tensor_spins(s1,s2)
    A_x, A_y, A_z = A
    mu = 1.6021/9.1093837/4/np.pi
    eigenvalues = np.zeros((len(B_arr),len(I_x)))

    #eigenvalue computation of the hamiltonian
    for i,B in enumerate(B_arr):
        B_x,B_y, B_z = B*np.cos(xi), B*np.sin(xi), B*0
        H = (A_x*S_x@I_x+A_y*S_y@I_y+A_z*S_z@I_z) + mu*(g_x*B_x*S_x+g_y*B_y*S_y+g_z*B_z*S_z)

        eigenvalues[i],v = la.eigh(H)
    return B_arr,eigenvalues

# for c in plane sample

def ev_Yb_12(B_arr, xi):
    g = (3.92,1.05,3.92)
    A = (3.082,.788,3.082)
    _,ev = get_ev(B_arr, s2 = 1/2, g = g, A = A, xi = xi)
    return ev

def ev_Yb_52(B_arr, xi):
    g = (3.92,1.05,3.92)
    A = (-.851,-.216,-.851)
    _,ev = get_ev(B_arr, s2 = 5/2, g = g, A = A, xi = xi)
    return ev

def ev_Yb_0(B_arr, xi):
    g = (3.92,1.05,3.92)
    _,ev = get_ev(B_arr, s2 = 0, g = g, A = (0, 0, 0), xi = xi)
    return ev

def ev_Er_72(B_arr, xi):
    g = (1.247, 8.38, 8.38)
    A = (-.130, -.873,-.873)
    # g = (8.38,1.247,8.38) 
    # A = (-.873,-.130,-.873)
    _,ev = get_ev(B_arr, s2 = 7/2, g = g, A = A, xi = xi)
    return ev

def ev_Er_0(B_arr, xi):
    g = (1.247, 8.38, 8.38)
    _,ev = get_ev(B_arr, s2 = 0, g = g, A = (0, 0, 0), xi = xi)
    return ev

def ev_Nd_0(B_arr, xi):
    g = (2.52, 2.03, 2.52)
    _,ev = get_ev(B_arr, s2= 0, g = g, A = (0, 0, 0), xi = xi)
    return ev

#Nd 143 isotope -> I = 7/2
def ev_Nd_72(B_arr, xi):
    g = (2.52, 2.03, 2.52)
    A = [(-.220/(8/11))*x for x in g]   #   A_J = -220 MHz
    _,ev = get_ev(B_arr, s2= 7/2, g = g, A = A, xi = xi)
    return ev

def cal_ev(f_res,Bmin,Bmax,N_B,xi):
    blue_color = 'darkblue'
    red_color = 'crimson'
    B_arr = np.linspace(Bmin, Bmax, N_B)
    B_list=[]

    #Erbium 7/2
    ev = ev_Er_72(B_arr, xi=xi)
    indices = np.array([np.searchsorted(arr,f_res) for arr in (-ev[:,:8]+ev[:,:-9:-1]).T])

    plt.plot(B_arr,ev,color=red_color,alpha=0.7)
    [plt.arrow(B_arr[x],ev[x,i],0,ev[x,-1-i]-ev[x,i],length_includes_head=True,head_width=0.2,head_length=0.4,linewidth=1.5,color='black', label=('Er I=7/2' if i==0 else None)) for i,x in enumerate(indices[(indices>0) & (indices < N_B)])]

    for i,x in enumerate(indices[(indices>0) & (indices < N_B)]):
        B_list.append( B_arr[x] )

    #Erbium 0
    ev = ev_Er_0(B_arr, xi=xi)
    index = np.searchsorted((-ev[:,:1]+ev[:,-1:]).T[0],f_res)

    plt.plot(B_arr,ev, color=blue_color,linewidth=2.5)
    plt.arrow(B_arr[min(index,N_B-1)],ev[min(index,N_B-1),0],0,ev[min(index,N_B-1),1]-ev[min(index,N_B-1),0],length_includes_head=True,head_width=0.2,head_length=0.6,color=red_color,linestyle='-',linewidth=2, label='I=0 Er')

    B_list.append(B_arr[min(index,N_B-1)])

    plt.xlabel('B (mT)')
    plt.ylabel('E/h (GHz)')
    plt.title('Zeeman spliting energy levels \n '+r"Resonator 3 (6.561GHz), $\xi$ = "+f"{xi*180/np.pi:.1f}", fontsize = 12)
    plt.legend()
    plt.twinx()
    plt.show()

    return np.array(B_list)