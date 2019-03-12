import numpy as np
import matplotlib.pyplot as plt
from project1_opacity import logT_arr, logkpa_arr, logR_arr
from project0 import eps_func
from scipy import interpolate

plt.matplotlib.rcParams.update({'font.size': 16})

"""
The program will call function solver (with dynamic step length True) and
it will call plot_func_r which will plot as a function of r. This can be hashed
out at line 209 and 210.
"""

#Solar values
R_sun = 6.96e8              #[m]
M_sun = 1.989e30            #[kg]
L_sun = 3.846e26            #[W]
rho_sun_avg = 1.408e3       #[kgm^-3]

#Initial parameters
L_0 = 1.0*L_sun               #[W]
R_0 = 0.72*R_sun*0.543       #[m]
M_0 = 0.8*M_sun               #[g]
rho_0 = 5.1*rho_sun_avg*0.63 #[kgm^-3]
T_0 = 5.7e6*1.5              #[K]

"""
#Values for model produces unphysical density. Unlocked at own risk.
T_0 = 5.7e6*0.40
rho_0 = 5.1*rho_sun_avg*1.61431
R_0 = 0.72*R_sun*0.4837
"""

X = 0.7
Y = 0.29
Z = 0.01

#Constants
mu = 1./(2*X+3*Y/4.+Z/2.)   #avg. particle mass
m_u = 1.660539e-27          #mass unit [kg]
k = 1.3806505e-23           #Boltzmanns constant [J/K]
sigma = 5.670367e-8         #Stefan-Boltzmann constant [W/(m^2K^4)]
c = 3e8                     #Speed of light [m/s]
a = 4.*sigma/c              #Radiation constant [J/(m^3K^4)]
G = 6.67408e-11             #Gravitational constant [m^3/(kg^1s^2)]

#Calculating the opacity, kappa
def kpa_func(rho_val,T_val):
    logR_val = np.log10(rho_val*1e-3/(T_val*1e-6)**3)  #Rho from SI to cgs by multiplying with 1e-3, and log10
    kpa_ip = interpolate.interp2d(logR_arr,logT_arr,logkpa_arr)
    #return 10**kpa_ip(rho_val,T_val)/10.                   #For kappa_test, replace this with the return statement below
    return 10**float(kpa_ip(logR_val,np.log10(T_val)))/10.  #Returns non-log SI


#Functions to calculate the density rho and the pressure P
rho_func = lambda P_val,T_val: P_val*mu*m_u/(k*T_val)
P_func = lambda rho_val,T_val: rho_val*k*T_val/(mu*m_u) + a*T_val**4/3.

dm = -1e-4*M_0
p = 0.0001
dynamic_stepl = True
if dynamic_stepl == True:   #fewer steps if euler solver
    N = 100000
else:
    N = abs(int(M_0/dm))


#Arrays for calculations
r = np.zeros(N+1)
P = np.zeros(N+1)
L = np.zeros(N+1)
T = np.zeros(N+1)
m = np.zeros(N+1)
rho = np.zeros(N+1)
eps = np.zeros(N+1)

#Initial conditions
r[0] = R_0
P[0] = P_func(rho_0,T_0)
L[0] = L_0
T[0] = T_0
m[0] = M_0
rho[0] = rho_0
eps[0] = eps_func(T_0,rho_0)

#Differential equations
dr_dm = lambda r_val,rho_val: 1./(4.*np.pi*r_val**2*rho_val)
dP_dm = lambda m_val,r_val: -G*m_val/(4.*np.pi*r_val**4)
dL_dm = lambda T_val,rho_val: eps_func(T_val,rho_val)
dT_dm = lambda L_val,r_val,T_val,rho_val: -3.*kpa_func(rho_val,T_val)*L_val/(256.
                                        *np.pi**2*sigma*r_val**4*T_val**3)

"""
Variable steplength and euler solver
"""
def solver(dm_in,p):        #dm_in is the dm value for non-dynamic steplength
    i = 0       #Counter to see how many indices are needed to plot
    if dynamic_stepl == True:
        while m[i] > 0 and i+1 <= N:
            #Finding a value for dm, which is the smallest of the calculated dm's
            dm_vals = [abs(p*r[i]/dr_dm(r[i],rho[i])), abs(p*P[i]/dP_dm(m[i],r[i])),
                       abs(p*L[i]/dL_dm(T[i],rho[i])), abs(p*T[i]/dT_dm(L[i],r[i],
                       T[i],rho[i]))]
            dm = -min(dm_vals)

            r[i+1] = r[i] + dr_dm(r[i],rho[i])*dm
            P[i+1] = P[i] + dP_dm(m[i],r[i])*dm
            L[i+1] = L[i] + dL_dm(T[i],rho[i])*dm
            T[i+1] = T[i] + dT_dm(L[i],r[i],T[i],rho[i])*dm

            m[i+1] = m[i] + dm
            rho[i+1] = (P[i+1] - a*T[i+1]**4/3.)*mu*m_u/(k*T[i+1])
            eps[i+1] = eps_func(T[i], rho[i])

            i += 1
        print 'm:', m[i]/M_0, 'rho:', rho[i]/rho_0, 'r:', r[i]/R_0
        print 'P:', P[i]/P[0], 'L:', L[i]/L_0, 'T:', T[i]/T_0, 'eps:', eps[i]/eps[0]
        return i

    elif dynamic_stepl == False:
        dm = dm_in
        while m[i] > 0 and i+1 <= N:
            P[i+1] = P[i] + dP_dm(m[i],r[i])*dm
            r[i+1] = r[i] + dr_dm(r[i],rho[i])*dm
            L[i+1] = L[i] + dL_dm(T[i],rho[i])*dm
            T[i+1] = T[i] + dT_dm(L[i],r[i],T[i],rho[i])*dm

            m[i+1] = m[i] + dm
            rho[i+1] = (P[i+1] - a*T[i+1]**4/3.)*mu*m_u/(k*T[i+1])
            eps[i+1] = eps_func(T[i], rho[i])

            i += 1
        print 'm:', m[i]/M_0, 'rho:', rho[i]/rho_0, 'r:', r[i]/R_0
        print 'P:', P[i]/P[0], 'L:', L[i]/L_0, 'T:', T[i]/T_0, 'eps:', eps[i]/eps[0]
        return i

"""
Following are the functions for plotting, and they are added only as a formality
to show how this was done.
"""
if __name__ =='__main__':
    def plot_func_m(ind):
        if abs(m[1]) > 0:
            plt.figure()
            plt.plot(m[:ind]/M_0, r[:ind]/R_sun)
            plt.xlabel('m/$M_0$'); plt.ylabel('r/$R_0$')
            plt.title('r(m)')

            plt.figure()
            plt.plot(m[:ind]/M_0, T[:ind]*1e-6)
            plt.xlabel('m/$M_0$'); plt.ylabel('T1e-6K')
            plt.title('T(m)')

            plt.figure()
            plt.plot(m[:ind]/M_0, L[:ind]/L_sun)
            plt.xlabel('m/$M_0$'); plt.ylabel('L/$L_{sun}$')
            plt.title('L(m)')

            plt.figure()
            plt.semilogy(m[:ind]/M_0, rho[:ind]/rho_0)
            plt.xlabel('m/$M_0$'); plt.ylabel('log($\\rho$/$\\bar{\\rho}_{sun})$')
            #plt.axis([0, 1, 1, 10])
            plt.title('$\\rho$(m)')

            plt.figure()
            plt.plot(m[:ind]/M_0,r[:ind]/R_0)
            plt.xlabel('m'); plt.ylabel('r')
            plt.title('r(m)')
            plt.show()
        else:
            print 'Need to calculate arrays m, r, P, T, rho'

    def plot_func_r(ind):
        if abs(m[1]) > 0:
            plt.figure()
            plt.plot(r[:ind]/R_0,T[:ind]/T_0)
            plt.xlabel('r/$R_0$'); plt.ylabel('T/$T_0$')
            plt.title('Temperature')

            plt.figure()
            plt.plot(r[:ind]/R_0,L[:ind]/L_0)
            plt.xlabel('r/$R_0$'); plt.ylabel('L/$L_0$')
            plt.title('Luminosity')

            plt.figure()
            plt.semilogy(r[:ind]/R_0,eps[:ind]/eps[0])
            plt.xlabel('r/$R_0$'); plt.ylabel('$\\epsilon$/$\\epsilon_0$')
            plt.title('Total energy')

            plt.figure()
            plt.semilogy(r[:ind]/R_0, rho[:ind]/rho_0)
            plt.xlabel('r/$R_0$'); plt.ylabel('$\\rho$/$\\rho_0$')
            plt.title('Density')

            plt.figure()
            plt.semilogy(r[:ind]/R_0, P[:ind]/P[0])
            plt.xlabel('r/$R_0$'); plt.ylabel('P/$P_0$')
            plt.title('Pressure')

            plt.figure()
            plt.plot(r[:ind]/R_0,m[:ind]/M_0)
            plt.xlabel('r/$R_0$'); plt.ylabel('m/$M_0$')
            plt.title('Mass')
            plt.show()
        else:
            print 'Need to calculate arrays m, r, P, T, rho'

    index = solver(dm,p)   #Number of indices to plot (index_counter)
    plot_func_r(index)

    dms_variation = False       #set to true if one wants to run test of values for step length
    if dms_variation == True:    #For testing values of dm
        dms = -M_0*np.array([1e-1,1e-2,5e-2,1e-3,5e-3,1e-4])
        for i in dms:
           imax = solver(i,p)
           plt.plot(m[:imax]/M_0,r[:imax]/R_0)
           plt.xlabel('m/$M_0$'); plt.ylabel('r/$R_0$')
           plt.title('Relative radius as function of mass')
        plt.legend(['-1e-1*M_0', '-1e-2*M_0','5e-2*M_0', '-1e-3*M_0',
                    '-5e-3*M_0', '-1e-4*M_0'])
        plt.show()

    def exp_values():
        """
        Made for plotting and printing end values when experimenting with the
        initial conditions for the radius, density, temperature and pressure. It
        is only included to show.
        """
        Rs = R_0*np.array([1./5, 1./4, 1./3, 1./2, 1., 2, 3, 4, 5])#([0.45, 0.5, 0.65, 0.70, 0.75, 0.8, 1.])
        Ts = T_0*np.array([1./5, 1./4, 1./3, 1./2, 1., 2, 3, 4, 5])#([1.8, 2, 3, 3.5, 4.1, 4.5, 4.8, 5])
        rhos = rho_0*np.array([1./5, 1./4, 1./3, 1./2, 1., 2, 3, 4, 5])
        Ps = P[0]*np.array([1./5, 1./4, 1./3, 1./2, 1., 2, 3, 4, 5])
        for i in R_0s:
            R_0 = i                         #Setting initial condition to that of the array above
            r[0] = R_0
            P[0] = P_func(rho_0,T_0)        #Used when experimenting with other values than P[0] (see line above). If P[0], please remove.
            eps[0] = eps_func(T_0,rho_0)
            index = solver(dm,p)
            plt.plot(r[:index]/R_0,L[:index]/L_0)
            #plt.plot(r[:index]/R_0,m[:index]/M_0)
            plt.xlabel('r'); plt.ylabel('m')
            plt.title('m(r)')
            print 'm:', m[index], 'rho:', rho[index], 'r:', r[index], 'P:', P[index], 'L:', L[index], 'T:', T[index], 'eps:', eps[index]
        plt.legend(['$P_0$/5','$P_0$/4','$P_0$/3' '$P_0$/2', '$P_0$','2$P_0$', '3$P_0$', '4$P_0$', '5$P_0$'])
        plt.show()


    """
    Following are the tests made for the program.
    """

    #Kappa_test
    """
    logR_test = np.array([-6.0,-5.95,-5.8,-5.7,-5.55,-5.95,-5.95,-5.95,-5.8,-5.75,-5.7,-5.55,-5.50])
    logT_test = np.array([3.750,3.755,3.755,3.755,3.755,3.770,3.780,3.795,3.770,3.775,3.780,3.795,3.800])
    for i in range(len(logR_test)):
        print kpa_func((logR_test[i]),logT_test[i])
    """

    #Test for rho and P
    """
    print rho_func(P_func(rho_0,T_0),T_0), rho_0
    print P_func(rho_0,T_0)
    """






#
