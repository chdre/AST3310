import numpy as np
import matplotlib.pyplot as plt

plt.matplotlib.rcParams.update({'font.size': 12})
"""
When the script is executed, it will provide 4 plots (same as the ones in the
paper) if the below statement is set to True.
"""

plots = True

#Setup
nx = 300        #Number of horizontal gridpoints
ny = 100        #Number of radial gridpoints
dx = 12e6/nx    #Basing dx of off the minimum requirement of the x-axis, 12 Mm.
dy = 4e6/ny     #Same as dx, but with the y-axis limit 4 Mm. Notice that dx = dy
N = 10          #Time steps
p = 0.1

#Solar values
R_sun = 6.96e8              #[m]
M_sun = 1.989e30            #[kg]
T_phot = 5778               #Temperature solar photosphere, [K]
P_phot = 1.8e8              #pressure solar photosphere, [Pa]

#Constants
G = 6.67408e-11             #Gravitational constant [m^3/(kg^1s^2)]
g = -G*M_sun/R_sun**2       #Constant gravitational acc
mu = 0.61                   #Chemical potential, ideal gas
m_u = 1.660539e-27          #mass unit [kg]
k = 1.3806505e-23           #Boltzmanns constant [J/K]
c_v = 3.*k/(2.*mu*m_u)      #Heat capacity, constant volume
c_p = 5.*k/(2.*mu*m_u)      #Heat capacity, constant pressure
gamma = c_p/c_v

#Matrices
ux = np.zeros ((nx+1,ny+1,N+1))  #x componenent of velocity
uy = np.zeros_like(ux)           #y componenent of velocity
rhoux = np.zeros_like(ux)        #Momentum in x
rhouy = np.zeros_like(ux)        #Momentum in y
rho = np.zeros_like(ux)          #Density in y
T = np.zeros_like(ux)            #Temperature
P = np.zeros_like(ux)            #pressure
e = np.zeros_like(ux)            #Energy

def droll(matrix,delta,a):
    """
    Function to roll matrix hortizontally with given step delta. "a" for which
    axis to roll, 0 for y, 1 for xself. Return phi_{j-1} - phi_{j+1} since we
    are calculating downwards from j = ny.
    """
    rolled_back = np.roll(matrix,-1,axis=a)
    rolled_forward = np.roll(matrix,1,axis=a)

    return (rolled_back - rolled_forward)/(2.*delta)

def droll_vert(matrix,delta,last,first,a):
    """
    Function to roll matrix vertically with given step delta. Last and first are
    variables to set the elements rolled from the end/beginning of the matrix
    to the beginning/end (circular) to zero. Return phi_{j-1} - phi_{j+1} since
    we are calculating downwards from j = ny.
    """
    rolled_back = np.roll(matrix,-1,axis=a)
    rolled_forward = np.roll(matrix,1,axis=a)
    rolled_back[-1] = last
    rolled_forward[0] = first

    return (rolled_back - rolled_forward)/(2.*delta)

def roll_time(matrix,last,first,a):
    """
    For rolling the time derivatives. Last and first are variables to set the
    elements rolled from the end/beginning of the matrix to the beginning/end
    (so its not circular) to zero.
    """
    rolled_back = np.roll(matrix,-1,axis=a)
    rolled_forward = np.roll(matrix,1,axis=a)
    rolled_back[-1] = last
    rolled_forward[0] = first

    return rolled_forward + rolled_back


#Initial conditions
nab = 0.5                   #d(lnT)/d(lnP)
T[:,-1,0] = T_phot          #Temperature at the top of the box
rho[:,-1,0] = P_phot*mu*m_u/(k*T_phot)
P[:,-1,0] = P_phot
for j in range(ny):
    j = ny-j                            #Starting at the top, so we calculate j-1 for next value.
    dP_dy = -g*rho[0,j,0]               #Hydrostatic equilibrium
    dT_dy = nab*T[0,j,0]/P[0,j,0]*dP_dy #Initial condition for dT/dy, found from nabla
    P[:,j-1,0] = P[:,j,0] + dP_dy*dy
    T[:,j-1,0] = T[:,j,0] + dT_dy*dy
    rho[:,j-1,0] = P[:,j-1,0]*mu*m_u/(k*T[:,j-1,0])     #Equation of state
e[:,:,0] = rho[:,:,0]*k*T[:,:,0]/(mu*m_u)*(gamma - 1.)  #Internal energy

if plots == True:
    #Plotting initial conditions for P and rho as function of T
    plt.figure()
    plt.plot(T[0,:,0],rho[0,:,0])
    plt.xlabel('T [K]'); plt.ylabel('$\\rho$ [kg/m^2]')
    plt.title('Initial condition')
    plt.figure()
    plt.plot(T[0,:,0], P[0,:,0])
    plt.xlabel('T [K]'); plt.ylabel('P [Pa]')
    plt.title('Initial condition')
    plt.show()

def BCs(t):
    """
    Function to set the boundary conditions.
    Calling function within the loop to set the boundary conditions for each
    time step.
    """
    uy[:,0,t] = 0.0                 #u_y at bottom
    uy[:,-1,t] = 0.0                #u_y at top
    ux[:,0,t] = 0.0                 #u_x at bottom
    ux[:,-1,t] = 0.0                #u_x at top
    T[:,-1,t] = 0.9*T_phot          #Temperature at bottom, 1.1 of the previous
    T[:,0,t] = 1.1*T_phot           #Temperature at bottom
    e[:,-1,t] = (gamma - 1.)*rho[:,-1,t]*k*T[:,-1,t]/(mu*m_u)   #Internal energy at top
    e[:,0,t] = (gamma - 1.)*rho[:,0,t]*k*T[:,0,t]/(mu*m_u)      #Internal energy at bottom
    rho[:,-1,t] = e[:,-1,t]*mu*m_u/((gamma - 1.)*k*T[:,-1,t])   #Density at top
    rho[:,0,t] = e[:,0,t]*mu*m_u/((gamma - 1.)*k*T[:,0,t])      #Density at bottom

for t in range(0,N):
    BCs(t)
    #Gradients
    de_dx = droll(e[:,:,t],dx,1)                            #de/dx
    de_dy = droll_vert(e[:,:,t],dy,0,0,0)                   #de/dy
    drho_dx = droll(rho[:,:,t],dx,1)                        #d(rho)/dx
    drho_dy = droll_vert(rho[:,:,t],dy,0,0,0)               #d(rho)/dy
    dP_dy = droll_vert(P[:,:,t],dy,0,0,0)                   #dP/dy
    drhoux_dx = droll(rho[:,:,t]*ux[:,:,t],dx,1)            #d(rho ux)/dx
    drhoux_dy = droll_vert(rho[:,:,t]*ux[:,:,t],dy,0,0,0)   #d(rho ux)/dy
    drhouy_dy = droll_vert(rho[:,:,t]*uy[:,:,t],dy,0,0,0)   #d(rho uy)/dy
    drhouy_dx = droll(rho[:,:,t]*uy[:,:,t],dx,1)            #d(rho uy)/dx

    #BC's for the gradients
    drhoux_dy[:,-1] = drhouy_dy[:,-1] = 0.0     #No momentum passing out of the star at the top
    drhoux_dy[:,0] = drhouy_dy[:,0] = 0.0
    dP_dy[:,0] = -g*rho[:,0,t]                  #BC for hydrostatic equilibrium
    dP_dy[:,-1] = -g*rho[:,-1,t]                #BC for hydrostatic equilibrium


    """
    If to use roll to calculate velocity gradients, or simply dividing the
    momentum by the density.
    """
    roll_for_u = True   #True to use roll, False to divide by rho
    if roll_for_u == False:
        dux_dx = drhoux_dx/rho[:,:,t]
        dux_dy = drhoux_dy/rho[:,:,t]
        duy_dy = drhouy_dy/rho[:,:,t]
        duy_dx = drhouy_dx/rho[:,:,t]
    elif roll_for_u == True:
        dux_dx = droll(ux[:,:,t],dx,1)
        dux_dy = droll_vert(ux[:,:,t],dy,0,0,0)
        duy_dy = droll_vert(uy[:,:,t],dy,0,0,0)
        duy_dx = droll(uy[:,:,t],dx,1)
    #BC
    dux_dy[:,-1] = dux_dy[:,0] = 0.0        #BC for vertical gradient of the horizontal compoenent

    deux_dx = droll(ux[:,:,t]*e[:,:,t],dx,1)            #d(e*ux)/dx
    deux_dy = droll_vert(ux[:,:,t]*e[:,:,t],dx,0,0,0)   #d(e*ux)/dy
    deuy_dx = droll(uy[:,:,t]*e[:,:,t],dy,1)            #d(e*uy)/dx
    deuy_dy = droll_vert(uy[:,:,t]*e[:,:,t],dy,0,0,0)   #d(e*uy)/dy

    #DE's to solve
    drho_dt = -rho[:,:,t]*(dux_dx + dux_dy + duy_dx + duy_dy) - (2.*drho_dx +
                2.*drho_dy)*(ux[:,:,t]* + uy[:,:,t]) #d(rho)/dt, continuity equation
    #The two lines below sum to be the equation of motion d(rho u)/dt
    drhoux_dt = -rho[:,:,t]*ux[:,:,t]*(dux_dx + duy_dy) - (ux[:,:,t]*(rho[:,:,t]
                *dux_dx + ux[:,:,t]*drho_dx) + uy[:,:,t]*(rho[:,:,t]*dux_dy + ux[:,:,t]*drho_dy))   #dP/dx = 0
    drhouy_dt = -rho[:,:,t]*uy[:,:,t]*(dux_dx + duy_dy) - (ux[:,:,t]*(rho[:,:,t]
                *duy_dx + uy[:,:,t]*drho_dx) + uy[:,:,t]*(rho[:,:,t]*duy_dy + uy[:,:,t]*drho_dy)) - dP_dy + g*rho[:,:,t]
    de_dt = -(deux_dx + deux_dy + deuy_dx + deuy_dy) - P[:,:,t]*(deux_dx + deux_dy
            + duy_dx + duy_dy)     #Energy equation
    dux_dt = drhoux_dt/rho[:,:,t]
    duy_dt = (drhouy_dt - g*rho[:,:,t])/rho[:,:,t]

    for j in range(ny):   #y-direction
        """
        Calculating
        """
        j = ny-j
        rels = np.array([abs(drhoux_dt[0,j]/(rho[0,j,t]*ux[0,j,t])), abs(drhouy_dt[0,j]/(rho[0,j,t]*uy[0,j,t])),
                abs(drho_dt[0,j]/rho[0,j,t]), abs(de_dt[0,j]/e[0,j,t])])
        delta = np.nanmax(rels)
        dt = p/delta

        #Below is the method used in the previous paper to find a dt
        #dt_vals = [abs(p*rho[0,j,t]/drho_dt[0,j]), abs(p*e[0,j,t]/de_dt[0,j]),
        #            abs(p*ux[0,j,t]/dux_dt[0,j]), abs(p*uy[0,j,t]/duy_dt[0,j])]
        #dt = np.nanmin(dt_vals)

        rho[:,j,t+1] = 0.5*roll_time(rho[:,j,t],0,0,0) + drho_dt[:,j]*dt
        e[:,j,t+1] = 0.5*roll_time(e[:,j,t],0,0,0) + de_dt[:,j]*dt
        ux[:,j,t+1] = 0.5*roll_time(ux[:,j,t],0,0,0) + dux_dt[:,j]*dt
        uy[:,j,t+1] = 0.5*roll_time(uy[:,j,t],0,0,0) + duy_dt[:,j]*dt
        rhoux[:,j,t+1] = 0.5*roll_time(rho[:,j,t]*ux[:,j,t],0,0,0) + drhoux_dt[:,j]*dt
        rhouy[:,j,t+1] = 0.5*roll_time(rho[:,j,t]*uy[:,j,t],0,0,0) + drhouy_dt[:,j]*dt
        P[:,j,t+1] = e[:,j,t+1]/(gamma - 1.)
        T[:,j,t+1] = P[:,j,t+1]*mu*m_u/(k*rho[:,j,t+1]*(gamma - 1.))


def write_to_file(matrix,time_step):
    """
    Function to write values of array at time time_step to file. Calling the
    function will create a .txt-file with the name 'test' showing the values
    of the array for a given time_step. To display values for all t unlock.
    Call example: write_to_file(rho,0)
    """
    if time_step == True:
        thefile = open('test.txt', 'w')
        #for item in matrix[0,:,:]:         #To display values for all t swap this line with the one below.
        for item in matrix[0,:,time_step]:
            thefile.write("%s\n" % item)


if plots == True:
    #Plotting timestep 1
    plt.figure()
    plt.plot(T[0,:,1],rho[0,:,1])
    plt.xlabel('T [K]'); plt.ylabel('$\\rho$ [kg/m^2]')
    plt.title('Time step 1')
    plt.figure()
    plt.plot(T[0,:,1], P[0,:,1])
    plt.xlabel('T [K]'); plt.ylabel('P [Pa]')
    plt.title('Time step 1')
    plt.show()
