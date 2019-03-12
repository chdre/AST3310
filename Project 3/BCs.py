"""
Function to change the boundary conditions so that dux/dy = 0, duy/dy = 0 and
uy = 0 at the vertical boundaries.
Calling function within the loop to set the boundary conditions for each
time step.
"""

def BCs(t):
    uy[0,0,0] = 100
    uy[:,0,t] = 0                   #u_y at bottom
    uy[:,-1,t] = 0                  #u_y at top
    du_dy[:,0,t] = 0                #du_x/dy at bottom
    du_dy[:,-1,t] = 0               #du_x/dy at top
    T[:,-1,t] = 0.9*T_phot          #Temperature at bottom, 1.1 of the previous
    T[:,0,t] = 1.1*T_phot           #Temperature at bottom
    e[:,-1,t] = P_phot/(gamma - 1.)     #Energy at top
    rho[:,-1,t] = e[:,-1,t]*mu*m_u/((gamma - 1.)*k*T[:,-1,t])
    #drhou_dt_0 = g*rho[:,-1,t]  #eq. 6.10, no momentum passing out from star
