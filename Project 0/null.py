import numpy as np
import matplotlib.pyplot as plt

"""
Constants
"""

def eps_func(T,rho):
    X = 0.7; Y = 0.29; Y_3 = 1e-10; Z = 0.01; Z_Li = 1e-7; Z_Be = Z_Li
    m_u = 1.660539e-27
    N_A = 6.022e23
    T9 = T*1e-9

    """
    Particle density
    """
    n_e = rho*(1+X)/(2.*m_u)
    n_p = X*rho/m_u
    n_He = Y*rho/(4.*m_u)
    n_He3 = Y_3*rho/(3.*m_u)
    n_Z_Li = Z_Li*rho/(7.*m_u)     #3 electrons
    n_Z_Be = Z_Be*rho/(7.*m_u)     #4 electrons

    """
    Lambdas
    """

    lambda_pp = (4.01e-15*T9**(-2./3.)*np.exp(-3.380*T9**(-1./3.))*(1 + 0.123*T9**\
                (1./3.) + 1.09*T9**(2./3.) + 0.938*T9))/(1e6*N_A)

    lambda_33 = (6.04e10*T9**(-2./3.)*np.exp(-12.276*T9**(-1./3.))*(1. + 0.034*\
                T9**(1./3.) - 0.522*T9**(2./3.) - 0.124*T9 + 0.353*T9**(4./3.)\
                + 0.213*T9**(-5./3.)))/(1e6*N_A)

    lambda_34 = (5.61e6*(T9/(1 + 4.95e-2*T9))**(5./6.)*T9**(-3./2.)*np.exp(-12.826\
                *(T9/(1 + 4.95e-2*T9))**(-1./3.)))/(1e6*N_A)

    lambda_7e = (1.34e-10*T9**(-1./2.)*(1 - 0.537*T9**(1./3.) + 3.86*T9**(2./3.)\
                + 0.0027*T9**(-1.)*np.exp(2.515e-3*T9**(-1.))))/(1e6*N_A)

    lambda_prime_17 = (1.096e9*T9**(-2./3.)*np.exp(-8.472*T9**(-1./3.)) - 4.830e8*\
                    (T9/(1 + 0.759*T9))**(5./6.)*T9**(-3./2.)*np.exp(
                    -8.472*(T9/(1 + 0.759*T9))**(-1./3.)) + 1.06e10*T9**(-3./2.)*\
                    np.exp(-30.442*T9**(-1.)))/(1e6*N_A)

    lambda_17 = (3.11e5*T9**(-2./3.)*np.exp(-10.262*T9**(-1./3.)) + 2.53e3*T9**\
                (-3./2.)*np.exp(-7.306*T9**(-1.)))/(1e6*N_A)


    """
    Values of Q
    """
    Q_pp = 1.17*1.602e-13         #[J]
    Q_dp = 5.49*1.602e-13         #[J]
    Q_33 = 12.86*1.602e-13        #[J]
    Q_34 = 1.59*1.602e-13         #[J]
    Q_7e = 0.05*1.602e-13         #[J], 0.05 MeV since neutrino is lost
    Q_71_prime = 17.35*1.602e-13  #[J]
    Q_71 = 0.14*1.602e-13         #[J]

    #PP1
    r_pp = n_p**2*lambda_pp/(rho*2)
    r_33 = n_He3**2*lambda_33/(rho*2)

    #PP2
    r_34 = n_He3*n_He*lambda_34/rho
    r_7e = n_Z_Be*n_e*lambda_7e/rho
    r_71_prime = n_Z_Li*n_p*lambda_prime_17/rho

    #PP3
    r_71 = n_Z_Be*n_p*lambda_17/rho

    """
    If the sum of the reactions rates producing He3 and He4 is larger than the
    reaction rate that produces H, there will not be enough He4 for there to be
    a reaction between He3 and He4 producing Be7. We therefore need to scale the
    reaction rate of r_33, producing He4
    """
    if r_33 + r_34 > r_pp:
        """
        Multiplying the above equation by r_33/(r_33 + r_44) as to scale r_33,
        and require that this is equal (This works because we are working with
        steps in integers of reactions). The same goes for r_34 which is multiplied
        by r_34*r_pp/(r_33 + r_34).
        """
        r_33_temp = r_33*r_pp/(r_33 + r_34)
        r_34 = r_34*r_pp/(r_33 + r_34)
        r_33 = r_33_temp


    """
    Now we consider the sum of the reaction rates of r_7e and r_71. Both of the
    reactions require Be7 to produce respectively Li7 and B8. We therefore scale
    the reaction rate r_34, that produces Be7.
    """
    if r_7e + r_71 > r_34:
        """
        We now need to scale both r_7e and r_71. Multiplying the equations
        respectively by r_7e/(r_7e + r_71) and r_71/(r_7e + r_71) and setting
        equality
        """
        r_7e_temp = r_34*r_7e/(r_7e + r_71)
        r_71 = r_34*r_71/(r_7e + r_71)
        r_7e = r_7e_temp

    """
    r_71_prime requires a Li7, which is dependant on r_7e.
    """
    if r_71_prime > r_7e:
        """
        Multiplying by r_7e/(r_34 + r_7e) and setting equal
        """
        r_71_prime = r_7e



    eps_r_pp = r_pp*(Q_pp+Q_dp)*rho
    eps_r_33 = r_33*Q_33*rho
    eps_r_34 = r_34*Q_34*rho
    eps_r_7e = r_7e*Q_7e*rho
    eps_r_71_prime = r_71_prime*Q_71_prime*rho
    eps_r_71 = r_71*Q_71*rho

    eps = eps_r_pp + eps_r_33 + eps_r_34 + eps_r_7e + eps_r_71_prime + eps_r_71

    return eps

T_core = 1.57e7
T2 = 1e8
rho = 1.62e5

func(T2,rho)
