import numpy as np
from sys import exit


# H = 200, F = 20, u_r = 0.08, mu_u = 1, W_r = 1, gamma_s = 0.33,
#                  m = 0.1, sigma = 0.5, delta = 1, alpha_2 = 0.25


def calibrate_model(H = 200, F = 20, Ah = 1, u_r = 0.08, mu_u = 0.3, W_u = 1, gamma_s = 0.33, m = 0.1, eta = 0.5, delta = 1, alpha_2 = 0.1):

    # get elasticity parameter
    rho = (eta-1)/eta

    if mu_u < 0:
        print("\nSorry, but mu_u has to be between 0 and 1.")
        exit()

    mu_s = 1 - mu_u

    # 1. get Omega, mu_s, W_s
    Omega = (1-gamma_s)/gamma_s
    X_1 = mu_s/mu_u
    W_s = (X_1**rho)*(Omega**(1-rho))*W_u

    # 2. get Nr, Nnr
    Nnr = np.round(H*(1-u_r)/(1+Omega))
    Nr = np.round(Omega*Nnr)

    # 3. get y
    Y = ((mu_u*Nr)**rho + (mu_s*Nnr)**rho)**(1/rho)
    y_f = Y/F

    # 4. get uc, p
    uc = (Nr*W_u + Nnr*W_s)/Y
    p = (1+m)*uc

    AF = uc * Y
    Af = AF / F

    # 5. get pi, DIV
    Pi = m*(Nr*W_u + Nnr*W_s)
    pi_f = Pi/F
    DIV = delta*Pi

    div_f = DIV/F
    div_h = DIV/H

    # get I, c, C, alpha_1, AF
    AH = H*Ah
    I = Nr*W_u + Nnr*W_s + DIV
    alpha_1 = (Y * p - alpha_2 * AH) / I

    c = Y/H # individual steady state consumption

    # print("rho: {}, mu_s: {}, W_s: {}, Ah: {}, Af: {}, uc: {}, p: {}, y_f: {}".format(rho, mu_s, W_s, Ah, Af, uc, p, y_f))
    # print("pi_f: {}, div_h: {}, div_f: {}, c: {}, alpha_1: {}, Nr: {}, Nnr: {}".format(pi_f, div_h, div_f, c, alpha_1, Nr, Nnr))

    return mu_s, W_s, Af, uc, p, y_f, pi_f, div_h, div_f, c, alpha_1






