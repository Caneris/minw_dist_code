import numpy as np
from step_function_methods import run
import concurrent.futures
import time

def Niederreiter(N=100, D=2, iStart=0):
    j = np.arange(1,D+1)
    nu = 2 ** (j/(D+1))
    i = np.arange(iStart+1,N+iStart+1).reshape(-1, 1)
    z = i * nu
    return z%1

def get_par_vals(y_row):
    N_app, lambda_LM, sigma_w, sigma_m, sigma_delta, alpha_2, lambda_exp = y_row
    return int(N_app), lambda_LM, sigma_w, sigma_m, sigma_delta, alpha_2, lambda_exp

def run_model(arg):
    y_row, nSim, sim_num = arg
    N_app, lambda_LM, sigma_w, sigma_m, sigma_delta, alpha_2, lambda_exp = get_par_vals(y_row)
    results = np.zeros((nSim, 3))
    for n in range(nSim):
        print("Simulation #{}".format(sim_num*nSim + n + 1))
        data_mat = run(alpha_2=alpha_2, lambda_LM=lambda_LM, sigma_m=sigma_m, sigma_w=sigma_w, sigma_delta=sigma_delta,
                       lambda_exp=lambda_exp, N_app=N_app)
        data = data_mat[[0, 3, 17], -200:].mean(axis=1)
        results[n, :] = data

    for n in range(nSim):
        print("Simulation #{}".format(sim_num*nSim + n + 1))
        data_mat = run(alpha_2=alpha_2, N_good=6, lambda_LM=lambda_LM, sigma_m=sigma_m, sigma_w=sigma_w,
                       sigma_delta=sigma_delta, lambda_exp=lambda_exp, N_app=N_app)
        price_mean2 = data_mat[[3, 17], -200:].mean(axis=1)
        results[n, [1,2]] -= price_mean2
    result = results.mean(axis=0)
    print(result)
    return list(result)

if __name__ == '__main__':

    nSim = 5 # number of simulations per parameter combination
    N = 125
    D = 7
    iStart = 0
    nRep = 5

    R = np.array([
        [3, 14], # N_app
        [1, 10], # lambda_LM
        [0.01, 0.5], # sigma_w
        [0.01, 0.5], # sigma_m
        [0.01, 0.5], # sigma_delta
        [0.1, 0.3], # alpha_2
        [0.2, 0.8] # lambda_exp
    ])
    R = np.transpose(R)

    y = (R[1] - R[0])/2
    xOpt = R[0] + y
    RC = np.zeros((2, D))
    RC[0,:], RC[1,:] = -y, y

    opt_arr = np.array([0.1, 0.1, 0.05])
    x = Niederreiter(int(N/nRep), D, iStart)

    start_time = time.time()
    for rep in range(nRep):
        par_vals_arr = np.minimum(xOpt + x*(RC[1] - RC[0]) + RC[0], R[1])
        par_vals_arr = np.maximum(par_vals_arr, R[0])

        with concurrent.futures.ProcessPoolExecutor() as executor:
            args = [(par_vals_arr[i], nSim, i) for i in range(len(par_vals_arr))]
            output_arr = np.array(list(executor.map(run_model, args)))

        dist_arr = np.linalg.norm(output_arr - opt_arr, axis=1)
        iOpt = np.argmin(dist_arr)
        xOpt, fOpt = par_vals_arr[iOpt], output_arr[iOpt]
        RC = RC/2

        print("Rep number: {} done".format(rep + 1))
        print("xOpt: {}, fOpt: {}".format(xOpt, fOpt))

    end_time = time.time() - start_time
    print(f'rep.: fOpt = {xOpt} at {fOpt}')
    print("{} simulations took {} seconds...".format(N*nSim, end_time))
