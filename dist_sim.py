from dist_sim_tools import *

import matplotlib.pyplot as plt
import concurrent.futures
import csv

if __name__ == '__main__':
    n_sim = 2
    seed = 123456
    T = 500
    lambda_LM_arr = np.arange(1, 16, 2)
    sigma_m = 0.35
    sigma_w = 0.4
    eta = 1.5
    min_w_par = 0.4
    d_mwp_arr = np.array([0.0, 0.1])
    q_arr = np.linspace(0.0, 1.0, 101)
    for lambda_LM in lambda_LM_arr:

        results = counterfact_sim(seed, d_mwp_arr, n_sim, q_arr, T, lambda_LM, sigma_m, sigma_w, eta, min_w_par)
        diff = results[1,:, :] - results[0,:, :]
        with open('quantile_diff_lambda_LM{}.csv'.format(lambda_LM), 'w', newline = '') as csvfile:
            filewriter = csv.writer(csvfile, delimiter = ',')
            filewriter.writerows(diff)