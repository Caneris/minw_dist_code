from dist_sim_tools import *

import matplotlib.pyplot as plt
import concurrent.futures
import csv

if __name__ == '__main__':
    n_sim = 25
    seed = 123456
    T = 1000
    lambda_LM_arr = np.array([1, 5, 10, 15])
    sigma_m_arr = np.array([0.3, 0.35, 0.4, 0.45]) # 0.35
    sigma_w_arr = np.array([0.3, 0.35, 0.4, 0.45]) # 0.4
    N_app_arr = np.arange(1,11)
    N_good_arr = np.arange(1,11)
    gamma_s_arr = np.array([0.1, 0.2, 0.3, 0.4])
    min_w_par_arr = np.array([1e-14, 0.2, 0.4, 0.6, 0.8])
    d_mwp_arr = np.array([0.0, 0.1])
    q_arr = np.linspace(0.01, 1.0, 100)
    csv_names = np.array(['lambda_LM', 'min_w_par', 'sigma_w', 'sigma_m', 'N_app', 'N_good'])
    par_vals_list = [lambda_LM_arr, min_w_par_arr, sigma_w_arr, sigma_m_arr, N_app_arr, N_good_arr]
    for gamma_s_i in range(2,4):
        gamma_s = gamma_s_arr[gamma_s_i]
        print('gamma_s = {}'.format(gamma_s))
        for i in range(len(par_vals_list)):
            csv_name = csv_names[i]
            parvals = par_vals_list[i]
            for parval in parvals:
                print("start {} simulations for {} = {} ....".format(n_sim, csv_name, parval))
                results1, results2 = counterfact_sim(seed, i, d_mwp_arr, n_sim, q_arr, T, parval, gamma_s)
                diff = results1[1,:, :] - results1[0,:, :]
                diff = np.insert(diff, 0, parval, axis=1)
                results2 = np.insert(results2, 0, parval, axis=2)
                with open('dist_gamma_results_minwpar_04/dist_sim_qvals_{}_{}.csv'.format(csv_name, gamma_s_i), 'a', newline = '') as csvfile:
                    filewriter = csv.writer(csvfile, delimiter = ',')
                    filewriter.writerows(diff)
                for j in range(results2.shape[0]):
                    with open('dist_gamma_results_minwpar_04/dist_sim_mean_data_{}_{}_{}.csv'.format(csv_name, j, gamma_s_i), 'a', newline = '') as csvfile:
                        filewriter = csv.writer(csvfile, delimiter = ',')
                        filewriter.writerows(results2[j, :, :])
