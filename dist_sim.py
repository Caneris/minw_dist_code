from dist_sim_tools import *

import matplotlib.pyplot as plt
import concurrent.futures
import csv

if __name__ == '__main__':
    n_sim = 8
    seed = 123456
    T = 1000
    lambda_LM_arr = np.array([1, 2.5, 5, 7.5, 10, 12.5, 15])
    sigma_m_arr = np.arange(0.25, 0.45, 0.05) # 0.35
    sigma_w_arr = np.arange(0.25, 0.45, 0.05) # 0.4
    N_app_arr = np.arange(1, 10)
    N_good_arr = np.arange(1, 10)
    eta = 1.5
    min_w_par_arr = np.array([1e-14, 0.2, 0.4, 0.6, 0.8, 0.9])
    d_mwp_arr = np.array([0.0, 0.1])
    q_arr = np.linspace(0.0, 1.0, 101)
    csv_names = np.array(['lambda_LM', 'min_w_par', 'sigma_w', 'sigma_m', 'N_app', 'N_good'])
    par_vals_list = [lambda_LM_arr, min_w_par_arr, sigma_w_arr, sigma_m_arr, N_app_arr, N_good_arr]
    for i in range(len(par_vals_list)):
        csv_name = csv_names[i]
        parvals = par_vals_list[i]
        for parval in parvals:
            print("start {} simulations for {} = {} ....".format(n_sim, csv_name, parval))
            results1, results2 = counterfact_sim(seed, i, d_mwp_arr, n_sim, q_arr, T, parval)
            diff = np.log(results1[1,:, :]) - np.log(results2[0,:, :])
            diff = np.insert(diff, 0, parval, axis=1)
            with open('dist_sim_results/dist_sim_qvals_{}.csv'.format(csv_name), 'a', newline = '') as csvfile:
                filewriter = csv.writer(csvfile, delimiter = ',')
                filewriter.writerows(diff)
            for j in range(results2.shape[0]):
                with open('dist_sim_results/dist_sim_mean_data_{}_{}.csv'.format(csv_name, j), 'a', newline = '') as csvfile:
                    filewriter = csv.writer(csvfile, delimiter = ',')
                    filewriter.writerow(["unemployment_rate", "nominal_GDP", "real_GDP", "mean_price",
                                         "mean_wage", "median_real_wage", "mean_real_wage", "gini_coefficient",
                                         "share_of_inactive", "share_of_refinanced", "unskilled_wage", "skilled_wages",
                                         "wage_ratio_s_to_u", "C", "DC", "total_expenditure", "A_ratio_FtoH",
                                         "mean_m", "mean_uc", "mean_delta", "tot_inv", "ur_unskilled",
                                         "ur_skilled"])
                    filewriter.writerows(results2[j, :, :].T)
