import numpy as np
import numpy.random as rd
import time
from multiprocessing import Pool
from setting_generator import pick_element
from switcher_module import run_OFAT_ID
import csv


def run_nc(args):
    ID, NC, eta, T, par_vals, par_name, eta_i = args
    print('start simulation {} with NC = {}'.format(ID, NC))
    run_perms(ID, NC, eta, T, par_vals, par_name, eta_i)


def run_perms(ID, NC, eta, T, par_vals, par_name, eta_i):

    with open('OFAT{}_{}.csv'.format(eta_i, par_name), 'w+', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([par_name,
                             "unemployment_rate", "nominal_GDP", "real_GDP", "mean_price",
                             "mean_wage", "median_real_wage", "mean_real_wage", "gini_coefficient",
                             "share_of_inactive", "share_of_refinanced", "unskilled_wage", "skilled_wages",
                             "wage_ratio_s_to_u", "C", "DC", "total_expenditure", "A_ratio_FtoH",
                             "mean_m", "mean_uc", "mean_delta", "tot_inv",
                             "unemployment_rate_sd", "nominal_GDP_sd", "real_GDP_sd", "mean_price_sd",
                             "mean_wage_sd", "median_real_wage_sd", "mean_real_wage_sd", "gini_coefficient_sd",
                             "share_of_inactive_sd", "share_of_refinanced_sd", "unskilled_wage_sd", "skilled_wages_sd",
                             "wage_ratio_s_to_u_sd", "C_sd", "DC_sd", "total_expenditure_sd", "A_ratio_FtoH_sd",
                             "mean_m_sd", "mean_uc_sd", "mean_delta_sd", "tot_inv_sd"])

    perm_seq = rd.permutation(NC)
    for j in range(NC):
        print("start ID: {} sim: {} ....".format(ID, j))
        num = perm_seq[j]
        i = pick_element(par_vals, num)[0]
        data = run_OFAT_ID(ID, T, par_vals[i])
        data = np.concatenate((data[:, -400:].mean(axis=1), data[:, -400:].std(axis=1)))
        data = np.insert(data, [0], par_vals[i])

        with open('OFAT{}_{}.csv'.format(eta_i, par_name), 'a', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(data)


def run_nc_with_mp(args_arr):

    start_time = time.time()

    p = Pool()
    p.map(run_nc, args_arr)

    p.close()
    p.join()

    end_time = time.time() - start_time
    print("Simulating {} mc simulations took {} time using mp".format(len(args_arr), end_time))


if __name__ == '__main__':

    N_app_arr = np.array([2, 3, 6, 12]).astype(int)
    N_good_arr = np.array([2, 3, 6, 12]).astype(int)
    lambda_LM_arr = np.array([1, 2, 5, 8, 10])
    sigma_w_arr = np.array([0.275, 0.3, 0.325, 0.35, 0.4])
    min_w_par_arr = np.array([1e-14, 0.2, 0.4, 0.6, 0.8])

    par_list = [N_app_arr, N_good_arr, lambda_LM_arr, sigma_w_arr, min_w_par_arr]
    par_names = ["N_app", "N_good", "lambda_LM", "sigma_w", "min_w_par"]
    etas = np.array([1.5])

    # Number of periods per simulation
    T = 1000
    # Number of replications (cores)
    NR = 5
    # number of simulations per parameter value
    nsim = 50
    # number of cases per parameter value array
    NC = [len(par_list[i])*nsim for i in range(len(par_list))]
    for eta_i in range(len(etas)):
        args_arr = [(ID, NC[ID], etas[eta_i], T, par_list[ID], par_names[ID], eta_i) for ID in range(NR)]
        print(args_arr)
        run_nc_with_mp(args_arr)