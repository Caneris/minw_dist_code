from dist_sim_tools import *

import matplotlib.pyplot as plt
import concurrent.futures
import csv

if __name__ == '__main__':

    n_sim = 25
    seed = 123456
    lambda_LM_arr = np.array([1, 5, 10, 15])
    sigma_m_arr = np.array([0.3, 0.35, 0.4, 0.45]) # 0.35
    sigma_w_arr = np.array([0.3, 0.35, 0.4, 0.45]) # 0.4
    N_arr = np.array[4, 8, 16, 32]
    gamma_s_arr = np.array([0.1, 0.2, 0.3, 0.4])
    min_w_par_arr = np.array([1e-14, 0.2, 0.4, 0.6, 0.8])
    d_mwp_arr = np.array([0.0, 0.1])
    q_arr = np.linspace(0.01, 1.0, 100)
    par_names = np.array(['lambda_LM', 'min_w_par', 'sigma_w', 'sigma_m', 'N'])
    par_vals_list = [lambda_LM_arr, min_w_par_arr, sigma_w_arr, sigma_m_arr, N_arr]

    sim_par_keys = np.array(['min_w_par', 'eta'])

    min_w_par_vals = np.array([1e-14, 0.2, 0.4, 0.6])
    min_w_par_path_ids = np.array(['00', '02', '04', '06'])

    eta_par_vals = np.array([0.5, 1.5, 2.0, 2.5])
    eta_par_path_ids = np.array(['05', '15', '20', '25'])

    sim_par_vals_list = [min_w_par_vals, eta_par_vals]
    sim_par_path_ids_list = [min_w_par_path_ids, eta_par_path_ids]

    def_params = {
        "sigma_w": 0.4,
        "sigma_m": 0.35,
        "lambda_LM": 10,
        "N": 6,
        "eta": 1.5,
        "min_w_par": 0.4,
        "d_mwp": 0.0
    }

    for spk_i in range(len(sim_par_keys)):
        sim_par_key = sim_par_keys[spk_i]
        sim_par_vals = sim_par_vals_list[spk_i]
        sim_par_path_ids = sim_par_path_ids_list[spk_i]
        for spv_i in range(len(sim_par_vals)):
            sim_par_val = sim_par_vals[spv_i]
            sim_par_path_id = sim_par_path_ids[spv_i]
            for i in range(len(par_vals_list)):
                par_name = par_names[i]
                parvals = par_vals_list[i]
                for parval in parvals:
                    print("start {} simulations for {} = {} ....".format(n_sim, par_name, parval))
                    results1, results2 = counterfact_sim(seed, par_name, d_mwp_arr, n_sim, q_arr, parval,
                                                         def_params, sim_par_key, sim_par_val)

                    diff = results1[1,:, :] - results1[0,:, :]
                    diff = np.insert(diff, 0, parval, axis=1)
                    results2 = np.insert(results2, 0, parval, axis=2)
                    with open('ofat_results/{}_res_{}/dist_sim_qvals_{}.csv'.format(sim_par_key, sim_par_path_id,
                                                                                    par_name), 'a', newline = '') as csvfile:
                        filewriter = csv.writer(csvfile, delimiter = ',')
                        filewriter.writerows(diff)

                    for j in range(results2.shape[0]):
                        with open('ofat_results/{}_res_{}/dist_sim_mean_data_{}_{}csv'.format(sim_par_key,
                                                                                              sim_par_path_id,
                                                                                              par_name,
                                                                                              j), 'a', newline = '') as csvfile:
                            filewriter = csv.writer(csvfile, delimiter = ',')
                            filewriter.writerows(results2[j, :, :])
