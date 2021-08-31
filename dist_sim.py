from step_function_methods import *
import matplotlib.pyplot as plt
import concurrent.futures
import csv

T = 1000
periods = T
H = 1000
F = 160
n_sim = 4

q_arr = np.linspace(0.0, 1.0, 101)
d_mwp_arr = np.array([0.0, 0.1])
results = np.zeros((d_mwp_arr.size, n_sim, q_arr.size))

# for j in range(d_mwp_arr.size):
#     seed = 12312
#     rd.seed(seed)
#     set_seed(seed)
#
#     for i in range(n_sim):
#         data_mat, w_dist_mat = run(T=T, alpha_2=0.25, N_good=4, lambda_LM=10, sigma_m=0.35, sigma_w=0.4, sigma_delta=0.0001,
#                                    lambda_F=0.5, lambda_H=1.0, F=F, H=H, N_app=4, eta=1.5, min_w_par=0.4,
#                                    d_min_w_par=d_mwp_arr[j], change_t=600, W_u=1, Ah=1, tol=1e-14)
#
#         q_mat = get_q_vals(q_arr, w_dist_mat)
#         q_vals = q_mat[-400:, :].mean(axis=0)
#         results[j, i, :] = q_vals
#
#
# plt.plot((results[1, :, :] - results[0, :, :]).mean(axis=0))
# plt.show()



def run_model(arg):
    q_arr, T, lambda_LM, sigma_m, sigma_w, eta, min_w_par, d_mwp, change_t = arg

    data_mat, w_dist_mat = run(T=T, alpha_2=0.25, N_good=6, lambda_LM=lambda_LM, sigma_m=sigma_m, sigma_w=sigma_w,
                               sigma_delta=0.0001, lambda_F=0.5, lambda_H=1.0, F=80, H=500, N_app=6, eta=eta,
                               min_w_par=min_w_par, d_min_w_par=d_mwp, change_t=change_t, W_u=1, Ah=1,
                               tol=1e-14)

    q_mat = get_q_vals(q_arr, w_dist_mat)
    q_vals = q_mat[-400:, :].mean(axis=0)
    return q_vals

def run_mp(args):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        X = np.array(list(executor.map(run_model, args)))
    q_mean = X.mean(axis=0)
    return q_mean

args = [(123456, 1000, 5, 0.35, 0.4, 1.5, 0.4, 0.0, 600) for i in range(n_sim)]

if __name__ == '__main__':
    n_sim = 50
    seed = 123456
    d_mwp_arr = np.array([0.0, 0.1])
    q_arr = np.linspace(0.0, 1.0, 101)
    results = np.zeros((d_mwp_arr.size, q_arr.size))
    for i in range(d_mwp_arr.size):
        rd.seed(seed)
        set_seed(seed)
        args = [(q_arr, 1000, 10, 0.35, 0.4, 1.5, 0.4+d_mwp_arr[i], 0.0, 600) for j in range(n_sim)]
        q_mean = run_mp(args)
        results[i, :] = q_mean

    diff = results[1,:] - results[0,:]
    plt.plot(diff)
    plt.show()
    with open('quantile_diff.csv', 'w', newline = '') as csvfile:
        filewriter = csv.writer(csvfile, delimiter = ',')
        filewriter.writerow(diff)