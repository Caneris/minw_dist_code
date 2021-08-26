from step_function_methods import *
import matplotlib.pyplot as plt

T = 1000
periods = T
H = 1000
F = 160
n_sim = 5

q_arr = np.linspace(0.0, 1.0, 21)
d_mwp_arr = np.array([0.0, 0.1])
results = np.zeros((d_mwp_arr.size, n_sim, q_arr.size))

for j in range(d_mwp_arr.size):
    seed = 12312
    rd.seed(seed)
    set_seed(seed)

    for i in range(n_sim):
        data_mat, w_dist_mat = run(T=T, alpha_2=0.25, N_good=4, lambda_LM=10, sigma_m=0.35, sigma_w=0.4, sigma_delta=0.0001,
                                   lambda_F=0.5, lambda_H=1.0, F=F, H=H, N_app=4, eta=1.5, min_w_par=0.4,
                                   d_min_w_par=d_mwp_arr[j], change_t=600, W_u=1, Ah=1, tol=1e-14)

        q_mat = get_q_vals(q_arr, w_dist_mat)
        q_vals = q_mat[-200:, :].mean(axis=0)
        results[j, i, :] = q_vals


plt.plot((results[1, :, :] - results[0, :, :]).mean(axis=0))
plt.show()
results[1, :, -1] - results[0, :, -1]