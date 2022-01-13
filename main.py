from step_function_methods import *
import matplotlib.pyplot as plt
import time

T = 1000
periods = T
H = 250
change_t = 500

gamma_s_arr = np.array([0.4, 0.5])
q_arr = np.linspace(0.01, 1.0, 100)
d_mwp_arr = np.array([0.0, 0.2])
results = np.zeros((d_mwp_arr.size, q_arr.size, gamma_s_arr.size))

start = time.time()

for i in range(d_mwp_arr.size):
    for j in range(len(gamma_s_arr)):
        seed = 1231
        rd.seed(seed)
        set_seed(seed)
        w_dist_mat = run(T=T, alpha_2=0.25, N_good=6, lambda_LM=10, sigma_m=0.35, sigma_w=0.40, sigma_delta=0.1,
                         lambda_F=0.5, lambda_H=1.0, F=40, H=H, N_app=6, eta=1.5, gamma_s=gamma_s_arr[j],
                         min_w_par=1e-14, W_u=1, Ah=1, tol=1e-14, change_t=change_t, d_mwp=d_mwp_arr[i])

        q_mat = get_q_vals(q_arr, w_dist_mat)
        q_vals = q_mat[-100:,:].mean(axis=0)
        results[i,:,j] = q_vals

end = time.time()
print(end - start)
i = 1
print(gamma_s_arr[i])
plt.plot(results[1,:,i] - results[0,:,i])
plt.show()