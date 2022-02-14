from step_function_methods import *
import matplotlib.pyplot as plt
import time

T = 1000
periods = T
H = 250
F = 40
change_t = 500
n_sim = 10
gamma_s_arr = np.array([0.2])
q_arr = np.linspace(0.01, 1.0, 100)
d_mwp_arr = np.array([0.0, 0.1])
results = np.zeros((2, q_arr.size, n_sim))

params = {
    "sigma_w": 0.4,
    "sigma_m": 0.35,
    "lambda_LM": 10,
    "N": 6,
    "eta": 1.5,
    "min_w_par": 0.4,
    "d_mwp": 0.0
}

for i in range(d_mwp_arr.size):
    for j in range(n_sim):
        seed = 1231
        rd.seed(seed)
        set_seed(seed)
        params["min_w_par"] = d_mwp_arr[i]
        data_mat, w_dist_mat = run_changed_params(params)

        q_mat = get_q_vals(q_arr, w_dist_mat)
        q_vals = q_mat[-400:,:].mean(axis=0)
        results[i, :, j] = q_vals

q2 = results[1, 5:95, :].mean(axis = 1)
q1 = results[0, 5:95, :].mean(axis = 1)
plt.plot(q2 - q1)
plt.show()

# unemployment
plt.plot(np.arange(periods), data_mat[0, -periods:])
plt.show()

# nominal GDP
plt.plot(np.arange(periods), data_mat[1, -periods:])
plt.show()

# real GDP
plt.plot(np.arange(periods), data_mat[2, -periods:])
plt.show()

# mark up
plt.plot(np.arange(periods), data_mat[17, -periods:])
plt.show()

# mean prices
plt.plot(np.arange(periods), data_mat[3, -periods:])
plt.show()

# mean wages
plt.plot(np.arange(periods), data_mat[4, -periods:])
plt.show()

# median wages
plt.plot(np.arange(periods), data_mat[5, -periods:])
plt.show()

# mean real wages
plt.plot(np.arange(periods), data_mat[6, -periods:])
plt.show()

# Gini coefficient
plt.plot(np.arange(periods), data_mat[7, -periods:])
plt.show()

# Share of inactive firms
plt.plot(np.arange(periods), data_mat[8, -periods:])
plt.show()

# mean real wages unskilled
plt.plot(np.arange(periods), data_mat[10, -periods:])
plt.show()
# mean real wages skilled
plt.plot(np.arange(periods), data_mat[11, -periods:])
plt.show()
# ratio
plt.plot(np.arange(periods), data_mat[11, -periods:]/data_mat[10, -periods:])
plt.show()
# share refinanced
plt.plot(np.arange(periods), data_mat[9, -periods:])
plt.show()

# mean delta
plt.plot(np.arange(periods), data_mat[19, -periods:])
plt.show()


plt.plot(np.arange(periods), data_mat[18, -periods:])
plt.show()