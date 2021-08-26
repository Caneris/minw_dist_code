from step_function_methods import *
import matplotlib.pyplot as plt
import time
import seaborn as sns
from scipy.stats import shapiro

T = 1000
periods = T
H = 1000

q_arr = np.linspace(0.01, 1.0, 100)
d_mwp_arr = np.array([0.0, 0.2])
results = np.zeros((2, q_arr.size))

start = time.time()

for i in range(d_mwp_arr.size):
    seed = 1231232
    rd.seed(seed)
    set_seed(seed)
    data_mat, w_dist_mat = run(T=T, alpha_2=0.25, N_good=4, lambda_LM=10, sigma_m=0.35, sigma_w=0.45, sigma_delta=0.0001,
                               lambda_F=0.5, lambda_H=1.0, F=160, H=H, N_app=4, eta=1.5, min_w_par=0.4,
                               d_min_w_par=d_mwp_arr[i], change_t=600, W_u=1, Ah=1, tol=1e-14)

    q_mat = get_q_vals(q_arr, w_dist_mat)
    q_vals = q_mat[-100:,:].mean(axis=0)
    results[i,:] = q_vals

end = time.time()
print(end - start)

plt.plot(results[1] - results[0])
plt.show()


sns.distplot(w_dist_mat[250, w_dist_mat[250] > 0])
plt.show()

sorted_wd_mat = np.sort(w_dist_mat)
test = sorted_wd_mat[-250:, :]
test_mean = test.mean(axis=0)

# seed = 1234123
# rd.seed(seed)
# set_seed(seed)
#
# data_mat, w_dist_mat = run(T=T, alpha_2=0.25, N_good=4, lambda_LM=10, sigma_m=0.35, sigma_w=0.4,
#                            sigma_delta=0.0001, lambda_F=0.5, lambda_H=1.0, F=160, H=1000, N_app=4,
#                            eta=1.5, min_w_par=0.4, Ah=1, tol=1e-14, d_min_w_par=0.0, change_t=500, t_after_minw=100,
#                            W_u=1)
#
# sns.distplot(w_dist_mat[0, w_dist_mat[0] > 0])
# plt.show()
#
# sns.distplot(w_dist_mat[1, w_dist_mat[1] > 0])
# plt.show()
#
# w_dist1 = np.log(w_dist_mat[0, w_dist_mat[0] > 0])
# w_dist2 = np.log(w_dist_mat[1, w_dist_mat[1] > 0])
# q_vals11 = get_q_vals(q_arr, w_dist1)
# q_vals22 = get_q_vals(q_arr, w_dist2)
# plt.plot(q_vals22 - q_vals11)
# plt.show()
#
# plt.plot(q_vals2 - q_vals22)
# plt.show()

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