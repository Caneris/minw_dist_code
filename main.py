from step_function_methods import *
import matplotlib.pyplot as plt
import time

seed = 12341
rd.seed(seed)
set_seed(seed)

T = 1000
periods = 300

start = time.time()
data_mat = run(T=T, alpha_2=0.25, N_good=4, lambda_LM=10, sigma_m=0.35, sigma_w=0.4, sigma_delta=0.0001, lambda_F=0.5,
               lambda_H=1.0, F=160, H=1000, N_app=4, eta=1.5, min_w_par=1e-14, Ah=1, tol=1e-14)
end = time.time()
print(data_mat[17, -400:].mean())
print(data_mat[18, -400:].mean())
print(data_mat[3, -400:].mean())
print(end - start)

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