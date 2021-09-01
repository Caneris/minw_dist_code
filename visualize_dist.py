import numpy as np
import matplotlib.pyplot as plt

my_data = np.genfromtxt('quantile_diff.csv', delimiter=',')
q_arr = np.linspace(0.0, 1.0, 101)
plt.title("Distributional effects of a minimum wage increase")
plt.ylabel("log($\\frac{Q2(p)}{Q1(p)}$", labelpad=-4)
plt.xlabel("p")
plt.plot(q_arr, my_data)
plt.savefig("preview_results.pdf")
plt.show()
