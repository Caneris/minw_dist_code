import numpy.random as rd
import time
from multiprocessing import Pool
from step_function_methods import run
import csv

def run_nc(args):
    ID, NC, T = args
    print('start simulation ID = {} with NC = {}'.format(ID, NC))
    run_perms(ID, NC, T)

def run_perms(ID, NC, T):
    for j in range(NC):

        print("start ID: {} sim: {} ....".format(ID, j))

        N_app = rd.randint(2, 11)
        N_good = rd.randint(2, 11)
        lambda_LM = rd.uniform(0.1, 5)
        sigma_adj = rd.uniform(0.1, 0.6)
        min_w_par = rd.uniform(0.1, 0.6)

        data_mat = run(T=T, N_good=N_good, lambda_LM=lambda_LM, N_app=N_app, min_w_par=min_w_par, Ah=10)

        with open('pretest_unemp_ID{}.csv'.format(ID), 'a', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(data_mat[0, :])

        with open('pretest_gini_ID{}.csv'.format(ID), 'a', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(data_mat[3, :])

        with open('pretest_price_ID{}.csv'.format(ID), 'a', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(data_mat[7, :])

        print("ID: {} sim: {} finished".format(ID, j))


def run_nc_with_mp(args_arr):

    start_time = time.time()

    p = Pool()
    p.map(run_nc, args_arr)

    p.close()
    p.join()

    end_time = time.time() - start_time
    print("Simulating {} mc simulations took {} time using mp".format(len(args_arr), end_time))


if __name__ == '__main__':

    # Number of periods per simulation
    T = 1000
    # Number of replications (cores)
    NR = 10
    # number of cases
    NC = 100
    args_arr = [(ID, NC, T) for ID in range(NR)]
    run_nc_with_mp(args_arr)
