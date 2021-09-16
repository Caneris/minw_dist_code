from step_function_methods import *
import concurrent.futures
from dist_stim_switcher_module import run_dist_sim_ID

def run_model(arg):
    ID, q_arr, T, d_mwp, parval = arg

    data_mat, w_dist_mat = run_dist_sim_ID(ID, T, d_mwp, parval)

    q_mat = get_q_vals(q_arr, w_dist_mat)
    q_vals = q_mat[-400:, :].mean(axis=0)
    mean_data = data_mat[:,-400:].mean(axis=1)
    return mean_data, q_vals


def run_mp(args):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        mean_d, q = np.array(list(executor.map(run_model, args)))
    q_vals_mat = q
    mean_data_mat = mean_d
    return mean_data_mat, q_vals_mat

def counterfact_sim(seed, ID, d_mwp_arr, n_sim, q_arr, T, parval):
    results1 = np.zeros((d_mwp_arr.size, n_sim, q_arr.size)) # qvals
    results2 = np.zeros((d_mwp_arr.size, 23, n_sim)) # mean_data
    for i in range(d_mwp_arr.size):
        rd.seed(seed)
        set_seed(seed)
        args = [(ID, q_arr, T, d_mwp_arr[i], parval) for j in range(n_sim)]
        mean_data_mat, q_vals_mat = run_mp(args)
        results1[i, :, :] = q_vals_mat
    return results1, results2