from step_function_methods import *
import concurrent.futures
from dist_stim_switcher_module import run_dist_sim_ID

def run_model(arg):
    ID, q_arr, T, d_mwp, parval, min_w_base = arg

    w_dist_mat = run_dist_sim_ID(ID, T, d_mwp, parval, min_w_base)

    q_mat = get_q_vals(q_arr, w_dist_mat)
    q_vals = q_mat[-100:, :].mean(axis=0)
    return q_vals


def run_mp(args):
    X = []
    for arg in args:
        x = run_model(arg)
        X.append(x)
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     X = np.array(list(executor.map(run_model, args)), dtype=object)
    X1 = np.stack(np.asarray(X), axis=0)
    return X1

def counterfact_sim(seed, ID, d_mwp_arr, n_sim, q_arr, T, parval, min_w_base):
    results1 = np.zeros((d_mwp_arr.size, n_sim, q_arr.size)) # qvals
    for i in range(d_mwp_arr.size):
        rd.seed(seed)
        set_seed(seed)
        args = [(ID, q_arr, T, d_mwp_arr[i], parval, min_w_base) for j in range(n_sim)]
        q_vals_mat = run_mp(args)
        results1[i, :, :] = q_vals_mat
    return results1