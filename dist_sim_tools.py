from step_function_methods import *
import concurrent.futures

def run_model(arg):
    q_arr, T, lambda_LM, sigma_m, sigma_w, eta, min_w_par = arg

    data_mat, w_dist_mat = run(T=T, alpha_2=0.25, N_good=6, lambda_LM=lambda_LM, sigma_m=sigma_m, sigma_w=sigma_w,
                               sigma_delta=0.0001, lambda_F=0.5, lambda_H=1.0, F=80, H=500, N_app=6, eta=eta,
                               min_w_par=min_w_par, W_u=1, Ah=1, tol=1e-14)

    q_mat = get_q_vals(q_arr, w_dist_mat)
    q_vals = q_mat[-400:, :].mean(axis=0)
    return q_vals


def run_mp(args):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        X = np.array(list(executor.map(run_model, args)))
    q_vals_mat = X
    return q_vals_mat

def counterfact_sim(seed, d_mwp_arr, n_sim, q_arr, T, lambda_LM, sigma_m, sigma_w, eta, min_w_par):
    results = np.zeros((d_mwp_arr.size, n_sim, q_arr.size))
    for i in range(d_mwp_arr.size):
        rd.seed(seed)
        set_seed(seed)
        args = [(q_arr, T, lambda_LM, sigma_m, sigma_w, eta, min_w_par+d_mwp_arr[i]) for j in range(n_sim)]
        q_vals_mat = run_mp(args)
        results[i, :, :] = q_vals_mat
    return results