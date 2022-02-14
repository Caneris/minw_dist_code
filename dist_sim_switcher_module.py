from step_function_methods import run

def ID0(T, d_mwp, par_val, gamma_s):
    data_mat, w_dist_mat = run(T=T, lambda_LM=par_val, gamma_s=gamma_s, d_mwp=d_mwp)
    return data_mat, w_dist_mat

def ID1(T, d_mwp, par_val, gamma_s):
    data_mat, w_dist_mat = run(T=T, gamma_s=gamma_s, min_w_par=par_val, d_mwp=d_mwp)
    return data_mat, w_dist_mat

def ID2(T, d_mwp, par_val, gamma_s):
    data_mat, w_dist_mat = run(T=T, sigma_w=par_val, gamma_s=gamma_s, d_mwp=d_mwp)
    return data_mat, w_dist_mat

def ID3(T, d_mwp, par_val, gamma_s):
    data_mat, w_dist_mat = run(T=T, sigma_m=par_val, gamma_s=gamma_s, d_mwp=d_mwp)
    return data_mat, w_dist_mat

def ID4(T, d_mwp, par_val, gamma_s):
    data_mat, w_dist_mat = run(T=T, N_app=par_val, gamma_s=gamma_s, d_mwp=d_mwp)
    return data_mat, w_dist_mat

def ID5(T, d_mwp, par_val, gamma_s):
    data_mat, w_dist_mat = run(T=T, N_good=par_val, gamma_s=gamma_s, d_mwp=d_mwp)
    return data_mat, w_dist_mat

switcher = {
    0: ID0,
    1: ID1,
    2: ID2,
    3: ID3,
    4: ID4,
    5: ID5
}

def run_dist_sim_ID(ID, T, d_mwp, par_val, gamma_s):
    func = switcher.get(ID, "no such ID...")
    return func(T, d_mwp, par_val, gamma_s)