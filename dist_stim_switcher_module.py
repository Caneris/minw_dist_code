from step_function_methods import run

def ID0(T, d_mwp, par_val):
    data_mat, w_dist_mat = run(T=T, lambda_LM=par_val, min_w_par=0.4, d_mwp=d_mwp)
    return data_mat, w_dist_mat

def ID1(T, d_mwp, par_val):
    data_mat, w_dist_mat = run(T=T, min_w_par=par_val, d_mwp=d_mwp)
    return data_mat, w_dist_mat

def ID2(T, d_mwp, par_val):
    data_mat, w_dist_mat = run(T=T, sigma_w=par_val, min_w_par=0.4, d_mwp=d_mwp)
    return data_mat, w_dist_mat

def ID3(T, d_mwp, par_val):
    data_mat, w_dist_mat = run(T=T, sigma_m=par_val, min_w_par=0.4, d_mwp=d_mwp)
    return data_mat, w_dist_mat

def ID4(T, d_mwp, par_val):
    data_mat, w_dist_mat = run(T=T, N_app=par_val, min_w_par=0.4, d_mwp=d_mwp)
    return data_mat, w_dist_mat

def ID5(T, d_mwp, par_val):
    data_mat, w_dist_mat = run(T=T, N_good=par_val, min_w_par=0.4, d_mwp=d_mwp)
    return data_mat, w_dist_mat

switcher = {
    0: ID0,
    1: ID1,
    2: ID2,
    3: ID3,
    4: ID4,
    5: ID5
}

def run_dist_sim_ID(ID, T, d_mwp, par_val):
    func = switcher.get(ID, "no such ID...")
    return func(T, d_mwp, par_val)