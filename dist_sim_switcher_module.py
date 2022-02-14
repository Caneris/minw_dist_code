from step_function_methods import run

def ID0(d_mwp, par_val):
    data_mat, w_dist_mat = run(lambda_LM=par_val, d_mwp=d_mwp)
    return data_mat, w_dist_mat

def ID1(d_mwp, par_val):
    data_mat, w_dist_mat = run(min_w_par=par_val, d_mwp=d_mwp)
    return data_mat, w_dist_mat

def ID2(d_mwp, par_val):
    data_mat, w_dist_mat = run(sigma_w=par_val, d_mwp=d_mwp)
    return data_mat, w_dist_mat

def ID3(d_mwp, par_val):
    data_mat, w_dist_mat = run(sigma_m=par_val, d_mwp=d_mwp)
    return data_mat, w_dist_mat

def ID4(d_mwp, par_val):
    data_mat, w_dist_mat = run(N_good=par_val, N_app=par_val, d_mwp=d_mwp)
    return data_mat, w_dist_mat


switcher = {
    0: ID0,
    1: ID1,
    2: ID2,
    3: ID3,
    4: ID4
}

def run_dist_sim_ID(ID, d_mwp, par_val):
    func = switcher.get(ID, "no such ID...")
    return func(d_mwp, par_val)