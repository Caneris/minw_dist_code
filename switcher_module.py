from step_function_methods import run

def change_min_w_par(T, par_val):
    data = run(T=T, min_w_par=par_val)
    return data

def change_eta(T, par_val):
    data = run(T=T, eta=par_val)
    return data

# def ID2(T, par_val):
#     data = run(T=T, lambda_LM=par_val)
#     return data
#
# def ID3(T, par_val):
#     data = run(T=T, sigma_w=par_val)
#     return data
#
# def ID4(T, par_val):
#     data = run(T=T, N_app=par_val)
#     return data

switcher = {
    'min_w_par': change_min_w_par,
    'eta': change_eta
}

def run_changed_param(par_string, T, par_val):
    func = switcher.get(par_string, "no such ID...")
    return func(T, par_val)