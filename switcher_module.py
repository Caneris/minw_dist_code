from step_function_methods import run

def ID0(T, par_val):
    data = run(T=T, min_w_par=par_val)
    return data

def ID1(T, par_val):
    data = run(T=T, N_good=par_val)
    return data

def ID2(T, par_val):
    data = run(T=T, lambda_LM=par_val)
    return data

def ID3(T, par_val):
    data = run(T=T, sigma_w=par_val)
    return data

def ID4(T, par_val):
    data = run(T=T, N_app=par_val)
    return data

switcher = {
    0: ID0,
    1: ID1,
    2: ID2,
    3: ID3,
    4: ID4
}

def run_OFAT_ID(ID, T, par_val):
    func = switcher.get(ID, "no such ID...")
    return func(T, par_val)