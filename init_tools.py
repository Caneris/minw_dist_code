import numpy as np
import numpy.random as rd

def init_float_mat(init_vals, N):
    n = init_vals.size
    float_mat = np.zeros((n, N), dtype=np.float64)
    for i in range(n):
        float_mat[i, :] = np.full((1, N), init_vals[i])
    return float_mat


def init_emp_mat(F, H, u_r):
    N = np.int32(H*(1-u_r))
    emp_matrix = np.zeros((F, H), dtype=bool)
    rand_f_ids = rd.permutation(N)
    rand_h_ids = rd.choice(np.arange(H), N, replace = False)

    for h_id, perm_num in zip(rand_h_ids, rand_f_ids):
        f_id = perm_num % F
        emp_matrix[f_id, h_id] = 1
    return emp_matrix


def get_skill_mat(H, H_u):
    skill_mat = np.full((2, H), 0, dtype=bool)
    skill_mat[0, 0:H_u] = np.full(H_u, True, dtype=bool)
    skill_mat[1] = np.invert(skill_mat[0,:])
    return skill_mat
