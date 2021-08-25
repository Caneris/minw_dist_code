# ## Integers: 8xF matrix ##
#
# ## Order of variables (rows of the matrix):
# #[0] Nu: Number of unskilled workers,
# #[1] d_Nu: demanded Nu,
# #[2] Ns: Number of skilled workers,
# #[3] d_Ns: demanded Ns,
# #[4] vu: open vacancies unskilled workers,
# #[5] vs: open vacancies skilled workers
#
# f_int_mat = np.zeros((6, F), dtype=np.int32)
#
# ## Floats: 17xF ##
#
# ## Order of variables (rows of the matrix):
# #[0] s: number of sales,
# #[1] s_hat: number of expected sales,
# #[2] W_u: Average Wage of unskilled workers,
# #[3] W_s: Average Wage of skilled workers,
# #[4] W_u_hat: expected W_u,
# #[5] W_s_hat: expected W_s,
# #[6] W_u_tot: total wage bill for unskilled workers
# #[7] W_s_tot: total wage bill for skilled workers
# #[8] y: number of goods produced
# #[9] d_y: desired y
# #[10] uc: unit costs (costs per good produced)
# #[11] m: mark-up
# #[12] inv: inventories (nu * y)
# #[13] pi: profits
# #[14] div: dividends paid to shareholders
# #[15] delta: fraction of profits paid as dividends
# #[16] A_f: Firm's liquid assets
# #[17] p: price per good
# #[18] theta: fraction of wage bills paid to workers
#
#
#
# ## Booleans: two 2xF matrices ##
# ## Order of variables (rows of the matrix):
# # [0] job_offer: If worker got job_offer True, otherwise False
# # [1] fired: If worker got fired True, otherwise False
#
#
# ## Order of variables in skill matrix (rows of the matrix):
# # [0] unskilled: If unskilled True, otherwise False
# # [1] skilled: If skilled True, otherwise False
#
#
# ## Integers: 1xF matrix ##
# ## Order of variables (rows of the matrix):
# #[0] fired_time: shows for how many periods the workers has been fired (but still has the job, because of job protection)
#
# ## floats: 9xH ##
# ## Order of variables (rows of the matrix):
# #[0] w: wage,
# #[1] d_w: demanded wage,
# #[2] p_h: average price paid for goods,
# #[3] p_h_hat: expected p_h,
# #[4] c: (real) consumption,
# #[5] d_c: desired consumption,
# #[6] expenditure: cash spent for consumption,
# #[7] div_h: dividends received by firms as shareholder,
# #[8] R_h: costs for refinancing defaulted firms
# #[9] A_h: household's wealth
