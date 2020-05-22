import numpy as np
# hidden contains number of nodes in each hidden layer
# n: inputs
# m: number of nodes in the first hidden layer
# You must make sure that Ws and bs contain the appropriate dimensions
# Ws: List or dictionary containing matrices representing parameters in each layer
# bs: the corresponding biases


def build_mlp_structure(nin, mlp_hidden):
    mlp_structure = [nin] + mlp_hidden + [1]
    wb_shapes = []
    for i in range(len(mlp_hidden) + 1):
        wb_shapes.append((mlp_structure[i], mlp_structure[i + 1]))
        wb_shapes.append((1, mlp_structure[i + 1]))
    wb_sizes = [h * w for h, w in wb_shapes]
    neurons_cnt = np.sum(np.array(wb_sizes))
    print('Total trainable parameters is like', neurons_cnt)
    return neurons_cnt, wb_shapes, wb_sizes


def func_format_weights(theta_params, wb_sizes, wb_shapes):
    wb_size_new = []
    sum_sizes = 0
    wb_sizes_array = np.asarray(wb_sizes)
    # Format the size entries using number python package
    for k in range(len(wb_sizes_array)):
        sum_sizes += wb_sizes_array[k]
        wb_size_new.append(sum_sizes)
    # ##### calculate length of input and bias params
    l_params_values = np.split(theta_params, wb_size_new)
    for i in range(len(l_params_values) - 1):
        l_params_values[i] = np.reshape(l_params_values[i], wb_shapes[i])
    ws_classify = l_params_values[0:][::2]
    bs_classify = l_params_values[1:][::2]
    return ws_classify, bs_classify


def process_ws_bs(ws0, bs0):
    ws_red = ws0[0:-1]
    lst_ws_bs = []
    for k in range(len(bs0)):
        reshaped_ws = np.r_[ws_red[k], bs0[k]]
        # print(reshaped_ws.shape)
        lst_ws_bs.append(reshaped_ws)
    return lst_ws_bs


def func_full_matrices(lst_weights):
    # empty lists
    full_flow_matrix = lst_weights.copy()
    # set index from forward flow from first layer

    # Check weights and determine flow coming from backward to front
    for w_idx in range(len(full_flow_matrix)-1):
        k0_index = w_idx + 1
        lst_new_matrices = [full_flow_matrix[0]]
        # print(full_flow_matrix[w_idx].shape)
        # print(full_flow_matrix[k0_index].shape)
        # original shape of matrix
        if k0_index <= len(full_flow_matrix):
            for col_idx in range(full_flow_matrix[w_idx].shape[1]):
                if np.sum(full_flow_matrix[w_idx][:, col_idx]) == 0.0:
                    if col_idx < full_flow_matrix[w_idx].shape[1]:
                        full_flow_matrix[k0_index][col_idx, :] = 0.0
                    else:
                        continue
    # reverse mode
    mat_arrange = full_flow_matrix.copy()
    mat_arrange.reverse()
    # print(len(mat_arrange))
    # from back to front
    for mat_in_m in range(len(mat_arrange)-1):
        k = mat_in_m + 1
        if k <= len(mat_arrange):
            for mat_idx in range(mat_arrange[mat_in_m].shape[0]-1):
                if np.sum(mat_arrange[mat_in_m][mat_idx, :]) == 0.0:
                    mat_arrange[k][:, mat_idx] = 0.0
        # print(k, len(mat_arrange))
    mat_arrange.reverse()
    return mat_arrange


def func_revers_wsbs(ws_bs_list):
    for mat_idx in range(len(ws_bs_list)):
        ws_list = []
        bs_list = []
        if np.mod(mat_idx, 2) == 0:
            bs_list.append(ws_bs_list[mat_idx])
        else:
            ws_list.append(ws_bs_list[mat_idx])
    return ws_list, bs_list


def func_all_params_concatenate(ws_mat, bs_mat):
    conc_matrix = []  # create an empty list to store concatenated matrix
    # ws_new = ws_mat[0:-1]
    for j_idx in range(len(ws_mat)):
        # reshape the ws matrix each
        reshape_ws_mat = np.reshape(ws_mat[j_idx], ws_mat[j_idx].shape[0] * ws_mat[j_idx].shape[1])
        # append to list
        conc_matrix.append(reshape_ws_mat)
        # concatenate bs and have x rows and 1 column
        re_bs0 = bs_mat[j_idx].reshape(-1)
        # re_bs0 = np.reshape(bs_mat[j_idx], bs_mat[j_idx].shape[0] * bs_mat[j_idx].shape[1])
        conc_matrix.append(re_bs0)  # append bs
    conc_arrays = np.concatenate(conc_matrix, axis=0)
    return np.array(conc_arrays)  # return the array of new matrix


def func_collect_allparams(allparams, sizes1, shapes1):
    # first pass the vector of parameters from program and format weight vector
    w, b = func_format_weights(allparams, sizes1, shapes1)
    params_in_list = process_ws_bs(w, b)
    # print(len(params_in_list))
    p_in_full = func_full_matrices(params_in_list)
    all_ps = func_organize(p_in_full)
    return all_ps

def func_set_to_zero(ws_and_bs):
    ws_and_bs.reverse()
    new_ws_bs = []
    for l in range(1, len(ws_and_bs)):
        # print(ws_and_bs[l])
        earlier_mat = ws_and_bs[l-1].copy()
        l_mat = ws_and_bs[l]
        for n_idx in range(earlier_mat.shape[0]):
            if np.sum(earlier_mat[n_idx, :]) == 0.0:
                l_mat[n_idx, :] = 0.0
        new_ws_bs.append(earlier_mat)
        new_ws_bs.append(l_mat)
    print(len(new_ws_bs))
    return new_ws_bs


def func_organize(new_ws_bs0):
    lsting_wsbs = []
    for k in new_ws_bs0:
        shaped_k = k[0:-1, :].shape
        ws_reshaped = np.reshape(k[0:-1, :], shaped_k[0]*shaped_k[1])
        bs_reshaped = np.reshape(k[-1, :], shaped_k[1])
        # 
        lsting_wsbs.append(ws_reshaped)
        lsting_wsbs.append(bs_reshaped)
    return np.array(np.concatenate(lsting_wsbs))
