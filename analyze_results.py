import numpy as np


print(np.count_nonzero(params))
def build_mlp_structure(nin, mlp_hidden):
    mlp_structure = [nin] + mlp_hidden + [1]
    wb_shapes = []
    for i in range(len(mlp_hidden) + 1):
        wb_shapes.append((mlp_structure[i], mlp_structure[i + 1]))
        wb_shapes.append((1, mlp_structure[i + 1]))
    wb_sizes = [h * w for h, w in wb_shapes]
    neurons_cnt = np.sum(np.array(wb_sizes))
    print('Total trainable parameters is like', neurons_cnt)
    print(wb_shapes)
    return neurons_cnt, wb_shapes, wb_sizes

def func_format_weights(theta_params, wb_sizes, wb_shapes):
    wb_size_new = []
    sum_sizes = 0
    # wb_sizes_array = np.asarray(wb_sizes)
    wb_sizes_array = [int(i) for i in wb_sizes] 
    # Format the size entries using number python package
    for k in range(len(wb_sizes_array)):
        sum_sizes += wb_sizes_array[k]
        wb_size_new.append(sum_sizes)
    print(wb_size_new)
    # ##### calculate length of input and bias params
    # ##### Another way of doing things
    l_params_values = np.split(theta_params, wb_size_new)
    l_params = l_params_values[0:-4]
    # print(l_params)
    for i in range(len(l_params)):
    	print(l_params[i].shape)
    	#l_params_values[i] = np.reshape(l_params_values[i], wb_shapes[i])
    print(np.reshape(l_params_values[0], wb_shapes[0]))
    ws_classify = l_params_values[0:][::2]
    bs_classify = l_params_values[1:][::2]
    return ws_classify, bs_classify

n = 32
mlp_hidden= [20, 5]
print(s.shape)
neurons_cnt, wbshapes, wbsizes = build_mlp_structure(n, mlp_hidden)
ws, bs = func_format_weights(s, wbsizes, wbshapes)
#print(ws, bs)
def process_ws_bs(ws0, bs0):
    ws_red = ws0[0:-1]
    lst_ws_bs = []
    for k in range(len(ws_red)):
        reshaped_ws = np.r_[ws_red[k], bs0[k]]
        lst_ws_bs.append(reshaped_ws)
    return lst_ws_bs