import os
import numpy as np
__path__=[os.path.dirname(os.path.abspath(__file__))]
from .py_func_loss import func_full_matrices


def paramreshape(allparams, wbshapes, wbsizesx, hidden):
    xwbsizenew = []
    sumwbsizes = 0
    wbsizesarray = np.asarray(wbsizesx)
    # Format the size entries using number python package
    for k in range(len(wbsizesarray)):
        sumwbsizes += wbsizesarray[k]
        xwbsizenew.append(sumwbsizes)
    # #######################calculate length of input and bias params ############
    # ##### Another way of doing things
    lpvals = np.split(allparams, xwbsizenew)	
    for i in range(len(lpvals) - 1):
        lpvals[i] = np.reshape(lpvals[i], wbshapes[i])
    ws_classif = lpvals[0:][::2]
    bs_classif = lpvals[1:][::2]	
    ##############################################################################
    eachlayerweights = []
    for ix in range(len(hidden)):
        layerwightsi = np.r_[ws_classif[ix], bs_classif[ix]]
        inputweightsbin = np.where(layerwightsi !=0, 1,0)
        xind = inputweightsbin.shape[0]
        zeroscol = np.zeros((xind, 1))
        inputweightsbin = np.c_[inputweightsbin, zeroscol]
        eachlayerweights.append(np.transpose(inputweightsbin))
    lastlayerwights = np.r_[ws_classif[-2], bs_classif[-1]]
    binlastlayerwights = np.where(lastlayerwights !=0, 1,0)
    eachlayerweights.append(np.transpose(binlastlayerwights)) 
    print(eachlayerweights[0]) 
    print(eachlayerweights[1]) 
    eachlayerweights = func_full_matrices_new(eachlayerweights)
    return eachlayerweights
	# each_lst_matrix = func_full_matrices(eachlayerweights)
	# return each_lst_matrix
##################################################################################

def func_full_matrices_new(lst_weights):
    # empty lists
    # mat_arrange = lst_weights.copy()
    full_flow_matrix = lst_weights.copy()
    # set index from forward flow from first layer
    
    # Check weights and determine flow coming from backward to front
    for w_idx in range(len(full_flow_matrix)-1):
    	k0_index = w_idx + 1
    	if k0_index <= len(full_flow_matrix):
            for col_idx in range(full_flow_matrix[w_idx].shape[0]):
                if np.sum(full_flow_matrix[w_idx][col_idx, :]) == 0:
                    full_flow_matrix[k0_index][:, col_idx] = 0
    full_flow_matrix.reverse()
    mat_arrange = lst_weights.copy()
    mat_arrange.reverse()
    for idx in range(len(mat_arrange)-1):
    	k_idex = idx + 1
    	if k_idex <= len(mat_arrange):
    		for row_idx in range(mat_arrange[idx].shape[1]):
                    if np.sum(mat_arrange[idx][:,row_idx]) == 0:
                        mat_arrange[k_idex][row_idx,:] = 0
    mat_arrange.reverse()
    return mat_arrange
