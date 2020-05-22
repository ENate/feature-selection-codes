import numpy as np

def paramreshape(allparams, wbshapes, wbsizesx, hidden):
	xwbsizenew = []
	sumwbsizes = 0
	wbsizesarray = np.asarray(wbsizesx)
	# Format the size entries using number python package
	for k in range(len(wbsizesarray)):
		sumwbsizes += wbsizesarray[k]
		xwbsizenew.append(sumwbsizes)
	########################calculate length of input and bias params ############
	###### Another way of doing things
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
	##############################################################################
	print(len(eachlayerweights))

	return eachlayerweights
##################################################################################