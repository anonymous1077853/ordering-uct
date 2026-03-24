#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python interface for running the R CAM (Causal Additive Models) pruning routine.

This script was integrated without modification from the RL-BIC repository.
Original Python interface written by Zhuangyan Fang, Peking University.

The public entry point is `pruning_cam(XX, Adj)`, which applies backward
selection (selGam) to remove spurious edges from a candidate adjacency matrix.
"""

import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import FloatVector, ListVector, IntVector
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from rpy2.robjects import numpy2ri


# install r package CAM, and package mboost will be automatically installed
base = rpackages.importr('base')
utils = rpackages.importr('utils')

if rpackages.isinstalled('CAM'):
    pass
else:
    utils.install_packages('CAM')


# import r packages
cam = rpackages.importr('CAM')
mboost = rpackages.importr('mboost')


def CAM(XX, pns_type=None, pns_thres=None, adj_after_pns=None, pruning_type=None):
    """
    Runs the full CAM greedy DAG search on the data matrix XX.

    CAM scores candidate parent sets using an additive Gaussian process model
    (SEMGAM) and greedily adds the highest-scoring edge at each step, stopping
    when no further improvement is possible. Returns the resulting adjacency
    matrix as a numpy array of shape (d, d).

    `pns_type` and `pns_thres` control optional preliminary neighbour screening
    (PNS); pass None to skip. `adj_after_pns` provides an external candidate
    set matrix when PNS is skipped. `pruning_type` is currently unused.
    """
    string = '''
    asSparseMatrix <- function(d){
        return(as(matrix(0, d, d), "sparseMatrix"))
    }

    whichMax <- function(input){
        return(which.max(input))
    }
    '''
    selfpack = SignatureTranslatedAnonymousPackage(string, "selfpack")

    n, d = XX.shape
    maxNumParents = min(d - 1, round(n / 20))
    X = numpy2ri.py2rpy(XX)

    if pns_type != None:
        if pns_thres != None & pns_thres >= 0 & pns_thres <= 1:
            selMat = pns_type(X, pns_thres=pns_thres, verbose=False)
        else:
            raise ValueError
    else:
        if adj_after_pns == None:
            selMat = np.ones((d, d))
        else:
            selMat = adj_after_pns

    computeScoreMatTmp = robjects.r.computeScoreMat(X, scoreName='SEMGAM',
                                                    numParents=1, numCores=1, output=False,
                                                    selMat=numpy2ri.py2rpy(selMat),
                                                    parsScore=ListVector({'numBasisFcts': 10}), intervMat=float('nan'),
                                                    intervData=False)
    scoreVec = []
    edgeList = []
    pathMatrix = robjects.r.matrix(0, d, d)
    Adj = robjects.r.matrix(0, d, d)
    scoreNodes = computeScoreMatTmp.rx('scoreEmptyNodes')[0]
    scoreMat = computeScoreMatTmp.rx('scoreMat')[0]
    counterUpdate = 0
    while (sum(scoreMat.ro != -float('inf')) > 0):
        ix_max = robjects.r.arrayInd(selfpack.whichMax(scoreMat),
                                     robjects.r.dim(scoreMat))
        ix_max_backward = robjects.r.matrix(IntVector([ix_max[1], ix_max[0]]), 1, 2)
        Adj.rx[ix_max] = 1
        scoreNodes.rx[ix_max[1]] = scoreNodes.rx(ix_max[1]).ro + scoreMat.rx(ix_max)
        scoreMat.rx[ix_max] = -float('inf')
        pathMatrix.rx[ix_max[0], ix_max[1]] = 1
        DescOfNewChild = robjects.r.which(pathMatrix.rx(ix_max[1], True).ro == 1)
        AncOfNewChild = robjects.r.which(pathMatrix.rx(True, ix_max[0]).ro == 1)
        pathMatrix.rx[AncOfNewChild, DescOfNewChild] = 1
        scoreMat.rx[robjects.r.t(pathMatrix).ro == 1] = -float('inf')
        scoreMat.rx[ix_max[1], ix_max[0]] = -float('inf')
        scoreVec.append(sum(scoreNodes))
        edgeList.append(list(ix_max))
        scoreMat = robjects.r.updateScoreMat(scoreMat, X, scoreName='SEMGAM', i=ix_max[0], j=ix_max[1],
                                             scoreNodes=scoreNodes, Adj=Adj, numCores=1, output=False,
                                             maxNumParents=maxNumParents, parsScore=ListVector({'numBasisFcts': 10}),
                                             intervMat=float('nan'), intervData=False)
        counterUpdate = counterUpdate + 1

    if pruning_type != None:
        pass

    return np.array(Adj)


def _pruning(X, G, pruneMethod=robjects.r.selGam,
             pruneMethodPars=ListVector({'cutOffPVal': 0.001, 'numBasisFcts': 10}), output=False):
    """
    Applies backward edge selection to the candidate graph G using the given pruning method.

    For each node i, fits an additive model of i on its current parents in G and
    removes parents whose p-value exceeds the cut-off threshold (default 0.001 via selGam).
    X must be a numpy array; G is the candidate adjacency matrix. Returns the pruned
    adjacency matrix as a numpy array.
    """
    d = G.shape[0]
    X = robjects.r.matrix(numpy2ri.py2rpy(X), ncol=d)
    G = robjects.r.matrix(numpy2ri.py2rpy(G), d, d)
    finalG = robjects.r.matrix(0, d, d)
    for i in range(d):
        parents = robjects.r.which(G.rx(True, i + 1).ro == 1)
        lenpa = robjects.r.length(parents)[0]
        if lenpa > 0:
            Xtmp = robjects.r.cbind(X.rx(True, parents), X.rx(True, i+1))
            selectedPar = pruneMethod(Xtmp, k=lenpa + 1, pars=pruneMethodPars, output=output)
            finalParents = parents.rx(selectedPar)
            finalG.rx[finalParents, i+1] = 1

    return np.array(finalG)


def pruning_cam(XX, Adj):
    """
    Public entry point for CAM edge pruning.

    Applies `_pruning` with fixed selGam parameters (p-value cut-off 0.001,
    10 basis functions) to remove spurious edges from the candidate adjacency
    matrix Adj. XX is the data matrix (n x d); returns the pruned adjacency
    matrix as a numpy array.
    """
    X2 = numpy2ri.py2rpy(XX)
    Adj = _pruning(X=X2, G=Adj, pruneMethod=robjects.r.selGam,
                   pruneMethodPars=ListVector({'cutOffPVal': 0.001, 'numBasisFcts': 10}), output=False)

    return Adj
