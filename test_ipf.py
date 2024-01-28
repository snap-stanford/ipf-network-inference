import helper_methods_for_aggregate_data_analysis as helper
from model_experiments import fit_disease_model_on_real_data

import argparse
import datetime
import itertools
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms.flow import preflow_push
import numpy as np
import os
from pulp import *
import pickle
import random
from scipy.sparse import csr_matrix
from scipy.linalg import eig, eigh
from scipy import optimize
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
import time

####################################################################
# Functions to do IPF / test IPF convergence
####################################################################
def do_ipf(X, p, q, num_iter=1000, start_iter=0, row_factors=None, col_factors=None, 
           eps=1e-8, return_all_factors=False, verbose=True):
    """
    X: initial matrix
    p: target row marginals
    q: target col marginals
    num_iter: number of iterations to run
    start_iter: which iteration to start at
                If start_iter > 0, then initial row_factors and col_factors must be provided
                Otherwise, we initialize to all ones
    row_factors: initial row factors
    col_factors: initial col factors
    eps: epsilon to check for convergence
    return_all_factors: whether to return row and column factors over all iterations or just 
                the final factors
    """
    assert X.shape == (len(p), len(q))
    if not np.isclose(np.sum(p), np.sum(q)):
        print('Warning: total row marginals do not equal total col marginals')
    # this allows us to continue from an earlier stopped iteration
    if start_iter > 0:
        assert row_factors is not None and col_factors is not None
        assert len(row_factors) == X.shape[0]
        assert len(col_factors) == X.shape[1]
        print(f'Starting from iter {start_iter}, received row and col factors')
        row_factors = row_factors.copy()
        col_factors = col_factors.copy()
    else:
        assert row_factors is None and col_factors is None
        row_factors = np.ones(X.shape[0])
        col_factors = np.ones(X.shape[1])
    if verbose:
        print(f'Running IPF for max {num_iter} iterations')
    
    all_row_factors = []
    all_col_factors = []
    all_est_mat = []
    row_errs = []
    col_errs = []
    for i in range(start_iter, start_iter+num_iter):
        if (i%2) == 0:  # adjust row factors
            row_sums = np.sum(X @ np.diag(col_factors), axis=1)
            # prevent divide by 0
            row_factors = p / np.clip(row_sums, 1e-8, None)
            # if marginals are 0, row factor should be 0
            row_factors[np.isclose(p, 0)] = 0
        else:  # adjust col factors
            col_sums = np.sum(np.diag(row_factors) @ X, axis=0)
            # prevent divide by 0
            col_factors = q / np.clip(col_sums, 1e-8, None)
            # if marginals are 0, column factor should be 0
            col_factors[np.isclose(q, 0)] = 0
        all_row_factors.append(row_factors)
        all_col_factors.append(col_factors)
     
        # get error from marginals
        est_mat = np.diag(row_factors) @ X @ np.diag(col_factors)
        all_est_mat.append(est_mat)
        row_err = np.sum(np.abs(p - np.sum(est_mat, axis=1)))
        col_err = np.sum(np.abs(q - np.sum(est_mat, axis=0)))
        row_errs.append(row_err)
        col_errs.append(col_err)
        if verbose:
            print('Iter %d: row err = %.4f, col err = %.4f' % (i, row_err, col_err))
        
        # check if converged
        if len(all_est_mat) >= 2:
            delta = np.sum(np.abs(all_est_mat[-1] - all_est_mat[-2]))
            if delta < eps:  # converged
                if verbose:
                    print(f'Converged; stopping after {i} iterations')
                break
            
        # check if stuck in oscillation
        if len(all_est_mat) >= 4:
            same1 = np.isclose(all_est_mat[-1], all_est_mat[-3]).all()
            same2 = np.isclose(all_est_mat[-2], all_est_mat[-4]).all()
            diff_consecutive = ~(np.isclose(all_est_mat[-1], all_est_mat[-2]).all())
            if same1 and same2 and diff_consecutive:
                if verbose:
                    print(f'Stuck in oscillation; stopping after {i} iterations')
                break                                
        
    if return_all_factors:  # return factors per iteration
        return i, np.array(all_row_factors), np.array(all_col_factors), row_errs, col_errs
    return i, row_factors, col_factors, row_errs, col_errs


def compute_error_bound(X, row_factors, col_factors):
    """
    Compute relative bound (without constant) from Theorem 4.2 given X and network parameters.
    """
    m, n = X.shape
    means = np.diag(row_factors) @ X @ np.diag(col_factors)
    bound = np.sum(means)
    square = convert_bipartite_matrix_to_square_matrix(X)
    laplacian = np.diag(np.sum(square, axis=1)) - square
    # we can use eigh since laplacian is symmetric
    w = eigh(laplacian, subset_by_index=[1,1], eigvals_only=True)  # ordered from smallest to largest, want second-smallest
    w = w[0]
    bound = bound/(w**2)
    return bound
    
    
def test_ipf_convergence_from_max_flow(X, p, q, return_flow_mat=False, flow_func=preflow_push):
    """
    Test whether IPF will converge via max flow algorithm.
    From networkx documentation: "Edges of the graph are expected to have 
    an attribute called ‘capacity’. If this attribute is not present, the edge 
    is considered to have infinite capacity."
    """
    assert np.isclose(np.sum(p), np.sum(q))
    G = nx.DiGraph()
    # add edges from source to row nodes with capacity p
    G.add_node('source')
    row_nodes = np.arange(len(p))
    G.add_nodes_from(row_nodes)
    edges = list(zip(['source'] * len(p), row_nodes))
    G.add_edges_from(edges)
    capacities = dict(zip(edges, p))
    nx.set_edge_attributes(G, name='capacity', values=capacities)
    
    # add edges from column nodes to sink with capacity q
    G.add_node('sink')
    col_nodes = np.arange(len(q)) + len(p)
    G.add_nodes_from(col_nodes)
    edges = list(zip(col_nodes, ['sink'] * len(q)))
    G.add_edges_from(edges)
    capacities = dict(zip(edges, q))
    nx.set_edge_attributes(G, name='capacity', values=capacities)
    
    # add edges between row and column nodes with infinite capacity
    nnz_row, nnz_col = np.nonzero(X)
    edges = zip(row_nodes[nnz_row], col_nodes[nnz_col])
    G.add_edges_from(edges)
    print('Constructed graph for max flow')
    
    # do maximum flow
    ts = time.time()
    f_val, f_dict = nx.maximum_flow(G, 'source', 'sink', flow_func=flow_func)
    print('Finished computing max flow [time=%.2fs]' % (time.time()-ts))
    print('Flow value = %.3f, marginal total = %.3f -> equal = %s' % (
        f_val, np.sum(p), np.isclose(f_val, np.sum(p))))
    
    if return_flow_mat:  # return a matrix representing row->column flows
        F = np.zeros(X.shape)
        m = len(p)
        for i in np.arange(m):  # iterate through row nodes
            cols, flows = zip(*list(f_dict[i].items()))  # get flows to col nodes
            F[i, np.array(cols, dtype=int)-m] = flows  # col nodes indexed by j+m in G
        return G, f_val, F
    # return the original flow dictionary
    return G, f_val, f_dict


def test_ipf_convergence_from_row_subsets(X, p, q, max_set_size=5, return_early=True):
    """
    Test whether IPF will converge by testing row subsets and their corresponding
    columns. This is not an efficient way to test for convergence, but explains more
    directly which constraint is getting violated.
    """
    # for each subset of rows, its total marginals must be less than or equal
    # to the total marginals of its corresponding POIs
    rows = np.arange(len(p), dtype=int)
    corresponding_cols = csr_matrix((X > 0).astype(int))  # maps rows to cols
    all_pass = True
    for set_size in range(1, min(len(rows), max_set_size)+1):
        ts = time.time()
        sets = np.array(list(itertools.combinations(rows, set_size)))
        print(f'Found {len(sets)} sets of size {set_size}')
        set_inds = np.zeros((len(sets), len(rows)), dtype=int)  # num_sets x num_rows
        for i in range(set_size):
            # indicators of which rows are in each set
            set_inds[np.arange(len(sets)), sets[:, i]] = 1
        set_inds = csr_matrix(set_inds)
            
        # get total row marginals per set
        set_row_marginals = (set_inds @ p.reshape(-1, 1)).reshape(-1)
        # get total col marginals per set
        set_cols = set_inds @ corresponding_cols  # which cols correspond to rows in set
        set_cols = set_cols.minimum(1)  # element-wise minimum, used to binarize
        set_col_marginals = (set_cols @ q.reshape(-1, 1)).reshape(-1)
        
        passed = set_row_marginals <= set_col_marginals
        print('Finished checks for this set size [time = %.3fs]' % (time.time()-ts))
        if not passed.all():
            all_pass = False
            print('Found %d violations' % (~passed).sum())
            violated_set_idx = np.arange(len(sets))[~passed]
            for si in violated_set_idx:
                print('Rows: %s -> row total = %.4f, col total = %.4f' % (
                    sets[si], set_row_marginals[si], set_col_marginals[si]))
            if return_early:
                return sets, set_row_marginals, set_col_marginals
        print()
    if all_pass:
        print('Passed all subsets!\n')
        
        
def find_blocking_row_subset(X, p, q, F=None):
    """
    Use max flow values to identify a blocking row subset.
    """
    if F is None:
        _, _, F = test_ipf_convergence_from_max_flow(X, p, q, return_flow_mat=True)
    assert np.sum(F) < np.sum(p)
    rows = np.arange(len(p))
    cols = np.arange(len(q))
    row_flows = np.sum(F, axis=1)
    below_cap = row_flows < p
    assert below_cap.sum() > 0  # there must be at least one row under capacity
    print('Found %d rows below capacity' % below_cap.sum())
    first_row = rows[below_cap][0]
    
    # run variant of BFS from this row
    curr_nodes = [first_row]
    layer = 0
    seen_rows = set()
    seen_cols = set()
    while len(curr_nodes) > 0:
        if (layer % 2) == 0:  # even layer, rows
            seen_rows = seen_rows.union(set(curr_nodes))
            col_sum = np.sum(X[curr_nodes], axis=0)  # sum over rows of columns
            col_neighbors = cols[col_sum > 0]
            curr_nodes = col_neighbors[~np.isin(col_neighbors, list(seen_cols))]
        else:  # odd layer, columns
            seen_cols = seen_cols.union(set(curr_nodes))
            row_sum = np.sum(F[:, curr_nodes], axis=1)  # sum over columns of rows with flows
            row_neighbors = rows[row_sum > 0]
            curr_nodes = row_neighbors[~np.isin(row_neighbors, list(seen_rows))]
        layer += 1
    print('Found blocking set of size', len(seen_rows))
    
    seen_rows, seen_cols = list(seen_rows), list(seen_cols)
    is_neighbor = np.sum(X[seen_rows], axis=0) > 0
    assert is_neighbor.sum() == len(seen_cols)  # all neighboring columns should've been visited
    row_total = p[seen_rows].sum()
    col_total = q[is_neighbor].sum()
    assert row_total > col_total
    print('Total row marginal = %.3f, total column marginal = %.3f' % (row_total, col_total))
    return seen_rows, seen_cols


def modify_x_num_edges(X, p, q, S, epsilon=1e-2, tiebreak='largest'):
    """
    Returns a modified matrix with added edges between rows in S and new non-neighboring columns,
    so that S is no longer blocking. Objective: minimize number of edges added.
    """
    assert np.isclose(p.sum(), q.sum())
    assert tiebreak in ['smallest', 'largest']
    m, n = X.shape
    columns = np.arange(n)
    is_neighbor = np.sum(X[S], axis=0) > 0
    remainder = np.sum(p[S]) - np.sum(q[is_neighbor])
    col_options = columns[~is_neighbor]  # can't be connected to S already
    print('Num column options', len(col_options))
    sorted_cols = sorted(col_options, key=lambda j:q[j], reverse=True)  # sort descending by q_j
    
    X_mod = X.copy().astype(float)
    if tiebreak == 'largest':
        row = S[np.argmax(p[S])]
    else:
        row = S[np.argmin(p[S])]
    print(f'Row with {tiebreak} p_i:', row)
    selected = []
    total = 0
    for col in sorted_cols:
        X_mod[row, col] = epsilon
        selected.append(col)
        total += q[col]
        if total > remainder or np.isclose(total, remainder):
            break
    assert total > remainder or np.isclose(total, remainder)
    print('Added top-%d columns:' % len(selected), selected)
    return X_mod
    
    
def convert_bipartite_matrix_to_square_matrix(B):
    """
    Convert a bipartite matrix that is m x n into a square matrix 
    that is (m+n) x (m+n).
    """
    m, n = B.shape
    square = np.zeros((m+n, m+n))
    square[:m][:, m:] = B
    square[m:][:, :m] = B.T
    return square

def get_largest_eigenvalue_and_eigenvectors(square):
    """
    Return a square matrix's largest eigenvalue and corresponding left/right eigenvector.
    """
    is_symmetric = (np.isclose(square, square.T)).all()
    # get first eigenvalue and first eigenvectors
    ts = time.time()
    if is_symmetric:
        index = square.shape[0]-1  # ordered from smallest to largest, want last one
        w, u = eigh(square, subset_by_index=[index,index])  # eigh is much faster
        w = w[0]
        u = u.reshape(-1)
        v = u  # left and right eigenvectors are the same for symmetric matrix
    else:
        w, u, v = eig(square, left=True, right=True)
        largest = np.argmax(w)
        w = w[largest]
        u = u[:, largest]
        v = v[:, largest]
    print('Found eigenvalues and eigenvectors [time=%.3f]' % (time.time()-ts))
    return w, u, v
    
    
def modify_x_lambda1(X, p, q, S, is_bipartite=True, epsilon=1e-2, debug=False):
    """
    Returns a modified matrix with added edges between rows in S and new non-neighboring columns,
    so that S is no longer blocking. Objective: minimize change in largest eigenvalue, lambda1.
    """
    m, n = X.shape
    if is_bipartite:
        square = convert_bipartite_matrix_to_square_matrix(X)
    else:
        square = X
    assert square.shape[0] == square.shape[1]
    w, u, v = get_largest_eigenvalue_and_eigenvectors(square)
    if min(u) < 0:  # from Tong et al., need to ensure that eigenscores are non-negative
        u = -u
    if min(v) < 0:
        v = -v
    S = list(sorted(S))
    if debug:
        print('u of S', u[S])
    smallest_row = S[np.argmin(u[S])]
    print('Row with smallest left eigenvector:', smallest_row)
    
    # select columns using integer linear program
    columns = np.arange(n)
    is_neighbor = np.sum(X[S], axis=0) > 0
    col_options = columns[~is_neighbor]  # can't be connected to S already
    if debug:
        print('Column options:', col_options)
    marginals = q[col_options]
    costs = v[m+col_options]
    remainder = np.sum(p[S]) - np.sum(q[is_neighbor])
    ts = time.time()
    prob = LpProblem("select_columns", LpMinimize)
    variables = [LpVariable(f"col{j}", 0, 1, LpInteger) for j in col_options]
    prob += lpDot(costs, variables)  # objective to minimize
    prob += lpDot(marginals, variables) >= remainder  # constraint
    status = prob.solve()
    solution = [value(v) for v in variables]
    print('Finished solving ILP [time=%.2fs]' % (time.time()-ts))
    
    # add selected columns to modified X
    X_mod = X.copy().astype(float)
    selected = []
    for j, col in enumerate(col_options):
        if solution[j] >= 0.99:
            selected.append(col)
            X_mod[smallest_row, col] = epsilon
    print('Selected columns:', selected)
    is_neighbor = np.sum(X_mod[S], axis=0) > 0
    assert np.sum(p[S]) <= np.sum(q[is_neighbor])  # should no longer be blocking
    return X_mod, w, u, v


def convergence_algorithm(X, p, q, modify_x):
    """
    Our algorithm to achieve IPF convergence.
    """
    assert modify_x in ['num_edges_smallest', 'num_edges_largest', 'lambda1']
    ts = time.time()
    curr_X = X.copy().astype(float)
    it = 0
    w_orig = None
    while True:
        print(f'=== ITER {it} ==')
        G, f_val, F = test_ipf_convergence_from_max_flow(curr_X, p, q, return_flow_mat=True)
        if np.isclose(np.sum(p), f_val):
            print('Finished convergence algorithm [total time=%.3fs]' % (time.time()-ts))
            return curr_X, w_orig
        S, _ = find_blocking_row_subset(curr_X, p, q, F=F)
        if modify_x == 'num_edges_smallest':
            curr_X = modify_x_num_edges(curr_X, p, q, S, tiebreak='smallest')
        elif modify_x == 'num_edges_largest':
            curr_X = modify_x_num_edges(curr_X, p, q, S, tiebreak='largest')
        else:
            curr_X, w, u, v = modify_x_lambda1(curr_X, p, q, S)
            if it == 0:
                w_orig = w  # save original lambda_1
        print()
        it += 1
        
def evaluate_change_in_x(X_orig, X_mod, w_orig, is_bipartite=True):
    """
    Compare original X and modified X on number of edges and largest eigenvalue.
    """
    nonzero_orig = X_orig > 0
    nonzero_mod = X_mod > 0
    assert nonzero_mod[nonzero_orig].all()  # all original nonzero entries should remain nonzero
    num_edges = (nonzero_mod & (~nonzero_orig)).sum()
    print('Number of new edges:', num_edges)
    
    if is_bipartite:
        square_mod = convert_bipartite_matrix_to_square_matrix(X_mod)
    else:
        square_mod = X_mod
    w_mod, _, _ = get_largest_eigenvalue_and_eigenvectors(square_mod)
    print('Change in largest eigenvalue', w_mod - w_orig)
    return num_edges, w_mod-w_orig
    

####################################################################
# Compare IPF to Poisson regression
####################################################################
def run_poisson_experiment(X, p, q, Y=None, F=None, method='IRLS'):
    """
    Do Poisson regression on IPF inputs.
    X, p, q: IPF inputs.
    Y: response variable. If Y is not provided, we use max-flow algorithm to find appropriate Y.
    F: optional additional interation feature of size m x n.
    method: method used for fitting model. Default is 'IRLS' (iteratively reweighted least squares), 
        which is default for statsmodels. Another option is 'lbfgs' (default for sklearn).
    """
    assert X.shape == (len(p), len(q))
    assert (p > 0).all(), 'Row marginals must be positive for Poisson regression to converge'
    assert (q > 0).all(), 'Col marginals must be positive for Poisson regression to converge'
    m, n = X.shape
    
    # construct one-hot matrix representing row and col indices
    row_nnz, col_nnz = X.nonzero()
    nnz = len(row_nnz)
    csr_rows = np.concatenate([np.arange(nnz, dtype=int), np.arange(nnz, dtype=int)])
    csr_cols = np.concatenate([row_nnz, col_nnz+m]).astype(int)
    csr_data = np.concatenate([np.ones(nnz), np.ones(nnz)])
    onehots = csr_matrix((csr_data, (csr_rows, csr_cols)), shape=(nnz, m+n)).toarray()
    assert (np.sum(onehots, axis=1) == 2).all()  # each row should have two nonzero entries
    print('Constructed one-hot mat:', onehots.shape)
    
    # construct explanatory variables
    if F is not None:  # add interaction feature
        log_f = np.log(F[row_nnz, col_nnz]).reshape(len(onehots), 1)
        explain_vars = np.concatenate([onehots, log_f], axis=1)
    else:
        explain_vars = onehots
    print('Constructed explanatory variables:', explain_vars.shape)

    # construct response variable
    if Y is None:  # get Y by running max-flow algorithm
        G, f_val, Y = test_ipf_convergence_from_max_flow(X, p, q, return_flow_mat=True)
        assert np.isclose(f_val, np.sum(p))  # IPF should converge
    assert Y.shape == X.shape
    assert (Y[X == 0] == 0).all()  # Y should inherit all zeros of X
    assert np.isclose(np.sum(Y, axis=1), p).all()  # Y should have target row marginals
    assert np.isclose(np.sum(Y, axis=0), q).all()  # Y should have target col marginals
    resp = Y[row_nnz, col_nnz]
    print('Constructed response variable:', resp.shape)
    
    ts = time.time()
    offset = np.log(X[row_nnz, col_nnz])  # include in linear model with coefficient of 1
    mdl = sm.GLM(resp, explain_vars, offset=offset, family=sm.families.Poisson())
    print('Initialized Poisson model [time=%.3fs]' % (time.time()-ts))
    ts = time.time()
    result = mdl.fit(method=method, maxiter=1000)
    print('Finished fitting model with statsmodels, method %s [time=%.3fs]' % (method, time.time()-ts))
    return mdl, result
    
def visualize_ipf_vs_poisson_params(ipf_row_factors, ipf_col_factors, reg_coefs, reg_cis=None,
                                    true_row_factors=None, true_col_factors=None, normalize=True,
                                    log_ipf=False, xlim=None, ylim=None):
    """
    Plot IPF parameters vs Poisson regression parameters. If log_ipf is True, log transform IPF parameters.
    If log_ipf is False, exponentiate the Poisson regression parameters.
    """
    m, n = len(ipf_row_factors), len(ipf_col_factors)
    reg_row_coefs = np.exp(reg_coefs[:m])
    reg_row_cis = np.exp(reg_cis[:m]) if reg_cis is not None else None
    reg_col_coefs = np.exp(reg_coefs[m:])
    reg_col_cis = np.exp(reg_cis[m:]) if reg_cis is not None else None
    if normalize:
        # normalize all factors by their mean, so that we can compare at y=x
        ipf_row_factors = ipf_row_factors / np.mean(ipf_row_factors)
        ipf_col_factors = ipf_col_factors / np.mean(ipf_col_factors)
        reg_row_cis = reg_row_cis / np.mean(reg_row_coefs) if reg_row_cis is not None else None
        reg_row_coefs = reg_row_coefs / np.mean(reg_row_coefs)
        reg_col_cis = reg_col_cis / np.mean(reg_col_coefs) if reg_col_cis is not None else None
        reg_col_coefs = reg_col_coefs / np.mean(reg_col_coefs)
        true_row_factors = true_row_factors / np.mean(true_row_factors) if true_row_factors is not None else None
        true_col_factors = true_col_factors / np.mean(true_col_factors) if true_col_factors is not None else None
    
    if log_ipf:
        ipf_row_factors = np.log(ipf_row_factors)
        ipf_col_factors = np.log(ipf_col_factors)
        ipf_row_label = '$\log(d^0_i)$'
        ipf_col_label = '$\log(d^1_j)$'
        reg_row_coefs = np.log(reg_row_coefs)
        reg_row_cis = np.log(reg_row_cis) if reg_row_cis is not None else None
        reg_col_coefs = np.log(reg_col_coefs)
        reg_col_cis = np.log(reg_col_cis) if reg_col_cis is not None else None
        reg_row_label = '$\\theta_i$'
        reg_col_label = '$\\theta_j$'
        true_row_factors = np.log(true_row_factors) if true_row_factors is not None else None
        true_col_factors = np.log(true_col_factors) if true_col_factors is not None else None    
        true_row_label = '$u_i$'
        true_col_label = '$-v_j$'
    else:
        ipf_row_label = '$d^0_i$'
        ipf_col_label = '$d^1_j$'
        reg_row_label = '$\exp(\\theta_i)$'
        reg_col_label = '$\exp(\\theta_j)$'
        true_row_label = '$\exp(u_i)$'
        true_col_label = '$\exp(-v_j)$'
        
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.3)
    # plot row params 
    ax = axes[0]
    ax.scatter(ipf_row_factors, reg_row_coefs, color='tab:blue')
    if reg_row_cis is not None:
        for i in range(m):
            ax.plot([ipf_row_factors[i], ipf_row_factors[i]], [reg_row_cis[i, 0], reg_row_cis[i, 1]], 
                    color='grey', alpha=0.5)
    ax.set_xlabel(f'{ipf_row_label} from IPF', fontsize=14)
    color = 'black' if true_row_factors is None else 'tab:blue'
    ax.set_ylabel(f'{reg_row_label} from Poisson regression', color=color, fontsize=14)
    ax.grid(alpha=0.2)
    if normalize:
        ax.plot(ipf_row_factors, ipf_row_factors, label='y=x')
        ax.legend(loc='lower right', fontsize=14)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(labelsize=12)
    
    # plot column params
    ax = axes[1]
    ax.scatter(ipf_col_factors, reg_col_coefs, color='tab:blue')
    if reg_col_cis is not None:
        for j in range(n):
            ax.plot([ipf_col_factors[j], ipf_col_factors[j]], [reg_col_cis[j, 0], reg_col_cis[j, 1]],
                    color='grey', alpha=0.5)
    ax.set_xlabel(f'{ipf_col_label} from IPF', fontsize=14)
    color = 'black' if true_col_factors is None else 'tab:blue'
    ax.set_ylabel(f'{reg_col_label} from Poisson regression', color=color, fontsize=14)
    ax.grid(alpha=0.2)
    if normalize:
        ax.plot(ipf_col_factors, ipf_col_factors, label='y=x')
        ax.legend(loc='lower right', fontsize=14)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(labelsize=12)
    
    # add true scaling factors in orange
    if true_row_factors is not None:
        ax_twin = axes[0].twinx()
        ax_twin.scatter(ipf_row_factors, true_row_factors, color='tab:orange', alpha=0.5)
        ax_twin.set_ylabel(f'{true_row_label} from Poisson model', color='tab:orange', fontsize=14)
        ax_twin.set_xlim(axes[0].get_xlim())
        ax_twin.set_ylim(axes[0].get_ylim())
        ax_twin.tick_params(labelsize=12)
    if true_col_factors is not None:
        ax_twin = axes[1].twinx()
        ax_twin.scatter(ipf_col_factors, true_col_factors, color='tab:orange', alpha=0.5)
        ax_twin.set_ylabel(f'{true_col_label} from Poisson model', color='tab:orange', fontsize=14)
        ax_twin.set_xlim(axes[1].get_xlim())
        ax_twin.set_ylim(axes[1].get_ylim())
        ax_twin.tick_params(labelsize=12)
    return fig, axes

    
if __name__ == '__main__':
    test_recoverable_process(sparsity_rate=0.8, seed=1)