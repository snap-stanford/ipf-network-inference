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
from scipy.linalg import eig
from scipy import optimize
from scipy.stats import pearsonr
from sklearn.linear_model import PoissonRegressor
import statsmodels.api as sm
import time

####################################################################
# Functions to do IPF / test IPF convergence
####################################################################
def do_ipf(Z, u, v, num_iter=1000, start_iter=0, row_factors=None, col_factors=None, 
           eps=1e-8, return_all_factors=False, verbose=True):
    """
    Z: initial matrix
    u: target row marginals
    v: target col marginals
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
    assert Z.shape == (len(u), len(v))
    if not np.isclose(np.sum(u), np.sum(v)):
        print('Warning: total row marginals do not equal total col marginals')
    # this allows us to continue from an earlier stopped iteration
    if start_iter > 0:
        assert row_factors is not None and col_factors is not None
        assert len(row_factors) == Z.shape[0]
        assert len(col_factors) == Z.shape[1]
        print(f'Starting from iter {start_iter}, received row and col factors')
        row_factors = row_factors.copy()
        col_factors = col_factors.copy()
    else:
        assert row_factors is None and col_factors is None
        row_factors = np.ones(Z.shape[0])
        col_factors = np.ones(Z.shape[1])
    print(f'Running IPF for max {num_iter} iterations')
    
    all_row_factors = []
    all_col_factors = []
    all_est_M = []
    row_errs = []
    col_errs = []
    for i in range(start_iter, start_iter+num_iter):
        if (i%2) == 0:  # adjust row factors
            row_sums = np.sum(Z @ np.diag(col_factors), axis=1)
            # prevent divide by 0
            row_factors = u / np.clip(row_sums, 1e-8, None)
            # if marginals are 0, row factor should be 0
            row_factors[np.isclose(u, 0)] = 0
        else:  # adjust col factors
            col_sums = np.sum(np.diag(row_factors) @ Z, axis=0)
            # prevent divide by 0
            col_factors = v / np.clip(col_sums, 1e-8, None)
            # if marginals are 0, column factor should be 0
            col_factors[np.isclose(v, 0)] = 0
        all_row_factors.append(row_factors)
        all_col_factors.append(col_factors)
     
        # get error from marginals
        est_M = np.diag(row_factors) @ Z @ np.diag(col_factors)
        all_est_M.append(est_M)
        row_err = np.sum(np.abs(u - np.sum(est_M, axis=1)))
        col_err = np.sum(np.abs(v - np.sum(est_M, axis=0)))
        row_errs.append(row_err)
        col_errs.append(col_err)
        if verbose:
            print('Iter %d: row err = %.4f, col err = %.4f' % (i, row_err, col_err))
        
        # check if converged
        if len(all_est_M) >= 2:
            delta = np.sum(np.abs(all_est_M[-1] - all_est_M[-2]))
            if delta < eps:  # converged
                print(f'Converged; stopping after {i} iterations')
                break
            
        # check if stuck in oscillation
        if len(all_est_M) >= 4:
            same1 = np.isclose(all_est_M[-1], all_est_M[-3]).all()
            same2 = np.isclose(all_est_M[-2], all_est_M[-4]).all()
            diff_consecutive = ~(np.isclose(all_est_M[-1], all_est_M[-2]).all())
            if same1 and same2 and diff_consecutive:
                print(f'Stuck in oscillation; stopping after {i} iterations')
                break                                
        
    if return_all_factors:  # return factors per iteration
        return i, np.array(all_row_factors), np.array(all_col_factors), row_errs, col_errs
    return i, row_factors, col_factors, row_errs, col_errs


def test_ipf_convergence_from_max_flow(A, p, q, return_flow_mat=False, flow_func=preflow_push):
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
    nnz_row, nnz_col = np.nonzero(A)
    edges = zip(row_nodes[nnz_row], col_nodes[nnz_col])
    G.add_edges_from(edges)
    print('Constructed graph for max flow')
    
    # do maximum flow
    ts = time.time()
    print(flow_func)
    f_val, f_dict = nx.maximum_flow(G, 'source', 'sink', flow_func=flow_func)
    print('Finished computing max flow [time=%.2fs]' % (time.time()-ts))
    print('Flow value = %.3f, marginal total = %.3f -> equal = %s' % (
        f_val, np.sum(p), np.isclose(f_val, np.sum(p))))
    
    if return_flow_mat:  # return a matrix representing row->column flows
        F = np.zeros(A.shape)
        m = len(p)
        for i in np.arange(m):  # iterate through row nodes
            cols, flows = zip(*list(f_dict[i].items()))  # get flows to col nodes
            F[i, np.array(cols, dtype=int)-m] = flows  # col nodes indexed by j+m in G
        return G, f_val, F
    # return the original flow dictionary
    return G, f_val, f_dict


def test_ipf_convergence_from_row_subsets(A, p, q, max_set_size=5, return_early=True):
    """
    Test whether IPF will converge by testing row subsets and their corresponding
    columns. This is not an efficient way to test for convergence, but explains more
    directly which constraint is getting violated.
    """
    # for each subset of rows, its total marginals must be less than or equal
    # to the total marginals of its corresponding POIs
    rows = np.arange(len(p), dtype=int)
    corresponding_cols = csr_matrix((A > 0).astype(int))  # maps rows to cols
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
        
        
def find_blocking_row_subset(A, p, q, F=None):
    """
    Use max flow values to identify a blocking row subset.
    """
    if F is None:
        _, _, F = test_ipf_convergence_from_max_flow(A, p, q, return_flow_mat=True)
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
            col_sum = np.sum(A[curr_nodes], axis=0)  # sum over rows of columns
            col_neighbors = cols[col_sum > 0]
            curr_nodes = col_neighbors[~np.isin(col_neighbors, list(seen_cols))]
        else:  # odd layer, columns
            seen_cols = seen_cols.union(set(curr_nodes))
            row_sum = np.sum(F[:, curr_nodes], axis=1)  # sum over columns of rows with flows
            row_neighbors = rows[row_sum > 0]
            curr_nodes = row_neighbors[~np.isin(row_neighbors, list(seen_rows))]
        layer += 1
    return seen_rows, seen_cols


def convert_bipartite_matrix_to_square_matrix(B):
    """
    Helper function to convert a bipartite matrix that is m x n into a square matrix 
    that is (m+n) x (m+n).
    """
    m, n = B.shape
    square = np.zeros((m+n, m+n))
    square[:m][:, m:] = B
    square[m:][:, :m] = B.T
    return square


def modify_x(A, p, q, S, epsilon=1e-2, debug=False):
    """
    Returns a modified matrix with added edges between rows in S and new non-neighboring columns,
    so that S is no longer blocking.
    """
    # get first eigenvalue and first eigenvectors
    m, n = A.shape
    square = convert_bipartite_matrix_to_square_matrix(A)
    if debug:
        print(square)
    w, u, v = eig(square, left=True, right=True)
    largest = np.argmax(w)
    smallest = np.argmin(w)
    assert np.isclose(w[largest], -w[smallest])  # for bipartite graph, first eigenvalue should be -last eigenvalue
    if debug:
        print('Largest eigenvalue:', largest)
    u = u[:, largest]
    v = v[:, largest]
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
    is_neighbor = np.sum(A[S], axis=0) > 0
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
    A_mod = A.copy().astype(float)
    selected = []
    for j, col in enumerate(col_options):
        if solution[j] >= 0.99:
            selected.append(col)
            A_mod[smallest_row, col] = epsilon
    print('Selected columns:', selected)
    
    # check modified X
    is_neighbor = np.sum(A_mod[S], axis=0) > 0
    assert np.sum(p[S]) <= np.sum(q[is_neighbor])  # should no longer be blocking
    square_mod = convert_bipartite_matrix_to_square_matrix(A_mod)
    w_mod = eig(square_mod, left=False, right=False)
    largest_mod = max(w_mod)
    print('Change in largest eigenvalue: %.4f' % (largest_mod - w[largest]))
    return A_mod


def test_ipf_convergence_from_choice_sets(A, p, q, sort_items=False, apportion_strategy='greedy'):
    """
    DEPRECATED. Test whether IPF will converge via choice set construction algorithm.
    """
    assert (A >= 0).all() and (p >= 0).all() and (q >= 0).all()
    assert A.shape == (len(p), len(q))
    assert np.isclose(np.sum(p), np.sum(q))  # sum of marginals should be the same
    assert apportion_strategy in {'greedy', 'proportional'}
    ts = time.time()
    choice_sets = np.arange(A.shape[0])  # choice sets as rows
    items = np.arange(A.shape[1])  # items as columns
    row_nnz, col_nnz = A.nonzero()  # indices of choice set, item pairs
    G = nx.DiGraph()  # item x item comparison graph
    G.add_nodes_from(items)
    p_remaining = p.copy()  # how many wins left per choice set
    
    # TEMP
    cbg_id = 720
    test_pois = row_nnz[col_nnz == cbg_id]
    
    # for each item, go through its choice sets
    if sort_items:
        print('Will iterate through items from smallest to largest q_j')
        item_order = np.argsort(q)  # sort from smallest to largest q_j
    else:
        item_order = items
    for j in item_order:
        if (j % 100) == 0:
            print(j)
        q_j = q[j].copy()
        if q_j > 0:  # if item j has wins to apportion
            choice_sets_j = row_nnz[col_nnz == j]  # choice sets that j belongs to
            choice_sets_remain = choice_sets[p_remaining > 0]  # choice sets with wins remaining
            intersect = sorted(set(choice_sets_j).intersection(set(choice_sets_remain))) 
            intersect_p_remaining = np.sum(p_remaining[intersect])
            if q_j > intersect_p_remaining:
                raise Exception('Stopped after apportioning %.2f visits. Could not apportion all of item %d\'s wins: q_j = %.2f, wins remaining in j\'s choice sets = %.2f' % (
                    np.sum(p)-np.sum(p_remaining), j, q_j, intersect_p_remaining))
            
            if apportion_strategy == 'greedy':  # apportion as much as you can to this choice set
                for i in intersect:
                    amount = min(p_remaining[i], q_j)
                    if amount > 0:  # item j wins choice set i at least once
                        q_j -= amount  # remove from item j's wins
                        p_remaining[i] = p_remaining[i] - amount  # remove from choice set i's wins
                        # TEMP
                        if i in test_pois:
                            print(f'CBG {j}: removing {amount} from POI {i}, {p_remaining[i]} remaining')
                        items_i = list(col_nnz[row_nnz == i])  # items in choice set i
                        items_i.remove(j)  # remove item j from list
                        new_edges = list(zip(items_i, [j] * len(items_i)))  # edges from other items to item j
                        G.add_edges_from(new_edges)
                    if q_j == 0:  # stop when all of q_j has been apportioned
                        break
                assert q_j == 0
            else:  # apportion proportionally
                amounts = p_remaining[intersect] * q_j / intersect_p_remaining  # so that sum is q_j
                assert np.isclose(np.sum(amounts), q_j)
                p_remaining[intersect] = p_remaining[intersect] - amounts
                items_intersect = list(set(col_nnz[np.isin(row_nnz, intersect)]))  # items in intersect choice sets
                items_intersect.remove(j)  # remove item j from list
                new_edges = list(zip(items_intersect, [j] * len(items_intersect)))  # edges from other items to item j
                G.add_edges_from(new_edges)

    print('Finished apportioning wins [time=%.2fs]' % (ts-time.time()))
    strongly_conected = nx.is_strongly_connected(G)
    print('Comparison graph is strongly connected:', strongly_conected)
    return G


####################################################################
# Experiments with synthetic matrices
####################################################################
def generate_M(dist='uniform', m=100, n=200, seed=0, sparsity_rate=0, exact_rate=False, verbose=True):
    """
    Generate M based on kwargs.
    sparsity_rate: each entry is set to 0 with probability sparsity_rate.
    """
    np.random.seed(seed)
    assert dist in {'uniform', 'poisson'}
    if verbose:
        print(f'Sampling M from {dist} distribution')
    if dist == 'uniform':
        M = np.random.rand(m, n)
    elif dist == 'poisson':
        M = np.random.poisson(lam=10, size=(m,n))
    if sparsity_rate > 0:
        assert sparsity_rate < 1
        if exact_rate:
            random.seed(seed)
            num_zeros = int(sparsity_rate * (m*n))  # sample exactly this number of entries to set to 0
            pairs = list(itertools.product(range(m), range(n)))  # all possible pairs
            set_to_0 = random.sample(pairs, num_zeros)  # sample without replacement
            set_to_0 = ([t[0] for t in set_to_0], [t[1] for t in set_to_0])
        else:
            # set each entry to 0 with independent probability sparsity_rate
            set_to_0 = np.random.rand(m, n) < sparsity_rate
        M[set_to_0] = 0
        if verbose:
            print('Num nonzero entries in M: %d out of %d' % (np.sum(M > 0), m*n))
    return M 

def evaluate_fitted_matrix(M, est_M, verbose=True):
    """
    Compare fitted matrix est_M to true matrix M.
    """
    M_err = np.sum(np.abs(M - est_M))
    row_err = np.sum(np.abs(np.sum(M, axis=1) - np.sum(est_M, axis=1)))
    col_err = np.sum(np.abs(np.sum(M, axis=0) - np.sum(est_M, axis=0)))
    if verbose:
        print('Diff in M: %.4f' % M_err)
        print('Diff in row marginals: %.4f' % row_err)
        print('Diff in col marginals: %.4f' % col_err)
    return M_err, row_err, col_err

def test_recoverable_process(m_kwargs, verbose=True):
    """
    M is recoverable, i.e., M = PZQ, for diagonal matrices P and Q.
    """
    M = generate_M(verbose=verbose, **m_kwargs)
    m, n = M.shape
    u = np.sum(M, axis=1)  # true row marginals
    v = np.sum(M, axis=0)  # true col marginals
    true_row_factors = np.clip(np.random.rand(m) * 2, 1e-5, None)  # range from (0, 2)
    true_col_factors = np.clip(np.random.rand(n) * 2, 1e-5, None)
    Z = np.diag(1/true_row_factors) @ M @ np.diag(1/true_col_factors)  # initial matrix
    assert np.isclose(M, np.diag(true_row_factors) @ Z @ np.diag(true_col_factors)).all()
    if not (np.sum(Z, axis=1) > 0).all():
        print('Warning: found zero row in Z')
    if not (np.sum(Z, axis=0) > 0).all():
        print('Warning: found zero column in Z')
    ## TODO: test if Z can be permuted to block-diagonal

    iterations, row_factors, col_factors, _, _ = do_ipf(Z, u, v, verbose=verbose)
    est_M = np.diag(row_factors) @ Z @ np.diag(col_factors)
    M_err, row_err, col_err = evaluate_fitted_matrix(M, est_M, verbose=verbose)
    return iterations, M_err, row_err, col_err

def test_process_with_missingness(m_kwargs, z_sparsity=0, verbose=True,
                                  return_matrices=False):
    """
    Generate Z with greater sparsity than M.
    """
    M = generate_M(verbose=verbose, **m_kwargs)
    m, n = M.shape
    u = np.sum(M, axis=1)  # true row marginals
    v = np.sum(M, axis=0)  # true col marginals
    true_row_factors = np.clip(np.random.rand(m) * 2, 1e-5, None)  # range from (0, 2)
    true_col_factors = np.clip(np.random.rand(n) * 2, 1e-5, None)
    Z = np.diag(1/true_row_factors) @ M @ np.diag(1/true_col_factors)  # initial matrix
    assert np.isclose(M, np.diag(true_row_factors) @ Z @ np.diag(true_col_factors)).all()
    if z_sparsity > 0:
        assert z_sparsity < 1
        set_to_0 = np.random.rand(m, n) < z_sparsity
        Z[set_to_0] = 0
        if verbose:
            print('Num nonzero entries in Z: %d out of %d' % (np.sum(Z > 0), m*n))
    if not (np.sum(Z, axis=1) > 0).all():
        print('Warning: found zero row in Z')
    if not (np.sum(Z, axis=0) > 0).all():
        print('Warning: found zero column in Z')
    ## TODO: test if Z can be permuted to block-diagonal

    iterations, row_factors, col_factors, _, _ = do_ipf(Z, u, v, verbose=verbose)
    est_M = np.diag(row_factors) @ Z @ np.diag(col_factors)
    M_err, row_err, col_err = evaluate_fitted_matrix(M, est_M, verbose=verbose)
    if return_matrices:
        return iterations, M, est_M, M_err, row_err, col_err
    return iterations, M_err, row_err, col_err

def test_unrecoverable_process():
    """
    Test unrecoverable process, where we know there is no P, Q such that 
    M = PZQ.
    """
    M = np.array([[1, 9],
                  [2, 0.5]])
    u = np.sum(M, axis=1)  # true row marginals
    v = np.sum(M, axis=0)  # true col marginals
    Z = np.ones(M.shape)
    iter, row_factors, col_factors, _, _ = do_ipf(Z, u, v)
    print(row_factors)
    print(col_factors)
    est_M = np.diag(row_factors) @ Z @ np.diag(col_factors)
    print(est_M)
    print('Abs diff in M: %.4f' % np.sum(np.abs(M - est_M)))

    
####################################################################
# Test IPF on SafeGraph mobility data
####################################################################
def prep_safegraph_data_for_ipf(msa_name, dt, msa_df_date_range):
    """
    Prep SafeGraph data for IPF.
    msa_name: name of metropolitan statistical area
    dt: datetime object, with year, month, day, and hour
    msa_df_date_range: SafeGraph data is stored per date range; this is the date range corresponding to this datetime
    """
    min_datetime = datetime.datetime(dt.year, dt.month, dt.day, 0)
    max_datetime = datetime.datetime(dt.year, dt.month, dt.day, 23)
    CBG_COUNT_CUTOFF = 100  # this doesn't matter since CBGs are prespecified
    POI_HOURLY_VISITS_CUTOFF = 'all'  # same, doesn't matter
    poi_ids = helper.load_poi_ids_for_msa(msa_name)
    cbg_ids = helper.load_cbg_ids_for_msa(msa_name)
    print('Loaded %d POI and %d CBG ids' % (len(poi_ids), len(cbg_ids)))
    msa_df = helper.prep_msa_df_for_model_experiments(msa_name, [msa_df_date_range])
    m = fit_disease_model_on_real_data(d=msa_df,
                                       min_datetime=min_datetime,
                                       max_datetime=max_datetime,
                                       msa_name=msa_name,
                                       exogenous_model_kwargs={'poi_psi':1, 
                                                               'home_beta':1, 
                                                               'p_sick_at_t0':0,  # don't need infections
                                                               'just_compute_r0':False},
                                       poi_attributes_to_clip={'clip_areas':True, 
                                                               'clip_dwell_times':True, 
                                                               'clip_visits':True},
                                       preload_poi_visits_list_filename=None,
                                       poi_cbg_visits_list=None,
                                       poi_ids=poi_ids,
                                       cbg_ids=cbg_ids,
                                       correct_poi_visits=True,
                                       multiply_poi_visit_counts_by_census_ratio=True,
                                       aggregate_home_cbg_col='aggregated_cbg_population_adjusted_visitor_home_cbgs',
                                       poi_hourly_visits_cutoff=POI_HOURLY_VISITS_CUTOFF,  
                                       cbg_count_cutoff=CBG_COUNT_CUTOFF,
                                       cbgs_to_filter_for=None,
                                       cbg_groups_to_track=None,
                                       counties_to_track=None,
                                       include_cbg_prop_out=True,
                                       include_inter_cbg_travel=False,
                                       include_mask_use=False,
                                       model_init_kwargs={'ipf_final_match':'poi',
                                                          'ipf_num_iter':100,
                                                          'num_seeds':2},
                                       simulation_kwargs={'do_ipf':True, 
                                                          'allow_early_stopping':False},
                                       counterfactual_poi_opening_experiment_kwargs=None,
                                       counterfactual_retrospective_experiment_kwargs=None,
                                       return_model_without_fitting=True,  # note: changed from False to True
                                       attach_data_to_model=True,
                                       model_quality_dict=None,
                                       verbose=True)
    Z, u, v = _prep_safegraph_data_for_ipf(m.POI_TIME_COUNTS, m.cbg_day_prop_out, 
                                          m.CBG_SIZES, m.POI_CBG_PROPORTIONS.toarray(), dt.hour)
    return Z, u, v
    
    
def _prep_safegraph_data_for_ipf(poi_time_counts, cbg_day_prop_out, cbg_sizes, 
                                 poi_cbg_props, t):
    """
    Helper function to prep IPF inputs from preprocessed SafeGraph data.
    poi_time_counts: n_pois x hours, represents hourly number of visits to each POI
    cbg_day_prop_out: n_cbgs x days, represents daily proportion of each CBG that is out.
                      we use median when value is NaN.
    cbg_sizes: n_cbgs, CBG population sizes
    poi_cbg_props: n_pois x n_cbgs, time-aggregated distributions over visitors' home CBGs for each POI
                   row sums represent total proportion of POI's visitors who come from this
                   set of CBGs. Usually using 20191230_20201019_aggregated_visitor_home_cbgs,
                   which accounts for CBG coverage.
    t: scalar, which hour we are calculating
    """
    poi_visits = poi_time_counts[:, t]
    poi_nan = np.isnan(poi_visits)
    day = int(t / 24)
    cbg_prop_out = cbg_day_prop_out[:, day]
    cbg_visits = cbg_prop_out * cbg_sizes
    cbg_nan = np.isnan(cbg_visits)
    if poi_nan.sum() > 0 or cbg_nan.sum() > 0:
        print('Removing %d POIs and %d CBGs with NaN marginals' % (poi_nan.sum(), cbg_nan.sum()))
        poi_visits = poi_visits[~poi_nan]
        cbg_visits = cbg_visits[~cbg_nan]
        poi_cbg_props = poi_cbg_props[~poi_nan][:, ~cbg_nan]
    assert poi_cbg_props.shape == (len(poi_visits), len(cbg_visits))
    
    # proportion of POI's visitors that we account for
    prop_poi_kept = poi_cbg_props @ np.ones(poi_cbg_props.shape[1])  
    u = poi_visits * prop_poi_kept
    v = cbg_visits
    v = v * np.sum(u) / np.sum(v)  # renormalize to match row sums
    assert np.isclose(np.sum(u), np.sum(v))
    return poi_cbg_props, u, v
    
    
def construct_networkx_bipartite_graph(poi_ids, cbg_ids, poi_cbg_props):
    """
    Construct Networkx bipartite graph.
    """
    # construct bipartite graph in networkx
    B = nx.Graph()
    B.add_nodes_from(poi_ids, bipartite=0)
    B.add_nodes_from(cbg_ids, bipartite=1)
    nnz_row_idx, nnz_col_idx = np.nonzero(poi_cbg_props)
    nnz_poi_ids = poi_ids[nnz_row_idx]
    nnz_cbg_ids = cbg_ids[nnz_col_idx]
    weights = poi_cbg_props[nnz_row_idx, nnz_col_idx]
    edge_list = list(zip(nnz_poi_ids, nnz_cbg_ids, weights))
    B.add_weighted_edges_from(edge_list)
    return B

def print_stats_of_aggregated_matrix(poi_ids, cbg_ids, poi_cbg_props):
    """
    Print stats of aggregated matrix.
    """
    print('Num POIs: %d. Num CBGs: %d.' % (len(poi_ids), len(cbg_ids)))
    assert poi_cbg_props.shape == (len(poi_ids), len(cbg_ids))
    prop_poi_kept = poi_cbg_props @ np.ones(poi_cbg_props.shape[1])  
    print('Total POI props kept: %.2f%%-%.2f%% (IQR)' % (100. * np.percentile(prop_poi_kept, 25),
                                                         100. * np.percentile(prop_poi_kept, 75)))
    print('Prop entries kept per threshold')
    for t in np.arange(0, 0.11, 0.01):
        prop = np.sum(poi_cbg_props > t) / (poi_cbg_props.shape[0] * poi_cbg_props.shape[1])
        print(t, prop)
    
    props_sorted = np.sort(poi_cbg_props, axis=1)  # sort row-wise
    cutoff = int(round(props_sorted.shape[1] * 0.01))  # top 1%
    top_sum = np.sum(props_sorted[:, -cutoff:], axis=1)
    print('Percent visits per POI from top 1%% (%d CBGs): mean=%.2f%%, median=%.2f%%' % (
        cutoff, 100. * np.mean(top_sum), 100. * np.median(top_sum)))
    cutoff = int(round(props_sorted.shape[1] * 0.1))  # top 10%
    top_sum = np.sum(props_sorted[:, -cutoff:], axis=1)
    print('Percent visits per POI from top 10%% (%d CBGs): mean=%.2f%%, median=%.2f%%' % (
        cutoff, 100. * np.mean(top_sum), 100. * np.median(top_sum)))
    
    B = construct_networkx_bipartite_graph(poi_ids, cbg_ids, poi_cbg_props)
    print('Connected:', nx.is_connected(B))
    if not nx.is_connected(B):
        largest_cc = max(nx.connected_components(B), key=len)
        num_pois = 0
        for k in largest_cc:
            if type(k) == str and k.startswith('sg:'):
                num_pois += 1
        print('Largest connected compomnent: %d POIs, %d CBGs' % (num_pois, len(largest_cc)-num_pois))
    return B
    
def run_safegraph_ipf_experiment(msa_name, dt, msa_df_date_range, max_iter=1000):
    """
    Run IPF on SafeGraph data for given datetime.
    msa_name: name of metropolitan statistical area
    dt: datetime object, with year, month, day, and hour
    msa_df_date_range: SafeGraph data is stored per date range; this is the date range corresponding to this datetime
    max_iter: max iterations to run IPF
    """
    Z, u, v = prep_safegraph_data_for_ipf(msa_name, dt, msa_df_date_range)
    print('Date: %s, marginals prop positive -> POIs = %.3f, CBGs = %.3f' % (
        dt.strftime('%Y-%m-%d-%H'), np.mean(u > 0), np.mean(v > 0)))
    ts = time.time()
    ipf_out = do_ipf(Z, u, v, num_iter=max_iter)
    print('Finished IPF: time=%.2fs' % (time.time()-ts))
    fn = 'ipf-output/%s_%s.pkl' % (msa_name, dt.strftime('%Y-%m-%d-%H'))
    print('Saving results in', fn)
    with open(fn, 'wb') as f:
        pickle.dump(ipf_out, f)
        
def run_safegraph_all_hours_in_day(msa_name, dt):
    """
    Outer function to run IPF for all hours in a given day.
    """
    print('Running IPF for %s, all hours on %s...' % (msa_name, dt.strftime('%Y-%m-%d')))
    for hr in range(24):
        curr_dt = datetime.datetime(year=dt.year, month=dt.month, day=dt.day, hour=hr)
        out_file = 'ipf-output/%s_%s.out' % (msa_name, curr_dt.strftime('%Y-%m-%d-%H'))
        cmd = f'nohup python -u test_ipf.py safegraph --msa_name {msa_name} --hour {hr} --mode inner > {out_file} 2>&1 &'
        print(cmd)
        os.system(cmd)
        time.sleep(1)


####################################################################
# Test IPF on bikeshare data from CitiBike
####################################################################
def prep_bikeshare_data_for_ipf(dt, timeagg='month', hours=None, networks=None):
    """
    Prep bikeshare data for IPF.
    dt: datetime object, with year, month, day, and hour
    timeagg: how much time to aggregate over
    """
    assert (dt >= datetime.datetime(2023, 9, 1)) and (dt < datetime.datetime(2023, 10, 1))
    assert timeagg in ['month', 'week', 'day']
    print('Prepping bikeshare data for %s...' % datetime.datetime.strftime(dt, '%Y-%m-%d %H'))
    if hours is None or networks is None:
        with open('bikeshare-202309.pkl', 'rb') as f:
            hours, networks = pickle.load(f)
    hour_idx = hours.index(dt)
    true_mat = networks[hour_idx].toarray()
    p = true_mat.sum(axis=1)  # row marginals
    q = true_mat.sum(axis=0)  # column marginals
    N = len(p)
    
    # get time-aggregated matrix
    if timeagg == 'month':
        start_idx = 0
        end_idx = len(networks)
    elif timeagg == 'week':
        week_idx = hour_idx // 168
        start_idx = 168 * week_idx
        end_idx = 168 * (week_idx + 1)
    else:
        day_idx = hour_idx // 24
        start_idx = 24 * day_idx
        end_idx = 24 * (day_idx + 1)
    X = 0
    for mat in networks[start_idx:end_idx]:
        X += mat
    nnz = X.count_nonzero()
    print('Aggregated to %s-level -> %d pairs (%.2f%%)' % (timeagg, nnz, 100 * nnz / (N*N)))
    X = X.toarray()
    return X, p, q, true_mat


def eval_est_mat(est_mat, real_mat, verbose=True):
    """
    Evaluate distance between real matrix and estimated matrix.
    """
    if not np.isclose(est_mat.sum(), real_mat.sum()):
        print('Warning: matrices do not have the same total, off by %.3f' % np.abs(est_mat.sum()-real_mat.sum()))
    if not np.isclose(real_mat.sum(axis=1), est_mat.sum(axis=1)).all():
        print('Warning: row marginals don\'t match')
    if not np.isclose(real_mat.sum(axis=0), est_mat.sum(axis=0)).all():
        print('Warning: col marginals don\'t match')
    l2 = np.sqrt(np.sum((est_mat - real_mat) ** 2))
    norm = np.sqrt(np.sum(real_mat ** 2))
    if verbose:
        print('Normalized L2 distance', l2/norm)
    corr = pearsonr(real_mat.flatten(), est_mat.flatten())
    if verbose:
        print('Pearson corr', corr)
    return l2/norm, corr
    

def run_bikeshare_ipf_experiment(dt, timeagg='month', max_iter=1000):
    """
    Run IPF experiment on bikeshare data.
    """
    X, p, q, true_mat = prep_bikeshare_data_for_ipf(dt, timeagg)
    ts = time.time()
    ipf_out = do_ipf(X, p, q, num_iter=max_iter)
    print('Finished IPF: time=%.2fs' % (time.time()-ts))    

    row_factors, col_factors = ipf_out[1], ipf_out[2]
    est_mat = np.diag(row_factors) @ X @ np.diag(col_factors)
    print('Comparing real matrix and estimated matrix')
    eval_est_mat(est_mat, true_mat)
    
    fn = 'ipf-output/bikeshare_%s_%s.pkl' % (timeagg, dt.strftime('%Y-%m-%d-%H'))
    print('Saving results in', fn)
    with open(fn, 'wb') as f:
        pickle.dump(ipf_out, f)
        
        
def run_bikeshare_all_hours_in_day(dt, timeagg='month', max_iter=1000):
    """
    Outer function to run IPF for all hours in a given day.
    """
    print('Running IPF on bikeshare data, all hours on %s...' % dt.strftime('%Y-%m-%d'))
    for hr in range(24):
        curr_dt = datetime.datetime(year=dt.year, month=dt.month, day=dt.day, hour=hr)
        out_file = 'ipf-output/bikeshare_%s_%s.out' % (timeagg, curr_dt.strftime('%Y-%m-%d-%H'))
        cmd = f'nohup python -u test_ipf.py bikeshare {dt.year} {dt.month} {dt.day} --hour {hr} --timeagg {timeagg} --max_iter {max_iter} --mode inner > {out_file} 2>&1 &'
        print(cmd)
        os.system(cmd)
        time.sleep(1)

        
def baseline_no_mat(p, q):
    """
    Baseline where we ignore time-aggregated matrix and only use marginals.
    """
    outer_prod = np.outer(p, q) * 1.0
    outer_prod /= np.sum(outer_prod)
    outer_prod *= np.sum(p)  # scale to sum to marginal total
    assert np.isclose(np.sum(outer_prod, axis=1), p).all()
    assert np.isclose(np.sum(outer_prod, axis=0), q).all()
    return outer_prod

def baseline_no_col(X, p):
    """
    Baseline where we ignore column marginals and only use X and p.
    """
    row_sums = X.sum(axis=1)
    row_factors = p / row_sums
    row_factors[row_sums == 0] = 0
    est_mat = np.diag(row_factors) @ X
    assert np.isclose(np.sum(est_mat, axis=1), p).all()
    return est_mat

def baseline_no_row(X, q):
    """
    Baseline where we ignore row marginals and only use X and q.
    """
    col_sums = X.sum(axis=0)
    col_factors = q / col_sums
    col_factors[col_sums == 0] = 0
    est_mat = X @ np.diag(col_factors)
    assert np.isclose(np.sum(est_mat, axis=0), q).all()
    return est_mat

def baseline_scale_mat(X, total):
    """
    Baseline where we rescale X so that its total is equal to the hourly total.
    """
    curr_total = X.sum()
    est_mat = X * total / curr_total
    assert np.isclose(est_mat.sum(), total)
    return est_mat

def evaluate_results_on_bikeshare(dt, methods=None):
    """
    Evaluate results from different methods over 24 hours of bikeshare data for a given day.
    """
    assert (dt >= datetime.datetime(2023, 9, 1)) and (dt < datetime.datetime(2023, 10, 1))
    if methods is None:
        methods = ['ipf_month', 'ipf_week', 'ipf_day', 
                   'baseline_no_mat', 'baseline_no_col', 'baseline_no_row',
                   'baseline_scale_month', 'baseline_scale_week', 'baseline_scale_day']
    with open('bikeshare-202309.pkl', 'rb') as f:
        hours, networks = pickle.load(f)
    
    Xs = {}
    Xs['month'] = prep_bikeshare_data_for_ipf(dt, timeagg='month', hours=hours, networks=networks)[0]
    Xs['week'] = prep_bikeshare_data_for_ipf(dt, timeagg='week', hours=hours, networks=networks)[0]
    Xs['day'] = prep_bikeshare_data_for_ipf(dt, timeagg='day', hours=hours, networks=networks)[0]
    l2_dict = {m:[] for m in methods}
    pearson_dict = {m:[] for m in methods}

    for hr in range(24):
        curr_dt = datetime.datetime(year=dt.year, month=dt.month, day=dt.day, hour=hr)
        print('\n', curr_dt.strftime('%Y-%m-%d-%H'))
        hour_idx = hours.index(curr_dt)
        true_mat = networks[hour_idx].toarray()
        p = true_mat.sum(axis=1)  # row marginals
        q = true_mat.sum(axis=0)  # column marginals
        for m in methods:
            est_mat = _get_estimated_matrix(Xs, p, q, curr_dt, m)
            if est_mat is not None:
                l2, pearson = eval_est_mat(est_mat, true_mat, verbose=False)
                print(m, 'L2=%.3f' % l2, 'Pearson r=%.3f (p=%.3f)' % pearson)
            else:
                l2, pearson = np.nan, (np.nan, np.nan)
            l2_dict[m].append(l2)
            pearson_dict[m].append(pearson[0])
    return l2_dict, pearson_dict
                
                
def _get_estimated_matrix(Xs, p, q, dt, method):
    """
    Helper method to get the estimated matrix for a given hour.
    """
    if method.startswith('ipf_'):
        timeagg = method.split('_', 1)[1]
        fn = 'ipf-output/bikeshare_%s_%s.pkl' % (timeagg, dt.strftime('%Y-%m-%d-%H'))
        if os.path.isfile(fn):
            with open(fn, 'rb') as f:
                ipf_out = pickle.load(f)
            row_factors, col_factors = ipf_out[1], ipf_out[2]
            X = Xs[timeagg]
            est_mat = np.diag(row_factors) @ X @ np.diag(col_factors)
        else:
            print('File is missing:', fn)
            est_mat = None
    elif method.startswith('baseline_scale_'):
        timeagg = method.rsplit('_', 1)[1]
        X = Xs[timeagg]
        est_mat = baseline_scale_mat(X, np.sum(p))
    else:
        assert method.startswith('baseline_no_')
        if method == 'baseline_no_mat':
            est_mat = baseline_no_mat(p, q)
        elif method == 'baseline_no_col':
            est_mat = baseline_no_col(Xs['month'], p)
        else:
            assert method == 'baseline_no_row'
            est_mat = baseline_no_row(Xs['month'], q)
    return est_mat

        
####################################################################
# Compare IPF to Poisson regression
####################################################################
def run_poisson_experiment(X, p, q, Y=None, package='sm', method='IRLS'):
    """
    Do Poisson regression on IPF inputs.
    X, p, q: IPF inputs.
    Y: response variable. If Y is not provided, we use max-flow algorithm to find appropriate Y.
    package: either 'sm' for statsmodels or 'sklearn' for scikit-learn.
    method: method used for fitting model. Default is 'IRLS' (iteratively reweighted least squares), 
        which is default for statsmodels. Another option is 'lbfgs' (default for sklearn).
    """
    assert (p > 0).all(), 'Row marginals must be positive for Poisson regression to converge'
    assert (q > 0).all(), 'Col marginals must be positive for Poisson regression to converge'
    assert package in ['sm', 'sklearn']
    if package == 'sklearn':
        raise Exception('Poisson regression not fully implemented for sklearn yet')
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

    # construct response variable
    if Y is None:  # get Y by running max-flow algorithm
        G, f_val, Y = test_ipf_convergence_from_max_flow(X, p, q, return_flow_mat=True)
        assert np.isclose(f_val, np.sum(p))  # IPF should converge
    assert (Y[X == 0] == 0).all()  # Y should inherit all zeros of X
    assert np.isclose(np.sum(Y, axis=1), p).all()  # Y should have target row marginals
    assert np.isclose(np.sum(Y, axis=0), q).all()  # Y should have target col marginals
    resp = Y[row_nnz, col_nnz]
    print('Constructed response variable:', resp.shape)
    
    if package == 'sm':  # statsmodels
        ts = time.time()
        offset = np.log(X[row_nnz, col_nnz])  # include in linear model with coefficient of 1
        mdl = sm.GLM(resp, onehots, offset=offset, family=sm.families.Poisson())
        print('Initialized Poisson model [time=%.3fs]' % (time.time()-ts))
        ts = time.time()
        result = mdl.fit(method=method, maxiter=1000)
        print('Finished fitting model with statsmodels, method %s [time=%.3fs]' % (method, time.time()-ts))
    else:  # sklearn
        ts = time.time()
        # TODO: figure out how to include offset
        mdl = PoissonRegressor(alpha=0, fit_intercept=False, solver=method)
        mdl.fit(onehots, resp)
        result = None
        print('Finished fitting model with sklearn, method %s [time=%.3fs]' % (method, time.time()-ts))
    return mdl, result

def test_poisson_with_mobility_data(dt, method):
    """
    Function to test Poisson regression on mobility data. Note: this function takes hours to run,
    since Poisson regression takes very long on large number of parameters.
    """
    msa_name = 'Richmond_VA'
    msa_df_date_range = '20200302_20200608'
    X, p, q = prep_safegraph_data_for_ipf(msa_name, dt, msa_df_date_range)
    print('Date: %s, marginals prop positive -> POIs = %.3f, CBGs = %.3f' % (
        dt.strftime('%Y-%m-%d-%H'), np.mean(p > 0), np.mean(q > 0)))
    
    # keep submatrix with nonzero row and column marginals
    nonzero_rows = p > 0
    X = X[nonzero_rows]
    p = p[nonzero_rows]
    nonzero_cols = q > 0
    X = X[:, nonzero_cols]
    q = q[nonzero_cols]
    print('Shape without zero marginals:', X.shape, len(p), len(q))
    
    fn = f'poisson-{method}.pkl'
    print('Will save results in', fn)
    mdl, result = run_poisson_experiment(X, p, q, Y=None, method=method)
    print(result.summary())
    with open(fn, 'wb') as f:
        pickle.dump((result.params, result.conf_int(alpha=0.05), result.conf_int(alpha=0.1)), f)
    
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
    # test_recoverable_process(sparsity_rate=0.8, seed=1)
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, choices=['safegraph', 'bikeshare'])
    parser.add_argument('year', type=int, choices=[2020, 2021, 2022, 2023])
    parser.add_argument('month', type=int, choices=np.arange(1, 13, dtype=int))
    parser.add_argument('day', type=int, choices=np.arange(1, 32, dtype=int))
    parser.add_argument('--hour', type=int, default=0, choices=np.arange(0, 25, dtype=int))
    parser.add_argument('--msa_name', default='Richmond_VA', type=str)  # only for SafeGraph
    parser.add_argument('--timeagg', default='month', choices=['month', 'week', 'day'], type=str)  # only for bikeshare
    parser.add_argument('--max_iter', type=int, default=10000)
    parser.add_argument('--mode', default='outer', choices=['outer', 'inner'])
    args = parser.parse_args()

    dt = datetime.datetime(args.year, args.month, args.day, args.hour)
    if args.data == 'safegraph':
        # hours to try: 2020/03/02 and 2020/04/06
        if args.mode == 'outer':
            run_safegraph_all_hours_in_day(args.msa_name, dt)
        else:
            # msa_df_date_range = '20191230_20200224'
            msa_df_date_range = '20200302_20200608'
            run_safegraph_ipf_experiment(args.msa_name, dt, msa_df_date_range, max_iter=args.max_iter)
    else:
        if args.mode == 'outer':
            run_bikeshare_all_hours_in_day(dt, timeagg=args.timeagg, max_iter=args.max_iter)
        else:
            run_bikeshare_ipf_experiment(dt, timeagg=args.timeagg, max_iter=args.max_iter)
    
#     dt = datetime.datetime(2020, 3, 2, 12)
#     method = 'irls'
#     test_poisson_with_mobility_data(dt, method)