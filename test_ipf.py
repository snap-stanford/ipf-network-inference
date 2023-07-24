import helper_methods_for_aggregate_data_analysis as helper
from model_experiments import fit_disease_model_on_real_data

import argparse
import datetime
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import pickle
from scipy.stats import pearsonr
import time

####################################################################
# Functions to do IPF / test IPF convergence
####################################################################
def do_ipf(Z, u, v, start_iter=0, num_iter=100, row_factors=None, col_factors=None, 
           eps=1e-8, verbose=True,):
    """
    Z: initial matrix
    u: target row marginals
    v: target col marginals
    """
    assert Z.shape == (len(u), len(v))
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
        
    row_errs = []
    col_errs = []
    prev_M = None
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
     
        # get error from marginals
        est_M = np.diag(row_factors) @ Z @ np.diag(col_factors)
        row_err = np.sum(np.abs(u - np.sum(est_M, axis=1)))
        col_err = np.sum(np.abs(v - np.sum(est_M, axis=0)))
        row_errs.append(row_err)
        col_errs.append(col_err)
        if verbose:
            print('Iter %d: row err = %.4f, col err = %.4f' % (i, row_err, col_err))
        
        # check if converged
        if prev_M is not None and (np.sum(np.abs(est_M - prev_M)) < eps):
            if verbose:
                print(f'Converged; stopping after {i} iterations')
            return i, row_factors, col_factors, row_errs, col_errs
        # check if stuck in oscillation
        if len(row_errs) >= 6:
            row_alternating = np.tile([row_errs[-2], row_errs[-1]], 3)  # create repeating pair 
            col_alternating = np.tile([col_errs[-2], col_errs[-1]], 3)  # create repeating pair
            if np.isclose(row_errs[-6:], row_alternating).all() and np.isclose(col_errs[-6:], col_alternating).all():
                if verbose:
                    print(f'Stuck in oscillation; stopping after {i} iterations')
                return i, row_factors, col_factors, row_errs, col_errs
        prev_M = est_M
    return i, row_factors, col_factors, row_errs, col_errs


def test_ipf_convergence_from_choice_sets(A, p, q, sort_items=False, apportion_strategy='greedy'):
    """
    Test whether IPF will converge via choice set construction algorithm.
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
def generate_M(dist='uniform', m=100, n=200, seed=0, sparsity_rate=0, verbose=True):
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
# Prepare SafeGraph data for IPF
####################################################################
def prep_safegraph_data_for_ipf(poi_time_counts, cbg_day_prop_out, cbg_sizes, 
                                poi_cbg_props, t):
    """
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
    
def run_ipf_experiment(msa_name, dt, msa_df_date_range, max_iter=1000):
    """
    Do IPF on SafeGraph data for given datetime.
    msa_name: name of metropolitan statistical area
    dt: datetime object, with year, month, day, and hour
    msa_df_date_range: the date range that includes datetime; we store MSA dataframes stratified by date ranges
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
                                       # important - changed this from False to True
                                       return_model_without_fitting=True,
                                       attach_data_to_model=True,
                                       model_quality_dict=None,
                                       verbose=True)
    
    Z, u, v = prep_safegraph_data_for_ipf(m.POI_TIME_COUNTS, m.cbg_day_prop_out, 
                                          m.CBG_SIZES, m.POI_CBG_PROPORTIONS.toarray(), dt.hour)
    print('Date: %s, marginals prop positive -> POIs = %.3f, CBGs = %.3f' % (
        dt.strftime('%Y-%m-%d-%H'), np.mean(u > 0), np.mean(v > 0)))
    ipf_out = do_ipf(Z, u, v, num_iter=max_iter)
    fn = '%s_%s.pkl' % (msa_name, dt.strftime('%Y-%m-%d-%H'))
    print('Saving results in', fn)
    with open(fn, 'wb') as f:
        pickle.dump(ipf_out, f)
    
    
if __name__ == '__main__':
    # test_recoverable_process(sparsity_rate=0.8, seed=1)
    parser = argparse.ArgumentParser()
    parser.add_argument('msa_name', type=str)
    parser.add_argument('hour', type=int)
    parser.add_argument('--max_iter', type=int, default=1000)
    args = parser.parse_args()
    
    dt = datetime.datetime(2020, 3, 1, args.hour)  # use March 1, 2020 for now
    msa_df_date_range = '20191230_20200224'
    run_ipf_experiment(args.msa_name, dt, msa_df_date_range, max_iter=args.max_iter)