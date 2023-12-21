import helper_methods_for_aggregate_data_analysis as helper
from model_experiments import fit_disease_model_on_real_data
from test_ipf import *

import argparse
import datetime
import itertools
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
import os
import pandas as pd
import pickle
import random
from scipy.sparse import csr_matrix
from scipy.linalg import eig
from scipy import optimize
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import pairwise_distances
import statsmodels.api as sm
import time


####################################################################
# Experiments with synthetic data
####################################################################
def generate_X(m, n, dist='uniform', seed=0, sparsity_rate=0, exact_rate=False, verbose=True):
    """
    Generate X based on kwargs.
    sparsity_rate: each entry is set to 0 with probability sparsity_rate.
    """
    np.random.seed(seed)
    assert dist in {'uniform', 'poisson'}
    if verbose:
        print(f'Sampling X from {dist} distribution')
    if dist == 'uniform':
        X = np.random.rand(m, n)
    elif dist == 'poisson':
        X = np.random.poisson(lam=10, size=(m,n))
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
        X[set_to_0] = 0
        if verbose:
            print('Num nonzero entries in X: %d out of %d' % (np.sum(X > 0), m*n))
    return X

def generate_row_and_col_factors(m, n, seed=0, scalar=4):
    """
    Generate ground-truth row factors and column factors.
    """
    np.random.seed(seed)
    row_factors = np.random.rand(m) * scalar
    col_factors = np.random.rand(n) * scalar
    return row_factors, col_factors
    
def generate_hourly_network(X, row_factors, col_factors, model='basic',
                            F=None, beta=None, prev_hour=None, seed=0):
    """
    Generate hourly network based on time-aggregated network X and hourly row/column factors,
    and potentially other information. 'model' defines which model is being used.
    """
    np.random.seed(seed)
    if model == 'basic':
        means = np.diag(row_factors) @ X @ np.diag(col_factors)
        Y = np.random.poisson(means)
    elif model == 'interaction':
        assert F is not None and beta is not None
        means = (np.diag(row_factors) @ X @ np.diag(col_factors)) * (F ** beta)
        Y = np.random.poisson(means)
    elif model == 'autocorrelated':
        assert prev_hour is not None and beta is not None
        # TODO
    else:
        assert model == 'negative_binom'
        # TODO
    return Y


####################################################################
# Experiment with SafeGraph mobility data
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
    X, p, q = _prep_safegraph_data_for_ipf(m.POI_TIME_COUNTS, m.cbg_day_prop_out, 
                                          m.CBG_SIZES, m.POI_CBG_PROPORTIONS.toarray(), dt.hour)
    return X, p, q
    
    
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
    p = poi_visits * prop_poi_kept
    q = cbg_visits
    q = q * np.sum(p) / np.sum(q)  # renormalize to match row sums
    assert np.isclose(np.sum(p), np.sum(q))
    return poi_cbg_props, p, q
    
    
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
    X, p, q = prep_safegraph_data_for_ipf(msa_name, dt, msa_df_date_range)
    print('Date: %s, marginals prop positive -> POIs = %.3f, CBGs = %.3f' % (
        dt.strftime('%Y-%m-%d-%H'), np.mean(p > 0), np.mean(q > 0)))
    ts = time.time()
    ipf_out = do_ipf(X, p, q, num_iter=max_iter)
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
        cmd = f'nohup python -u experiments_with_data.py ipf_single_hour safegraph --msa_name {msa_name} --hour {hr} > {out_file} 2>&1 &'
        print(cmd)
        os.system(cmd)
        time.sleep(1)
        
        
def poisson_regression_on_safegraph_data(dt, method):
    """
    Function to test Poisson regression on SafeGraph data. Note: this function takes hours to run,
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


####################################################################
# Experiments with bikeshare data from CitiBike
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

def get_distances_between_stations():
    """
    Get pairwise distances between bike stations.
    """
    stations = pd.read_csv('202309-bike-stations.csv').sort_values('station_num')
    locations = stations[['lat_mean', 'lng_mean']].values
    pairwise_dist = pairwise_distances(locations)
    return pairwise_dist

    
def l2_norm(mat):
    """
    Return L2 norm of a matrix.
    """
    return np.sqrt(np.sum(mat ** 2))
    
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
    norm_l2 = l2_norm(est_mat - real_mat) / l2_norm(real_mat)
    if verbose:
        print('Normalized L2 distance', norm_l2)
    corr = pearsonr(real_mat.flatten(), est_mat.flatten())
    if verbose:
        print('Pearson corr', corr)
    return norm_l2, corr
    

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
        cmd = f'nohup python -u experiments_with_data.py ipf_single_hour bikeshare {dt.year} {dt.month} {dt.day} --hour {hr} --timeagg {timeagg} --max_iter {max_iter} > {out_file} 2>&1 &'
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


def poisson_regression_on_bikeshare_data(dt, timeagg, model='basic'):
    assert model in ['basic', 'interaction', 'autocorrelated', 'neg_binom']
    """
    Function to test Poisson regression on bikeshare data. Note: this function takes hours to run,
    since Poisson regression takes very long on large number of parameters.
    """
    X, p, q, true_mat = prep_bikeshare_data_for_ipf(dt, timeagg)
    
    # keep submatrix with nonzero row and column marginals
    nonzero_rows = p > 0
    p = p[nonzero_rows]
    nonzero_cols = q > 0
    q = q[nonzero_cols]
    X = X[nonzero_rows][:, nonzero_cols]
    true_mat = true_mat[nonzero_rows][:, nonzero_cols]
    print('Shape without zero marginals:', X.shape, len(p), len(q))
    
    if model == 'interaction':
        # provide inverse distance as a feature
        distances = get_distances_between_stations()
        distances = np.clip(distances, 0.001, None)  # clip so we don't have distance of 0
        F = 1/distances
        F = F[nonzero_rows][:, nonzero_cols]
    else:
        F = None
    
    fn = 'poisson-bikeshare-%s-%s-%s.pkl' % (dt.strftime('%Y-%m-%d-%H'), timeagg, model)
    print('Will save results in', fn)
    mdl, result = run_poisson_experiment(X, p, q, Y=true_mat, F=F)
    print(result.summary())
    with open(fn, 'wb') as f:
        pickle.dump((result.params, result.conf_int(alpha=0.05), result.conf_int(alpha=0.1), result.llf), f)

        
def compute_residuals(obs, exp, residual_type):
    """
    Compute Pearson or deviance residuals.
    """
    if residual_type == 'pearson':
        residuals = (obs - exp) / np.sqrt(exp)
    else:
        assert residual_type == 'deviance'
        first_term = obs * np.log(obs / exp)
        first_term[np.isclose(obs, 0)] = 0
        second_term = obs - exp
        residuals = np.sign(obs - exp) * np.sqrt(2 * (first_term - second_term))
    return residuals
    
def analyze_poisson_residuals(dt, timeagg, dist_mat=None, residual_type='pearson', verbose=True):
    """
    Analyze either Pearson or deviance residuals for fitted Poisson model / IPF estimates.
    """
    assert residual_type in ['pearson', 'deviance']
    X, p, q, true_mat = prep_bikeshare_data_for_ipf(dt, timeagg)
    fn = 'ipf-output/bikeshare_%s_%s.pkl' % (timeagg, dt.strftime('%Y-%m-%d-%H'))
    with open(fn, 'rb') as f:
        ipf_out = pickle.load(f)
    row_factors, col_factors = ipf_out[1], ipf_out[2]
    est_mat = np.diag(row_factors) @ X @ np.diag(col_factors)
    
    # we only keep observations where X_ij > 0, p_i > 0, and q_j > 0
    X_keep = X[p>0][:,q>0]
    if verbose:
        print('Kept shape with nonzero marginals:', X_keep.shape)
    obs = true_mat[p>0][:,q>0][X_keep > 0]  # observed data
    is_zero = np.isclose(obs, 0).sum()
    exp = est_mat[p>0][:,q>0][X_keep > 0]  # expected values
    assert np.isclose(exp, 0).sum() == 0
    if verbose:
        print('Num observations kept: %d, %d (%.2f%%) is zero' % (len(obs), is_zero, 100. * is_zero / len(obs)))
        print('Corr between observed values and expected values: r=%.3f, p=%.3f' % pearsonr(obs, exp))
    
    num_obs = len(obs)
    num_params = (p>0).sum() + (q>0).sum()
    residuals = compute_residuals(obs, exp, residual_type)
    corr = pearsonr(exp, residuals)
    if verbose:
        print(f'Corr between expected values and {residual_type} residuals: r=%.3f, p=%.3f' % corr)
    
    if dist_mat is not None:  # pairwise distances between stations provided
        assert dist_mat.shape == X.shape
        dist_keep = dist_mat[p>0][:,q>0][X_keep>0]
        assert dist_keep.shape == residuals.shape
        corr = pearsonr(dist_keep, residuals)
        if verbose:
            print(f'Corr btwn distances and {residual_type} residuals: r=%.3f, p=%.3f' % corr)
        return num_params, residuals, exp, dist_keep
    return num_params, residuals, exp


def get_consecutive_residuals(dt1, timeagg, residual_type='pearson'):
    """
    Analyze either Pearson or deviance residuals for fitted Poisson model / IPF estimates.
    """
    X, p1, q1, true_mat1 = prep_bikeshare_data_for_ipf(dt1, timeagg)
    fn = 'ipf-output/bikeshare_%s_%s.pkl' % (timeagg, dt1.strftime('%Y-%m-%d-%H'))
    with open(fn, 'rb') as f:
        ipf_out = pickle.load(f)
    row_factors, col_factors = ipf_out[1], ipf_out[2]
    est_mat1 = np.diag(row_factors) @ X @ np.diag(col_factors)

    dt2 = dt1 + datetime.timedelta(hours=1)
    X, p2, q2, true_mat2 = prep_bikeshare_data_for_ipf(dt2, timeagg)
    fn = 'ipf-output/bikeshare_%s_%s.pkl' % (timeagg, dt2.strftime('%Y-%m-%d-%H'))
    with open(fn, 'rb') as f:
        ipf_out = pickle.load(f)
    row_factors, col_factors = ipf_out[1], ipf_out[2]
    est_mat2 = np.diag(row_factors) @ X @ np.diag(col_factors)

    row_keep = (p1 > 0) & (p2 > 0)
    col_keep = (q1 > 0) & (q2 > 0)
    X_keep = X[row_keep][:, col_keep]
    obs1 = true_mat1[row_keep][:, col_keep][X_keep > 0]
    exp1 = est_mat1[row_keep][:, col_keep][X_keep > 0]
    obs2 = true_mat2[row_keep][:, col_keep][X_keep > 0]
    exp2 = est_mat2[row_keep][:, col_keep][X_keep > 0]
    print('Num pairs left:', len(obs1))

    res1 = compute_residuals(obs1, exp1, residual_type)
    res2 = compute_residuals(obs2, exp2, residual_type)
    return res1, res2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['ipf_all_hours', 'ipf_single_hour', 'poisson'])
    parser.add_argument('data', type=str, choices=['safegraph', 'bikeshare'])
    parser.add_argument('year', type=int, choices=[2020, 2021, 2022, 2023])
    parser.add_argument('month', type=int, choices=np.arange(1, 13, dtype=int))
    parser.add_argument('day', type=int, choices=np.arange(1, 32, dtype=int))
    parser.add_argument('--hour', type=int, default=0, choices=np.arange(0, 25, dtype=int))
    parser.add_argument('--msa_name', default='Richmond_VA', type=str)  # only for SafeGraph
    parser.add_argument('--timeagg', default='month', choices=['month', 'week', 'day'], type=str)  # only for bikeshare
    parser.add_argument('--max_iter', type=int, default=10000)
    parser.add_argument('--poisson_model', type=str, default='basic', choices=['basic', 'interaction', 
                                                            'autocorrelated', 'neg_binom'])  # only for mode=poisson
    args = parser.parse_args()

    dt = datetime.datetime(args.year, args.month, args.day, args.hour)
    if args.data == 'safegraph':
        # hours to try: 2020/03/02 and 2020/04/06
        if args.mode == 'ipf_all_hours':
            run_safegraph_all_hours_in_day(args.msa_name, dt)
        elif args.mode == 'ipf_single_hour':
            # msa_df_date_range = '20191230_20200224'
            msa_df_date_range = '20200302_20200608'
            run_safegraph_ipf_experiment(args.msa_name, dt, msa_df_date_range, max_iter=args.max_iter)
        else:
            assert args.mode == 'poisson'
            poisson_regression_on_safegraph_data(dt, 'IRLS')
    else:
        if args.mode == 'ipf_all_hours':
            run_bikeshare_all_hours_in_day(dt, args.timeagg, args.max_iter)
        elif args.mode == 'ipf_single_hour':
            run_bikeshare_ipf_experiment(dt, args.timeagg, args.max_iter)
        else:
            poisson_regression_on_bikeshare_data(dt, args.timeagg, model=args.poisson_model)