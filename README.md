# IPF for dynamic network inference

This respository contains code to run all experiments for "Inferring dynamic networks from marginals with iterative proportional fitting". The main files are:

- **test_ipf.py**: functions to run IPF, test and enable IPF convergence (our ConvIPF algorithm), and run the equivalent Poisson regression.
- **experiments_with_data.py**: functions to run experiments with data - synthetic data, SafeGraph mobility data, and CitiBike bikeshare data
- **icml_experiments.ipynb**: notebook to recreate all results and figures reported in the paper.

The other files are imported from other [repos](https://github.com/snap-stanford/covid-mobility-tool) (Chang et al., 2021) to use SafeGraph mobility data.
