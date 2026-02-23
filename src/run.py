import argparse
import numpy as np
import pandas as pd
import re
from pathlib import Path
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from glob import glob
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.hierarchical import UnifiedHierarchical


def load_dataset(path):
    data = pd.read_csv(path, header=3)
    timestamp = data.iloc[:, 0].to_numpy()
    thermistor_temps = data.iloc[:, 3:].to_numpy()
    match = re.search(r'_(\d+)s\.csv', path)
    period_str = match.group(1) if match else None
    return timestamp, thermistor_temps, period_str


def load_all_datasets(data_dir, pattern='al_*.csv', skip_initial=1000):
    files = sorted(glob(str(Path(data_dir) / pattern)))
    datasets = []

    for fpath in files:
        timestamp, thermistor_temps, period_str = load_dataset(fpath)
        t = timestamp[skip_initial:] - timestamp[skip_initial]
        temp = thermistor_temps[skip_initial:, :].T

        period = float(period_str)
        omega = 2 * np.pi / period

        datasets.append({
            't': t,
            'temp': temp,
            'omega': omega,
            'period': period_str,
            'file': Path(fpath).name
        })

        print(f"loaded {Path(fpath).name}: period={period}s, omega={omega:.4f}, shape={temp.shape}")

    return datasets


def run_inference(model, n_samples=2000, n_tune=1000, n_chains=2):
    with model:
        trace = pm.sample(draws=n_samples,
            tune=n_tune, chains=n_chains, target_accept=0.95,
            return_inferencedata=True,random_seed=42
        )
    return trace


def plot_results(trace, datasets, x_sensors):
    var_names = list(trace.posterior.data_vars.keys())

    global_vars = [v for v in ['D', 'gamma', 'H'] if v in var_names]
    if global_vars:
        fig = az.plot_trace(trace, var_names=global_vars, compact=False)
        plt.tight_layout()
        plt.savefig('results/hierarchical_trace_global.png', dpi=150)
        plt.close()
        fig = az.plot_posterior(trace, var_names=global_vars, hdi_prob=0.95)
        plt.tight_layout()
        plt.savefig('results/hierarchical_posterior_global.png', dpi=150)
        plt.close()

    freq_vars = [v for v in ['Tbar', 'A', 'phi'] if v in var_names]
    if len(datasets) > 0 and freq_vars:
        fig = az.plot_forest(trace, var_names=freq_vars, combined=True, figsize=(10, 8))
        plt.tight_layout()
        plt.savefig('results/hierarchical_forest_freqparams.png', dpi=150)
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='../data')
    parser.add_argument('--pattern', type=str, default='al_*.csv')
    parser.add_argument('--model', type=str, default='FullSolutionNewton',
                       choices=['SimpleForward', 'FullSolution', 'FullSolutionNewton', 'FullSolutionRobin'])
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--tune', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=2)
    parser.add_argument('--skip', type=int, default=1000)
    parser.add_argument('--L', type=float, default=0.046)
    args = parser.parse_args()
    x_sensors = np.array([0.003, 0.008, 0.013, 0.018, 0.023, 0.028, 0.033, 0.043])

    print(f"using forward model: {args.model}, L={args.L}m")
    datasets = load_all_datasets(args.data_dir, args.pattern, args.skip)

    model_instance = UnifiedHierarchical({})
    pymc_model = model_instance.build_model(datasets, x_sensors, L=args.L, model_type=args.model)
    print("model built!")
    trace = run_inference(pymc_model, args.samples, args.tune, args.chains)

    if args.model in ['FullSolutionNewton', 'FullSolutionRobin']:
        print(az.summary(trace, var_names=['D', 'gamma'], hdi_prob=0.95))
    else:
        print(az.summary(trace, var_names=['D'], hdi_prob=0.95))

    print(az.summary(trace, var_names=['Tbar', 'A', 'phi'], hdi_prob=0.95))

    Path('results').mkdir(exist_ok=True)
    plot_results(trace, datasets, x_sensors)
    trace.to_netcdf('results/hierarchical_trace.nc')
if __name__ == '__main__':
    main()
