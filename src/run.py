"""
PyMC Bayesian inference for thermal wave propagation analysis.
Uses existing model classes which already have build_model methods.
"""

import argparse
import numpy as np
import pandas as pd
import re
from pathlib import Path
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.models import SimpleForward, FullSolution, FullSolutionNewton, FullSolutionRobin

def load_dataset(path):
    data = pd.read_csv(path, header=3)
    timestamp = data.iloc[:, 0].to_numpy()
    output_voltage = data.iloc[:, 1].to_numpy()
    output_current = data.iloc[:, 2].to_numpy()
    thermistor_temps = data.iloc[:, 3:].to_numpy()
    start_index = path.rfind('_') + 1
    end_index = path.find('.csv', start_index)
    period_str = path[start_index:end_index]    
    with open(path) as f:
        text = f.read()
    match = re.search(r"comments:\s*(.*)", text, re.IGNORECASE)
    comments = match.group(1) if match else None
    return timestamp, output_voltage, output_current, thermistor_temps, comments, period_str


MODEL_REGISTRY = {
    'SimpleForward': SimpleForward,
    'FullSolution': FullSolution,
    'FullSolutionNewton': FullSolutionNewton,
    'FullSolutionRobin': FullSolutionRobin
}


def run_inference(model, n_samples=2000, n_tune=1000, n_chains=2, target_accept=0.95):
    '''run mcmc sampling'''
    with model:
        trace = pm.sample(draws=n_samples, tune=n_tune, chains=n_chains, target_accept=target_accept, return_inferencedata=True, random_seed=42)
    return trace

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, default='FullSolutionNewton', choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--sensor', type=int, default=0)
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--tune', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=2)
    parser.add_argument('--skip', type=int, default=1000)
    args = parser.parse_args()
    
    x_sensors = np.array([0.003, 0.008, 0.013, 0.018, 0.023, 0.028, 0.033, 0.043])    
    timestamp, _, _, thermistor_temps, comments, period_str = load_dataset(args.dataset)
    t = timestamp[args.skip:] - timestamp[args.skip]
    temp = thermistor_temps[args.skip:, args.sensor]
    x = x_sensors[args.sensor]
    period = float(period_str.replace('s', ''))
    omega = 2 * np.pi / period
    
    print(f"model: {args.model}")
    print(f"sensor {args.sensor} at x = {x*1000:.1f} mm")
    print(f"period: {period}s, omega: {omega:.4f} rad/s")
    print(f"data points: {len(t)}")
    
    ModelClass = MODEL_REGISTRY[args.model]
    model_instance = ModelClass({})
    pymc_model = model_instance.build_model(x, t, temp)    
    trace = run_inference(pymc_model, args.samples, args.tune, args.chains)

    print(az.summary(trace, hdi_prob=0.95))
    az.plot_trace(trace)
    plt.tight_layout()
    plt.savefig(f"results/{Path(args.dataset).stem}_{args.model}_trace.png", dpi=150)
    plt.show()
    az.plot_posterior(trace, hdi_prob=0.95)
    plt.tight_layout()
    plt.savefig(f"results/{Path(args.dataset).stem}_{args.model}_posterior.png", dpi=150)
    plt.show()
    
    print("done")

if __name__ == '__main__':
    main()
