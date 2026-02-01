"""
pymc bayesian inference for thermal wave propagation anal 
models supported:
- SimpleForward: single propagating wave (high-frequency limit)
- FullSolution: finite rod with adiabatic boundary
- FullSolutionNewton: finite rod with Newton cooling (heat loss)
- FullSolutionRobin: finite rod with Robin BC (convective loss at tip)
"""

import argparse
import numpy as np
import pandas as pd
import re
import sys
from pathlib import Path
import pymc as pm
import arviz as az
# import pymc3 as pm
# import arviz as az
import matplotlib.pyplot as plt
from scipy import signal
from models.models import SimpleForward, FullSolution, FullSolutionNewton, FullSolutionRobin


def load_dataset(path):
    data = pd.read_csv(path, header=3)
    timestamp = data.iloc[:, 0].to_numpy()
    output_voltage = data.iloc[:, 1].to_numpy()
    output_current = data.iloc[:, 2].to_numpy()
    thermistor_temperatures = data.iloc[:, 3:].to_numpy()    
    start_index = path.rfind('_') + 1
    end_index = path.find('.csv', start_index)
    period_str = path[start_index:end_index]

    with open(path) as f:
        text = f.read()
    match = re.search(r"comments:\\s*(.*)", text, re.IGNORECASE)
    comments = match.group(1) if match else None    
    return timestamp, output_voltage, output_current, thermistor_temperatures, comments, period_str

MODEL_REGISTRY = {
    'SimpleForward': SimpleForward,
    'FullSolution': FullSolution,
    'FullSolutionNewton': FullSolutionNewton,
    'FullSolutionRobin': FullSolutionRobin}

def get_model_class(model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"{model_name} invalid")
    return MODEL_REGISTRY[model_name]

