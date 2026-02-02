import numpy as np
from scipy.optimize import curve_fit 
import matplotlib.pyplot as plt
import os

def load_dataset(path): #load a data set
	import pandas as pd
	import re
	
	data = pd.read_csv(path, header=3)
	timestamp = data.iloc[:, 0].to_numpy()
	output_voltage = data.iloc[:, 1].to_numpy()
	output_current = data.iloc[:, 2].to_numpy()
	thermistor_temperatures = data.iloc[:, 3:].to_numpy()

	comments = re.search(r"Comments: (.*)$", open(path).read(), re.MULTILINE)[1]

	return timestamp, output_voltage, output_current, thermistor_temperatures, comments

def sin_fit(t, A, phi, omega, T_0):
	return A * np.sin(omega * t + phi) + T_0

timestamp, output_voltage, output_current, thermistor_temperatures, comments = (
	load_dataset(r"/Users/isaacreid/nonlinear-heat/data/al_25xs.csv")
)

popt, pcov = curve_fit(sin_fit, timestamp)