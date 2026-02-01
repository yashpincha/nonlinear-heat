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

timestamp, output_voltage, output_current, thermistor_temperatures, comments = (
	load_dataset(r'/Users/yash/nonlinear-heat/data/al_25s.csv')
)

print(thermistor_temperatures.shape)

print(len(thermistor_temperatures))
print(len(thermistor_temperatures[0]))
def remove_phase_diff(thermistor_temperatures):
    temps_phase_removed = []
    i0s = []

    for j in range(len(thermistor_temperatures[0])):
        mean = np.mean(thermistor_temperatures[:, j])
        detrended = thermistor_temperatures[:, j] - mean
        sign_change = np.where(np.sign(detrended[:-1]) < np.sign(detrended[1:]))[0]
        i0 = sign_change[0] + 1
        temps_phase_removed.append(detrended[i0:] + mean)
        i0s.append(i0)

    return temps_phase_removed, i0s
temps_phase_removed, i0 = remove_phase_diff(thermistor_temperatures)

# for i in range(thermistor_temperatures.shape[1]):
# 	plt.plot(timestamp[i0[i]:], temps_phase_removed[i])
# plt.show()