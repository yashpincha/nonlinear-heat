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

directory = r"/Users/isaacreid/nonlinear-heat/data"
full_path = r"/Users/isaacreid/nonlinear-heat/data/al_70s.csv"

for file in os.listdir(directory):
	file_path = os.path.join(directory, file)
	if os.path.isfile(file_path):
		times, output_voltage, output_current, temps, comments = (load_dataset(file_path)) 
		plt.plot(times, temps)
		plt.title(file)
		plt.show()
		
"""times, output_voltage, output_current, temps, comments = (load_dataset(full_path)) 
plt.plot(times, temps)
plt.title(full_path)
plt.show()"""