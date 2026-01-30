import argparse
import pandas as pd
import re

def load_dataset(path):

	data = pd.read_csv(path, header=3)
	timestamp = data.iloc[:, 0].to_numpy()
	output_voltage = data.iloc[:, 1].to_numpy()
	output_current = data.iloc[:, 2].to_numpy()
	thermistor_temperatures = data.iloc[:, 3:].to_numpy()
	
	start_index = path.rfind('_') + 1
	end_index = path.find('.csv', start_index)
	freq = path[start_index:end_index]
	with open(path) as f:
		text = f.read()
	match = re.search(r"comments:\s*(.*)", text, re.IGNORECASE)
	if match:
		comments = match.group(1)
	else:
		comments = None
	return timestamp, output_voltage, output_current, thermistor_temperatures, comments, freq

def plot_dataset(timestamp, output_voltage, thermistor_temperatures, freq):
	import matplotlib.pyplot as plt

	plt.figure(figsize=(12, 8))

	plt.subplot(2, 1, 1)
	plt.plot(timestamp, output_voltage, label='output voltage')
	plt.xlabel('time')
	plt.ylabel('voltage')
	plt.legend()

	plt.subplot(2, 1, 2)
	for i in range(thermistor_temperatures.shape[1]):
		plt.plot(timestamp, thermistor_temperatures[:, i], label=f'thermistor {i+1}')
	plt.xlabel('time')
	plt.ylabel('temperature degrees')
	plt.legend()
	plt.title(f'frequency {freq}')
	plt.tight_layout()
	plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
args = parser.parse_args()
timestamp, output_voltage, output_current, thermistor_temperatures, comments, freq = load_dataset(args.dataset)
plot_dataset(timestamp, output_voltage, thermistor_temperatures, freq)

