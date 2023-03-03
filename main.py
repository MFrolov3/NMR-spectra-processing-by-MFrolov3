import numpy as np
import matplotlib.pyplot as plt
import random

NUMBERSPEC = 10
x_shift = 0.13
spectrum_linear_comb = []


# function for adding white noise to spectra linear combination
def addnoise(spectra_sum,ampl, mu, sigma, n):
    return spectra_sum + ampl*np.random.normal(mu, sigma, n)

spec_real_data = np.loadtxt(f'assets\mix_metb001.txt')
data = [np.loadtxt(f'assets\met{i+1}.txt') for i in range(20)]
sizes_of_points = len(data[12][:, 0]) * [1]
sizes_of_points_for_real = len(spec_real_data[:, 0]) * [1]

#setting 0 intensity near 0 ppm for 2nd - 20th  NMR spectra
for iter in range(1, 20):
    array = data[iter]
    data[iter][:, 1][(array[:, 0] > -1) & (array[:, 0] < 1)] = 0

for index in range(NUMBERSPEC):
    c = np.array([random.random() for index in range(20)])
    new_data = np.array([c[index] * data[index][:, 1] for index in range(20)])
    spectrum_linear_comb.append(np.sum(new_data, axis=0))

plt.scatter(spec_real_data[:, 0], 1.25*spec_real_data[:, 1], sizes_of_points_for_real)
plt.scatter(data[12][:, 0] + x_shift, addnoise(spectrum_linear_comb[1], 100000, 0, 1, 65536), sizes_of_points)
plt.show()
