import numpy as np
import matplotlib.pyplot as plt
import random
import csv


# function calculating linear combination of spectra
def gen_spec(data, max_shifts='assets\max_shifts.csv', c=np.array([random.random() for index in range(20)])):
    for index in range(20):
        array = data[index]
        data[index][:, 1][(array[:, 0] > -0.5) & (array[:, 0] < 0.5)] = 0

    with open(max_shifts) as file:
        mshifts_rows = [row for row in csv.reader(file) if row.index != 0]
        mshifts_rows.pop(0)

    max_shift_list = list(sorted(dict(mshifts_rows).items(), key=lambda x: x[0].lower()))
    shift_list = np.array([random.uniform(-float(max_shift_list[index][1]), float(max_shift_list[index][1])) for index in range(20)])

    new_data_x = np.array([data[index][:, 0] + shift_list[index] for index in range(20)])
    new_data_y = np.array([c[index] * data[index][:, 1] for index in range(20)])
    z = np.dstack((new_data_x, new_data_y))
    list_cut = []

    for i in range(20):
        g = z[i].copy()
        g = g[(g[:, 0] > -4.5) & (g[:, 0] < 14)].copy()
        if (len(g[(g[:, 0] > -4.5) & (g[:, 0] < 14)]) == 60607):
            g = np.vstack((np.array([14, 0]), g))
        list_cut.append(g)

    spec_cut = np.array(list_cut)
    spectrum_linear_comb_cut = np.sum(spec_cut[:, :, 1], axis=0)
    return (spec_cut[12, :, 0], spectrum_linear_comb_cut)


# function for adding white noise to spectra linear combination
def addnoise(spectra_sum, ampl, mu, sigma, n):
    return spectra_sum + ampl * np.random.normal(mu, sigma, n)


spec_real_data = np.loadtxt(f'assets\mix_metb001.txt')
data = [np.loadtxt(f'assets\met{i + 1}.txt') for i in range(20)]
sizes_of_points = 60608 * [1]
sizes_of_points_for_real = len(spec_real_data[:, 0]) * [1]

y = data.copy()
y[0][:, 1][(y[0][:, 0] < -0.5) | (y[0][:, 0] > 0.5)] = 0
signal_zero = y[0]

# setting 0 intensity near 0 ppm for 2nd - 20th  NMR spectra
next_data = gen_spec(data)
plt.figure(1)
plt.scatter(next_data[0], addnoise(next_data[1], 1000, 0, 1, 60608), sizes_of_points, 'blue')
plt.scatter(spec_real_data[:, 0], 1.25*spec_real_data[:, 1], sizes_of_points_for_real, 'orange')
plt.show()
