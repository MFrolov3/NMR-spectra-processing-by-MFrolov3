import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import csv

METABOLITE_QUANTITY = 20
ZERO_NEIGHBORHOOD = 0.5


# function that extracts zero signal from first metabolite
def extract_zero(input_data):
    temp = input_data.copy()
    temp[0][:, 1][(temp[0][:, 0] < -ZERO_NEIGHBORHOOD)
                  | (temp[0][:, 0] > ZERO_NEIGHBORHOOD)] = 0
    signal_zero = temp[0]
    return signal_zero


# setting signal to zero
def remove_zero(in_data):
    input_data = in_data.copy()
    for index in range(METABOLITE_QUANTITY):
        array = input_data[index]
        input_data[index][:, 1][(array[:, 0] > -ZERO_NEIGHBORHOOD)
                                & (array[:, 0] < ZERO_NEIGHBORHOOD)] = 0
    return input_data


def get_max_shifts_list(max_shifts):
    # opening file that contains max_shifts
    with open(max_shifts) as file:
        shifts_rows = [row for row in csv.reader(file)]
        # deleting first string
        shifts_rows.pop(0)

    # sorting list of metabolites by alphabet
    return list(sorted(shifts_rows, key=lambda x: x[0].lower()))


def generate_shift_list(max_shift_list):
    border = [float(max_shift_list[index][1])
              for index in range(METABOLITE_QUANTITY)]
    return np.array([random.uniform(-border[index], border[index])
                     for index in range(20)])


# setting concentration random value
def set_concentration():
    c = np.random.uniform(0, 1, METABOLITE_QUANTITY)
    return 100 * c / np.sum(c)


# performing shifts by cutting numpy array from one side and adding
# missing datapoints with zero intensity from another
def shift_performing(input_data, joint, shift_list):
    shifted_lst = []
    for index in range(METABOLITE_QUANTITY):
        if shift_list[index] > 0:
            p = np.array(input_data.copy())
            p[index, :, 1] = 0
            temp1 = p[index][p[index, :, 0] < p[index, -1, 0] + shift_list[index]]
            temp2 = joint[index][joint[index, :, 0] < joint[index, 0, 0] - shift_list[index]]
            shifted_lst.append(np.vstack((temp2, temp1)))
        if shift_list[index] < 0:
            p = np.array(input_data.copy())
            p[index, :, 1] = 0
            temp1 = p[index][p[index, :, 0] > p[index, 0, 0] + shift_list[index]]
            temp2 = joint[index][joint[index, :, 0] > joint[index, -1, 0] - shift_list[index]]
            shifted_lst.append(np.vstack((temp1, temp2)))
    return shifted_lst


# delete extra data points in order to concatenate
# in numpy array when shifts are performed
def make_equal_length(data_input):
    tmp1 = data_input.copy()
    min_lst = []
    for index in range(METABOLITE_QUANTITY):
        min_lst.append(len(tmp1[index][:, 1]))
    u = min(min_lst)
    for index in range(METABOLITE_QUANTITY):
        while len(tmp1[index][:, 1]) != u:
            tmp1[index] = np.delete(tmp1[index], len(tmp1[index]) - 1, 0)
    return tmp1


# function calculating linear combination of spectra
def gen_spec(input_dat, max_shifts=r'assets\max_shifts.csv', n=1):
    input_data = input_dat.copy()
    input_data = remove_zero(input_data)
    max_shift_list = get_max_shifts_list(max_shifts)
    output_lst = []
    for index in range(n):
        output_c = set_concentration()
        shift_list = generate_shift_list(max_shift_list)

        new_data_x = np.array([input_data[index][:, 0] + shift_list[index]
                               for index in range(METABOLITE_QUANTITY)])
        new_data_y = np.array([output_c[index] * input_data[index][:, 1]
                               for index in range(METABOLITE_QUANTITY)])
        joint = np.dstack((new_data_x, new_data_y))  # concatenation

        processed_data_lst = shift_performing(input_data, joint, shift_list)
        processed_data_np = np.array(make_equal_length(processed_data_lst))
        spectrum_linear_comb_cut = np.sum(processed_data_np[:, :, 1], axis=0)
        output_lst.append((processed_data_np[12, :, 0], spectrum_linear_comb_cut, output_c))
    return output_lst


# function that saves simulated spectra
# and their own concentration in following csv files
def save_to_csv(released_product):
    for index in range(len(released_product)):
        df = pd.DataFrame(released_product[index][0:2]).transpose()
        df.to_csv(f'{index + 1}_spec', index=False, header=False)
        ds = pd.Series(released_product[index][2:3])
        ds.to_csv(f'{index + 1}_concentration', index=False, header=False)


# function for adding white noise to spectra linear combination
def add_noise(spectra_sum, ampl, mu, sigma, n):
    return spectra_sum + ampl * np.random.normal(mu, sigma, n)


# function for adding baseline to spectra
def add_baseline(spectra_sum, ampl, mu, sigma, n):
    return spectra_sum + ampl * np.random.normal(mu, sigma, n)


# load all spectra from txt files considering initial concentrations
def load_some_data(c=np.array([0.1] * METABOLITE_QUANTITY)):
    spec_real_data_temp = np.loadtxt(r'assets\mix_met001.txt')
    data_temp = [np.loadtxt(r'assets\met' + f'{i + 1}.txt') for i in range(METABOLITE_QUANTITY)]
    temp1 = np.array([c[i]*data_temp[i][:, 1] for i in range(METABOLITE_QUANTITY)])
    output_lst = []
    for i in range(METABOLITE_QUANTITY):
        output_lst.append(np.dstack((data_temp[i][:, 0], temp1[i]))[0])
    return output_lst, spec_real_data_temp


r = load_some_data()
data = r[0]
spec_real_data = r[1]

# main generation function performance
next_data = gen_spec(data)[0]
# example of using function
save_to_csv(gen_spec(data, n=3))
# set size of points in scatter plot
sizes_of_points = len(next_data[0]) * [1]
sizes_of_points_for_real = len(spec_real_data[:, 0]) * [1]

# plots of real spectra (orange) and generated (blue)
plt.figure(2)
plt.scatter(next_data[0], add_noise(next_data[1], 1000, 0, 1, len(next_data[0])), sizes_of_points, 'blue')
plt.scatter(spec_real_data[:, 0], 1.25 * spec_real_data[:, 1], sizes_of_points_for_real, 'orange')
plt.show()
