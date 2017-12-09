filename = "../../data/twor2010"

#stats I want:
#mean, median, and std dev of gaps between event and light change event
#save them and plot them

import datetime
import numpy as np
import matplotlib.pyplot as plt

light_type ='L'
desired_types = ['L', 'M', 'D']

window_size = 40 #events

def main():
    with open(filename) as f:
        lines = f.read().splitlines()
    num_lines = len(lines)
    window_gaps = []
    for i in range(window_size):
        window_gaps.append([])

    for i in range(num_lines):
        line = lines[i]
        if is_well_formed(line):
            device = get_device(line)
            device_type = get_device_type(device)
            timestamp = get_timestamp(line)
            if device_type == light_type and i >= window_size:
                for j in range(window_size):
                    event_line = lines[i-(j+1)]
                    event_timestamp = get_timestamp(event_line)
                    gap = timestamp - event_timestamp
                    window_gaps[j].append(gap)

        if ((i+1) % 200000) == 0:
            print("Processed " + str(i+1) + " lines out of " + str(num_lines) + " (" + str(round(((i+1)/num_lines*100), 2)) + "%)")

    window_gap_means = []
    window_gap_std = []
    for i in range(window_size):
        window_gap_means.append(np.array(window_gaps[i]).mean())
        window_gap_std.append(np.array(window_gaps[i]).std())

    print("Window size: {}".format(window_size))
    for i in range(window_size):
        print("   t-{} duration until light event: {} second mean, {} second std. dev.".format(i+1, window_gap_means[i], window_gap_std[i]))




            #Science is great! But it's not very forgiving. - Mr. Clarke, Stranger Things


def is_well_formed(line):
    fields = line.split()
    return (len(fields) >= 4)

def get_device(line):
    return line.split()[2]

def get_device_type(device):
    return device[0]

def get_date(line):
    fields = line.split()
    timestamp_str = fields[0] + ' ' + fields[1]
    try:
        date = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f").date()
    except ValueError:
        date = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").date()
    return date

def get_timestamp(line):
    fields = line.split()
    timestamp_str = fields[0] + ' ' + fields[1]
    try:
        timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f").timestamp()
    except ValueError:
        timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").timestamp()
    return timestamp

main()
