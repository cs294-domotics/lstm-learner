filename = "data/twor2010"

#stats I want:
#mean, median, and std dev of gaps between event and light change event
#save them and plot them

import datetime
import numpy as np

light_type ='L'

def main():
    with open(filename) as f:
        lines = f.read().splitlines()
    prev_device_timestamp = None
    prev_nonlight_timestamp = None
    diffs = []
    nonlight_diffs = []
    num_lines = len(lines)
    curr_line = 0
    for line in lines:
        if is_well_formed(line):
            device_type = get_device_type(get_device(line))
            timestamp = get_timestamp(line)
            if curr_line < 5:
                print(timestamp)
            if device_type == light_type:
                if prev_device_timestamp != None:
                    diff = timestamp - prev_device_timestamp
                    diffs.append(diff)
                if prev_nonlight_timestamp != None:
                    nonlight_diff = timestamp - prev_nonlight_timestamp
                    nonlight_diffs.append(nonlight_diff)
            else:
                prev_nonlight_timestamp = timestamp
            prev_device_timestamp = timestamp
            curr_line += 1
            if curr_line % 200000 == 0:
                print("Processed " + str(curr_line) + " lines out of " + str(num_lines) + " (" + str(round(curr_line/num_lines*100, 2)) + "%)")

    # convert lists for easy stat calculation
    diffs = np.array(diffs)
    nonlight_diffs = np.array(nonlight_diffs)

    # display results
    print("\nTime gap between a light event and the previous event (light or non-light)")
    print("Mean: " + str(np.mean(diffs)))
    print("Median: " + str(np.median(diffs)))
    print("Standard Dev: " + str(np.std(diffs)))

    print("\nTime gap between a light event and the previous non-light event")
    print("Mean: " + str(np.mean(nonlight_diffs)))
    print("Median: " + str(np.median(nonlight_diffs)))
    print("Standard Dev: " + str(np.std(nonlight_diffs)))



            #Science is great! But it's not very forgiving. - Mr. Clarke, Stranger Things


def is_well_formed(line):
    fields = line.split()
    return (len(fields) >= 4)

def get_device(line):
    return line.split()[2]

def get_device_type(device):
    return device[0]

def get_timestamp(line):
    fields = line.split()
    timestamp_str = fields[0] + ' ' + fields[1]
    try:
        timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f").timestamp()
    except ValueError:
        timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").timestamp()
    return timestamp

main()
