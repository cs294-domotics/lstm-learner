filename = "data/twor2010"

#stats I want:
#mean, median, and std dev of gaps between event and light change event
#save them and plot them

import datetime
import numpy as np
import matplotlib.pyplot as plt

light_type ='L'
desired_types = ['L', 'M', 'D']

def main():
    with open(filename) as f:
        lines = f.read().splitlines()
    event_counts_per_light = {}
    prev_device_timestamp = None
    prev_nonlight_timestamp = None
    all_event_diffs = []
    diffs = []
    nonlight_diffs = []
    num_lines = len(lines)
    curr_line = 0
    for line in lines:
        if is_well_formed(line):
            device = get_device(line)
            device_type = get_device_type(device)
            timestamp = get_timestamp(line)
            if device_type in desired_types:
                if prev_device_timestamp != None:
                    all_event_diffs.append(timestamp - prev_device_timestamp)
                if device_type == light_type:
                    if device not in event_counts_per_light:
                        event_counts_per_light[device] = 1
                    else:
                        event_counts_per_light[device] += 1
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
    all_event_diffs = np.array(all_event_diffs)
    diffs = np.array(diffs)
    nonlight_diffs = np.array(nonlight_diffs)

    # display results
    total_light_events = 0
    # break it down per light
    print("\nLight change events")
    for light in event_counts_per_light:
        print(light + ": "+ str(event_counts_per_light[light]))
        total_light_events += event_counts_per_light[light]
    print("Number of total light change events: " + str(total_light_events))

    print("\nTime gap between a light event and the previous event (light or non-light)")
    print("Mean: " + str(np.mean(diffs)))
    print("Median: " + str(np.median(diffs)))
    print("Standard Dev: " + str(np.std(diffs)))

    print("\nTime gap between a light event and the previous non-light event")
    print("Mean: " + str(np.mean(nonlight_diffs)))
    print("Median: " + str(np.median(nonlight_diffs)))
    print("Standard Dev: " + str(np.std(nonlight_diffs)))


    print("\nTime gap between each event")
    print("Mean: " + str(np.mean(all_event_diffs)))
    print("Median: " + str(np.median(all_event_diffs)))
    print("Standard Dev: " + str(np.std(all_event_diffs)))
    all_event_diffs.sort()
    print("Smallest: " + str(all_event_diffs[0]))
    print("Largest: " + str(all_event_diffs[-1]))
    custom_hist = []
    i = 0
    count = 0
    while i < len(all_event_diffs) and all_event_diffs[i] < 1:
        count += 1
        i += 1
    custom_hist.append(count)
    print("0 < 1 sec: " + str(count))
    count = 0
    while i < len(all_event_diffs) and all_event_diffs[i] < 10:
        count += 1
        i += 1
    custom_hist.append(count)
    print("1 < 10 sec: " + str(count))
    count = 0
    while i < len(all_event_diffs) and all_event_diffs[i] < 100:
        count += 1
        i += 1
    custom_hist.append(count)
    print("10 < 100 sec: " + str(count))
    count = 0
    while i < len(all_event_diffs) and all_event_diffs[i] < 600:
        count += 1
        i += 1
    custom_hist.append(count)
    print("100 < 600 sec: " + str(count))
    count = 0
    while i < len(all_event_diffs) and all_event_diffs[i] < 1200:
        count += 1
        i += 1
    custom_hist.append(count)
    print("600 (1 min) < 1200 sec (20 min): " + str(count))
    count = 0
    while i < len(all_event_diffs) and all_event_diffs[i] < 3600:
        count += 1
        i += 1
    custom_hist.append(count)
    print("1200 (20 min) < 3600 (1 hr): " + str(count))
    count = 0
    while i < len(all_event_diffs) and all_event_diffs[i] < 18000:
        count += 1
        i += 1
    custom_hist.append(count)
    print("3600 (1 hr) < 18000 (5 hrs): " + str(count))
    count = 0
    while i < len(all_event_diffs):
        count += 1
        i += 1
    custom_hist.append(count)
    print("over 18000 (5 hrs): " + str(count))

    # make plot
    labels = ["0 < 1", "1 < 10", "10 < 100", "100 < 600", "600 < 1200", "over 1200"]
    y_pos = np.arange(len(custom_hist))
    plt.bar(y_pos, custom_hist)
    plt.xticks(y_pos, labels)
    plt.title("Timing gap between events")
    plt.show()


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
