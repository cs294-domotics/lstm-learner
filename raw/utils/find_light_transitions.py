LOAD_FOLDER = "../stateful_data/"
FILENAME = "twor2010_1s_full"
LOAD_FILENAME = LOAD_FOLDER + FILENAME

ONE_DAY_SECS = 60 * 60 * 24
ONE_WEEK_SECS = ONE_DAY_SECS * 7
TWO_WEEK_SECS = ONE_WEEK_SECS * 2
ONE_MONTH_SECS = ONE_DAY_SECS * 30

def main():
    print("loading data...")
    lines = load_data(LOAD_FILENAME)
    # strip off the column labels
    column_labels = lines.pop(0)
    num_lights = get_num_lights(column_labels)
    print("found {} lights, searching for light transitions...".format(num_lights))
    num_samples = len(lines)
    transition_indices, transition_timestamps = get_transition_indices(lines, num_lights)#, WINDOW_SIZE)
    print(transition_timestamps[0:5])
    print("found {} light transitions...".format(len(transition_indices)))
    print("generating month, week timestamps...")
    start_timestamp = get_timestamp(lines[0])
    end_timestamp = get_timestamp(lines[-1])
    month_timestamps = get_month_timestamps(start_timestamp, end_timestamp)
    week_timestamps = get_week_timestamps(start_timestamp, end_timestamp)
    all_timestamps = month_timestamps #+ week_timestamps
    all_timestamps.sort()

    print("")

    curr_transition_index = 0
    for i in range(1,len(month_timestamps)):
        curr_month = month_timestamps[i-1]
        next_month = month_timestamps[i]
        print("Month {} start: {}".format(i,curr_month))
        light_count = 0
        while curr_transition_index < len(transition_timestamps) and transition_timestamps[curr_transition_index] < next_month:
            light_count += 1
            curr_transition_index += 1
        print("\tNum light events: {}".format(light_count))

    print("")

    curr_transition_index = 0
    for i in range(1,len(week_timestamps)):
        curr_week = week_timestamps[i-1]
        next_week = week_timestamps[i]
        print("Week {} start: {}".format(i,curr_week))
        light_count = 0
        while curr_transition_index < len(transition_timestamps) and transition_timestamps[curr_transition_index] < next_week:
            light_count += 1
            curr_transition_index += 1
        print("\tNum light events: {}".format(light_count))

def get_week_timestamps(start_timestamp, end_timestamp):
    week_timestamps = [start_timestamp]
    week_timestamp = start_timestamp
    while week_timestamp < end_timestamp:
        week_timestamp += ONE_WEEK_SECS
        week_timestamps.append(week_timestamp)
    return week_timestamps

def get_month_timestamps(start_timestamp, end_timestamp):
    month_timestamps = [start_timestamp]
    month_timestamp = start_timestamp
    while month_timestamp < end_timestamp:
        month_timestamp += ONE_MONTH_SECS
        month_timestamps.append(month_timestamp)
    return month_timestamps

def load_data(filename):
    f = open(filename, "r")
    lines = f.readlines()
    f.close()
    return lines


def get_num_lights(column_labels):
    fields = column_labels.split()
    num_lights = 0
    for field in fields:
        if field[0] == 'L':
            num_lights += 1
    return num_lights

def get_transition_indices(lines, num_lights):
    transition_indices = []
    transition_timestamps = []
    s2_index = 0
    num_processed = 0
    num_lines = len(lines)# - window_size
    update_chunk = 1000000
    while s2_index < len(lines):
        s1_light_state = get_light_state(lines[s2_index-1], num_lights)
        s2_light_state = get_light_state(lines[s2_index], num_lights)
        if s1_light_state != s2_light_state:
            transition_indices.append(s2_index)
            transition_timestamps.append(get_timestamp(lines[s2_index]))
        s2_index += 1
        num_processed += 1
        if num_processed >= update_chunk and num_processed % update_chunk == 0:
            print("{} processed of {} ({}%)".format(num_processed, num_lines, round(num_processed/num_lines*100, 2)))
    return transition_indices, transition_timestamps


def get_light_state(line, num_lights):
    fields = line.split()
    return fields[(-1*num_lights):]

def get_timestamp(line):
    timestamp = None
    fields = line.split()
    if len(fields) > 1:
        timestamp = float(fields[0])
    return timestamp

main()
