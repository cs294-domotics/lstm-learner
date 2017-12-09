# Takes in a fixed-interval stateful data file formatted like:
#
# Timestamp M050 M044 M045 L005
# 1251097367.069235 0 0 0 1
# 1251097368.069235 0 0 0 1
# 1251097369.069235 0 0 1 1
# ...
#
# and slices it up into smaller segments for generating training/testing. Say
# the dataset is three months. This might produce datasets for a set of periods
# like a day, a week, two weeks, a month. It will produce two files of each
# duration, one for creating training data and one for creating testing data.
# Since the goal is that a learner trained on any of the training data could be
# tested on any of the testing data, then you can think of the time alignment as
# "centered." That is, all the training durations will start at different times
# but will END at the same time, and the testing durations will all START at the
# same time (right after the training datasets end) but will end at different
# times. This means that the max duration is what affects when the datasets all
# start. Assume one month is the max duration. Then the training data for one
# month will be the first month. The training data for one week will be the last
# week of that month, and the training data for one day will be the last day
# of the month. The testing month/week/day all start on the same day, the first
# of the next month. This is done so that each of the training datasets can
# be evaluated against any of the testing data.

# training data
# Start of all data: Monday, August 24, 2009 12:02:47.069 AM
#{86400: 1253602967.069235, Tuesday, September 22, 2009 12:02:47.069 AM
#604800: 1253084567.069235, Wednesday, September 16, 2009 12:02:47.069 AM
#1209600: 1252479767.069235, Wednesday, September 9, 2009 12:02:47.069 AM
#2592000: 1251097367.069235} Monday, August 24, 2009 12:02:47.069 AM
#1253689367.069235 Wednesday, September 23, 2009 12:02:47.069 AM
# testing data
#1253689367.069235 Wednesday, September 23, 2009 12:02:47.069 AM
#{86400: 1253775767.069235, Thursday, September 24, 2009 12:02:47.069 AM
#604800: 1254294167.069235, Wednesday, September 30, 2009 12:02:47.069 AM
#1209600: 1254898967.069235,  Wednesday, October 7, 2009 12:02:47.069 AM
#2592000: 1256281367.069235} Friday, October 23, 2009 12:02:47.069 AM

from time import time

LOAD_FOLDER = "stateful_data/action/"
FILENAME_STEM = "twor2010_1s"
#FILENAME_STEM = "stupid_simple_1s"
DATA_FILE = FILENAME_STEM + "_full"
FILENAME = LOAD_FOLDER + DATA_FILE
SAVE_FOLDER = LOAD_FOLDER

ONE_DAY_SECS = 60 * 60 * 24
ONE_WEEK_SECS = ONE_DAY_SECS * 7
TWO_WEEK_SECS = ONE_WEEK_SECS * 2
ONE_MONTH_SECS = ONE_DAY_SECS * 30

INTERVAL_SECS = 1 #needs to divide evenly into WINDOW_TIME_SECS
TIMEFRAMES = [ONE_DAY_SECS, ONE_WEEK_SECS, TWO_WEEK_SECS, ONE_MONTH_SECS]
TIMEFRAME_NAMES = {ONE_DAY_SECS: "one_day",
                   ONE_WEEK_SECS: "one_week",
                   TWO_WEEK_SECS: "two_weeks",
                   ONE_MONTH_SECS: "one_month"}

#BIAS = 3 * ONE_MONTH_SECS

START_ON = 1267594672.070476 #based on results from find_light_transitions.py

INDENT = "    "

def main():
    TIMEFRAMES.sort() # just in case
    print("loading data...")
    lines = load_data(FILENAME)
    print("\n" + INDENT + lines[0].rstrip()),
    print(INDENT + lines[1].rstrip()),
    print(INDENT + "...")
    print(INDENT + lines[-1])
    print("removing column headers...")
    headers = lines.pop(0)
    #print("applying bias...")
    #lines = leftstrip(lines, BIAS)
    print("shifting timeline ahead to desired start time...")
    lines = leftstrip(lines, START_ON)
    # get start time for testing
    print("calculating start and end times for timeframes...")
    start_times_training, end_time_training = get_training_times(lines, TIMEFRAMES)
    start_time_testing, end_times_testing = get_testing_times(lines, TIMEFRAMES)
    print("opening files...")
    training_file_handles, training_file_names = open_files(TIMEFRAME_NAMES, "train")
    testing_file_handles, testing_file_names = open_files(TIMEFRAME_NAMES, "test")
    #print(start_times_training)
    #print(end_time_training)
    #print(start_time_testing)
    #print(end_times_testing)
    print("copying data out to the right files...")
    start_time = get_start_time(start_times_training)
    end_time = get_end_time(end_times_testing)

    # progress report info
    update_chunk = 200000 #samples
    num_samples = int((end_time - start_time)/INTERVAL_SECS)
    samples_processed = 0
    start_time = time()
    start_chunk_time = start_time

    # write the column headings
    for timeframe in TIMEFRAMES:
        training_file_handles[timeframe].write(headers)
        testing_file_handles[timeframe].write(headers)

    # copy each data line to the relevant files
    for line in lines:
        timestamp = get_timestamp(line)
        if timestamp <= end_time:
            for timeframe in TIMEFRAMES:
                if timestamp >= start_times_training[timeframe] and timestamp <= end_time_training:
                    training_file_handles[timeframe].write(line)
                if timestamp >= start_time_testing and timestamp <= end_times_testing[timeframe]: #if not elif bc/the midpoint is shared
                    testing_file_handles[timeframe].write(line)
        else:
            break
        # progress report info
        samples_processed += 1
        if samples_processed % update_chunk == 0:
            print_progress_report(samples_processed, num_samples, start_chunk_time, update_chunk)
            start_chunk_time = time()


    # final progress report
    total_minutes = round((time() - start_time)/60.0, 2)
    print(INDENT + "total elapsed time: " + str(total_minutes) + " minutes")

    close_files(training_file_handles)
    close_files(testing_file_handles)
    print("saved data to the following files: ")
    print(get_save_file_str(training_file_names, testing_file_names))


# removes a bias's worth of lines from the beginning
def leftstrip(lines, start_time):
    new_start_line = 0
    first_timestamp = get_timestamp(lines[0])
    timestamp = first_timestamp
    while timestamp < start_time:
        new_start_line += 1
        timestamp = get_timestamp(lines[new_start_line])
    return lines[new_start_line:]


def print_progress_report(samples_processed, num_samples, start_chunk_time, update_chunk):
    elapsed_minutes = round((time() - start_chunk_time)/60.0, 2)
    remaining_samples = num_samples - samples_processed
    remaining_chunks = remaining_samples / update_chunk
    remaining_minutes = round(remaining_chunks * elapsed_minutes, 2)
    print(INDENT + "processed " + str(samples_processed) + " out of " + str(num_samples))
    print(INDENT + str(elapsed_minutes) + " minutes elapsed since last update, " + str(remaining_minutes) + " estimated minutes remaining")

def get_end_time(end_times_testing):
    end_times = list(end_times_testing.values())
    end_times.sort()
    return end_times[-1]

def get_start_time(start_times_training):
    start_times = list(start_times_training.values())
    start_times.sort()
    return start_times[0]

def get_save_file_str(training_file_names, testing_file_names):
    s = "\n" + INDENT + "training files:\n"
    for name in training_file_names:
        s += INDENT + INDENT + name + "\n"
    s += "\n" + INDENT + "testing files:\n"
    for name in testing_file_names:
        s += INDENT + INDENT + name + "\n"
    return s

def close_files(file_handle_dict):
    for name in file_handle_dict:
        f = file_handle_dict[name]
        f.close()

def get_timestamp(line):
    timestamp = None
    fields = line.split()
    if len(fields) > 1:
        timestamp = float(fields[0])
    return timestamp

def load_data(filename):
    f = open(filename, "r")
    lines = f.readlines()#f.read().splitlines()
    f.close()
    return lines

def get_testing_times(lines, timeframes):
    start_time_testing = None
    end_times_testing = {}
    max_duration_secs = timeframes[-1]
    data_start_timestamp = get_timestamp(lines[0])
    start_time_testing = data_start_timestamp + max_duration_secs
    end_time_testing = start_time_testing + max_duration_secs
    for timeframe in timeframes:
        end_time_testing = start_time_testing + timeframe
        end_times_testing[timeframe] = end_time_testing
    return start_time_testing, end_times_testing

def get_training_times(lines, timeframes):
    start_times_training = {}
    end_time_training = None
    max_duration_secs = timeframes[-1]
    data_start_timestamp = get_timestamp(lines[0])
    end_time_training = data_start_timestamp + max_duration_secs
    for timeframe in timeframes:
        start_time_training = end_time_training - timeframe
        start_times_training[timeframe] = start_time_training
    return start_times_training, end_time_training

def open_files(timeframe_names, suffix):
    file_handles = {}
    file_names = []
    for timeframe in timeframe_names:
        timeframe_name = timeframe_names[timeframe]
        filename = LOAD_FOLDER + FILENAME_STEM + "_" + timeframe_name + "_" + suffix
        f = open(filename, "w")
        file_handles[timeframe] = f
        file_names.append(filename)
    return file_handles, file_names

main()
