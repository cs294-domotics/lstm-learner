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
