There are many different experiments in here sort of willy-nilly. The best-performing approach was an event-based approach. The code that is relevant to that version follows.

### How to run:

1. `event-slicer-dicer.py` takes in a CASAS file and generates a whole bunch of files with subsets of the original file for different durations of time. Currently the durations are one day, one week, two weeks, and one month. For each duration the program creates raw files for training and testing, and any of the training ones can be used with any of the testing ones. These files are also saved to the `stateful_data` folder. See the comments in the file for more details.
2. `preprocessor-sampled-events.py` and `preprocessor-sampled-events-activities.py` take in any CASAS-like event-based data file and spits out files containing numpy arrays for the LSTM representing windows of time that the LSTM will try to learn. These files will be written to the `build` folder. Each program lets you determine whether or not to add time or light state. The former file does not include activity info as a feature, the latter file does.
3. `lstm_windows.py` takes in the numpy array files created by the preprocessor and tries to learn!
