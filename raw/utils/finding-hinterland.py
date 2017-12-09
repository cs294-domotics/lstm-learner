# looks for correlations between activities and lights

LOAD_FILENAME = "../stateful_data/activities/twor2010_1s_full"
DESIRED_LIGHT = "L005"

def main():
    #open up the full stateful data file
    #go through each line and calculate
    # find correlations between L005 and activities
    # L005 is on X% of the time that activity happens
    # activity happens Y% of the time that L5 is on (just to make sure it's not just L5 always being on)
    print("loading data...")
    lines = load_data(LOAD_FILENAME)

    header = lines.pop(0)
    headers = header.split()
    index_of_light = headers.index(DESIRED_LIGHT)
    print(index_of_light)

    end_of_activities = 0
    for i in range(len(headers)):
        if headers[i][:3] == "R1_" or headers[i][:3] == "R2_":
            end_of_activities += 1
    end_of_activities += 1

    activity_names = headers[1:end_of_activities]
    print(activity_names)

    print("processing lines...")
    light_on = 0
    activity_transition_on_d = {}
    activity_on_d = {}
    both_on_d = {}
    neither_on_d = {}
    for a in activity_names:
        activity_transition_on_d[a] = 0
        activity_on_d[a] = 0
        both_on_d[a] = 0
        neither_on_d[a] = 0

    first_fields = lines[0].split()
    prev_activity_states = [int(x) for x in first_fields[1:end_of_activities]]
    for line in lines:
        fields = line.split()
        activity_states = [int(x) for x in fields[1:end_of_activities]]
        light_state = int(fields[index_of_light])
        #print(activity_states),
        if light_state == 1:
            light_on += 1
        for i in range(len(activity_states)):
            activity_name = activity_names[i]
            activity_state = activity_states[i]
            if activity_state == 1:
                activity_on_d[activity_name] += 1
                if light_state == 1:
                    both_on_d[activity_name] += 1
            elif activity_state == 0 and light_state == 0:
                neither_on_d[activity_name] += 1

            if activity_state != prev_activity_states[i] and activity_state == 1:
                activity_transition_on_d[activity_name] += 1

        prev_activity_states = activity_states

    print("")
    print("Light " + DESIRED_LIGHT + " on:")
    print(light_on)
    print("")
    for activity_name in activity_names:
        print(activity_name)
        print(activity_transition_on_d[activity_name])
        print(activity_on_d[activity_name])
        print(both_on_d[activity_name])
        if activity_on_d[activity_name] != 0:
            print("({})%".format(round(both_on_d[activity_name]/activity_on_d[activity_name]*100, 2)))
        print(neither_on_d[activity_name])
        print("")




def load_data(filename):
    f = open(filename, "r")
    lines = f.readlines()
    f.close()
    return lines

main()
