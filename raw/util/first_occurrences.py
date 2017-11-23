# This script prints out the first occurrence of each device in the dataset

filename = "../data/twor2010"

def main():
    with open(filename) as f:
		data = f.read().splitlines()
    device_type_buckets, device_first_occurrences = find_devices(data)
    for device_type in device_type_buckets:
        devices = device_type_buckets[device_type]
        for device in devices:
            first_occurrence = str(device) + "\t" + str(device_first_occurrences[device])
            print(first_occurrence)

def find_devices(lines):
    device_type_buckets = {}
    device_first_occurrences = {}
    for line in lines:
        if is_well_formed(line):
            device = get_device(line)
            device_type = get_device_type(device)
            if device_type not in device_type_buckets:
                device_type_buckets[device_type] = [device]
                device_first_occurrences[device] = get_timestamp(line)
            else:
                if device not in device_type_buckets[device_type]:
                    device_type_buckets[device_type].append(device)
                    device_first_occurrences[device] = get_timestamp(line)
    return device_type_buckets, device_first_occurrences

def is_well_formed(line):
    fields = line.split()
    return (len(fields) >= 4)

def get_device(line):
    return line.split()[2]

def get_device_type(device):
    return device[0]

def get_timestamp(line):
    fields = line.split()
    return fields[0] + ' ' + fields[1]

main()
