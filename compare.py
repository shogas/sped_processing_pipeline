import os
import sys

def parseParameters(filename):
    # TODO(simonhog): Error handling
    with open(filename, 'r') as file:
        lines = file.read().splitlines()
    options = {}
    for line in lines:
        parts = line.split('=')
        key = parts[0].strip()
        value = parts[1].strip() # TODO(simonhog): Value conversion
        options[key] = value
    return options

def main(parameter_file):
    run_parameters = parseParameters(parameter_file)
    print(run_parameters)

if __name__ == '__main__':
    main(sys.argv[1])
