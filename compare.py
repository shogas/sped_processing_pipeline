from datetime import datetime
import os
import sys

def parseParameters(filename):
    # TODO(simonhog): Error handling
    with open(filename, 'r') as file:
        lines = file.read().splitlines()
    parameters = {}
    for line in lines:
        parts = line.split('=', 1)
        key = parts[0].strip()
        value = parts[1].strip() # TODO(simonhog): Value conversion
        parameters[key] = value

    now = datetime.now()
    parameters['__date'] = now.isoformat() # TODO(simonhog): Ensure readable, add timezone. pytz library?
    parameters['__date_string'] = '{0:%Y}{0:%m}{0:%d}_{0:%H}_{0:%M}_{0:%S}_{0:%f}'.format(now) # TODO(simonhog): Ensure readable, add timezone. pytz library?
    parameters['__parameter_file'] = os.path.abspath(filename)
    if 'shortname' not in parameters:
        parameters['shortname'] = 'unnamed'
    return parameters

def saveParameters(parameters):
    out_filename = 'run_metadata_{}_{}'.format(parameters['shortname'], parameters['__date_string'])
    print(out_filename)
    with open(out_filename, 'w') as file:
        for key, value in parameters.items():
            file.write('{} = {}\n'.format(key, value))

def main(parameter_file):
    run_parameters = parseParameters(parameter_file)
    print(run_parameters)
    saveParameters(run_parameters)

if __name__ == '__main__':
    main(sys.argv[1])
