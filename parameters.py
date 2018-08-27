from datetime import datetime
import os
import subprocess

def getCommandlineOutput(command):
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (stdout, stderr) = p.communicate()
    return stdout

def gitGetHashCurrentRepo():
    return 'tempfortesting'
    # return getCommandlineOutput('git rev-parse HEAD').strip()

def gitUncommittedChanges():
    return False
    # res = getCommandlineOutput('git diff --name-only')
    # return len(res) > 0

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
    parameters['__code_git_hash'] = gitGetHashCurrentRepo()
    if 'shortname' not in parameters:
        parameters['shortname'] = 'unnamed'
    if gitUncommittedChanges():
        print("[WARN]: Uncommitted changes in git repository")
    return parameters

def saveParameters(parameters, output_directory):
    out_filename = os.path.join(
            output_directory,
            'run_metadata_{}_{}'.format(parameters['shortname'], parameters['__date_string']))
    with open(out_filename, 'w') as file:
        for key, value in parameters.items():
            file.write('{} = {}\n'.format(key, value))

