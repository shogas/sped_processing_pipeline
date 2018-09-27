from datetime import datetime
import os
import subprocess


def get_commandline_output(command):
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (stdout, stderr) = p.communicate()
    return stdout


def git_current_commit_hash():
    return get_commandline_output('git rev-parse HEAD').strip()


def git_uncommited_changes():
    res = get_commandline_output('git diff --name-only')
    return len(res) > 0


def try_convert_to_float(string):
    try:
        return float(string)
    except ValueError:
        return string

# TODO(simonhog): Allow comments
def parameters_parse(filename):
    # TODO(simonhog): Error handling
    with open(filename, 'r') as file:
        lines = file.read().splitlines()
    parameters = {}
    for line in lines:
        comment_start = line.find('#')
        if comment_start >= 0:
            line = line[:comment_start]
        parts = line.split('=', 1)
        key = parts[0].strip()
        value = parts[1].strip()
        if value.isdigit():
            value = int(value)
        else:
            value = try_convert_to_float(value)
        parameters[key] = value

    now = datetime.now()
    parameters['__date'] = now.isoformat()  # TODO(simonhog): Ensure readable, add timezone. pytz library?
    parameters['__date_string'] = '{0:%Y}{0:%m}{0:%d}_{0:%H}_{0:%M}_{0:%S}_{0:%f}'.format(now)  # TODO(simonhog): Ensure readable, add timezone. pytz library?
    parameters['__parameter_file'] = os.path.abspath(filename)
    if parameters.get('development', 'true').lower() != 'true':
        parameters['__code_git_hash'] = git_current_commit_hash()
        if git_uncommited_changes():
            print("[WARN]: Uncommitted changes in git repository")
    if 'shortname' not in parameters:
        parameters['shortname'] = 'unnamed'
    return parameters


def parameters_save(parameters, output_directory):
    out_filename = os.path.join(output_directory, 'metadata.txt')
    del parameters['__date_string']
    with open(out_filename, 'w') as file:
        for key, value in parameters.items():
            file.write('{} = {}\n'.format(key, value))
