import sys

from parameters import parseParameters, saveParameters

def main(parameter_file):
    run_parameters = parseParameters(parameter_file)
    print(run_parameters)
    saveParameters(run_parameters, '../../Data/Tmp')

if __name__ == '__main__':
    main(sys.argv[1])
