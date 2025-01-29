
"""
# Add the next four lines to the main script if it is external to the module folders
import sys
module_path="/path/to/gfn2-xtb_paramfitter"
if module_path not in sys.path:
    sys.path.append(module_path)
"""

import numpy as np
from algorithms import DDGeneticAlgorithm as DDGA
from objectivefunctions import ECGFittingV1
from parameters import parm_HCONS

"""
We set the output folder where the calculation results will be located.
in the out_folder path will be created a folder at each evaluation of the objective function
If the evaluation runs without errors each folder will contain:
 1) a pkl binary file per data set, codifying a python dictionary with the results
 2) the parameter file (always exported)

The file at '{out_folder}/logger.log' could be loaded as pandas dataframe for plotting the Scores and identifying 
the best solution at some point of the optimization. By plotting the Scores, you could also get information about
the convergence rates of the calculations for any score individually. 

This script also prints the real-time results of the calculations, once you finished the optimization, you could copy
the string with the list of parameters for the best solution to set "orig" variable for the next optimization. 
"""

out_folder = 'CCR_OPT/out'
logger = f'{out_folder}/logger.log'

# Locations for the folders containing the data sets
path1 = 'DATASETS/CCR-IRC'
path2 = 'DATASETS/CCR-SC'

# Creating the name patterns for data sets' files
# Our data sets are constituted by gaussian output files ranging from 1.log to 299.log and 94.log for IRC and SC, resp.
bn1 = [str(st) for st in range(1, 300, 1)]
bn2 = [str(st) for st in range(1, 95, 1)]


# We substitute string patterns to generate the parameter files with modifications
patterns = [f'VARIABLE{i}#' for i in range(1, 76)]

# Creating the function object for the optimization
funct = ECGFittingV1(patterns=patterns, parm=parm_HCONS, path1=path1, path2=path2, out_folder=out_folder,
                     bn1=bn1, bn2=bn2, bias_grad=False, bgrad=1, bias_chrg=False, bchrg=1)



def objective(inp):
    return funct.evaluate(solution=inp[0], thr_id=inp[1])

# Setting up the guess solution, as the starting point in parameters' space
# The values shown here are the original in GFN2-xTB's parametrization
# Each value in the list will substitute each of the string patterns given by
# patterns list (["VARIABLE1#", "VARIABLE2#", ...])

orig = [-10.707211, 1.230000, 0.405771, 0.800000, -0.500000, 5.563889, 0.027431, 2.213717, 1.105388, -0.953618,
        -13.970922, -10.063292, 2.096432, 1.800000, 0.538015, 1.500000, -0.102144, 0.161657, -0.411674, 0.213583,
        1.247655, 4.231078, -2.294321, -0.271102, 1.056358, -16.686243, -12.523956	, 2.339881, 2.014332, 0.461493,
        -0.639780, -1.955336, 0.561076, 3.521273, 2.026786, 1.682689, 5.242592, -8.506003, -2.504201, 1.164892,
        -20.229985, -15.503117	, 2.439742, 2.137023, 0.451896, -0.517134, 0.117826, -0.145102, -4.935670, -0.310828,
        2.165712, 5.784415, -14.955291, -3.350819, 1.497020, -20.029654, -11.377694, -0.420282	, 1.981333, 2.025643,
        1.702555, 0.339971, -0.501722, -0.256951, -0.098465, 2.007690, -0.151117, 0.442859, 1.214553, 14.995090,
        -25.855520, -8.048064, 25.993857, -1.085866, -2.500000]

# Setting up the initial boundaries for each parameter

var_bound = []
for i in orig:
    var_bound.append([i - 0.001 * abs(i), i + 0.001 * abs(i)])

var_bound = np.array(var_bound)

# Dictionary containing the initial configurations
algorithm_param = {'max_num_iteration': 10000000,
                   'population_size': 100,
                   'mutation_probability': 0.9,
                   'elit_ratio': 0.15,
                   'crossover_probability': 0.95,
                   'parents_portion': 0.4,
                   'crossover_type': 'uniform',
                   'max_iteration_without_improv': 120}

# As "patterns" list represent the list of string labels to substitute for the generation of the parameter files,
# here "dimension" variable is equal to the length of "patterns"

model = DDGA(function=objective, dimension=75, variable_boundaries=var_bound, algorithm_parameters=algorithm_param)

if __name__ == "__main__":
    # The optimization runs in parallel, if you have 44 threads in your server, then n_cpus should be equal to 44
    # For improving computational performance, “n_cpus” should be a multiple of population_size*(1-parents_portion)

    model.run(guess=orig, log_path=logger, n_cpus=44)
    convergence = model.report
    print(convergence)
    solution = model.output_dict
    print(str(list(solution)))
