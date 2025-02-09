import sys
# you must set this path
module_path="</path/to/gfn2-xtb_paramfitter>"
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
from algorithms import DDGeneticAlgorithm as DDGA
from objectivefunctions import ECGFittingV1
from parameters import parm_HCONS
import os, shutil

"""
We set the output folder where the calculation results will be located.
In the \"out_folder\" path, a folder at each evaluation of the objective function will be created
If the evaluation runs without errors each folder will contain:
 1) a pkl binary file per data set, codifying a python dictionary with the results (if thread's calculation 
    runs without errors)
 2) the parameter file (always exported)

The file at '{out_folder}/logger.log' could be loaded as pandas dataframe for plotting the Scores and identifying 
the best solution at some point of the optimization. By plotting the Scores, you could also get information about
the convergence rates of the calculations for any score individually. 

This script also prints the real-time results of the calculations, once you finished the optimization, you could copy
the string with the list of parameters for the best solution to set "orig" variable for the next optimization. 
"""

out_folder = 'DHFR_OPT/out'
if os.path.isdir(out_folder) and os.path.exists(out_folder):
    shutil.move(out_folder,out_folder + '_old')
    os.makedirs(out_folder)
else:
    os.makedirs(out_folder)

logger = f'{out_folder}/logger.log'

# List with the locations of the folders containing the data sets
paths=['DATASETS/CCR-IRC', 'DATASETS/DHFR-SC']

# Creating the name patterns for data sets' files
# Our data sets are constituted by gaussian output files ranging from 1.log to 299.log and 94.log for IRC and SC,
# respectively. "base_names" is a list of lists with the base names of the log files for each data set
bn1 = [str(st) for st in range(1, 300, 1)]
bn2 = [str(st) for st in range(1, 41, 1)]

base_names = [bn1, bn2]

# We substitute string patterns to generate the parameter files with modifications
patterns = [f'VARIABLE{i}#' for i in range(1, 76)]

# Creating the function object for the optimization
funct = ECGFittingV1(patterns=patterns, parm=parm_HCONS, paths=paths, base_names=base_names, out_folder=out_folder)

def objective(inp):
    return funct.evaluate(solution=inp[0], thr_id=inp[1])

# Setting up the guess solution, as the starting point in parameters' space
# The values shown here are the ones from the best solution from CCR optimization
# Each value in the list will substitute each of the string patterns given by
# patterns list (["VARIABLE1#", "VARIABLE2#", ...])

orig = [-14.018360, 1.236881, 0.135807, 0.324620, -4.432015, 1.025551, 0.066759, 2.073899, 0.960531, -4.533439, 
        -21.495342, -12.455139, 2.085313, 1.749797, 0.238371, 1.568646, -0.097512, 0.427945, -7.170988, 0.009064,
        1.233117, 4.221647, -0.984165, -0.756659, 1.536683, -22.129294, -14.561101, 2.269020, 2.049129, 0.290707,
        -1.289163, -5.680366, 0.789148, 1.312330, 0.297606, 1.668586, 5.535607, -7.475305, -3.558430, 0.341320,
        -19.829569, -17.334429, 2.408997, 2.172372, 0.465040, -0.263811, 0.175448, -0.131264, -3.342943,
        -4.057403, 2.085458, 5.666641, -20.024156, -0.691650, 0.652937, -12.239941, -15.282456, -1.096666, 
        3.108659, 1.865446, 1.686218, 0.572461, -0.332953, -1.706018, -1.354320, 1.977948, -0.172909, 0.206569,
        1.088428, 19.464433, -24.025970, -6.340825, 38.310046, -0.689027, -0.968582]

# Setting up the initial boundaries for each parameter

var_bound = []
for i in orig:
    var_bound.append([i - 0.001 * abs(i), i + 0.001 * abs(i)])

var_bound = np.array(var_bound)

# Dictionary containing the initial configurations
algorithm_param = {'max_num_iteration': 10000000,
                   'population_size': 680,
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
