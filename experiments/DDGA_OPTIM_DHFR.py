
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

out_folder = 'DHFR_OPT/out'
logger = f'{out_folder}/logger.log'

# Locations for the folders containing the data sets
path1 = 'DATASETS/CCR-IRC'
path2 = 'DATASETS/DHFR-SC'

# Creating the name patterns for data sets' files
# Our data sets are constituted by gaussian output files ranging from 1.log to 299.log and 94.log for IRC and SC, resp.
bn1 = [str(st) for st in range(1, 300, 1)]
bn2 = [str(st) for st in range(1, 41, 1)]


# We substitute string patterns to generate the parameter files with modifications
patterns = [f'VARIABLE{i}#' for i in range(1, 76)]

# Creating the function object for the optimization
funct = ECGFittingV1(patterns=patterns, parm=parm_HCONS, path1=path1, path2=path2, out_folder=out_folder,
                     bn1=bn1, bn2=bn2, bias_grad=False, bgrad=1, bias_chrg=False, bchrg=1)



def objective(inp):
    return funct.evaluate(solution=inp[0], thr_id=inp[1])

# Setting up the guess solution, as the starting point in parameters' space
# The values shown here are the ones from the best solution from CCR optimization
# Each value in the list will substitute each of the string patterns given by
# patterns list (["VARIABLE1#", "VARIABLE2#", ...])

orig = [-14.018360479020165, 1.236881069525325, 0.13580706972905662, 0.3246201212118799, -4.432014990923696,
        1.0255514937829773, 0.06675887731846529, 2.073899481917549, 0.9605312741040215, -4.533438778219989,
        -21.495341797847455, -12.455138968737609, 2.0853130300169327, 1.7497965921967473, 0.2383705371924053,
        1.5686463725021, -0.0975123283211986, 0.4279446114197464, -7.170988290926198, 0.00906433759237281,
        1.2331166997144956, 4.221646920375334, -0.9841654776436481, -0.7566586419524275, 1.5366829361031245,
        -22.129294446627746, -14.561100800412442, 2.2690202742193386, 2.0491294066846764, 0.29070662842784434,
        -1.2891628961245343, -5.680366217178571, 0.7891483626520734, 1.3123300408457044, 0.2976058944611557,
        1.66858564280436, 5.535606876139688, -7.475304558400378, -3.5584304512947775, 0.3413203882229962,
        -19.82956941809607, -17.33442916243415, 2.408997136891434, 2.1723719253106726, 0.46504038142750675,
        -0.2638106255820574, 0.17544813233694517, -0.13126412518464242, -3.342943055074073, -4.05740296182527,
        2.085458157029403, 5.666641043692198, -20.024155702961536, -0.6916504670913654, 0.6529365222226943,
        -12.239941213766064, -15.282455962860277, -1.096665972461735, 3.108659176393753, 1.8654460587012847,
        1.686217500037922, 0.5724606767043273, -0.3329527863969472, -1.7060181705584554, -1.3543201553593158,
        1.9779483323153317, -0.17290939702016442, 0.2065688451447048, 1.0884280538319338, 19.464433229339615,
        -24.025970115922917, -6.3408246781350766, 38.31004584262612, -0.6890266450361957, -0.9685821289890321]

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
