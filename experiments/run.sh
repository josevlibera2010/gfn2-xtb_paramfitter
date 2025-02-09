#!/bin/bash
#---------------Script SBATCH----------------
#SBATCH -J CCR
#SBATCH -p main
#SBATCH -n 30
#SBATCH --ntasks-per-node=44
#SBATCH -c 1
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-user=my_mail@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -o %j.out
#SBATCH -e %j.err
#---------------------------CONDA and XTB setups-------------------------------------------
export OMP_NUM_THREADS=1
export OMP_STACKSIZE=20G
export OMP_MAX_ACTIVE_LEVELS=1
export MKL_NUM_THREADS=1


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('~/bin/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "~/bin/anaconda3/etc/profile.d/conda.sh" ]; then
        . "~/bin/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="~/bin/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
#------------------------Loading the environment----------------------------------
conda activate xtbfitter
# you must set this path to load "gfn2-xtb_paramfitter" modules
export PYTHONPATH="</path/to/gfn2-xtb_paramfitter>":$PYTHONPATH

date
conda run -n xtbfitter --no-capture-output python DDGA_OPTIM_CCR.py
date

