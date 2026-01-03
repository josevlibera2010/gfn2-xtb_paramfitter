import logging
import os

import numpy as np
from xtb.interface import Calculator
from xtb.utils import get_method

from constants import ANGSTROM_TO_BOHR, HARTREE_TO_KCAL_MOL
from readers import atmNToSymb, XYZReader, CHRGReader, toSymbols
from validation import validate_directory_exists, validate_file_exists, validate_env_var

logger = logging.getLogger(__name__)


class XtbRunnerFromFiles:
    def __init__(self, chrg: float, multip: float, parm_folder: str, coord_file: str, charg_coord_file: str = None):
        validate_directory_exists(parm_folder, "parm_folder")
        validate_file_exists(coord_file, "coord_file")

        self.parm_folder = parm_folder
        self.coord_file = coord_file
        self.charg_coord_file = charg_coord_file

        self.chrg = chrg
        self.multip = multip
        self.energy = None
        self.gradients = None
        self.elem_indexes = {}

    def run(self):
        base_fold = validate_env_var('CONDA_PREFIX')
        os.environ['XTBHOME'] = base_fold
        os.environ['XTBPATH'] = self.parm_folder
        os.environ['PKG_CONFIG_PATH'] = base_fold + '/lib/pkgconfig'
        os.environ['MKL_NUM_THREADS'] = '8'
        os.environ['OMP_NUM_THREADS'] = '8,1'
        os.environ['OMP_STACKSIZE'] = '10G'
        numbers, positions = XYZReader(self.coord_file)
        tmp_symb = toSymbols(numbers)
        for i in range(len(numbers)):
            elem = atmNToSymb[int(numbers[i])]
            if elem in self.elem_indexes:
                self.elem_indexes[elem].append(i)
            else:
                self.elem_indexes[elem] = [i]

        calc = Calculator(param=get_method("GFN2-xTB"), numbers=numbers, positions=positions * ANGSTROM_TO_BOHR,
                          charge=self.chrg, uhf=(self.multip-1))
        if self.charg_coord_file and os.path.isfile(self.charg_coord_file):
            charges, positions = CHRGReader(self.charg_coord_file)
            numbers = np.ones(len(charges), dtype=np.int8)
            calc.set_external_charges(numbers=numbers, charges=charges, positions=positions)

        calc.set_verbosity(0)
        calc.set_max_iterations(150)
        calc.set_accuracy(0.0001)
        try:
            res = calc.singlepoint()
        except Exception as e:
            print(f'Error in single-point calculation!\n{e}')
            return None, None, None

        return res.get_energy(), res.get_charges(), res.get_gradient()


class XtbRunner:
    def __init__(self, chrg: float, multip: float, parm_folder: str, atomic_numbers: np.array, coords: np.array,
                 id_: str, ext_chrg_data: str = None, nthreads: int = 16):
        validate_directory_exists(parm_folder, "parm_folder")

        self.parm_folder = parm_folder
        self.__id = id_
        self.ext_chrg_data = ext_chrg_data

        self.coords = coords
        self.atomic_numbers = atomic_numbers
        self.chrg = chrg
        self.multip = multip
        self.nthreads = nthreads

        self.energy = None
        self.gradients = None

    def run(self):
        #import time
        #start = time.time()
        base_fold = validate_env_var('CONDA_PREFIX')
        os.environ['XTBHOME'] = base_fold
        os.environ['XTBPATH'] = self.parm_folder
        os.environ['PKG_CONFIG_PATH'] = base_fold + '/lib/pkgconfig'
        os.environ['MKL_NUM_THREADS'] = str(self.nthreads)
        os.environ['OMP_NUM_THREADS'] = f'{self.nthreads},1'
        os.environ['OMP_MAX_ACTIVE_LEVELS'] = '1'
        os.environ['OMP_STACKSIZE'] = '10G'
        try:
            calc = Calculator(param=get_method("GFN2-xTB"), numbers=self.atomic_numbers,
                              positions=self.coords * ANGSTROM_TO_BOHR, charge=self.chrg, uhf=(self.multip - 1))
            if self.ext_chrg_data is not None:
                charges = np.array(self.ext_chrg_data['charge'])
                positions = self.ext_chrg_data[['x', 'y', 'z']].to_numpy()
                numbers = np.ones(len(charges), dtype=np.int8)
                calc.set_external_charges(numbers=numbers, charges=charges, positions=positions)

            calc.set_verbosity(0)
            calc.set_accuracy(0.0001)
            calc.set_max_iterations(150)
            res = calc.singlepoint()
        except Exception as e:
            print(f'\nError in single-point calculation: {self.__id} \n{e}')
            #end = time.time()
            #print(f'Execution time: {end - start}')
            return None, None, None

        #end = time.time()
        #print(f'Execution time: {end - start}')
        return res.get_energy() * HARTREE_TO_KCAL_MOL, res.get_charges(), res.get_gradient()

