import pandas as pd

from inputhandlers.CurveHandler import CurveHandler
from readers import QMResultsReader as QMR


class DFTCurveHandler(CurveHandler):

    def __init__(self, root_path: str, base_names: list, res_format: str = 'log', external_chrg: bool = False):
        super(DFTCurveHandler, self).__init__(root_path=root_path, base_names=base_names, res_format=res_format,
                                              external_chrg=external_chrg)

    def read_curve(self):
        self.energies = []
        self.charges = {}
        self.atomic_numbers = {}
        self.gradients = {}
        self.coordinates = {}
        if self.external_chrg:
            self.external_charges = {}

        for i in self.base_names:
            gout = QMR(logfile=self.calc_outputs[i])
            self.system['chrg'][i] = gout.data.charge
            self.system['mult'][i] = gout.data.mult
            self.energies.append(gout.get_energy())
            self.charges[i] = gout.get_charges()
            self.coordinates[i] = gout.get_coords()
            self.atomic_numbers[i] = gout.get_atomic_numbers()
            self.gradients[i] = [0.0 for i in self.energies]
            try:
                self.gradients[i] = gout.get_gradients()
            except AttributeError as e:
                print(f'{e}\nGradient calculations not available for file: {i}\n')
            if self.external_chrg:
                self.external_charges[i] = pd.read_csv(self.ext_chrg_paths[i], sep=',')

