import json
import numpy as np
from inputhandlers.CurveHandler import CurveHandler


class JSONCurveHandler(CurveHandler):

    def __init__(self, json_path: str, external_chrg: bool = False):
        with open(json_path, 'r') as f:
            self.json_data = json.load(f)

        base_names = list(self.json_data['system']['chrg'].keys())

        super(JSONCurveHandler, self).__init__(
            root_path='',
            base_names=base_names,
            res_format='json',
            external_chrg=external_chrg
        )

    def read_curve(self):
        self.energies = self.json_data['energies']

        # Convert atomic_numbers, coordinates, charges, and gradients to numpy arrays
        self.atomic_numbers = {}
        for key, value in self.json_data['atomic_numbers'].items():
            self.atomic_numbers[key] = np.array(value)

        self.coordinates = {}
        for key, value in self.json_data['coordinates'].items():
            self.coordinates[key] = np.array(value)/1.8897259885789

        self.charges = {}
        for key, value in self.json_data['charges'].items():
            self.charges[key] = np.array(value)

        # Convert gradients (nested structure: dict -> list -> 2D array)
        # XTBFitters expects gradients[key][0] to be a 2D array of shape (n_atoms, 3)
        self.gradients = {}
        for key, value in self.json_data['gradients'].items():
            # Convert list of [x,y,z] vectors to a single 2D array wrapped in a list
            self.gradients[key] = [np.array(value)]

        self.system = self.json_data['system']

        if self.external_chrg:
            self.external_charges = self.json_data.get('external_charges', {})
