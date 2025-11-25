import json
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
        self.charges = self.json_data['charges']
        self.atomic_numbers = self.json_data['atomic_numbers']
        self.gradients = self.json_data['gradients']
        self.coordinates = self.json_data['coordinates']
        self.system = self.json_data['system']

        if self.external_chrg:
            self.external_charges = self.json_data.get('external_charges', {})
