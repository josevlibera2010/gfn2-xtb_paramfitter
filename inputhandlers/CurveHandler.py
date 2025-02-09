class CurveHandler:

    def __init__(self, root_path: str, base_names: list, res_format: str='log', external_chrg: bool=False,
                 read_irc: bool=False):
        self.root_path = root_path
        self.base_names = base_names
        self.external_chrg = external_chrg
        self.system = {'chrg': {}, 'mult': {}}

        self.calc_outputs = {}
        for nm in base_names:
            self.calc_outputs[nm] = f'{root_path}/{nm}.{res_format}'

        self.ext_chrg_paths = None
        if external_chrg:
            self.ext_chrg_paths = {}
            for nm in base_names:
                self.ext_chrg_paths[nm] = f'{root_path}/{nm}.chrg'

        self.energies = None
        self.gradients = None
        self.coordinates = None
        self.atomic_numbers = None
        self.charges = None
        self.external_charges = None
        if read_irc:
            self.read_irc()
        else:
            self.read_curve()


    def read_curve(self):
        pass

    def get_charges(self):
        return self.charges

    def get_energies(self):
        return self.energies

    def get_coordinates(self):
        return self.coordinates

    def get_gradients(self):
        return self.gradients

    def get_atomic_numbers(self):
        return self.atomic_numbers

    def get_external_charges(self):
        return self.external_charges

