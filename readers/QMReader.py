import cclib
import numpy as np


class QMResultsReader:
    def __init__(self, logfile: str):
        print(logfile)
        self.data = cclib.io.ccopen(logfile).parse()

    def get_energy(self) -> float:
        return self.data.scfenergies[-1] * 627.509391/27.2113961318

    def get_charges(self) -> np.array:
        return self.data.atomcharges['mulliken']

    def get_coords(self) -> np.array:
        return self.data.atomcoords

    def get_gradients(self) -> np.array:
        return self.data.grads

    def get_atomic_numbers(self) -> np.array:
        return self.data.atomnos

    def get_cclib_object(self):
        return self.data


class QMIRCReader(QMResultsReader):
    def __init__(self, logfile: str):
        super(QMIRCReader, self).__init__(logfile=logfile)

    def get_energy(self) -> np.array:
        return self.data.scfenergies/27.2113961318

    def get_coords(self) -> np.array:
        return self.data.atomcoords[1:]

