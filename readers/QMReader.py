import cclib
import numpy as np

from constants import HARTREE_TO_KCAL_MOL, EV_TO_HARTREE


class QMResultsReader:
    def __init__(self, logfile: str):
        print(logfile)
        self.data = cclib.io.ccopen(logfile).parse()

    def get_energy(self) -> float:
        # cclib returns scfenergies in eV; convert to kcal/mol
        return self.data.scfenergies[-1] * HARTREE_TO_KCAL_MOL / EV_TO_HARTREE

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
        # cclib returns scfenergies in eV; convert to Hartree
        return self.data.scfenergies / EV_TO_HARTREE

    def get_coords(self) -> np.array:
        return self.data.atomcoords[1:]

