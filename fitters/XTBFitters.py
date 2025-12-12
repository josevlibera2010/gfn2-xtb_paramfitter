import pickle
import numpy as np
from inputhandlers import DFTCurveHandler
from runners import XtbRunner


def get_intern_vect_analysis(vector: list):
    pair_dists = []
    for i in range(0, len(vector)):
        for j in range(i+1, len(vector)):
            pair_dists.append(vector[i]-vector[j])
    return pair_dists

class XTBCurveAnalyzer:
    def __init__(self, root_dir: str, curve: DFTCurveHandler, id_: str):
        self.__id = id_
        self.__curve = curve
        self.__curve.energies=np.array(self.__curve.energies)
        self.hl_vect = np.array(get_intern_vect_analysis(self.__curve.energies))
        self.hl_energ = np.array(self.__curve.energies) - np.average(self.__curve.energies)

        self.__root_dir = root_dir
        self.__xtb_curve = {}

    def get_xtb_data(self):
        return self.__xtb_curve

    def get_abinitio_data(self):
        return self.__curve

    def do_intern_curve_ecf_analysis(self):
        glob_data = {}
        xtb_grad = {}
        xtb_chrg = {}
        xtb_energ = []

        for nm in self.__curve.base_names:
            if self.__curve.external_chrg:
                runner = XtbRunner(chrg=self.__curve.system['chrg'][nm], multip=self.__curve.system['mult'][nm],
                                   parm_folder=self.__root_dir, atomic_numbers=self.__curve.atomic_numbers[nm],
                                   coords=self.__curve.coordinates[nm], ext_chrg_data=self.__curve.external_charges[nm],
                                   nthreads=1, id_=f'Curve {self.__id} Base name: {nm}')
            else:
                runner = XtbRunner(chrg=self.__curve.system['chrg'][nm], multip=self.__curve.system['mult'][nm],
                                   parm_folder=self.__root_dir, atomic_numbers=self.__curve.atomic_numbers[nm],
                                   coords=self.__curve.coordinates[nm], ext_chrg_data=None, nthreads=1,
                                   id_=f'Curve {self.__id} Base name: {nm}')

            energ, chrg, grad = runner.run()

            if energ is None:
                return None, None, None, None

            elif np.isinf(chrg).any() or np.isnan(chrg).any() or np.isinf(grad).any() or np.isnan(grad).any():
                return None, None, None, None

            xtb_energ.append(energ)
            xtb_grad[nm] = grad
            xtb_chrg[nm] = chrg

        self.__xtb_curve['ENERGY'] = xtb_energ
        self.__xtb_curve['GRADIENT'] = xtb_grad
        self.__xtb_curve['CHARGE'] = xtb_chrg

        #np.savetxt(f'{self.__root_dir}/{self.__id}_energies.log', np.array(xtb_energ))
        ll_vect = np.array(get_intern_vect_analysis(xtb_energ))
        evect_diff = (self.hl_vect - ll_vect)

        ll_energ = np.array(xtb_energ) - np.average(xtb_energ)
        glob_data["HL_energy"] = list(self.hl_energ)
        glob_data["LL_energy"] = list(ll_energ)

        energ_diff = self.hl_energ - ll_energ
        dist = []
        chrg_diffs = []

        glob_data['data'] = {}
        glob_data['Z'] = list(self.__curve.atomic_numbers[self.__curve.base_names[0]])
        for nm in self.__curve.base_names:
            tmp = []
            ctmp = []
            for idx in range(len(xtb_grad[nm])):
                cdiff = xtb_chrg[nm][idx] - self.__curve.charges[nm][idx]
                ctmp.append(cdiff)
                chrg_diffs.append(np.abs(cdiff))
                ll_frc = -xtb_grad[nm][idx]
                hl_frc = self.__curve.gradients[nm][0][idx]
                dif = hl_frc - ll_frc
                try:
                    d = np.sqrt(dif.dot(dif))
                except OverflowError:
                    return None, None, None, None
                    
                tmp.append(d)
                dist.append(d)

            glob_data['data'][nm] = {'d_charg': ctmp, 'd_grad': tmp}
            dt = open(f'{self.__root_dir}/{self.__id}_data.pkl', 'wb')

            pickle.dump(glob_data, file=dt, protocol=pickle.HIGHEST_PROTOCOL)
            dt.close()
        return evect_diff, energ_diff, chrg_diffs, dist
