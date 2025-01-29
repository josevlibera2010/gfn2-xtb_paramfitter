from parameters import XTBParam
from fitters import XTBCurveAnalyzer
from inputhandlers import DFTCurveHandler
import os.path
import shutil
import numpy as np
import sys
import time


def sigma_scaler(x: np.float64, k: np.float64) -> np.float64:
    return 1 / (1 + np.exp(- (x-k)/k))

class ECGFittingV1:

    def __init__(self, patterns: list, parm: str, out_folder: str, bn1: list = None, bn2: list = None, path1: str = None,
                 path2: str = None, irc_path1: bool = False, irc_path2: bool = False, bias_grad: bool = False,
                 bgrad: float = 0, bias_chrg: bool = False, bchrg: float = 0, bias_ener: bool = False, bener: float = 0,
                 elim: float = 0, chrg_lim: float = 0, grad_lim: float = 0, ewall: float = 1.0E+100,
                 cwall: float = 1.0E+100, gwall: float = 1.0E+100, scale: bool = False,
                 scale_factors: tuple = (1.0E+6, 1.0E+6, 1.0E+6), read_hl_curves: bool = True):
        self.__patterns = patterns
        self.__parm = parm
        self.__path1 = None
        self.__path2 = None
        self.__bias_grad = bias_grad
        self.__bgrad = bgrad
        self.__bias_chrg = bias_chrg
        self.__bchrg = bchrg
        self.__bias_ener = bias_ener
        self.__bener = bener
        self.__elim = elim
        self.__chrg_lim = chrg_lim
        self.__grad_lim = grad_lim
        self.__ewall = ewall
        self.__cwall = cwall
        self.__gwall = gwall
        self.__scale = scale
        self.__scale_factors = scale_factors
        self.out_folder = out_folder


        self.curve1 = None
        self.curve2 = None
        if read_hl_curves:
            self.__path1 = path1
            self.__path2 = path2
            sys.stdout.write(f'DFT inputs for path1 are in: {path1}\n')
            sys.stdout.write(f'Base names for inputs: {bn1}\n')

            sys.stdout.write(f'DFT inputs for path2 are in: {path2}\n')
            sys.stdout.write(f'Base names for inputs: {bn2}\n')
            sys.stdout.flush()
            self.curve1 = DFTCurveHandler(root_path=path1, base_names=bn1, read_irc=irc_path1)
            if (path2 is not None) and (bn2 is not None):
                self.curve2 = DFTCurveHandler(root_path=path2, base_names=bn2, read_irc=irc_path2)

    def evaluate(self, solution: list, thr_id: int):
        if self.curve1 is None:
            raise Exception("Curve object has not being read/created")

        curr_dir = self.out_folder + '/' + str(thr_id)
        log = ''
        log = f'Objective ECGFittingV1 # {thr_id}\nDir: {curr_dir}\nSolution: {list(solution)}\n'

        xtb_parms = XTBParam(content=self.__parm, patterns=self.__patterns, values=solution)
        if os.path.isdir(curr_dir):
            log += f'The folder: {curr_dir} exists and is going to be cleaned!\n'
            shutil.rmtree(curr_dir)
            os.makedirs(curr_dir)
            xtb_parms.print_param_file(f'{curr_dir}/param_gfn2-xtb.txt')
        else:
            os.makedirs(curr_dir)
            xtb_parms.print_param_file(f'{curr_dir}/param_gfn2-xtb.txt')

        evect_diff1 = None
        energ_diff1 = None
        chrg_diffs1 = None
        dist1 = None
        try:
            npoints = len(self.curve1.energies)
            fitter1 = XTBCurveAnalyzer(root_dir=curr_dir, curve=self.curve1, id_='curve1')
            start = time.time()
            evect_diff1, energ_diff1, chrg_diffs1, dist1 = fitter1.do_intern_curve_ecf_analysis()
            end = time.time()
            log += f'Execution time for curve 1: {end - start}\n'

        except Exception as e:
            log += str(e)
            log += f'\nSCORE: None GRADSCORE: None CHRGSCORE: None ESCORE: None\n'
            sys.stdout.write(log)
            sys.stdout.flush()
            if self.__scale:
                return np.array([300, 1.0E+300, 1.0E+300, 1.0E+300])
            else:
                return np.array([1.0E+300, 1.0E+300, 1.0E+300, 1.0E+300])


        if energ_diff1 is None:
            log += f'\nSCORE: None GRADSCORE: None CHRGSCORE: None ESCORE: None\n'
            sys.stdout.write(log)
            sys.stdout.flush()
            if self.__scale:
                return np.array([300, 1.0E+300, 1.0E+300, 1.0E+300])
            else:
                return np.array([1.0E+300, 1.0E+300, 1.0E+300, 1.0E+300])

        evect_diff2 = None
        energ_diff2 = None
        chrg_diffs2 = None
        dist2 = None

        if self.curve2:
            try:
                npoints = npoints + len(self.curve2.energies)
                fitter2 = XTBCurveAnalyzer(root_dir=curr_dir, curve=self.curve2, id_='curve2')
                start = time.time()
                evect_diff2, energ_diff2, chrg_diffs2, dist2 = fitter2.do_intern_curve_ecf_analysis()
                end = time.time()
                log += f'Execution time for curve 2: {end - start}\n'

            except Exception as e:
                log += str(e)
                log += f'\nSCORE: None GRADSCORE: None CHRGSCORE: None ESCORE: None\n'
                sys.stdout.write(log)
                sys.stdout.flush()
                if self.__scale:
                    return np.array([300, 1.0E+300, 1.0E+300, 1.0E+300])
                else:
                    return np.array([1.0E+300, 1.0E+300, 1.0E+300, 1.0E+300])

            if energ_diff2 is None:
                log += f'\nSCORE: None GRADSCORE: None CHRGSCORE: None ESCORE: None\n'
                sys.stdout.write(log)
                sys.stdout.flush()
                if self.__scale:
                    return np.array([300, 1.0E+300, 1.0E+300, 1.0E+300])
                else:
                    return np.array([1.0E+300, 1.0E+300, 1.0E+300, 1.0E+300])

        evect_diff = None
        energ_diff = None
        chrg_diffs = None
        dist = None
        if energ_diff2 is not None:
            evect_diff = list(evect_diff1) + list(evect_diff2)
            energ_diff = list(energ_diff1) + list(energ_diff2)
            chrg_diffs = list(chrg_diffs1) + list(chrg_diffs2)
            dist = list(dist1) + list(dist2)
        else:
            evect_diff = list(evect_diff1)
            energ_diff = list(energ_diff1)
            chrg_diffs = list(chrg_diffs1)
            dist = list(dist1)

        evect_diff = np.abs(evect_diff)
        energ_diff = np.array(energ_diff)
        chrg_diffs = np.array(chrg_diffs)
        dist = np.array(dist)

        grad_score = 0
        chrg_score = 0
        e1score = 0
        if np.all(np.isfinite(chrg_diffs)) and np.all(np.isfinite(dist)) and np.all(np.isfinite(energ_diff)):
            try:
                for i in chrg_diffs:
                    chrg_score = chrg_score + i * np.exp(i * 2)

                for j in dist:
                    grad_score = grad_score + j * 100 * np.exp(j * 100)

                for k in evect_diff:
                    e1score = e1score + k * np.exp(0.5 * k)
            except Exception as e:
                log += str(e)
                log += f'\nSCORE: None GRADSCORE: None CHRGSCORE: None ESCORE: None\n'
                sys.stdout.write(log)
                sys.stdout.flush()
                if self.__scale:
                    return np.array([300, 1.0E+300, 1.0E+300, 1.0E+300])
                else:
                    return np.array([1.0E+300, 1.0E+300, 1.0E+300, 1.0E+300])

        else:
            log += f'\nSCORE: None GRADSCORE: None CHRGSCORE: None ESCORE: None\n'
            sys.stdout.write(log)
            sys.stdout.flush()
            if self.__scale:
                return np.array([300, 1.0E+300, 1.0E+300, 1.0E+300])
            else:
                return np.array([1.0E+300, 1.0E+300, 1.0E+300, 1.0E+300])

        escore = None
        try:
            e1score = 1.0 + e1score / (npoints * (npoints - 1) / 2)
            e2score = 1.0 + np.sqrt(np.dot(energ_diff, energ_diff) / npoints)
            escore = np.power(e1score * e2score, 1.5)
            grad_score = grad_score / len(dist)
            chrg_score = chrg_score / len(chrg_diffs)
            tmp_out = f'GRADSCORE: {grad_score:.10f} CHRGSCORE: {chrg_score:.10f} ESCORE: {escore:.10f}'
            tdata = np.array([grad_score, chrg_score, escore])
            if grad_score > self.__gwall or chrg_score > self.__cwall or escore > self.__ewall:
                if self.__scale:
                    log += f'SCORE: {300} ' + tmp_out + '\n'
                    sys.stdout.write(log)
                    sys.stdout.flush()
                    return np.array([300, 1.0E+300, 1.0E+300, 1.0E+300])
                else:
                    log += f'SCORE: {1.0E+300} ' + tmp_out + '\n'
                    sys.stdout.write(log)
                    sys.stdout.flush()
                    return np.array([1.0E+300, 1.0E+300, 1.0E+300, 1.0E+300])

            if escore < self.__elim:
                escore = self.__elim

            if grad_score < self.__grad_lim:
                grad_score = self.__grad_lim

            if chrg_score < self.__chrg_lim:
                chrg_score = self.__chrg_lim

            if self.__scale:
                if self.__bias_grad:
                    self.__bgrad = sigma_scaler(self.__bgrad, self.__scale_factors[0])
                if self.__bias_chrg:
                    self.__bchrg = sigma_scaler(self.__bchrg, self.__scale_factors[1])
                if self.__bias_ener:
                    self.__bener = sigma_scaler(self.__bener, self.__scale_factors[2])

                grad_score = sigma_scaler(grad_score, self.__scale_factors[0])
                chrg_score = sigma_scaler(chrg_score, self.__scale_factors[1])
                escore = sigma_scaler(escore, self.__scale_factors[2])

            def apply_biases(grad_score: float, chrg_score: float, escore: float):
                if self.__bias_grad and (grad_score > self.__bgrad) and (grad_score < 1.0E+50):
                    grad_score = grad_score * ((1 + grad_score - self.__bgrad) ** 2)

                if self.__bias_chrg and (chrg_score > self.__bchrg) and (chrg_score < 1.0E+50):
                    chrg_score = chrg_score * ((1 + chrg_score - self.__bchrg) ** 2)

                if self.__bias_ener and (escore > self.__bener) and (escore < 1.0E+50):
                    escore = escore * ((1 + escore - self.__bener) ** 2)

                return grad_score, chrg_score, escore

            grad_score, chrg_score, escore = apply_biases(grad_score, chrg_score, escore)

            tmp_gs = (1 + grad_score) * (1 + chrg_score) * (1 + escore)

        except Exception as e:
            log += str(e)
            log += f'\nSCORE: None GRADSCORE: None CHRGSCORE: None ESCORE: None\n'
            sys.stdout.write(log)
            sys.stdout.flush()
            if self.__scale:
                return np.array([300, 1.0E+300, 1.0E+300, 1.0E+300])
            else:
                return np.array([1.0E+300, 1.0E+300, 1.0E+300, 1.0E+300])



        log += f'SCORE: {tmp_gs:.10f} ' + tmp_out + '\n'
        sys.stdout.write(log)
        sys.stdout.flush()

        return np.insert(tdata, 0, tmp_gs, axis=0)

    def evaluate_parm_file(self, parm_file: str, thr_id: int):
        if self.curve1 is None:
            raise Exception("Curve object has not being read/created")

        curr_dir = self.out_folder + '/' + str(thr_id)
        log = f'Objective ECGFittingV1 # {thr_id}\nDir: {curr_dir}\nParameter file: {parm_file}\n'
        if os.path.isdir(curr_dir):
            log += f'The folder: {curr_dir} exists and is going to be cleaned!\n'
            shutil.rmtree(curr_dir)
            os.makedirs(curr_dir)
            shutil.copy(parm_file, curr_dir)
        else:
            os.makedirs(curr_dir)
            shutil.copy(parm_file, curr_dir)

        evect_diff1 = None
        energ_diff1 = None
        chrg_diffs1 = None
        dist1 = None
        try:
            npoints = len(self.curve1.energies)
            fitter1 = XTBCurveAnalyzer(root_dir=curr_dir, curve=self.curve1, id_='curve1')
            start = time.time()
            evect_diff1, energ_diff1, chrg_diffs1, dist1 = fitter1.do_intern_curve_ecf_analysis()
            end = time.time()
            log += f'Execution time for curve 1: {end - start}\n'

        except Exception as e:
            log += str(e)
            log += f'\nSCORE: None GRADSCORE: None CHRGSCORE: None ESCORE: None\n'
            sys.stdout.write(log)
            sys.stdout.flush()
            if self.__scale:
                return np.array([300, 1.0E+300, 1.0E+300, 1.0E+300])
            else:
                return np.array([1.0E+300, 1.0E+300, 1.0E+300, 1.0E+300])


        if energ_diff1 is None:
            log += f'\nSCORE: None GRADSCORE: None CHRGSCORE: None ESCORE: None\n'
            sys.stdout.write(log)
            sys.stdout.flush()
            if self.__scale:
                return np.array([300, 1.0E+300, 1.0E+300, 1.0E+300])
            else:
                return np.array([1.0E+300, 1.0E+300, 1.0E+300, 1.0E+300])

        evect_diff2 = None
        energ_diff2 = None
        chrg_diffs2 = None
        dist2 = None

        if self.curve2:
            try:
                npoints = npoints + len(self.curve2.energies)
                fitter2 = XTBCurveAnalyzer(root_dir=curr_dir, curve=self.curve2, id_='curve2')
                start = time.time()
                evect_diff2, energ_diff2, chrg_diffs2, dist2 = fitter2.do_intern_curve_ecf_analysis()
                end = time.time()
                log += f'Execution time for curve 2: {end - start}\n'

            except Exception as e:
                log += str(e)
                log += f'\nSCORE: None GRADSCORE: None CHRGSCORE: None ESCORE: None\n'
                sys.stdout.write(log)
                sys.stdout.flush()
                if self.__scale:
                    return np.array([300, 1.0E+300, 1.0E+300, 1.0E+300])
                else:
                    return np.array([1.0E+300, 1.0E+300, 1.0E+300, 1.0E+300])

            if energ_diff2 is None:
                log += f'\nSCORE: None GRADSCORE: None CHRGSCORE: None ESCORE: None\n'
                sys.stdout.write(log)
                sys.stdout.flush()
                if self.__scale:
                    return np.array([300, 1.0E+300, 1.0E+300, 1.0E+300])
                else:
                    return np.array([1.0E+300, 1.0E+300, 1.0E+300, 1.0E+300])

        evect_diff = None
        energ_diff = None
        chrg_diffs = None
        dist = None

        if energ_diff2 is not None:
            evect_diff = list(evect_diff1) + list(evect_diff2)
            energ_diff = list(energ_diff1) + list(energ_diff2)
            chrg_diffs = list(chrg_diffs1) + list(chrg_diffs2)
            dist = list(dist1) + list(dist2)
        else:
            evect_diff = list(evect_diff1)
            energ_diff = list(energ_diff1)
            chrg_diffs = list(chrg_diffs1)
            dist = list(dist1)

        evect_diff = np.abs(evect_diff)
        energ_diff = np.array(energ_diff)
        chrg_diffs = np.array(chrg_diffs)
        dist = np.array(dist)

        grad_score = 0.0
        chrg_score = 0.0
        e1score = 0.0
        if np.all(np.isfinite(chrg_diffs)) and np.all(np.isfinite(dist)) and np.all(np.isfinite(energ_diff)):
            try:
                for i in chrg_diffs:
                    chrg_score = chrg_score + i * np.exp(i * 2)

                for j in dist:
                    grad_score = grad_score + j * 100 * np.exp(j * 100)

                for k in evect_diff:
                    e1score = e1score + k * np.exp(k / (npoints * (npoints - 1) / 2))
            except Exception as e:
                log += str(e)
                log += f'\nSCORE: None GRADSCORE: None CHRGSCORE: None ESCORE: None\n'
                sys.stdout.write(log)
                sys.stdout.flush()
                if self.__scale:
                    return np.array([300, 1.0E+300, 1.0E+300, 1.0E+300])
                else:
                    return np.array([1.0E+300, 1.0E+300, 1.0E+300, 1.0E+300])

        else:
            log += f'\nSCORE: None GRADSCORE: None CHRGSCORE: None ESCORE: None\n'
            sys.stdout.write(log)
            sys.stdout.flush()
            if self.__scale:
                return np.array([300, 1.0E+300, 1.0E+300, 1.0E+300])
            else:
                return np.array([1.0E+300, 1.0E+300, 1.0E+300, 1.0E+300])

        escore = None
        try:
            e1score = 1.0 + e1score / (npoints * (npoints - 1) / 2)
            e2score = 1.0 + np.sqrt(np.dot(energ_diff, energ_diff) / npoints)
            escore = np.power(e1score * e2score, 1.5)
            grad_score = grad_score / len(dist)
            chrg_score = chrg_score / len(chrg_diffs)
            tmp_out = f'GRADSCORE: {grad_score:.10f} CHRGSCORE: {chrg_score:.10f} ESCORE: {escore:.10f}'
            tdata = np.array([grad_score, chrg_score, escore])
            if grad_score > self.__gwall or chrg_score > self.__cwall or escore > self.__ewall:
                if self.__scale:
                    log += f'SCORE: {300} ' + tmp_out + '\n'
                    sys.stdout.write(log)
                    sys.stdout.flush()
                    return np.array([300, 1.0E+300, 1.0E+300, 1.0E+300])
                else:
                    log += f'SCORE: {1.0E+300} ' + tmp_out + '\n'
                    sys.stdout.write(log)
                    sys.stdout.flush()
                    return np.array([1.0E+300, 1.0E+300, 1.0E+300, 1.0E+300])

            if escore < self.__elim:
                escore = self.__elim

            if grad_score < self.__grad_lim:
                grad_score = self.__grad_lim

            if chrg_score < self.__chrg_lim:
                chrg_score = self.__chrg_lim

            if self.__scale:
                if self.__bias_grad:
                    self.__bgrad = sigma_scaler(self.__bgrad, self.__scale_factors[0])
                if self.__bias_chrg:
                    self.__bchrg = sigma_scaler(self.__bchrg, self.__scale_factors[1])
                if self.__bias_ener:
                    self.__bener = sigma_scaler(self.__bener, self.__scale_factors[2])

                grad_score = sigma_scaler(grad_score, self.__scale_factors[0])
                chrg_score = sigma_scaler(chrg_score, self.__scale_factors[1])
                escore = sigma_scaler(escore, self.__scale_factors[2])

            def apply_biases(grad_score: float, chrg_score: float, escore: float):
                if self.__bias_grad and (grad_score > self.__bgrad) and (grad_score < 1.0E+50):
                    grad_score = grad_score * ((1 + grad_score - self.__bgrad) ** 2)

                if self.__bias_chrg and (chrg_score > self.__bchrg) and (chrg_score < 1.0E+50):
                    chrg_score = chrg_score * ((1 + chrg_score - self.__bchrg) ** 2)

                if self.__bias_ener and (escore > self.__bener) and (escore < 1.0E+50):
                    escore = escore * ((1 + escore - self.__bener) ** 2)

                return grad_score, chrg_score, escore

            grad_score, chrg_score, escore = apply_biases(grad_score, chrg_score, escore)

            tmp_gs = (1 + grad_score) * (1 + chrg_score) * (1 + escore)

        except Exception as e:
            log += str(e)
            log += f'\nSCORE: None GRADSCORE: None CHRGSCORE: None ESCORE: None\n'
            sys.stdout.write(log)
            sys.stdout.flush()
            if self.__scale:
                return np.array([300, 1.0E+300, 1.0E+300, 1.0E+300])
            else:
                return np.array([1.0E+300, 1.0E+300, 1.0E+300, 1.0E+300])



        log += f'SCORE: {tmp_gs:.10f} ' + tmp_out + '\n'
        sys.stdout.write(log)
        sys.stdout.flush()

        return np.insert(tdata, 0, tmp_gs, axis=0)

    def is_scaled(self):
        return self.__scale



