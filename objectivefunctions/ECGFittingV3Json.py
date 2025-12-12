from parameters import XTBParam
from fitters import XTBCurveAnalyzer
from inputhandlers import JSONCurveHandler
import os.path
import shutil
import numpy as np
import sys
import time
from typing import List, Dict, Tuple, Optional, Union


def sigma_scaler(x: np.float64, k: np.float64) -> np.float64:
    """
    Apply sigmoid scaling to a value.

    Args:
        x: Value to scale
        k: Scaling factor

    Returns:
        Scaled value using sigmoid function
    """
    return 1 / (1 + np.exp(- (x - k) / k))

class ECGFittingV3Json:
    """
    Energy Charge Gradient fitting (version 3) class for evaluating
    parameter sets in quantum mechanical calculations using a multi-objective
    optimization approach.

    This version reads curve data from JSON files using JSONCurveHandler.
    Each JSON file contains all the data for one curve.
    """

    def __init__(self, patterns: list, parm: str, out_folder: str, paths: list = None,
                 cv_objectives: list[list] = None, bias_grad: bool = False, bgrad: float = 0, bias_chrg: bool = False,
                 bchrg: float = 0, bias_ener: bool = False, bener: float = 0, elim: float = 0, clim: float = 0,
                 glim: float = 0, ewall: float = 1.0E+100, cwall: float = 1.0E+100, gwall: float = 1.0E+100,
                 scale: bool = False, scale_factors: tuple = (1.0E+6, 1.0E+6, 1.0E+6), read_hl_curves: bool = True,
                 ref_grad: float = 0.020, ref_ener: float = 10.0, external_chrg: bool = False):
        self.__patterns = patterns
        self.__parm = parm
        self.__paths = None
        self.__bias_grad = bias_grad
        self.__bgrad = np.float64(bgrad)
        self.__bias_chrg = bias_chrg
        self.__bchrg = np.float64(bchrg)
        self.__bias_ener = bias_ener
        self.__bener = np.float64(bener)
        self.__elim = elim
        self.__chrg_lim = clim
        self.__grad_lim = glim
        self.__ewall = ewall
        self.__cwall = cwall
        self.__gwall = gwall
        self.__scale = scale
        self.__scale_factors = scale_factors
        self.out_folder = out_folder
        self.__ref_grad = ref_grad
        self.__ref_ener = ref_ener
        self.__external_chrg = external_chrg


        self.curves = []
        self.cv_objectives=[]

        if read_hl_curves:
            assert paths is not None, "Error: \"paths\" argument must be provided"

            self.__paths = paths

            pos=0
            for json_path in paths:
                pos+=1
                sys.stdout.write(f'JSON file for curve {pos}: {json_path}\n')
                sys.stdout.flush()

                self.curves.append(JSONCurveHandler(json_path=json_path, external_chrg=self.__external_chrg))
                self.cv_objectives.append([True, True, True])

            pos = 0
            for crv in self.curves:
                pos += 1
                assert crv is not None, f"Error parsing Curve{pos} from JSON file {self.__paths[pos - 1]}"

        if cv_objectives:
            assert len(cv_objectives) == len(self.curves), \
                "Error, the \"cv_objectives\" parameter must contain one object per added curve"
            for i in range(len(cv_objectives)):
                assert (isinstance(cv_objectives[i],list) and len(cv_objectives[i]) == 3 and
                        False not in [isinstance(v,bool) for v in cv_objectives[i]]), \
                    (f"Error, in \"cv_objectives\" parameter, object in position \"{i}\", must be a length three list "
                     f"of booleans")

            self.cv_objectives=cv_objectives



    @staticmethod
    def format_sol(sol: list) -> str:
        fout = f"[{sol[0]}"

        for i in sol[1:]:
            fout += f", {i:.6f}"
        fout += "]"
        return fout

    def apply_biases(self, grad_score: float, chrg_score: float, escore: float):
        if self.__bias_grad and (grad_score > self.__bgrad) and (grad_score < 1.0E+50):
            grad_score = grad_score * ((1 + grad_score - self.__bgrad) ** 2)

        if self.__bias_chrg and (chrg_score > self.__bchrg) and (chrg_score < 1.0E+50):
            chrg_score = chrg_score * ((1 + chrg_score - self.__bchrg) ** 2)

        if self.__bias_ener and (escore > self.__bener) and (escore < 1.0E+50):
            escore = escore * ((1 + escore - self.__bener) ** 2)

        return grad_score, chrg_score, escore

    def evaluate(self, thr_id: int, solution: list=None, pfile: bool = False, parm_file: str = None) -> np.array:
        log = ''
        curr_dir = self.out_folder + '/' + str(thr_id)

        if pfile and type(parm_file) is str:
            assert os.path.isfile(parm_file), \
                "Error, if you set \"pfile\", \"parm_file\" must contain a valid file path"
            log += (f'Objective ECGFittingV3Json # {thr_id}\n'
                    f'Dir: {curr_dir}\n'
                    f'Parameter file: {parm_file}\n')

            if os.path.isdir(curr_dir):
                log += f'The folder: {curr_dir} exists and is going to be cleaned!\n'
                shutil.rmtree(curr_dir)
                os.makedirs(curr_dir)
                shutil.copy(parm_file, curr_dir)
            else:
                os.makedirs(curr_dir)
                shutil.copy(parm_file, curr_dir)
        else:
            log = (f'Objective ECGFittingV3Json # {thr_id}\n'
                   f'Dir: {curr_dir}\n'
                   f'Solution: {self.format_sol(solution)}\n')

            assert (type(solution) is list) or (type(solution) is np.ndarray), \
                (f"Error: \"solution\" must be a list, "
                 f"your input variable's type: \"{type(solution)}\"")
            assert len(solution) > 0, "Error: \"solution\" list is empty"

            xtb_parms = XTBParam(content=self.__parm, patterns=self.__patterns, values=solution)
            if os.path.isdir(curr_dir):
                log += f'The folder: {curr_dir} exists and is going to be cleaned!\n'
                shutil.rmtree(curr_dir)
                os.makedirs(curr_dir)
                xtb_parms.print_param_file(f'{curr_dir}/param_gfn2-xtb.txt')
            else:
                os.makedirs(curr_dir)
                xtb_parms.print_param_file(f'{curr_dir}/param_gfn2-xtb.txt')

        evect_diff = []
        energ_diff = []
        chrg_diffs = []
        dist = []
        npoints = 0
        for cv in range(len(self.curves)):
            try:

                fitter = XTBCurveAnalyzer(root_dir=curr_dir, curve=self.curves[cv], id_=f'curve{cv+1}')
                start = time.time()
                evect, energ, chrg, dst = fitter.do_intern_curve_ecf_analysis()
                end = time.time()
                log += f'Execution time for curve {cv+1}: {end - start}\n'

                if (evect is None) or (energ is None) or (chrg is None) or (dst is None):
                    log += f"One or more OFs' value(s) could not be computed!\n"
                    log += f'\nSCORE: 1.0E+300 GRADSCORE: None CHRGSCORE: None ESCORE: None\n'
                    sys.stdout.write(log)
                    sys.stdout.flush()
                    return np.array([1.0E+300, 1.0E+300, 1.0E+300, 1.0E+300])

                if self.cv_objectives[cv][0]:
                    npoints += len(self.curves[cv].energies)
                    evect_diff+=list(evect)
                    energ_diff+=list(energ)
                if self.cv_objectives[cv][1]:
                    chrg_diffs+=list(chrg)
                if self.cv_objectives[cv][2]:
                    dist+=list(dst)

            except Exception as e:
                log += str(e)
                log += f'\nSCORE: 1.0E+300 GRADSCORE: None CHRGSCORE: None ESCORE: None\n'
                sys.stdout.write(log)
                sys.stdout.flush()
                return np.array([1.0E+300, 1.0E+300, 1.0E+300, 1.0E+300])

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
                    pen=0
                    if j > self.__ref_grad and self.__ref_grad > 0:
                        pen = (j - self.__ref_grad) / self.__ref_grad

                    grad_score = grad_score + j * 100 * np.exp(j * 100 * (1+pen))

                for k in evect_diff:
                    e1score = e1score + k * np.exp(k) / (npoints * (npoints - 1) / 2)

                for i in range(len(energ_diff)):
                    if np.abs(energ_diff[i]) > self.__ref_ener:
                        energ_diff[i]=energ_diff[i] * np.exp(3.0*(np.abs(energ_diff[i])-self.__ref_ener))



            except Exception as e:
                log += str(e)
                log += f'\nSCORE: None GRADSCORE: None CHRGSCORE: None ESCORE: None\n'
                sys.stdout.write(log)
                sys.stdout.flush()
                return np.array([1.0E+300, 1.0E+300, 1.0E+300, 1.0E+300])


        else:
            log += f'\nSCORE: None GRADSCORE: None CHRGSCORE: None ESCORE: None\n'
            sys.stdout.write(log)
            sys.stdout.flush()
            return np.array([1.0E+300, 1.0E+300, 1.0E+300, 1.0E+300])

        escore = None
        try:
            e1score = 1.0 + e1score
            e2score = 1.0 + np.sqrt(np.dot(energ_diff, energ_diff) / npoints)
            escore = np.power(e1score * e2score, 1.5)
            grad_score = 1.0 + grad_score / len(dist)
            chrg_score = 1.0 + chrg_score / len(chrg_diffs)
            tmp_out = f'GRADSCORE: {grad_score:.10f} CHRGSCORE: {chrg_score:.10f} ESCORE: {escore:.10f}'
            tdata = np.array([grad_score, chrg_score, escore])
            if grad_score > self.__gwall or chrg_score > self.__cwall or escore > self.__ewall:
                if self.__scale:
                    grad_score = sigma_scaler(grad_score, self.__scale_factors[0])
                    chrg_score = sigma_scaler(chrg_score, self.__scale_factors[1])
                    escore = sigma_scaler(escore, self.__scale_factors[2])

                    log += f'SCORE: {1.0E+300} ' + tmp_out + '\n'
                    sys.stdout.write(log)
                    sys.stdout.flush()
                    return np.array([1.0E+300, grad_score, chrg_score, escore])
                else:
                    log += f'SCORE: {1.0E+300} ' + tmp_out + '\n'
                    sys.stdout.write(log)
                    sys.stdout.flush()
                    return np.array([1.0E+300, grad_score, chrg_score, escore])

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

            grad_score, chrg_score, escore = self.apply_biases(grad_score, chrg_score, escore)

            tmp_gs = grad_score * chrg_score * escore

        except Exception as e:
            log += str(e)
            log += f'\nSCORE: None GRADSCORE: None CHRGSCORE: None ESCORE: None\n'
            sys.stdout.write(log)
            sys.stdout.flush()
            return np.array([1.0E+300, 1.0E+300, 1.0E+300, 1.0E+300])


        log += f'SCORE: {tmp_gs:.10f} ' + tmp_out + '\n'
        sys.stdout.write(log)
        sys.stdout.flush()

        return np.insert(tdata, 0, tmp_gs, axis=0)

    def evaluate_parm_file(self, parm_file: str, thr_id: int):
        return self.evaluate(thr_id=thr_id, pfile=True, parm_file=parm_file)

    def is_scaled(self):
        return self.__scale



