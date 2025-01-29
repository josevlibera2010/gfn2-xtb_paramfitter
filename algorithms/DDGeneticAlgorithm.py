import numpy as np
import sys
import time
from multiprocessing import Pool,TimeoutError
import matplotlib.pyplot as plt
from scipy.stats import qmc

class DDGeneticAlgorithm:
    DEFAULT_ALGORITHM_PARAMETERS={'max_num_iteration': None, 'population_size': 100, 'mutation_probability': 0.1,
                                  'elit_ratio': 0.01, 'crossover_probability': 0.5, 'parents_portion': 0.3,
                                  'crossover_type': 'uniform', 'max_iteration_without_improv': None}

    def __init__(self, function: callable, dimension: int, variable_boundaries: np.array = None,
                 algorithm_parameters=DEFAULT_ALGORITHM_PARAMETERS, convergence_curve=True, progress_bar=True):

        self.__name__ = "DDGeneticAlgorithm"
        # input function
        assert (callable(function)), "function must be callable"
        self.f = function
        # dimension
        self.dim = int(dimension)
        # input variable type
        assert (type(variable_boundaries).__module__ == 'numpy'), \
            "\n variable_boundaries must be a numpy array"

        assert (len(variable_boundaries) == self.dim), \
            "\n variable_boundaries must have a length equal dimension"

        for i in variable_boundaries:
            assert (len(i) == 2), \
                "\n boundary for each variable must be a tuple of length two."
            assert (i[0] <= i[1]), \
                "\n lower_boundaries must be smaller than upper_boundaries [lower,upper]"
        self.var_bound = variable_boundaries
        # convergence_curve
        if convergence_curve is not None:
            self.convergence_curve = convergence_curve

        # progress_bar
        if progress_bar is not None:
            self.progress_bar = progress_bar

        # input algorithm's parameters
        self.param = algorithm_parameters
        self.pop_s = int(self.param['population_size'])

        assert (1 >= self.param['parents_portion'] >= 0), "parents_portion must be in range [0,1]"

        self.par_s = int(self.param['parents_portion'] * self.pop_s)
        trl = self.pop_s - self.par_s
        if trl % 2 != 0:
            self.par_s += 1

        self.prob_mut = self.param['mutation_probability']

        assert (1 >= self.prob_mut >= 0), "mutation_probability must be in range [0,1]"
        self.orig_prob_mut = self.param['mutation_probability']

        self.prob_cross = self.param['crossover_probability']

        assert (1 >= self.prob_cross >= 0), "cross_probability must be in range [0,1]"

        self.orig_prob_cross = self.param['crossover_probability']

        assert (1 >= self.param['elit_ratio'] >= 0), "elit_ratio must be in range [0,1]"

        trl = self.pop_s * self.param['elit_ratio']

        if trl < 1 and self.param['elit_ratio'] > 0:
            self.num_elit = 1
        else:
            self.num_elit = int(trl)

        assert (self.par_s >= self.num_elit), "\n number of parents must be greater than number of elits"

        if self.param['max_num_iteration'] is None:
            self.iterate = 0
            for i in range(0, self.dim):
                if self.var_type[i] == 'int':
                    self.iterate += (self.var_bound[i][1] - self.var_bound[i][0]) * self.dim * (100 / self.pop_s)
                else:
                    self.iterate += (self.var_bound[i][1] - self.var_bound[i][0]) * 50 * (100 / self.pop_s)
            self.iterate = int(self.iterate)
            if (self.iterate * self.pop_s) > 10000000:
                self.iterate = 10000000 / self.pop_s
        else:
            self.iterate = int(self.param['max_num_iteration'])

        self.c_type = self.param['crossover_type']

        assert (self.c_type == "uniform" or self.c_type == "one_point" or self.c_type == "two_point"), \
            "\n crossover_type must 'uniform', 'one_point', or 'two_point' Enter string"

        self.stop_mniwi = False
        if self.param['max_iteration_without_improv'] is None:
            self.mniwi = self.iterate + 1
        else:
            self.mniwi = int(self.param['max_iteration_without_improv'])

        # the final report list
        self.report = []
        self.eval_id = 0

    def run(self, log_path: str, guess: list = None, n_cpus: int = 1, timeout: int = 1000.0, pre_scaled=False):
        assert (len(guess) == self.dim), \
            "\n guess must have the same number of dimensions as the dimensions of the solutions"



        # Initial Population using Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=self.dim)
        initial_samples = sampler.random(n=self.pop_s)
        pop = qmc.scale(initial_samples, self.var_bound[:, 0], self.var_bound[:, 1])
        pop = np.column_stack((pop, np.zeros(self.pop_s)))

        ini = 0
        if guess is not None:
            ini = 1
            guess.append(0)
            pop[0]=np.array(guess)

        sols = [(pop[i, :self.dim].copy(), self._generate_eval_id()) for i in range(self.pop_s)]

        logger = open(log_path, 'w')
        logger.write('ID\tGlobalScore\tGradientScore\tChargeScore\tEnergyScore\n')

        obj_values = []
        with Pool(n_cpus) as pool:
            results_pool = [pool.apply_async(self.f, (s,)) for s in sols]
            for result in results_pool:
                try:
                    obj_values.append(result.get(timeout=timeout))
                except TimeoutError:
                    if pre_scaled:
                        obj_values.append(np.array([300, 1.0E+300, 1.0E+300, 1.0E+300]))
                    else:
                        obj_values.append(np.array([1.0E+300, 1.0E+300, 1.0E+300, 1.0E+300]))

            #sys.stdout.write(f'Evaluation:\n {obj_values}\n\n')
            #sys.stdout.flush()

        for p in range(len(obj_values)):
            pop[p] = np.append(sols[p][0], obj_values[p][0])
            #sys.stdout.write(f'Iteración:\n {p}\n')
            #sys.stdout.flush()
            logger.write(f'{sols[p][1]}\t{obj_values[p][0]}\t{obj_values[p][1]}\t{obj_values[p][2]}\t{obj_values[p][3]}\n')
            logger.flush()

        # Report
        self.report = []
        self.best_function = pop[0, self.dim].copy()
        self.best_variable = pop[0, : self.dim].copy()

        t = 1
        counter = 0
        while t <= self.iterate:
            if self.progress_bar:
                self.progress(t, self.iterate, status="GA is running...")

            # Sort
            pop = pop[pop[:, self.dim].argsort()]

            if pop[0, self.dim] < self.best_function:
                counter = 0
                self.best_function = pop[0, self.dim].copy()
                self.best_variable = pop[0, : self.dim].copy()
                for ig in range(len(self.best_variable)):
                    if self.best_variable[ig] < (self.var_bound[ig][0] + abs(self.var_bound[ig][0]-self.var_bound[ig][1]) * 0.05):
                        self.var_bound[ig][0] = self.best_variable[ig] - abs(self.best_variable[ig]) * 0.2

                    elif self.best_variable[ig] > (self.var_bound[ig][1] - abs(self.var_bound[ig][0]-self.var_bound[ig][1]) * 0.05):
                        self.var_bound[ig][1] = self.best_variable[ig] + abs(self.best_variable[ig]) * 0.2

            else:
                counter += 1
            # Report

            self.report.append(pop[0, self.dim])

            # Normalizing objective function
            minobj = pop[0, self.dim]
            if minobj < 0:
                normobj = pop[:, self.dim] + abs(minobj)

            else:
                normobj = pop[:, self.dim].copy()

            maxnorm = np.amax(normobj)
            normobj = maxnorm - normobj + 1

            # Calculate probability
            sum_normobj = np.sum(normobj)
            prob = np.zeros(self.pop_s)
            prob = normobj / sum_normobj
            cumprob = np.cumsum(prob)

            # Select parents
            par = np.array([np.zeros(self.dim + 1)] * self.par_s)

            for k in range(0, self.num_elit):
                par[k] = pop[k].copy()
            for k in range(self.num_elit, self.par_s):
                index = np.searchsorted(cumprob, np.random.random())
                par[k] = pop[index].copy()

            ef_par_list = np.array([False] * self.par_s)
            par_count = 0
            while par_count == 0:
                for k in range(0, self.par_s):
                    if np.random.random() <= self.prob_cross:
                        ef_par_list[k] = True
                        par_count += 1

            ef_par = par[ef_par_list].copy()

            # New generation
            pop = np.array([np.zeros(self.dim + 1)] * self.pop_s)

            for k in range(0, self.par_s):
                pop[k] = par[k].copy()

            if t > 1 and self.prob_mut > 0.1:
                self.prob_mut = self.prob_mut * 0.7

            sols = [()] * (self.pop_s - self.par_s)
            pos = 0
            for k in range(self.par_s, self.pop_s, 2):
                r1 = np.random.randint(0, par_count)
                r2 = np.random.randint(0, par_count)
                pvar1 = ef_par[r1, : self.dim].copy()
                pvar2 = ef_par[r2, : self.dim].copy()
                ch = self.cross(pvar1, pvar2, self.c_type)
                ch1 = ch[0].copy()
                ch2 = ch[1].copy()

                ch1 = self.mut(ch1)
                ch2 = self.mutmidle(ch2, pvar1, pvar2)
                self.eval_id += 1
                sols[pos] = (ch1, self.eval_id)
                pos += 1
                self.eval_id += 1
                sols[pos] = (ch2, self.eval_id)
                pos += 1

            obj_values = []
            with Pool(n_cpus) as pool:
                results_pool = [pool.apply_async(self.f, (s,)) for s in sols]
                for result in results_pool:
                    try:
                        obj_values.append(result.get(timeout=timeout))
                    except TimeoutError:
                        if pre_scaled:
                            obj_values.append(np.array([300, 1.0E+300, 1.0E+300, 1.0E+300]))
                        else:
                            obj_values.append(np.array([1.0E+300, 1.0E+300, 1.0E+300, 1.0E+300]))

            for p in range(self.par_s, self.pop_s):
                pop[p] = np.append(sols[p - self.par_s][0], obj_values[p - self.par_s][0])
                logger.write(f'{sols[p - self.par_s][1]}\t{obj_values[p - self.par_s][0]}\t{obj_values[p - self.par_s][1]}\t'
                             f'{obj_values[p - self.par_s][2]}\t{obj_values[p - self.par_s][3]}\n')
                logger.flush()

            t += 1

            sys.stdout.write('\r The best solution found:\n %s' % (self.best_variable))
            sys.stdout.write('\n\n Objective function:\n %s\n' % (self.best_function))
            sys.stdout.flush()

            if counter > self.mniwi:
                pop = pop[pop[:, self.dim].argsort()]
                if pop[0, self.dim] >= self.best_function:
                    t = self.iterate
                    if self.progress_bar:
                        self.progress(t, self.iterate, status="GA is running...")
                    time.sleep(2)
                    t += 1
                    self.stop_mniwi = True

        # Sort
        pop = pop[pop[:, self.dim].argsort()]

        if pop[0, self.dim] < self.best_function:
            self.best_function = pop[0, self.dim].copy()
            self.best_variable = pop[0, : self.dim].copy()
        # Report

        self.report.append(pop[0, self.dim])

        self.output_dict = {'variable': self.best_variable, 'function': \
            self.best_function}
        if self.progress_bar:
            show = ' ' * 100
            sys.stdout.write('\r%s' % (show))
        sys.stdout.write('\r The best solution found:\n %s' % (self.best_variable))
        sys.stdout.write('\n\n Objective function:\n %s\n' % (self.best_function))
        sys.stdout.flush()
        re = np.array(self.report)
        if self.convergence_curve == True:
            plt.plot(re)
            plt.xlabel('Iteration')
            plt.ylabel('Objective function')
            plt.title('Genetic Algorithm')
            plt.show()

        if self.stop_mniwi == True:
            sys.stdout.write('\nWarning: GA is terminated due to the' + \
                             ' maximum number of iterations without improvement was met!')
        logger.close()

    def cross(self, x, y, c_type):

        ofs1 = x.copy()
        ofs2 = y.copy()

        if c_type == 'one_point':
            ran = np.random.randint(0, self.dim)
            for i in range(0, ran):
                if np.random.random() < self.prob_cross:
                    ofs1[i] = y[i].copy()
                    ofs2[i] = x[i].copy()

        if c_type == 'two_point':
            ran1 = np.random.randint(0, self.dim)
            ran2 = np.random.randint(ran1, self.dim)

            for i in range(ran1, ran2):
                if np.random.random() < self.prob_cross:
                    ofs1[i] = y[i].copy()
                    ofs2[i] = x[i].copy()

        if c_type == 'uniform':
            for i in range(0, self.dim):
                ran = np.random.random()
                if ran < self.prob_cross:
                    ofs1[i] = y[i].copy()
                    ofs2[i] = x[i].copy()

        return np.array([ofs1, ofs2])

    def mut(self, x):

        for i in range(0, self.dim):
            ran = np.random.random()
            if ran < self.prob_mut:
                x[i] = (self.var_bound[i][1] -
                        np.random.random() * (self.var_bound[i][1] - self.var_bound[i][0]))

        return x

    def mutmidle(self, x, p1, p2):

        for i in range(0, self.dim):
            ran = np.random.random()
            if ran < self.prob_mut:
                if p1[i] < p2[i]:
                    x[i] = p1[i] + np.random.random() * (p2[i] - p1[i])
                elif p1[i] > p2[i]:
                    x[i] = p2[i] + np.random.random() * (p1[i] - p2[i])
                else:
                    x[i] = self.var_bound[i][0] + np.random.random() * \
                           (self.var_bound[i][1] - self.var_bound[i][0])
        return x

    def _generate_eval_id(self):
        """Generates a unique evaluation ID."""
        self.eval_id += 1
        return self.eval_id

    @staticmethod
    def progress(count, total, status=''):
        bar_len = 50
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '|' * filled_len + '_' * (bar_len - filled_len)

        sys.stdout.write('\r%s %s%s %s' % (bar, percents, '%', status))
        sys.stdout.flush()
