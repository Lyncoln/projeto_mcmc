import numpy as np
import random
import itertools
import math


class MDGP:
    def __init__(self, distance_matrix: np.array, number_of_groups: int) -> None:
        self.distance_matrix = distance_matrix
        self.n = len(distance_matrix)
        self.g = number_of_groups
        self.group_size = int(self.n/self.g)

        if self.n % self.g != 0:
            raise ValueError('Os elementos devem ser perfeitamente particionados entre os grupos')

        self._set_initial_state()

    def _set_initial_state(self):
        initial_state = []
        for j in range(self.g):
            for i in range(self.group_size):
                initial_state.append(j)
        random.shuffle(initial_state)
        self.initial_state = initial_state

    @staticmethod
    def _evaluate_groups(cost, state):
        groups = list(set(state))
        sums = 0
        for i in groups:
            for j in np.where(np.array(state) == i):
                for element in list(itertools.combinations(j, 2)):
                    sums += cost[element[0]][element[1]]

        return(sums)

    @staticmethod
    def _generate_neighbors(state):
        state = list(state)
        groups = list(set(state))
        groups_change = random.sample(groups, 2)

        index_choose = []
        for i in groups_change:
            index_choose.append(np.random.choice(np.where(np.array(state) == i)[0], 1)[0])

        aux = state[index_choose[0]]
        state[index_choose[0]] = state[index_choose[1]]
        state[index_choose[1]] = aux

        return(state)

    def _boltzmann(self, state, cost, candidate_state, T):
        f_state = self._evaluate_groups(cost, state)
        f_candidate = self._evaluate_groups(cost, candidate_state)
        dif = math.exp(-abs(f_state - f_candidate)/T)
        return(dif)

    def _switch_states(self, state, candidate, cost, T, best):
        state_cost = self._evaluate_groups(cost, state)
        candidate_cost = self._evaluate_groups(cost, candidate)
        best_cost = self._evaluate_groups(cost, best)

        if state_cost < candidate_cost:
            state = candidate
            if(best_cost < candidate_cost):
                best = candidate

        else:
            if random.uniform(0, 1) < self._boltzmann(state, cost, candidate, T):
                state = candidate

        return state, state_cost, best, best_cost

    @staticmethod
    def alpha_schedule(T0, TMin, iters_per_temperature, alpha):
        schedule = []
        T = T0
        while T > TMin:
            for t in range(iters_per_temperature):
                schedule.append(T)
            T = T * alpha

        return schedule

    @staticmethod
    def exponential_schedule(T0, iters, beta):
        schedule = []
        for t in reversed(range(1, iters+1)):
            T = T0*beta**t
            if T > 0:
                schedule.append(T)

        return schedule

    @staticmethod
    def linear_schedule(T0, iters, beta):
        schedule = []
        for t in reversed(range(1, iters+1)):
            T = T0-beta*t
            if T > 0:
                schedule.append(T)

        return schedule

    @staticmethod
    def log_schedule(iters, a, b):
        schedule = []
        for t in range(1, iters+1):
            if np.log(t+b) > 0:
                T = a/np.log(t+b)
                schedule.append(T)

        return schedule

    def simulated_annealing_iterator(self, cooling_schedule):
        state = self.initial_state
        best = self.initial_state
        for T in cooling_schedule:
            neighbor = self._generate_neighbors(state)

            state, state_cost, best, best_cost = self._switch_states(state, neighbor, self.distance_matrix, T, best)

            yield T, state_cost, state, best_cost, best

    def simulated_annealing(self, cooling_schedule):
        for _, _, _, best_cost, best in self.simulated_annealing_iterator(cooling_schedule):
            final_best = best
            final_best_cost = best_cost

        return final_best_cost, final_best
