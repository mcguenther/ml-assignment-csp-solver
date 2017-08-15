import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.lines
import re
import time
import sys
import pycosat
import random
import xml.etree.ElementTree as ET
import numpy as np
import csv
import os
from random import randint
from operator import itemgetter
from mpl_toolkits.mplot3d import Axes3D

EXIT_ARGUMENT_ERROR = 2

EXIT_FILE_ERROR = 3

FILE_MODEL_IDENTIFYER = ".*(\.dimacs|\.xml)"

FILE_MODEL_DIMACS_IDENTIFYER = ".*(\.dimacs)"

FILE_MODEL_XML_IDENTIFYER = ".*(\.xml)"

FILE_MODEL_FEATURE_IDENTIFYER = ".*feature([0-9]*).txt"

FILE_MODEL_INTERATIONS_IDENTIFYER = ".*interactions([0-9]*).txt"


def timeit(method):
    def timed(*args, **kw):
        start = time.time()
        result = method(*args, **kw)
        end = time.time()
        print('%r (%r, %r) %2.4f sec' % \
              (method.__name__, args, kw, end - start))
        return result

    return timed


class VM:
    def __init__(self, dimacs):
        pass


class Visualizer:
    def __init__(self, model, sleep_cycles=10, sleep_cycles_pheromones=None):
        self.model = model
        if not sleep_cycles_pheromones:
            sleep_cycles_pheromones = sleep_cycles
        self.sequences = []
        plt.ion()

        self.max_pheromones = 30
        self.last_annotation = None

        if self.model.num_objectives == 1:
            self.fig = plt.figure()
            self.ax_cost_history = self.fig.add_subplot(211)
            self.ax_cost_history.set_title("Cost of best candidates over past epochs")
            self.ax_cost_history.set_xlabel('#sample')
            self.ax_cost_history.set_ylabel('cost of best solution')
            self.plot_data, = self.ax_cost_history.plot(self.sequences)

            self.ax_pheromone_history = self.fig.add_subplot(212)
            self.init_pheromone_graph()
            self.fig.tight_layout()
        else:
            # for pareto front visualization
            self.fig = plt.figure(figsize=(9, 6))
            self.ax = Axes3D(self.fig)
            self.ax.set_xlabel("\nObjective1")
            self.ax.set_ylabel("\nObjective2")
            self.ax.set_zlabel("\n\n\nObjective3")
            self.ax.set_xlim3d(70, 130)
            self.ax.set_ylim3d(14000, 20000)
            self.ax.set_zlim3d(3500000, 7000000)
            legend1 = matplotlib.lines.Line2D([0], [0], linestyle="none", c="blue", marker="o")
            legend2 = matplotlib.lines.Line2D([0], [0], linestyle="none", c="black", marker="o")
            legend3 = matplotlib.lines.Line2D([0], [0], linestyle="none", c="red", marker="s")
            self.ax.legend([legend1, legend2, legend3],
                           ["Current Population", "Local Pareto Front", "Global Pareto Front"],
                           numpoints=1)
            # plt.ion()

        # self.fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=10)
        pylab.pause(1.e-8)
        self.sleep_cycles_costs = sleep_cycles
        self.sleep_state = 1
        self.sleep_state_pheromones = 1
        self.sleep_cycles_pheromones = sleep_cycles_pheromones

    def update_pareto(self, population, local_front, global_front):
        # update pareto front visualization
        for solution in population:
            self.ax.scatter(solution.cost[0], solution.cost[1], solution.cost[2], s=30, color="blue", marker="o")
        for solution in global_front:
            self.ax.scatter(solution.cost[0], solution.cost[1], solution.cost[2], s=80, color="red", marker="s")
        for solution in local_front:
            self.ax.scatter(solution.cost[0], solution.cost[1], solution.cost[2], s=30, color="black", marker="o")
        plt.pause(0.005)

    def init_pheromone_graph(self):
        self.ax_pheromone_history.set_title("Current values of pheromones")
        self.ax_pheromone_history.set_xlabel('component id')
        self.ax_pheromone_history.set_ylabel('pheromone value')

    #    def add_sequence(self):
    #       self.sequences = []

    def add_solution(self, cost):
        if self.sleep_state % (self.sleep_cycles_costs + 1) == 0:
            self.add_solution_forced(cost)
            self.sleep_state = 1
        else:
            self.sleep_state += 1

    def add_solution_forced(self, cost):
        # if not self.sequences:
        #    self.add_sequence()
        self.sequences.append(cost)
        self.update_cost_graph()

    def update_cost_graph(self):
        self.plot_data.set_data(np.arange(len(self.sequences)), self.sequences)
        if self.last_annotation:
            self.last_annotation.remove()

        min_index = self.sequences.index(min(self.sequences))
        xy = (min_index, self.sequences[min_index] + 1)
        self.last_annotation = self.ax_cost_history.annotate(round(self.sequences[min_index], 3),
                                                             xy=xy, textcoords='data',
                                                             bbox=dict(boxstyle="round", fc="0.8"))
        self.ax_cost_history.plot(self.sequences, "b-")
        pylab.pause(1.e-8)

    def visualize(self):
        costs = self.sequences
        if self.model.num_objectives == 1:
            print("Best: " + str(min(costs)) + " | Worst: " +
                  str(max(costs)) + " | Avg: " + str(sum(costs) / len(costs)))
            self.update_cost_graph()
        else:
            pass
        plt.show(block=True)

    def set_sleep_time_costs(self, x):
        self.sleep_cycles_costs = x

    def set_sleep_time_pheromones(self, x):
        self.sleep_cycles_pheromones = x

    def update_pheromone_graph(self, model, literals, pheromones):
        if self.model.num_objectives == 1:
            if self.sleep_state_pheromones % (self.sleep_cycles_pheromones + 1) == 0:
                self.update_pheromone_graph_forced(model, literals, pheromones)
                self.sleep_state_pheromones = 1
            else:
                self.sleep_state_pheromones += 1

    # @timeit
    def update_pheromone_graph_forced(self, model, literals, pheromones):
        truncated_literals = literals[:self.max_pheromones]
        p_list = list(pheromones)
        truncated_pheromones = p_list[:self.max_pheromones]
        comp_names = list(
            (model.variable2name[abs(literal)] + "=" + str(0 if literal < 0 else 1) for literal in truncated_literals))
        self.ax_pheromone_history.clear()
        # Set number of ticks for x-axis
        # self.ax_pheromone_history.xticks(np.arange(num_pheromones), comp_names, rotation='vertical')
        self.ax_pheromone_history.set_xticks(np.arange(self.max_pheromones))
        # Set ticks labels for x-axis
        self.ax_pheromone_history.set_xticklabels(comp_names, rotation='vertical', fontsize=8)

        self.init_pheromone_graph()
        # let's use two nice colours for bars of the same component:
        # #FC89AC "Tickle Me Pink" and #DE5285 "Fandango Pink"
        colors = ['#FC89AC'] * 2 + ['#DE5285'] * 2
        self.ax_pheromone_history.bar(np.arange(self.max_pheromones), truncated_pheromones, color=colors)
        self.fig.tight_layout()
        pylab.pause(1.e-8)


class DummyVisualizer:
    def __init__(self):
        pass

    def update_pareto(self, population, local_front, global_front):
        pass

    def add_sequence(self):
        pass

    def add_solution(self, solution):
        pass

    def add_solution_forced(self, solution):
        pass

    def visualize(self):
        pass

    def set_sleep_time_costs(self, x):
        pass

    def set_sleep_time_pheromones(self, x):
        pass

    def update_pheromone_graph(self, model, literals, pheromones):
        pass

    def update_pheromone_graph_forced(self, model, literals, pheromones):
        pass


class Model:
    def __init__(self, vm_path, features_by_objective, interactions_by_objective):
        self.vm_path = vm_path

        self.num_features = len(features_by_objective[0])
        self.num_objectives = len(features_by_objective)
        self.num_interactions = len(interactions_by_objective[0])

        self.costs_single_matrix = np.zeros((self.num_features, self.num_objectives))
        self.interactions = np.zeros((self.num_objectives, self.num_features, self.num_interactions), dtype=np.bool)
        self.influences_interactions = np.zeros((self.num_interactions, self.num_objectives), dtype=np.float64)

        if is_model_dimacs(vm_path):
            self.name2variable, self.variable2name, self.constraint_list = self.read_vm_dimacs(vm_path)
        else:
            self.name2variable, self.variable2name, self.constraint_list = self.read_vm_xml(vm_path)

        self.name2variable["root"] = 0
        self.variable2name[0] = "root"

        for o in range(self.num_objectives):
            for variable, cost in [[self.name2variable[feature_name], features_by_objective[o][feature_name]]
                                   for feature_name in features_by_objective[o]]:
                self.costs_single_matrix[variable, o] = cost

            # self.costs_single = self.costs_single_matrix.copy().reshape(len(features))

            for i, features in enumerate(interactions_by_objective[o]):
                influence = interactions_by_objective[o][features]
                variables = [self.name2variable[feature] for feature in features]
                self.interactions[o, variables, i] = 1
                self.influences_interactions[i, o] = influence
                # self.expected_interaction_matches = np.sum(self.interactions, axis=1)
        self.expected_interaction_matches = np.sum(self.interactions, axis=1)

        self.constraints_by_variable = {}
        for constraint in self.constraint_list:
            for literal in constraint:
                variable = abs(literal)
                if variable not in self.constraints_by_variable:
                    self.constraints_by_variable[variable] = []
                self.constraints_by_variable[variable].append(constraint)

        worst_solution = Solution(self, start_literals=range(self.num_features))
        self.highest_costs_possible = worst_solution.get_cost()

    # @timeit
    def violates_constraints(self, current_literals, literal):
        var = abs(literal)
        new_literals = set(x for x in current_literals)
        if (literal * -1) in new_literals:
            return True
        new_literals.add(literal)

        if var in self.constraints_by_variable:
            constraints = self.constraints_by_variable[var]
            for constraint in constraints:
                num_False = 0
                for constraint_literal in constraint:
                    if constraint_literal in new_literals:
                        break
                    elif constraint_literal * -1 in new_literals:
                        num_False += 1
                if num_False == len(constraint):
                    return True
        return False

    def assess_cost_delta(self, current_literals, new_literal):
        if new_literal < 0:
            delta = 0
        else:
            delta = self.costs_single[new_literal]
            assumed_features = set(list(current_literals) + [new_literal])
            if new_literal in self.influences_interactions:
                interactions = self.influences_interactions[new_literal]
                for interaction, cost in interactions:
                    if set(interaction) <= assumed_features:
                        delta += cost
        return delta

        # @timeit

    def read_vm_xml(self, file_model):
        print("it's an XML!")
        # content = read_file(file_model)
        # tree = ET.fromstring(content)

        tree = ET.parse(file_model)
        root = tree.getroot()
        name2literal = {}
        literal2name = {}
        name2excluded = {}
        name2is_optional = {}
        parent2children = {}

        cnf = []
        for option in root.iter('configurationOption'):
            key = option.find('name').text
            is_optional = str2bool(option.find('optional').text)
            new_literal = len(name2literal) + 1
            name2literal[key] = new_literal
            literal2name[new_literal] = key
            name2is_optional[key] = is_optional

            parent_option = option.find('parent')
            if parent_option.text and parent_option.text.strip():
                parent_id = name2literal[parent_option.text]
                if parent_id not in parent2children:
                    parent2children[parent_id] = set()
                parent2children[parent_id].add(new_literal)

            excluded_options_tag = option.find('excludedOptions')
            excluded_options = excluded_options_tag.findall("options")
            if excluded_options:
                name2excluded[key] = set()
                for ex_opt in excluded_options:
                    name2excluded[key].add(ex_opt.text)
            else:
                if not is_optional:
                    cnf.append([new_literal])

        candidate_cnf = []
        for key in name2excluded:
            # try to add restrictions only once per excluded list
            literal_key = name2literal[key]
            excluded_ids = []

            for excluded_name in name2excluded[key]:
                excluded_id = name2literal[excluded_name]
                excluded_ids.append(excluded_id)
                # add key -> excluded_name  <=> -key excluded_id
                candidate_cnf.append(sorted([-literal_key, -excluded_id]))

            if not name2is_optional[key]:
                # one of the group needs to be selected
                all_options = list(excluded_ids)
                all_options.append(literal_key)
                candidate_cnf.append(sorted(all_options))

        for parent_id in parent2children:
            children = parent2children[parent_id]
            # add child -> parent
            for child in children:
                candidate_cnf.append(sorted([-child, parent_id]))

            # add parent -> any child
            parent_implicates_any_child = [-parent_id] + list(children)
            candidate_cnf.append(sorted(parent_implicates_any_child))

        for candidate in candidate_cnf:
            if candidate not in cnf:
                cnf.append(candidate)

        return name2literal, literal2name, cnf

    # @timeit
    def read_vm_dimacs(self, file_model):
        content = read_file(file_model)
        name2literal_dict = {}
        literal2name_dict = {}
        for line in content:
            if line.startswith("c"):
                line = line.split()
                if len(line) == 3:
                    line = line[1:]
                    (val, key) = line
                    val = val.replace("$", "")
                    val = int(val)
                    name2literal_dict[key] = val
                    literal2name_dict[val] = key

        # from http://techqa.info/programming/question/28890268/parse-dimacs-cnf-file-python
        cnf = list()
        cnf.append(list())

        for line in content:
            tokens = line.split()
            if len(tokens) != 0 and tokens[0] not in ("p", "c"):
                for tok in tokens:
                    lit = int(tok)
                    if lit == 0:
                        cnf.append(list())
                    else:
                        cnf[-1].append(lit)

        # TODO remove after testing for performance
        assert len(cnf[-1]) == 0
        cnf.pop()

        return name2literal_dict, literal2name_dict, cnf


def str2bool(val):
    return val.lower() in ("true", "y", "yes", "t", "1")


class Solution:
    def __init__(self, model, start_features=None, start_decisions=None, start_literals=None):
        self.model = model

        self.features = np.zeros(self.model.num_features)
        self.features[0] = 1
        self.decisions = np.zeros(self.model.num_features)
        self.decisions[0] = 1
        if start_literals is not None:
            for literal in start_literals:
                self.process_literal(literal)
        else:
            if start_features is not None:
                self.features = start_features
            if start_decisions is not None:
                self.decisions = start_decisions

        self.cost = None
        self.has_changed_since_eval = True

    def to_literal_set(self):
        decision_indexes, = np.nonzero(self.decisions)
        literals = set()
        for i in decision_indexes:
            if i != 0:
                literal = i if self.features[i] > 0 else -1 * i
                literals.add(literal)
        return literals

    def process_literal(self, literal):
        index = abs(literal)
        self.features[index] = 1 if literal > 0 else 0
        self.decisions[index] = 1

    def __str__(self):
        return "Solution( " + str(self.get_fitness()) + " | " + str(self.features) + ")"
        # return "Solution( )"

    def __repr__(self):
        return "Solution( " + "complete" if self.is_complete() else "incomplete" + " )"

    def is_complete(self):
        return np.all(self.decisions)

    # @timeit
    def get_valid_literals(self, remaining_literals):
        config_literals = list(self.to_literal_set())
        # new_cnf_entries = list([x] for x in config_literals)
        # constraints = self.model.constraint_list + new_cnf_entries

        valid_literals = set()
        for lit in remaining_literals:
            if not self.model.violates_constraints(config_literals, lit):
                # let's get this party started!
                valid_literals.add(lit)

        return valid_literals

    def append(self, new_literal):
        self.process_literal(new_literal)
        self.has_changed_since_eval = True

    def get_fitness(self):
        return self.model.highest_costs_possible - self.get_cost()

    def get_cost(self):
        if self.cost is None or self.has_changed_since_eval:
            # self.fitness = 1
            self.has_changed_since_eval = False
            self.cost = self.compute_cost()
        return self.cost

    def __hash__(self):
        return hash(tuple(self.features)) ^ hash(tuple(self.decisions))

    def __eq__(self, other):
        return np.all(self.features == other.features) and np.all(self.decisions == other.decisions)

    # @timeit
    def compute_cost(self, objectives=None):
        objectives_to_return = range(self.model.num_objectives)

        result_costs_single = self.features.dot(self.model.costs_single_matrix)
        matches = self.features.dot(self.model.interactions)
        valid_interactions = self.model.expected_interaction_matches == matches
        result_costs_interactions = np.zeros(self.model.num_objectives)
        for i in objectives_to_return:
            result_costs_interactions[i] = valid_interactions[i, :].dot(self.model.influences_interactions[:, i])
        # result_costs_interactions = np.sum(valid_interactions.dot(self.model.influences_interactions))
        final_cost = result_costs_single + result_costs_interactions

        return final_cost


class ParetoFront:
    def __init__(self, model):
        self.global_pareto_front = set()
        self.model = model

    def pareto_dominates(self, sol1, sol2):
        dominates = False
        costs1 = sol1.get_cost()
        costs2 = sol2.get_cost()
        for o in range(self.model.num_objectives):
            if costs1[o] < costs2[o]:
                dominates = True
            elif costs2[o] < costs1[o]:
                return False
        return dominates

    def update_front(self, new_population):
        local_front = self.get_front(new_population)
        self.global_pareto_front = self.get_front(self.global_pareto_front.union(local_front))
        return local_front, self.global_pareto_front

    def get_front(self, pop):
        front = set()
        for sol_new in pop:
            front.add(sol_new)
            front_solutions = [x for x in front]
            for sol_existent in front_solutions:
                if self.pareto_dominates(sol_existent, sol_new):
                    front.remove(sol_new)
                    break
                elif self.pareto_dominates(sol_new, sol_existent):
                    front.remove(sol_existent)

        return front

        # def compute_pareto_front(self):
        # sort first objective
        # self.all_solutions = self.all_solutions[self.all_solutions[:, 0].argsort()]
        # print("All Solutions:", self.all_solutions)
        # # add first row to pareto_front
        # self.pareto_front = self.all_solutions[0:1, :]
        # print("Pareto Front:", self.pareto_front)
        # # test next row against the last row in pareto_front
        # for row in self.all_solutions[1:, :]:
        #     if sum([row[x] >= self.pareto_front[-1][x]
        #             for x in range(len(row))]) == len(row):
        #         # if it is better on all objectives add the row to pareto_front
        #         self.pareto_front = np.concatenate((self.pareto_front, [row]))

        # self.global_pareto_front.append(self.pareto_front)
        # if len(self.global_pareto_front) > len(population):
        #     del self.global_pareto_front[len(population):]

        # sort by cost of first objective
        # itemgetter = 0?
        # solutions = sorted(self.all_solutions, key=itemgetter(0))
        # add first row = objective?
        # self.pareto_front = solutions[0]
        # test next row against last row in pareto front -> why last?
        # for row in solutions[1]:
        #    # row, p or p, row
        #    if dominates(row, self.pareto_front)

        # return self.pareto_front

    # row, c or c, row
    def dominates(row, candidate_row):
        return sum([row[x] >= candidate_row[x] for x in range(len(row))]) == len(row)


class BruteForce:
    def __init__(self, model, visualizer):
        # init
        self.model = model
        # do we need to init these components?
        self.components = []
        for feature in model.features:
            # print(feature)
            component_off = Component(feature, 0, model)
            component_on = Component(feature, 1, model)
            self.components.append(component_off)
            self.components.append(component_on)
        self.max_run_time = 5  # in seconds
        self.visualizer = visualizer

    def find_best_solution(self):
        # main part
        best = None
        cnf_solutions = pycosat.itersolve(self.model.constraint_list)
        top_200 = []

        # calculate all solutions
        counter = 0
        for sol in cnf_solutions:
            solution = Solution(self.model)
            for number in sol:
                feature_name = next(key for key, value in self.model.name2variable.items() if value == abs(number))
                toggle = lambda x: (1, 0)[x < 0]
                new_component = Component(feature_name, toggle(number), self.model)
                solution.append(new_component)

            # append top 200 list
            sol_fitness = solution.get_fitness()
            top_200.append((sol_fitness, solution))
            top_200.sort(key=itemgetter(0))
            if len(top_200) > 200:
                top_200.pop()

            if not best or (sol_fitness < best.get_fitness()):
                best = solution
                self.visualizer.add_solution_forced(solution)
            else:
                self.visualizer.add_solution(solution)

        # save top 200 list to csv file
        header = ["Fitness"]
        for feature in self.model.name2variable:
            header.append(feature)

        plain_solutions = [sub_list[1] for sub_list in top_200]
        csv_list = []
        for sol in plain_solutions:
            mini_list = []
            mini_list.append(sol.fitness)
            for i in range(len(sol.components)):
                mini_list.append(sol.components[i].state)
            csv_list.append(mini_list)

        vm = os.path.splitext(os.path.basename(self.model.vm_path))[0]
        file = "brute_" + vm + ".csv"
        with open(file, "w") as csv_file:
            out = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
            out.writerows([header])
            out.writerows(csv_list)

        self.visualizer.visualize()
        return best


class ACS:
    @timeit
    def __init__(self, model, visualizer):
        # init
        self.model = model
        self.pop_size = 10
        self.evaporation_rate = 0.05
        self.pheromones_init = 0.5
        self.hill_climbing_its = 0
        self.elitist_select_prob = 0.95
        self.tuning_heuristic_selection = 1
        self.tuning_pheromone_selection = 3
        self.max_pheromone_step = 0.003
        self.visualizer = visualizer

        self.literals = []

        for var in range(1, self.model.num_features):
            # if var == self.model.name2variable["root"] or var not in constraints_by_variable:
            #    variable_constraints = []
            # else:
            #    variable_constraints = constraints_by_variable[var]
            literal_on = var
            literal_off = -1 * var
            self.literals.append(literal_on)
            self.literals.append(literal_off)

        self.elitist_learning_rate = self.estimate_elitist_learning_rate()
        self.start_literals, self.start_features, self.start_decisions = self.get_minimum_solution()
        self.pheromones = {}
        for literal in self.literals:
            self.pheromones[literal] = self.pheromones_init
        self.max_run_time = 5  # in seconds

    @timeit
    def get_minimum_solution(self):
        solution = Solution(self.model)
        valid_literals = solution.get_valid_literals(self.literals)

        solution_start_features = np.zeros(self.model.num_features, dtype=np.int8)
        solution_start_features[0] = 1
        solution_start_decisions = np.zeros(self.model.num_features, dtype=np.int8)
        solution_start_decisions[0] = 1

        global_start_literals = set(x for x in valid_literals)
        for i in valid_literals:
            if -1 * i not in valid_literals:
                var = abs(i)
                solution_start_features[var] = 1 if i > 0 else 0
                solution_start_decisions[var] = 1
                global_start_literals.remove(i)

        return global_start_literals, solution_start_features, solution_start_decisions

    def time_up(self, start, seconds=None):
        if not seconds:
            seconds = self.max_run_time
        return time.time() - start > seconds

    def find_best_solution(self, seconds=None):
        if not seconds:
            seconds = self.max_run_time
        # main part
        start = time.time()
        best = None
        top_30 = []
        # self.visualizer.add_sequence()

        pareto = ParetoFront(self.model)
        while not self.time_up(start, seconds):
            population = self.construct_population(seconds, start)

            for solution in population:
                # append top 30 list
                sol_cost = solution.get_cost()[0]
                top_30.append((sol_cost, solution))
                top_30.sort(key=itemgetter(0))
                if len(top_30) > 30:
                    top_30.pop()

                if self.model.num_objectives == 1:
                    if not best or (sol_cost < best.get_cost()[0]):
                        best = solution
                        self.visualizer.add_solution_forced(best.get_cost()[0])
                        self.visualizer.update_pheromone_graph_forced(self.model, self.literals,
                                                                      list(self.pheromones.values()))
                    else:
                        self.visualizer.add_solution(sol_cost)

            local_front, global_front = pareto.update_front(population)
            if self.model.num_objectives == 3:
                self.visualizer.update_pareto(population, local_front, global_front)

            self.update_pheromones(local_front)
            print("finished epoch")

        self.visualizer.visualize()
        while True:
            plt.pause(0.005)

        # save top 30 list to csv file
        # self.save_top_candidates_csv(top_30)

        return global_front

    def update_pheromones(self, pareto_front):
        best = np.random.choice(list(pareto_front), 1)[0]

        literals_of_best = best.to_literal_set()
        for literal in self.pheromones:
            p = self.pheromones[literal]
            self.pheromones[literal] = (
                                           1 - self.evaporation_rate) * p + self.evaporation_rate * self.pheromones_init
            if literal in literals_of_best:
                self.pheromones[literal] = (1 - self.elitist_learning_rate) * p \
                                           + self.elitist_learning_rate * best.get_fitness()[0]
        self.visualizer.update_pheromone_graph(self.model, self.literals, self.pheromones.values())

    def construct_population(self, seconds, start):
        population = []
        for n in range(self.pop_size):
            if self.time_up(start, seconds):
                break
            solution = self.construct_solution()
            solution = self.hill_climbing(solution)
            population.append(solution)
        return population

    # @timeit
    def construct_solution(self, possible_literals=None, solution=None):
        while solution is None or not solution.is_complete():
            if not possible_literals or not solution:
                # print("ended up with invalid solution; starting over")
                solution = Solution(self.model,
                                    start_features=np.array(list(x for x in self.start_features)),
                                    start_decisions=np.array(list(x for x in self.start_decisions)))
                possible_literals = [x for x in self.start_literals]

            new_literal = self.dummy_literal_selection(solution, possible_literals)
            if not self.model.violates_constraints(solution.to_literal_set(), new_literal):
                solution.append(new_literal)
                possible_literals.remove(-1 * new_literal)
            else:
                opposite_literal = -1 * new_literal
                if not self.model.violates_constraints(solution.to_literal_set(), opposite_literal):
                    solution.append(opposite_literal)
                    possible_literals.remove(opposite_literal)
                else:
                    solution = None
            possible_literals.remove(new_literal)
        return solution

    def save_top_candidates_csv(self, top_30):
        header = ["Fitness"]
        for feature in self.model.name2variable:
            header.append(feature)
        plain_solutions = [sub_list[1] for sub_list in top_30]
        csv_list = []
        for sol in plain_solutions:
            mini_list = []
            mini_list.append(sol.fitness)
            for i in range(len(sol.components)):
                mini_list.append(sol.components[i].state)
            csv_list.append(mini_list)
        vm = os.path.splitext(os.path.basename(self.model.vm_path))[0]
        file = "acs_" + vm + ".csv"
        with open(file, "w") as csv_file:
            out = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
            out.writerows([header])
            out.writerows(csv_list)

            # compare top_30 with top_200 from brute force
            # self.compare_lists(top_30, vm)

    # @timeit
    def elitist_literal_selection(self, solution, possible_literals):
        current_literals = solution.to_literal_set()
        cost_map = {}
        for literal in possible_literals:
            cost_delta = self.model.assess_cost_delta(current_literals, literal)
            cost_map[literal] = -1 * cost_delta

        min_val = min(list(cost_map.values()))
        fitness_map = {}
        for literal in cost_map:
            fitness_map[literal] = (cost_map[literal] - min_val)  # / (max_val - min_val)
        q = random.random()

        if q <= self.elitist_select_prob:
            # do elitist exploitation
            # TODO check min/max
            des_map = {}
            for literal in fitness_map:
                des_map[literal] = self.desirebility(literal, fitness_map[literal], pheromone_exp=1)
            best = max(des_map, key=des_map.get)
        else:
            # des_vec_function = np.vectorize(self.desirebility)
            # des = list(des_vec_function(np.array(list(fitness_map.keys())), np.array(list(fitness_map.values()))))
            des = []
            for literal in fitness_map:
                des_for_literal = self.desirebility(literal, fitness_map[literal])
                des.append(des_for_literal)
            sum_des = np.sum(des)
            desirebility_probabilities = np.array(des) / sum_des
            best = np.random.choice(list(possible_literals), p=desirebility_probabilities)
            # biased exploration
        return best

    def dummy_literal_selection(self, solution, possible_literals):
        q = random.random()
        if q <= self.elitist_select_prob:
            # do elitist exploitation
            best_pheromone = None
            best = None
            # best_literal = None
            for literal in possible_literals:
                p = self.pheromones[literal]
                if not best_pheromone or p > best_pheromone:
                    best_pheromone = p
                    best = literal
                    # best = max(self.pheromones, key=self.pheromones.get)
        else:
            # des_vec_function = np.vectorize(self.desirebility)
            # des = list(des_vec_function(np.array(list(fitness_map.keys())), np.array(list(fitness_map.values()))))
            best = np.random.choice(list(possible_literals))
            # biased exploration

        return best

    def desirebility(self, literal, value, pheromone_exp=None):
        if not pheromone_exp:
            pheromone_exp = self.tuning_pheromone_selection
        p = self.pheromones[literal]
        des = pow(p, pheromone_exp) * pow(value, self.tuning_heuristic_selection)
        return des

    def assess_fitness_complete(self, fitness_old, new_comp, solution):
        old_components = solution.components
        new_solution = Solution(self.model, old_components + [new_comp])
        fitness_new = new_solution.get_fitness()
        fitness_delta = fitness_new - fitness_old
        return fitness_delta

    def assess_fitness_only_one_feature(self, fitness_old, new_comp, old_solution):
        fitness_delta_isolated = old_solution.model.features[new_comp.feature]
        return fitness_delta_isolated

    def hill_climbing(self, solution):
        return solution

    # @timeit
    def estimate_elitist_learning_rate(self):
        cnf_solutions = pycosat.itersolve(self.model.constraint_list)
        cost_list = []
        # for sol in cnf_solutions:
        for i in range(20):
            solution = Solution(self.model)
            sol = np.array(next(cnf_solutions))
            solution = Solution(self.model, start_literals=sol)
            cost_list.append(solution.get_fitness())

        median = np.median(np.array(cost_list))
        # 0.0001 * 100 =!= x * median
        rate = self.max_pheromone_step / median
        return rate  # optimizer python module

    def compare_lists(self, top_30, vm):
        file_200 = "brute_" + vm + ".csv"
        top_200 = []

        with open(file_200, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                top_200.append(row)
        top_200.pop(0)  # remove header

        print()
        for item in top_30:
            if item in top_200:
                print("Equal item:", item)


def is_model(search_space):
    return match(search_space, FILE_MODEL_IDENTIFYER)


def is_model_xml(search_space):
    return match(search_space, FILE_MODEL_XML_IDENTIFYER)


def is_model_dimacs(search_space):
    return match(search_space, FILE_MODEL_DIMACS_IDENTIFYER)


def is_model_feature(search_space):
    return match(search_space, FILE_MODEL_FEATURE_IDENTIFYER)


def is_model_interactions(search_space):
    return match(search_space, FILE_MODEL_INTERATIONS_IDENTIFYER)


def match(search_space, pattern):
    pattern_compiled = re.compile(pattern)
    its_a_match = pattern_compiled.match(search_space)
    return its_a_match


def get_feature_name_mappings(content):
    pass


def get_disjunction_list(content, feature_name_mappings):
    pass


def read_file(file_name):
    with open(file_name) as f:
        content = f.readlines()
    return content


def read_features(file_model_feature_files):
    feature_costs_per_objective = []
    for i, file_model_feature in enumerate(file_model_feature_files):
        content = read_file(file_model_feature)
        feature_cost = {}
        for line in content:
            line = clean_line(line)
            feature_name, influence = line.split(":")
            feature_cost[feature_name] = float(influence)
        feature_costs_per_objective.append(feature_cost)

    return feature_costs_per_objective


def read_interactions(file_model_interaction_files):
    interaction_costs_per_objective = []
    for i, file_model_interactions in enumerate(file_model_interaction_files):
        feature_influences = {}
        content = read_file(file_model_interactions)
        for line in content:
            line = clean_line(line)
            features, influence = line.split(":")
            features_list = features.split("#")
            interaction = tuple(features_list)
            feature_influences[interaction] = float(influence)
        interaction_costs_per_objective.append(feature_influences)
    return interaction_costs_per_objective


def clean_line(line):
    line = line.replace(" ", "")
    line = line.replace("\n", "")
    return line


def create_dummy_config(model, min, max):
    dummy_config = []
    num_features = randint(min, max)
    for n in range(num_features):
        max_index = len(model.features)
        i = randint(0, max_index - 1)
        feature = list(model.features.keys())[i]
        dummy_config.append(feature)
    return dummy_config


def help_str():
    return "USAGE: Optimize model.xml model_feature.txt model_interactions.txt [brute] [visualize]"


def parse_args(argv):
    file_model = ""
    file_model_feature_files = []
    file_model_interations_files = []
    found_options = False
    do_brute_force = False
    do_visualization = False
    for (i, arg) in enumerate(argv):
        if arg != "Optimize":
            if is_model(arg):
                file_model = arg
            elif is_model_feature(arg):
                file_model_feature_files.append(arg)
            elif is_model_interactions(arg):
                file_model_interations_files.append(arg)
            elif arg == "brute":
                do_brute_force = True
            elif arg == "visualize":
                do_visualization = True
            else:
                print("Unsupported File Name: " + arg)

    if file_model and file_model_feature_files and file_model_interations_files:
        found_options = True

    return found_options, file_model, file_model_feature_files, \
           file_model_interations_files, do_brute_force, do_visualization


def main(argv):
    # first read terminal arguments
    found_options, file_model, file_model_feature_files, \
    file_model_interations_files, do_brute_force, do_visualization = parse_args(argv)

    if not found_options:
        print(help_str())
        sys.exit(EXIT_ARGUMENT_ERROR)

    print("Reading Files")
    features_per_objective = read_features(file_model_feature_files)
    interactions_per_objective = read_interactions(file_model_interations_files)

    model = Model(file_model, features_per_objective, interactions_per_objective)

    if not file_model or not features_per_objective or not interactions_per_objective:
        print(help_str())
        sys.exit(EXIT_FILE_ERROR)

    if do_visualization:
        visualizer = Visualizer(model)
    else:
        visualizer = DummyVisualizer()

    if do_brute_force:
        visualizer.set_sleep_time_costs(50)
        brute_force = BruteForce(model, visualizer)
        pareto_front = brute_force.find_best_solution()
    else:
        visualizer.set_sleep_time_costs(100)
        visualizer.set_sleep_time_pheromones(20)
        acs = ACS(model, visualizer)
        pareto_front = acs.find_best_solution(seconds=20)

    print("Pareto front: ")
    for solution in pareto_front:
        print(str(solution))

    return pareto_front


if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
