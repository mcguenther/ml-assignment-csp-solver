import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import re
import time
import sys
import pycosat
import random
import xml.etree.ElementTree as ET
import numpy as np
from random import randint

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


class Component:
    def __init__(self, feature, state, pheromone=None):
        self.feature = feature
        # untoggled feature: state = 0
        # toggled feature: state = 1
        self.state = state
        self.pheromone = pheromone

    def to_literal(self, name_dict):
        literal = name_dict[self.feature]
        if not self.state:
            literal = literal * -1

        return literal

    def __str__(self):
        return "Component( " + self.feature + ", " + str(self.state) + ", " + str(self.pheromone) + " )"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.feature) ^ hash(self.state) ^ hash(self.pheromone)

    def __eq__(self, other):
        return self.feature == other.feature and self.state == other.state and self.pheromone == other.pheromone


class VM:
    def __init__(self, dimacs):
        pass


class Visualizer:
    def __init__(self, sleep_cycles=10):
        self.sequences = []
        plt.ion()
        self.fig = plt.figure()
        self.last_annotation = None
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Cost of best candidates over past epochs")
        self.ax.set_xlabel('#sample')
        self.ax.set_ylabel('cost of best solution')
        self.plot_data, = self.ax.plot(self.sequences)
        pylab.pause(1.e-8)
        self.sleep_cycles = sleep_cycles
        self.sleep_state = 1

    def add_sequence(self):
        self.sequences = []

    def add_solution(self, solution):
        if self.sleep_state % (self.sleep_cycles + 1) == 0:
            self.add_solution_forced(solution)
            self.sleep_state = 1
        else:
            self.sleep_state += 1

    def add_solution_forced(self, solution):
        if not self.sequences:
            self.add_sequence()
        self.sequences.append(solution.get_fitness())
        self.update_cost_graph()

    def update_cost_graph(self):
        self.plot_data.set_data(np.arange(len(self.sequences)), self.sequences)
        if self.last_annotation:
            self.last_annotation.remove()
        xy = (len(self.sequences) - 2, self.sequences[-1] + 1)
        self.last_annotation = self.ax.annotate(self.sequences[-1], xy=xy, textcoords='data')
        self.ax.plot(self.sequences, "b-")
        pylab.pause(1.e-8)

    def visualize(self):
        costs = self.sequences
        print("Best: " + str(min(costs)) + " | Worst: " +
              str(max(costs)) + " | Avg: " + str(sum(costs) / len(costs)))
        self.update_cost_graph()
        plt.show(block=True)

    def set_sleep_time(self, x):
        self.sleep_cycles = x


class DummyVisualizer:
    def __init__(self):
        pass

    def add_sequence(self):
        pass

    def add_solution(self, solution):
        pass

    def add_solution_forced(self, solution):
        pass

    def visualize(self):
        pass

    def set_sleep_time(self, x):
        pass


class Model:
    def __init__(self, vm_path, features, interactions):
        self.features = features
        self.interactions = interactions

        if is_model_dimacs(vm_path):
            self.name_dict, self.constraint_list = self.read_vm_dimacs(vm_path)
        else:
            self.name_dict, self.constraint_list = self.read_vm_xml(vm_path)

    # @timeit
    def read_vm_xml(self, file_model):
        print("it's an XML!")
        # content = read_file(file_model)
        # tree = ET.fromstring(content)

        tree = ET.parse(file_model)
        root = tree.getroot()
        name2literal = {}
        name2excluded = {}
        name2is_optional = {}
        parent2children = {}

        cnf = []
        for option in root.iter('configurationOption'):
            key = option.find('name').text
            is_optional = str2bool(option.find('optional').text)
            new_literal = len(name2literal) + 1
            name2literal[key] = new_literal
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

        return name2literal, cnf

    @timeit
    def read_vm_dimacs(self, file_model):
        content = read_file(file_model)
        d = {}
        for line in content:
            if line.startswith("c"):
                line = line.split()
                if len(line) == 3:
                    line = line[1:]
                    (val, key) = line
                    val = val.replace("$", "")
                    val = int(val)
                    d[key] = val

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

        return d, cnf


def str2bool(val):
    return val.lower() in ("true", "y", "yes", "t", "1")


class Solution:
    def __init__(self, model, components=None):
        self.model = model
        if components is None:
            self.components = []
        else:
            self.components = components
        self.fitness = None
        self.has_changed_since_eval = True

    def __str__(self):
        return "Solution( " + str(self.get_fitness()) + " | " + str(self.components) + ")"

    def __repr__(self):
        return self.__str__()

    def is_complete(self):
        all_features = set(self.model.features)
        all_features.remove("root")
        current_features = set(comp.feature for comp in self.components)
        is_complete = all_features == current_features
        return is_complete

    # @timeit
    def get_valid_components(self, global_components):
        """
        to slow!
        """
        valid_components = []
        all_features = set(self.model.features)
        ok_features = set((comp.feature for comp in self.components))
        rest_features = all_features - ok_features
        rest_features.remove("root")
        rest_components = set()
        for comp in global_components:
            if comp.feature in rest_features:
                rest_components.add(comp)

        config_literals = list(([comp.to_literal(self.model.name_dict)] for comp in self.components))
        constraints = self.model.constraint_list + config_literals

        for comp in rest_components:
            tmp_literal = comp.to_literal(self.model.name_dict)
            constraints.append([tmp_literal])
            # showtime
            result = pycosat.solve(constraints)
            constraints.pop()
            if result in ("UNSAT", "UNKNOWN"):
                continue
            # let's get this party started!
            valid_components.append(comp)

        return valid_components

    def clear(self):
        self.components = []
        self.has_changed_since_eval = True
        self.fitness = None

    def append(self, new_component):
        self.components.append(new_component)
        self.has_changed_since_eval = True

    def get_fitness(self):
        if not self.fitness or self.has_changed_since_eval:
            self.fitness = self.assess_fitness()
            self.has_changed_since_eval = False

        return self.fitness

    # @timeit
    def assess_fitness(self):
        """
        assesses fitness
        to slow!
        """
        fitness = self.model.features["root"]

        for component in self.components:
            if component.state:
                feature = component.feature
                value = self.model.features[feature]
                fitness += value

        for interaction_features in self.model.interactions:
            matches = 0
            for comp in self.components:
                if comp.feature in interaction_features and comp.state == 1:
                    matches += 1

            if matches == len(interaction_features):
                value = self.model.interactions[interaction_features]
                fitness += value

        return fitness


class BruteForce:
    def __init__(self, model, visualizer):
        # init
        self.model = model
        # do we need to init these components?
        self.components = []
        for feature in model.features:
            print(feature)
            component_off = Component(feature, 0)
            component_on = Component(feature, 1)
            self.components.append(component_off)
            self.components.append(component_on)
        self.max_run_time = 5  # in seconds
        self.visualizer = visualizer

    def find_best_solution(self):
        # main part
        best = None
        print()
        cnf_solutions = pycosat.itersolve(self.model.constraint_list)

        counter = 0
        for sol in cnf_solutions:
            solution = Solution(self.model)
            print("Init solution:", solution.get_fitness())
            for number in sol:
                print("Number:", number)
                feature_name = next(key for key, value in self.model.name_dict.items() if value == abs(number))
                toggle = lambda x: (1, 0)[x < 0]
                new_component = Component(feature_name, toggle(number))
                print(new_component)
                solution.append(new_component)
            counter += 1
            if not best or (solution.get_fitness() < best.get_fitness()):
                best = solution
                self.visualizer.add_solution_forced(solution)
            else:
                self.visualizer.add_solution(solution)
            print("Solution components:", solution.components)
            print("Solution fitness:", solution.get_fitness())
            print()

        print(counter, "solutions")
        print("Best fitness:", best.get_fitness())
        self.visualizer.visualize()
        return best


class ACS:
    def __init__(self, model, visualizer):
        # init
        self.model = model
        self.pop_size = 20
        self.elitist_learning_rate = 0.00006
        self.evaporation_rate = 0.1
        self.pheromones_init = 0.5
        # TODO: check parameters for component selection
        self.hill_climbing_its = 0
        self.elitist_select_prob = 0.5
        self.tuning_heuristic_selection = 1
        self.tuning_pheromone_selection = 1
        self.visualizer = visualizer

        # from the 1997 paper reflecting the impact of the fitness
        self.beta = 2

        self.components = []
        for feature in model.features:
            component_off = Component(feature, 0, self.pheromones_init)
            component_on = Component(feature, 1, self.pheromones_init)
            self.components.append(component_off)
            self.components.append(component_on)

        self.max_run_time = 10  # in seconds

    def time_up(self, start, seconds=None):
        if not seconds:
            seconds = self.max_run_time
        return time.time() - start > seconds

    def find_best_solution(self, seconds=None):
        if not seconds:
            seconds = self.max_run_time
        # main part
        best = None
        start = time.time()
        self.visualizer.add_sequence()
        while not self.time_up(start, seconds):
            for n in range(self.pop_size):
                if self.time_up(start, seconds):
                    break
                solution = Solution(self.model)
                while not solution.is_complete():
                    component_selection = solution.get_valid_components(self.components)
                    if not component_selection:
                        print("ended up with invalid solution; starting over")
                        solution.clear()
                        continue
                    else:
                        new_component = self.elitist_component_selection(solution, component_selection)
                        solution.append(new_component)
                # print("found a valid solution!")
                solution = self.hill_climbing(solution)

                if not best or (solution.get_fitness() < best.get_fitness()):
                    best = solution
                    self.visualizer.add_solution_forced(solution)
                else:
                    self.visualizer.add_solution(solution)

            for component in self.components:
                component.pheromone = (1 - self.evaporation_rate) * component.pheromone \
                                      + self.evaporation_rate * self.pheromones_init
                if component in set(best.components):
                    # TODO: check if formula for elitist pheromone increase is valid
                    component.pheromone = (1 - self.elitist_learning_rate) * component.pheromone \
                                          + self.elitist_learning_rate * best.get_fitness()
        self.visualizer.visualize()
        return best

    # @timeit
    def elitist_component_selection(self, solution, component_selection):
        """
        too slow!
        """
        fitness_old = solution.get_fitness()
        fitness_map = {}
        for new_comp in component_selection:
            # fitness_delta = self.assess_fitness_complete(fitness_old, new_comp, old_components)
            fitness_delta = self.assess_fitness_complete(fitness_old, new_comp, solution)
            if fitness_delta == 0:
                score = sys.maxsize
            else:
                score = new_comp.pheromone * pow(1 / fitness_delta, self.beta)
            fitness_map[new_comp] = score

        q = random.random()
        # q = 0.0
        if q <= self.elitist_select_prob:
            # do elitist exploitation
            # TODO check min/max
            best = max(fitness_map, key=fitness_map.get)
        else:
            des_map = {}
            for new_comp in component_selection:
                des_map[new_comp] = self.desirebility(new_comp, fitness_map)
            best_list = sorted(des_map, key=fitness_map.get)
            index = np.random.geometric(p=0.4)
            index = min(index, (len(best_list) - 1))
            best = best_list[index]
            # biased exploration

        return best

    def desirebility(self, component, val_map):
        p = component.pheromone
        # TODO define value of component

        des = pow(p, self.tuning_pheromone_selection)
        if val_map:
            val = val_map[component]  # component.value
            des = des * pow(val, self.tuning_heuristic_selection)
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


# optimizer python module




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


def read_features(file_model_feature):
    content = read_file(file_model_feature)
    feature_influences = {}
    for line in content:
        line = clean_line(line)
        feature_name, influence = line.split(":")
        feature_influences[feature_name] = float(influence)

    return feature_influences


def read_interactions(file_model_interactions):
    feature_influences = {}
    content = read_file(file_model_interactions)
    for line in content:
        line = clean_line(line)
        features, influence = line.split(":")
        features_list = features.split("#")
        interaction = tuple(features_list)
        feature_influences[interaction] = float(influence)
    return feature_influences


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
    return "USAGE: Optimize model.xml model_feature.txt model_interactions.txt"


def parse_args(argv):
    file_model = ""
    file_model_feature = ""
    file_model_interations = ""
    found_options = False
    do_brute_force = False
    do_visualization = False
    for (i, arg) in enumerate(argv):
        if arg != "Optimize":
            if is_model(arg):
                file_model = arg
            elif is_model_feature(arg):
                file_model_feature = arg
            elif is_model_interactions(arg):
                file_model_interations = arg
            elif arg == "brute":
                do_brute_force = True
            elif arg == "visualize":
                do_visualization = True
            else:
                print("Unsupported File Name: " + arg)

    if file_model and file_model_feature and file_model_interations:
        found_options = True

    return found_options, file_model, file_model_feature, \
           file_model_interations, do_brute_force, do_visualization


def main(argv):
    # first read terminal arguments
    found_options, file_model, file_model_feature, \
    file_model_interations, do_brute_force, do_visualization = parse_args(argv)

    if not found_options:
        print(help_str())
        sys.exit(EXIT_ARGUMENT_ERROR)

    print("Reading Files")
    features = read_features(file_model_feature)
    interactions = read_interactions(file_model_interations)
    print(interactions)

    model = Model(file_model, features, interactions)

    if not file_model or not features or not interactions:
        print(help_str())
        sys.exit(EXIT_FILE_ERROR)

    if do_visualization:
        visualizer = Visualizer()
    else:
        visualizer = DummyVisualizer()

    if do_brute_force:
        visualizer.set_sleep_time(50)
        brute_force = BruteForce(model, visualizer)
        optimum = brute_force.find_best_solution()
    else:
        visualizer.set_sleep_time(50)
        acs = ACS(model, visualizer)
        optimum = acs.find_best_solution(seconds=60)

    print("Optimum:", optimum)
    return optimum


if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
