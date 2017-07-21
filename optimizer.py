import re
import time
import sys
import pycosat
import random

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
        print('%r (%r, %r) %2.2f sec' % \
              (method.__name__, args, kw, end - start))
        return result

    return timed


class Component:
    def __init__(self, feature, state, pheromone):
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


class VM:
    def __init__(self, dimacs):
        pass


class Model:
    def __init__(self, vm_path, features, interactions):
        self.features = features
        self.interactions = interactions

        if is_model_dimacs(vm_path):
            self.name_dict, self.constraint_list = self.read_vm_dimacs(vm_path)
        else:
            self.vm = self.read_vm_xml(vm_path)

    def read_vm_xml(self, file_model):
        return True

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


class Solution:
    def __init__(self, model, components=None):
        self.model = model
        if components is None:
            self.components = []
        else:
            self.components = components
        self.fitness = None
        self.has_changed_since_eval = True

    def is_complete(self):
        for feature in self.model.features:
            feature_gen = (comp.feature for comp in self.components)
            if feature not in set(feature_gen):
                return False

        return True

    @timeit
    def get_valid_components(self, pheromones_init):
        valid_components = []
        all_features = set(self.model.features)
        ok_features = set((comp.feature for comp in self.components))
        rest_features = all_features - ok_features
        rest_features.remove("root")

        config_literals = list(([comp.to_literal(self.model.name_dict)] for comp in self.components))
        constraints = self.model.constraint_list + config_literals

        for test_feature in rest_features:
            for toggle in (0, 1):
                tmp_component = Component(test_feature, toggle, pheromones_init)
                tmp_literal = tmp_component.to_literal(self.model.name_dict)
                constraints.append([tmp_literal])

                # show time
                result = pycosat.solve(constraints)
                constraints.pop()
                if result in ("UNSAT", "UNKNOWN"):
                    continue

                # let's get this party started!
                valid_components.append(tmp_component)

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

    @timeit
    def assess_fitness(self):
        fitness = self.model.features["root"]

        for component in self.components:
            feature = component.feature
            value = self.model.features[feature]
            fitness += value

        for interaction_features in self.model.interactions:
            feature_generator = (comp.feature for comp in self.components)
            if set(interaction_features).issubset(set(feature_generator)):
                value = self.model.interactions[interaction_features]
                fitness += value

        return fitness


class ACS:
    def __init__(self, model):
        # init
        self.model = model
        self.pop_size = 5
        self.elitist_learning_rate = 0.1
        self.evaporation_rate = 0.1
        self.pheromones_init = 0.5
        # TODO: check parameters for component selection
        self.hill_climbing_its = 0
        self.elitist_select_prob = 0.5

        # from the 1997 paper reflecting the impact of the fitness
        self.beta = 2

        self.components = []
        for feature in model.features:
            component_off = Component(feature, 0, self.pheromones_init)
            component_on = Component(feature, 1, self.pheromones_init)
            self.components.append(component_off)
            self.components.append(component_on)

        self.max_run_time = 10  # in seconds

    def find_best_solution(self):
        # main part
        best = None
        start = time.time()
        while time.time() - start < self.max_run_time:
            population = []
            for n in range(self.pop_size):
                solution = Solution(self.model)
                while not solution.is_complete():
                    component_selection = solution.get_valid_components(self.pheromones_init)
                    if not component_selection:
                        print("ended up with invalid solution; starting over")
                        solution.clear()
                        continue
                    else:
                        new_component = self.elitist_component_selection(solution, component_selection)
                        solution.append(new_component)
                solution = self.hill_climbing(solution)

                if not best or (solution.get_fitness() > best.get_fitness()):
                    best = solution
            for component in self.components:
                component.pheromone = (1 - self.evaporation_rate) * component.pheromone \
                                      + self.evaporation_rate * self.pheromones_init
                if set(component).issubset(best.components):
                    # TODO: check if formula for elitist pheromone increase is valid
                    component.pheromone = (1 - self.elitist_learning_rate) * component.pheromone \
                                          + self.elitist_learning_rate * best.get_fitness()
        return best

    @timeit
    def elitist_component_selection(self, solution, component_selection):
        fitness_old = solution.get_fitness()
        old_components = solution.components
        fitness_map = {}
        best = None
        for new_comp in component_selection:
            # fitness_delta = self.assess_fitness_complete(fitness_old, new_comp, old_components)
            fitness_delta = self.assess_fitness_complete(fitness_old, new_comp, solution)
            score = new_comp.pheromone * pow(1 / fitness_delta, self.beta)
            fitness_map[new_comp] = score

        q = random.random()
        q = 0.0
        if q <= self.elitist_select_prob:
            # do elitist exploitation
            best = min(fitness_map, key=fitness_map.get)
        else:
            pass
        # biased exploration

        return best

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
    for (i, arg) in enumerate(argv):
        if arg != "Optimize":
            if is_model(arg):
                file_model = arg
            elif is_model_feature(arg):
                file_model_feature = arg
            elif is_model_interactions(arg):
                file_model_interations = arg
            else:
                print("Unsupported File Name: " + arg)

    if file_model and file_model_feature and file_model_interations:
        found_options = True

    return found_options, file_model, file_model_feature, \
           file_model_interations


def main(argv):
    # first read terminal arguments
    found_options, file_model, file_model_feature, file_model_interations \
        = parse_args(argv)

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

    acs = ACS(model)
    optimum = acs.find_best_solution()
    print(optimum)
    return optimum


if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
