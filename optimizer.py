import re
import time
import sys
from random import randint

EXIT_ARGUMENT_ERROR = 2

EXIT_FILE_ERROR = 3

FILE_MODEL_IDENTIFYER = ".*(\.dimacs|\.xml)"

FILE_MODEL_DIMACS_IDENTIFYER = ".*(\.dimacs)"

FILE_MODEL_XML_IDENTIFYER = ".*(\.xml)"

FILE_MODEL_FEATURE_IDENTIFYER = ".*feature([0-9]*).txt"

FILE_MODEL_INTERATIONS_IDENTIFYER = ".*interactions([0-9]*).txt"


class Component:
    def __init__(self, feature, state, pheromone):
        self.feature = feature
        self.state = state
        self.pheromone = pheromone


class VM:
    def __init__(self, dimacs):
        pass


class Model:
    def __init__(self, vm_path, features, interactions):
        self.features = features
        self.interactions = interactions

        if is_model_dimacs(vm_path):
            self.vm = self.read_vm_dimacs(vm_path)
        else:
            self.vm = self.read_vm_xml(vm_path)

    def read_vm_xml(self, file_model):
        return True

    def read_vm_dimacs(self, file_model):
        content = read_file(file_model)
        # feature_name_mappings = get_feature_name_mappings(content)
        # disjunctions = get_disjunction_list(content, feature_name_mappings)
        d = {}
        constraints = {}
        for line in content:
            if line.startswith("c"):
                line = line.split(" ")
                if len(line) == 3:
                    line = line[1:]
                    (val, key) = line
                    d[key] = val
            elif not line.startswith("p"):
                feature_ids = line.split(" ")
                feature_ids = feature_ids[:-1]  # remove trailing 0
                feature_signs = {}
                for f_id in feature_ids:
                    f_id = int(f_id)
                    sign = 1
                    if str(f_id).startswith("-"):
                        sign = -1
                        f_id = abs(f_id)
                    f_name = d[str(f_id)]
                    feature_signs[f_name] = sign

                # lambda clause to formulate constraint fo scp solver
                # they take a list of features, apply OR to them and should return True to be valid
                lambda_expr = lambda *x: self.apply_or_to_tuple(x) is True
                constraints[lambda_expr] = feature_signs

        return constraints

    def apply_or_to_tuple(self, or_tuple):
        # or_val = False
        # for entry in or_tuple:
        # or_val = or_val or entry
        # return or_val
        for entry in or_tuple:
            if entry is True:
                return True

        return False


class Solution:
    def __init__(self, model):
        self.model = model
        self.components = []
        self.fitness = None

    def is_complete(self):
        for feature in self.model.features:
            feature_gen = (comp.feature for comp in self.components)
            if feature not in set(feature_gen):
                return False

        return True

    def get_valid_components(self):
        pass

    def clear(self):
        self.components = []
        self.fitness = None

    def append(self, new_component):
        self.components.append(new_component)

    def get_fitness(self):
        if not self.fitness:
            self.fitness = self.assess_fitness()

        return self.fitness

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
        self.elitist_learning_rate = 0.001
        self.evaporation_rate = 0.1
        self.pheromones_init = 0.5
        # TODO: check parameters for component selection
        self.hill_climbing_its = 0
        self.elitist_select_prob = 0.5
        self.components = []
        for feature in model.features:
            component_off = Component(feature, 0, self.pheromones_init)
            component_on = Component(feature, 1, self.pheromones_init)
            self.components.append(component_off)
            self.components.append(component_on)

        self.max_run_time = 30  # in seconds

    def find_best_solution(self):
        # main part
        best = None
        start = time.time()
        while time.time() - start < self.max_run_time:
            population = []
            for n in range(self.pop_size):
                solution = Solution(self.model)
                while not solution.is_complete():
                    component_selection = solution.get_valid_components()
                    if not component_selection:
                        solution.clear()
                        continue
                    else:
                        new_component = self.elitist_component_selection(component_selection)
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

    def elitist_component_selection(self, component_selection):
        pass

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
