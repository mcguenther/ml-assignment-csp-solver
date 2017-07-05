import re
import sys
from random import randint

# optimizer python module
EXIT_ARGUMENT_ERROR = 2
EXIT_FILE_ERROR = 3

FILE_MODEL_IDENTIFYER = ".*(\.dimacs|\.xml)"
FILE_MODEL_DIMACS_IDENTIFYER = ".*(\.dimacs)"
FILE_MODEL_XML_IDENTIFYER = ".*(\.xml)"
FILE_MODEL_FEATURE_IDENTIFYER = ".*feature([0-9]*).txt"
FILE_MODEL_INTERATIONS_IDENTIFYER = ".*interactions([0-9]*).txt"


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


class Model:
    def __init__(self, vm, features, interactions):
        self.vm = vm
        self.features = features
        self.interactions = interactions

    def assess_fitness(self, dummy_config):
        fitness = self.features["root"]

        for feature in dummy_config:
            value = self.features[feature]
            fitness += value

        for interaction_features in self.interactions:
            if set(interaction_features).issubset(set(dummy_config)):
                value = self.interactions[interaction_features]
                fitness += value

        return fitness


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
    if is_model_dimacs(file_model):
        vm = read_vm_dimacs(file_model)
    else:
        vm = read_vm_xml(file_model)

    model = Model(vm, features, interactions)

    if not vm or not features or not interactions:
        print(help_str())
        sys.exit(EXIT_FILE_ERROR)

    optimum = acs(model)
    print(optimum)
    return optimum


def get_feature_name_mappings(content):
    pass


def get_disjunction_list(content, feature_name_mappings):
    pass


def read_vm_dimacs(file_model):
    content = read_file(file_model)
    feature_name_mappings = get_feature_name_mappings(content)
    disjunctions = get_disjunction_list(content, feature_name_mappings)
    # for line in content:
    #    if line.startswith("c"):
    #        if
    #        continue
    return True


def read_vm_xml(file_model):
    return True


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


def acs(model):
    dummy_config = []
    num_features = randint(30, 50)
    for n in range(num_features):
        max_index = len(model.features)
        i = randint(0, max_index - 1)
        feature = list(model.features.keys())[i]
        dummy_config.append(feature)

    print(dummy_config)
    fitness = model.assess_fitness(dummy_config)
    print(fitness)

    pass


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


class VmXML:
    def __init__(self, xml):
        pass


class VmDimacs:
    def __init__(self, dimacs):
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
