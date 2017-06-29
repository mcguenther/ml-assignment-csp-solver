import re
import sys

# optimizer python module
EXIT_ARGUMENT_ERROR = 2
EXIT_FILE_ERROR = 3

FILE_MODEL_IDENTIFYER = ".*(\.dimacs|\.xml)"
FILE_MODEL_DIMACS_IDENTIFYER = ".*(\.dimacs)"
FILE_MODEL_XML_IDENTIFYER = ".*(\.xml)"
FILE_MODEL_FEATURE_IDENTIFYER = ".*feature([1-9]*).txt"
FILE_MODEL_INTERATIONS_IDENTIFYER = ".*interactions([1-9]*).txt"


def is_model(seach_space):
    return match(seach_space, FILE_MODEL_IDENTIFYER)


def is_model_xml(seach_space):
    return match(seach_space, FILE_MODEL_XML_IDENTIFYER)


def is_model_dimacs(seach_space):
    return match(seach_space, FILE_MODEL_DIMACS_IDENTIFYER)


def is_model_feature(seach_space):
    return match(seach_space, FILE_MODEL_FEATURE_IDENTIFYER)


def is_model_interactions(seach_space):
    return match(seach_space, FILE_MODEL_INTERATIONS_IDENTIFYER)


def match(seach_space, pattern):
    pattern_compiled = re.compile(pattern)
    its_a_match = pattern_compiled.match(seach_space)
    return its_a_match


def main(argv):
    # base_dir = os.path.abspath(os.getcwd())
    found_options = False

    # first read terminal arguments
    found_options, file_model, file_model_feature, file_model_interations \
        = parse_args(argv)

    if not found_options:
        print(help_str())
        sys.exit(EXIT_ARGUMENT_ERROR)

    print("Reading Files")
    if is_model_dimacs(file_model):
        vm = read_vm_dimacs(file_model)
    else:
        vm = read_vm_xml(file_model)

    features = read_features(file_model_feature)
    interactions = read_interactions(file_model_interations)
    print(interactions)

    if not vm or not features or not interactions:
        print(help_str())
        sys.exit(EXIT_FILE_ERROR)

    optimum = acs(vm, features, interactions)
    print(optimum)
    return optimum


def read_vm_dimacs(file_model):
    pass


def read_vm_xml(file_model):
    pass


def read_features(file_model_feature):
    with open(file_model_feature) as f:
        content = f.readlines()
        feature_influences = {}
        for line in content:
            line = line.replace(" ", "")
            line = line.replace("\n", "")
            feature_name, influence = line.split(":")
            feature_influences[feature_name] = influence

    return feature_influences


def read_interactions(file_model_interations):
    feature_influences = {}
    with open(file_model_interations) as f:
        content = f.readlines()

        for line in content:
            line = line.replace(" ", "")
            line = line.replace("\n", "")
            features, influence = line.split(":")
            features_list = features.split("#")
            interaction = Interaction(list(features_list))
            feature_influences[interaction] = influence
    return feature_influences


def acs(vm, features, interactions):
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


class Interaction:
    def __init__(self, arg):
        arg.sort()
        self.features = arg

    def __str__(self):
        return str(self.features) 

    def __unicode__(self):
        return self.__str__()

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    main(sys.argv[1:])
