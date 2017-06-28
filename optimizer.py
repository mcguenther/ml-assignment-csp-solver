import re
import sys

# optimizer python module
EXIT_ARGUMENT_ERROR = 2
EXIT_FILE_ERROR = 3

FILE_MODEL_IDENTIFYER = "model([1-9]*).xml"
FILE_MODEL_FEATURE_IDENTIFYER = "model_feature([1-9]*).txt"
FILE_MODEL_INTERATIONS_IDENTIFYER = "model_interactions([1-9]*).txt"


def is_model(seach_space):
    return match(seach_space, FILE_MODEL_IDENTIFYER)


def is_model_feature(seach_space):
    return match(seach_space, FILE_MODEL_FEATURE_IDENTIFYER)


def is_model_interactions(seach_space):
    return match(seach_space, FILE_MODEL_INTERATIONS_IDENTIFYER)


def match(seach_space, pattern):
    pattern = re.compile(pattern)
    return pattern.match(seach_space)


def main(argv):
    # base_dir = os.path.abspath(os.getcwd())
    found_options = False

    # first read terminal arguments
    found_options, file_model, file_model_feature, file_model_interations = parse_args(
        argv)

    if not found_options:
        print(help_str())
        sys.exit(EXIT_ARGUMENT_ERROR)

    vm = read_vm(file_model)
    features = read_features(file_model_feature)
    interactions = read_interactions(file_model_interations)

    if not vm or not features or not interactions:
        print(help_str())
        sys.exit(EXIT_FILE_ERROR)

    optimum = acs(vm, features, interactions)
    print(optimum)
    return optimum


def read_vm(file_model):
    pass


def read_features(file_model):
    pass


def read_interactions(file_model):
    pass


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

    if file_model and file_model_feature and file_model_interations:
        found_options = True

    return found_options, file_model, file_model_feature, file_model_interations


if __name__ == "__main__":
    main(sys.argv[1:])
