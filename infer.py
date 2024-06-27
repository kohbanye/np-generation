import os
from pprint import pprint
from np_generation.generation import Generator

ROOT_DIR = os.environ["ROOT_DIR"]


def main():
    model_dir = os.path.join(ROOT_DIR, "model", "no_chiral")

    generator = Generator(model_dir)
    pprint(generator.batch_generate(10))


if __name__ == "__main__":
    main()
