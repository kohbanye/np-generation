import os
from np_generation.generation import Generator

ROOT_DIR = os.environ["ROOT_DIR"]


def main():
    model_dir = os.path.join(ROOT_DIR, "model")

    generator = Generator(model_dir)
    print(generator.batch_generate(3))


if __name__ == "__main__":
    main()
