import argparse

# Local Imports
from ppo.train import train_cartpole, train_spaceinvaders_ram
from ppo.utils import save_spaceinvaders_demo

# TODO: Make appropriate docker file
# TODO: Get it working in separate file format
# TODO: Add command line interface
# TODO: Add a readme

if __name__ == "__main__":
    # train_spaceinvaders_ram()
    save_spaceinvaders_demo()