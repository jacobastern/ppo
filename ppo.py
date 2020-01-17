import argparse

# Local Imports
from ppo.train import train, train_cartpole, train_spaceinvaders_ram
from ppo.utils import save_spaceinvaders_demo

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--function', required=False, type=str, default='train',
        help="Whether to train a model or save a demo", choices=['train', 'demo'])
    parser.add_argument('--env_name', required=False, type=str, default='SpaceInvaders',
        help="The environment to train or demo", choices=['SpaceInvaders', 'CartPole'])
    parser.add_argument('--train_default', required=False, type=bool, default=True,
        help="Whether to train with default hyperparameters")
    
    # parser.add_argument('-e', '--n_epochs', required=False, type=int, default=5000, 
    #     help='The number of epochs to train for.')
    # parser.add_argument('-f', '--file_name', required=False, type=str, default='michael-jackson.txt', 
    #     help='The name of the file to train on.')
    
    # parser.add_argument('--print_every', required=False, type=int, default=500, 
    #                     help='How often to print an evaluation string.')
    # parser.add_argument('--plot_every', required=False, type=int, default=10, 
    #                     help='How often to track the training loss.')
    # parser.add_argument('--hidden_size', required=False, type=int, default=100, 
    #                     help='The hidden dimension of the recurrent unit.')
    # parser.add_argument('--n_layers', required=False, type=int, default=1, 
    #                     help='The number of layers in the model.')
    # parser.add_argument('--lr', required=False, type=int, default=.005, 
    #                     help='The learning rate for the model.')

    # print("Training {env_name} agent with hyperparameters as follows:",
    #     "N epochs:", args.n_epochs,
    #     ""
    #     , sep='\n')
        
    
    args = parser.parse_args()
        
    if args.function == 'demo':
        if args.env_name == "SpaceInvaders":
            save_spaceinvaders_demo()
        elif args.env_name == "CartPole":
            save_cartpole_demo()

    elif args.function == 'train':
        if args.train_default:
            if args.env_name == "SpaceInvaders":
                train_spaceinvaders_ram()
            elif args.env_name == "CartPole":
                train_cartpole()
        else:
            # TODO: train with given arguments
            pass
