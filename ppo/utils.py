################################################################
# utils.py - utility functions for PPO algorithm, training, and 
#   Atari environments
# 
# calc_true_returns()
#   Calculates the discounted sum of rewards at each state and 
#   updates rollouts.
# generate_rollouts()
#   Generates experience rollouts to use for training
# pad_image()
#    Pads an image with zeros so it is square
# save_checkpoint()
#    Saves a checkpoint of the current model parameters
################################################################

import torch
import gym
import shutil
import numpy as np
from threading import Lock
from PIL import Image

# Local imports
from .model import FCNetwork, PolicyNetwork

def calc_true_returns(rollouts, gamma):
    """Calculates the discounted sum of rewards at each state and updates rollouts.
        Also returns a list of the total rewards for each rollout.
    Args:
        rollouts (list) : a list of rollouts
        gamma (float) : the discount factor
    """
    updated_rollouts = []
    rewards = []
    for rollout in rollouts:
        updated_episode = []
        discounted_sum_of_rewards = 0
        rollout_reward = 0
        for step in rollout[::-1]:
            reward = step[3]
            # Update the discounted rewards and the full reward
            discounted_sum_of_rewards = reward + discounted_sum_of_rewards * gamma
            rollout_reward += 1
            # Update this step with discounted reward
            updated_step = step + (discounted_sum_of_rewards,)
            updated_episode.append(updated_step)
        # We now have rollouts updated to include discounted reward
        updated_rollouts.append(updated_episode)
        # Also, a list of the total rewards for each rollout.
        rewards.append(rollout_reward)
        
    return updated_rollouts, rewards
            
def generate_rollouts(env, policy_net, rollouts, rewards, episodes_per_epoch, episode_length, gamma, device, pad_img, feature_net, thread_id):
    """Generates experience rollouts to use for training
    Args:
        env (openai gym environment): the environment we will train on
        policy_net (PolicyNetwork): a network that takes in state and outputs actions
        episodes_per_epoch (int): the number of episodes that we train on each epoch
        episode_length (int): the number of frames to run each episode
        gamma (float): the discount factor for distant rewards
    """
    cnt = 0
    thread_rollouts = []
    for i in range(episodes_per_epoch):            
        state = env.reset()
        episode = []
        for step in range(episode_length):
            input_state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            if pad_img:
                input_state = pad_image(input_state).to(device)
            if feature_net:
                input_state = feature_net(input_state)
            # Select an action
            action_dist = policy_net(input_state)
            action_dist = action_dist[0]
            action = np.random.choice(np.arange(len(action_dist)), p=action_dist.cpu().detach().numpy())
            next_state, reward, terminated, _ = env.step(action)
            # Add this experience to memory
            episode.append((state, action_dist.cpu().detach().numpy(), np.array([action]), reward))
            if terminated:
                break
            state = next_state
        thread_rollouts.append(episode)
    # Update rollouts to include partially discounted rewards, get reward for each rollout
    thread_rollouts, thread_rewards = calc_true_returns(thread_rollouts, gamma)
    lock = Lock()
    # Lock these objects to avoid race condition with other workers
    with lock:
        rollouts.extend(thread_rollouts)
        rewards.extend(thread_rewards)

def pad_image(img):
    """Pads an image with zeros so it is square"""
    max_dim = max(img.shape[2], img.shape[3])
    padded_img = torch.zeros((img.shape[0], img.shape[1], max_dim, max_dim))
    padded_img[:, :, :img.shape[2], :img.shape[3]] = img
    return padded_img

def save_checkpoint(state, is_best, env_name):
    ckpt_file = os.path.join("saved_models", "{}.pth.tar".format(env))
    best_ckpt_file = os.path.join("saved_models", "{}_best.pth.tar".format(env))
    torch.save(state, ckpt_file)
    if is_best:
        shutil.copyfile(ckpt_file, best_ckpt_file)

def save_demo(env, policy_net, episode_length, device, pad_img, feature_net, filename):
    """Runs an agent through an environment, saves the simulation as a gif"""
    state = env.reset()
    frames = []

    for step in range(episode_length):
        input_state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        if pad_img:
            input_state = pad_image(input_state).to(device)
        if feature_net:
            input_state = feature_net(input_state)
        # Select an action
        action_dist = policy_net(input_state)[0]
        action = np.random.choice(np.arange(len(action_dist)), p=action_dist.cpu().detach().numpy())
        next_state, reward, terminated, _ = env.step(action)
        # Record the frame
        env_image = env.unwrapped._get_image()
        env_image = Image.fromarray(env_image)
        frames.append(env_image)
        if terminated:
            break
        state = next_state
    
    frames[0].save('results/space_invaders.gif',
                save_all=True, append_images=frames[1:], optimize=False, duration=40, loop=0)

def save_spaceinvaders_demo():
    
    state_space_dim = 128
    action_space_dim = 6
    feature_dim = 100
    env_name = 'SpaceInvaders-ram-v0'
    episode_length = 1000

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    env = gym.make(env_name)
    policy_net = PolicyNetwork(feature_dim, action_space_dim, hidden_dim=100)
    feature_net = FCNetwork(state_space_dim, feature_dim)

    # Load saved model
    resume = 'saved_models/SpaceInvaders-ram-v0_best.pth'
    print("=> loading checkpoint '{}'".format(resume))
    checkpoint = torch.load(resume, map_location=device)
    policy_net.load_state_dict(checkpoint['policy_state_dict'])
    feature_net.load_state_dict(checkpoint['feature_state_dict'])
    print("=> loaded checkpoint")

    save_demo(env, policy_net, episode_length=episode_length, device=device, pad_img=False, feature_net=feature_net, filename='results/space_invaders_ppo.gif')

def save_cartpole_demo():
    # TODO: implement this
    pass