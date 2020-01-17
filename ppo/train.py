################################################################
# train.py - training loop for PPO algorithm and default training
# configurations for CartPole and SpaceInvaders
# 
# train()
#   Trains policy and value networks on the given environment
# train_cartpole()
#   Trains a CartPole agent using default hyperparameters.
# train_spaceinvaders_ram()
#   Trains a SpaceInvaders agent using default hyperparameters.
################################################################

from ppo.model import FCNetwork, PolicyNetwork, ValueNetwork
from ppo.datasets import ExperienceDataset

def train(env_name, 
         policy_net,
         value_net,
         feature_net=None,
         device='cpu', 
         epochs=50, 
         episodes_per_epoch=100, 
         episode_length=200, 
         gamma=0.99, 
         policy_epochs=5, 
         batch_size=256, 
         epsilon=0.2,
         environment_threads=8,
         pad_img=False,
         resume=None):
            
    value_criterion = nn.MSELoss()

    optimizer = optim.Adam(chain(policy_net.parameters(), value_net.parameters()), 
                           lr=1e-3, 
                           betas=(0.9, 0.999), 
                           weight_decay=0.01)
    
    policy_net = policy_net.to(device)
    value_net = value_net.to(device)
    if feature_net:
        feature_net = feature_net.to(device)
    
    value_losses = []
    policy_losses = []
    avg_rewards = []
        
    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            value_losses = checkpoint['value_losses']
            policy_losses = checkpoint['policy_losses']
            avg_rewards = checkpoint['avg_rewards']
            start_epoch = checkpoint['epoch']
            value_net.load_state_dict(checkpoint['value_state_dict'])
            policy_net.load_state_dict(checkpoint['policy_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            
    # Prepare the environments
    environments = [gym.make(env_name) for _ in range(environment_threads)]
    rollouts_per_thread = episodes_per_epoch // environment_threads
    remainder = episodes_per_epoch % environment_threads
    rollout_nums = ([rollouts_per_thread + 1] * remainder) + ([rollouts_per_thread] * (environment_threads - remainder))

    start_epoch = 0
    loop = tqdm(total=epochs * policy_epochs, position=0)
    try:
        for e in range(start_epoch, epochs):

            rollouts = []
            rewards = []

            # Run the environments
            threads = [Thread(target=generate_rollouts, args=(environments[i],
                                                      policy_net,
                                                      rollouts,
                                                      rewards,
                                                      rollout_nums[i],
                                                      episode_length,
                                                      gamma,
                                                      device,
                                                      pad_img,
                                                      feature_net,
                                                      i)) for i in range(environment_threads)]
            for x in threads:
                x.start()
            for x in threads:
                x.join()

            avg_r = sum(rewards) / len(rewards)
            avg_rewards.append(avg_r)
            experience_dataset = ExperienceDataset(rollouts)
            experience_loader = DataLoader(experience_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          pin_memory=True)
            
            for _ in range(policy_epochs):
                for state, old_action_dist, old_action, _, discounted_return in experience_loader:
                    # Format the data from the rollout
                    state = state.detach().float().to(device)
                    if pad_img:
                        state = pad_image(state).to(device)
                    if feature_net:
                        state = feature_net(state)
                    old_action_dist = old_action_dist.detach().float().to(device)
                    old_action = old_action.detach().float().to(device)
                    discounted_return = discounted_return.detach().float().to(device).unsqueeze(1)

                    # Compare the estimated returns to the actual returns for the value loss
                    estimated_returns = value_net(state)
                    val_loss = value_criterion(estimated_returns, discounted_return)
                    value_losses.append(val_loss.item())

                    # Compute the ratio of the current value of the action to the former value of the action
                    new_action_dist = policy_net(state)
                    new_action_probs = new_action_dist[range(new_action_dist.shape[0]), old_action.long()[:, 0]].unsqueeze(1)
                    old_action_probs = old_action_dist[range(old_action_dist.shape[0]), old_action.long()[:, 0]].unsqueeze(1)
                    # This is a measure of how much our network changes with each epoch
                    ratio = new_action_probs / old_action_probs
                    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)

                    advantage = discounted_return - estimated_returns.detach()

                    # Train the policy to maximize the advantage, but clipped to avoid large steps
                    policy_loss = -torch.mean(torch.min(ratio * advantage, clipped_ratio * advantage))
                    policy_losses.append(policy_loss.item())

                    # Compute gradients and step
                    optimizer.zero_grad()
                    loss = policy_loss + val_loss
                    loss.backward()
                    optimizer.step()

                loop.update(1)

            loop.set_description("Epoch {:d} : Avg. Reward: {:6.2f}, Value Loss: {:6.2f}, Policy Loss: {:6.2f}."
                             .format(e + 1, avg_rewards[-1], value_losses[-1], policy_losses[-1]))
    except KeyboardInterrupt:
        pass
    finally:
        # Save the model
        print("\nSaving model...")
        save_checkpoint( { 
                    'epoch' : e + 1,
                    'value_losses' : value_losses,
                    'policy_losses' : policy_losses,
                    'avg_rewards' : avg_rewards,
                    'value_state_dict' : value_net.state_dict(),
                    'policy_state_dict' : policy_net.state_dict(),
                         }, is_best=True, env_name=env_name )
        print("Model saved.")
        
        # Cleanup tqdm, plot rewards
        loop.close()
        plt.plot(np.arange(len(avg_rewards)), avg_rewards)
        plt.title("Average Reward over Training")
        plt.xlabel("Epochs")
        plt.ylabel("Reward")
        plt.show()

def train_spaceinvaders_ram():
    """Trains a Space Invaders agent using default hyperparameters."""

    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        
    env = 'SpaceInvaders-ram-v0'
    
    state_space_dim = 128
    action_space_dim = 6
    feature_dim = 100
    
    policy_net = PolicyNetwork(feature_dim, action_space_dim, hidden_dim=100)
    value_net = ValueNetwork(feature_dim, hidden_dim=100)
    feature_net = FCNetwork(state_space_dim, feature_dim)
    main(env, 
         policy_net, 
         value_net, 
         feature_net=feature_net,
         device=device, 
         epochs=400, 
         episodes_per_epoch=30, 
         episode_length=1000, 
         gamma=0.99, 
         policy_epochs=5, 
         batch_size=256, 
         epsilon=0.2, 
         pad_img=False)
         resume="{}.pth.tar".format(env))
    
def train_cartpole():
    """Trains a CartPole agent using default hyperparameters."""
    
    state_space_dim = 4
    action_space_dim = 2
    epochs = 30
    env = 'CartPole-v0'
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    policy_net = PolicyNetwork(state_space_dim, action_space_dim)
    value_net = ValueNetwork(state_space_dim)
    
    train(env, policy_net, value_net, device=device, epochs=epochs)