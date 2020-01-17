################################################################
# datasets.py - dataset classes for Atari environments
# 
# ExperienceDataset(Dataset)
#   Dataset holding experience in the form of (state, action_distribuion,
#   action, reward, discounted_reward) tuples
################################################################

from torch.utils.data import Dataset

class ExperienceDataset(Dataset):
    """Dataset holding experience in the form of (state, action_distribuion,
    action, reward, discounted_reward) tuples
    """
    def __init__(self, rollouts):
        super(ExperienceDataset, self).__init__()
        
        self.experience = self.parse_rollouts(rollouts)
        
    def parse_rollouts(self, rollouts):
        experience = []
        for rollout in rollouts:
            experience.extend(rollout)
        return experience
    
    def __getitem__(self, index):
        return self.experience[index]
    
    def __len__(self):
        return len(self.experience)