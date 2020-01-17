
################################################################
# models.py - Models for use in PPO algorithm
# 
# ConvNetwork
#   A feature extractor from pixel environments
# FCNetwork
#   A feature extractor from ram-based environments
# ValueNetwork
#   A network to learn values of actions based on the current 
#   state features
# PolicyNetwork
#   A network to learn the optimal distributions over actions
#   based on the current state
################################################################

class ConvNetwork(nn.Module):
    """Network to extract features from images, as a precursor to the Policy and
    Value Networks"""
    def __init__(self, size, input_channels):
        super(ConvNetwork, self).__init__()
        
        self.net = nn.Sequential(nn.Conv2d(input_channels, 64, 3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                 nn.ReLU(),
                                 nn.Conv2d(128, 128, 3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv2d(128, 256, 3, stride=2, padding=1),
                                 nn.ReLU(),
                                 nn.Conv2d(256, 256, 3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv2d(256, 512, 3, stride=2, padding=1),
                                 nn.ReLU(),
                                 nn.Conv2d(512, 512, 3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv2d(512, 1024, 3, stride=2, padding=1))
        
    def forward(self, img):
        embedding = self.net(img)
        return embedding.view(1, -1)
    
class FCNetwork(nn.Module):
    """Network to extract features from non-structured data, as a precursor to the Policy and
    Value Networks"""
    def __init__(self, input_dim, output_dim, hidden_dim=100):
        super(FCNetwork, self).__init__()
        
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim))
        
    def forward(self, img):
        embedding = self.net(img)
        return embedding
    
class PolicyNetwork(nn.Module):
    """Takes in a state, outputs a distribution over actions"""
    def __init__(self, state_dim=4, act_space_dim=2, hidden_dim=10):
        super(PolicyNetwork, self).__init__()
        
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, act_space_dim),
        )
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, state):
        action_probs = self.softmax(self.net(state))
        return action_probs
    
class ValueNetwork(nn.Module):
    """Takes in a state, outputs the estimated time-discounted value of that state"""
    def __init__(self, state_dim=4, hidden_dim=10):
        super(ValueNetwork, self).__init__()
        
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state):
        value = self.net(state)
        return value