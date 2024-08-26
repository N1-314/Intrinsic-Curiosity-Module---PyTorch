import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

class UniversalHead(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.univers_head = nn.Sequential(
            nn.Conv2d(inputs, 32, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Flatten()
        )
    def forward(self, x):
        return self.univers_head(x)

class ActorCritic(nn.Module):
    def __init__(self, env):
        super().__init__()
        inputs = env.observation_space.shape[0]
        # self.is_discrete = env.action_space.__class__.__name__ == 'Discrete'

        action_dim = env.action_space.n
        units = 256

        # The input state is passed through a sequence of four convolutional layers with 32 filters each, kernel size of 3x3, stride of 2 and padding of 1.
        # An exponential linear unit (ELU) is used after each convolutional layer.
        self.univers_head = nn.Sequential( # (4, 42, 42) -> (32, 21, 21) -> (32, 11, 11) -> (32, 6, 6) -> (32, 3, 3) -> (288)
            nn.Conv2d(inputs, 32, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Flatten() # 배치가 있다고 가정 그래야 start_dim=1이 타당
        )

        # The output of the last convolution layer is fed into a LSTM with 256 units.
        self.lstm = nn.LSTMCell(32*3*3, hidden_size=units)

        # Two separate fully connected layers are used to predict the value function and the action from the LSTM feature representation.
        self.value = nn.Linear(units, 1)
        self.logit = nn.Linear(units, action_dim)
    
        
    def forward(self, x, hx=None):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.univers_head(x)
        hx, cx = self.lstm(x, hx)
        
        # Critic
        value = self.value(hx)

        # Actor
        logit = self.logit(hx)
        policy = F.softmax(logit, dim=-1) # policy: probs

        # log_policy = torch.log(policy + 1e-10)
        log_policy = F.log_softmax(logit, dim=-1)
        entropy = -(policy * log_policy).sum(1, keepdim=True)

        m = Categorical(probs=policy)
        action = m.sample()#.item()
        log_prob = m.log_prob(action) # log_policy[0, action] == m.log_prob(action)

        return action.item(), value, log_prob, entropy, policy, hx, cx
        
    @torch.no_grad()
    def get_action(self, x, hx=None):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.univers_head(x)
        hx, cx = self.lstm(x, hx)
        logit = self.logit(hx)
        policy = F.softmax(logit, dim=-1)
        action = logit.argmax(dim=-1)

        return action.item(), policy, hx, cx
    
    def f(self, x, hx=None): # for captum
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.univers_head(x)
        hx, cx = self.lstm(x, hx)
        
        # Critic
        value = self.value(hx)

        # Actor
        logit = self.logit(hx)
        policy = F.softmax(logit, dim=-1) # policy: probs

        # log_policy = torch.log(policy + 1e-10)
        log_policy = F.log_softmax(logit, dim=-1)
        entropy = -(policy * log_policy).sum(1, keepdim=True)

        m = Categorical(probs=policy)
        action = m.sample()#.item()
        log_prob = m.log_prob(action) # log_policy[0, action] == m.log_prob(action)

        return policy
    
class ICM(nn.Module):
    def __init__(self, env):
        super().__init__()
        inputs = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.feature_size = 32*3*3

        self.conv = UniversalHead(inputs)
        self.inverse_net = nn.Sequential(
            nn.Linear(self.feature_size * 2, 256),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.forward_net = nn.Sequential(
            nn.Linear(self.feature_size + action_dim, 256),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(256, self.feature_size)
        )
    
    def forward(self, state, next_state, action):
        if state.dim() == 3:
            state = state.unsqueeze(0)
        if next_state.dim() == 3:
            next_state = next_state.unsqueeze(0)
            
        state_ft = self.conv(state).view(-1, self.feature_size)
        next_state_ft = self.conv(next_state).view(-1, self.feature_size)
        
        return self.inverse_net(torch.cat([state_ft, next_state_ft], dim=1)),\
               self.forward_net(torch.cat([state_ft, action], dim=1)),\
               next_state_ft