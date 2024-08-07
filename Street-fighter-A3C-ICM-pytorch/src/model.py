"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn


class BaseConv(nn.Module):
    def __init__(self, num_inputs):
        super(BaseConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1),
                                  nn.ELU(),
                                  nn.Conv2d(32, 32, 3, stride=2, padding=1),
                                  nn.ELU(),
                                  nn.Conv2d(32, 32, 3, stride=2, padding=1),
                                  nn.ELU(),
                                  nn.Conv2d(32, 32, 3, stride=2, padding=1),
                                  nn.ELU(),
                                #   nn.Conv2d(32, 32, 3, stride=2, padding=1),
                                #   nn.ELU()
                                  )

    def forward(self, x):
        return self.conv(x)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        self.conv = BaseConv(num_inputs)
        # self.lstm = nn.LSTMCell(64 * 6 * 6, 1024)
        self.lstm = nn.LSTM(input_size=288, hidden_size=256, batch_first=True) # n_layer
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_actions)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hx, cx):
        x = self.conv(x)
        _,(hx,cx) = self.lstm(x.view(1,1,-1), (hx, cx)) # Input : (seq_len : 1, batch:1, input_size:288) 
        return self.actor_linear(hx), self.critic_linear(hx), hx, cx
        # x = x.view(x.size(0), 1, -1)  # 배치 크기 유지, 시퀀스 길이 1 추가
        # hx, cx = self.lstm(x, (hx, cx))
        # return self.actor_linear(hx[-1]), self.critic_linear(hx[-1]), hx, cx


class IntrinsicCuriosityModule(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(IntrinsicCuriosityModule, self).__init__()
        self.conv = BaseConv(num_inputs)
        self.feature_size = 64 * 6 * 6
        self.inverse_net = nn.Sequential(
            nn.Linear(self.feature_size * 2, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, num_actions)
        )
        self.forward_net = nn.Sequential(
            nn.Linear(self.feature_size + num_actions, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, self.feature_size)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, state, next_state, action):
        state_ft = self.conv(state)
        next_state_ft = self.conv(next_state)
        state_ft = state_ft.view(-1, self.feature_size)
        next_state_ft = next_state_ft.view(-1, self.feature_size)
        return self.inverse_net(torch.cat((state_ft, next_state_ft), 1)), self.forward_net(
            torch.cat((state_ft, action), 1)), next_state_ft
