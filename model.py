import torch
import torch.nn as nn


class BaseConv(nn.Module):
    def __init__(self, c_in):
        '''
        - {Conv > ELU > } * 4
        - "A3C: a sequence of four convolution layers with 32 filters each, kernel size of 3x3,
        stride of 2 and padding of 1. An ELU is used after each convolution
        layer"
        - "ICM: "
        
        Parameters:
            c_in : input channel
    
        '''
        super(BaseConv, self).__init__()

        filter = 32
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, filter, kernel_size=(3,3), stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(filter, filter, kernel_size=(3,3), stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(filter, filter, kernel_size=(3,3), stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(filter, filter, kernel_size=(3,3), stride=2, padding=1),
            nn.ELU(),
        )

        self.output_size = self._get_output_size(c_in)

    def _get_output_size(self, c_in):
        "The input images are re-sized to 42 x 42."
        x = torch.randn(1, c_in, 42, 42)
        x = self.conv(x)
        return x.view(1,-1).size(1)

    def forward(self, x):
        return self.conv(x)

class ActorCritic(nn.Module):
    def __init__(self, c_in, num_actions):
        '''
        - "The output of the last convolution layer is fed into a LSTM with 256
        units. Two seperate fully connected layers are used to predict the
        value function and the action from the LSTM feature representation."


        Parameters:
            c_in : input channel
            num_actions : size of action space. 4 for Doom, 12 for Mario.
        
        '''
        super(ActorCritic, self).__init__()

        self.num_lstm_unit = 256
        self.conv = BaseConv(c_in)
        self.lstm = nn.LSTM(input_size=self.conv.output_size,
                            hidden_size=self.num_lstm_unit,
                            num_layers=4,
                            batch_first=True,
                            )
            # Inputs
            #   - input: (N,L,Hin) : [1, 4, 288]
            #   - h_0, c_0 : same as h_n, c_n
            # Outputs
            #   - h_n: (D*num_layers, N, Hout) : [4, 1, 256]
            #   - c_n: (D*num_layers, N, Hcell) : [4, 1, 256]
        self.critic = nn.Linear(self.num_lstm_unit, 1)
        self.actor = nn.Linear(self.num_lstm_unit, num_actions)
    
    def forward(self, x, h0, c0):
        x = self.conv(x)    # x.shape == torch.Size([4, 32, 3, 3])

        if isinstance(h0, int):
            h0 = torch.zeros((4, 1, self.num_lstm_unit))
        if isinstance(c0, int):
            c0 = torch.zeros((4, 1, self.num_lstm_unit))
    
        _, (hn, cn) = self.lstm(x.view(1, x.size(0), -1), (h0, c0))
        return self.actor(hn[-1]), self.critic(hn[-1]), hn, cn
