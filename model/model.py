from common.layers import NoisyLinear
import torch
import torch.nn as nn
import random
import numpy as np

# Based off Facebook's RL Assembly [9] Repo.
class DistributionalIQNDuelingLSTMNet(nn.Module):
    def __init__(self, device, num_action, nenvs, multi=True, noisy=True, noisy_std=0.5, dueling=True, distributional=True, num_tau=8, dropout=False, conv_dropout=0.1, linear_dropout=0.4, actions_and_rewards=True, deeper=True, skip=True, use_lstm=True):
        super().__init__()

        # Parameters
        self.frame_stack = 4
        self.conv_out_dim = 3136
        self.hid_dim = 512
        self.skip = skip
        self.hid_dim_intermediary = self.hid_dim if not self.skip else self.hid_dim * 2
        self.num_lstm_layer = 1
        self.device = device
        self.num_action = num_action
        self.nenvs = nenvs
        self.multi = multi
        self.noisy = noisy
        self.noisy_std = noisy_std
        self.dueling = dueling
        self.distributional = distributional
        self.actions_and_rewards = actions_and_rewards
        self.deeper = deeper
        self.use_lstm = use_lstm
        
        self.num_tau = num_tau
        self.n_cos = 64
        self.pis = torch.FloatTensor([np.pi*i for i in range(0,self.n_cos)]).view(1,1,1,self.n_cos).to(device) # Starting from 0 as in the paper 
        
        # I believe the way that noisy linear layers are set up would break in a strange multi gpu setting as the device numba is not retained?
        if device.type == 'cuda':
            self.use_cuda = True
        else:
            self.use_cuda = False
            

        # Convolutional state encoder.
        self.net = nn.Sequential(
            nn.Conv2d(self.frame_stack, 32, 8, stride=4),
            nn.ReLU() if not dropout else nn.Sequential(nn.ReLU(), nn.Dropout(conv_dropout)),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU() if not dropout else nn.Sequential(nn.ReLU(), nn.Dropout(conv_dropout)),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU() if not dropout else nn.Sequential(nn.ReLU(), nn.Dropout(conv_dropout))
        ).to(self.device)

        if not self.actions_and_rewards:
            if self.use_lstm:
                # LSTM to take in encoded states from CNN (and the hidden state at beginning of sequence) and output a state representation with longer term dependencies and information.
                self.lstm = nn.LSTM(
                    self.conv_out_dim, self.hid_dim, num_layers=self.num_lstm_layer
                ).to(self.device)
            else:
                self.lstm = nn.Sequential(
                    NoisyLinear(self.conv_out_dim, self.hid_dim, self.use_cuda, self.noisy_std) if self.noisy else nn.Linear(self.conv_out_dim, self.hid_dim),
                    nn.ReLU() if not dropout else nn.Sequential(nn.ReLU(), nn.Dropout(linear_dropout))
                ).to(self.device)
        else:
            # If passing actions and rewards too (instead) like R2D2 does.
            if self.use_lstm:
                # LSTM to take in encoded states from CNN (and the hidden state at beginning of sequence) and output a state representation with longer term dependencies and information.
                self.lstm = nn.LSTM(
                    self.conv_out_dim + self.num_action + 1, self.hid_dim, num_layers=self.num_lstm_layer
                ).to(self.device)
            else:
                self.lstm = nn.Sequential(
                    NoisyLinear(self.conv_out_dim + self.num_action + 1, self.hid_dim, self.use_cuda, self.noisy_std) if self.noisy else nn.Linear(self.conv_out_dim + self.num_action + 1, self.hid_dim),
                    nn.ReLU() if not dropout else nn.Sequential(nn.ReLU(), nn.Dropout(linear_dropout))
                ).to(self.device)

        self.cos_embedding = nn.Sequential(
            nn.Linear(self.n_cos, self.hid_dim),
            nn.ReLU(),
        ).to(self.device)
        
        

        # Value function fully connected layers: {
        
        if self.dueling:
            # This design is inspired by the findings of the D2RL paper, which allows our network to become much deeper than usual.
            # Directly connected to the LSTM output.
            self.fc_v_1 = nn.Sequential(
                NoisyLinear(self.hid_dim, self.hid_dim, self.use_cuda, self.noisy_std) if self.noisy else nn.Linear(self.hid_dim, self.hid_dim),
                nn.ReLU() if not dropout else nn.Sequential(nn.ReLU(), nn.Dropout(linear_dropout))
            ).to(self.device)
            
            # # Contains a skip connection to the CNN output.
            # self.fc_v_2 = nn.Sequential(
                # NoisyLinear(self.hid_dim + self.conv_out_dim, self.hid_dim, self.use_cuda, self.noisy_std) if self.noisy else nn.Linear(self.hid_dim + self.conv_out_dim, self.hid_dim),
            #nn.ReLU() if not dropout else nn.Sequential(nn.ReLU(), nn.Dropout(linear_dropout))
            # ).to(self.device)

            # Contains a skip connection to the LSTM output.
            self.fc_v_2 = nn.Sequential(
                NoisyLinear(self.hid_dim_intermediary, self.hid_dim, self.use_cuda, self.noisy_std) if self.noisy else nn.Linear(self.hid_dim_intermediary, self.hid_dim),
                nn.ReLU() if not dropout else nn.Sequential(nn.ReLU(), nn.Dropout(linear_dropout))
            ).to(self.device)
            
            if self.deeper:
                # Contains a skip connection to the LSTM output.
                self.fc_v_3 = nn.Sequential(
                    NoisyLinear(self.hid_dim_intermediary, self.hid_dim, self.use_cuda, self.noisy_std) if self.noisy else nn.Linear(self.hid_dim_intermediary, self.hid_dim),
                    nn.ReLU() if not dropout else nn.Sequential(nn.ReLU(), nn.Dropout(linear_dropout))
                ).to(self.device)

            
            # Contains a skip connection to the LSTM output.
            self.fc_v_4 = NoisyLinear(self.hid_dim_intermediary, 1, self.use_cuda, self.noisy_std).to(self.device) if self.noisy else nn.Linear(self.hid_dim_intermediary, 1).to(self.device)

            # }

        # Advantage function fully connected layers (or normal Q-network if dueling==False): {
            
        # This design is inspired by the findings of the D2RL paper, which allows our network to become much deeper than usual.
        # Directly connected to the LSTM output.
        self.fc_a_1 = nn.Sequential(
            NoisyLinear(self.hid_dim, self.hid_dim, self.use_cuda, self.noisy_std) if self.noisy else nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU() if not dropout else nn.Sequential(nn.ReLU(), nn.Dropout(linear_dropout))
        ).to(self.device)

        # # Contains a skip connection to the CNN output.
        # self.fc_a_2 = nn.Sequential(
            # NoisyLinear(self.hid_dim + self.conv_out_dim, self.hid_dim, self.use_cuda, self.noisy_std) if self.noisy else nn.Linear(self.hid_dim + self.conv_out_dim, self.hid_dim),
        #nn.ReLU() if not dropout else nn.Sequential(nn.ReLU(), nn.Dropout(linear_dropout))
        # ).to(self.device)

        # Contains a skip connection to the LSTM output.
        self.fc_a_2 = nn.Sequential(
            NoisyLinear(self.hid_dim_intermediary, self.hid_dim, self.use_cuda, self.noisy_std) if self.noisy else nn.Linear(self.hid_dim_intermediary, self.hid_dim),
            nn.ReLU() if not dropout else nn.Sequential(nn.ReLU(), nn.Dropout(linear_dropout))
        ).to(self.device)

        if self.deeper:
            # Contains a skip connection to the LSTM output.
            self.fc_a_3 = nn.Sequential(
                NoisyLinear(self.hid_dim_intermediary, self.hid_dim, self.use_cuda, self.noisy_std) if self.noisy else nn.Linear(self.hid_dim_intermediary, self.hid_dim),
                nn.ReLU() if not dropout else nn.Sequential(nn.ReLU(), nn.Dropout(linear_dropout))
            ).to(self.device)

        # Contains a skip connection to the LSTM output.
        self.fc_a_4 = NoisyLinear(self.hid_dim_intermediary, self.num_action, self.use_cuda, self.noisy_std).to(self.device) if self.noisy else nn.Linear(self.hid_dim_intermediary, self.num_action).to(self.device)

        # }
        
        if self.use_lstm:
            self.lstm.flatten_parameters()
        
    # https://github.com/BY571/IQN-and-Extensions
    def calc_cos(self, seq, batch_size, n_tau=8):
        """
        Calculating the cosine values depending on the number of tau samples
        """
        taus = torch.rand(seq, batch_size, n_tau, 1, device=self.device) #(seq, batch_size, n_tau, 1)
        cos = torch.cos(taus*self.pis)

        assert cos.shape == (seq, batch_size,n_tau,self.n_cos), "cos shape is incorrect"
        return cos, taus

    def reset_noise(self):
        if self.dueling:
            self.fc_v_1[0].reset_noise()
            self.fc_v_2[0].reset_noise()
            self.fc_v_4.reset_noise()
        self.fc_a_1[0].reset_noise()
        self.fc_a_2[0].reset_noise()
        self.fc_a_4.reset_noise()
    
    def get_h0(self, batchsize):
        """
        Retrieve initial hidden state of LSTM.
        """
        shape = (self.num_lstm_layer, batchsize, self.hid_dim)
        hid = {"h0": torch.zeros(*shape, device=self.device), "c0": torch.zeros(*shape, device=self.device)}
        return hid

    def duel(self, v, a):
        """
        Takes in Q-value outputs from Value and Advantage networks, and produces the Duelling Q-Networks outputs.
        """
        q = v + a - a.mean(2, keepdim=True)
        return q

    def _conv_forward(self, s):
        """
        Send observation through CNN.
        """
        assert s.dim() == 4  # [batch, c, h, w]
        x = self.net(s) #state to representaion
        x = x.view(s.size(0), self.conv_out_dim)
        return x

    def advantage(self, o, s):
        """
        Retrieve Q-values for each action from the current state.
        Taken from the D2RL [2] paper's insights and architecture.
        o: LSTM output
        s: CNN output
        """
        a = self.fc_a_1(o)
        if self.skip:
            a = torch.cat([a, o], dim=-1)
        a = self.fc_a_2(a)
        if self.deeper:
            if self.skip:
                a = torch.cat([a, o], dim=-1)
            a = self.fc_a_3(a)
        if self.skip:
            a = torch.cat([a, o], dim=-1)
        a = self.fc_a_4(a)
        return a

    def value(self, o, s):
        """
        Retrieve Q-value for the value of the current state.
        Taken from the D2RL [2] paper's insights and architecture.
        o: LSTM output
        s: CNN output
        """
        v = self.fc_v_1(o)
        if self.skip:
            v = torch.cat([v, o], dim=-1)
        v = self.fc_v_2(v)
        if self.deeper:
            if self.skip:
                v = torch.cat([v, o], dim=-1)
            v = self.fc_v_3(v)
        if self.skip:
            v = torch.cat([v, o], dim=-1)
        v = self.fc_v_4(v)
        return v

    def act(self, obs, actions, rewards, hid, epsilon=0.01, eps_greedy=True, multi=None):
        """
        Retrieve the action an agent(s) should take according to the Duelling Network (in this case simply the Advantage Network), with epsilon greedy policy support.
        Simultaneously return the next hidden state of the LSTM.
        """
        batch = obs.size(0)
        if multi == None:
            multi = self.multi
        
        x = self._conv_forward(obs)
        # x: [batch, hid]
        x = x.unsqueeze(0)
        # x: [1, batch, hid]

        # If passing actions and rewards too (instead) like R2D2 does.
        if self.actions_and_rewards:
            x = torch.cat([x,actions,rewards], axis = -1)

        if self.use_lstm:
            o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        else:
            o = self.lstm(x)
            (h, c) = (hid["h0"], hid["c0"])
        
        del obs, actions, rewards, hid
        
        if self.distributional:
            cos, taus = self.calc_cos(1, batch, self.num_tau) # cos shape (batch, num_tau, hid_dim)
            sample_embeddings = self.cos_embedding(cos)
            
            # x has shape (seq, batch, hid_dim), reshape to (seq, batch, 1, hid_dim) for multiplication 
            o = (o.unsqueeze(2)*sample_embeddings)
            # o = o.view(1, batch, self.num_tau*self.hid_dim)
        # else:
        #     taus = None

        if multi:
            a = self.advantage(o, x)
            if self.distributional:
                # Calculate the mean of the quantiles
                a = a.mean(dim=-2)
            a = a.squeeze(0)
            # a: [batch, num_action]
            # legal_a = (1 + a - a.min())
            greedy_action = a.argmax(1).detach()
            if eps_greedy:
                random_actions = torch.randint(low=0, high=self.num_action, size=(batch,), device=self.device)
                probs = torch.rand(batch, device=self.device)
                greedy_action = torch.where(probs < epsilon, random_actions, greedy_action)
            return greedy_action.tolist(), {"h0": h.detach(), "c0": c.detach()}
        else:
            if eps_greedy and random.random() < epsilon:
                greedy_action = random.randrange(self.num_action)
            else:
                a = self.advantage(o, x)
                if self.distributional:
                    # Calculate the mean of the quantiles
                    a = a.mean(dim=2)
                a = a.squeeze(0)
                # a: [batch, num_action]
                # legal_a = (1 + a - a.min())
                greedy_action = a.argmax(1).detach()
            
            return int(greedy_action), {"h0": h.detach(), "c0": c.detach()}

    def unroll_rnn(self, obs, actions, rewards, hid):
        """
        Send observation through both CNN and LSTM.
        """
        s = obs
        assert s.dim() == 5  # [seq, batch, c, h, w]
        seq, batch, c, h, w = s.size()
        s = s.view(seq * batch, c, h, w)
        # Send through CNN.
        x = self.net(s)
        x = x.view(seq, batch, self.conv_out_dim)
        
        # If passing actions and rewards too (instead) like R2D2 does.
        if self.actions_and_rewards:
            x = torch.cat([x,actions,rewards], axis = -1)

        if self.use_lstm:
            o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        else:
            o = self.lstm(x)

        # o: LSTM output, x: CNN output.
        
        if self.distributional:
            cos, taus = self.calc_cos(seq, batch, self.num_tau) # cos shape (batch, num_tau, layer_size)
            sample_embeddings = self.cos_embedding(cos)
            
            # o has shape (seq, batch, hid_dim), reshape to (seq, batch, 1, hid_dim) for multiplication 
            o = (o.unsqueeze(2)*sample_embeddings)
        else:
            taus = None
        
        return o, x, {"h0": h, "c0": c}, taus

    def forward(self, obs, actions, rewards, hid):
        """
        Send observation through the network and return Q-values.
        return:
            q(s, a): [seq, batch, num_action]
            hid(s_n): [batch] (n=seq, final hidden state, used for LSTM burnin)
        """
        o, x, hid, taus = self.unroll_rnn(obs, actions, rewards, hid)
        # o: LSTM output, x: CNN output.
        if self.dueling:
            a = self.advantage(o, x)
            v = self.value(o, x)
            q = self.duel(v, a)
        else:
            # Note here the advantage network is in fact just the normal network not an advantage network, it only shows the same name to not duplicate code.
            q = self.advantage(o, x)
        # if self.distributional:
        #     # Calculate the mean of the quantiles
        #     q = q.mean(dim=2)
        return q, hid, taus