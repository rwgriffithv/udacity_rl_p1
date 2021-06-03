# Deep Q-Learning With Experience Replay (V. Mnih et al.)

import torch
import torch.nn as tnn
import torch.optim as topt
import random as rand

# local imports
from .nn import polyak_update


class DeepQ:
    def __init__(self, qnet, target_qnet, replay_buf, learning_rate=0.001, discount_factor=0.99, polyak_factor=0.95):
        self.qnet = qnet
        self.target_qnet = target_qnet
        self.replay_buf = replay_buf
        self.discount_factor = discount_factor
        self.polyak_factor = polyak_factor
        # initialize optimizer
        self.optimizer = topt.Adam(qnet.parameters(), learning_rate)

    def step(self, batch_size):
        # get sample batch
        states, actions, rewards, terminals, next_states = self.replay_buf.sample(batch_size)
        # zero/clear gradients
        self.optimizer.zero_grad()
        self.qnet.train(True) # ensure qnet is training so back propogation can occur
        # q_vals according to sampled action taken from sampled states
        q_vals = self.qnet(states).gather(-1, actions)
        targ_q_vals = torch.max(self.target_qnet(next_states), -1)
        discounted_targ_q_vals = self.discount_factor * targ_q_vals
        # calculate MSE loss and perform gradient step
        loss = tnn.MSELoss()(q_vals, rewards + (1 - terminals) * discounted_targ_q_vals)
        loss.backward()
        self.optimizer.step()
        # polyak update target qnet
        polyak_update(self.target_qnet, self.qnet, self.polyak_factor)

    def get_action(self, state):
        self.qnet.train(False) # evaluation/inference mode
        q_val_probs = torch.softmax(self.qnet(state), -1).numpy()
        self.qnet.train(True)
        p = rand.uniform(0, 1)
        cumulative_qvp = 0
        for i, qvp in enumerate(q_val_probs):
            cumulative_qvp += qvp
            if p <= cumulative_qvp:
                return i
        return len(q_val_probs) - 1