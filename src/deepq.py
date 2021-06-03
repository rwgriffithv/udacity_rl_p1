# Deep Q-Learning With Experience Replay (V. Mnih et al.)

from numpy.lib.polynomial import poly
import torch
import torch.nn as tnn
import torch.optim as topt
import torch.cuda as tcuda
import numpy as np
import random as rand

# local imports
from .nn import polyak_update


class DeepQ:
    def __init__(self, qnet, target_qnet, replay_buf, learning_rate=0.0003, discount_factor=0.99, polyak_factor=0.99):
        self.qnet = qnet
        self.target_qnet = target_qnet
        # set target_qnet to qnet's weights
        polyak_update(self.qnet, self.target_qnet, 0)
        self.replay_buf = replay_buf
        # initialize optimizer
        self.optimizer = topt.Adam(qnet.parameters(), learning_rate)
        # get devices
        self.dev_gpu = torch.device("cuda" if tcuda.is_available() else "cpu")
        self.dev_cpu = torch.device("cpu")
        # create constant tensors
        self.discount_factor = torch.tensor(discount_factor, dtype=torch.float32, device=self.dev_gpu)
        self.polyak_factor = torch.tensor(polyak_factor, dtype=torch.float32, device=self.dev_gpu)

    def optimize(self, num_steps=1, batch_size=1000):
        for _ in range(num_steps): # number of gradient steps
            # get sample batch, convert numpy arrays to tensors and send to GPU
            batch_tuple = self.replay_buf.sample(batch_size)
            states, actions, rewards, terminals, next_states = [torch.from_numpy(b).float().to(self.dev_gpu) for b in batch_tuple]
            torch.clamp(rewards, min=-1.0, max=1.0)
            self.qnet.train(True) # ensure qnet is training so back propogation can occur
            # q_vals according to sampled action taken from sampled states
            q_vals = torch.gather(self.qnet(states), -1, actions.type(torch.int64))
            targ_q_vals = torch.unsqueeze(torch.max(self.target_qnet(next_states).detach(), axis=-1)[0], dim=-1) # taking the maximum of the target q values
            discounted_targ_q_vals = self.discount_factor * targ_q_vals
            # calculate MSE loss and perform gradient step
            loss = tnn.MSELoss()(q_vals, rewards + (1 - terminals) * discounted_targ_q_vals)
            self.optimizer.zero_grad() # zero/clear previous gradients
            loss.backward()
            self.optimizer.step()
            # polyak update target qnet
            polyak_update(self.qnet, self.target_qnet, self.polyak_factor)

    def get_action(self, state, epsilon=0.0):
        qnet_in = torch.from_numpy(np.array(state)).float().to(self.dev_gpu) # convert state, state can be numpy array or list
        self.qnet.train(False) # evaluation/inference mode
        with torch.no_grad():
            q_val_probs = torch.softmax(self.qnet(qnet_in), -1).to(self.dev_cpu).numpy()
        self.qnet.train(True)
        if rand.random() > epsilon:
            return int(np.argmax(q_val_probs))
        else:
            p = rand.uniform(0, 1)
            cumulative_qvp = 0
            for i, qvp in enumerate(q_val_probs):
                cumulative_qvp += qvp
                if p <= cumulative_qvp:
                    return i
            return len(q_val_probs) - 1