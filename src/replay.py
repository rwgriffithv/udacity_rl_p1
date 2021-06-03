# transition replay classes
# currently implemented to only support single-agent functionality

import torch
import numpy as np
import random as rand


# states are observed states
# more closely related to policy nn inputs
# can include things like previous actions (does not have to be a pure environment state)
class Transition:
    def __init__(self, state, action, reward, terminal, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.terminal = terminal # terminal flag 1 or 0, 1 signifying the agent will no longer receive rewards
        self.next_state = next_state

class ReplayBuffer:
    def __init__(self, capacity, state_size):
        self.capacity = capacity
        self.size = 0
        self.states = np.zeros((capacity, state_size), np.dtype(float))
        # for discrete actions dtype is always int; using float for future continuous action use
        self.actions = np.zeros((capacity, 1), np.dtype(float))
        self.rewards = np.zeros((capacity, 1), np.dtype(float))
        self.terminals = np.zeros((capacity, 1), np.dtype(int))
        self.next_states = np.zeros((capacity, state_size), np.dtype(float))

    def insert(self, transitions):
        # append transitions if possible
        append_trans = transitions[:self.capacity - self.size]
        for t in append_trans:
            self.states[self.size] = t.state
            self.actions[self.size] = t.action
            self.rewards[self.size] = t.reward
            self.terminals[self.size] = t.terminal
            self.next_states[self.size] = t.next_state
            self.size += 1
        # insert remaining transitions (randomly replacing prev trans)
        insert_trans = transitions[len(append_trans):]
        idxs = rand.sample(range(self.capacity), len(insert_trans))
        for i, t in zip(idxs, insert_trans):
            self.states[i] = t.state
            self.actions[i] = t.action
            self.rewards[i] = t.reward
            self.terminals[i] = t.terminal
            self.next_states[i] = t.next_state

    def sample(self, batch_size):
        idxs = rand.choices(range(self.size), k=batch_size)
        b_states = self.states[idxs]
        b_actions = self.actions[idxs]
        b_rewards = self.rewards[idxs]
        b_terminals = self.terminals[idxs]
        b_next_states = self.next_states[idxs]
        return b_states, b_actions, b_rewards, b_terminals, b_next_states