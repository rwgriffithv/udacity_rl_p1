# training a single angent for a given UnityEnviroment executable using Deep Q-Learning

import sys
import numpy as np
from unityagents import UnityEnvironment, brain
from torch import save

# local imports
from .nn import build_network
from .replay import Transition, ReplayBuffer
from .deepq import DeepQ


def train(banana_bin_path):
    # environment solution constants
    REQ_AVG_SCORE = 13
    # training constants
    REPBUF_CAPCITY = int(1e5)
    REPBUF_PREFILL_RATIO = 0.5
    LEARNING_RATE = 0.001
    DISCOUNT_FACTOR = 0.99
    POLYAK_FACTOR = 0.95
    
    # instantiate environment
    env = UnityEnvironment(file_name=banana_bin_path)
    # get default brain name
    brain_name = env.brain_names[0]
    
    # get environment state and action size
    env_info = env.reset(train_mode=True)[brain_name]
    env_state_size = len(env_info.vector_observations[0])
    action_size = env.brains[brain_name].vector_action_space_size
    
    # build sequential artificial neural networks for the q-function and target q-function
    state_size = env_state_size + 1 # make states that include previous action id
    qnet = build_network(state_size, action_size)
    target_qnet = build_network(state_size, action_size) # target q network

    # states stored in replay buffer are environment states concatenated with previous action id
    replay_buf = ReplayBuffer(REPBUF_CAPCITY, state_size)

    # prefill replay buffer with transitions collected from taking random actions
    # do not track scores here, qnet is not being used and is not training
    num_rand_episodes = 0
    while replay_buf.size < int(REPBUF_PREFILL_RATIO * REPBUF_CAPCITY):
        num_rand_episodes += 1
        env_info = env.reset(train_mode=False)[brain_name]
        prev_action = 0
        state = [*env_info.vector_observations[0], prev_action] # including action id in state
        while True:
            action = np.random.randint(action_size)
            env_info = env.step(action)[brain_name]
            reward = env_info.rewards[0]
            terminal = 1 if env_info.local_done[0] else 0
            next_state = [*env_info.vector_observations[0], action] # including action id in state
            replay_buf.insert([Transition(state, action, reward, terminal, next_state)])
            state = next_state # roll over state
            if terminal: # check if episode is done
                break
    print("executed %d episodes with random actions" % num_rand_episodes)
    print("replay buffer prefilled with %d transitions" % replay_buf.size)
    
    # actually train using actions specified by qnet
    deepq = DeepQ(qnet, target_qnet, replay_buf, LEARNING_RATE, DISCOUNT_FACTOR, POLYAK_FACTOR)
    scores = [] # sum of rewards throughout an episode, used to determine if the agent has solved the environment
    while True:
        score = 0
        env_info = env.reset(train_mode=True)[brain_name]
        prev_action = 0
        state = [*env_info.vector_observations[0], prev_action] # including action id in state
        while True:
            action = deepq.get_action(state) # get action using softmax and random probability
            env_info = env.step(action)[brain_name]
            reward = env_info.rewards[0]
            terminal = 1 if env_info.local_done[0] else 0
            next_state = [*env_info.vector_observations[0], action] # including action id in state
            replay_buf.insert([Transition(state, action, reward, terminal, next_state)])
            deepq.step() # apply gradient step and update target qnet
            state = next_state  # roll over state
            score += reward # accumulate score
            if terminal: # check if episode is done
                break
        # update scores, check for environment being solved
        scores.append(score)
        num_prev_scores = min(100, len(scores))
        avg_score = sum(scores[-num_prev_scores:]) / num_prev_scores
        print("\raverage score for episodes [%d, %d):\t%f" % (len(scores) - num_prev_scores, len(scores), avg_score), end="")
        if avg_score > REQ_AVG_SCORE:
            break

    # save models and plot final rewards curve
    print("\n\nenvironment solved, saving model to qnet.pt")
    save(qnet.state_dict(), "qnet.pt")


if __name__ == "__main__":
    if len(sys.argv != 2):
        print("\nERROR:\tinvalid arguments\nUSAGE:\ttrain.py <unity_environment_executable>\n")
    else:
        train(sys.argv[1])