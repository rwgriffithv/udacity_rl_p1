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
    REPBUF_CAPCITY = int(1e6)
    REPBUF_PREFILL_RATIO = 0.1
    LEARNING_RATE = 0.001
    DISCOUNT_FACTOR = 0.99
    POLYAK_FACTOR = 0.95
    NUM_GRAD_STEPS_PER_UPDATE = 1
    NUM_GRAD_STEPS_INIT = 1000
    BATCH_SIZE = 2000
    
    # instantiate environment
    env = UnityEnvironment(file_name=banana_bin_path)
    # get default brain name
    brain_name = env.brain_names[0]
    
    # get environment state and action size
    env_info = env.reset(train_mode=True)[brain_name]
    env_state_size = len(env_info.vector_observations[0])
    action_size = env.brains[brain_name].vector_action_space_size
    
    # build sequential artificial neural networks for the q-function and target q-function
    state_size = env_state_size + action_size # make states that include previous action one-hot encoding
    qnet = build_network(state_size, action_size)
    target_qnet = build_network(state_size, action_size) # target q network

    # states stored in replay buffer are environment states concatenated with previous action id
    replay_buf = ReplayBuffer(REPBUF_CAPCITY, state_size)

    # prefill replay buffer with transitions collected from taking random actions
    # do not track scores here, qnet is not being used and is not training
    num_rand_episodes = 0
    if REPBUF_PREFILL_RATIO > 0:
        print("\nprefilling replay buffer using random actions...")
    while replay_buf.size < int(REPBUF_PREFILL_RATIO * REPBUF_CAPCITY):
        num_rand_episodes += 1
        env_info = env.reset(train_mode=True)[brain_name]
        state = [*env_info.vector_observations[0]] + ([0] * action_size) # including action one hot encoding in state
        while True:
            action = np.random.randint(action_size)
            env_info = env.step(action)[brain_name]
            reward = env_info.rewards[0]
            terminal = 1 if env_info.local_done[0] else 0
            next_state = [*env_info.vector_observations[0]] + ([0] * action_size) # including action id in state
            next_state[state_size - action_size + action] = 1
            replay_buf.insert([Transition(state, action, reward, terminal, next_state)])
            state = next_state # roll over state
            if terminal: # check if episode is done
                break
        print("\rreplay buffer size [transitions]:\t%d" % replay_buf.size, end="")

    # actually train using actions specified by qnet
    deepq = DeepQ(qnet, target_qnet, replay_buf, LEARNING_RATE, DISCOUNT_FACTOR, POLYAK_FACTOR)
    if REPBUF_PREFILL_RATIO > 0:
        deepq.optimize(NUM_GRAD_STEPS_INIT, BATCH_SIZE) # initial training on random actions
    scores = [] # sum of rewards throughout an episode, used to determine if the agent has solved the environment
    print("\n\ntraining....")
    while True:
        score = 0
        env_info = env.reset(train_mode=True)[brain_name]
        state = [*env_info.vector_observations[0]] + ([0] * action_size)
        while True:
            action = deepq.get_action(state) # get action using softmax and random probability
            env_info = env.step(action)[brain_name]
            reward = env_info.rewards[0]
            terminal = 1 if env_info.local_done[0] else 0
            next_state = [*env_info.vector_observations[0]] + ([0] * action_size)
            next_state[state_size - action_size + action] = 1
            replay_buf.insert([Transition(state, action, reward, terminal, next_state)])
            deepq.optimize(NUM_GRAD_STEPS_PER_UPDATE, BATCH_SIZE)
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

    env.close()
    # save models and plot final rewards curve
    print("\n\nenvironment solved, saving model to qnet.pt and scores to scores.csv")
    with open("scores.csv", "w") as f:
        f.write(str(scores)[0:-1])
    save(qnet.state_dict(), "qnet.pt")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("\nERROR:\tinvalid arguments\nUSAGE:\ttrain.py <unity_environment_executable>\n")
    else:
        train(sys.argv[1])