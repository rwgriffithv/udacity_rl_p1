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
    LEARNING_RATE = 0.001 # small due to frequency of gradient steps
    DISCOUNT_FACTOR = 1
    POLYAK_FACTOR = 0.999 # very large due to frequency of gradient steps
    NUM_GRAD_STEPS_PER_UPDATE = 1
    BATCH_SIZE = 256
    K = 3 # number of simulation steps per RL algorithm step
    EPSILON_MIN = 0.1
    EPSILON_MAX = 1.0
    EPSILON_DECAY = 0.99
    # epsilon refreshing to encourage exploration after standard epsilon annealing
    EPSILON_REFRESH = 0.5 # for refreshing the value of epsilon
    STAGNANT_EPS_TO_REFRESH = 50 # number of sequential stagnant episodes that prompts an epsilon refresh
    SCORE_DECREASE_TO_REFRESH = 0.5 # average score decrease that prompts an epsilon refresh
    
    # instantiate environment
    env = UnityEnvironment(file_name=banana_bin_path)
    # get default brain name
    brain_name = env.brain_names[0]
    
    # get environment state and action size
    env_info = env.reset(train_mode=True)[brain_name]
    state_size = len(env_info.vector_observations[0])
    action_size = env.brains[brain_name].vector_action_space_size
    
    # build sequential artificial neural networks for the q-function and target q-function
    qnet = build_network(state_size, action_size)
    target_qnet = build_network(state_size, action_size) # target q network

    # states stored in replay buffer are environment states concatenated with previous action id
    replay_buf = ReplayBuffer(REPBUF_CAPCITY, state_size)

    # training using Deep Q-Learning
    deepq = DeepQ(qnet, target_qnet, replay_buf, LEARNING_RATE, DISCOUNT_FACTOR, POLYAK_FACTOR)
    scores = [] # sum of rewards throughout an episode, used to determine if the agent has solved the environment
    epsilon = EPSILON_MAX
    stagnation_count = 0
    max_score = int(-1e6)
    print("\n\ntraining....")
    while True:
        score = 0
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        while True:
            action = deepq.get_action(state, epsilon)
            reward = 0
            for _ in range(K):
                env_info = env.step(action)[brain_name]
                reward += env_info.rewards[0]
                terminal = 1 if env_info.local_done[0] else 0
                if terminal: # check if episode is done
                    break
            next_state = env_info.vector_observations[0]
            replay_buf.insert([Transition(state, action, reward, 0, next_state)])
            deepq.optimize(NUM_GRAD_STEPS_PER_UPDATE, BATCH_SIZE)
            state = next_state  # roll over state
            score += reward # accumulate score
            if terminal:
                break
        # update scores
        max_score = max(score, max_score)
        scores.append(score)
        # check for stagnation, update epsilon
        score_diff = max_score - score
        stagnation_count = 0 if score_diff == 0 else stagnation_count + 1
        if epsilon == EPSILON_MIN and (stagnation_count == STAGNANT_EPS_TO_REFRESH or score_diff >= SCORE_DECREASE_TO_REFRESH):
            epsilon = EPSILON_REFRESH
            stagnation_count = 0
        else:
            epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)
        # check for environment being solved
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