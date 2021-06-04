# Udacity Deep Reinforcement Learning Project 1 Submission

## Project Details

The goal of this project was to use reinforcement learning techniques to train  
an agent to navigate and collect bananas in a provided UnityEnvironment  
following the architecture of the open-source Unity plugin **Unity Machine  
Learning Agents (ML-Agents)**. The Banana Collector environment is provided  
by Udacity, and is a modified version of one of the environments in the  
ML-Agents [GitHub repository](https://github.com/Unity-Technologies/ml-agents).

In this Banana Collector environment, agents are provided a reward of +1 for  
collecting a yellow banana while a blue banana yields a reward of -1. The goal  
of an agent in this environment is to maximize its rewards over the course of  
an episode, thus it attempts to collect as many yellow bananas as possible while  
avoiding blue bananas.

The state space that the agent perceives to perform actions in the environment  
has 37 dimensions and contains the agent's velocity, along with ray-based  
perceptions of objects around the agent's forward direction. There are four  
discrete actions available for the agent to select at each timestep:
* `0` - move forward
* `1` - move backward
* `2` - turn left
* `3` - turn right  

This task is episodic, and the environment is considered "solved" upon the agent  
earning an average score of +13 over 100 consecutive episodes.

## Repository Contents

| directory | contents |
| ----------| -------- |
| model | saved model weights (qnet.pt) |
| src | python source files used to train model |

| source file | description |
| ----------- | ----------- |
| deepq.py | Deep Q-Learning class (gradient step and action inference) |
| nn.py | Neural network utility functions (model definition, polyak update) |
| replay.py | Experience replay buffer and (S, A, R, T, S') transition classes
| train.py | Main function for training an agent to solve a UnityEnvironment |

`Report.md` contains the project report that is required for the submission of  
this project, and contains more detailed descriptions of the methods used in the  
source files listed above.

`src/train.py` utilizes capitalized constants defined at the top of `train(...)` and  
accepts a path to a UnityEnvironment executable so while its hyperparameters and  
required average score are tuned for the environment used in this project  
specifically, it is designed to be generalizable. The executable path is taken as  
a command line argument when run as the main script. This is done to prevent the  
need to modify the file when running on various machines where the executable  
path will naturally differ.

## Getting Started

### Step 1: Clone the DRLND Repository
Follow the [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) provided by Udacity to  
set up your Python environment. These instructions are found in the repository's  
own `README.md` file. For this project submission, **steps 2, 4, 5 are not required**  
as OpenAI gym is not utilized and neither are any IPython notebooks.  
**The required steps 1, 3** will install PyTorch, the ML-Agents toolkit, and other  
dependencies.

## Step 2: Download the Unity Environment
You will **not** need to install Unity; only the executable itself is required.  
You will need to select the environment binary that matches your operating system:
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

These links are provided and maintained by Udacity.  
You can unzip this executable wherever you wish. As described earlier, `train(...)`  
from `src/train.py` will take the executable path as an argument.

With this, all dependencies will have been installed. To make your PyTorch  
installation specific to a particular CUDA toolkit version `XY.Z`, within Anaconda  
run the following command: 

`conda install pytorch cudatoolkit=XY.Z -c pytorch`

## Instructions

`src/train.py` contains the function `train(...)` that will be used to train an  
agent to solve this environment. This function can of course be invoked from  
within a Python interpreter, or be invoked from the command line in the following  
way from the root of this repository:

`python -m src.train <PATH_TO_UNITY_ENVIRONMENT_BINARY>`

Upon the completion of training, a PyTorch model `qnet.pt` and a CSV file of  
episode scores (cumulative rewards) `scores.csv` will both be written to the  
current working directory.