[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Dependencies
 - Python 3.6
 - PyTorch 0.4.0
 - Unity Agents
 - Numpy
 - Matplotlib
 - Tqdm
 
 (*You just need to have Python 3.6. Rest is being installed from the Navigation.ipynb file.*) 
### Instructions

This work consists of 4 essential scripts: 
 - agent.py: Consists of agent behaviour and related functions
 - dqn_utils.py: Consists of training procedures and other future training related utils functions
 - model.py: Consists of the neural network employed within the agent
 - Navigation.ipynb: Where all parts come together
 
 **Please note that you need a proper Unity environment for your PC. If you do not have the environment, please download the one fits for your system from the **Getting Started** section.**
 
##### How to run the code
 - Create an environment. Example anaconda environment creation:
 ```
 conda create -n project_bananas python==3.6
 conda activate project_bananas
 ```
 - Open the **Navigation.ipynb** file.
 - If it's your first run with your freshly created environment, run the first cell. You can skip this one if it isn't your first run with the code.
 - Run rest of the cells.
 
Further report, obtained results, future improvements and ideas can be seen in **Report.html** file.

HAVE A NICE ONE!