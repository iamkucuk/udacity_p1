# import numpy as np
# from PIL import Image
#
#
# def load_image(infilename):
#     img = Image.open(infilename)
#     img.load()
#     data = np.asarray(img, dtype="int32")
#     return data
#
#
# def save_image(npdata, outfilename):
#     img = Image.fromarray(np.asarray(np.clip(npdata, 0, 255), dtype="uint8"), "L")
#     img.save(outfilename)
#
#
# data = load_image("banana.png")
#
# outimg = Image.fromarray(data.astype("uint8"), "RGB")
# outimg.show()



from unityagents import UnityEnvironment
import numpy as np


#%%
from dqn_utils import train_dqn
env = UnityEnvironment(base_port=5003, file_name="Banana_Windows_x86_64/Banana.exe")


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

#%%

# env.close()

#%% md

### 4. It's Your Turn!

#%%

from agent import Agent

agent = Agent(state_size=state_size, action_size=action_size, seed=0)

#%%


#%%

from tqdm.auto import tqdm, trange
from collections import deque

scores = train_dqn(env, agent, brain_name=brain_name)

#%%



#%%


