import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from engine.three import PoorMansGymEnv

ENV_NAME='three'

env = PoorMansGymEnv(stop_on_invalid_move = True)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

window = 8

model = Sequential()
model.add(Flatten(input_shape=(window,16)))
model.add(Dense(16, init='random_uniform', activation='relu'))
model.add(Dense(16, init='random_uniform', activation='relu'))
model.add(Dense(16, init='random_uniform', activation='relu'))
model.add(Dense(nb_actions, init='random_uniform', activation='sigmoid'))
print(model.summary())


# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=window)
args = {
    'model':               model,
    'nb_actions':          nb_actions,
    'memory':              memory,
    'nb_steps_warmup':     100,
    'batch_size':          512,
    # 'batch_size':          32,
    'target_model_update': 1e-2,
    'policy':              BoltzmannQPolicy(),
}

dqn = DQNAgent(**args)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=50000000, visualize=False, verbose=1)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=False)
