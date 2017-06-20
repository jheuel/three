from __future__ import division
import argparse

from PIL import Image
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, Permute, Dropout
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from engine.three import PoorMansGymEnv

INPUT_SHAPE = (4, 4)
WINDOW_LENGTH = 1


# class AtariProcessor(Processor):
    # def process_observation(self, observation):
        # # assert observation.ndim == 3  # (height, width, channel)
        # # img = Image.fromarray(observation)
        # # img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        # processed_observation = np.array(observation)
        # # assert processed_observation.shape == INPUT_SHAPE
        # return processed_observation.astype(
            # 'uint8')  # saves storage in experience memory

    # def process_state_batch(self, batch):
        # # We could perform this processing step in `process_observation`. In this case, however,
        # # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # # an `uint8` array. This matters if we store 1M observations.
        # processed_batch = batch.astype('float32') / 255.
        # return processed_batch

    # def process_reward(self, reward):
        # return np.clip(reward, -1., 1.)


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='BreakoutDeterministic-v3')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('-l', '--load_weights', default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
# env = gym.make(args.env_name)
env = PoorMansGymEnv(mocked_game=True, square_grid=True)
np.random.seed(123)
nb_actions = env.action_space.n

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
input_shape = (WINDOW_LENGTH, ) + INPUT_SHAPE
model = Sequential()
# (width, height, channels)
model.add(Permute((2, 3, 1), input_shape=input_shape))
model.add(Convolution2D(256, (2,1), strides=1, padding='same'))#, input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(256, (1,2), strides=1, padding='same'))#, input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(512, (2,1), strides=1, padding='same'))#, input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(512, (1,2), strides=1, padding='same'))#, input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(512, (2,1), strides=1, padding='same'))#, input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(512, (1,2), strides=1, padding='same'))#, input_shape=input_shape))
model.add(Activation('relu'))
# model.add(Dropout(0.1))
# model.add(MaxPooling2D(pool_size=2))#, strides=1, padding='same'))
# model.add(Dropout(0.1))
# model.add(MaxPooling2D(pool_size=2, padding='same'))
# model.add(Convolution2D(16, (2,2)))
# model.add(Activation('relu'))
# model.add(Dropout(0.1))
# model.add(MaxPooling2D(pool_size=1, padding='same'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
# model.add(Dropout(0.1))
# model.add(Dense(100))
# model.add(Activation('relu'))
# model.add(Dropout(0.1))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=int(5e6), window_length=WINDOW_LENGTH)
# processor = AtariProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(
    EpsGreedyQPolicy(),
    attr='eps',
    value_max=1.,
    value_min=.1,
    value_test=.05,
    nb_steps=1e6)

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy()
# Feel free to give it a try!

warmup_steps = 100000
if args.load_weights:
    warmup_steps = 0
    # policy = LinearAnnealedPolicy(
        # EpsGreedyQPolicy(),
        # attr='eps',
        # value_max=1.00,
        # value_min=0.01,
        # value_test=.05,
        # nb_steps=1e6)


dqn = DQNAgent(
    model=model,
    nb_actions=nb_actions,
    policy=policy,
    memory=memory,
    # processor=processor,
    nb_steps_warmup=warmup_steps,
    gamma=.9,
    target_model_update=1e-3,
    # target_model_update=3000,
    # train_interval=4,
    delta_clip=1
)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

if args.load_weights:
    weights_filename = args.load_weights
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)

if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [
        ModelIntervalCheckpoint(checkpoint_weights_filename, interval=25000)
    ]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=175e6, log_interval=10000, verbose=1)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

elif args.mode == 'test':
    env = PoorMansGymEnv(square_grid=True)
    dqn.test(env, nb_episodes=10, visualize=False)
