""" Line 1 - 19 was for this to work on GCP CPU """
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus

import numpy as np
import keras
import tensorflow.keras.backend as backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Activation, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.activations import softmax
from tensorflow.keras.callbacks import TensorBoard
from collections import deque
import time, gym
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
import signal
import sys
import argparse
from wrapper import ChannelsFirstImageShape, FrameStack, ProcessFrame84
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#tf.compat.v1.enable_eager_execution()
#tf.enable_eager_execution()
DISCOUNT = 0.95
REPLAY_MEMORY_SIZE = 200000  # last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 500  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 32  # (samples) to use for training
UPDATE_TARGET_EVERY = 4  
MODEL_NAME = 'Breakout'
MIN_REWARD = 0  # For model save
MEMORY_FRACTION = 0.20
# Environment settings
EPISODES = 500000
# Exploration settings
epsilon = 0.4  # will be decayed
EPSILON_DECAY = 0.999985
MIN_EPSILON = 0.001
#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = True

"""
class PreProcessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PreProcessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=1,
                                                shape=(80*4, 86, 1), dtype=np.int8)
        self.frames = deque(maxlen=4)
        init_frames = np.zeros((3, 80, 86, 1)).astype(np.int8)
        self.frames.append(i) for i in init_frames

    def observation(self, obs):
        obs = obs[25:]
        obs = obs[:-12]
        inx,iny,inc = obs.shape
        inx = int(inx*0.5)
        iny = int(iny*0.5)
        obs = cv2.resize(obs,(inx,iny)) # Ob is the current frame
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY) # convert to grayscale
        # obs = np.array(obs).astype(np.float32)
        obs = np.expand_dims(obs, axis=2)
        # obs = np.array(obs)/255.0
        # obs = preprocess(downsample(obs))
        # obs = np.reshape(obs,(inx,iny))
        # obs = np.ndarray.flatten(obs)
        return list(np.concatenate(self.frames, obs))

class SkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=2):
        super(SkipEnv, self).__init__(env)
        self._skip = skip

    def step(self, action):
        t_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            if done:
                break
        return obs, t_reward, done, info
"""


env = gym.make('BreakoutDeterministic-v4')
env = ProcessFrame84(env)
env = ChannelsFirstImageShape(env)
env = FrameStack(env, 4)

ep_rewards = [0]

random.seed(1)
np.random.seed(1)
#tf.set_random_seed(1)

if not os.path.isdir('models'):
    os.makedirs('models')

parser = argparse.ArgumentParser(description='Train DEEP Q network ')
parser.add_argument('--load-from-file', type=str, help='Load model from file')
args = parser.parse_args()

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        #self.writer = tf.summary.create_file_writer(self.log_dir) #tf.summary.FileWriter(self.log_dir)
        self.writer = tf.summary.FileWriter(self.log_dir)
    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)
        #self.writer.add_summary(stats, self.step)

    def _write_logs(self, logs, index):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            if isinstance(value, np.ndarray):
                summary_value.simple_value = value.item()
            else:
                summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, index)
        self.writer.flush()

# Agent class
class DQNAgent:

    def __init__(self):
        # Main model
        self.model = self.create_model_alt_1()
        # Target network
        self.target_model = self.create_model_alt_1()
        self.target_model.set_weights(self.model.get_weights())
        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))
        # count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Dense(256, input_shape=env.observation_space.shape))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Activation('relu'))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(env.action_space.n, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # This is the model that worked finally. This was on GCP CPU p100 accelerator - 
    # GPU input NCHW | CPU input NHWC
    def create_model_alt_1(self):
        model = Sequential()
        model.add(Conv2D(32,8,
                              strides=(4, 4),
                              padding="valid",
                              activation="relu",
                              input_shape=env.observation_space.shape,
                              data_format="channels_first"))
        model.add(Conv2D(64,4,
                              strides=(2, 2),
                              padding="valid",
                              activation="relu",
                              input_shape=env.observation_space.shape,
                              data_format="channels_first"))
        model.add(Conv2D(64,
                              3,
                              strides=(1, 1),
                              padding="valid",
                              activation="relu",
                              input_shape=env.observation_space.shape,
                              data_format="channels_first"))
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(env.action_space.n, activation='linear'))
        model.compile(loss="mean_squared_error",
                           optimizer=RMSprop(lr=0.00025,
                                             rho=0.95,
                                             epsilon=0.01),
                           metrics=["accuracy"])
        return model

    # Adds step's data to a replay memory
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):
        # Start training only if certain number of samples is ready
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([np.array(transition[0]) for transition in minibatch])
        current_qs_list = self.model.predict(current_states/255.0)
        # Get future states from minibatch, then query NN model for Q values
        new_current_states = np.array([np.array(transition[3]) for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states/255.0)
        X = []
        y = []
        # enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            # append to training data
            X.append(np.array(current_state))
            y.append(current_qs)
        # Fit on all samples as one batch, log only on terminal state
        # divide /255 only while passing because we don't want to store it that way to save memory- 
        # i.e, have input as int8 array 
        self.model.fit(np.array(X)/255.0, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1
        # update target network with weights of main if counter reached 
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        state = np.array(state)
        state = state/255.0
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

agent = DQNAgent()
if args.load_from_file:
    print('loading model from file: {}'.format(args.load_from_file))
    agent.model = tf.keras.models.load_model('{}'.format(args.load_from_file))
    agent.target_model = tf.keras.models.load_model('{}'.format(args.load_from_file))
    print('weights restored successfully..')

print(agent.model.summary())

def save(sig, fr):
    print('saving model & exiting')
    agent.model.save('models/breakout-{}.model'.format(int(time.time())))
    exit(0)

signal.signal(signal.SIGINT, save)

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    # Update tensorboard step every episode
    agent.tensorboard.step = episode
    episode_reward = 0
    step = 1
    lives = 5
    current_state = env.reset()
    done = False
    while not done:
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)
        new_state, reward, done, info = env.step(action)
        episode_reward += reward
        if not done and info['ale.lives'] < lives:
            #fire to drop the ball
            lives = info['ale.lives']
            obs, rew, done, info = env.step(1)
        if episode == 1 or (SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY):
            # env.render()
            pass
        # Every step - update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1
    # Append episode reward to a list and log stats
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
        print("reward_avg={}, reward_min={}, reward_max={}, epsilon={}".format(average_reward,
        min_reward, max_reward, epsilon))
        # Save model when min reward is greater or equal a set value
        if min_reward > MIN_REWARD:
            print('min_reward>old_min_reward saving model...')
            agent.model.save('models/{}__{}max_{}avg_{}min__{}.model'.format(MODEL_NAME,
            max_reward, average_reward,
            min_reward,
            int(time.time())))
            MIN_REWARD = min_reward
    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

save(None, None)
