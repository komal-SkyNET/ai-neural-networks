import numpy as np
import tensorflow as tf
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
parser = argparse.ArgumentParser(description='Train DEEP Q network ')
parser.add_argument('--gpu', type=bool, help='Use GPU?', default=False)
parser.add_argument('--load-from-file', type=str, help='Load model from file')
args = parser.parse_args()
print(args.gpu, args)
from wrapper import ChannelsFirstImageShape, FrameStack, ProcessFrame84
tf.compat.v1.disable_eager_execution()
DISCOUNT = 0.95
REPLAY_MEMORY_SIZE = 200000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 500  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 32  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 4  # Terminal states (end of episodes)
MODEL_NAME = 'Breakout'
MIN_REWARD = 0  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 300000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = True
env = gym.make('BreakoutDeterministic-v4')
env = ProcessFrame84(env)
if args.gpu == True:
    env = ChannelsFirstImageShape(env)
env = FrameStack(env, 4)

ep_rewards = [0]

random.seed(1)
np.random.seed(1)
tf.compat.v1.set_random_seed(1)

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

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
        self.tensorboard = TensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

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

    def create_model_alt_1(self):
        model = Sequential()
        model.add(Conv2D(32,8,
                            strides=(4, 4),
                            padding="valid",
                            activation="relu",
                            input_shape=env.observation_space.shape,
                            data_format="channels_first" if args.gpu else "channels_last"))
        model.add(Conv2D(64,4,
                            strides=(2, 2),
                            padding="valid",
                            activation="relu",
                            input_shape=env.observation_space.shape,
                            data_format="channels_first" if args.gpu else "channels_last"))
        model.add(Conv2D(64,
                            3,
                            strides=(1, 1),
                            padding="valid",
                            activation="relu",
                            input_shape=env.observation_space.shape,
                            data_format="channels_first" if args.gpu else "channels_last"))
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(env.action_space.n, activation='linear'))
        model.compile(loss="mean_squared_error",
                        optimizer=RMSprop(lr=0.00025,
                                            rho=0.95,
                                            epsilon=0.01),
                        metrics=["accuracy"])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):
        # Start training only if certain number of samples is already saved
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
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(np.array(current_state))
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255.0, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        if terminal_state:
            self.target_update_counter += 1
            
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
    agent.model.save('models/breakout-{}.model'.format(int(time.time())), save_format='h5')
    exit(0)

signal.signal(signal.SIGINT, save)
# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    # Update tensorboard step every episode
    agent.tensorboard.step = episode
    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1
    lives = 5
    # Reset environment and get initial state
    current_state = env.reset()
    # Reset flag and start iterating until episode ends
    done = False
    while not done:
        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, info = env.step(action)
        if not done and info['ale.lives'] < lives:
            #fire to drop the ball
            lives = info['ale.lives']
            obs, rew, done, info = env.step(1)
        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            pass
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        tf.summary.scalar('reward_avg', average_reward, step)
        tf.summary.scalar('reward_min', min_reward, step)
        tf.summary.scalar('reward_max', max_reward, step)
        tf.summary.scalar('epsilon', epsilon, step)
        print("reward_avg={}, reward_min={}, reward_max={}, epsilon={}".format(average_reward,
        min_reward, max_reward, epsilon))
        # Save model, but only when min reward is greater or equal a set value
        if min_reward > MIN_REWARD:
            print('min_reward>old_min_reward saving model...')
            agent.model.save('models/{}__{}max_{}avg_{}min__{}.model'.format(MODEL_NAME,
            max_reward, average_reward,
            min_reward,
            int(time.time())), save_format='h5')
            MIN_REWARD = min_reward
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
save(None, None)
