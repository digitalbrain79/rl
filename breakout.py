import tensorflow as tf
import gym
import numpy as np
import random

from collections import deque
from skimage.transform import resize
from skimage.color import rgb2gray
from functools import reduce

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('mode', 'train', '')

env = gym.make('BreakoutDeterministic-v0')
env._max_episode_steps = 100000

MINIBATCH_SIZE = 32
HISTORY_SIZE = 4
TRAIN_START = 50000
FINAL_EXPLORATION = 0.1
TARGET_UPDATE = 10000
MEMORY_SIZE = 1000000
EXPLORATION = 1000000
START_EXPLORATION = 1.
INPUT = env.observation_space.shape
OUTPUT = env.action_space.n
HEIGHT = 84
WIDTH = 84
LEARNING_RATE = 0.00025
DISCOUNT = 0.99
EPSILON = 0.01
MOMENTUM = 0.95
MAX_EPISODE = 1000000

model_path = 'save'

class DQNAgent:
    def __init__(self, sess, height, width, history_size, output, name='main'):
        self.sess = sess
        self.height = height
        self.width = width
        self.history_size = history_size
        self.output = output
        self.name = name

        self.build_network()

    def build_network(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder('float', [None, self.height, self.width, self.history_size])
            self.Y = tf.placeholder('float', [None])
            self.action = tf.placeholder('int64', [None])

            conv1 = tf.layers.conv2d(self.X, 32, [8, 8], (4, 4), activation=tf.nn.relu, use_bias=False, name='conv1')
            conv2 = tf.layers.conv2d(conv1, 64, [4, 4], (2, 2), activation=tf.nn.relu, use_bias=False, name='conv2')
            conv3 = tf.layers.conv2d(conv2, 64, [3, 3], activation=tf.nn.relu, use_bias=False, name='conv3')
            shape = conv3.get_shape().as_list()
            flat = tf.reshape(conv3, [-1, reduce(lambda x, y: x * y, shape[1:])])
            fc1 = tf.layers.dense(flat, 512, tf.nn.relu, use_bias=False, name='fc1')
            self.q = tf.layers.dense(fc1, OUTPUT, use_bias=False, name='q')

        action_one_hot = tf.one_hot(self.action, self.output, 1.0, 0.0)
        q_val = tf.reduce_sum(tf.multiply(self.q, action_one_hot), reduction_indices=1)
        error = clipped_error(self.Y - q_val)

        self.loss = tf.reduce_mean(error)
        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=EPSILON)
        self.train = optimizer.minimize(self.loss)

        self.saver = tf.train.Saver()

    def predict(self, history):
        return self.sess.run(self.q, feed_dict={self.X: np.reshape(np.float32(history / 255.),
            [-1, HEIGHT, WIDTH, HISTORY_SIZE])})

    def get_action(self, q, e):
        if e > np.random.rand(1):
            action = np.random.randint(self.output)
        else:
            action = np.argmax(q)
        return action

def clipped_error(error):
    return tf.where(tf.abs(error) < 1.0, 0.5 * tf.square(error), tf.abs(error) - 0.5)

def preprocess(X):
    x = np.uint8(resize(rgb2gray(X), (HEIGHT, WIDTH), mode='reflect') * 255)
    return x

def get_copy_var_ops(dest_scope_name='target', src_scope_name='main'):
    op_holder = []

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

def get_init_state(history, s):
    for i in range(HISTORY_SIZE):
        history[:, :, i] = preprocess(s)

def get_game_type(count, info, no_life_game, start_live):
    if count == 1:
        start_live = info['ale.lives']
        if start_live == 0:
            no_life_game = True
        else:
            no_life_game = False
    return [no_life_game, start_live]

def get_terminal(start_live, info, reward, no_life_game, terminal):
    if no_life_game:
        if reward < 0:
            terminal = True
    else:
        if start_live > info['ale.lives']:
            terminal = True
            start_live = info['ale.lives']

    return [terminal, start_live]

def train_minibatch(mainDQN, targetDQN, minibatch):
    state_stack = []
    action_stack = []
    reward_stack = []
    next_state_stack = []
    done_stack = []

    for state, action, reward, done in minibatch:
        state_stack.append(state[:, :, :HISTORY_SIZE])
        action_stack.append(action)
        reward_stack.append(reward)
        next_state_stack.append(state[:, :, 1:])
        done_stack.append(done)

    done_stack = np.array(done_stack) + 0
    Q1 = targetDQN.predict(np.array(next_state_stack))
    y = reward_stack + (1 - done_stack) * DISCOUNT * np.max(Q1, axis=1)

    mainDQN.sess.run(mainDQN.train, feed_dict={mainDQN.X: np.float32(np.array(state_stack) / 255.),
        mainDQN.Y: y, mainDQN.action: action_stack})

def play(mainDQN):
    state = env.reset()
    reward_sum = 0
    same_action_count = 0
    prev_action = 0
    e = 0.
    history = np.zeros([HEIGHT, WIDTH, HISTORY_SIZE], dtype=np.uint8)
    get_init_state(history, state)

    while True:
        env.render()
        Q = mainDQN.predict(history)
        action = mainDQN.get_action(Q, e)
        if prev_action == action:
            same_action_count += 1
            if same_action_count > 100:
                e = 1.
        else:
            same_action_count = 0
            e = 0.
        prev_action = action
        state, reward, done, info = env.step(action)
        history[:, :, :3] = history[:, :, 1:]
        history[:, :, 3] = preprocess(state)
        reward_sum += reward
        if done:
            print('Total score: %d' % reward_sum)
            break

def main():
    episode_op = tf.get_variable('episode', shape=(), dtype=tf.int32, initializer=tf.zeros_initializer())

    with tf.Session() as sess:
        mainDQN = DQNAgent(sess, HEIGHT, WIDTH, HISTORY_SIZE, OUTPUT, name='main')
        targetDQN = DQNAgent(sess, HEIGHT, WIDTH, HISTORY_SIZE, OUTPUT, name='target')

        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            mainDQN.saver.restore(sess, ckpt.model_checkpoint_path)
            print('Model restored: ', ckpt.model_checkpoint_path)

        if FLAGS.mode == 'play':
            play(mainDQN)
            return

        copy_ops = get_copy_var_ops(dest_scope_name='target', src_scope_name='main')
        sess.run(copy_ops)
        episode = sess.run(episode_op)
        episode = 0

        recent_reward = deque(maxlen=100)
        e = 1.
        frame = 0
        save_model = False
        average_Q = deque()
        no_life_game = False
        replay_memory = deque(maxlen=MEMORY_SIZE)

        while episode < MAX_EPISODE:
            episode += 1
            history = np.zeros([HEIGHT, WIDTH, HISTORY_SIZE + 1], dtype=np.uint8)
            reward_all, count = 0, 0
            done = False
            state = env.reset()

            get_init_state(history, state)

            while not done:
                frame += 1
                count += 1

                if e > FINAL_EXPLORATION and frame > TRAIN_START:
                    e -= (START_EXPLORATION - FINAL_EXPLORATION) / EXPLORATION

                Q = mainDQN.predict(history[:, :, :HISTORY_SIZE])
                average_Q.append(np.max(Q))

                action = mainDQN.get_action(Q, e)

                next_state, reward, done, info = env.step(action)
                terminal = done
                reward = np.clip(reward, -1, 1)

                no_life_game, start_lives = get_game_type(count, info, no_life_game, done)

                terminal, start_lives = get_terminal(start_lives, info, reward, no_life_game, terminal)

                history[:, :, HISTORY_SIZE] = preprocess(next_state)

                replay_memory.append((np.copy(history[:, :, :]), action, reward, terminal))
                history[:, :, :HISTORY_SIZE] = history[:, :, 1:]

                reward_all += reward

                if frame > TRAIN_START:
                    minibatch = random.sample(replay_memory, MINIBATCH_SIZE)
                    train_minibatch(mainDQN, targetDQN, minibatch)

                    if frame % TARGET_UPDATE == 0:
                        copy_ops = get_copy_var_ops(dest_scope_name='target', src_scope_name='main')
                        sess.run(copy_ops)

                if (frame - TRAIN_START) % TRAIN_START == 0:
                    save_model = True

            recent_reward.append(reward_all)

            print('Episode:{0:6d} | Frames:{1:9d} | Steps:{2:5d} | Reward:{3:3.0f} | e-greedy:{4:.5f} | '
                'Avg_Max_Q:{5:2.5f} | Recent reward:{6:.5f}'.format(episode, frame, count, reward_all, e,
                np.mean(average_Q), np.mean(recent_reward)))

            if save_model:
                sess.run(episode_op.assign(episode))
                save_path = mainDQN.saver.save(mainDQN.sess, model_path + '/model.ckpt', global_step=episode)
                print('Model(episode: %d) saved in file: %s' % (episode, save_path))
                save_model = False
                average_Q = deque()

if __name__ == '__main__':
    main()
