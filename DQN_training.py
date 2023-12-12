import tensorflow as tf
from tensorflow.keras import layers, models
from Custum_environment import TradingEnvironment
import numpy as np
import random
class DQN(tf.keras.Model):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(num_actions, activation='linear')
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)
class DQNAgent:
    def __init__(self, num_actions, num_states):
        self.num_actions = num_actions
        self.num_states = num_states
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 8
        self.memory = []  # Experience replay memory
        self.model = DQN(num_actions)
        self.target_model = DQN(num_actions)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.update_target_network()
        self.model.compile(optimizer=self.optimizer, loss='mse')
    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def replay(self):
        for experience in self.memory:
            state, action, reward, next_state, done = experience
            print("state:",state,"experiance:",experience)
            if state is None or next_state is None:
                continue
            target = self.model.predict(np.array([state]))[0]
            if done:
                target[action] = reward
            else:
                next_state_target = self.target_model.predict(np.array([next_state]))[0]
                next_best_action = np.argmax(self.model.predict(np.array([next_state]))[0])
                target[action] = reward + self.gamma * next_state_target[next_best_action]
            self.model.train_on_batch(np.array([state]), np.array([target]))
    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice([-1,0,1])
        else:
            q_values = self.model.predict(np.array([state]))[0]
            return np.argmax(q_values)-1
    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            while True:
                if state == None:
                    state = env.get_state()
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                # state = tf.convert_to_tensor(env.get_state(), dtype=tf.float32)
                total_reward += reward
                if done == True:
                    break
            if len(self.memory) > self.batch_size:
                self.replay()
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            self.update_target_network()
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")


prices = [10,20,30,40,50,90,30,39,29]
env = TradingEnvironment(prices)

# Create an instance of the DQNAgent
num_actions = 3  # Sell, Hold, Buy
num_states = len(prices)
agent = DQNAgent(num_actions, num_states)
# state = tf.convert_to_tensor(env.get_state(), dtype=tf.float32)

# Train the DQN model
num_episodes = 10  # Number of episodes for training
agent.train(env, num_episodes)