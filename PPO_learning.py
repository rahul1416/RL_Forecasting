import numpy as np
import gym
from gym_environment import TradingEnvironment
from ray.tune.registry import register_env

prices = [1, 1100, 105, 115, 120, 10, 592, 216, 210, 215]  # Replace this with your future price values
probabilities = [0.4, 0.3, 0.25, 0.15, 0.1, 0.1, 0.01, 0.01, 0.4]  # Replace with corresponding probabilities

env = TradingEnvironment(prices,probabilities)
register_env("my_custom_env", lambda config: TradingEnvironment(prices,probabilities))

# Define a simple neural network for policy approximation
class PolicyNetwork:
    def __init__(self, input_size, output_size, learning_rate=0.01):
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.zeros(output_size)
        self.learning_rate = learning_rate

    def forward(self, state):
        return np.dot(state, self.weights) + self.bias

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0, keepdims=True)

    def predict(self, state):
        logits = self.forward(state)
        return self.softmax(logits)

    def train(self, states, actions, advantages):
        num_samples = len(states)

        # One-hot encode actions
        action_masks = np.eye(self.weights.shape[1])[actions]

        # Compute gradient of the policy
        policy_gradient = np.dot(states.T, action_masks - self.predict(states))

        # Update weights and bias using policy gradient and advantages
        self.weights += self.learning_rate * policy_gradient * advantages / num_samples
        self.bias += self.learning_rate * np.sum(action_masks - self.predict(states), axis=0) / num_samples

# Proximal Policy Optimization (PPO) algorithm
def ppo(env_name, epochs=100, max_steps=1000, gamma=0.99, epsilon=0.2, learning_rate=0.01):
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy_network = PolicyNetwork(input_size=state_size, output_size=action_size, learning_rate=learning_rate)

    for epoch in range(epochs):
        states = []
        actions = []
        rewards = []
        advantages = []

        for _ in range(max_steps):
            state = env.reset()
            episode_states = []
            episode_actions = []
            episode_rewards = []

            for _ in range(max_steps):
                action_prob = policy_network.predict(state)
                action = np.random.choice(action_size, p=action_prob)

                next_state, reward, done, _ = env.step(action)

                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)

                state = next_state

                if done:
                    break

            states.extend(episode_states)
            actions.extend(episode_actions)
            rewards.extend(episode_rewards)

            # Calculate advantages using rewards-to-go
            discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
            running_add = 0
            for t in reversed(range(len(rewards))):
                running_add = running_add * gamma + rewards[t]
                discounted_rewards[t] = running_add

            # Standardize advantages
            advantages.extend(discounted_rewards - np.mean(discounted_rewards))
            
        states = np.array(states)
        actions = np.array(actions)
        advantages = np.array(advantages)

        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Train policy network
        policy_network.train(states, actions, advantages)

        # Print total reward for the epoch
        total_reward = sum(rewards)
        print(f"Epoch {epoch + 1}, Total Reward: {total_reward}")

    env.close()

# Example usage
ppo("my_custom_env", epochs=50)
