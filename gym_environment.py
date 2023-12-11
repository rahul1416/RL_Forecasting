import gym
from gym import spaces
import numpy as np

class TradingEnvironment(gym.Env):
    def __init__(self, prices, probabilities):
        super(TradingEnvironment, self).__init__()
        self.prices = prices
        self.probabilities = probabilities
        self.current_step = 0
        self.balance = 1000
        self.shares_held = 0
        self.max_steps = len(prices) - 1
        self.can_buy = True

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # Buy (1), Sell (2), Hold (0)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = 1000
        self.shares_held = 0
        self.can_buy = True
        return self.get_state()

    def get_state(self):
        return np.array([self.prices[self.current_step], self.balance, self.shares_held])

    def take_action(self, action):
        current_price = self.prices[self.current_step]
        if action == 1:  # Buy
            if self.can_buy and self.balance >= current_price:
                shares_to_buy = self.balance / current_price
                self.shares_held += shares_to_buy
                self.balance -= shares_to_buy * current_price
                self.can_buy = False
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.shares_held = 0
                self.can_buy = True

    def step(self, action):
        self.take_action(action)
        self.current_step += 1

        if self.current_step >= self.max_steps:
            done = True
        else:
            done = False

        next_state = self.get_state()
        current_price = self.prices[self.current_step - 1]

        reward = 0
        if self.current_step == self.max_steps:
            final_balance = self.balance + self.shares_held * self.prices[-1]
            reward = final_balance - self.balance - current_price * self.shares_held
            done = True

        return next_state, reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Shares Held: {self.shares_held}")

    def close(self):
        pass

# Example usage
prices = [100, 110, 105, 115, 120, 110, 512, 216, 210, 215]
probabilities = [0.4, 0.3, 0.25, 0.15, 0.1, 0.1, 0.01, 0.01, 0.4]

env = TradingEnvironment(prices, probabilities)
print(env)
# action = 0
# for _ in range(len(prices)):
#     if action == 0:
#         action = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])  # Replace with your RL agent's action
#     elif action == 1:
#         action = np.random.choice([-1, 0])
#     elif action == -1:
#         action = np.random.choice([1, 0])

#     next_state, reward, done = env.step(action)
#     if done:
#         break

# # Retrieve final state, reward, balance, shares held
# final_state = env.get_state()
# final_balance = env.balance
# final_shares_held = env.shares_held
# print("Final State:", final_state)
# # print("Final Balance:", final_balance)
# print("Final Shares Held:", final_shares_held)
# print("action:", action)
# print("reward",reward)