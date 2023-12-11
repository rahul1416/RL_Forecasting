import numpy as np

class TradingEnvironment:
    def __init__(self, prices, probabilities):
        self.prices = prices
        self.probabilities = probabilities
        self.current_step = 0
        self.balance = 1000  # Initial balance
        self.shares_held = 0
        self.max_steps = len(prices)-1
        
    def reset(self):
        self.current_step = 0
        self.balance = 1000
        self.shares_held = 0
        
    def get_state(self):
        return (self.prices[self.current_step], self.balance, self.shares_held)
    
    def take_action(self, action):
        current_price = self.prices[self.current_step]
        if action == 1:  # Buy
            if self.balance >= current_price:
                # Buy as many shares as possible with the available balance
                shares_to_buy = self.balance / current_price
                self.shares_held += shares_to_buy
                self.balance -= shares_to_buy * current_price
        elif action == -1:  # Sell
            if self.shares_held > 0:
                # Sell all shares held at once
                self.balance += self.shares_held * current_price
                self.shares_held = 0
        
    def step(self, action):
        self.take_action(action)
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        else:
            done = False
        
        next_state = self.get_state()
        current_price = self.prices[self.current_step - 1]  # Get the current price

        # Print price and stock values at each step
        print(f"Step: {self.current_step}, Price: {current_price}, Balance: {self.balance}, Shares Held: {self.shares_held}, action:{action}")


        reward = 0
        if self.current_step == self.max_steps:
            final_balance = self.balance + self.shares_held * self.prices[-1]
            if (self.balance == 0 and action == 1 ) and (self.shares_held == 0 and action == -1):
                reward = -e^(-10)
            else:
                reward = final_balance -  1000  # Initial balance
            done = True
            print("Final_balance:",final_balance)
        
        return next_state, reward, done


# Example usage
prices = [100, 110, 105, 115, 120, 110, 512, 216, 210, 215]  # Replace this with your future price values
probabilities = [0.4, 0.3, 0.25, 0.15, 0.1, 0.1, 0.01, 0.01, 0.4]  # Replace with corresponding probabilities

env = TradingEnvironment(prices, probabilities)

# Example episode
env.reset()
action = 0
for _ in range(len(prices)):
    if action == 0:
        action = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])  # Replace with your RL agent's action
    elif action == 1:
        action = np.random.choice([-1, 0])
    elif action == -1:
        action = np.random.choice([1, 0])

    next_state, reward, done = env.step(action)
    if done:
        break

# Retrieve final state, reward, balance, shares held
final_state = env.get_state()
final_balance = env.balance
final_shares_held = env.shares_held
print("Final State:", final_state)
# print("Final Balance:", final_balance)
print("Final Shares Held:", final_shares_held)
print("action:", action)
print("reward",reward)

