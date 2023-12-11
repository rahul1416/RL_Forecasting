import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from Custum_environment import TradingEnvironment
# TradingEnvironment class (as provided in your previous messages)

# Hyperparameters
input_size = 3  # Size of the state space (prices, balance, shares_held)
output_size = 3  # Buy, Sell, Hold
action = 0
prices = [1, 1100, 105, 115, 120, 10, 592, 216, 210, 215]  # Replace this with your future price values
probabilities = [0.4, 0.3, 0.25, 0.15, 0.1, 0.1, 0.01, 0.01, 0.4]  # Replace with corresponding probabilities

# Create the trading environment and the model
env = TradingEnvironment(prices, probabilities)
model = Sequential([
    Dense(64, input_dim=input_size, activation='relu'),
    Dense(32, activation='relu'),
    Dense(output_size, activation='linear')
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Training parameters
episodes = 10
epsilon = 0.1  # Exploration-exploitation trade-off
gamma = 0.95  # Discount factor

# Training loop
for episode in range(episodes):
    state = env.get_state()
    state = np.reshape(state, [1, input_size])

    for step in range(len(env.prices) - 1):
        # Epsilon-greedy exploration
        if np.random.rand() <= epsilon:
            if action == 0:
                action = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])  # Replace with your RL agent's action
            elif action == 1:
                action = np.random.choice([-1, 0])
            elif action == -1:
                action = np.random.choice([1, 0])
        else:
            q_values = model.predict(state)
            if action == 0:
                action = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])  # Replace with your RL agent's action
            elif action == 1:
                action = np.random.choice([-1, 0])
            elif action == -1:
                action = np.random.choice([1, 0])

        # Take the chosen action and observe the new state and reward
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, [1, input_size])

        # Q-value update using the Bellman equation
        target = reward + gamma * np.max(model.predict(next_state))
        q_values = model.predict(state)
        q_values[0][action] = target

        # Train the model on the updated Q-values
        model.fit(state, q_values, epochs=100, verbose=0)

        # Move to the next time step
        state = next_state

    # Reset the environment at the end of each episode
    env.reset()

# Save the trained model for later use
model.save("Qlearning_trading_model.h5")
