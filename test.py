# Create a test environment
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from Custum_environment import TradingEnvironment
from tensorflow.keras.models import load_model

# Load the trained model
prices_test = [100, 110, 105, 115, 120, 10, 590, 216, 210, 215]  # Replace this with your future price values
probabilities_test = [0.4, 0.3, 0.25, 0.15, 0.1, 0.1, 0.01, 0.01, 0.4]
model = load_model("trading_model.h5")

# Create a test environment
test_env = TradingEnvironment(prices_test, probabilities_test)  # Replace with your test data
input_size = 3
# Test the model
state = test_env.get_state()
state = np.reshape(state, [1, input_size])

total_reward = 0
done = False

while not done:
    # Use the trained model to predict the Q-values
    q_values = model.predict(state)
    print("Q_value:",q_values)
    # Choose the action with the highest Q-value
    action = np.argmax(q_values)
    print("action:",action)
    # Map the action index back to -1, 0, or 1
    mapped_action = action - 1
    # Take the action in the environment
    next_state, reward, done = test_env.step(mapped_action)
    next_state = np.reshape(next_state, [1, input_size])
    total_reward += reward
    print("next_state:",next_state,"reward:",reward)
    state = next_state

print("Total Reward:", total_reward)

