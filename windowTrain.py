import pandas as pd

# Read the CSV file
file_path = 'BTCUSDT-1m-2018-01.csv'  # Replace with your file path
column_names = ['time', 'open', 'high','low','close','volume','closetime','quote','trades','assest','q_quote','ignore']  # Replace with your actual column names

data = pd.read_csv(file_path,names=column_names)
data.head()
data1 = data['close'].copy()
data = data1.head(500)
data

window_size = 10  # Set the size of the sliding window
stride = 1
for i in range(0, len(data)//5 - window_size + 1, stride):
    window = data.iloc[i:i+window_size]
    
    env = TradingEnvironment(np.array(window))

    # Create an instance of the DQNAgent
    num_actions = 3  # Sell, Hold, Buy
    num_states = len(prices)
    agent = DQNAgent(num_actions, num_states)
    # state = tf.convert_to_tensor(env.get_state(), dtype=tf.float32)
    
    # Train the DQN model
    num_episodes = 2  # Number of episodes for training
    agent.train(env, num_episodes)
    print("Window:", i // stride + 1)
    print(window)