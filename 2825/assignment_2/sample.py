## Necessary Libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
#### Load the dataset
data=pd.read_csv(r'dataset/energydata_complete.csv')
# Display the first few rows of the dataset to inspect it
data.head()
# Check for missing values in the dataset
missing_values = data.isnull().sum()

# Convert the 'date' column to datetime format for easier handling
data['date'] = pd.to_datetime(data['date'])

# Display missing values and data types
missing_values, data.dtypes
# Pre process the dataset to get the features and target and scale them
# Selecting the features to be normalized
features_to_normalize = data.columns.drop(['date', 'Appliances', 'lights'])  # Excluding date, Appliances, and lights

# Applying Min-Max Scaling
scaler = MinMaxScaler()
data[features_to_normalize] = scaler.fit_transform(data[features_to_normalize])

# Define features and target variable
features = data.drop(['Appliances', 'date'], axis=1)  # Exclude 'date' if it's not used as a feature
target = data['Appliances']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

### Define Actor Model

data = data.drop(['date', 'Appliances'], axis=1)  # Dropping non-feature columns

# Check the number of features now
state_space = data.shape[1]  # This should be the actual number of features
print(f"State space (number of features): {state_space}")

### Redefine Actor and Critic Models to match the actual state space
def build_actor_model():
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(state_space,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')  # Assuming 3 actions: Decrease, Maintain, Increase
    ])
    return model

def build_critic_model():
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(state_space,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # Output is the value estimation
    ])
    return model

# Rebuild models with the corrected state space
actor_model = build_actor_model()
critic_model = build_critic_model()


### Calculate Reward Function

def calculate_reward(current_state, next_state, energy_before, energy_after):
    target_temperature = 22  # Target temperature for minimal energy use and comfort
    temp_features = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9']  # Temperature features in the dataset

    # Calculate deviation penalty: sum of absolute differences from target temperature
    deviation_penalty_current = sum(abs(current_state[temp_features] - target_temperature))
    deviation_penalty_next = sum(abs(next_state[temp_features] - target_temperature))

    # Calculate the comfort penalty based on average deviation
    comfort_penalty = (deviation_penalty_current + deviation_penalty_next) / 2

    # Calculate energy savings: positive if energy use decreases
    energy_savings = energy_before - energy_after

    # Combine the comfort penalty and energy savings to get the final reward
    reward = energy_savings - comfort_penalty

    return reward


def simulate_environment(current_state, action, data, index, energy_data):
    # Adjust temperature based on action
    temp_adjustment = action - 1  # action: 0 (decrease by 1°C), 1 (maintain), 2 (increase by 1°C)
    temp_features = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9']
    
    # Update temperatures
    next_state = current_state.copy()
    next_state[temp_features] += temp_adjustment
    
    # Ensure we don't go out of bounds in the dataset
    if index + 1 < len(data):
        next_index = index + 1
    else:
        next_index = index  # Stay at the last index if we're at the end of the data set
    
    # Get energy consumption before and after the action
    energy_before = energy_data[index]
    energy_after = energy_data[next_index]
    
    # Calculate the reward
    reward = calculate_reward(current_state, next_state, energy_before, energy_after)
    
    # Return the new state and the reward
    return next_state, reward

# Train the Actor-Critic models

def update_models(current_state, action, advantage, target, actor_model, critic_model, optimizer_actor, optimizer_critic):
    # Convert current state to a suitable tensor for prediction
    state_tensor = tf.convert_to_tensor(current_state.values.reshape(1, -1), dtype=tf.float32)

    # Update Critic Model
    with tf.GradientTape() as tape:
        # Get value prediction from the critic model
        value = critic_model(state_tensor, training=True)
        # Calculate critic loss as mean squared error between target values and predicted values
        loss_critic = tf.keras.losses.MSE(target, value)
    # Calculate gradients for the critic model
    grads = tape.gradient(loss_critic, critic_model.trainable_variables)
    # Apply gradients to update the critic model
    optimizer_critic.apply_gradients(zip(grads, critic_model.trainable_variables))
    
    # Update Actor Model
    with tf.GradientTape() as tape:
        # Get action probabilities from the actor model
        action_probs = actor_model(state_tensor, training=True)
        # Calculate the log probability of the selected action
        action_log_probs = tf.math.log(action_probs[0, action])
        # Calculate actor loss using the advantage (scaled by log probability of the action)
        loss_actor = -action_log_probs * advantage
    # Calculate gradients for the actor model
    grads = tape.gradient(loss_actor, actor_model.trainable_variables)
    # Apply gradients to update the actor model
    optimizer_actor.apply_gradients(zip(grads, actor_model.trainable_variables))



def train_function(features, energy_data, episodes=500):
    discount_factor = 0.99
    optimizer_actor = tf.keras.optimizers.Adam(learning_rate=0.001)
    optimizer_critic = tf.keras.optimizers.Adam(learning_rate=0.001)

    for episode in range(episodes):
        print(f"Starting episode {episode+1}")
        total_reward = 0
        current_state = features.iloc[0]  # Reset to initial state at start of each episode
        index = 0
        
        while index < len(features) - 1:
            # Predict action probabilities from actor model
            print(f"Processing step {index+1}/{len(features)}")
            action_probabilities = actor_model.predict(current_state.values.reshape(1, -1))
            action = np.argmax(action_probabilities)
            
            # Simulate the environment with the chosen action
            next_state, reward = simulate_environment(current_state, action, features, index, energy_data)
            total_reward += reward
            
            # Compute target and advantage for updating critic and actor
            value_current = critic_model.predict(current_state.values.reshape(1, -1))
            value_next = critic_model.predict(next_state.values.reshape(1, -1))
            target = reward + discount_factor * value_next
            advantage = target - value_current
            
            # Update models
            update_models(current_state, action, advantage, target, actor_model, critic_model, optimizer_actor, optimizer_critic)
            
            # Update state and index
            current_state = next_state
            index += 1
        
        print(f"Episode {episode+1} completed. Mean reward: {total_reward / len(features)}")
        # Print mean reward for episode
        mean_reward = total_reward / index
if __name__ == '__main__':
    train_function(features, target.values)