# %%
# Constants
import numpy as np
# Constants
goal = 100
gamma = 1.0  # Discount factor
prob_roll = 1/6  # Probability of rolling any number between 1 and 6
# Since we have varied dice faces so won't be using it anywhere instead computing it dynamically at run time.


class DiceGameEnvironment:
    def __init__(self, goal=100, sides=6):
        self.goal = goal  # The target score to win the game
        self.sides = sides  # Number of sides on the die
        self.current_score = 0  # Initial score at the start of the game

    def roll_die(self):
        """Simulate rolling the die and return the result as an integer between 1 and the number of sides."""
        return np.random.randint(1, self.sides + 1)

    def reset(self):
        """Reset the environment to the initial state."""
        self.current_score = 0  # Reset the score to 0
        return self.current_score  # Return the initial state

    def step(self, current_score, action):
        """
        Take an action from the current score and return the new score and the reward.
        Parameters:
            current_score (int): The current score of the player.
            action (str): 'roll' or 'stop' indicating the player's action.
        Returns:
            tuple: A tuple containing the new score and the reward obtained.
        """
        if action == "stop":
            return current_score, self.calculate_reward(current_score, current_score, action)
        elif action == "roll":
            roll = self.roll_die()
            if roll == 1:
                return 0, self.calculate_reward(current_score, 0, action)
            new_score = current_score + roll
            if new_score > self.goal:
                return new_score, self.calculate_reward(current_score, 0, action)
            elif new_score == self.goal:
                return new_score, 100  # Large reward for winning
            else:
                return new_score, 0  # Continue with the new score

    def calculate_reward(self, old_score, new_score, action):
        if action == "stop":
            if new_score == self.goal:
                return 100  # Large reward for winning
            else:
                return -1  # Small penalty for stopping without winning
        elif new_score == 0:
            return -old_score  # Penalty for rolling a 1
        elif new_score > self.goal:
            return -100  # Large penalty for exceeding the goal
        else:
            return 0  # No immediate reward for other rolls

    def is_terminal(self, score):
        """Check if the given score is a terminal state."""
        return score == 0 or score >= self.goal


def policy_evaluation(policy, env, gamma=1.0, threshold=0.01):
    V = np.zeros(env.goal + 1)  # Initialize value function
    iteration = 0
    while True:
        delta = 0
        for s in range(1, env.goal):  # Skip terminal state
            v = 0
            if policy[s] == 0:  # Stop
                v = env.calculate_reward(s, s, 'stop')
            elif policy[s] == 1:  # Roll
                for roll in range(1, env.sides + 1):
                    next_s = s + roll
                    if next_s > env.goal:
                        next_s = 0
                    prob = 1 / env.sides
                    v += prob * (env.calculate_reward(s, next_s, 'roll') + gamma * V[next_s])
            delta = max(delta, np.abs(V[s] - v))
            V[s] = v
        if delta < threshold:
            break
        iteration += 1
        if iteration % 100 == 0:
            print(f"Policy Evaluation Iteration {iteration}: delta={delta}")
    return V

def policy_improvement(V, env, gamma=1.0):
    policy = np.zeros(env.goal + 1, dtype=int)
    for s in range(1, env.goal):
        A = np.zeros(2)  # Store the expected returns for each action
        
        # Calculate stop action value
        A[0] = env.calculate_reward(s, s, 'stop')
        
        # Calculate roll action value
        for roll in range(1, env.sides + 1):
            next_s = s + roll
            if next_s > env.goal:
                next_s = 0
            prob = 1 / env.sides
            A[1] += prob * (env.calculate_reward(s, next_s, 'roll') + gamma * V[next_s])
        
        # Choose the best action
        best_action = np.argmax(A)
        policy[s] = best_action
        
        # Debugging output
        print(f"State {s}: Stop Value={A[0]}, Roll Value={A[1]}, Best Action={'Stop' if best_action == 0 else 'Roll'}")
        
    return policy

def policy_iteration(env, gamma=1.0):
    policy = np.random.choice([0, 1], size=(env.goal + 1))  # Random initial policy
    stable = False
    iteration = 0
    while not stable:
        print(f"Policy Iteration {iteration} Start")
        V = policy_evaluation(policy, env, gamma)
        new_policy = policy_improvement(V, env, gamma)
        
        # Debugging output to show policy changes
        for s in range(1, env.goal):
            if policy[s] != new_policy[s]:
                print(f"State {s} Policy Changed: {policy[s]} -> {new_policy[s]}")
        
        stable = np.array_equal(new_policy, policy)
        print(f"Policy Iteration {iteration}: Policy Stable={stable}")
        policy = new_policy
        iteration += 1
    return policy, V



def value_iteration(env, gamma=1.0, threshold=0.01, epsilon=0.1):
    V = np.zeros(env.goal + 1)  # Initialize value function for all states
    
    iteration = 0
    while True:
        delta = 0
        for s in range(1, env.goal):  # Iterate over all states except the terminal state
            if env.is_terminal(s):
                continue
            
            # Initialize a list to store rewards for all actions
            stop_reward = env.calculate_reward(s, s, 'stop')
            expected_roll_value = 0
            for roll in range(1, env.sides + 1):
                next_s = s + roll
                if next_s > env.goal:
                    next_s = 0  # Reset if the score exceeds the goal
                expected_roll_value += (1 / env.sides) * (env.calculate_reward(s, next_s, 'roll') + gamma * V[next_s])
            
            # Choose the best expected reward
            max_reward = max(stop_reward, expected_roll_value)
            
            # Update the value table and calculate delta
            delta = max(delta, abs(V[s] - max_reward))
            V[s] = max_reward
            print(f"State {s}: V[s]={V[s]}, max_reward={max_reward}, delta={delta}")
        
        # Check for convergence
        if delta < threshold:
            break
        iteration += 1
        if iteration % 100 == 0:
            print(f"Value Iteration {iteration}: delta={delta}")
    
    # Extract the optimal policy based on the value function
    policy = np.zeros(env.goal + 1, dtype=int)
    for s in range(1, env.goal):
        if env.is_terminal(s):
            continue
        
        stop_value = env.calculate_reward(s, s, 'stop')
        roll_values = [(1 / env.sides) * (env.calculate_reward(s, s + roll, 'roll') + gamma * V[s + roll if s + roll <= env.goal else 0])
                       for roll in range(1, env.sides + 1)]
        best_action_value = max(stop_value, np.sum(roll_values))
        
        # Choose the action with the highest expected return
        policy[s] = 0 if stop_value >= best_action_value else 1

    return policy, V



def simulate_game(env, policy, num_simulations=100):
    cumulative_rewards = 0
    for i in range(num_simulations):
        state = env.reset()
        total_reward = 0
        steps = 0
        while not env.is_terminal(state):
            action = 'stop' if policy[state] == 0 else 'roll'
            old_state = state
            state, reward = env.step(state, action)
            total_reward += reward
            steps += 1
            if env.is_terminal(state):
                break
            if steps > 1000:  # Add a safeguard for runaway simulations
                print(f"Simulation {i} exceeded 1000 steps, breaking.")
                break
        cumulative_rewards += total_reward
        print(f"Simulation {i}: Total Reward = {total_reward}, Steps = {steps}")
    return cumulative_rewards


env_goal_50 = DiceGameEnvironment(goal=50, sides=6)
env = DiceGameEnvironment(goal=100, sides=6)
policy_vi_50, V_vi_50 = value_iteration(env_goal_50)
print("Value Iteration Results for Goal 50:")
print("Policy:", policy_vi_50)
print("Value Function:", V_vi_50)

# Environment with die having 8 sides
env_sides_8 = DiceGameEnvironment(goal=100, sides=8)
policy_vi_8, V_vi_8 = value_iteration(env_sides_8)
print("Value Iteration Results for 8-sided Die:")
print("Policy:", policy_vi_8)
print("Value Function:", V_vi_8)

# Environment with discount factor 0.9
policy_vi_09, V_vi_09 = value_iteration(env, gamma=0.9)
print("Value Iteration Results with Gamma 0.9:")
print("Policy:", policy_vi_09)
print("Value Function:", V_vi_09)

# Environment with discount factor 0.95
policy_vi_095, V_vi_095 = value_iteration(env, gamma=0.95)
print("Value Iteration Results with Gamma 0.95:")
print("Policy:", policy_vi_095)
print("Value Function:", V_vi_095)

policy_vi, V_vi = value_iteration(env)



import matplotlib.pyplot as plt

def plot_value_function(V, title):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(V)), V, marker='o')
    plt.title(title)
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

# Plot value function for original environment
plot_value_function(V_vi, "Value Function for Goal 100 and 6-sided Die")

# Plot value function for goal 50
plot_value_function(V_vi_50, "Value Function for Goal 50 and 6-sided Die")

# Plot value function for 8-sided die
plot_value_function(V_vi_8, "Value Function for Goal 100 and 8-sided Die")

# Plot value function for discount factor 0.9
plot_value_function(V_vi_09, "Value Function with Gamma 0.9")

# Plot value function for discount factor 0.95
plot_value_function(V_vi_095, "Value Function with Gamma 0.95")


# Scenario 1: Change Goal Score to 50
env_goal_50 = DiceGameEnvironment(goal=50, sides=6)
policy_vi_50, V_vi_50 = value_iteration(env_goal_50)
print("Value Iteration Results for Goal 50:")
print("Policy:", policy_vi_50)
print("Value Function:", V_vi_50)

# Scenario 2: Change Die to 8 Sides
env_sides_8 = DiceGameEnvironment(goal=100, sides=8)
policy_vi_8, V_vi_8 = value_iteration(env_sides_8)
print("Value Iteration Results for 8-sided Die:")
print("Policy:", policy_vi_8)
print("Value Function:", V_vi_8)

# Scenario 3: Experiment with Different Discount Factors
policy_vi_09, V_vi_09 = value_iteration(env, gamma=0.9)
print("Value Iteration Results with Gamma 0.9:")
print("Policy:", policy_vi_09)
print("Value Function:", V_vi_09)

policy_vi_095, V_vi_095 = value_iteration(env, gamma=0.95)
print("Value Iteration Results with Gamma 0.95:")
print("Policy:", policy_vi_095)
print("Value Function:", V_vi_095)

# Scenario 4: Visualize the Value Function
import matplotlib.pyplot as plt
import seaborn as sns

def plot_value_function(V, title):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(V)), V, marker='o')
    plt.title(title)
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

# Plot value function for original environment
plot_value_function(V_vi, "Value Function for Goal 100 and 6-sided Die")

# Plot value function for goal 50
plot_value_function(V_vi_50, "Value Function for Goal 50 and 6-sided Die")

# Plot value function for 8-sided die
plot_value_function(V_vi_8, "Value Function for Goal 100 and 8-sided Die")

# Plot value function for discount factor 0.9
plot_value_function(V_vi_09, "Value Function with Gamma 0.9")

# Plot value function for discount factor 0.95
plot_value_function(V_vi_095, "Value Function with Gamma 0.95")