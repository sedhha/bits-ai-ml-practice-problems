import numpy as np
from typing import List, Tuple

# Creating Dice Environment
class DiceGameEnvironment:
    def __init__(self, goal: int = 100, sides: int = 6) -> None:
        self.goal: int = goal
        self.state: int = 0  # Start at 0 points
        self.turn_start_state: int = 0  # Points at the start of the turn
        self.sides: int = sides  # Number of sides on the die
        self.is_terminal: bool = False  # Tracks whether the current state is terminal

    def roll_die(self) -> int:
        """Simulate rolling the die and return the result as an integer between 1 and the number of sides."""
        return np.random.randint(1, self.sides + 1)
    
    def reset(self) -> int:
        """Reset the game to the initial state and return the new state."""
        self.state = 0
        self.turn_start_state = 0
        self.is_terminal = False
        return self.state

    def step(self, action: int) -> Tuple[int, bool]:
        if self.is_terminal:
            return self.state, True

        if action == 0:  # Stop
            self.is_terminal = True  # Ending the game by player choice
            return self.state, self.is_terminal

        if action == 1:  # Roll the dice
            roll = self.roll_die()
            if roll == 1:
                self.state = self.turn_start_state  # Reset to score at start of turn
                return self.state, False  # End the turn but not the game
            else:
                self.state += roll
                if self.state >= self.goal:
                    self.is_terminal = True  # Win by reaching or exceeding the goal
                return self.state, self.is_terminal

        return self.state, self.is_terminal


def calculate_reward(action: int, next_state: int) -> int:
    """
    Calculate the reward for the given action in the given state.
    Args:
        state (int): the current state of the game.
        action (int): the action taken ('0' for stop, '1' for roll).
        next_state (int): the state after the action is taken.
    Returns:
        int: A numeric reward based on the game outcome.
    """
    if action == 0:  # Stop
        # Reward or penalize based on the closeness to the goal
        if next_state == 100:
            return 100  # Big positive reward for winning the game
        else:
            # Penalize based on how far from 100 the stop was made
            return -abs(100 - next_state)  # e.g., -5 points if stopped at 95

    elif action == 1:  # Roll
        if next_state > 100:
            return -100  # Large penalty for losing the game by exceeding 100
        elif next_state == 100:
            return 100  # Big positive reward for winning the game
        else:
            # No intermediate rewards for rolling unless it directly results in winning or losing
            return 0

    return 0  # Default case (should not be reached, but just in case)

def policy_evaluation(policy:np.ndarray, env:DiceGameEnvironment, gamma:int=1.0, threshold=1e-6) -> np.ndarray:
    V = np.zeros(env.goal + 1)  # Initialize value function
    iteration = 0
    while True:
        delta = 0
        for s in range(1, env.goal):  # Skip terminal state
            v = 0
            if policy[s] == 0:  # Stop
                v = calculate_reward(0,s)
            elif policy[s] == 1:  # Roll
                for roll in range(1, env.sides + 1):
                    next_s = s + roll
                    if next_s > env.goal:
                        next_s = 0
                    prob = 1 / env.sides
                    v += prob * (calculate_reward(1, next_s) + gamma * V[next_s])
            delta = max(delta, np.abs(V[s] - v))
            V[s] = v
        if delta < threshold:
            break
        iteration += 1
    return V

def policy_improvement(V:np.ndarray, env:DiceGameEnvironment, gamma:int=1.0) -> np.ndarray:
    policy = np.zeros(env.goal + 1, dtype=int)
    for s in range(1, env.goal):
        A = np.zeros(2)  # Store the expected returns for each action
        
        # Calculate stop action value
        A[0] = calculate_reward(0,s)
        # Calculate roll action value
        for roll in range(1, env.sides + 1):
            next_s = s + roll
            if next_s > env.goal:
                next_s = 0
            prob = 1 / env.sides
            A[1] += prob * (calculate_reward(0, next_s) + gamma * V[next_s])
        
        # Choose the best action
        best_action = np.argmax(A)
        policy[s] = best_action
                
    return policy

def policy_iteration(env:DiceGameEnvironment, gamma=1.0):
    policy = np.random.choice([0, 1], size=(env.goal + 1))  # Random initial policy
    stable = False
    iteration = 0
    while not stable:
        V = policy_evaluation(policy, env, gamma)
        new_policy = policy_improvement(V, env, gamma)
        stable = np.array_equal(new_policy, policy)
        policy = new_policy
        iteration += 1
    return policy, V

def value_iteration(env: DiceGameEnvironment, gamma: float = 1.0, threshold: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    V = np.zeros(env.goal + 1)  # Initialize value function for all states

    while True:
        delta = 0
        for s in range(1, env.goal):  # Iterate over all non-terminal states
            stop_reward = calculate_reward(0, s)
            roll_rewards = 0
            for roll in range(1, env.sides + 1):
                next_s = s + roll
                if next_s > env.goal:
                    next_s = env.goal  # Stop adding if the score exceeds the goal
                roll_rewards += (1 / env.sides) * (calculate_reward(1, next_s) + gamma * V[next_s])
            
            new_value = max(stop_reward, roll_rewards)
            delta = max(delta, abs(V[s] - new_value))
            V[s] = new_value
        
        if delta < threshold:  # Check for convergence
            break

    # Derive policy from value function
    policy = np.zeros(env.goal + 1, dtype=int)
    for s in range(1, env.goal):
        stop_value = calculate_reward(0, s)
        roll_values = [(1 / env.sides) * (calculate_reward(1, s + roll) + gamma * V[s + roll if s + roll <= env.goal else s])
                       for roll in range(1, env.sides + 1)]
        roll_value = sum(roll_values)
        policy[s] = 0 if stop_value >= roll_value else 1

    return policy, V


def simulate_game(env:DiceGameEnvironment, policy:np.ndarray, num_simulations:int=100)->int:
    cumulative_rewards = 0
    for i in range(num_simulations):
        state = env.reset()
        total_reward = 0
        steps = 0
        while not env.is_terminal:
            action = 0 if policy[state] == 0 else 1
            state, reward = env.step(action)
            total_reward += reward
            steps += 1
            if env.is_terminal:
                break
        cumulative_rewards += total_reward
    return cumulative_rewards

# Assuming env is an instance of DiceGameEnvironment
env = DiceGameEnvironment(goal=100, sides=6)

# Policy Iteration
policy_pi, V_pi = policy_iteration(env)
print("Policy Iteration Results:")
print("Policy:", policy_pi)
print("Value Function:", V_pi)
cumulative_rewards_pi = simulate_game(env, policy_pi, 100)
print("Total Cumulative Reward from Policy Iteration:", cumulative_rewards_pi)

# Value Iteration
policy_vi, V_vi = value_iteration(env)
print("Value Iteration Results:")
print("Optimal Policy Iteration:", policy_vi)
print("Optimal Value Iteration:", V_vi)
cumulative_rewards_vi = simulate_game(env, policy_vi, 100)
print("Total Cumulative Reward from Value Iteration:", cumulative_rewards_vi)


env_goal_50 = DiceGameEnvironment(goal=50, sides=6)
policy_vi_50, V_vi_50 = value_iteration(env_goal_50)
print("Value Iteration Results for Goal 50:")
print("Optimal Policy Iteration:", policy_vi_50)
print("Optimal Value Iteration:", V_vi_50)

# Environment with die having 8 sides
env_sides_8 = DiceGameEnvironment(goal=100, sides=8)
policy_vi_8, V_vi_8 = value_iteration(env_sides_8)
print("Value Iteration Results for 8-sided Die:")
print("Optimal Policy Iteration:", policy_vi_8)
print("Optimal Value Iteration:", V_vi_8)

# Environment with discount factor 0.9
policy_vi_09, V_vi_09 = value_iteration(env, gamma=0.9)
print("Value Iteration Results with Gamma 0.9:")
print("Optimal Policy Iteration:", policy_vi_09)
print("Optimal Value Iteration:", V_vi_09)

# Environment with discount factor 0.95
policy_vi_095, V_vi_095 = value_iteration(env, gamma=0.95)
print("Value Iteration Results with Gamma 0.95:")
print("Optimal Policy Iteration:", policy_vi_095)
print("Optimal Value Iteration:", V_vi_095)