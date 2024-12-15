from __future__ import annotations  
import pandas as pd
from numpy import random, argmin, argmax, sqrt, log

# Concepts learned -> N Armed Bandit, Random Search, Epsilon Greedy, Upper Confidence Bound
# Given a CTR dataset assuming the click probability 
# depends upon Gender (with 0.7 times males clicking it and 0.3 times female clicking it)
# Design a CTR Environment using - Random Approahc, Epsilon Greedy, Upper Confidence Bound
# CTR Environment should have -> (Data Loaded, the probabilities and reward function)
# simulate_random_policy should simulate the policy for N times and see its effect
class CTREnvironment:
    
    def __init__(self: CTREnvironment, num_arms: int, data: pd.DataFrame) -> CTREnvironment:
        self.arm_probabilities = {}
        self.total_arms = num_arms
        self.data = data
        self.compute_probabilities()
        
    def compute_probabilities(self: CTREnvironment):
        for i in range(self.total_arms):
            arm_data = self.data[self.data['armid'] == i]       
            self.male_data = (self.data['Gender'] == 'Male').mean()
            self.female_data = (self.data['Gender'] == 'Female').mean()
            prob = (0.7 * (arm_data['Gender'] == 'Male').mean() +
                       0.6 * (arm_data['Gender'] == 'Female').mean())
            self.arm_probabilities[i] = prob
        

    def select_arm(self: CTREnvironment) -> int:
        return random.randint(0, self.total_arms)
    
    def get_reward_for_selection(self: CTREnvironment, arm_id: int) -> float:
        probability = self.arm_probabilities.get(arm_id, None)
        if probability is None:
            print(f'Probability not found for arm_id: {arm_id}')
            probability = 0
        return 1 if random.random() < probability else 0
    
def simulate_random_policy(env: CTREnvironment, iterations:int = 10, result:list[(int, int)] = []):
    for i in range(iterations):
        selection = env.select_arm()
        reward = env.get_reward_for_selection(selection)
        result.append((selection, reward)) 
    return result

def simulate_greedy_policy(env: CTREnvironment, iterations:int = 10, epsilon: float = 0.0, result:list[(int, int)] = []):
    pulled_arms_count = [0] * env.total_arms # Contains all the arms pull count
    arms_reward = [0] * env.total_arms # Contains the reward for each arm
    for _ in range(iterations):
        # If all arms are pulled once and explored take the one which gave maximum results
        arm_id = -1
        if argmin(pulled_arms_count) > 0 and epsilon < random.random():
            arm_id = argmax(arms_reward)
        # Explore all the arms
        else:
            arm_id = env.select_arm()
        
        current_reward = env.get_reward_for_selection(arm_id)
        pulled_arms_count[arm_id] += 1
        total_visits_to_current_arm = pulled_arms_count[arm_id]
        arms_reward[arm_id] = current_reward + (arms_reward[arm_id] - current_reward)/total_visits_to_current_arm
    result = []
    for index,reward in enumerate(arms_reward):
        result.append((index, reward))
    return result

def ucb_policy(env: CTREnvironment, iterations: int = 10, result: list[(int, int)] = []):    
    # SELECT ARM
    def select_arm(estimates: list[int]):
        ucb_values = [0.0] * env.total_arms
        for i in range(env.total_arms):
            if visits[i] == 0:
                return i # Ensure that the node has been visited once
            bonus = sqrt((2 * log(total_visits))/visits[i])
            updated_ucb_value = estimates[i] + bonus
            ucb_values[i] = updated_ucb_value
        return argmax(ucb_values) 
    
    # UPDATE ESTIMATES 
    def update_estimates(
        arm_id: int, 
        reward: int, 
        total_visits: int, 
        visits: list[int], 
        estimates: list[int]
        ) -> tuple[int, list[int], list[int]]:
        total_visits += 1
        visits[arm_id] += 1
        total_visits_so_far = visits[arm_id]
        existing_estimate = estimates[arm_id]
        new_estimate = existing_estimate + (reward - existing_estimate)/total_visits_so_far
        estimates[arm_id] = new_estimate
        return (total_visits, visits, estimates)
    
    estimates = [0] * env.total_arms
    visits = [0] * env.total_arms
    total_visits = 0
    for i in range(iterations):
        # Pick up an Arm
        arm_id = select_arm(estimates=estimates)
        # Fetch the Reward
        reward = env.get_reward_for_selection(arm_id)
        # Update the Policy
        total_visits, visits, estimates = update_estimates(arm_id, reward, total_visits, visits, estimates)
    results = []
    for index, estimate in enumerate(estimates):
        results.append((index, estimate))
    return results
        
    
        

def pre_process():
    data = r'2825/ctr_dataset.csv'
    data = pd.read_csv(data)
    data['armid'] = data.groupby(['Age', 'Gender', 'City', 'Phone_OS']).ngroup()
    num_arms = data['armid'].nunique()
    env = CTREnvironment(num_arms, data)
    return env

def generate_results(env: CTREnvironment, iterations: int = 10, simulation_type: str = 'random', epsilon: float = 0.0) -> list[(int,int,int)]:
    # Match against different simulation_types
    if simulation_type == 'random':
        return simulate_random_policy(env, iterations)
    elif simulation_type == 'greedy':
        return simulate_greedy_policy(env, iterations, epsilon = epsilon)
    elif simulation_type == 'ucb':
        return ucb_policy(env, iterations)
if __name__ == "__main__":
    env = pre_process()
    iterations = 10000
    # random_results = generate_results(env, iterations = iterations, simulation_type = 'random')
    # greedy_results = generate_results(env, iterations = iterations, simulation_type = 'greedy')
    # ucb_results = generate_results(env, iterations = iterations, simulation_type = 'ucb')
    epsilon_greedy_results = generate_results(env, iterations = iterations, epsilon = 0.01, simulation_type = 'greedy')
    for result in epsilon_greedy_results:
        print(result)
    