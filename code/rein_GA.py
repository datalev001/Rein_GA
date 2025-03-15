import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
import gym
from gym import spaces

def generate_sales_data_realistic(num_samples=2163, seed=42):
    """
    Generate 2163 samples of sales data with realistic behavior.
    """
    np.random.seed(seed)
    weeks = np.arange(1, num_samples + 1)
    
    advertising_spend = np.random.randint(50, 300, size=num_samples)
    competitor_price = np.random.randint(80, 150, size=num_samples)
    unit_cost = np.random.randint(60, 120, size=num_samples)
    price_level = np.random.randint(90, 200, size=num_samples)
    
    season_index = np.sin(2 * np.pi * weeks / 52) + np.random.normal(0, 0.3, num_samples)
    
    base_sales = (
        300 + 0.9 * advertising_spend
        - 0.5 * (price_level - competitor_price)
        + 70 * season_index
        - 1.2 * (unit_cost - 60)
    )
    noise = np.random.normal(0, 50, num_samples) + np.random.randint(-20, 20, size=num_samples)
    sales = (base_sales + noise).clip(10, None).astype(int)
    
    df = pd.DataFrame({
        "Week": weeks,
        "AdvertisingSpend": advertising_spend,
        "CompetitorPrice": competitor_price,
        "UnitCost": unit_cost,
        "PriceLevel": price_level,
        "SeasonIndex": season_index,
        "Sales": sales
    })
    return df

df = generate_sales_data_realistic(num_samples=2163, seed=42)
print (df.head(8))
df.to_csv(r"C:\backupcgi\final_bak\RG.csv")


df = pd.read_csv("RG.csv")

class SalesProfitEnv(gym.Env):
    """
    Custom environment simulating sales and pricing optimization.
    """
    def __init__(self, df):
        super(SalesProfitEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.max_index = len(self.df) - 1
        self.action_space = spaces.Discrete(11)  # Actions: -5 to +5 price changes
        
        low_state = np.array([0, 0, 0, 0, -2], dtype=np.float32)
        high_state = np.array([1000, 200, 200, 120, 2], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_state, high=high_state, dtype=np.float32)
        
        self.current_step = 0
        self.done = False
        self.prev_action = None
    
    def reset(self):
        self.current_step = 0
        self.done = False
        return self._get_state()
    
    def step(self, action):
        delta_price = action - 5  # Mapping action to price change (-5 to +5)
        
        row = self.df.iloc[self.current_step]
        adv_spend = row["AdvertisingSpend"]
        comp_price = row["CompetitorPrice"]
        unit_cost = row["UnitCost"]
        old_price = row["PriceLevel"]
        season_idx = row["SeasonIndex"]
        sales_real = row["Sales"]
        
        price_used = max(old_price + delta_price, 1.0)
        sales_pred = sales_real + np.random.normal(-2 * delta_price, 30) + np.random.randint(-10, 10)
        sales_pred = max(sales_pred, 0)
        
        # Reward function modification
        profit = sales_pred * (price_used - unit_cost) - adv_spend
        
        # New penalty for extreme jumps in pricing (reduces oscillation)
        if self.prev_action is not None:
            price_smoothness_penalty = abs(delta_price - self.prev_action) * 3  # Adjust penalty strength
            profit -= price_smoothness_penalty  # Encourages gradual changes

        # Soft penalty for excessive same-action repetition
        if self.prev_action == delta_price:
            profit -= 1  # Small discouragement for repeated actions
        
        self.prev_action = delta_price  # Store previous action
        
        self.current_step += 1
        if self.current_step >= self.max_index:
            self.done = True
        
        next_state = self._get_state(price_used)
        return next_state, float(profit), self.done, {}

    def _get_state(self, new_price=None):
        row = self.df.iloc[self.current_step]
        adv_spend = row["AdvertisingSpend"]
        comp_price = row["CompetitorPrice"]
        unit_cost = row["UnitCost"]
        price_level = new_price if new_price is not None else row["PriceLevel"]
        season_idx = row["SeasonIndex"]
        
        return np.array([adv_spend, comp_price, price_level, unit_cost, season_idx], dtype=np.float32)

def train_ppo_ga(df, num_generations=3, population_size=4):
    """
    Train a population of PPO agents with Genetic Algorithm enhancements.
    """
    env = SalesProfitEnv(df)
    max_steps = len(df)
    population = [PPO_Agent() for _ in range(population_size)]
    
    best_global_agent = None
    best_global_score = -1e9
    
    for gen in range(num_generations):
        print(f"\n=== Generation {gen} ===")
        
        fitness_scores = []
        
        for i, agent in enumerate(population):
            memory = generate_trajectory(env, agent, max_steps=max_steps)
            agent.update(memory)
            
            eval_memory = generate_trajectory(env, agent, max_steps=max_steps)
            total_reward = sum(eval_memory["rewards"])
            fitness_scores.append(total_reward)
        
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Sort scores descending
        best_idx = sorted_indices[0]
        
        if fitness_scores[best_idx] > best_global_score:
            best_global_score = fitness_scores[best_idx]
            best_global_agent = copy.deepcopy(population[best_idx])
        
        print(f"  Best agent in this generation = {best_idx}, score={fitness_scores[best_idx]:.2f}")
        
        new_population = []
        elite_agent = population[best_idx]  # Keep best agent
        new_population.append(elite_agent)
        
        top2 = sorted_indices[:2]  # Take top 2 for crossover
        while len(new_population) < population_size:
            p1, p2 = np.random.choice(top2, 2, replace=True)
            child = crossover_and_mutate(population[p1], population[p2], mutation_rate=0.15)
            new_population.append(child)
        
        population = new_population  # Replace old population
    
    print("\n=== Training Done ===")
    print(f"Best global agent score: {best_global_score:.2f}")
    return best_global_agent


# Run training and visualization
train_and_visualize()
