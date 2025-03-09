import torch
import numpy as np
from models.ctr_model import CTRModel
from models.dqn_model import DQN
from agents.bidding_agent import BiddingAgent
from utils.budget_manager import BudgetManager


state_dim = 10  
action_dim = 20  
ctr_model = CTRModel(input_dim=state_dim)
dqn_model = DQN(state_dim, action_dim)

# Initialize agent and budget manager
agent = BiddingAgent(state_dim, action_dim, ctr_model, dqn_model)
budget_manager = BudgetManager(budget=1000)

# Simulating Bidding Process
for i in range(100):  
    features = np.random.rand(state_dim) 
    p_click = ctr_model.predict(torch.FloatTensor(features))  # Get CTR estimate
    state = np.append(features, p_click)  # Enhanced state
    bid = agent.select_bid(state)  # Get bid price
    
    # Simulated auction outcome
    cost = np.random.randint(1, 20) 
    click = np.random.choice([0, 1], p=[0.9, 0.1]) 
    
    # Update budget and report
    budget_manager.update_budget(cost)
    print(f"Auction {i+1}: Bid {bid} cents, Cost {cost} cents, Click {click}")
    
    if budget_manager.current_budget <= 0:
        break
