import numpy as np 
import itertools
from Grid_World import GridWorld
from Policy_iteration import policy_iteration

problem = GridWorld()
GAMMA = 0.5
sa_values = np.zeros((len(problem.states), len(problem.actions)), dtype=float)

def epsilon_greedy(a_values, epsilon):
    # -------------------------
    # This function takes a list a_values where each index i corresponds to an estimate of the value of action i and then performs
    # epsilon-greedy action selection to sample and return an action
    
    p = np.random.random(1)
    if p<epsilon:
        action = np.random.choice(len(a_values))
    else:
        action = np.argmax(a_values)
    return action

for i in range(100000):
    s = problem.reset()
    done = False
    while not done:
        # -------------------------
        # Perform one step of Q-learning here using sa_values to store the action-value estimates and epsilon_greedy to perform the
        # action selection
        
        lr = 0.01 #Learning rate
        
        action  = epsilon_greedy(sa_values[s,:],epsilon=0.5)
        next_state, reward, done = problem.step(action)                
        sa_values[s,action] = (1-lr)*sa_values[s,action]+lr*(reward+GAMMA*max(sa_values[next_state,:]))
        s = next_state

optimal_policy_state_values = policy_iteration(problem, gamma=0.5)[1]
learned_policy_state_values = np.max(sa_values, axis=1)

print('Optimal Policy State Values:')
print(optimal_policy_state_values)
print('Learned Policy State Values:')
print(learned_policy_state_values)
print('Root Mean Squared Value Error: {0:.8f}'.format(
    np.sqrt(np.mean(np.square(optimal_policy_state_values - learned_policy_state_values)))))