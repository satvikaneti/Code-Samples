import sys
from environment import MountainCar
import numpy as np
import random 
#import pyglet
import pandas as pd

#position can be between -1.2 and 0.6 inclusive
#velocity can be between -0.07 and 0.07 inclusive
#actions are 0-> left, 1-> doing nothing, 2-> pushing right

def main(args):
    """
    #read in command line arguments
    mode = args[1]
    weight_out = args[2]
    returns_out = args[3]
    episodes = int(args[4])
    maxiters = int(args[5])
    epsilon = float(args[6])
    gamma = float(args[7])
    alpha= float(args[8])
    
    """
    mode = "tile"
    episodes = 400
    maxiters = 200
    epsilon = 0.05
    gamma = 0.99
    alpha = 0.00005
    weight_out = "C:/Users/sneti/Documents/school/ML/hw8/handout/examples/sample_weight.csv"
    returns_out = "C:/Users/sneti/Documents/school/ML/hw8/handout/examples/sample_returns.csv"
    rewards_empirical = "C:/Users/sneti/Documents/school/ML/hw8/handout/examples/tilerewardsgraph.csv"
    
    
    #initialize mountain car environment, weights, bias, and rewards list 
    mtcar = MountainCar(mode)
        
    weights = np.zeros((mtcar.state_space, mtcar.action_space))
    bias = 0
    
    rewards = np.zeros((episodes, 1))
    
    #loop through all episodes (reinitializing the mountain car each time)     
    for episode in range(episodes):   
        
        #initialize current state by resetting the car randomly 
        currentstatedict = mtcar.reset()
        
        #go through all keys in the dictionary to create an array of current state from sparse vectors
        currentstate = np.zeros((1, mtcar.state_space))
        for key in currentstatedict:
            currentstate[0][key] = currentstatedict[key] 
        
        #loop through all iterations to learn weights
        for iteration in range(maxiters):
            
            #pick random probability, if within epsilon, then take a random step
            prob = random.random()
            
            if prob < epsilon:
                bestaction = random.randrange(0,mtcar.action_space,1)
            
            #otherwise, create Q values for each action using the dot product of state * weights_action + bias
            #then pick the index (action) of the largest Q value
            else:
            
                Q = np.zeros((mtcar.action_space, 1))
                for action in range(len(Q)):
                    Q[action] = currentstate[0].dot(weights[:,action]) + bias
                bestaction = np.argmax(Q)
            
            #take a step in the direction of the best action
            nextstatedict, reward, doneflag = mtcar.step(bestaction)
            
            #take the dictionary for next state and convert to array again
            nextstate = np.zeros((1, mtcar.state_space))
            for key in nextstatedict:
                nextstate[0][key] = nextstatedict[key]
            
            #add rewards to list 
            rewards[episode] += reward 
            
            #do Q values again for this next state, same way as before
            Q2 = np.zeros((mtcar.action_space, 1))
            for action in range(len(Q2)):
                Q2[action] = nextstate[0].dot(weights[:,action]) + bias
            
            #but this time get the max Q value of the list, not argmax 
            Q2max = np.max(Q2)
            
            #update weights of the action taken based on the equation, using dQ/dw, which is just the currentstate vector
            weights[:,bestaction] = weights[:,bestaction] - alpha*(Q[bestaction] - (reward+gamma*Q2max))*currentstate[0]
            
            #update bias based on same equation, but dQ/db is just 1
            bias = bias - alpha*(Q[bestaction] - (reward+gamma*Q2max))
            
            #if game is over (reached the flag), update weights again and then close the game and break the loop
            if doneflag == True:
                mtcar.close()
                break 
            
            #otherwise, set current state to next state (since we've taken that step), and keep iterating 
            currentstate = nextstate 
                
    #print(weights)
    #print(bias)
    #print(rewards)
    
    #print out weights
    with open(weight_out, "w") as f:
        f.write("%s\n" % (str(bias[0])))
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                f.write("%s\n" % (str(weights[i][j])))
    
    #print out rewards 
    with open(returns_out, "w") as f:
        for i in range(len(rewards)):
            f.write("%s\n" % (str(rewards[i][0])))
        
    #mtcar.render() 
    
    k = 25
    
    final = pd.DataFrame(rewards)
    #calculate k-rolling average
    data = pd.DataFrame(np.array(rewards).T) 
    data = data.rolling(k, min_periods = 1).mean()       
    final["Avg"] = data.T
    
    final.to_csv(rewards_empirical)



if __name__ == "__main__":
    main(sys.argv)
