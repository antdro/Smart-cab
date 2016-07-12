
# coding: utf-8

# In[ ]:

# %load Agent.py
import random
import numpy as np
import pandas as pd

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # TODO: Initialize any additional variables here
        self.Q_table = {}
        self.actions = ['forward', 'left', 'right', None]
        self.gamma = 0.5
        self.alpha = 0.75
        self.epsilon = [np.exp(-val) for val in np.linspace(2.5,6,100)]
        self.rand_count = 0
        self.steps = 0
        self.total_reward = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.rand_count +=1
        self.steps = 0
        self.total_reward = 0
                
    #find the best q_value and best action    
    def best_qvalue_action(self, state):
        best_action = random.choice(self.actions)
        best_q_value = self.Q_table.get((state, best_action), 0)
        for action in self.actions:
            self.Q_table[(state, action)] = self.Q_table.get((state, action), 0)
            if self.Q_table[(state, action)] > best_q_value:
                best_q_value = self.Q_table[(state, action)]
                best_action = action
                
        return best_q_value, best_action

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.steps += 1

        # TODO: Update state
        state = (inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], self.next_waypoint)
        
        # TODO: Select action according to your policy
        q_value, action = self.best_qvalue_action(state)  # best q_value for current state
        
        #choose action with probability less than epsilon
        if random.uniform(0,1) < self.epsilon[self.rand_count-1]:
            action = random.choice(self.actions)
        
        # Execute action and get reward
        reward = self.env.act(self, action)
        self.total_reward += reward
        
        # TODO: Learn policy based on state, action, reward
        
        #sense the environment
        inputs = self.env.sense(self)
        next_state = (inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], self.next_waypoint)
        
        #find the Q value for the next state
        next_q_value, next_action = self.best_qvalue_action(next_state)    
        
        #find the Q value for the current state
        self.Q_table[(state, action)] = (1 - self.alpha)*q_value + self.alpha*(reward + self.gamma*next_q_value)
        
        print "\nLearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, \ntotal steps = {}, \ntotal reward = {}\n"         .format(deadline, inputs, action, reward, self.steps, self.total_reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials
        
    # Now simulate it
    sim = Simulator(e, update_delay=0.05, display=False)  # create simulator (uses pygame when display=True, if available)
    
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    
    # return performance rate for currrent simulation
    #print 'Success: {}'.format(e.get_success()/float(100))

if __name__ == '__main__':
        run()

