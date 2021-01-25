import numpy as np
from ple.games.waterworld import *
from ple import PLE
from tensorflow import keras
from keras import layers
from random import Random
import math

def inside(x, center_x, y, center_y, radius):
	return (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2

def get_x(radians, r, x):
	return r * math.cos(radians) + x

def get_y(radians, r, y):
	return r * math.sin(radians) + y

def get_function(x1, x2, y1, y2):
	return lambda x: ((x - x1) * (y2 - y1) / (x2 - x1)) + y1 

def getState(p, RADIUS):
	state = p.getGameState()
	score = p.score()

	pi = 3.14
	number_sensors = 30
	look_for_radius = 250
	player_x = state['player_x']
	player_y = state['player_y']
	sensors_functions = []
	extremums = []

	for i in range(1, number_sensors + 1):
		extreme_x = get_x((2 * pi / number_sensors) * i, look_for_radius, player_x)
		extreme_y = get_y((2 * pi / number_sensors) * i, look_for_radius, player_y)
		extremums.append([extreme_x, extreme_y])
		sensors_functions.append(get_function(player_x, extreme_x, player_y, extreme_y))

	sensors = np.zeros((number_sensors,1))

	for idx, f in enumerate(sensors_functions):
		for x_coord in range(min(int(player_x), int(extremums[idx][0]) + 1), max(int(player_x), int(extremums[idx][0]) + 1)):
			for index, creep_pos in enumerate(state['creep_pos']['GOOD']):
				if (0 <= x_coord <= 512) and inside(x_coord, creep_pos[0], f(x_coord), creep_pos[1], RADIUS):
					if sensors[idx] != 0:
						if abs(sensors[idx]) > state['creep_dist']['GOOD'][index]:
							sensors[idx] = state['creep_dist']['GOOD'][index] // 25
					else:
						sensors[idx] = state['creep_dist']['GOOD'][index] // 25


	for idx, f in enumerate(sensors_functions):
		for x_coord in range(min(int(player_x), int(extremums[idx][0]) + 1), max(int(player_x), int(extremums[idx][0]) + 1)):
			for index, creep_pos in enumerate(state['creep_pos']['BAD']):
				if (0 <= x_coord <= 512) and inside(x_coord, creep_pos[0], f(x_coord), creep_pos[1], RADIUS):
					if sensors[idx] != 0:
						if abs(sensors[idx]) > state['creep_dist']['BAD'][index]:
							sensors[idx] = -1 * (state['creep_dist']['BAD'][index] // 25)
					else:
						sensors[idx] = -1 * (state['creep_dist']['BAD'][index] // 25)

	sensors = np.array(sensors).flatten()

	# game_state = np.concatenate([np.array([state['player_x'], state['player_y'], state['player_velocity_x'], state['player_velocity_y']]), sensors])
	
	return sensors, score


class ReplayMemory(object):	
	def __init__(self, capacity):
		self.capacity = capacity
		self.states = []
	
	def add(self, state):
		if len(self.states) < self.capacity:
			self.states.insert(0, state)
		else:
			self.states.pop()
			self.states.insert(0, state)

	def perform_batch(self, agent):
		r = Random()
		random_memory_states = r.choices(self.states, k=100)
		states = np.array([i[0] for i in random_memory_states])
		targets = []

		for memorized in random_memory_states:
			action_taken = memorized[1]
			current_q_vals = agent.predict_move(memorized[0])[1] # current
			next_q_values = agent.predict_move(memorized[3])[1] # next 
			q_max = np.max(next_q_values)
			target = reward + gamma * q_max

			targets.append(np.array([i if index != agent.actions.index(action_taken) else target for index, i in enumerate(current_q_vals[0])]))
		
		targets = np.array([i for i in targets])

		return agent.model.train_on_batch(states, targets)
		

class Agent(object):
	def __init__(self, actions, model=None):
		self.actions = actions[:4]
		self.model = model

	def build_model(self):
		model = keras.Sequential(
    	[
	        layers.Dense(30, input_shape=(30,), activation="relu", name="layer1"),
	        layers.Dense(200, activation="relu", name="layer2"),
	        layers.Dense(4, activation="linear", name="layer3"),
    	])

		model.compile(loss=keras.losses.Huber(), optimizer='RMSprop')
		self.model = model	


	
	def predict_move(self, state):
		q_values = self.model.predict(np.array([state]))
		return [self.actions[np.argmax(q_values)], q_values]

	def rand_act(self):
		r = Random()
		return self.actions[r.randint(0,3)]


if __name__ == '__main__':

	fps = 30  
	num_steps = 1
	force_fps = True
	display_screen = True


	pygame.init()

	game = WaterWorld(width=512, height=512, num_creeps=10)
	p = PLE(game, fps=fps, num_steps=num_steps,
        force_fps=force_fps, display_screen=display_screen)

	RADIUS = game.AGENT_RADIUS
	agent = Agent(actions=p.getActionSet())
	agent.build_model()
	memory_size = 50000
	min_memory_size = 100
	memory = ReplayMemory(memory_size)
	epochs = 50
	num_steps = 1000
	epsilon = 1
	epsilon_min = 0.1
	epsilon_decay = (epsilon - epsilon_min) / 50000
	gamma = 0.9
	freq = 10
	rewards = {10:1, 9:2, 8:3, 7:4, 6:5, 5:6, 4:7, 3:8, 2:9, 1:10, 0:0}
	steps = 0

	for epoch in range(1, epochs + 1):
		steps = 0
		g = 0
		b = 0

		if p.game_over():
			p.init()

		while not p.game_over() and steps < num_steps:
			current_state, score = getState(p, RADIUS)
				
			r = Random()
			randint = r.uniform(0., 1.)
			if randint < epsilon:					# alegem actiunea random sau din experienta
				action = agent.rand_act()
			else:
				action = agent.predict_move(current_state)[0]

			p.act(action)

			next_state, next_score = getState(p, RADIUS)
			reward =  next_score - score

			
			if reward not in [-1, 1]:				# primim reward-ul in dependenta de distanta 
				reward -= 5
				for s in next_state:
					if s > 0:
						reward += rewards[s]
					else:
						reward -= rewards[abs(s)]
			else:
				if reward == 1:
					reward = 1000
					g += 1
				else:
					reward = -2000
					b += 1
				

			memory.add([current_state, action, reward, next_state])
		
			if len(memory.states) >= min_memory_size and steps % freq == 0:		# invatarea
				loss = memory.perform_batch(agent)
				print(steps)
			
			steps += 1
			if epsilon > epsilon_min:		# scadem probabilitatea de a alege actiunea random
				epsilon = epsilon - epsilon_decay

		print(epoch)
		print(g,b)

	agent.model.save('model_18_dec' + str(g))


