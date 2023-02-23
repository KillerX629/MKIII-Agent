import retro
import gym
from agent import *







"""Tamaño de la entrada:
    224  320  3
    height, width, channels
    
número de acciones posibles:
    8
    [izq,der]
"""


class BasicWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.past_action = [self.env.action_space.n]
        for i in  range (len(self.past_action)):
            self.past_action[i]=0

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        #comienzo de las modificaciones
        # dejamos guardada la última entrada para que se le pueda dar al agente en la observación
        self.past_action = action
        #fin de las modificaciones
        return next_state, reward, done, info






if __name__ == '__main__':
    env = BasicWrapper(retro.make(game="MortalKombat3-Genesis",
                    state="Level1.ShangTsungVsLiuKang",
                    scenario="scenario"))
    obs = env.reset()
    height, width, channels = env.observation_space.shape
    #print("", height, "", width, "", channels)
    actions = env.action_space.n
    agent = build_agent(build_model(actions),actions)
    try:
        agent.load_weights("mkiii.h5")
    except:
        FileNotFoundError("")
    agent.compile(optimizer=adam_v2.Adam(lr=1e-3), metrics=['mae'])
    
    agent.fit({"inputImage":obs, "inputPastInputs":env.past_action},env=env, nb_steps=1000000, visualize=True, verbose=2)

"""
while True:
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
        env.close()
        agent.save_weights("mkiii.h5", overwrite=True)
"""



class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, obs):
        #modificaciones a la observación
        
        #fin de modificaciones
        return obs
    #para usar el Wrapper:
    # env = ObservationWrapper(retro.make(scenario))




"""
    este código es una de las formas en las que podemos conseguir guardar
    los píxeles de la pantalla en variables.
    
    alto, ancho, canales = obs.shape
    
    acciones = env.action_space.n
    
    
    para conocer las acciones disponibles en el juego:
    env.unwrapped.get_action_meanings()
    """
    
"""
while True:
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
        env.close()
"""
