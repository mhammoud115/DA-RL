from os import path
from typing import Optional
import numpy as np

import scipy.io

import gym
from gym import spaces
# from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled

# from gym.envs.classic_control import utils

# Set up some global constants, which shall be referred to later in the code...
# This is the initial condition for the reference simulation
DEFAULT_REF = scipy.io.loadmat('IC_L96.mat')
DEFAULT_REF = DEFAULT_REF['Xa0'][0,:]

# This is the initial condition for the actual solution
DEFAULT_Stt = scipy.io.loadmat('IC_L96.mat')
DEFAULT_Stt = DEFAULT_Stt['Xa0'][0,:]

class Lorenz96Env(gym.Env):

    def __init__(self, render_mode: Optional[str] = None, Nx = 40, F = 7, integ_steps = 5):
        super(Lorenz96Env, self).__init__()

        # Set up the environment variables
        self.num_steps = integ_steps    # Assimilation Frequency
        self.dt = 0.05                  # Model Timestep
        self.Nx = Nx                    # Number of Lorenz Variables
        self.counterSteps = 0           # Counter to link both

        self.F = F                      # Forcing Term

        self.render_mode = render_mode  # TODO

        self.screen_dim = 500           # TODO
        self.screen = None              # TODO
        self.clock = None               # TODO
        self.isopen = True              # TODO

        self.previousCost = -10000.


        a = -400.*np.ones(Nx)
        b = -25. *np.ones(Nx)
        c = -20. *np.ones(Nx)
        
        self.lowObs  = np.concatenate((a,b,c))
        self.highObs = np.concatenate((-a,-b,-c))

        self.highAct  = 6. * np.ones(self.Nx, dtype=np.float32)

        self.action_space = spaces.Box(low=-self.highAct, high=self.highAct, dtype=np.float32)
        self.observation_space = spaces.Box(low=self.lowObs, high=self.highObs, dtype=np.float32)

    def rk4singlestep(self, fun, dt, y0, F, Nx):
        f1 = fun(y0, F, Nx)
        f2 = fun(y0 + (dt / 2.) * f1, F, Nx)
        f3 = fun(y0 + (dt / 2.) * f2, F, Nx)
        f4 = fun(y0 + dt * f3, F, Nx)
        
        yout = y0 + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4)
        
        return yout
   
    def lorenz96(self, y, F, Nx):
        
        dy = np.zeros(Nx)
        
        dy[0] = (y[1] - y[Nx-2]) * y[Nx-1] - y[0] + F
        dy[1] = (y[2] - y[Nx-1]) * y[0]    - y[1] + F
        for i in range(2, Nx-1):
            dy[i] = (y[i+1] - y[i-2])  * y[i-1] - y[i] + F
        dy[Nx-1]  = (y[0]   - y[Nx-3]) * y[Nx-2] - y[Nx-1] + F

        return np.array(dy)
    
    def step(self, action):
        
        x  = self.state[-40:]
        xR = self.state_ref[-40:]

        F  = self.F
        Nx = self.Nx
        costs = 0
        dt = self.dt

        temp  = []
        temp2 = []
        tempRD = []
        tmp_  = []
        self.last_action = action

        new_xDot  = np.zeros(Nx)
        new_xRDot = np.zeros(Nx)

        x_new = x
        xR_new = xR

        for i in range(self.num_steps):
            
            if i == 0 :
                x_old = x_new
                x_new = self.rk4singlestep(self.lorenz96, dt, x_old, F, Nx) + action
                x_dot = self.lorenz96(x_new, F, Nx)
                
                xR_old = xR_new
                xR_new = self.rk4singlestep(self.lorenz96, dt, xR_old, F, Nx)
            else:
                x_old = x_new
                x_new = self.rk4singlestep(self.lorenz96, dt, x_old, F, Nx)
                x_dot = self.lorenz96(x_new, F, Nx)
                
                xR_old = xR_new
                xR_new = self.rk4singlestep(self.lorenz96, dt, xR_old, F, Nx)

            if i == self.num_steps - 1:
                temp.append(x_new - xR_new + np.random.lognormal(0, 1., size=Nx))
                temp.append(x_dot)
                temp.append(x_new)

                temp2.append(xR_new)
        
        self.state = np.array(temp)
        self.state_ref = np.array(temp2)

        self.state = np.ndarray.flatten(self.state)
        self.state_ref = np.ndarray.flatten(self.state_ref)

        x = self.state[-40:]
        xR = self.state_ref[-40:]
        
        costs = np.sqrt(np.mean((x - xR)**2))

        terminated = bool(self.counterSteps >= 10000000)
        err = np.max(np.abs(x_new - xR_new), axis=0)
        if err > 25.:
            terminated = True

        self.counterSteps += self.num_steps

        return self._get_obs(), -costs, terminated, {}
    
    def reset(self):

        self.counterSteps = 0
        np.random.seed()

        temp  = []
        low2  = []
        high2 = []

        def_ref = DEFAULT_REF

        a = np.zeros(self.Nx)
        b = np.zeros(self.Nx)
        c = def_ref + np.random.randn(def_ref.shape)

        res_state      = np.concatenate((a,b,c))
        res_state_ref  = def_ref

        self.state     = res_state
        self.state_ref = res_state_ref

        self.last_u = None
        self.counterSteps = 0
        self.previousCost = -10000

        if self.render_mode == "human":
            self.render()
        
        return self._get_obs()

    
    def _get_obs(self):
        return np.array(self.state, dtype=np.float64)


    def _get_ref(self, i):
        return np.array(self.state_ref, dtype=np.float64)

    def _get_refDerv(self):
        return np.array(self.state_refDerv, dtype=np.float64)


    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )


    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False



