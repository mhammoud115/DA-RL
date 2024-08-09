from os import path
from typing import Optional
import numpy as np

import gym
from gym import spaces
# from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled

# from gym.envs.classic_control import utils

# Set up some global constants, which shall be referred to later in the code...
# This is the initial condition for the reference simulation
DEFAULT_X_REF = 0.587276
DEFAULT_Y_REF = 0.563678
DEFAULT_Z_REF = 16.8708

# This is the initial condition for the actual solution
DEFAULT_X = 1.
DEFAULT_Y = 1.
DEFAULT_Z = 17.


class Lorenz63Env(gym.Env):
    """
       ### Description
    The Lorenz-63 system is a chaotic problem with a highly non-linear behavior.
    The system consists of three variables governed by three coupled equations.
    The states start at random position and the goal is to apply a correction such that the solution 
    follows that of a reference one. 

    
    The diagram below specifies the coordinate system used for the implementation of the pendulum's
    dynamic equations.
    ![Pendulum Coordinate System](./diagrams/pendulum.png)
    -  `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta` : angle in radians.
    - `tau`: torque in `N m`. Defined as positive _counter-clockwise_.
    
    
    ### Action Space
    The action is a `ndarray` with shape `(3,)` representing the correction applied to each of the variables.
    # TODO..................
    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   |    x   | -5.0 | 5.0 |
    | 0   |    y   | -5.0 | 5.0 |
    | 0   |    z   | -5.0 | 5.0 |

    ### Observation Space
    The observation is a `ndarray` with shape `(3,)` representing the x-y-z variables of the Lorenz 63 system
    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   |         x        | -5.0 | 5.0 |
    | 1   |         y        | -5.0 | 5.0 |
    | 2   |         z        | -5.0 | 5.0 |
    
    
    ### Rewards
    The reward function is defined as:
    TODO
    *r = -(theta<sup>2</sup> + 0.1 * theta_dt<sup>2</sup> + 0.001 * torque<sup>2</sup>)*
    where `$\theta$` is the pendulum's angle normalized between *[-pi, pi]* (with 0 being in the upright position).
    Based on the above equation, the minimum reward that can be obtained is
    *-(pi<sup>2</sup> + 0.1 * 8<sup>2</sup> + 0.001 * 2<sup>2</sup>) = -16.2736044*,
    while the maximum reward is zero (pendulum is upright with zero velocity and no torque applied).
    
    
    ### Starting State
    The starting state is Random TODO
    
    ### Episode Truncation
    The episode truncates at TODO time steps.
    
    ### Arguments
    - `\sigma`: 
    - `\rho`: 
    - `\beta`: 
    ```
    gym.make('Pendulum-v1', g=9.81)
    ```
    ### Version History
    * v0: Initial versions release (1.0.0)
    """

    metadata = {
        "render_modes": ["2D", "3D"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, sigma=10.0, rho=28.0, beta=8/3, integ_steps = 5):
        super(Lorenz63Env, self).__init__()

        # Set the environment variables
        self.num_steps = integ_steps    # assimilation frequency
        self.dt = 0.001                 # model timestep 
        self.counterSteps = 0           # Counter to link both
        
        self.sigma = sigma              # Lorenz 63 parameters
        self.rho = rho
        self.beta = beta
        
        self.render_mode = render_mode  # TODO

        self.screen_dim = 500           # TODO
        self.screen = None              # TODO
        self.clock = None               # TODO
        self.isopen = True              # TODO

        self.previousCost = -10000.

        # Initialize ranges for the states, this will be used to clip the predictions to a certain range
        # Observations are [x', y', z', x, y, z] x (num_steps - 1) then [xf-xo, yf-yo, zf-zo, x', y', z', x, y, z]
        # For POMDP, I no longer have any observations of one or more variables, so I have to remove those from the states...
        # Lets remove y first...
        low_Obs = np.tile([-4000., -4000., -4000., -60., -60., -10.], self.num_steps -1)
        low_Obs = np.concatenate((low_Obs, [-30., -30., -30., -4000., -4000., -4000., -60., -60., -10.]))

        high_Obs = np.tile([4000., 4000., 4000., 60., 60., 100.], self.num_steps -1)
        high_Obs = np.concatenate((high_Obs, [30., 30., 30., 4000., 4000., 4000., 60., 60., 100.]))

        self.lowObs  = np.array(low_Obs, dtype=np.float32)
        self.highObs  = np.array(high_Obs, dtype=np.float32)
        self.highAct = np.array([6., 6., 6.], dtype=np.float32)
        
        # Define the action and observational space: dimensions and range
        self.action_space = spaces.Box(low=-self.highAct, high=self.highAct, dtype=np.float32)
        self.observation_space = spaces.Box(low=self.lowObs, high=self.highObs, dtype=np.float32)

    def rk4singlestep(self, fun, dt, y0, sigma, rho, beta):
        """
            Long comments describing functions or other complicated
            classes can be left with the triple-quotes notation like this.
            
            This function does a single 4th-order Runge-Kutta step for ODE integration,
            where fun is the ODE, dt is the timestep, t0 is the current time, and y0 is
            the current initial condition. 
        """
        f1 = fun(y0, sigma, rho, beta)
        f2 = fun(y0 + (dt / 2) * f1, sigma, rho, beta)
        # f3 = fun(y0 + (dt / 2) * f2, sigma, rho, beta)
        # f4 = fun(y0 + dt * f3, sigma, rho, beta)
        # X1 = X0 + dt*K2
        yout = y0 + dt*f2
        # yout = y0 + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4)
        return yout
    
    def lorenz(self, y, sigma, rho, beta):
        """
            This function defines the dynamical equations
            that represent the Lorenz system. 
            
            Normally we would need to pass the values of
            sigma, beta, and rho, but we have already defined them
            globally above.
        """
        # y is a three dimensional state-vector
        dy = [sigma * (y[1] - y[0]), 
            y[0] * (rho - y[2]) - y[1],
            y[0] * y[1] - beta * y[2]]
        return np.array(dy)
       
    def step(self, action):
        '''
        Step is the function that says, given the current state, what is the best action and evolves the system by one step...
        '''
        *_, x, y, z = self.state
        *_, xR, yR, zR = self.state_ref

        # Specify the Lorenz63 parameters 
        sigma = self.sigma
        rho = self.rho
        beta = self.beta
        costs = 0

        # Specify the timestep of the RK2
        dt = self.dt

        temp  = []
        temp2 = []
        tempRD = []
        tmp_  = []
        self.last_action = action

        new_x = x # + np.random.normal(0, np.sqrt(1.))
        new_y = y # + np.random.normal(0, np.sqrt(1.))
        new_z = z # + np.random.normal(0, np.sqrt(1.))

        new_xR = xR
        new_yR = yR
        new_zR = zR        

        old_x = new_x
        old_y = new_y
        old_z = new_z

        old_xR = new_xR
        old_yR = new_yR
        old_zR = new_zR
        
        [new_x, new_y, new_z] = self.rk4singlestep(self.lorenz, dt, [old_x, old_y, old_z], sigma, rho, beta) + action
        # rk4singlestep(self.lorenz, dt, np.array([old_x, old_y, old_z]), sigma, rho, beta) + action
            
        new_xDot = sigma*(new_y - new_x)
        new_yDot = new_x*(rho - new_z) - new_y
        new_zDot = new_x*new_y - beta*new_z

        [new_xR, new_yR, new_zR] = self.rk4singlestep(self.lorenz, dt, [old_xR, old_yR, old_zR], sigma, rho, beta)
            
        new_x = np.clip(new_x, -60., 60.)
        new_y = np.clip(new_y, -60., 60.)
        new_z = np.clip(new_z, -10, 100.)

        new_xDot = np.clip(new_xDot, -4000., 4000.)
        new_yDot = np.clip(new_yDot, -4000., 4000.)
        new_zDot = np.clip(new_zDot, -4000., 4000.)

        temp.append(new_xDot)
        temp.append(new_yDot)
        temp.append(new_zDot)

        temp.append(new_x)
        temp.append(new_y)
        temp.append(new_z)

        temp2.append(new_xR)
        temp2.append(new_yR)
        temp2.append(new_zR)

        # Take a RL step, then, run the model for n_assim_steps-1 (relaxation)
        for i in range(self.num_steps -1):
            
            old_x = new_x
            old_y = new_y
            old_z = new_z

            old_xR = new_xR
            old_yR = new_yR
            old_zR = new_zR

            [new_x, new_y, new_z] = self.rk4singlestep(self.lorenz, dt, [old_x, old_y, old_z], sigma, rho, beta)
            
            new_xDot = sigma*(new_y - new_x)
            new_yDot = new_x*(rho - new_z) - new_y
            new_zDot = new_x*new_y - beta*new_z

            [new_xR, new_yR, new_zR] = self.rk4singlestep(self.lorenz, dt, [old_xR, old_yR, old_zR], sigma, rho, beta)

            new_x = np.clip(new_x, -60., 60.)
            new_y = np.clip(new_y, -60., 60.)
            new_z = np.clip(new_z, -10, 100.)

            new_xDot = np.clip(new_xDot, -4000., 4000.)
            new_yDot = np.clip(new_yDot, -4000., 4000.)
            new_zDot = np.clip(new_zDot, -4000., 4000.)

            # error += (new_x-new_xR) + (new_y-new_yR) + (new_z-new_zR)

            if i < self.num_steps -2:
                temp.append(new_xDot)
                temp.append(new_yDot)
                temp.append(new_zDot)

                temp.append(new_x)
                temp.append(new_y)
                temp.append(new_z)

                temp2.append(new_xR)
                temp2.append(new_yR)
                temp2.append(new_zR)
            
            if i == self.num_steps -2:
                temp.append(new_x - new_xR + np.random.normal(0, 1.))
                temp.append(new_y - new_yR + np.random.normal(0, 1.))
                temp.append(new_z - new_zR + np.random.normal(0, 1.))
                
                temp.append(new_xDot)
                temp.append(new_yDot)
                temp.append(new_zDot)

                temp.append(new_x)
                temp.append(new_y)
                temp.append(new_z)

                temp2.append(new_xR)
                temp2.append(new_yR)
                temp2.append(new_zR)
            
        self.state = np.array(temp)
        self.state_ref = np.array(temp2)
        self.state_refDerv = np.array(tempRD)

        if self.render_mode == "human":
            self.render()

        costs = ((new_x - new_xR)**2. + (new_y - new_yR)**2. + (new_z - new_zR)**2. + 1.e-7)/(3.)

        self.counterSteps += self.num_steps
        terminated = bool(self.counterSteps >= 10000000 or np.abs(new_x - new_xR) > 10. or np.abs(new_y - new_yR) > 10. or np.abs(new_z - new_zR) > 20.)

        # return self._get_obs(), self._get_ref(), -costs, False, False, {}
        return self._get_obs(), -costs, terminated, {}


    def reset(self, *, seed: Optional[int] = 20, options: Optional[dict] = None):
        # super().reset(seed=seed)
        # super().reset(seed=seed)
        np.random.seed()

        x = DEFAULT_X
        y = DEFAULT_Y  
        z = DEFAULT_Z
        
        temp = []
        low2 = []
        high2 = []
        for i in range(self.num_steps -1):
            temp.append(0.)
            temp.append(0.)
            temp.append(0.)

            temp.append(0.)
            temp.append(0.)
            temp.append(0.)

            low2.append(0.)
            low2.append(0.)
            low2.append(0.)

            high2.append(0.)
            high2.append(0.)
            high2.append(0.)

        low2.append(DEFAULT_X -1.)
        low2.append(DEFAULT_Y -1.)
        low2.append(DEFAULT_Z -2.)
        
        high2.append(DEFAULT_X +1.)
        high2.append(DEFAULT_Y +1.)
        high2.append(DEFAULT_Z +2.)
        
        self.state_ref = np.random.uniform(low=low2, high=high2)

        temp.append(0.)
        temp.append(0.)
        temp.append(0.)
        
        temp.append(0.)
        temp.append(0.)
        temp.append(0.)

        # temp.append(self.state_ref[-3] + np.random.rand()*0.8)
        # temp.append(self.state_ref[-2] + np.random.rand()*0.8)
        # temp.append(self.state_ref[-1] + np.random.rand()*0.8)
        temp.append(self.state_ref[-3])
        temp.append(self.state_ref[-2])
        temp.append(self.state_ref[-1])

        self.state = np.array(temp)

        # self.state = np.array([DEFAULT_X_REF + 5., DEFAULT_Y_REF + 5., DEFAULT_Z_REF + 5., 0.0, 0.0, 0.0])

        # self.state_ref = np.array([DEFAULT_X_REF, DEFAULT_Y_REF, DEFAULT_Z_REF])

        self.last_u = None
        self.counterSteps = 0
        self.previousCost = -10000

        if self.render_mode == "human":
            self.render()
        return self._get_obs()


    def _get_obs(self):
        # x, y, z, xDot, yDot, zDot = self.state
        return np.array(self.state, dtype=np.float32)


    def _get_ref(self, i):
        # xR, yR, zR = self.state_ref
        return np.array(self.state_ref, dtype=np.float32)

    def _get_refDerv(self):
        # xR, yR, zR = self.state_ref
        return np.array(self.state_refDerv, dtype=np.float32)



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



