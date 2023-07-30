import numpy as np


class PID_controller:
    def __init__(self):
        self.prev_action = 0
        
        self.Kp = 11
        self.Kd = 2
        self.Ki = 0.1

        self.timeStep = 0
        self.lastError = 0.0
        self.curr_err = 0.0

        self.integralTerm = 0.0

    def reset(self):
        self.prev_action = 0
        
        self.Kp = 11
        self.Kd = 2
        self.Ki = 0.1

        self.timeStep = 0
        self.lastError = 0.0
        self.curr_err = 0.0

        self.integralTerm = 0.0


    def get_action(self, state, image_state, random_controller=False):

        terminal, timestep, _, _, theta, theta_dot, reward = state

        if random_controller:
            return np.random.uniform(-1, 1)
        else:
            self.lastError = self.curr_err
            self.curr_err = theta
            dt = (timestep+1) - self.timeStep
            self.integralTerm += self.curr_err * dt
            self.timeStep = timestep + 1

            KpTerm = self.curr_err
            KdTerm =  theta_dot
            KiTerm = self.integralTerm

            action = (KpTerm * self.Kp) + (KiTerm * self.Ki) + (KdTerm * self.Kd)
            self.prev_action = action
            return action

    def get_action_with_disturbances(self, state, image_state, random_controller=False):

        terminal, timestep, x, x_dot, theta, theta_dot, reward = state

        if random_controller:
            return np.random.uniform(-1, 1)
        else:
            if np.random.rand() > 0.99:
                return 10
            else:
                self.lastError = self.curr_err
                self.curr_err = theta
                deltaTime = (timestep+1) - self.timeStep
                self.integralTerm += self.curr_err * deltaTime
                self.timeStep = timestep + 1

                KpTerm = self.curr_err
                KdTerm =  theta_dot
                KiTerm = self.integralTerm

                action = (KpTerm * self.Kp) + (KiTerm * self.Ki) + (KdTerm * self.Kd)
                self.prev_action = action
                return action