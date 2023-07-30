"""
Here are the matrices and parameters used in this class: 
F -> State Transition Matrix
B -> Control Input Matrix
Q -> Covariance of the Process Noise
R -> Covariance of the Measurement Noise
H -> Observation Matrix
S     ->  residual covarience
K     -> Kalman gain
x_pri -> priori state estimate
P_pri -> priori estimate covariance
x_post -> updated state estimate
P_post -> updated estiamte covariance
r_hat -> measurement pre-fit residual
"""
import numpy as np
from numpy.linalg import inv


class KalmanFilter:
    def __init__(self, F, B, H, Q, R):
        self.F = np.array(F)
        self.B = np.array(B)
        self.H = np.array(H)
        self.Q = np.array(Q)
        self.R = np.array(R)
        self.x_last = np.zeros([np.size(F, 1), 1])
        self.P_last = np.zeros(np.shape(F))
        self.theta_variance = []
        self.theta_dot_variance = []

    def predict_update(self, u, z):

        # z = measurement [theta]
        # u = control force [action]

        u = np.array(u)
        z = np.array(z)

        ## Predict
        x_pri = np.matmul(self.F, self.x_last) + self.B * u
        P_pri = np.matmul(np.matmul(self.F, self.P_last), np.transpose(self.F)) + self.Q

        ## Update
        r_hat = z - np.matmul(self.H, x_pri)
        S = np.matmul(np.matmul(self.H, P_pri), np.transpose(self.H)) + self.R
        K = np.matmul(np.matmul(P_pri, np.transpose(self.H)), inv(S))
        x_post = x_pri + np.matmul(K, r_hat)
        P_post = np.matmul((np.eye(2) - np.matmul(K, self.H)), P_pri)

        # Logging the variances from the P matrix for analysis purposes

        self.theta_variance.append(P_post[0][0])
        self.theta_dot_variance.append(P_post[1][1])

        ## Store for next step
        self.x_last = x_post
        self.P_last = P_post

        return x_post
    
    def get_variance(self):
        return self.theta_variance, self.theta_dot_variance