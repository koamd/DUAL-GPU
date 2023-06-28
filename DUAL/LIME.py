import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
#from scipy.fft import *
from skimage import exposure
import cv2
from tqdm import trange

#scipy.fft.set_global_backend(cufft)

class LIME:
    # initiate parameters
    def __init__(self, iterations, alpha, rho, gamma, strategy, exact):
        self.iterations = iterations
        self.alpha = alpha
        self.rho = rho
        self.gamma = gamma
        self.strategy = strategy
        self.exact = exact

    # load pictures and normalize
    def load(self, imgPath):
        self.loadimage(cv2.imread(imgPath) / 255)

    # initiate Dx,Dy,DTD
    def loadimage(self,L):
        self.L = cp.array(L)
        self.row = self.L.shape[0]
        self.col = self.L.shape[1]

        self.T_esti = cp.max(self.L, axis=2)
        self.Dv = -cp.eye(self.row) + cp.eye(self.row, k=1)
        self.Dh = -cp.eye(self.col) + cp.eye(self.col, k=-1)

        dx = cp.zeros((self.row, self.col))
        dy = cp.zeros((self.row, self.col))
        dx[1, 0] = 1
        dx[1, 1] = -1
        dy[0, 1] = 1
        dy[1, 1] = -1

        dxf = cufft.fft2(dx)
        dyf = cufft.fft2(dy)
        
        self.DTD = cp.conj(dxf) * dxf + cp.conj(dyf) * dyf
        self.W = self.Strategy()

    # strategy 2
    def Strategy(self):
        if self.strategy == 2:
            self.Wv = 1 / (cp.abs(self.Dv @ self.T_esti) + 1)
            self.Wh = 1 / (cp.abs(self.T_esti @ self.Dh) + 1)
            return cp.vstack((self.Wv, self.Wh))
        else:
            return cp.ones((self.row * 2, self.col))

    # T subproblem
    def T_sub(self, G, Z, miu):
        X = G - Z / miu
        Xv = X[:self.row, :]
        Xh = X[self.row:, :]

        cp_input = 2 * self.T_esti + miu * (self.Dv @ Xv + Xh @ self.Dh)
        numerator_cp = cufft.fft2(cp_input)

        denominator_cp = cp.array(self.DTD * miu + 2)
        
        T = cp.asnumpy(cp.real(cufft.ifft2(numerator_cp / denominator_cp)))

        return cp.array(exposure.rescale_intensity(T, (0, 1), (0.001, 1)))

    # G subproblem
    def G_sub(self, T, Z, miu, W):
        epsilon = self.alpha * W / miu
        temp = cp.vstack((self.Dv @ T, T @ self.Dh)) + Z / miu
        return cp.sign(temp) * cp.maximum(cp.abs(temp) - epsilon, 0)

    # Z subproblem
    def Z_sub(self, T, G, Z, miu):
        return Z + miu * (cp.vstack((self.Dv @ T, T @ self.Dh)) - G)

    # miu subproblem
    def miu_sub(self, miu):
        return miu * self.rho

    def run(self):
        # accurate algorithm
        if self.exact:
            T = cp.zeros((self.row, self.col))
            G = cp.zeros((self.row * 2, self.col))
            Z = cp.zeros((self.row * 2, self.col))
            miu = 1

            for i in trange(0,self.iterations):
                T = self.T_sub(G, Z, miu)
                G = self.G_sub(T, Z, miu, self.W)
                Z = self.Z_sub(T, G, Z, miu)
                miu = self.miu_sub(miu)

            self.T = T ** self.gamma
            self.R = self.L / cp.repeat(self.T[..., None], 3, axis = -1)
            R_np = cp.asnumpy(self.R)
            return exposure.rescale_intensity(R_np,(0,1)) * 255
        # TODO: rapid algorithm
        else:
            pass
