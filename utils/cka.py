# inspired by
# https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment/blob/master/CKA.py

import math
import torch
import numpy as np

class CKA(object):
    def __init__(self):
        pass 
    
    def centering(self, K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n
        return np.dot(np.dot(H, K), H) 

    def rbf(self, X, sigma=None):
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return np.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = X @ X.T
        L_Y = Y @ Y.T
        return np.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = np.sqrt(self.linear_HSIC(X, X))
        var2 = np.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = np.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = np.sqrt(self.kernel_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)

class CudaCKA_old(object):
    def __init__(self, device):
        self.device = device
    
    def centering(self, K): # K 16*16
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  # 16*16

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)

class CudaCKA(object):
    def __init__(self, device):
        self.device = device

    def mat_permute(self,X):
        return X.permute(0,1,3,2)

    def centering(self, K):
        n = K.shape[2] 
        unit = torch.ones([n, n], device=self.device) 
        I = torch.eye(n, device=self.device)
        H = I - unit / n 
        return torch.matmul(torch.matmul(H, K), H)  

    def linear_HSIC(self, X, Y): 
        L_X = torch.matmul(X, self.mat_permute(X)) 
        L_Y = torch.matmul(Y, self.mat_permute(Y))
        return torch.sum(self.centering(L_X) * self.centering(L_Y),dim=(2,3)) 

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y) 
        var1 = torch.sqrt(self.linear_HSIC(X, X)) 
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))
        # return torch.mean(hsic / (var1 * var2))
        return torch.sum(hsic / (var1 * var2))/(X.shape[0]*X.shape[1])

def cka_loss(net_partial_features: list, pret_partial_features: list,device):
    # net_partial_features[2].shape=torch.Size([16,128,16,16])
    del net_partial_features[0]
    del pret_partial_features[0]
    cuda_cka = CudaCKA(device)
    cka_loss = 0

    for X4d,Y4d in zip(net_partial_features,pret_partial_features):
        cka_loss += cuda_cka.linear_CKA(X4d,Y4d)
    return 1-cka_loss/len(net_partial_features) 

