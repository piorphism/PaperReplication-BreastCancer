
# coding: utf-8

# In[1]:


import warnings
import sys
import time

import numpy as np
from scipy import linalg

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.extmath import logsumexp
from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics.pairwise import pairwise_kernels

class KernelFisher(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    Kernel Fisher Discriminant Analysis (KFDA)

    Parameters
    ----------
    
    sigma_sqrd:  float
    tol:  float
    kernel: "linear","poly","rbf","sigmoid" 
    degree : Degree for poly
    gamma : gamma as in LDA 
    coef0 : coefficient in poly and sigmoid
    """
    def __init__(self, sigma_sqrd=1e-8, tol=1.0e-3,
                 kernel="linear", gamma=None, degree=3, coef0=1):

        self.sigma_sqrd = sigma_sqrd
        self.tol = tol
        self.kernel = kernel.lower()
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self._centerer = KernelCenterer()

                
                
    @property
    def _pairwise(self):
        return self.kernel == "kerenl"
    
    def _get_kernel(self, X, Y=None):
        params = {"gamma": self.gamma,
                  "degree": self.degree,
                  "coef0": self.coef0}
        
        return pairwise_kernels(X, Y, metric=self.kernel,
                                    filter_params=True, **params)


    def fit(self, X, y):
        X, y = check_X_y(X, y) #does not accept sparse arrays
        self.classes_, y = np.unique( y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        n_samples_perclass = np.bincount(y)
                        
        self.means_ = []
        for ind in range(n_classes):
            Xg = X[y == ind, :]
            meang = Xg.mean(0)
            self.means_.append(np.asarray(meang))

        PI_diag = np.diag( 1.0*n_samples_perclass )                                        # shape(PI_diag) = n_classes x n_classes
        PI_inv = np.diag( 1.0 / (1.0*n_samples_perclass) )                                 # shape(PI_inv) = n_classes x n_classes
        PI_sqrt_inv = np.sqrt( PI_inv )                                                    # shape(PI_sqrt_inv) = n_classes x n_classes
        E=np.zeros( (n_samples,n_classes) )                                                # shape(E) = n_samples x n_classes
        E[[range(n_samples),y]]=1
        EPI = np.dot(E, PI_sqrt_inv)
        #One_minus_E_Pi_Et = np.identity(n_samples) - np.inner( E, np.inner(PI_diag, E).T ) # shape(One_minus_E_Pi_Et) = n_samples x n_samples
        C = self._get_kernel(X) 
        K_mean = np.sum(C, axis=1) / (1.0*C.shape[1])
        C = self._centerer.fit_transform(C)
        Uc, Sc, Utc, Sc_norm = self.svd_comp( C, self.tol, flag=True )
        reg_factor = self.sigma_sqrd * Sc_norm 
        St_reg_inv = np.inner( Uc, np.inner(np.diag(1.0/(Sc + reg_factor)), Utc.T).T )   
        R = np.inner(EPI.T, np.inner(C, np.inner( St_reg_inv, EPI.T ).T ).T )
        Vr, Lr, Vtr, Lr_norm =  self.svd_comp( R, tol=1e-6 )
        Z = np.inner( np.inner( np.inner( np.inner( np.diag(1.0 / np.sqrt(Lr)), Vtr.T), EPI), C.T), St_reg_inv)
        Z = (Z.T - (Z.sum(axis=1) / (1.0*Z.shape[1])) ).T
        self.Z = Z
        self.n_components_found_ = Z.shape[0]
        self.K_mean = K_mean
        self.X_fit_ = X
        
        return self

    def svd_comp(self, M, tol=1e-3, flag=False):
        U, S, Vt = linalg.svd(M, full_matrices=False)
        if flag:
            self.singular_vals = S

        S_norm = np.sum(S)
        rank = np.sum( (S/S_norm) > tol )

        return U[:,:rank], S[:rank], Vt[:rank,:], S_norm


    @property
    def classes(self):
        return self.classes_

    def _decision_function(self, X):
        return self.transform(X)

    def decision_function(self, X):
        return self._decision_function(X)

    def transform(self, X):
        
        k = self._get_kernel(X, self.X_fit_)
        z = np.inner(self.Z, (k-self.K_mean) ).T

        return z
        

    def fit_transform(self, X, y, sigma_sqrd=1e-8, tol=1.0e-3):

        return self.fit(X, y, sigma_sqrd=sigma_sqrd, tol=tol).transform(X)

