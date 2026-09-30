import numpy as np
from numpy import ndarray
from typing import Optional, Union

class Tau:
    def __init__(self, nx: int, nu: int, T: int, tau_init: Optional[ndarray]=None):
        self._nx = nx; self._nu = nu; self._T = T
        self._ntau = self._nx + self._nu
        self._var_dims = self._T*self._ntau + self._nx
        self._is_flatten: bool = False
        self._is_flatten_before_sync: Optional[bool] = None
        self._sync: bool = False
        self._build(tau_init)

    @classmethod
    def build_from_numpy(cls, tau: ndarray, nx: int, ntau: Optional[int]=None):
        if ntau is None:
            T_, ntau = tau.shape
            nu = ntau - nx; T = T_-1
        else:
            nu = ntau - nx
            tau = np.concatenate([tau, np.zeros(nu)])
            tau = tau.reshape(-1, ntau)
            T = tau.shape[0]-1
        return cls(nx, nu, T, tau)

    def sync(self, tau_other: 'Tau'):
        self._is_flatten_before_sync = self._is_flatten
        if tau_other.is_flatten: self.flatten()
        else: self.unflatten()
        self._sync = True

    def recover(self):
        if not self._sync: return
        if self._is_flatten_before_sync: self.flatten()
        else: self.unflatten()
        self._sync = False

    def __getitem__(self, t: int) -> ndarray:
        if self._is_flatten:
            return self._tau[t*self._ntau: (t+1)*self._ntau] # (ntau, )
        else:
            return self._tau[t, :] # (ntau, )
    
    def __add__(self, tau_other: 'Tau') -> 'Tau':
        tau_other.sync(self)
        _tau = self.tau + tau_other.tau
        tau_other.recover()
        return Tau.build_from_numpy(_tau, self._nx, ntau=self._ntau if self._is_flatten else None)

    def __rmul__(self, other: Union[int, float]):
        if isinstance(other, (int, float)):
            _tau = other * self._tau
            return Tau.build_from_numpy(_tau, self._nx, ntau=self._ntau if self._is_flatten else None)
        return NotImplemented

    def __mul__(self, other: Union[int, float]):
        if isinstance(other, (int, float)):
            return other * self
        return NotImplemented

    def __iadd__(self, tau_other: 'Tau') -> 'Tau':
        tau_other.sync(self)
        self._tau = self.tau + tau_other.tau
        tau_other.recover()
        return self

    def get_state(self, t: int) -> ndarray:
        return self._tau[t*self._ntau: t*self._ntau+self._nx] if self._is_flatten else \
        self._tau[t, :self._nx]

    def get_control(self, t: int) -> ndarray:
        return self._tau[t*self._ntau+self._nx: (t+1)*self._ntau] if self._is_flatten else \
        self._tau[t, self._nx:]
    
    def set_state_init(self, x0: ndarray): # (nx, )
        self._tau[0, :self._nx] = x0

    def set_tau(self, tau: ndarray):
        self._tau = tau
        return tau

    def set_state(self, x: ndarray, t: int): # (nx, )
        if self._is_flatten:
            self._tau[t*self._ntau: t*self._ntau+self._nx] = x
        else:
            self._tau[t, :self._nx] = x
        return x

    def set_control(self, u: ndarray, t: int): # (nu, )
        if self._is_flatten:
            self._tau[t*self._ntau+self._nx: (t+1)*self._ntau] = u
        else:
            self._tau[t, self._nx:] = u
        return u

    def flatten(self, is_clip: bool=True) -> ndarray:
        if self._is_flatten: return self._tau
        self._is_clip = is_clip
        self._is_flatten = True
        self._tau = self._tau.reshape(-1) # ((T+1)*ntau, )
        if is_clip: self._tau = self._tau[:self._var_dims]
        return self._tau

    def unflatten(self, is_clip: bool=True) -> ndarray:
        if not self._is_flatten: return self._tau
        if self._is_clip != is_clip: raise ValueError('clip value set exist conflict')
        self._is_flatten = False
        if is_clip: self._tau = np.concatenate([self._tau, np.zeros(self._nu)])
        self._tau = self._tau.reshape(self._T+1, self._ntau)
        return self._tau

    def _build(self, tau_init: Optional[ndarray]):
        # (T+1, ntau)
        self._tau = np.zeros([self._T+1, self._ntau]) if tau_init is None else tau_init.copy()
        self._is_flatten = False

    @property
    def tau(self):
        return self._tau
    
    @property
    def x(self):
        _recover: bool = False
        if self._is_flatten:
            _recover = True
            self.unflatten()
        _x = self._tau[:, :self._nx]
        if _recover: self.flatten()
        return _x # (T+1, nx)

    @property
    def u(self):
        _recover: bool = False
        if self._is_flatten:
            _recover = True
            self.unflatten()
        _u = self._tau[:self._T, self._nx:]
        if _recover: self.flatten()
        return _u # (T, nu)

    @property
    def is_flatten(self):
        return self._is_flatten
