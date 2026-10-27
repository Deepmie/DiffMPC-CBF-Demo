from typing import Optional, Union
import torch
from torch import Tensor

class Tau:
    def __init__(
            self,
            nx: int,
            nu: int,
            T: int,
            tau_init: Optional[Tensor]=None,
            device: torch.device = torch.device('cpu'),
            dtype: torch.dtype = torch.float32,
        ):
        self._nx = nx; self._nu = nu; self._T = T
        self._ntau = self._nx + self._nu
        self._var_dims = self._T*self._ntau + self._nx
        self._device = device; self._dtype = dtype
        self.reset(tau_init)

    @classmethod
    def build_from_tensor(cls, tau: Tensor, nx: int, ntau: Optional[int]=None):
        _device, _dtype = tau.device, tau.dtype
        if ntau is None:
            T_, ntau = tau.shape
            nu = ntau - nx; T = T_-1
        else:
            nu = ntau - nx
            tau = torch.concat([tau, torch.zeros(nu, device=_device, dtype=_dtype)])
            tau = tau.reshape(-1, ntau)
            T = tau.shape[0]-1
        return cls(nx, nu, T, tau, _device, _dtype)

    def reset(self, tau_init: Optional[Tensor]):
        self._is_flatten: bool = False
        self._is_flatten_before_sync: Optional[bool] = None
        self._sync: bool = False
        self._build(tau_init)

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

    def __getitem__(self, t: int) -> Tensor:
        return self.get_tau(t)

    def __add__(self, tau_other: 'Tau') -> 'Tau':
        tau_other.sync(self)
        _tau = self.tau + tau_other.tau
        tau_other.recover()
        return Tau.build_from_tensor(_tau, self._nx, ntau=self._ntau if self._is_flatten else None)

    def __rmul__(self, other: Union[int, float]):
        if isinstance(other, (int, float)):
            _tau = other * self._tau
            return Tau.build_from_tensor(_tau, self._nx, ntau=self._ntau if self._is_flatten else None)
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

    def norm(self, p: int=2) -> float:
        return torch.norm(self._tau, p=p)

    def get_tau(self, t: int) -> Tensor:
        tau = self._tau[t*self._ntau: (t+1)*self._ntau] if self._is_flatten else \
        self._tau[t]
        return tau.unsqueeze(-1) # (ntau, 1)

    def get_state(self, t: int) -> Tensor:
        x = self._tau[t*self._ntau: t*self._ntau+self._nx] if self._is_flatten else \
        self._tau[t, :self._nx]
        return x.unsqueeze(-1) # (nx, 1)

    def get_control(self, t: int) -> Tensor:
        u = self._tau[t*self._ntau+self._nx: (t+1)*self._ntau] if self._is_flatten else \
        self._tau[t, self._nx:]
        return u.unsqueeze(-1) # (nu, 1)
    
    def set_state_init(self, x0: Tensor, need_shift: bool=False): # (nx, )
        self._tau[0, :self._nx] = x0
        if need_shift:
            self._tau[:self._T, self._nx:] = self._tau[1:self._T+1, self._nx:].clone()
        else:
            self._tau[:, self._nx:] = self._torch_zeros([self._T+1, self._nu])

    def set_tau(self, tau: Tensor):
        self._tau = tau
        return tau

    def set_state(self, x: Tensor, t: int): # (nx, )
        if x.ndimension() == 2: x = x.flatten()
        if self._is_flatten:
            self._tau[t*self._ntau: t*self._ntau+self._nx] = x
        else:
            self._tau[t, :self._nx] = x
        return x

    def set_control(self, u: Tensor, t: int): # (nu, )
        if u.ndimension() == 2: u = u.flatten()
        if self._is_flatten:
            self._tau[t*self._ntau+self._nx: (t+1)*self._ntau] = u
        else:
            self._tau[t, self._nx:] = u
        return u        

    def flatten(self, is_clip: bool=True) -> Tensor:
        if self._is_flatten: return self._tau
        self._is_clip = is_clip
        self._is_flatten = True
        self._tau = self._tau.reshape(-1) # ((T+1)*ntau, )
        if is_clip: self._tau = self._tau[:self._var_dims]
        return self._tau

    def unflatten(self, is_clip: bool=True) -> Tensor:
        if not self._is_flatten: return self._tau
        if self._is_clip != is_clip: raise ValueError('clip value set exist conflict')
        self._is_flatten = False
        if is_clip: self._tau = torch.concat([self._tau, self._torch_zeros(self._nu)])
        self._tau = self._tau.reshape(self._T+1, self._ntau)
        return self._tau

    def require_grad_(self, mode: bool=True) -> Tensor:
        return self._tau.requires_grad_(mode)

    def detach(self) -> 'Tau':
        _tau = Tau(self._nx, self._nu, self._T, self._tau.detach().clone(), self._device, self._dtype)
        _tau._is_flatten = self._is_flatten
        return _tau

    def to(self, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None) -> Tensor:
        self._tau = self._tau.to(device=device, dtype=dtype)
        if device is not None: self._device = device
        if dtype is not None: self._dtype = dtype
        return self

    @property
    def device(self) -> torch.device:
        return self._tau.device

    @property
    def dtype(self) -> torch.dtype:
        return self._tau.dtype

    def _build(self, tau_init: Optional[Tensor]):
        # (T+1, ntau)
        self._tau = self._torch_zeros([self._T+1, self._ntau]) \
        if tau_init is None else tau_init.clone()
        self._is_flatten = False

    def _torch_zeros(self, size) -> Tensor:
        return torch.zeros(size, device=self._device, dtype=self._dtype)

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
