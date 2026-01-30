import numpy as np
from base import BaseModel

class SimpleForward(BaseModel):
    '''
    single propagating wave
    assumes high-frequency + heavily-damped limit where reflections are negligible.
    T(x,t) = Tbar + A * exp(-beta*x) * cos(omega*t - beta*x + phi)
    '''
    def forward(self, x, t):
        D = self.params['D']
        w = self.params['w']
        A = self.params['A']
        phi = self.params['phi']
        Tbar = self.params['Tbar']
        beta = np.sqrt(w / (2 * D))
        return Tbar + A * np.exp(-beta * x) * np.cos(w * t - beta * x + phi)

class FullSolution(BaseModel):
    '''
    finite rod, adiabatic boundary at x=L
    T(x,t) = Tbar + Re[A * exp(i(wt+phi)) * cosh(k(L-x)) / cosh(kL)]
    where k = sqrt(i*w / D)
    '''
    def forward(self, x, t):
        D = self.params['D']
        w = self.params['w']
        A = self.params['A']
        phi = self.params['phi']
        L = self.params['L']
        Tbar = self.params['Tbar']
        k = np.sqrt((1j * w) / D)
        thermal_wave = A * np.exp(1j * (w * t + phi)) * (np.cosh(k * (L - x)) / np.cosh(k * L))
        return Tbar + np.real(thermal_wave)

class FullSolutionNewton(BaseModel):
    '''
    finite rod, adiabatic boundary at x=L, with a newton cooling term, gamma
    dT/dt = D*d2T/dx2 - gamma(T-Tbar)
    T(x,t) = Tbar + Re[A*exp(i(wt+phi)) * cosh(k(L-x))/cosh(kL)]
    where k = sqrt((gamma + i*w)/D)
    '''
    def forward(self, x, t):
        D = self.params['D']
        w = self.params['w']
        A = self.params['A']
        phi = self.params['phi']
        L = self.params['L']
        gamma = self.params['gamma']
        Tbar = self.params['Tbar']
        k = np.sqrt((gamma + 1j * w) / D)
        thermal_wave = A * np.exp(1j * (w * t + phi)) * (np.cosh(k * (L - x)) / np.cosh(k * L))
        return Tbar + np.real(thermal_wave)

class FullSolutionRobin(BaseModel):
    '''
    finite rod, newton cooling w/ gamma and robin b.c. at x=L
    accounts for convective loss at the tip: -dT/dx = H(T-Tbar)
    where H = h/k_th
    '''
    def forward(self, x, t):
        D = self.params['D']
        w = self.params['w']
        A = self.params['A']
        phi = self.params['phi']
        L = self.params['L']
        gamma = self.params['gamma']
        H = self.params['H']
        Tbar = self.params['Tbar']
        k = np.sqrt((gamma + 1j * w) / D)
        num = k * np.cosh(k * (L - x)) + H * np.sinh(k * (L - x))
        den = k * np.cosh(k * L) + H * np.sinh(k * L)        
        thermal_wave = A * np.exp(1j * (w * t + phi)) * (num / den)
        return Tbar + np.real(thermal_wave)