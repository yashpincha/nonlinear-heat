import numpy as np
import pymc as pm
from base import BaseModel

class SimpleForward(BaseModel):
    '''
    single propagating wave
    assumes high-frequency + heavily-damped limit where reflections are negligible.
    T(x,t) = Tbar + A * exp(-beta*x) * cos(omega*t - beta*x + phi)
    '''
    def forward(self, x, t, params):
        D = params['D']
        w = params['w']
        A = params['A']
        phi = params['phi']
        Tbar = params['Tbar']
        beta = np.sqrt(w / (2 * D))
        return Tbar + A * np.exp(-beta * x) * np.cos(w * t - beta * x + phi)
    
    def build_model(self, x, t, data, sigma=None):
        with pm.Model() as model:
            D = pm.Uniform("D", lower=0.001, upper=0.1)
            w = pm.Uniform("w", lower=0.01, upper=10.0)
            A = pm.Uniform("A", lower=0.0, upper=10.0)
            phi = pm.Uniform("phi", lower=-np.pi, upper=np.pi)
            Tbar = pm.Uniform("Tbar", lower=0.0, upper=50.0)
            
            mu = self.forward(x, t, {'D': D, 'w': w, 'A': A, 'phi': phi, 'Tbar': Tbar})
            
            if sigma is None:
                sigma = pm.HalfNormal("sigma", sigma=1.0)            
            pm.Normal("obs", mu=mu, sigma=sigma, observed=data)
        return model


class FullSolution(BaseModel):
    '''
    finite rod, adiabatic boundary at x=L
    T(x,t) = Tbar + Re[A * exp(i(wt+phi)) * cosh(k(L-x)) / cosh(kL)]
    where k = sqrt(i*w / D)
    '''
    def forward(self, x, t, params):
        D = params['D']
        w = params['w']
        A = params['A']
        phi = params['phi']
        L = params['L']
        Tbar = params['Tbar']
        k = np.sqrt((1j * w) / D)
        thermal_wave = A * np.exp(1j * (w * t + phi)) * (np.cosh(k * (L - x)) / np.cosh(k * L))
        return Tbar + np.real(thermal_wave)
    
    def build_model(self, x, t, data, sigma=None):
        with pm.Model() as model:
            D = pm.Uniform("D", 0.001, 0.1)
            w = pm.Uniform("w", 0.01, 10.0)
            A = pm.Uniform("A", 0.0, 10.0)
            phi = pm.Uniform("phi", -np.pi, np.pi)
            L = pm.Uniform("L", 0.01, 1.0)
            Tbar = pm.Uniform("Tbar", 0.0, 50.0)
            
            mu = self.forward(x, t, {'D': D, 'w': w, 'A': A, 'phi': phi, 'L': L, 'Tbar': Tbar})
            
            if sigma is None:
                sigma = pm.HalfNormal("sigma", sigma=1.0)
            
            pm.Normal("obs", mu=mu, sigma=sigma, observed=data)
        return model


class FullSolutionNewton(BaseModel):
    '''
    finite rod, adiabatic boundary at x=L, with a newton cooling term, gamma
    dT/dt = D*d2T/dx2 - gamma(T-Tbar)
    T(x,t) = Tbar + Re[A*exp(i(wt+phi)) * cosh(k(L-x))/cosh(kL)]
    where k = sqrt((gamma + i*w)/D)
    '''
    def forward(self, x, t, params):
        D = params['D']
        w = params['w']
        A = params['A']
        phi = params['phi']
        L = params['L']
        gamma = params['gamma']
        Tbar = params['Tbar']
        k = np.sqrt((gamma + 1j * w) / D)
        thermal_wave = A * np.exp(1j * (w * t + phi)) * (np.cosh(k * (L - x)) / np.cosh(k * L))
        return Tbar + np.real(thermal_wave)
    
    def build_model(self, x, t, data, sigma=None):
        with pm.Model() as model:
            D = pm.Uniform("D", 0.001, 0.1)
            w = pm.Uniform("w", 0.01, 10.0)
            A = pm.Uniform("A", 0.0, 10.0)
            phi = pm.Uniform("phi", -np.pi, np.pi)
            L = pm.Uniform("L", 0.01, 1.0)
            gamma = pm.Uniform("gamma", 0.0, 1.0)
            Tbar = pm.Uniform("Tbar", 0.0, 50.0)
            
            mu = self.forward(x, t, {'D': D, 'w': w, 'A': A, 'phi': phi, 'L': L, 'gamma': gamma, 'Tbar': Tbar})
            
            if sigma is None:
                sigma = pm.HalfNormal("sigma", sigma=1.0)
            
            pm.Normal("obs", mu=mu, sigma=sigma, observed=data)
        return model


class FullSolutionRobin(BaseModel):
    '''
    finite rod, newton cooling w/ gamma and robin b.c. at x=L
    accounts for convective loss at the tip: -dT/dx = H(T-Tbar)
    where H = h/k_th
    '''
    def forward(self, x, t, params):
        D = params['D']
        w = params['w']
        A = params['A']
        phi = params['phi']
        L = params['L']
        gamma = params['gamma']
        H = params['H']
        Tbar = params['Tbar']
        k = np.sqrt((gamma + 1j * w) / D)
        num = k * np.cosh(k * (L - x)) + H * np.sinh(k * (L - x))
        den = k * np.cosh(k * L) + H * np.sinh(k * L)        
        thermal_wave = A * np.exp(1j * (w * t + phi)) * (num / den)
        return Tbar + np.real(thermal_wave)
    
    def build_model(self, x, t, data, sigma=None):
        with pm.Model() as model:
            D = pm.Uniform("D", 0.001, 0.1)
            w = pm.Uniform("w", 0.01, 10.0)
            A = pm.Uniform("A", 0.0, 10.0)
            phi = pm.Uniform("phi", -np.pi, np.pi)
            L = pm.Uniform("L", 0.01, 1.0)
            gamma = pm.Uniform("gamma", 0.0, 1.0)
            H = pm.Uniform("H", 0.0, 10.0)
            Tbar = pm.Uniform("Tbar", 0.0, 50.0)
            
            mu = self.forward(x, t, {'D': D, 'w': w, 'A': A, 'phi': phi, 'L': L, 'gamma': gamma, 'H': H, 'Tbar': Tbar})
            
            if sigma is None:
                sigma = pm.HalfNormal("sigma", sigma=1.0)
            
            pm.Normal("obs", mu=mu, sigma=sigma, observed=data)
        return model
