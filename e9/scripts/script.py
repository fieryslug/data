import numpy as np
from matplotlib import pyplot as plt
import json
from scipy import stats
from scipy.fftpack import fft, ifft
from scipy import optimize
from scipy.stats import linregress

g = 9.8

def spring(name):
    with open('data/spring.json', 'r') as f:
        dat = json.load(f)
    spr = dat[name]
    ms = spr['ms']
    m = np.array(spr['m']) / 1000
    y = np.array(spr['y']) / 100

    F = m * g

    slope, intcpt, r, p, err = stats.linregress(y, F)

    yy = np.linspace(min(y), max(y), 10)
    FF = slope * yy + intcpt
    plt.plot(yy, FF)
    
    plt.scatter(y, F, s=4)
    plt.grid()
    plt.title(name + ': ' + f'slope={slope}')
    plt.xlabel('y (m)')
    plt.ylabel('F (N)')

    plt.show()


def spring_d(name):
    with open('data/spring.json', 'r') as f:
        dat = json.load(f)
    spr = dat[name]
    ms = spr['ms']
    m = np.array(spr['m'])
    T20 = np.array(spr['T20'])




def read(fname):
    with open(fname, 'r') as f:
        s = f.read()
    d = s.split('\n')
    x = []
    t = []
    for d1 in d:
        if len(d) > 0:
            p = d1.split('.')
            x.append(float(p[1] + '.' + p[2]))
            t.append(float(p[3]))
    x = np.array(x)
    t = np.array(t)
    return x, t/1e6

def read2(fname):
    with open(fname, 'r') as f:
        s = f.read()
    d = s.split('\n')
    x = []
    t = []
    for d1 in d :
        if len(d) > 0:
            p = d1.split(',')
            x.append(float(p[1])/100)
            t.append(int(p[0])/1000)
    return np.array(x), np.array(t)


def test_func(t, a, b, c, d, e, h):
    return (a - b*t + h*t**2) * np.cos(c*t + d) + e


def proc(x, t):
    params, cov = optimize.curve_fit(test_func, t, x, p0=[0.1, 0.1, 2.86, 0, 0.45, 0])
    tt = np.linspace(min(t), max(t), 1000)
    xx = test_func(tt, *params)
    print(params)
    print(cov)
    plt.plot(tt, xx)
    plt.scatter(t, x, s=1)
    plt.grid()
    plt.xlabel('t(s)', fontsize=14)
    plt.ylabel('x(m)', fontsize=14)
    (a, b, c, d, e, h) = np.round(params, 2)
    plt.title(f'x-t plot\n$x(t)=({a}-{b}t)\cos({c}t+{d})+{e}$', fontsize=15)
    
    plt.show()  


def test_damp(t, b, A, phi, B):
    m = 0.46228
    k = 10.92
    return np.exp(-b * t / (2*m)) * A * np.cos(np.sqrt(4*m*k-b**2)/(2*m) * t + phi) + B
    

def proc_damp(x, t):
    m = 0.46228
    k = 10.92
    plt.scatter(t, x, s=1)
    params, cov = optimize.curve_fit(test_damp, t, x, p0=[1, 0.03, 0, 0.5])
    tt = np.linspace(min(t), max(t), 1000)
    xx = test_damp(tt, *params)
    b = params[0]
    omega = np.sqrt(4*m*k - b**2) / (2*m)
    print(f'T={np.pi * 2 / omega}')
    plt.grid()
    plt.xlabel('$t(s)$')
    plt.ylabel('$x(m)$')
    (b, A, phi, B) = np.round(params, 2)
    c0 = np.round(-b/(2*m), 2)
    c1 = np.round(np.sqrt(4*m*k-b**2)/(2*m), 2)
    plt.title(f'x-t plot\n$x(t) = e^{{{c0}t}} [{A} \cos({c1}t  {phi})] + {B}$')
    plt.plot(tt, xx)
    plt.show()

def test_friction(t, a, b, c):
    return a*t**2 + b*t + c


def friction(x, t):
    t = t[1:-5]
    x = x[1:-5]
    plt.scatter(t, x, s=1)
    plt.show()

    params, cov = optimize.curve_fit(test_friction, t, x, p0=[0, 0, 0])
    tt = np.linspace(min(t), max(t), 1000)
    xx = test_friction(tt, *params)
    plt.scatter(t, x, s=1)
    plt.plot(tt, xx)
    print(params[0])
    plt.grid()
    plt.xlabel('t(s)', fontsize=14)
    plt.ylabel('x(m)', fontsize=14)
    (a, b, c) = np.round(params, 3)
    plt.title(f'x-t plot\n$x(t) = {a}t^2 + {b}t + {c}$', fontsize=143)
    plt.show()


def proc_overdamp(x, t):
    plt.scatter(t, x, s=3)
    params, cov = optimize.curve_fit(test_overdamp, t, x, p0=[-0.1609, -0.844, 0.4])
    tt = np.linspace(min(t), max(t), 1000)
    xx = test_overdamp(tt, *params)
    plt.plot(tt, xx)
    plt.grid()
    (A, B, E) = np.round(params, 3)
    plt.title(f'$x(t) = {A}e^{{{B}t}}+ {E}$')
    plt.xlabel('t(s)')
    plt.ylabel('x(m)')
    plt.show()


def test_overdamp(t, A, B, E):
    return (A * np.exp(B*t)) + E


def tracker():
    with open('data/tracker', 'r') as f:
        s = f.read()
    d = s.split('\n')
    t = []
    x = []
    v = []
    for da in d:
        p = da.split()
        t.append(float(p[0]))
        x.append(float(p[1]))

    for i in range(len(t)):
        if i > 0 and i < len(t)-1:
            v.append((x[i+1] - x[i-1])/(t[i+1]-t[i-1]))
        else:
            v.append(np.nan)
    
    t = np.array(t)
    x = np.array(x)
    v = np.array(v)
    k = 9.6
    m = 0.13926
    

    params, cov = optimize.curve_fit(test_2, t, x, p0=[0.04, 10, 0, 0.04, 0])

    tt = np.linspace(min(t), max(t), 1000)
    xx = test_2(tt, *params)
    (A, C, D, E, F) = np.round(params, 3)
    plt.title(f'x-t plot\n$x(t) = {A} \cos({C}t+{D}) e^{{{F}t}}+{E}$')
    plt.grid()
    plt.xlabel('t(s)')
    plt.ylabel('x(m)')

    plt.plot(tt, xx)
    
    U = 0.5 * k * (x-params[3])**2
    T = 0.5 * m * v**2

    plt.scatter(t, x, s=4)
    plt.show()


    plt.scatter(t, v, s=4)
    plt.grid()
    plt.xlabel('t(s)')
    plt.ylabel('v(m/s)')
    plt.title('v-t plot')
    plt.show()


    plt.plot(t, U, label='U')
    plt.plot(t, T, label='T')
    plt.plot(t, T+U, label='E=T+U')
    plt.legend()
    plt.grid()
    plt.title('Energy - time plot')
    plt.xlabel('t(s)')
    plt.ylabel('energy(J)')
    params1, cov1 = optimize.curve_fit(test_U, t, U, p0=[0.03, 12, 0])
    UU = test_U(tt, *params1)
    
    plt.show()

    N = len(t)
    dt = (t[-1] - t[0]) / (N-1)
    dk = 2 * np.pi / (N * dt)
    M = N//2
    k = np.array([j * dk for j in range(0, N-M)] + [j * dk for j in range(-M, 0)])
    xf = fft(x)
    vf = 1j * k * xf
    print(k)
    for j in range(N):
        if np.abs(k[j]) > k[M * 6 // 8]:
            vf[j] = 0
    v = ifft(vf)
    plt.scatter(k, xf, s=2)
    plt.show()
    plt.scatter(t, v, s=2)
    #plt.plot(t, v)
    plt.show()




def test_2(t, A, C, D, E, F):
    return (A) * np.cos(C*t + D) * np.exp(F*t) + E


def test_U(t, A, B, C):
    return A * np.cos(B*t + C)**2


def spring():
    with open('data/spring/fat', 'r') as f:
        s = f.read()
    d = s.split('\n')
    l = []
    m = []
    for d1 in d:
        p = d1.split()
        l.append(float(p[0]))
        m.append(float(p[1]))
    l = np.array(l)
    m = np.array(m)
    F = m * g / 1000

    k, i, r, p, err = linregress(l, F)

    ll = np.linspace(min(l), max(l), 10)
    FF = k * ll + i
    plt.plot(ll, FF)

    plt.scatter(l, F, s=4)

    plt.grid()
    plt.xlabel('$x(m)$')
    plt.ylabel('$F(N)$')
    plt.title('F-x plot')
    plt.text(max(l)*0.9, max(F)*0.7, f'$F={np.round(k, 3)}x + {np.round(i, 3)}$\n$r^2={np.round(r**2, 3)}$')

    plt.show()

  

    with open('data/spring/fat_dyn', 'r') as f:
        s = f.read()
    d = s.split('\n')
    m = []
    N = []
    NT = []
    for d1 in d:
        p = d1.split()
        m.append(float(p[1]))
        N.append(float(p[2]))
        NT.append((float(p[3]) + float(p[4]))/2)
    m = np.array(m)
    N = np.array(N)
    NT = np.array(NT)
    T = NT / N

    mu = 4 * np.pi**2 * m
    k, i, r, p, err = linregress(T**2, mu)
    TT2 = np.linspace(min(T**2), max(T**2), 10)
    mmu = k * TT2 + i
    plt.plot(TT2, mmu)
    plt.scatter(T**2, mu, s=4)
    plt.grid()
    plt.xlabel('$T^2 (s^2)$')
    plt.ylabel('$4\pi^2 m (kg)$')
    plt.title('$4\pi^2m$ - $T^2$ plot')
    plt.text(0.548, 4.2, f'$4\pi^2m = {np.round(k, 3)}T^2 + ({np.round(i, 3)}) $\n$r^2={np.round(r**2, 3)}$')

    plt.show()
    

    

def test_lin(F, A, B):
    return A*F + B


def main():
    x, t = read2('data/overdamping')
    t = t
    proc_overdamp(x, t)


tracker()