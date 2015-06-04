import time
import numpy as np

x = np.random.normal(size=(23358, 3)).astype('float')
v = np.random.normal(size=(23358, 3)).astype('float')
buff = np.random.normal(size=(23358, 3)).astype('float')

def update(x, v, n):
  for dt in np.linspace(0, 1, n).astype('float'):
    temp = (dt * buff)
    x += temp
    v += temp

n_iter = 100000
t0 = time.time()
update(x, v, n_iter)
dt = time.time() - t0

ips = n_iter / dt
ips
