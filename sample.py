import numpy as np

"""
Migration of CJ's code into a regular python script.
Mild modifications applied.
"""

def f1(x):
    return abs(x)

def f2(x):
    return x * np.sin(5*x)

def f3(x):
    ret = []
    for i in x:
        if i > 0:
            ret.append(1 + 0.2 * np.sin(5*i))
        else:
            ret.append(0.2 * np.sin(5*i))
    return np.array(ret)

def f4(x1, x2):
    y1 = np.abs(x1 + x2)
    y2 = np.abs(x1 - x2)
    return [[y1], [y2]]

def sample_1d(f):
    st = -np.sqrt(3)
    ed = np.sqrt(3)
    x = np.random.uniform(st, ed, size=3000)
    y = f(x)
    return x, y

def sample_2d(f):
    st = -np.sqrt(3)
    ed = np.sqrt(3)
    x1 = np.random.uniform(st, ed, size=3000)
    x2 = np.random.uniform(st, ed, size=3000)
    y = f(x1, x2)
    return (x1, x2), y

