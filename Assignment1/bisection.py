import numpy as np
def find_q(u, pa, t):
    error = 10.0
    left = float(pa)
    right = 1.0
    g = lambda y : pa*(np.log((pa+1e-15)/(y+1e-15))) + (1-pa)*np.log((1-pa)/(1-y+1e-15)) - np.log(t)/(u+1e-15)
    while abs(error) > 1e-2:
        middle = (left+right)/2.0
        print(left, right, error, np.sign(g(left)), np.sign(g(right)), np.sign(g(middle)))
        # if (abs(left - right)<1e-2):
        #     return middle
        if np.sign(g(left)) == np.sign(g(middle)):     
            left = middle
        elif np.sign(g(right)) == np.sign(g(middle)):
            right = middle
        error = g(middle)
    print(middle, g(middle))
    return middle

if __name__ == '__main__':
    print(find_q(1, 0.01 , 1))