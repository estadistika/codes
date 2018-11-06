import numpy as np

def simulate(n : int = 1000, w0 : float = -.3, w1 : float = -.7, alpha : float = 1 / 5) -> tuple:
    x = np.random.uniform(-1, 1, n)
    
    w = np.array([w0, w1])
    A = np.column_stack((np.ones(x.shape), x))
    
    y = A.dot(w) + np.random.normal(0, alpha, size = n)
    return (x, y, A, alpha)

def posterior(y : np.array, A : np.array, alpha : float, beta : float = (1 / 2)**2) -> tuple:
    sigmainv = alpha * A.T.dot(A) + beta * np.eye(2)
    sigma = np.linalg.inv(sigmainv)
    mu = alpha * sigma.dot(A.T).dot(y)
    return (mu, sigma)

x, y, A, a = simulate();
mu, sigma = posterior(y, A, a);