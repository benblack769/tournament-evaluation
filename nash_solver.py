import cvxpy as cp
import numpy as np

def equilibrium(A):
    x = cp.Variable(A.shape[0])
    z = cp.Variable()

    bounds = [
        z - A.T @ x <= 0,
        cp.sum(x) == 1,
        x >= 0]
    cp.Problem(cp.Maximize(z), bounds).solve()

    # x = pd.Series(x.value, A.index)
    # y = pd.Series(bounds[0].dual_value, A.columns)
    return x.value, bounds[0].dual_value

matrix = np.array([
    [0,1,-1],
    [-1,0,1],
    [1,-1,0],
])
print(equilibrium(matrix))

matrix = np.array([
    [0,1,-1,-1],
    [-1,0,1,1],
    [1,-1,0,0],
    [1,-1,0,0],
])
print(equilibrium(matrix))
