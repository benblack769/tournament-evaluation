import cvxpy as cp
import numpy as np

def max_entropy_nash(A):
    x = cp.Variable(A.shape[0])
    z = cp.Variable()

    bounds = [
        z - A.T @ x <= 0,
        cp.sum(x) == 1,
        x >= 0
    ]
    cp.Problem(cp.Maximize(z), bounds).solve()#solver='GLPK')
    minimax_value = z.value
    nash_result = x.value

    try:
        x = cp.Variable(A.shape[0])
        bounds = [
            minimax_value - A.T @ x <= 0,
            cp.sum(x) == 1,
            x >= 0
        ]
        cp.Problem(cp.Maximize(cp.sum(cp.entr(x))), bounds).solve()
        nash_result = x.value
    except cp.error.SolverError:
        # if the solver failed, then default back to the regular nash,
        # don't worry about the entropy maximation
        pass
    # print(x.value)
    # x = pd.Series(x.value, A.index)
    # y = pd.Series(bounds[0].dual_value, A.columns)
    return nash_result#, bounds[0].dual_value

if __name__ == "__main__":
    matrix = np.array([
        [0,1,-1],
        [-1,0,1],
        [1,-1,0],
    ])
    print(max_entropy_nash(matrix))

    matrix = np.array([
        [0,1,-1,-1],
        [-1,0,1,1],
        [1,-1,0,0],
        [1,-1,0,0],
    ])
    print(max_entropy_nash(matrix))
