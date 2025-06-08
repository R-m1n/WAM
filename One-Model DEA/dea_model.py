# dea_model.py
import cvxpy as cp

def dea_one_model(X, Y, dmu_index, epsilon=1e-4):
    m, n = X.shape
    s = Y.shape[0]
    
    x0 = X[:, dmu_index]
    y0 = Y[:, dmu_index]

    phi = cp.Variable()
    lambdas = cp.Variable(n)
    s_c = cp.Variable(m, nonneg=True)
    s_plus_i2 = cp.Variable(m, nonneg=True)
    s_plus_r = cp.Variable(s, nonneg=True)

    constraints = []
    for i in range(m):
        constraints.append(x0[i] == X[i, :] @ lambdas + s_c[i] - s_plus_i2[i])

    for r in range(s):
        constraints.append(0 == Y[r, :] @ lambdas - phi * y0[r] - s_plus_r[r])

    constraints.append(cp.sum(lambdas) == 1)
    constraints.append(lambdas >= 0)

    obj = cp.Maximize(phi + epsilon * (-cp.sum(s_c) + cp.sum(s_plus_r) - cp.sum(s_plus_i2)))
    prob = cp.Problem(obj, constraints)
    
    prob.solve()

    return {
        "phi": phi.value,
        "lambdas": lambdas.value,
        "s_c": s_c.value,
        "s_plus_i2": s_plus_i2.value,
        "s_plus_r": s_plus_r.value
    }