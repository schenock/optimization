import time

import cupy as np
import matplotlib.pyplot as plt


def moore_penrose_pinv(J: np.ndarray) -> np.ndarray:
    r"""
    Calculate A* (Moore-Penrose pseudoinverse).
    """
    return np.linalg.inv(J.T @ J) @ J.T


class GaussNewtonSolver(object):
    r"""
    Gauss-Newton solver

    argmin [sum(residual^2)], residual = f(x, theta) - y
    y - response vector
    theta - coeffs

    """

    def __init__(self, fn, **gn_params):

        self.fn = fn
        self.max_iter = gn_params['max_iter']
        self.min_rmse_diff = gn_params['min_rmse_diff']
        self.min_rmse = gn_params['min_rmse']

        self.theta0 = gn_params['theta0'] or None

        self.theta = None
        self.x = None
        self.y = None

    def fit(self, x, y, theta0):
        self.x = x
        self.y = y
        self.theta = theta0 if theta0 is not None else self.theta0
        return self._iterative_optim()

    def _iterative_optim(self):

        sum_time = 0

        rmse_p = np.inf
        for k in range(self.max_iter):
            s = time.perf_counter()
            residual = self._calc_residual()
            jacobian = self._calc_jacobian(analytical=False, eps=10 ** (-6))

            # beta update
            self.theta = self.theta - moore_penrose_pinv(jacobian) @ residual

            rmse = np.sqrt(np.sum(residual ** 2))
            if self._termination_condition(rmse, rmse_p, k):
                return sum_time / k
            rmse_p = rmse
            sum_time += (time.perf_counter() - s)

        print("Max iterations exceeded, optimization did not converge.")
        return sum_time / k

    def _termination_condition(self, rmse, rmse_p, k):
        if self.min_rmse_diff is not None:
            diff = np.abs(rmse_p - rmse)
            if diff < self.min_rmse_diff:
                print("RMSE diff smaller than diff: rmse={} prev_rmse={}".format(rmse, rmse_p))
                print("Total iterations: ", k)
                return True

        if self.min_rmse is not None and self.min_rmse > rmse:
            print("RMSE({}) smaller than: {}".format(rmse, self.min_rmse))
            print("Total iterations: ", k)
            return True
        return False

    @property
    def residual(self):
        return self._calc_residual()

    def _calc_residual(self):
        y_fit = self.fn(self.x, self.theta)
        return y_fit - self.y

    def _calc_jacobian(self, analytical=False, eps=10 ** (-6)):
        if analytical:
            raise NotImplementedError("Analytical compute of jacobian not implemented.")

        gradients = []
        for i in range(len(self.theta)):
            bt = np.zeros_like(self.theta).astype(float)
            bt[i] += eps
            gradient = (self.fn(self.x, self.theta + bt) - self.fn(self.x, self.theta - bt)) / (2 * eps)
            gradients.append(gradient)

        return np.column_stack(gradients)

    def get_estimate(self) -> np.ndarray:
        """
        Get estimated response
        """
        return self.fn(self.x, self.theta)


def sample_fn(x, theta):
    return theta[0] * x ** 3 + theta[1] * x ** 2 + theta[2] * x + 2 * np.cos(np.sqrt(x))


def test(max_iter, runs, size):
    x = np.array(range(1, size))
    y = sample_fn(x, theta=[1.1, 0.1, 0.5])
    theta0 = [10 ** (-6)] * 3

    gn_params = {
        'max_iter': max_iter,
        'min_rmse_diff': 10 ** (-6),
        'min_rmse': 10 ** (-9),
        'theta0': None
    }

    sum_avg = 0
    for i in range(runs):
        solver = GaussNewtonSolver(fn=sample_fn, **gn_params)

        avg_iter_time = solver.fit(x, y, theta0=theta0)
        print("Avg time per iter: {} for sample size: {}".format(avg_iter_time, len(x)))
        sum_avg += avg_iter_time

    return sum_avg / runs


def main():
    xx = []
    yy = []
    plt.figure(figsize=(11, 11))
    for size in [x for x in np.arange(100 ** 2, 1000 ** 2, 200 ** 2)]:
        result = test(max_iter=20, runs=5, size=size)
        xx.append(size.get())
        yy.append(result)

    plt.plot(xx, yy, linestyle='--', marker='o', color='r')

    plt.title("Cupy")
    axes = plt.gca()
    axes.set_ylim([0, 1])

    plt.xlabel("Sample size")
    plt.ylabel("Avg. iteration (s)")
    plt.show()


if __name__ == "__main__":
    main()
