from autograd import grad
from autograd import hessian as hess
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from tabulate import tabulate


MAX_X_VALUE = 1e12


class GradientDescent:
    def __init__(self, step, tolerance, max_iterations):
        self.step = step
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def optim(self, f, x0):
        x = np.array(x0, dtype=np.float64)
        grad_f = grad(f)
        iteration = 0
        values = []

        while iteration < self.max_iterations:
            gradient = grad_f(x)
            x_new = x - self.step * gradient
            if (x_new > MAX_X_VALUE).any():
                print("\033[91mThe maximum value of x has been exceeded\033[0m")
                return x_new, values
            if f(x) < self.tolerance:
                return x_new, values
            x = x_new
            values.append(f(x))
            iteration += 1

        return x, values


def q(x, alfa):
    alfa_exponents = (np.arange(len(x)) - 1) / (len(x) - 1)
    return np.sum(np.power(alfa, alfa_exponents) * np.power(x, 2))


def generate_report(alfa, x0, result, values, time):
    data = [
        ["Alfa", float(alfa)],
        ["x0", x0],
        ["Determined x", np.array2string(result, separator=", ")],
        ["Determined minimal value", float(values[-1])],
        ["Number of iterations", len(values)],
        ["Time", f"{time} s"],
    ]
    table = tabulate(data, tablefmt="grid", maxcolwidths=[25, 45])
    return table


def step_requirements(x0, alfa):
    x = np.array(x0, dtype=np.float64)
    for a in alfa:
        hessian_func = hess(lambda x: q(x, a))
        hessian = hessian_func(x)
        greatest_eigenvalue = np.abs(np.linalg.eigvals(hessian)).max()
        print(
            f"The largest eigenvalue modulus of the Hessian (alfa = {a}):",
            greatest_eigenvalue,
        )
        print(f"Oscillation when step > {1 / greatest_eigenvalue}")
        print(f"Discrepancy when step > {2 / greatest_eigenvalue}", "\n")


if __name__ == "__main__":
    x0 = [random.uniform(-100, 100) for _ in range(10)]
    alfas = [1, 10, 100]
    steps = [0.6, 0.1, 0.05, 0.01, 0.001]
    tolerance = pow(10, -6)
    max_iterations = pow(10, 4)

    step_requirements(x0, alfas)

    results = {}

    for step in steps:
        for alfa in alfas:
            gd = GradientDescent(
                step=step, tolerance=tolerance, max_iterations=max_iterations
            )
            start_time = time.time()
            result, values = gd.optim(lambda x: q(x, alfa), x0)
            results[(step, alfa)] = values
            end_time = time.time()
            plt.plot(range(len(values)), values, label=f"Alfa: {alfa}")
            print(
                f"Step = {step}, Max iterations = {max_iterations}, Tolerance = {tolerance}"
            )
            print(
                generate_report(alfa, x0, result, values, start_time - end_time), "\n\n"
            )
        plt.title(
            f"Alpha dependency chart\n(Step = {step}, Max iterations = {max_iterations}, Tolerance = {tolerance})"
        )
        plt.xlabel("Iteration number")
        plt.ylabel("Value of the target function")
        plt.yscale("log")
        plt.ylim(0, pow(10, 5))
        plt.legend()
        plt.show()

    for alfa in alfas:
        for step in steps:
            plt.plot(
                range(len(results[(step, alfa)])),
                results[(step, alfa)],
                label=f"Step: {step}",
            )
        plt.title(
            f"Step dependency chart\n(Alpha = {alfa}, Max iterations = {max_iterations}, Tolerance = {tolerance})"
        )
        plt.xlabel("Iteration number")
        plt.ylabel("Value of the target function")
        plt.yscale("log")
        plt.xscale("log")
        plt.ylim(0, pow(10, 6))
        plt.legend()
        plt.show()
