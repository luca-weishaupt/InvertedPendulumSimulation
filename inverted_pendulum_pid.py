from io import BytesIO

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from numpy import *
from scipy.optimize import minimize
from tqdm import tqdm

GRAV = 9.8  # acceleration due to GRAV, positive is downward, m/sec^2
MARM = 1.0  # arm mass in kg
MROD = 0.1  # rod mass in kg
LROD = 0.5  # half the rod length in meters
TIME_STEP = 0.05  # time step in seconds
STEPS = 150
NSIMS = 10


class PID:
    def __init__(self, Kp, Kd, Ki):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.last_error = 0
        self.sum = 0

    def compute(self, time_delta, error):
        derivative = (error - self.last_error) / time_delta
        self.sum += error * time_delta
        self.last_error = error
        return self.Kp * error + self.Kd * derivative + self.Ki * self.sum


def find_error(theta):
    # We only care about the minimum angle between the equilibrium and the current angle
    new_error = theta % (2 * pi)
    if new_error > pi:
        new_error = new_error - 2 * pi
    return new_error


class ArmPole(object):
    # This function was adapted from
    # https://github.com/CodeReclaimers/neat-python/blob/master/examples/single-pole-balancing/cart_pole.py
    def __init__(
        self,
        x=None,
        theta=None,
        dx=None,
        dtheta=None,
        position_limit=2.4,
        angle_limit_radians=45 * pi / 180,
    ):
        self.position_limit = position_limit
        self.angle_limit_radians = angle_limit_radians

        if x is None:
            x = random.uniform(
                -0.5 * self.position_limit, 0.5 * self.position_limit
            )

        if theta is None:
            theta = random.uniform(
                -0.5 * self.angle_limit_radians, 0.5 * self.angle_limit_radians
            )

        if dx is None:
            dx = random.uniform(-1.0, 1.0)

        if dtheta is None:
            dtheta = random.uniform(-1.0, 1.0)

        self.t = 0.0
        self.x = x
        self.theta = theta

        self.dx = dx
        self.dtheta = dtheta

        self.xacc = 0.0
        self.tacc = 0.0

    def step(self, force):
        """
        Update the system state using leapfrog integration.
            x_{i+1} = x_i + v_i * dt + 0.5 * a_i * dt^2
            v_{i+1} = v_i + 0.5 * (a_i + a_{i+1}) * dt
        """
        # Locals for readability.
        mt = MROD + MARM
        dt = TIME_STEP

        # Remember acceleration from previous step.
        tacc0 = self.tacc
        xacc0 = self.xacc

        # Update position/angle.
        self.x += dt * self.dx + 0.5 * xacc0 * dt**2
        self.theta += dt * self.dtheta + 0.5 * tacc0 * dt**2

        # Compute new accelerations as given in "Correct equations for the dynamics of the cart-pole system"
        # by Razvan V. Florian (http://florian.io).
        # http://coneural.org/florian/papers/05_cart_pole.pdf
        st = sin(self.theta)
        ct = cos(self.theta)
        tacc1 = (
            GRAV * st
            + ct * (-force - MROD * LROD * self.dtheta**2 * st) / mt
        ) / (LROD * (4.0 / 3 - MROD * ct**2 / mt))
        xacc1 = (
            force + MROD * LROD * (self.dtheta**2 * st - tacc1 * ct)
        ) / mt

        # Update velocities.
        self.dx += 0.5 * (xacc0 + xacc1) * dt
        self.dtheta += 0.5 * (tacc0 + tacc1) * dt

        # Remember current acceleration for next step.
        self.tacc = tacc1
        self.xacc = xacc1
        self.t += dt


def simulate(params, AP, doublePID=True):
    xs = zeros(STEPS)
    thetas = zeros(STEPS)
    Fs = zeros(STEPS)
    ts = zeros(STEPS)
    angle_PID = PID(params[0], params[1], params[2])

    if doublePID:
        pos_PID = PID(params[3], params[4], params[5])
    else:
        pos_PID = PID(0, 0, 0)

    for i in range(STEPS):
        error = find_error(AP.theta)
        F = angle_PID.compute(TIME_STEP, error)
        F += pos_PID.compute(TIME_STEP, AP.x)
        AP.step(F)
        xs[i] = AP.x
        thetas[i] = AP.theta
        Fs[i] = F
        ts[i] = AP.t
    return xs, thetas, Fs, ts


def err_func(args, n_perturbations=10):
    # an error function that we try to minimize
    sum = 0
    for seed in range(n_perturbations):
        np.random.seed(seed)
        AP = ArmPole(
            x=np.random.uniform(-1.5, 1.5),
            dx=0,
            theta=np.random.uniform(-0.3, 0.3),
            dtheta=0,
        )
        xs, thetas, Fs, ts = simulate(args, AP)
        sum += np.log(np.sum(np.square(thetas) + np.square(xs)))
    return sum


def make_simulations(params, fname, simple_plot=False, doublePID=True):
    for seed in range(NSIMS):
        np.random.seed(seed)
        AP = ArmPole(
            x=np.random.uniform(-1.5, 1.5),
            dx=0,
            theta=np.random.uniform(-0.3, 0.3),
            dtheta=0,
        )
        xs, thetas, Fs, ts = simulate(params, AP, doublePID)

        if simple_plot:
            print(f"Initial anlge: {thetas[0]}\nInitial position:{xs[0]}")
            fig, ax = plt.subplots(3)
            ax[0].set_xlim(ts[0], ts[-1])
            ax[0].axes.xaxis.set_ticklabels([])
            ax[0].set_ylabel("theta [rad]")
            ax[0].set_ylim(np.min(thetas), np.max(thetas))
            ax[0].plot(ts, thetas, "blue")
            ax[1].set_xlim(ts[0], ts[-1])
            ax[1].axes.xaxis.set_ticklabels([])
            ax[1].set_ylim(np.min(xs), np.max(xs))
            ax[1].set_ylabel("X [m]")
            ax[1].plot(ts, xs, "blue")
            ax[2].set_xlim(ts[0], ts[-1])
            ax[2].set_ylim(np.min(Fs), np.max(Fs))
            ax[2].set_xlabel("time [s]")
            ax[2].set_ylabel("Force [N]")
            ax[2].plot(ts, Fs, "blue")
            plt.show()
            continue

        fig = plt.figure(figsize=(16, 9))
        gs = GridSpec(9, 16, figure=fig)
        ax1 = fig.add_subplot(gs[:, :9], projection="3d")
        ax2 = fig.add_subplot(gs[:3, 9:])
        ax3 = fig.add_subplot(gs[3:6, 9:])
        ax4 = fig.add_subplot(gs[6:, 9:])
        images = []
        ax1.set_axis_off()
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.set_zlim(-1, 1)
        ax2.set_xlim(ts[0], ts[-1])
        ax2.axes.xaxis.set_ticklabels([])
        ax2.set_ylabel("theta [rad]")
        ax2.set_ylim(np.min(thetas), np.max(thetas))
        ax3.set_xlim(ts[0], ts[-1])
        ax3.axes.xaxis.set_ticklabels([])
        ax3.set_ylim(np.min(xs), np.max(xs))
        ax3.set_ylabel("X [m]")
        ax4.set_xlim(ts[0], ts[-1])
        ax4.set_ylim(np.min(Fs), np.max(Fs))
        ax4.set_xlabel("time [s]")
        ax4.set_ylabel("Force [N]")

        for i in tqdm(range(len(xs))):
            # Data for a three-dimensional line
            phi = xs[i] / AP.position_limit * 2 * np.pi
            (a1,) = ax1.plot3D(
                [0, np.cos(phi)], [0, np.sin(phi)], [0, 0], "gray"
            )
            (a2,) = ax1.plot3D([0, 1], [0, 0], [0, 0], "red", linestyle="--")
            (a3,) = ax1.plot3D(
                [
                    np.cos(phi),
                    np.cos(phi) + 2 * LROD * np.sin(-phi) * np.sin(thetas[i]),
                ],
                [
                    np.sin(phi),
                    np.sin(phi) + 2 * LROD * np.cos(-phi) * np.sin(thetas[i]),
                ],
                [0, 2 * LROD * np.cos(thetas[i])],
                "black",
            )
            (a4,) = ax2.plot(ts[:i], thetas[:i], "blue")
            (a5,) = ax3.plot(ts[:i], xs[:i], "blue")
            (a6,) = ax4.plot(ts[:i], Fs[:i], "blue")
            buf = BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            images.append(imageio.imread(buf))
            a1.remove()
            a2.remove()
            a3.remove()
            a4.remove()
            a5.remove()
            a6.remove()

        imageio.mimsave(fname, images, duration=TIME_STEP)


def main():
    # guess = [18, 7, 11]
    guess = [
        31.6084103,
        7.21465755,
        27.33140888,
        3.5096769,
        1.3153921,
        0.04832309,
    ]
    guess = [30, 7, 30, 3.5, 1.3, 0.05]
    fname = "movie.gif"

    # make_simulations(guess, fname, True, doublePID=len(guess) == 6)

    fit = minimize(err_func, guess)
    print(fit)

    make_simulations(fit.x, fname, False, doublePID=len(guess) == 6)


if __name__ == "__main__":
    main()
