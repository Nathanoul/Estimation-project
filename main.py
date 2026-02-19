import matplotlib.pyplot as plt
from functions import *

if __name__ == '__main__':
    q_true = quat_normalize(np.array([0.5, 0.5, 0.5, 0.5]))
    mu_true = 1e-5*np.ones(3)
    T = 600

    q0 = quat_normalize(np.array([0.0, 0.0, 0.0, 1]))
    mu0 = np.zeros(3)
    #q0 = quat_normalize(np.array([0.378, 0.756, 0.378, -0.378]))
    #mu0 = 200*np.ones(3) * np.pi/180 * 1/3600
    x = np.concatenate((q0, mu0))

    e1, n1 = simulate_filter(T, q_true, mu_true, init_guess=x, mode="AEKF1")
    e2, n2 = simulate_filter(T, q_true, mu_true, init_guess=x, mode="AEKF2")
    e3, n3 = simulate_filter(T, q_true, mu_true, init_guess=x, mode="AEKF3")

    t = np.arange(len(e1))*0.1

    plt.figure(figsize=(10,6))
    ax = plt.plot(t, e1, label="AEKF1")
    plt.plot(t, e2, label="AEKF2")
    plt.plot(t, e3, label="AEKF3")
    plt.yscale("log")

    plt.xlabel("Time (s)")
    plt.ylabel("Attitude Error (rad)")
    plt.title("Filter Comparison")
    plt.legend()
    plt.grid()
    plt.show()


    plt.figure(figsize=(10,6))
    plt.plot(t[2:], n1[2:], label="AEKF1")
    plt.plot(t[2:], n2[2:], label="AEKF2")
    plt.plot(t[2:], n3[2:], label="AEKF3")
    plt.yscale("log")

    plt.xlabel("Time (s)")
    plt.ylabel("q norm")
    plt.title("Filter Comparison")
    plt.legend()
    plt.grid()
    plt.show()