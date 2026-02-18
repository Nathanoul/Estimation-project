import numpy as np

def quat_normalize(q):
    return q / np.linalg.norm(q)


def omega_matrix(w):
    w1, w2, w3 = w
    return np.array([
        [0,  w3, -w2, w1],
        [-w3, 0,  w1, w2],
        [w2, -w1, 0,  w3],
        [-w1, -w2, -w3, 0]
    ])


def ksi_matrix(q):
    e = q[:3]
    q4 = q[3]
    return np.array([
        [q4, -e[2], e[1]],
        [e[2], q4, -e[0]],
        [-e[1], e[0], q4],
        [-e[0], -e[1], -e[2]]
    ])


def quat_propagate(q, w, dt):
    Omega = omega_matrix(w)
    q_dot = 0.5 * Omega @ q
    return q + q_dot * dt

def cross_matrix(a):
    return np.array([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0]
    ])

def quat_to_A(q):
    e = q[:3]
    q4 = q[3]
    ex = cross_matrix(e)
    return (q4**2 - e @ e) * np.eye(3) + 2*np.outer(e,e) - 2*q4*ex


def true_omega(t):
    return np.sin(2*np.pi*t/150) * np.array([1,1,1]) * np.pi/180


def gyro_measurement(w, mu, sigma_eps, dt):
    noise = sigma_eps*np.sqrt(dt)*np.random.randn(3)
    return w + mu + noise


def star_measurement(q, r, sigma_b):
    A = quat_to_A(q)
    noise = sigma_b*np.random.randn(3)
    return A @ r + noise


def ekf_predict(x, P, w_meas, Q, dt):
    q = x[:4]
    mu = x[4:]

    w_est = w_meas - mu
    q_pred = quat_propagate(q, w_est, dt)
    mu_pred = mu

    x_pred = np.hstack((q_pred, mu_pred))

    F = np.zeros((7,7))
    F[0:4, 0:4] = 1/2 * omega_matrix(w_est)
    F[0:4, 4:7] = -1/2 * ksi_matrix(q_pred)
    F = np.eye(7) + dt * F

    G = np.zeros((7,6))
    G[0:4, 0:3] = -1/2 * ksi_matrix(q_pred)
    G[4:7, 3:6] = np.eye(3)

    P_pred = F @ P @ F.T + G @ Q @ G.T

    return x_pred, P_pred


def ekf_star_update(x, P, z, r, R):
    q = x[:4]
    e = x[:3]
    q4 = x[3]

    A = quat_to_A(q)
    h = A @ r

    H = np.zeros((3,7))
    rx = cross_matrix(r)
    ex = cross_matrix(e)
    H[0:3, 0:3] = -2*r @ e.T + 2*e @ r.T + 2*e.T @ r * np.eye(3) - 2*q4 * rx
    H[0:3, 3] = 2*q4 * r - 2*ex @ r

    y = z - h
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.pinv(S)


    x_upd = x + K @ y
    P_upd = (np.eye(7) - K @ H) @ P

    return x_upd, P_upd


def ekf_norm_update(x, P, sigma_nor):
    q = x[:4]

    g = q @ q
    r = 1 - g

    H = np.zeros((1,7))
    H[0,:4] = 2*q

    S = H @ P @ H.T + sigma_nor**2
    K = P @ H.T / S

    x_new = x + (K.flatten()*r)
    P_new = (np.eye(7) - K @ H) @ P

    return x_new, P_new


def attitude_error(q_true, q_est):
    A = quat_to_A(q_true)
    Ahat = quat_to_A(q_est)

    dA = A @ np.linalg.inv(Ahat)
    val = 0.5*(np.trace(dA)-1)
    val = np.clip(val, -1, 1)

    return np.arccos(val)


def simulate_filter(T, q_true, mu_true, init_guess, mode):

    dt = 0.1
    T = 600
    steps = int(T/dt)

    sigma_eps = 1e-6
    sigma_b = 1e-4
    sigma_nor = 1e-7

    Q = np.eye(6)*1e-8*dt
    R = np.eye(3)*sigma_b**2/dt

    r1 = np.array([1,0,0])
    r2 = np.array([0,1,0])

    P = 5*np.eye(7)

    x = init_guess

    errors = []
    norms_errors = []

    for k in range(steps):

        t = k*dt

        # true dynamics
        w = true_omega(t)
        q_true = quat_propagate(q_true, w, dt)
        q_true = quat_normalize(q_true)

        # gyro
        w_meas = gyro_measurement(w, mu_true, sigma_eps, dt)

        # predict
        x, P = ekf_predict(x, P, w_meas, Q, dt)

        # star tracker
        if k % int(2/dt) == 0:

            r = r1 if (k//int(2/dt))%2==0 else r2
            z = star_measurement(q_true, r, sigma_b)

            x, P = ekf_star_update(x, P, z, r, R)
        # mode-dependent step
        if mode == "AEKF2":
            x[:4] = quat_normalize(x[:4])

        if mode == "AEKF3":
            x, P = ekf_norm_update(x, P, sigma_nor)

        # log error
        q_norm_err = abs(np.linalg.norm(x[:4]) - 1)
        norms_errors.append(q_norm_err)
        err = attitude_error(q_true, x[:4])
        errors.append(err)

    return np.array(errors), np.array(norms_errors)
