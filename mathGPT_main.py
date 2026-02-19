import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Quaternion utilities (project-exact)
# ============================================================

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
    return quat_normalize(q + 0.5 * omega_matrix(w) @ q * dt)

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

# ============================================================
# True dynamics & sensors (project-exact)
# ============================================================

def true_omega(t):
    return np.sin(2*np.pi*t/150) * np.array([1,1,1]) * np.pi/180

def gyro_measurement(w, mu, sigma_eps, dt):
    return w + mu + sigma_eps*np.sqrt(dt)*np.random.randn(3)

def star_measurement(q, r, sigma_b):
    return quat_to_A(q) @ r + sigma_b*np.random.randn(3)

# ============================================================
# AEKF prediction & updates (unchanged)
# ============================================================

def ekf_predict(x, P, w_meas, Q, dt):
    q = x[:4]
    mu = x[4:]

    w_est = w_meas - mu
    q_pred = quat_propagate(q, w_est, dt)
    x_pred = np.hstack((q_pred, mu))

    F = np.zeros((7,7))
    F[0:4,0:4] = 0.5 * omega_matrix(w_est)
    F[0:4,4:7] = -0.5 * ksi_matrix(q_pred)

    G = np.zeros((7,6))
    G[0:4,0:3] = -0.5 * ksi_matrix(q_pred)
    G[4:7,3:6] = np.eye(3)

    Phi = np.eye(7) + F*dt
    P = Phi @ P @ Phi.T + G @ Q @ G.T * dt

    return x_pred, P

def ekf_star_update(x, P, z, r, R):
    q = x[:4]
    e = q[:3]
    q4 = q[3]

    h = quat_to_A(q) @ r
    rx = cross_matrix(r)
    ex = cross_matrix(e)

    H = np.zeros((3,7))
    H[:,0:3] = 2*(-np.outer(r,e) + np.outer(e,r) + (e@r)*np.eye(3) + q4*rx)
    H[:,3] = 2*(q4*r - ex@r)

    y = z - h
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    x = x + K @ y
    P = (np.eye(7) - K@H) @ P @ (np.eye(7) - K@H).T + K@R@K.T

    return x, P

def ekf_norm_update(x, P, sigma_nor):
    q = x[:4]
    r = 1 - q@q

    H = np.zeros((1,7))
    H[0,:4] = -2*q

    S = H @ P @ H.T + sigma_nor**2
    K = P @ H.T / S

    x = x + K.flatten()*r
    P = (np.eye(7) - K@H) @ P @ (np.eye(7) - K@H).T + K@K.T * sigma_nor**2

    return x, P

# ============================================================
# Attitude error
# ============================================================

def attitude_error(q_true, q_est):
    dA = quat_to_A(q_true) @ quat_to_A(q_est).T
    return np.arccos(np.clip((np.trace(dA)-1)/2, -1, 1))

# ============================================================
# MEKF helpers
# ============================================================

def small_angle_quat(dtheta):
    th = np.linalg.norm(dtheta)
    if th < 1e-12:
        return np.array([0,0,0,1])
    return np.hstack((dtheta/th*np.sin(th/2), np.cos(th/2)))

def quat_multiply(q1, q2):
    e1,s1 = q1[:3],q1[3]
    e2,s2 = q2[:3],q2[3]
    return np.hstack((s1*e2+s2*e1+np.cross(e1,e2), s1*s2-e1@e2))

# ============================================================
# Main simulation
# ============================================================

def run_all_filters(T=600):

    dt = 0.1
    steps = int(T/dt)

    sigma_eps = np.sqrt(1e-13)
    sigma_n = np.sqrt(1e-11)
    sigma_b = 10*np.pi/(180*3600)
    sigma_nor = 1e-5

    Q = np.diag([sigma_eps**2]*3 + [sigma_n**2]*3)
    R = sigma_b**2 * np.eye(3)

    r1 = np.array([1,0,0])
    r2 = np.array([0,1,0])

    q_true = quat_normalize(np.array([0.378,-0.378,0.756,0.378]))
    mu_true = 3e-5*np.ones(3)

    x0 = np.hstack((np.array([0,0,0,1]), np.zeros(3)))
    P0 = 5*np.eye(7)

    filters = {
        "AEKF1":[x0.copy(),P0.copy()],
        "AEKF2":[x0.copy(),P0.copy()],
        "AEKF3":[x0.copy(),P0.copy()]
    }

    q_mekf = x0[:4].copy()
    mu_mekf = np.zeros(3)
    P_mekf = 5 * np.eye(6)

    err = {k:[] for k in ["AEKF1","AEKF2","AEKF3","MEKF"]}
    norm_err = {k:[] for k in err}

    for k in range(steps):

        t = k*dt
        w = true_omega(t)
        q_true = quat_propagate(q_true, w, dt)
        mu_true += sigma_n*np.sqrt(dt)*np.random.randn(3)
        w_meas = gyro_measurement(w, mu_true, sigma_eps, dt)

        for name in filters:
            x,P = filters[name]
            x,P = ekf_predict(x,P,w_meas,Q,dt)

            if k % int(2/dt) == 0:
                r = r1 if (k//int(2/dt))%2==0 else r2
                z = star_measurement(q_true,r,sigma_b)
                x,P = ekf_star_update(x,P,z,r,R)

            if name=="AEKF2":
                x[:4] = quat_normalize(x[:4])
            if name=="AEKF3":
                x,P = ekf_norm_update(x,P,sigma_nor)

            filters[name] = [x,P]
            err[name].append(attitude_error(q_true,x[:4]))
            norm_err[name].append(abs(np.linalg.norm(x[:4])-1))

        # MEKF
        q_mekf = quat_propagate(q_mekf, w_meas-mu_mekf, dt)
        F = np.block([
            [-cross_matrix(w_meas-mu_mekf), -np.eye(3)],
            [np.zeros((3,3)), np.zeros((3,3))]
        ])
        P_mekf = (np.eye(6)+F*dt)@P_mekf@(np.eye(6)+F*dt).T + Q*dt

        if k % int(2/dt) == 0:
            r = r1 if (k//int(2/dt))%2==0 else r2
            z = star_measurement(q_true,r,sigma_b)

            Ahat = quat_to_A(q_mekf)
            h = Ahat @ r

            H = -np.hstack((cross_matrix(Ahat @ r), np.zeros((3, 3))))
            K = P_mekf@H.T@np.linalg.pinv(H@P_mekf@H.T+R)
            dx = K@(z-h)
            q_mekf = quat_multiply(q_mekf, small_angle_quat(dx[:3]))
            mu_mekf += dx[3:]
            P_mekf = (np.eye(6) - K@H) @ P_mekf @ (np.eye(6) - K@H).T + K@R@K.T

        err["MEKF"].append(attitude_error(q_true,q_mekf))
        norm_err["MEKF"].append(abs(np.linalg.norm(q_mekf)-1))

    return err, norm_err

# ============================================================
# Run & plot
# ============================================================

err, norm_err = run_all_filters()
t = np.linspace(0,6000,len(err["MEKF"]))

plt.figure(figsize=(9,5))
for k in err:
    plt.semilogy(t/60, err[k], label=k)
plt.xlabel("Time [min]")
plt.ylabel("Angular error [rad]")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,5))
for k in norm_err:
    plt.semilogy(t/60, norm_err[k], label=k)
plt.xlabel("Time [min]")
plt.ylabel(r"$|\|q\|-1|$")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()
