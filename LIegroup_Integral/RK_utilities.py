


def rk4_step_increament(func, t0, y0, h, f0=None):
    k1 = f0
    if k1 is None:
        k1 = func(t0, y0)
    half_dt = h * 0.5
    k2 = func(t0 + half_dt, y0 + half_dt * k1)
    k3 = func(t0 + half_dt, y0 + half_dt * k2)
    k4 = func(t0 + h, y0 + h * k3)
    return (k1 + 2 * (k2 + k3) + k4) * h / 6 