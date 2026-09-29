import jax
import numpy as np
from functools import partial
import jax.numpy as jnp

import immrax as irx
from .TVGPR import TVGPR # Import the TVGPR class for Gaussian Process Regression
from px4_rta_mm_gpr.utilities.jax_setup import jit


GRAVITY: float = 9.806

class NpToJaxDrainRing:
    """
    Collect rows in a fast mutable NumPy ring buffer.
    On drain(), return a jnp.array (oldest→newest) and reset to empty.
    """
    def __init__(self, capacity=50, cols=3, dtype=np.float64):
        self.capacity = capacity
        self.cols = cols
        self._data = np.zeros((capacity, cols), dtype=dtype)
        self._start = 0      # index of oldest row
        self._size = 0       # number of valid rows

    def add(self, row):
        """Append one row; overwrite oldest if full."""
        r = np.asarray(row)
        if r.shape != (self.cols,):
            raise ValueError(f"expected shape ({self.cols},), got {r.shape}")
        end = (self._start + self._size) % self.capacity
        self._data[end] = r
        if self._size < self.capacity:
            self._size += 1
        else:
            self._start = (self._start + 1) % self.capacity

    def extend(self, rows):
        for r in rows:
            self.add(r)

    def _chron_view(self):
        if self._size == 0:
            return self._data[0:0]
        s = self._start
        e = (self._start + self._size) % self.capacity
        if s < e:
            return self._data[s:e]
        return np.vstack((self._data[s:], self._data[:e]))

    def drain(self):
        """
        Return all rows as a jnp.array (oldest→newest) and reset buffer.
        Note: this converts/copies to device as needed.
        """
        out_np = self._chron_view().copy()
        # reset
        self._start = 0
        self._size = 0
        # optional: zero out data to avoid holding old values
        # self._data[:] = 0
        return jnp.array(out_np)
    
    def __call__(self):
        self.drain()

    # Subscriptable / iterable (chronological)
    def __len__(self): return self._size
    def __getitem__(self, idx): return self._chron_view()[idx]
    def __iter__(self): return iter(self._chron_view())
    def __repr__(self):
        return f"NpToJaxDrainRing(size={self._size}, cap={self.capacity}, data=\n{self._chron_view()}\n)"


## Planar Case
class PlanarMultirotorTransformed(irx.System) :
    def __init__ (self, mass):
        self.xlen = 5
        self.evolution = 'continuous'
        self.G = GRAVITY # gravitational acceleration in m/s^2
        self.M = mass # mass of the multirotor in kg

    def f(self, t, x, u, w_y, w_z): #jax version of Eq.30 in "Trajectory Tracking Runtime Assurance for Systems with Partially Unknown Dynamics"
        px, py, h, v, theta = x.ravel()
        u1, u2 = u.ravel()
        w_y = w_y[0]
        w_z = w_z[0]

        G = self.G # gravitational acceleration
        M = self.M # mass of the vehicle

        ydot = h*jnp.cos(theta) - v*jnp.sin(theta) # ydot = h*cos(theta) - v*sin(theta)
        zdot = h*jnp.sin(theta) + v*jnp.cos(theta) # zdot = h*sin(theta) + v*cos(theta)

        # depending on the distubrance direction in the sim we may need to flip the sign on the wz terms
        hdot = G*jnp.sin(theta) + (w_y/M)*jnp.cos(theta) + (w_z/M)*jnp.sin(theta) # hdot = G*sin(theta) + (wy/M)*cos(theta) + (wz/M)*sin(theta)
        vdot = G*jnp.cos(theta) - (w_y/M)*jnp.sin(theta) + (w_z/M)*jnp.cos(theta) - (u1/M)

        theta_dot = u2 # thetadot = u2


        return jnp.array([
            ydot,
            zdot,
            hdot,
            vdot,
            theta_dot
        ])
    

## Get mean of GP at a time based on height
def get_gp_mean(GP, t, x) :
    return GP.mean(jnp.array([t, x[1]])).reshape(-1)

## JIT: Get input from rollout feedforward input and error between rollout reference with LQR feedback
@jit
def u_applied(x, xref, uref, K_feedback, ulim):
    error = x - xref  # Compute the error between the reference and the current state
    u_fb = -K_feedback @ error
    u_total = u_fb + uref 
    u_clipped = jnp.clip(u_total, ulim.lower, ulim.upper)  # Clip the control input to the limits
    return u_clipped
 
## JIT: Collection idx function
# The certificate is SPATIAL: a row fails when the tube's position bounds (py, pz) are more than `threshold` (m) from
# the reference position, or the tube reaches the floor. `threshold` is a scalar or a per-axis pair (py, pz): the node
# uses 0.25 m + delta, delta = the position-estimate uncertainty (0 in simulation). Velocity and attitude bounds are not limited (a NaN in any
# state still fails: the rollout broke down).
SPATIAL = 2   # the first two states, py and pz


@jit
def collection_id_jax(xref, xemb, threshold=0.25, z_max=jnp.inf):
    n = xref.shape[1]
    diff1 = jnp.abs(xref[:, :SPATIAL] - xemb[:, :SPATIAL]) > threshold
    diff2 = jnp.abs(xref[:, :SPATIAL] - xemb[:, n:n + SPATIAL]) > threshold
    nan_mask = jnp.isnan(xref).any(axis=1) | jnp.isnan(xemb).any(axis=1)
    ground = xemb[:, xref.shape[1] + 1] > z_max  # tube's lowest altitude (upper pz bound, NED) below the floor
    fail_mask = diff1.any(axis=1) | diff2.any(axis=1) | nan_mask | ground

    # Safe handling using lax.cond
    return jax.lax.cond(
        jnp.any(fail_mask),
        lambda _: jnp.argmax(fail_mask),  # return first failing index
        lambda _: -1,                     # otherwise -1
        operand=None
    )


# Which planar state each wind GP is a function of (see TVGPR data in the node's wind callback):
#   y-wind (horizontal push) is learned as a function of altitude  pz = x[1]
#   z-wind (vertical push)   is learned as a function of lateral py = x[0]
WY_INPUT_IDX = 1
WZ_INPUT_IDX = 0


def _make_step(obs_wy, obs_wz, K_feed, K_reference, dt, perm, sys_mjacM, MASS, ulim, quad_sys, x_des, div=50,
               gp_feedforward=False):
    """Build the one-step update of the (embedding system, reference) pair.

    Shared by the full-horizon scan (jitted_rollout) and the early-exit loop
    (jitted_rollout_until_violation), so both compute identical rows.

    gp_feedforward: add the GP mean disturbance's component along the thrust axis to the reference thrust,
    u1 = M g + (wz cos(theta) - wy sin(theta)) - K (x - x_des). Without it the reference LQR (no integral action)
    settles where K_pz * e = wz, e.g. 0.98 N / 0.32 N/m = 3 m short of the goal in SITL.
    """
    GPY = TVGPR(obs_wy, sigma_f = 5.0, l=2.0, sigma_n = 0.01, epsilon = 0.25) # define the GP model for the disturbance in Y
    GPZ = TVGPR(obs_wz, sigma_f = 5.0, l=2.0, sigma_n = 0.01, epsilon = 0.25) # define the GP model for the disturbance in Z

    def mean_disturbance_wy(t, x) :
            return GPY.mean(jnp.hstack((t, x[WY_INPUT_IDX]))).reshape(-1)

    def mean_disturbance_wz(t, x) :
            return GPZ.mean(jnp.hstack((t, x[WZ_INPUT_IDX]))).reshape(-1)

    def sigma_wy_sq(t, x):
        return GPY.variance(jnp.hstack((t, x[WY_INPUT_IDX]))).reshape(-1)

    def sigma_wz_sq(t, x):
        return GPZ.variance(jnp.hstack((t, x[WZ_INPUT_IDX]))).reshape(-1)

    def sigma_bruteforce_both(t, ix):
        """Vectorized computation of sigma bounds for both Y and Z wind directions.

        Computes min/max of 3*sigma across the interval ix for both wind GPs.
        Optimized to share discretization and use vectorized operations.
        """
        x_array = ix.lower + ((ix.upper - ix.lower))*jnp.linspace(0., 1., div).reshape(-1, 1)

        # Vectorized computation using vmap instead of list comprehension
        sigma_wy_sq_vals = jax.vmap(lambda x: sigma_wy_sq(t, x))(x_array)
        sigma_wz_sq_vals = jax.vmap(lambda x: sigma_wz_sq(t, x))(x_array)

        sigma_wy_sq_scaled = 9.0 * sigma_wy_sq_vals  # 3.0**2 for 3-sigma bounds
        sigma_wz_sq_scaled = 9.0 * sigma_wz_sq_vals

        w_diff_Y = irx.interval(jnp.array([jnp.min(sigma_wy_sq_scaled)]), jnp.array([jnp.max(sigma_wy_sq_scaled)]))
        w_diff_Z = irx.interval(jnp.array([jnp.min(sigma_wz_sq_scaled)]), jnp.array([jnp.max(sigma_wz_sq_scaled)]))

        return w_diff_Y, w_diff_Z

    sigma_wy_sq_jacM = irx.jacM(sigma_wy_sq)
    sigma_wz_sq_jacM = irx.jacM(sigma_wz_sq)

    G_mjacM_Y = irx.mjacM(mean_disturbance_wy)
    G_mjacM_Z = irx.mjacM(mean_disturbance_wz)
    G_perm = irx.Permutation((0, 1, 2, 4, 5, 3))

    def step (carry, t) :
        xt_emb, xt_ref = carry #(py,pz,h,v,theta)

        xt_des = x_des # desired final state
        error = xt_ref - xt_des  # error between the current rollout reference state and the desired state
        nominal = -K_reference @ error  # nominal input based on the reference state and feedback gain

        ### Wind GP Interval Work (Y and Z directions computed together for efficiency)
        GP_mean_t_Y = GPY.mean(jnp.array([t, xt_ref[WY_INPUT_IDX]])).reshape(-1) # mean y-wind at the current time and altitude
        GP_mean_t_Z = GPZ.mean(jnp.array([t, xt_ref[WZ_INPUT_IDX]])).reshape(-1) # mean z-wind at the current time and lateral position

        hover = MASS * GRAVITY
        if gp_feedforward: # cancel the GP's mean disturbance along the thrust axis (see docstring)
            theta = xt_ref[4]
            hover = hover + GP_mean_t_Z[0] * jnp.cos(theta) - GP_mean_t_Y[0] * jnp.sin(theta)
        uG = nominal + jnp.array([hover, 0.0])  # Add the gravitational (+ disturbance) force to the nominal input
        u_ref_clipped = jnp.clip(uG, ulim.lower, ulim.upper)  # Clip the reference input to the input saturation limits

        MSY = sigma_wy_sq_jacM(irx.interval(0.), irx.ut2i(xt_emb))[1] # (1, 5) Jacobian bound of sigma^2 wrt x
        MSZ = sigma_wz_sq_jacM(irx.interval(0.), irx.ut2i(xt_emb))[1]

        xint = irx.ut2i(xt_emb) # buffer sampled sigma bound with lipschitz constant to recover guarantee
        x_div = (xint.upper - xint.lower)/(div*2) # half the sampling spacing along each state
        sigma_lip_Y = 9.0 * MSY.upper @ x_div.T # (1,) Lipschitz buffer for the sampled sigma^2 bound
        sigma_lip_Z = 9.0 * MSZ.upper @ x_div.T

        # Compute sigma bounds for both wind directions in one vectorized call
        w_diff_Y, w_diff_Z = sigma_bruteforce_both(t, irx.ut2i(xt_emb))
        # [0]: the (only) element. The original code indexed [1], which JAX silently clamps to [0].
        sig_upper_y = jnp.sqrt(w_diff_Y.upper + sigma_lip_Y[0])
        sig_upper_z = jnp.sqrt(w_diff_Z.upper + sigma_lip_Z[0])

        w_diffint_Y = irx.icentpert(0.0, sig_upper_y)
        w_diffint_Z = irx.icentpert(0.0, sig_upper_z)

        wint_Y = irx.interval(GP_mean_t_Y) + w_diffint_Y # type: ignore
        wint_Z = irx.interval(GP_mean_t_Z) + w_diffint_Z # type: ignore

        # Compute the mixed Jacobian inclusion matrix for the system dynamics function and the disturbance function
        Mt, Mx, Mu, MwY, MwZ = sys_mjacM( irx.interval(t), irx.ut2i(xt_emb), ulim, wint_Y, wint_Z,
                                    centers=((jnp.array([t]), xt_ref, u_ref_clipped, GP_mean_t_Y, GP_mean_t_Z),),
                                    permutations=(perm,))[0]

        _, MGY = G_mjacM_Y(irx.interval(jnp.array([t])), irx.ut2i(xt_emb),
                        centers=((jnp.array([t]), xt_ref,),),
                        permutations=(G_perm,))[0]

        _, MGZ = G_mjacM_Z(irx.interval(jnp.array([t])), irx.ut2i(xt_emb),
                        centers=((jnp.array([t]), xt_ref,),),
                        permutations=(G_perm,))[0]

        Mt = irx.interval(Mt)
        Mx = irx.interval(Mx)
        Mu = irx.interval(Mu)
        MwY = irx.interval(MwY)
        MwZ = irx.interval(MwZ)


        # Embedding system for reachable tube overapproximation due to state/input/disturbance uncertainty around the quad_sys_planar.f reference system under K_ref
        F = lambda t, x, u, wy, wz: (Mx + Mu@K_feed + MwY@MGY + MwZ@MGZ)@(x - xt_ref) + MwY@w_diffint_Y + MwZ@w_diffint_Z + quad_sys.f(0., xt_ref, u_ref_clipped, GP_mean_t_Y, GP_mean_t_Z) # with GP Jac
        embsys = irx.ifemb(quad_sys, F)
        xt_emb_p1 = xt_emb + dt*embsys.E(irx.interval(jnp.array([t])), xt_emb, u_ref_clipped, wint_Y, wint_Z)

        # Move the reference forward in time as well
        xt_ref_p1 = xt_ref + dt*quad_sys.f(t, xt_ref, u_ref_clipped, GP_mean_t_Y, GP_mean_t_Z)


        return ((xt_emb_p1, xt_ref_p1), (xt_emb_p1, xt_ref_p1, u_ref_clipped))

    return step


def _row_fails(xref_row, xemb_row, threshold, z_max=jnp.inf):
    """Same test as collection_id_jax, for one row: a position bound (py, pz) too far from the reference position,
    NaN, or the tube's lowest altitude (upper bound of pz in NED) below the floor z_max = -min_altitude."""
    n = xref_row.shape[0]
    return (jnp.any(jnp.abs(xref_row[:SPATIAL] - xemb_row[:SPATIAL]) > threshold)
            | jnp.any(jnp.abs(xref_row[:SPATIAL] - xemb_row[n:n + SPATIAL]) > threshold)
            | jnp.any(jnp.isnan(xref_row)) | jnp.any(jnp.isnan(xemb_row))
            | (xemb_row[n + 1] > z_max))


## JIT: Rollout function (full horizon)
@partial(jax.jit, static_argnames=['T', 'dt', 'perm', 'sys_mjacM', 'MASS', 'ulim', 'quad_sys', 'gp_feedforward'])
def jitted_rollout(t_init, ix, xc, K_feed, K_reference, obs_wy, obs_wz, T, dt, perm, sys_mjacM, MASS, ulim, quad_sys, x_des=jnp.array([0., -2.4, 0., 0., 0.]), gp_feedforward=False):
    step = _make_step(obs_wy, obs_wz, K_feed, K_reference, dt, perm, sys_mjacM, MASS, ulim, quad_sys, x_des,
                      gp_feedforward=gp_feedforward)
    tt = jnp.arange(0, T, dt) + t_init # define the time horizon for the rollout

    final_carry, (embedding_sys_traj, reference_traj, control_traj) = jax.lax.scan(step, (irx.i2ut(ix), xc), tt)

    # Return with initial conditions prepended
    embedded_states_full = jnp.vstack((irx.i2ut(ix), embedding_sys_traj))
    reference_states_full = jnp.vstack((xc, reference_traj))
    control_inputs_full = jnp.vstack(control_traj)
    return embedded_states_full, reference_states_full, control_inputs_full


def rollout_until_violation(t_init, ix, xc, K_feed, K_reference, obs_wy, obs_wz, x_des, threshold, *, n_steps, dt,
                            perm, sys_mjacM, MASS, ulim, quad_sys, margin_steps, z_max=jnp.inf,
                            gp_feedforward=False):
    """Early-exit rollout: integrate only until the tube first leaves the certification threshold, plus a margin.

    The scan is causal, so every computed row is IDENTICAL to the corresponding row of jitted_rollout and the
    returned violation index equals collection_id_jax on the full rollout (when it lies within the horizon).
    Rows beyond ``n_valid`` are NaN. Cost is proportional to (violation index + margin), not to the horizon.

    Returns: tube (n_steps+1, 10), ref (n_steps+1, 5), u (n_steps, 2), violation_idx (-1 if none), n_valid
    """
    step = _make_step(obs_wy, obs_wz, K_feed, K_reference, dt, perm, sys_mjacM, MASS, ulim, quad_sys, x_des,
                      gp_feedforward=gp_feedforward)
    x0_emb = irx.i2ut(ix)
    tube = jnp.full((n_steps + 1, x0_emb.shape[0]), jnp.nan).at[0].set(x0_emb)
    ref = jnp.full((n_steps + 1, xc.shape[0]), jnp.nan).at[0].set(xc)
    u = jnp.full((n_steps, 2), jnp.nan)

    fail0 = _row_fails(xc, x0_emb, threshold, z_max)
    viol0 = jnp.where(fail0, 0, -1)
    stop0 = jnp.where(fail0, margin_steps, n_steps)

    def cond(c):
        i, _, _, _, _, _, _, stop = c
        return (i < n_steps) & (i < stop)

    def body(c):
        i, x_emb, x_ref, tube, ref, u, viol, stop = c
        (x_emb1, x_ref1), (_, _, u_i) = step((x_emb, x_ref), t_init[0] + i * dt)
        tube = tube.at[i + 1].set(x_emb1)
        ref = ref.at[i + 1].set(x_ref1)
        u = u.at[i].set(u_i)
        first_fail = (viol < 0) & _row_fails(x_ref1, x_emb1, threshold, z_max)
        viol = jnp.where(first_fail, i + 1, viol)
        stop = jnp.where(first_fail, jnp.minimum(i + 1 + margin_steps, n_steps), stop)
        return (i + 1, x_emb1, x_ref1, tube, ref, u, viol, stop)

    i_end, _, _, tube, ref, u, viol, _ = jax.lax.while_loop(
        cond, body, (jnp.array(0), x0_emb, xc, tube, ref, u, viol0, stop0))
    return tube, ref, u, viol, i_end + 1


## JAX Linearization Function
@partial(jit, static_argnums=0)
def jitted_linearize_system(sys, x0, u0, w0y, w0z):
    """Compute the Jacobian of the system dynamics function with respect to state and input at the initial conditions."""
    A, B = jax.jacfwd(sys.f, argnums=(1, 2))(0, x0, u0, w0y, w0z) # Compute the Jacobian of the system dynamics function with respect to state and input at the initial conditions
    return A, B

# ## 3D Case
# class ThreeDMultirotorTransformed(irx.System):
#     def __init__(self, mass):
#         self.xlen = 9
#         self.evolution = 'continuous'
#         self.G = GRAVITY  # gravitational acceleration in m/s^2
#         self.M = mass  # mass of the multirotor in kg
#         self.C = jnp.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 1, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 1, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 1]])

#     def f(self, t, state, input, w):
#         """Quadrotor dynamics. xdot = f(x, u, w)."""
#         x, y, z, vx, vy, vz, roll, pitch, yaw = state
#         curr_thrust = input[0]
#         body_rates = input[1:]
#         GRAVITY = self.G
#         MASS = self.M
#         wz = w # horizontal wind disturbance as a function of height

#         T = jnp.array([[1, jnp.sin(roll) * jnp.tan(pitch), jnp.cos(roll) * jnp.tan(pitch)],
#                         [0, jnp.cos(roll), -jnp.sin(roll)],
#                         [0, jnp.sin(roll) / jnp.cos(pitch), jnp.cos(roll) / jnp.cos(pitch)]])
#         curr_rolldot, curr_pitchdot, curr_yawdot = T @ body_rates

#         sr = jnp.sin(roll)
#         sy = jnp.sin(yaw)
#         sp = jnp.sin(pitch)
#         cr = jnp.cos(roll)
#         cp = jnp.cos(pitch)
#         cy = jnp.cos(yaw)

#         vxdot = -(curr_thrust / MASS) * (sr * sy + cr * cy * sp)
#         vydot = -(curr_thrust / MASS) * (cr * sy * sp - cy * sr)
#         vzdot = GRAVITY - (curr_thrust / MASS) * (cr * cp)

#         return jnp.hstack([vx, vy, vz, vxdot, vydot, vzdot, curr_rolldot, curr_pitchdot, curr_yawdot])

# quad_sys_3D = ThreeDMultirotorTransformed(mass=1.75)
# ulim_3D = irx.interval([0, -1, -1, -1], [21, 1, 1, 1])  # Input saturation interval -> 0 <= u1 <= 20, -1 <= u2 <= 1, -1 <= u3 <= 1
# Q_3D = jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1]) * jnp.eye(quad_sys_3D.xlen)  # weights that prioritize overall tracking of the reference (defined below)
# R_3D = jnp.array([1, 1, 1, 1]) * jnp.eye(4)  # weights for the control input (thrust and body rates)


if __name__ == "__main__":
    import control

    GP_instantiation_values = jnp.array([[-2, 0.0], #make the second column all zeros
                                        [0, 0.0],
                                        [2, 0.0],
                                        [4, 0.0],
                                        [6, 0.0],
                                        [8, 0.0],
                                        [10, 0.0],
                                        [12, 0.0]]) # at heights of y in the first column, disturbance to the values in the second column
    # add a time dimension at t=0 to the GP instantiation values for TVGPR instantiation
    actual_disturbance_GP = TVGPR(jnp.hstack((jnp.zeros((GP_instantiation_values.shape[0], 1)), GP_instantiation_values)), 
                                        sigma_f = 5.0, 
                                        l=2.0, 
                                        sigma_n = 0.01,
                                        epsilon=0.1,
                                        discrete=False
                                        )
    
    # Initial conditions
    x0 = jnp.array([-1.5, -2., 0., 0., 0.1]) # [x1=py, x2=pz, x3=h, x4=v, x5=theta]
    MASS = 1.75
    u0 = jnp.array([MASS*GRAVITY, 0.0]) # [u1=thrust, u2=roll angular rate]
    w0 = jnp.array([0.01]) # [w1= unkown horizontal wind disturbance]
    x0_pert = jnp.array([0.01, 0.01, 0.01, 0.01, 0.01])
    ix0 = irx.icentpert(x0, x0_pert)

    n_obs = 9
    obs = jnp.tile(jnp.array([[0, x0[1], get_gp_mean(actual_disturbance_GP, 0.0, x0)[0]]]),(n_obs,1))

    quad_sys_planar = PlanarMultirotorTransformed(mass=MASS)
    ulim_planar = irx.interval([0, -1],[19, 1]) # type: ignore # Input saturation interval -> -5 <= u1 <= 15, -5 <= u2 <= 5
    Q_planar = jnp.array([1, 1, 1, 1, 1]) * jnp.eye(quad_sys_planar.xlen) # weights that prioritize overall tracking of the reference (defined below)
    R_planar = jnp.array([1, 1]) * jnp.eye(2)

    # OR [50, 20, 50, 20, 10]
    Q_ref_planar =jnp.array([50, 15, 50, 20, 10]) * jnp.eye(quad_sys_planar.xlen) # Different weights that prioritize reference reaching origin
    #(py,pz,h,v,theta)
    R_ref_planar = jnp.array([20, 20]) * jnp.eye(2)

    A,B = jitted_linearize_system(quad_sys_planar, x0, u0, w0)
    K_reference, P, _ = control.lqr(A, B, Q_ref_planar, R_ref_planar)
    K_feedback, P, _ = control.lqr(A, B, Q_planar, R_planar)


    reachable_tube, rollout_ref, rollout_feedfwd_input = jitted_rollout(0.0, ix0, x0, K_feedback, K_reference, obs, 30., 0.01, irx.Permutation((0, 1, 2, 3, 4, 5, 6, 7, 8)), irx.mjacM(quad_sys_planar.f))


    n_obs = 9
    obs = jnp.tile(jnp.array([[0, x0[1], get_gp_mean(actual_disturbance_GP, 0.0, x0)[0]]]),(n_obs,1))
    print(f"{get_gp_mean(actual_disturbance_GP, 0.0, jnp.array([0, 0])) = }")
    print(f"{get_gp_mean(actual_disturbance_GP, 0.0, jnp.array([0, -0.55])) = }")
    print(f"{get_gp_mean(actual_disturbance_GP, 0.0, jnp.array([0, -5])) = }")
    print(f"{get_gp_mean(actual_disturbance_GP, 0.0, jnp.array([0, -10])) = }")
    print(f"{obs = }")

    print("This module is not intended to be run directly. Please import it in your main script.")
