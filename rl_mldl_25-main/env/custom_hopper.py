"""Implementation of the Hopper environment supporting
domain randomization optimization.

See more at: https://www.gymlibrary.dev/environments/mujoco/hopper/
"""
from copy import deepcopy
import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv


class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None):
        """
        domain: one of {None, "source", "target", "udr", "massdr",
                         "frictiondr", "dampingdr", "extdr"}
        """

        # Domain identifier for this instance
        self.domain = domain

        # Used by 'extdr' to inject action noise (must exist before MujocoEnv.__init__)
        self.action_noise_std = 0.01

        # Initialize base MuJoCo environment (this may call step/reset internally)
        MujocoEnv.__init__(self, frame_skip=4)
        utils.EzPickle.__init__(self)

        # ---- Mass configurations ----
        # Keep arrays as numpy arrays to support arithmetic
        self.target_masses = np.copy(self.sim.model.body_mass[1:])  # shape (4,)

        # Source masses: torso scaled by 0.7, others same as target
        self.source_masses = self.target_masses.copy()
        self.source_masses[0] *= 0.7  # -30% torso mass

        # ---- UDR ranges for masses (only links 1,2,3 in your representation) ----
        scale = 0.3
        
        # udr_low/upr_high refer to the last 3 masses (exclude torso index 0)
        self.udr_low = self.source_masses[1:] * (1.0 - scale)
        self.udr_high = self.source_masses[1:] * (1.0 + scale)

        # ---- Extended DR: nominal damping + friction (saved to reset later) ----
        self.nominal_damping = np.copy(self.sim.model.dof_damping)
        self.nominal_friction = np.copy(self.sim.model.geom_friction)

        # Multiplicative ranges for damping and friction
        self.damping_scale_low = 0.5
        self.damping_scale_high = 1.5

        self.friction_scale_low = 0.7
        self.friction_scale_high = 1.3

    # ----------------------------------------------------------------------
    # Domain randomization helpers
    # ----------------------------------------------------------------------
    def set_random_parameters(self):
        """Set random masses using UDR sampling distribution."""
        self.set_parameters(self.sample_parameters())

    def sample_parameters(self):
        """
        Sample masses for UDR: torso fixed, other 3 links randomized.
        """
        masses = self.source_masses.copy()
        rand_rest = self.np_random.uniform(low=self.udr_low,
                                           high=self.udr_high)
        masses[1:] = rand_rest
        return masses

    # ----------------------------------------------------------------------
    def get_parameters(self):
        """Get value of mass for each link."""
        return np.array(self.sim.model.body_mass[1:])

    def set_parameters(self, task):
        """Set each hopper link's mass to a new value."""
        self.sim.model.body_mass[1:] = task

    # ----------------------------------------------------------------------
    def step(self, a):
        """Step the simulation to the next timestep."""

        # Safe RNG: fallback to np.random if np_random not initialized yet
        rng = getattr(self, "np_random", np.random)

        # Extended DR: add action noise only for full extdr
        if self.domain == "extdr":
            a = a + rng.normal(0.0, self.action_noise_std, size=a.shape)
            a = np.clip(a, -1.0, 1.0)

        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]

        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()

        s = self.state_vector()
        done = not (
            np.isfinite(s).all()
            and (np.abs(s[2:]) < 100).all()
            and (height > 0.7)
            and (abs(ang) < 0.2)
        )
        ob = self._get_obs()
        return ob, reward, done, {}

    # ----------------------------------------------------------------------
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    # ----------------------------------------------------------------------
    def reset_model(self):
        # Use env RNG if available
        rng = getattr(self, "np_random", np.random)

        # Sample initial qpos, qvel
        qpos = self.init_qpos + rng.uniform(
            low=-.005, high=.005, size=self.model.nq
        )
        qvel = self.init_qvel + rng.uniform(
            low=-.005, high=.005, size=self.model.nv
        )

        # Reset damping and friction to nominal
        self.sim.model.dof_damping[:] = self.nominal_damping
        self.sim.model.geom_friction[:] = self.nominal_friction

        # ----- DOMAIN SELECTION -----
        if self.domain == "target":
            self.set_parameters(self.target_masses)

        elif self.domain == "source":
            self.set_parameters(self.source_masses)

        elif self.domain == "udr":
            self.set_random_parameters()

        elif self.domain == "massdr":  # Ablation: Mass-only (same as UDR)
            self.set_random_parameters()

        elif self.domain == "frictiondr":  # Ablation: Friction-only
            # Randomize friction
            friction_scale = rng.uniform(
                self.friction_scale_low,
                self.friction_scale_high
            )
            self.sim.model.geom_friction[:] = (
                self.nominal_friction * friction_scale
            )

        elif self.domain == "dampingdr":  # Ablation: Damping-only
            # Randomize damping
            damping_scale = rng.uniform(
                self.damping_scale_low,
                self.damping_scale_high,
                size=self.nominal_damping.shape
            )
            self.sim.model.dof_damping[:] = (
                self.nominal_damping * damping_scale
            )

        elif self.domain == "extdr":  # Full: Mass + damping + friction + action noise (original)
            # Mass randomization (same as UDR)
            self.set_random_parameters()

            # Randomize damping
            damping_scale = rng.uniform(
                self.damping_scale_low,
                self.damping_scale_high,
                size=self.nominal_damping.shape
            )
            self.sim.model.dof_damping[:] = (
                self.nominal_damping * damping_scale
            )

            # Randomize friction
            friction_scale = rng.uniform(
                self.friction_scale_low,
                self.friction_scale_high
            )
            self.sim.model.geom_friction[:] = (
                self.nominal_friction * friction_scale
            )

        else:
            # default = target
            self.set_parameters(self.target_masses)

        # Finalize state
        self.set_state(qpos, qvel)
        return self._get_obs()

    # ----------------------------------------------------------------------
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    # ----------------------------------------------------------------------
    def set_mujoco_state(self, state):
        mjstate = deepcopy(self.get_mujoco_state())

        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]

        self.set_sim_state(mjstate)

    def set_sim_state(self, mjstate):
        return self.sim.set_state(mjstate)

    def get_mujoco_state(self):
        return self.sim.get_state()


"""
    Registered environments
"""
gym.envs.register(
    id="CustomHopper-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
)

gym.envs.register(
    id="CustomHopper-source-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "source"},
)

gym.envs.register(
    id="CustomHopper-target-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "target"},
)

gym.envs.register(
    id="CustomHopper-udr-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "udr"},
)

gym.envs.register(
    id="CustomHopper-massdr-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "massdr"},
)

gym.envs.register(
    id="CustomHopper-frictiondr-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "frictiondr"},
)

gym.envs.register(
    id="CustomHopper-dampingdr-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "dampingdr"},
)

gym.envs.register(
    id="CustomHopper-extdr-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "extdr"},
)
