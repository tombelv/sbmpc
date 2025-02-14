import os
import time

import aligator
from aligator import manifolds
import hppfcl as fcl
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
from proxsuite_nlp import constraints

from quadrotor import get_quadroter_config, quadrotor_dynamics
from sbmpc.simulation import build_model_from_config
from sbmpc import settings
from sbmpc.geometry import quat2rotm

URDF_FILE_NAME_PATH = os.path.join(os.path.dirname(__file__), "crazyfly_urdf", "cf2x.urdf")

SPHERE_CENTERS = np.array([[0, 1, 0.5],
                           [0, 2, 0.5]])
SPHERE_RADIUS = 0.3
MASS = 0.027
GRAVITY = 9.81


# The actuation matrix below maps controls to forces and torques
# it is used when going forward in dynamics, in the below code it is actuation_matrix_
"""
template <typename Scalar>
void MultibodyConstraintFwdDynamicsTpl<Scalar>::forward(const ConstVectorRef &x,
                                                    const ConstVectorRef &u,
                                                    BaseData &data) const {
Data &d = static_cast<Data &>(data);
d.tau_ = actuation_matrix_ * u;
const pinocchio::ModelTpl<Scalar> &model = space_.getModel();
const int nq = model.nq;
const int nv = model.nv;
const auto q = x.head(nq);
const auto v = x.segment(nq, nv);
d.xdot_.head(nv) = v;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
d.xdot_.segment(nv, nv) = pinocchio::constraintDynamics(
    model, d.pin_data_, q, v, d.tau_, constraint_models_, d.constraint_datas_,
    d.settings);
#pragma GCC diagnostic pop
}
"""


# the below matrix transforms velocity at propellor to thrust and the torque components in body frame
# in piniocchio, when you apply in a joint, always applied in body frame of joint
# d_cog, cf, cm, u_lim, _ = 0.1525, 6.6e-5, 1e-6, 5.0, 0.1
# QUAD_ACT_MATRIX = np.array(
#     [
#         [0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0],
#         [1.0, 1.0, 1.0, 1.0],
#         [0.0, d_cog, 0.0, -d_cog],
#         [-d_cog, 0.0, d_cog, 0.0],
#         [-cm / cf, cm / cf, -cm / cf, cm / cf],
#     ]
# )

# in our dynamics we consider drone as one ridged body, and replace inertial data with what we have in our dynamics model we use in mppi
# this matrix assumes we are controlling thrust (along the body-frame z axis), and 3 components torque in body frame
# we have a normalizing coefficient to scale down commands
NORMALIZING_COEFF = 10e-3
QUAD_ACT_MATRIX = np.array(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1 * NORMALIZING_COEFF, 0, 0],
        [0, 0, 1 * NORMALIZING_COEFF, 0],
        [0, 0, 0, 1 * NORMALIZING_COEFF]
    ]
)

builder = RobotWrapper.BuildFromURDF
robot = builder(
                URDF_FILE_NAME_PATH,
                os.path.join(os.path.dirname(__file__), "crazyfly_urdf"),
                pin.JointModelFreeFlyer(),
            )

ub = robot.model.upperPositionLimit
ub[0:3] = 10
ub[3:7] = 2
robot.model.upperPositionLimit = ub
lb = robot.model.lowerPositionLimit
lb[0:2] = -10
lb[2] = 0
ub[3:7] = -2
robot.model.lowerPositionLimit = lb

rmodel = robot.model
rdata = robot.data
nq = rmodel.nq
nv = rmodel.nv
ROT_NULL = np.eye(3)
SPACE = manifolds.MultibodyPhaseSpace(rmodel)
X_TARGET = SPACE.neutral()
X_TARGET[:3] = (0, 3, 0.5)
X0_INIT = np.concatenate([robot.q0, np.zeros(nv)])

def setup(x0, u0, nu, weights, w_u, floor, ctrl_cstr, dt, dynmodel, quad_radius, nsteps) -> aligator.TrajOptProblem:

    # def running_cost(self, state: jnp.array, inputs: jnp.array, reference) -> jnp.float32:
    #     state_ref = reference[:13]
    #     state_ref = state_ref.at[7:10].set(-1*(state[0:3] - state_ref[0:3]))
    #     input_ref = reference[13:13+4]
    #     pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, state_ref)
    #     return (5 * vel_err.transpose() @ vel_err +
    #             1 * ang_vel_err.transpose() @ ang_vel_err +
    #             (inputs-input_ref).transpose() @ jnp.diag(jnp.array([10, 10, 10, 100])) @ (inputs-input_ref))

    # def final_cost(self, state, reference):
    #     pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, reference[:13])
    #     return (10 * pos_err.transpose() @ pos_err +
    #             1 * att_err.transpose() @ att_err +
    #             5 * vel_err.transpose() @ vel_err +
    #             1 * ang_vel_err.transpose() @ ang_vel_err)


        term_cost = aligator.QuadraticStateCost(SPACE, nu, X_TARGET, np.diag(weights))
        prob = aligator.TrajOptProblem(x0, nu, SPACE, term_cost=term_cost)

        for i in range(nsteps):
            rcost = aligator.CostStack(SPACE, nu)

            xreg_cost = aligator.QuadraticStateCost(
                SPACE, nu, X_TARGET, np.diag(weights) * dt
            )
            rcost.addCost(xreg_cost)

            ureg_cost = aligator.QuadraticControlCost(SPACE, u0, w_u * dt)
            rcost.addCost(ureg_cost)

            stage = aligator.StageModel(rcost, dynmodel)
            stage.addConstraint(*ctrl_cstr)
            for i in range(SPHERE_CENTERS.shape[0]):
                sphere_coords = SPHERE_CENTERS[i]
                sphr = Sphere(rmodel, SPACE.ndx, nu, sphere_coords, SPHERE_RADIUS, quad_radius
                )
                stage.addConstraint(floor, constraints.NegativeOrthant())
                stage.addConstraint(sphr, constraints.NegativeOrthant())

            prob.addStage(stage)
        return prob


def create_halfspace_z(ndx, nu, offset: float = 0.0, neg: bool = False):
    r"""
    Constraint :math:`z \geq offset`.
    """
    root_frame_id = 1
    p_ref = np.zeros(3)
    frame_fun = aligator.FrameTranslationResidual(ndx, nu, rmodel, p_ref, root_frame_id)
    A = np.array([[0.0, 0.0, 1.0]])
    b = np.array([-offset])
    sign = -1.0 if neg else 1.0
    frame_fun_z = aligator.LinearFunctionComposition(frame_fun, sign * A, sign * b)
    return frame_fun_z

class Sphere(aligator.StageFunction):
    def __init__(
        self,
        rmodel: pin.Model,
        ndx,
        nu,
        center,
        radius,
        margin: float = 0.0,
    ) -> None:
        super().__init__(ndx, nu, 1)
        self.rmodel = rmodel.copy()
        self.rdata = self.rmodel.createData()
        self.ndx = ndx
        self.center = center.copy()
        self.radius = radius
        self.margin = margin

    def __getinitargs__(self):
        return (self.rmodel, self.ndx, self.nu, self.center, self.radius, self.margin)

    def evaluate(self, x, u, data):  # distance function
        q = x[:nq]
        pin.forwardKinematics(self.rmodel, self.rdata, q)
        M: pin.SE3 = pin.updateFramePlacement(self.rmodel, self.rdata, 1)
        err = M.translation[:3] - self.center
        res = np.dot(err, err) - (self.radius + self.margin) ** 2
        data.value[:] = -res

    def computeJacobians(self, x, u, data):
        q = x[:nq]
        J = pin.computeFrameJacobian(
            self.rmodel, self.rdata, q, 1, pin.LOCAL_WORLD_ALIGNED
        )
        err = x[:3] - self.center
        data.Jx[:nv] = -2 * J[:3].T @ err


def main(display: bool = True, integrator: str = "euler"):
    sphere = fcl.Sphere(SPHERE_RADIUS)

    for i in range(SPHERE_CENTERS.shape[0]):
        sphere_coords = SPHERE_CENTERS[i]
        geom_sphr = pin.GeometryObject(
        f"sphere_{i}", 0, sphere, pin.SE3(ROT_NULL, sphere_coords)
    )
        sphr_color = np.array([1.0, 0, 0, 0.4])
        geom_sphr.meshColor = sphr_color
        
        robot.collision_model.addGeometryObject(geom_sphr)
        robot.visual_model.addGeometryObject(geom_sphr)
    
    robot.collision_model.geometryObjects[0].geometry.computeLocalAABB()
    quad_radius = robot.collision_model.geometryObjects[0].geometry.aabb_radius

    nu = QUAD_ACT_MATRIX.shape[1]

    ode_dynamics = aligator.dynamics.MultibodyFreeFwdDynamics(SPACE, QUAD_ACT_MATRIX)

    dt = 0.01
    # 6 or 10 is interesting
    nsteps = 25
    Tf = nsteps * dt
    times = np.linspace(0, Tf, nsteps + 1)
    print("nsteps: {:d}".format(nsteps))

    if integrator == "euler":
        dynmodel = aligator.dynamics.IntegratorEuler(ode_dynamics, dt)
    elif integrator == "semieuler":
        dynmodel = aligator.dynamics.IntegratorSemiImplEuler(ode_dynamics, dt)
    elif integrator == "rk2":
        dynmodel = aligator.dynamics.IntegratorRK2(ode_dynamics, dt)
    elif integrator == "midpoint":
        dynmodel = aligator.dynamics.IntegratorMidpoint(ode_dynamics, dt)
    else:
        raise ValueError()
    

    # solving dynamics, tau for position, velocity, rotation. solving direct dynamics.
    # tau = pin.rnea(rmodel, rdata, robot.q0, np.zeros(nv), np.zeros(nv))
    # u0, _, _, _ = np.linalg.lstsq(QUAD_ACT_MATRIX, tau, rcond=-1)
    u0 = np.zeros((nu,), dtype=np.float32)
    u0[0] = MASS * GRAVITY

    # adding targets 
    objective_color = np.array([5, 104, 143, 200]) / 255.0
    sp1_obj = pin.GeometryObject(
                "obj1", 0, fcl.Sphere(0.05), pin.SE3(ROT_NULL, X_TARGET[:3])
    )
    sp1_obj.meshColor[:] = objective_color
    robot.visual_model.addGeometryObject(sp1_obj)

    weights = np.zeros(SPACE.ndx)
    weights[:3] = 0.1
    weights[3:6] = 1e-2
    weights[nv:] = 1e-3


    u_max = np.zeros(nu, dtype=np.float32)
    u_max[0] = 50
    u_max[1:4] = np.pi * 2 * 10
    u_min = np.zeros(nu, dtype=np.float32)
    u_min[1:4] = -np.pi * 2 * 10

    w_u = np.eye(nu) * 1e-1
    floor = create_halfspace_z(SPACE.ndx, nu, 0.0, True)

    u_identity_fn = aligator.ControlErrorResidual(SPACE.ndx, np.zeros(nu))
    box_set = constraints.BoxConstraint(u_min, u_max)
    ctrl_cstr = (u_identity_fn, box_set)

    problem = setup(X0_INIT, u0, nu, weights, w_u, floor, ctrl_cstr, dt, dynmodel, quad_radius, nsteps)

    tol = 1e-4
    mu_init = 1.0
    # verbose = aligator.VerboseLevel.VERBOSE
    history_cb = aligator.HistoryCallback()
    solver = aligator.SolverProxDDP(tol, mu_init) # , verbose=verbose
    solver.max_iters = 200
    solver.registerCallback("his", history_cb)
    solver.bcl_params.dyn_al_scale = 1e-6
    solver.setup(problem)

    us_init = [u0] * nsteps
    xs_init = aligator.rollout(dynmodel, X0_INIT, us_init)
    solver.run(problem, xs_init, us_init)
    results = solver.results
    xs_opt = results.xs.tolist()
    us_opt = results.us.tolist()
    control_input = results.us.tolist()[0]

    config = get_quadroter_config()
    sim_dynamics_model_setting = config.sim_dynamics
    dyn_model = settings.DynamicsModel.MJX
    sim_dynamics_model, _, new_state = build_model_from_config(dyn_model, config, None)

    xs_opt_arr = np.array(xs_opt)

    stage_model = problem.stages
    stage_data = stage_model[nsteps - 1].createData()

    # solver.cycleProblem(problem, stage_data)

    x0 = X0_INIT

    trajectory = [x0]
    controls = []
    dd = dynmodel.createData()

    iter_counter = 0
    max_iterations = 2000
    times = []

    while np.linalg.norm(x0[[0, 1, 2, 7,8,9,10,11,12]] - X_TARGET[[0, 1, 2, 7,8,9,10,11,12]]) > 0.1 and iter_counter <= max_iterations:
        start_time = time.time()
        us_init = [u0] * nsteps
        xs_init = aligator.rollout(dynmodel, x0, us_init)
        converged = solver.run(problem, xs_init, us_init)
        if not converged:
            print(x0)
            print(iter_counter)
            print(xs_init[0])
            print(us_init)
            raise RuntimeError
        results = solver.results
        control_input = results.us.tolist()[0]
        controls.append(np.copy(control_input))
        dynmodel.forward(x0, control_input, dd)
        # new_state = sim_dynamics_model.integrate(new_state, control_input, dt)
        # x0 = np.concatenate(
        #         [new_state.qpos, new_state.qvel])
        # # re-ordering quaternion, pinocchio does x y z w, mjx does w x y z
        # x0[3:7] = x0[[4, 5, 6, 3]]
        # rot = quat2rotm(x0[3:7])
        # euler = pin.rpy.matrixToRpy(np.matrix(rot))

        x0 = np.copy(dd.xnext)

        trajectory.append(x0)
        
        # res = pin.isNormalized(rmodel, x0)
        # res = SPACE.isNormalized(x0)
        # problem.replaceStageCircular(problem.stages[0])
        # aligator.rotate_vec_left(problem.stages)
        # solver.cycleProblem(problem, problem.stages[nsteps - 1].createData())

        problem = setup(x0, u0, nu, weights, w_u, floor, ctrl_cstr, dt, dynmodel, quad_radius, nsteps)
        iter_counter += 1
        times.append(time.time() - start_time)

        

    trajectory = np.array(trajectory)
    controls = np.array(controls)
    print(f"avg time per iteration: {round(sum(times) / len(times), 5)} seconds")
    print(f"total iterations: {iter_counter}, delta_time: {dt}")
    print(f"goal: {X_TARGET}, final_state: {x0}, norm diff: {np.linalg.norm(X_TARGET - x0)}")



    if display:

        times = np.arange(trajectory.shape[0]) * dt

        fig = plt.figure(figsize=(10, 8))
        fig.suptitle("Trajectory with Obstacles, Goals, and Initial Position")
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(0, 5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label="trajectory")

        ax.scatter(X_TARGET[0], X_TARGET[1], X_TARGET[2], s=[10], c="red", label="goal")
        ax.scatter(X0_INIT[0], X0_INIT[1], X0_INIT[2], s=[10], c="green", label="x0")
        for i in range(SPHERE_CENTERS.shape[0]):
            sphere_coords = SPHERE_CENTERS[i]
            legend = None
            if i == 0:
                legend = "obstacle"
            ax.scatter(sphere_coords[0], sphere_coords[1], sphere_coords[2], c="purple", label=legend, s=[10])
        ax.legend()
        plt.show()

        fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(10, 8))
        labels = ["x_position_meters", "y_position_meters", "z_position_meters"]
        for i in range(3):
            axes[i].plot(times, trajectory[:, i])
            axes[i].set_ylabel(labels[i])
        axes[2].set_xlabel("time (seconds)")
        fig.suptitle("Position over Time")
        plt.show()

        fig, axes = plt.subplots(4, 1, sharex=True, sharey=False, figsize=(10, 8))
        labels = ["thrust_z_axis", "torque_x", "torque_y", "torque_z"]
        for i in range(4):
            axes[i].plot(times[1:], controls[:, i])
            axes[i].set_ylabel(labels[i])
        axes[2].set_xlabel("time (seconds)")
        fig.suptitle("Control Effort over Time")
        plt.show()

        orientation_angles = []
        for i in range(trajectory.shape[0]):
            quat_ori = trajectory[i, 3:7]
            rot = quat2rotm(quat_ori[[3, 0, 1, 2]])
            euler = pin.rpy.matrixToRpy(np.matrix(rot))
            orientation_angles.append(euler)
        orientation_angles = np.array(orientation_angles)

        fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(10, 8))
        labels = ["x_orientation_radians", "y_orientation_radians", "z_orientation_radians"]
        for i in range(3):
            axes[i].plot(times, orientation_angles[:, i])
            axes[i].set_ylabel(labels[i])
        axes[2].set_xlabel("time (seconds)")
        fig.suptitle("Orientation over Time")
        plt.show()
        


if __name__ == "__main__":
    main()