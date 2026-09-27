//! Joints solved with velocity impulses, in the same passes as contacts.
//!
//! An [XPBD joint](super::xpbd) corrects positions once contacts and friction have been
//! solved and the bodies moved, so friction gets no say in the motion the joint makes:
//! a body held by friction under a steady joint load is carried along a little every
//! substep. A joint solved here is [warm started], [solved] and [relaxed] next to the
//! contacts instead, each answering the other's impulses before anything moves.
//!
//! Every joint but the [`PrismaticJoint`] is solved this way.
//!
//! # Stiffness
//!
//! A constraint with no compliance is held as contacts are, by a soft constraint of
//! [`SolverConfig::joint_frequency`](crate::dynamics::solver::SolverConfig::joint_frequency)
//! and [`joint_damping_ratio`](crate::dynamics::solver::SolverConfig::joint_damping_ratio), with its
//! position error left out of the relaxing pass.
//!
//! A constraint with compliance is a spring of that stiffness, integrated implicitly. It
//! is the velocity form of what XPBD solves: the compliance over the substep squared is
//! added to the inverse effective mass, and the whole position error is asked for in one
//! substep.
//!
//! [warm started]: crate::dynamics::solver::schedule::SubstepSolverSystems::WarmStart
//! [solved]: crate::dynamics::solver::schedule::SubstepSolverSystems::SolveConstraints
//! [relaxed]: crate::dynamics::solver::schedule::SubstepSolverSystems::Relax

mod distance;
mod fixed;
mod point;
mod revolute;
#[cfg(feature = "3d")]
mod spherical;

pub use distance::DistanceJointImpulses;
pub use fixed::FixedJointImpulses;
#[cfg(feature = "3d")]
pub use fixed::PrincipalCompliance;
pub use point::PointImpulses;
pub use revolute::RevoluteJointImpulses;
#[cfg(feature = "3d")]
pub use spherical::SphericalJointImpulses;

use core::cmp::Ordering;

use crate::{
    dynamics::{
        joints::EntityConstraint,
        solver::{
            SolverConfig,
            softness_parameters::{SoftnessCoefficients, SoftnessParameters},
            solver_body::{SolverBodies, SolverBody, SolverBodyIndex, SolverBodyInertia},
        },
    },
    prelude::*,
};
use bevy::{ecs::component::Mutable, prelude::*};

/// A joint between two bodies that is solved with velocity impulses.
pub trait ImpulseJoint: Component + EntityConstraint<2> + Clone {
    /// What the solver keeps for the joint: where it stood when the step began, and the
    /// impulses it holds its bodies with.
    type Impulses: Component<Mutability = Mutable> + Default + Clone;

    /// Records where the joint stands as the step begins, and forgets the last step's
    /// totals. The impulses themselves are kept, for warm starting.
    fn prepare(&self, bodies: [&RigidBodyQueryReadOnlyItem; 2], impulses: &mut Self::Impulses);

    /// Applies the impulses the joint held its bodies with last, scaled by `coefficient`.
    fn warm_start(
        &self,
        bodies: [&mut SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
        impulses: &mut Self::Impulses,
        coefficient: f32,
    );

    /// Solves the joint's constraints once, and at the end of a substep adds its impulses
    /// to the step's totals.
    fn solve(
        &self,
        bodies: [&mut SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
        impulses: &mut Self::Impulses,
        pass: &Pass,
    );

    /// The linear, angular and motor impulses of the step's substeps, added up, as the
    /// second body was given them.
    fn totals(impulses: &Self::Impulses) -> (Vector, AngularVector, f32);
}

/// One pass over the constraints in a substep.
#[derive(Clone, Copy, Debug)]
pub struct Pass {
    /// Whether position errors are solved for. They are left out to relax the velocities
    /// that solving for them added.
    pub use_bias: bool,
    /// How a constraint with no compliance of its own is held in this pass.
    pub softness: SoftnessCoefficients,
    /// The stiffest spring a substep can follow: what it adds to the inverse effective
    /// mass it works against, as a share of that.
    pub follows: f32,
    /// The length of the substep.
    pub delta_secs: f32,
}

impl Pass {
    /// Softness that solves for the speed alone, with all of the effective mass.
    const RELAXED: SoftnessCoefficients = SoftnessCoefficients {
        bias: 0.0,
        mass_scale: 1.0,
        impulse_scale: 0.0,
    };

    fn new(use_bias: bool, config: &SolverConfig, delta_secs: f32) -> Self {
        // A frequency the substep cannot follow would only ring.
        let frequency = config.joint_frequency.min(0.25 / delta_secs);
        Self {
            use_bias,
            softness: if use_bias {
                SoftnessParameters::new(config.joint_damping_ratio, frequency)
                    .compute_coefficients(delta_secs)
            } else {
                Self::RELAXED
            },
            follows: (core::f32::consts::TAU * frequency * delta_secs)
                .powi(2)
                .recip(),
            delta_secs,
        }
    }

    /// What is added to the inverse effective mass for a spring of some compliance, for
    /// how much an impulse along it moves anything at the most: the [size](size3) of its
    /// inverse effective mass.
    ///
    /// A spring stiffer than a substep can follow is no spring to the solver. Solved as
    /// one, it would be asked to close its whole error every substep, with nothing to
    /// take out the speed that adds, and light bodies on it buzz and spin. It is solved
    /// as the stiffest spring that the substep can follow.
    pub(super) fn give(&self, compliance: f32, stiffest: f32) -> f32 {
        (compliance / (self.delta_secs * self.delta_secs)).max(self.follows * stiffest)
    }

    /// The step in impulse that answers a constraint's speed and error, for what it has
    /// given so far.
    ///
    /// `solve` applies the inverse of a matrix to a vector: the inverse effective mass,
    /// with the given amount added to its diagonal. `stiffest` is the size of the inverse
    /// effective mass.
    pub(super) fn step<V>(
        &self,
        solve: impl Fn(f32, V) -> V,
        compliance: f32,
        stiffest: f32,
        speed: V,
        error: V,
        given: V,
    ) -> V
    where
        V: Copy
            + core::ops::Add<Output = V>
            + core::ops::Sub<Output = V>
            + core::ops::Mul<f32, Output = V>
            + core::ops::Neg<Output = V>,
    {
        if compliance > 0.0 {
            // A spring, integrated implicitly.
            let give = self.give(compliance, stiffest);
            -solve(give, speed + error * self.delta_secs.recip() + given * give)
        } else {
            -solve(0.0, speed + error * self.softness.bias) * self.softness.mass_scale
                - given * self.softness.impulse_scale
        }
    }
}

impl Pass {
    /// What a limit holds with after this pass, for what it held with before. A limit only
    /// pushes back in: `room` is how far within it the constraint is, `closing` how fast
    /// the room grows, and `k` the inverse effective mass along it.
    pub(super) fn limit(
        &self,
        k: f32,
        compliance: f32,
        room: f32,
        closing: f32,
        given: f32,
    ) -> f32 {
        let impulse = if room > 0.0 {
            // Not there yet: stop only what would pass the limit within the substep.
            -(closing + room / self.delta_secs) / k
        } else {
            self.step(
                |give, rhs| rhs / (k + give),
                compliance,
                k,
                closing,
                room,
                given,
            )
        };
        (given + impulse).max(0.0)
    }
}

/// A turn of the second body about an axis on the first, as the bodies stand within a
/// substep.
pub(super) struct Turning {
    /// The axis.
    #[cfg(feature = "3d")]
    pub axis: Vector,
    /// The second body's angle about the axis from the first's.
    pub angle: f32,
    /// How fast an impulse about the axis turns the bodies apart: its inverse effective mass.
    pub k: f32,
}

impl Turning {
    pub fn new(
        #[cfg(feature = "3d")] axis: Vector,
        angle: f32,
        [inertia1, inertia2]: [&SolverBodyInertia; 2],
    ) -> Self {
        let k = inertia1.effective_inv_angular_inertia() + inertia2.effective_inv_angular_inertia();
        Self {
            #[cfg(feature = "3d")]
            axis,
            angle,
            #[cfg(feature = "2d")]
            k,
            #[cfg(feature = "3d")]
            k: axis.dot(k * axis),
        }
    }

    /// How fast the second body turns about the axis, past the first.
    pub fn speed(&self, [body1, body2]: [&SolverBody; 2]) -> f32 {
        let speed = body2.angular_velocity - body1.angular_velocity;
        #[cfg(feature = "2d")]
        return speed;
        #[cfg(feature = "3d")]
        return speed.dot(self.axis);
    }

    /// An impulse about the axis.
    pub fn about(&self, impulse: f32) -> AngularVector {
        #[cfg(feature = "2d")]
        return impulse;
        #[cfg(feature = "3d")]
        return self.axis * impulse;
    }

    /// Keeps the angle within its limits, by the impulses `lower` about the axis and
    /// `upper` against it.
    #[allow(clippy::too_many_arguments)]
    pub fn limit(
        &self,
        [body1, body2]: [&mut SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
        limit: AngleLimit,
        compliance: f32,
        lower: &mut f32,
        upper: &mut f32,
        pass: &Pass,
    ) {
        if self.k <= f32::EPSILON {
            return;
        }
        let held = pass.limit(
            self.k,
            compliance,
            self.angle - limit.min,
            self.speed([&*body1, &*body2]),
            *lower,
        );
        turn(
            [&mut *body1, &mut *body2],
            inertias,
            self.about(held - *lower),
        );
        *lower = held;

        let held = pass.limit(
            self.k,
            compliance,
            limit.max - self.angle,
            -self.speed([&*body1, &*body2]),
            *upper,
        );
        turn([body1, body2], inertias, self.about(*upper - held));
        *upper = held;
    }
}

/// How much an impulse moves anything at the most, by an inverse effective mass: no less
/// than its largest eigenvalue, no more than that by the square root of its rank, and the
/// same however the bodies are turned in the world.
pub(super) fn size2(k: Mat2) -> f32 {
    (k.x_axis.length_squared() + k.y_axis.length_squared()).sqrt()
}

/// How much an impulse moves anything at the most, by an inverse effective mass: no less
/// than its largest eigenvalue, no more than that by the square root of its rank, and the
/// same however the bodies are turned in the world.
#[cfg(feature = "3d")]
pub(super) fn size3(k: SymmetricTensor) -> f32 {
    let across = k.m01 * k.m01 + k.m02 * k.m02 + k.m12 * k.m12;
    (k.m00 * k.m00 + k.m11 * k.m11 + k.m22 * k.m22 + 2.0 * across).sqrt()
}

/// How small a share of a scaled matrix a direction may be and still be one that impulses
/// move anything in. Smaller is what rounding leaves of nothing.
const NOTHING: f32 = 1e-6;

/// The inverse of `k` applied to `rhs`, in the directions `k` can move anything in.
///
/// Locked axes and bodies that cannot move leave directions where no impulse has any
/// effect. None is given there. A direction that little moves in is not one of them: a
/// thin rod turns about its length thousands of times as readily as across it, and is
/// held across it all the same. So `k` is first scaled to a diagonal of ones, where how
/// near it is to having such a direction no longer depends on the bodies' proportions.
pub(super) fn solve2(k: Mat2, rhs: Vec2) -> Vec2 {
    let diagonal = Vec2::new(k.x_axis.x, k.y_axis.y);
    if diagonal.max_element() <= 0.0 {
        return Vec2::ZERO;
    }
    if diagonal.min_element() > 0.0 {
        let scale = diagonal.map(f32::sqrt).recip();
        let across = k.x_axis.y * scale.x * scale.y;
        let determinant = 1.0 - across * across;
        if determinant > NOTHING {
            let rhs = rhs * scale;
            let solved = Vec2::new(rhs.x - across * rhs.y, rhs.y - across * rhs.x);
            return solved / determinant * scale;
        }
    }
    // One direction is left: k = l v v^T, whose trace is l.
    let trace = diagonal.x + diagonal.y;
    k * rhs / (trace * trace)
}

/// The inverse of `k` applied to `rhs`, in the directions `k` can move anything in.
///
/// Locked axes and bodies that cannot move leave directions where no impulse has any
/// effect. None is given there. A direction that little moves in is not one of them: a
/// thin rod turns about its length thousands of times as readily as across it, and is
/// held across it all the same. So `k` is first scaled to a diagonal of ones, where how
/// near it is to having such a direction no longer depends on the bodies' proportions.
#[cfg(feature = "3d")]
pub(super) fn solve3(k: SymmetricTensor, rhs: Vec3) -> Vec3 {
    let diagonal = k.diagonal();
    if diagonal.max_element() <= 0.0 {
        return Vec3::ZERO;
    }
    // Nothing moves along an axis with nothing on the diagonal: k is positive
    // semidefinite, so its row and column are nothing too.
    let scale = Vec3::select(
        diagonal.cmpgt(Vec3::ZERO),
        diagonal.map(f32::sqrt).recip(),
        Vec3::ZERO,
    );
    let scaled = SymmetricTensor::new(
        k.m00 * scale.x * scale.x,
        k.m01 * scale.x * scale.y,
        k.m02 * scale.x * scale.z,
        k.m11 * scale.y * scale.y,
        k.m12 * scale.y * scale.z,
        k.m22 * scale.z * scale.z,
    );
    if scaled.determinant() > 1e-4 {
        return scaled.inverse() * (rhs * scale) * scale;
    }
    let (values, directions) = eigen(scaled);
    if values.min_element() > NOTHING {
        // Every direction moves something, some of them little.
        return (0..3)
            .map(|i| directions.col(i) * (directions.col(i).dot(rhs * scale) / values[i]))
            .sum::<Vec3>()
            * scale;
    }
    // Some direction moves nothing. What is asked for along it cannot be had, and what
    // is left is solved as nearly as it can be.
    let (values, directions) = eigen(k);
    let least = values.max_element() * NOTHING;
    (0..3)
        .filter(|&i| values[i] > least)
        .map(|i| directions.col(i) * (directions.col(i).dot(rhs) / values[i]))
        .sum()
}

/// The eigenvalues of a symmetric matrix and, in the columns of the other, their
/// directions, by Jacobi's rotations: slower than a closed form, and as exact in the
/// small eigenvalues as in the large.
#[cfg(feature = "3d")]
pub(super) fn eigen(k: SymmetricTensor) -> (Vec3, Mat3) {
    let mut a = k.to_mat3();
    let mut directions = Mat3::IDENTITY;
    for _ in 0..8 {
        let mut turned = false;
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            let across = a.col(q)[p];
            let (first, second) = (a.col(p)[p], a.col(q)[q]);
            if across.abs() <= f32::EPSILON * 0.5 * (first.abs() + second.abs()) {
                continue;
            }
            // The turn in the plane of the two axes that leaves nothing across them.
            let theta = (second - first) / (2.0 * across);
            let tangent = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let cosine = (tangent * tangent + 1.0).sqrt().recip();
            let sine = tangent * cosine;
            let mut turn = Mat3::IDENTITY;
            turn.col_mut(p)[p] = cosine;
            turn.col_mut(q)[q] = cosine;
            turn.col_mut(q)[p] = sine;
            turn.col_mut(p)[q] = -sine;
            a = turn.transpose() * a * turn;
            directions *= turn;
            turned = true;
        }
        if !turned {
            break;
        }
    }
    (Vec3::new(a.x_axis.x, a.y_axis.y, a.z_axis.z), directions)
}

/// Gives the second body an angular impulse, and takes it from the first.
pub(super) fn turn(
    [body1, body2]: [&mut SolverBody; 2],
    [inertia1, inertia2]: [&SolverBodyInertia; 2],
    impulse: AngularVector,
) {
    body1.angular_velocity -= inertia1.effective_inv_angular_inertia() * impulse;
    body2.angular_velocity += inertia2.effective_inv_angular_inertia() * impulse;
}

/// The joints of one kind that are solved this step, each with its bodies found and what
/// the solver keeps for it, side by side: the passes of a substep go through them
/// without asking the world for anything.
#[derive(Resource)]
pub struct ActiveJoints<J: ImpulseJoint> {
    joints: Vec<ActiveJoint<J>>,
}

impl<J: ImpulseJoint> Default for ActiveJoints<J> {
    fn default() -> Self {
        Self { joints: Vec::new() }
    }
}

impl<J: ImpulseJoint> ActiveJoints<J> {
    /// How many joints are solved this step.
    pub fn len(&self) -> usize {
        self.joints.len()
    }

    /// Whether no joint is solved this step.
    pub fn is_empty(&self) -> bool {
        self.joints.is_empty()
    }
}

struct ActiveJoint<J: ImpulseJoint> {
    entity: Entity,
    bodies: [SolverBodyIndex; 2],
    joint: J,
    impulses: J::Impulses,
}

impl<J: ImpulseJoint> ActiveJoint<J> {
    /// The joint's two bodies, with stand-ins for those the solver does not move.
    fn solve(
        &mut self,
        solver_bodies: &mut SolverBodies,
        solve: impl FnOnce(&J, [&mut SolverBody; 2], [&SolverBodyInertia; 2], &mut J::Impulses),
    ) {
        let mut dummy_body1 = SolverBody::DUMMY;
        let mut dummy_body2 = SolverBody::DUMMY;
        let (mut body1, mut inertia1) = (&mut dummy_body1, &SolverBodyInertia::DUMMY);
        let (mut body2, mut inertia2) = (&mut dummy_body2, &SolverBodyInertia::DUMMY);

        let access = solver_bodies.access();
        // SAFETY: The two jointed bodies are distinct, and joints are processed serially here.
        let (b1, b2) = unsafe { access.get_pair_unchecked_mut(self.bodies[0], self.bodies[1]) };
        if let Some((body, inertia)) = b1 {
            body1 = body;
            inertia1 = inertia;
        }
        if let Some((body, inertia)) = b2 {
            body2 = body;
            inertia2 = inertia;
        }

        // If a body has a higher dominance, it is treated as a static or kinematic body.
        match (inertia1.dominance() - inertia2.dominance()).cmp(&0) {
            Ordering::Greater => inertia1 = &SolverBodyInertia::DUMMY,
            Ordering::Less => inertia2 = &SolverBodyInertia::DUMMY,
            _ => {}
        }

        solve(
            &self.joint,
            [body1, body2],
            [inertia1, inertia2],
            &mut self.impulses,
        );
    }
}

/// Finds the joints to solve this step, and records where each stands as it begins. A
/// joint whose bodies are at rest carries no load that is known.
pub fn prepare_impulse_joints<J: ImpulseJoint>(
    bodies: Query<RigidBodyQueryReadOnly, Without<RigidBodyDisabled>>,
    indices: Query<&SolverBodyIndex, Without<RigidBodyDisabled>>,
    mut joints: Query<
        (Entity, &J, &mut J::Impulses, Option<&mut JointForces>),
        (Without<RigidBody>, Without<JointDisabled>),
    >,
    mut active: ResMut<ActiveJoints<J>>,
) {
    active.joints.clear();
    for (entity, joint, mut impulses, forces) in &mut joints {
        let entities = joint.entities();
        let [index1, index2] = entities.map(|body| {
            indices
                .get(body)
                .copied()
                .unwrap_or(SolverBodyIndex::INVALID)
        });
        let at_rest = index1 == index2;
        if !at_rest && let Ok([body1, body2]) = bodies.get_many(entities) {
            joint.prepare([&body1, &body2], &mut impulses);
            active.joints.push(ActiveJoint {
                entity,
                bodies: [index1, index2],
                joint: joint.clone(),
                impulses: impulses.clone(),
            });
        } else if let Some(mut forces) = forces
            && *forces != JointForces::new()
        {
            *forces = JointForces::new();
        }
    }
}

/// Applies the impulses each joint held its bodies with last.
pub fn warm_start_impulse_joints<J: ImpulseJoint>(
    mut solver_bodies: ResMut<SolverBodies>,
    mut active: ResMut<ActiveJoints<J>>,
    solver_config: Res<SolverConfig>,
) {
    let coefficient = solver_config.warm_start_coefficient;
    for active in &mut active.joints {
        active.solve(&mut solver_bodies, |joint, bodies, inertias, impulses| {
            joint.warm_start(bodies, inertias, impulses, coefficient);
        });
    }
}

/// Solves each joint, with its position error if `USE_BIAS`, and without it to relax
/// what that added.
pub fn solve_impulse_joints<J: ImpulseJoint, const USE_BIAS: bool>(
    mut solver_bodies: ResMut<SolverBodies>,
    mut active: ResMut<ActiveJoints<J>>,
    solver_config: Res<SolverConfig>,
    time: Res<Time>,
) {
    let delta_secs = time.delta_secs();
    if delta_secs <= 0.0 || active.joints.is_empty() {
        return;
    }
    let pass = Pass::new(USE_BIAS, &solver_config, delta_secs);
    for active in &mut active.joints {
        active.solve(&mut solver_bodies, |joint, bodies, inertias, impulses| {
            joint.solve(bodies, inertias, impulses, &pass);
        });
    }
}

/// Keeps the impulses each joint holds its bodies with for the next step's warm start,
/// and reports the force and the torque about its anchor that the joint applied to its
/// first body over the step.
pub fn finish_impulse_joints<J: ImpulseJoint>(
    mut joints: Query<(&mut J::Impulses, Option<&mut JointForces>), With<J>>,
    active: Res<ActiveJoints<J>>,
    time: Res<Time>,
) {
    let per_second = time.delta_secs().recip_or_zero();
    for active in &active.joints {
        let Ok((mut impulses, forces)) = joints.get_mut(active.entity) else {
            continue;
        };
        impulses.clone_from(&active.impulses);
        if let Some(mut forces) = forces {
            let (linear, angular, motor) = J::totals(&active.impulses);
            forces.set_force(-linear * per_second);
            forces.set_torque(-angular * per_second);
            forces.set_motor_force(motor * per_second);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A turn of the axes, to stand things at no special angle.
    #[cfg(feature = "3d")]
    fn askew() -> Mat3 {
        Mat3::from_quat(Quat::from_euler(EulerRot::XYZ, 0.4, -0.7, 1.1))
    }

    #[cfg(feature = "3d")]
    fn turned(values: Vec3, turn: Mat3) -> SymmetricTensor {
        SymmetricTensor::from_mat3_unchecked(turn * Mat3::from_diagonal(values) * turn.transpose())
    }

    /// A rod is held across its length however thin it is, along the axes or askew of
    /// them: what it turns least readily in is still a direction it turns in.
    #[cfg(feature = "3d")]
    #[test]
    fn a_thin_rod_is_held_in_every_direction() {
        // The inverse inertia of 1 kg rods 2 m long: 2 cm square, 1 cm and 3 mm.
        for readiest in [15_000.0, 60_000.0, 660_000.0] {
            let values = Vec3::new(readiest, 3.0, 3.0);
            for turn in [Mat3::IDENTITY, askew()] {
                let k = turned(values, turn);
                for across in [Vec3::X, Vec3::Y, Vec3::Z] {
                    let rhs = turn * across;
                    let solved = solve3(k, rhs);
                    let expected = turn * (across / values);
                    assert!(
                        solved.distance(expected) < 0.02 * expected.length(),
                        "a rod turning {readiest} times as readily about its length: \
                         {solved} for {expected}"
                    );
                }
            }
        }
    }

    /// Where nothing can move, no impulse is given, and what can move is solved as if
    /// that direction were not there.
    #[cfg(feature = "3d")]
    #[test]
    fn a_direction_nothing_moves_in_is_left_out() {
        for turn in [Mat3::IDENTITY, askew()] {
            let k = turned(Vec3::new(4.0, 0.0, 0.5), turn);
            let solved = solve3(k, turn * Vec3::new(2.0, 7.0, 1.0));
            let expected = turn * Vec3::new(0.5, 0.0, 2.0);
            assert!(solved.distance(expected) < 1e-3, "{solved} for {expected}");
        }
    }

    /// How much an impulse moves anything at the most is the same however things are
    /// turned, and no less than the most it moves anything in any one direction.
    #[cfg(feature = "3d")]
    #[test]
    fn the_size_of_a_matrix_is_the_same_however_it_is_turned() {
        for values in [Vec3::new(60_000.0, 3.0, 3.0), Vec3::new(2.0, 5.0, 0.0)] {
            let size = size3(turned(values, Mat3::IDENTITY));
            assert!(size >= values.max_element() && size <= values.max_element() * 1.74);
            let askew = size3(turned(values, askew()));
            assert!((askew - size).abs() < 1e-4 * size, "{askew} for {size}");
        }
    }

    /// Against the same solved in double precision, for matrices from well conditioned to
    /// very badly, at every angle: the solve is as exact as single precision allows.
    #[cfg(feature = "3d")]
    #[test]
    fn the_solve_is_as_exact_as_its_numbers_allow() {
        use bevy::math::{DMat3, DQuat, DVec3};
        let mut seed = 0x2545_f491_u32;
        let mut random = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as f64 / u32::MAX as f64
        };
        let mut worst = [0.0_f64; 2];
        for _ in 0..20_000 {
            let spread = 10f64.powf(random() * 5.0);
            let values = DVec3::new(
                spread * (0.5 + random()),
                0.5 + random(),
                (0.5 + random()) * spread.powf(random()),
            );
            let turn = DMat3::from_quat(
                DQuat::from_euler(
                    EulerRot::XYZ,
                    random() * 6.0,
                    random() * 6.0,
                    random() * 6.0,
                )
                .normalize(),
            );
            let k = turn * DMat3::from_diagonal(values) * turn.transpose();
            let rhs = DVec3::new(random() - 0.5, random() - 0.5, random() - 0.5);
            // What single precision is given, solved exactly.
            let given = SymmetricTensor::from_mat3_unchecked(k.as_mat3());
            let exact = given.to_mat3().as_dmat3().inverse() * rhs.as_vec3().as_dvec3();
            let error = |solved: Vec3| solved.as_dvec3().distance(exact) / exact.length();
            worst[0] = worst[0].max(error(solve3(given, rhs.as_vec3())));
            worst[1] = worst[1].max(error(given.inverse() * rhs.as_vec3()));
        }
        println!("worst error: scaled {:e}, plain {:e}", worst[0], worst[1]);
        assert!(worst[0] < 0.02, "the solve was out by {}", worst[0]);
    }

    #[test]
    fn two_directions_are_solved_as_three_are() {
        let turn = Mat2::from_angle(0.6);
        let turned = |values: Vec2| turn * Mat2::from_diagonal(values) * turn.transpose();
        for values in [Vec2::new(60_000.0, 3.0), Vec2::new(2.0, 5.0)] {
            let solved = solve2(turned(values), turn * Vec2::new(1.0, 1.0));
            let expected = turn * values.recip();
            assert!(
                solved.distance(expected) < 0.02 * expected.length(),
                "{solved} for {expected}"
            );
        }
        let solved = solve2(turned(Vec2::new(4.0, 0.0)), turn * Vec2::new(2.0, 7.0));
        let expected = turn * Vec2::new(0.5, 0.0);
        assert!(solved.distance(expected) < 1e-3, "{solved} for {expected}");
        let solved = solve2(
            Mat2::from_diagonal(Vec2::new(0.0, 4.0)),
            Vec2::new(7.0, 2.0),
        );
        assert!(solved.distance(Vec2::new(0.0, 0.5)) < 1e-6, "{solved}");
    }
}
