//! Joints solved with velocity impulses, in the same passes as contacts.
//!
//! An [XPBD joint](super::xpbd) corrects positions once contacts and friction have been
//! solved and the bodies moved, so friction gets no say in the motion the joint makes:
//! a body held by friction under a steady joint load is carried along a little every
//! substep. A joint solved here is [warm started], [solved] and [relaxed] next to the
//! contacts instead, each answering the other's impulses before anything moves.
//!
//! The [`FixedJoint`] and the [`RevoluteJoint`] are solved this way.
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

mod fixed;
mod point;
mod revolute;

pub use fixed::FixedJointImpulses;
pub use point::PointImpulses;
pub use revolute::RevoluteJointImpulses;

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
pub trait ImpulseJoint: Component + EntityConstraint<2> {
    /// What the solver keeps for the joint: where it stood when the step began, and the
    /// impulses it holds its bodies with.
    type Impulses: Component<Mutability = Mutable> + Default;

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
            delta_secs,
        }
    }

    /// The step in impulse that answers a constraint's speed and error, for what it has
    /// given so far.
    ///
    /// `solve` applies the inverse of a matrix to a vector: the inverse effective mass,
    /// with the given amount added to its diagonal.
    pub(super) fn step<V>(
        &self,
        solve: impl Fn(f32, V) -> V,
        compliance: f32,
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
            let give = compliance / (self.delta_secs * self.delta_secs);
            -solve(give, speed + error * self.delta_secs.recip() + given * give)
        } else {
            -solve(0.0, speed + error * self.softness.bias) * self.softness.mass_scale
                - given * self.softness.impulse_scale
        }
    }
}

/// The inverse of `k` applied to `rhs`, in the directions `k` can move anything in.
///
/// Locked axes and bodies that cannot move leave directions where no impulse has any
/// effect. None is given there.
pub(super) fn solve2(k: Mat2, rhs: Vec2) -> Vec2 {
    let scale = k.x_axis.x.max(k.y_axis.y);
    if scale <= 0.0 {
        return Vec2::ZERO;
    }
    if k.determinant() > 1e-6 * scale * scale {
        return k.inverse() * rhs;
    }
    // One direction is left: k = l v v^T, whose trace is l.
    let trace = k.x_axis.x + k.y_axis.y;
    k * rhs / (trace * trace)
}

/// The inverse of `k` applied to `rhs`, in the directions `k` can move anything in.
///
/// Locked axes and bodies that cannot move leave directions where no impulse has any
/// effect. None is given there.
#[cfg(feature = "3d")]
pub(super) fn solve3(k: SymmetricTensor, rhs: Vec3) -> Vec3 {
    let scale = k.diagonal().max_element();
    if scale <= 0.0 {
        return Vec3::ZERO;
    }
    if k.determinant() > 1e-6 * scale * scale * scale {
        return k.inverse() * rhs;
    }
    let eigen = glam_matrix_extras::SymmetricEigen3::new(k);
    (0..3)
        .filter(|&i| eigen.eigenvalues[i] > 1e-4 * scale)
        .map(|i| {
            let direction = eigen.eigenvectors.col(i);
            direction * (direction.dot(rhs) / eigen.eigenvalues[i])
        })
        .sum()
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

/// The two bodies of a joint, with stand-ins for those the solver does not move.
fn for_bodies(
    solver_bodies: &mut SolverBodies,
    indices: &Query<&SolverBodyIndex, Without<RigidBodyDisabled>>,
    [entity1, entity2]: [Entity; 2],
    solve: impl FnOnce([&mut SolverBody; 2], [&SolverBodyInertia; 2]),
) {
    let index1 = indices
        .get(entity1)
        .copied()
        .unwrap_or(SolverBodyIndex::INVALID);
    let index2 = indices
        .get(entity2)
        .copied()
        .unwrap_or(SolverBodyIndex::INVALID);
    if index1 == index2 {
        return;
    }

    let mut dummy_body1 = SolverBody::DUMMY;
    let mut dummy_body2 = SolverBody::DUMMY;
    let (mut body1, mut inertia1) = (&mut dummy_body1, &SolverBodyInertia::DUMMY);
    let (mut body2, mut inertia2) = (&mut dummy_body2, &SolverBodyInertia::DUMMY);

    let access = solver_bodies.access();
    // SAFETY: The two jointed bodies are distinct, and joints are processed serially here.
    let (b1, b2) = unsafe { access.get_pair_unchecked_mut(index1, index2) };
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

    solve([body1, body2], [inertia1, inertia2]);
}

/// Records where each joint stands as the step begins.
pub fn prepare_impulse_joints<J: ImpulseJoint>(
    bodies: Query<RigidBodyQueryReadOnly, Without<RigidBodyDisabled>>,
    mut joints: Query<(&J, &mut J::Impulses), (Without<RigidBody>, Without<JointDisabled>)>,
) {
    for (joint, mut impulses) in &mut joints {
        if let Ok([body1, body2]) = bodies.get_many(joint.entities()) {
            joint.prepare([&body1, &body2], &mut impulses);
        }
    }
}

/// Applies the impulses each joint held its bodies with last.
pub fn warm_start_impulse_joints<J: ImpulseJoint>(
    mut solver_bodies: ResMut<SolverBodies>,
    indices: Query<&SolverBodyIndex, Without<RigidBodyDisabled>>,
    mut joints: Query<(&J, &mut J::Impulses), (Without<RigidBody>, Without<JointDisabled>)>,
    solver_config: Res<SolverConfig>,
) {
    let coefficient = solver_config.warm_start_coefficient;
    for (joint, mut impulses) in &mut joints {
        for_bodies(
            &mut solver_bodies,
            &indices,
            joint.entities(),
            |bodies, inertias| joint.warm_start(bodies, inertias, &mut impulses, coefficient),
        );
    }
}

/// Solves each joint, with its position error if `USE_BIAS`, and without it to relax
/// what that added.
pub fn solve_impulse_joints<J: ImpulseJoint, const USE_BIAS: bool>(
    mut solver_bodies: ResMut<SolverBodies>,
    indices: Query<&SolverBodyIndex, Without<RigidBodyDisabled>>,
    mut joints: Query<(&J, &mut J::Impulses), (Without<RigidBody>, Without<JointDisabled>)>,
    solver_config: Res<SolverConfig>,
    time: Res<Time>,
) {
    let delta_secs = time.delta_secs();
    if delta_secs <= 0.0 {
        return;
    }
    let pass = Pass::new(USE_BIAS, &solver_config, delta_secs);
    for (joint, mut impulses) in &mut joints {
        for_bodies(
            &mut solver_bodies,
            &indices,
            joint.entities(),
            |bodies, inertias| joint.solve(bodies, inertias, &mut impulses, &pass),
        );
    }
}

/// Reports the force and the torque about its anchor that each joint applied to its first
/// body over the step.
pub fn write_impulse_joint_forces<J: ImpulseJoint>(
    mut joints: Query<(&J::Impulses, &mut JointForces), With<J>>,
    time: Res<Time>,
) {
    let per_second = time.delta_secs().recip_or_zero();
    for (impulses, mut forces) in &mut joints {
        let (linear, angular, motor) = J::totals(impulses);
        forces.set_force(-linear * per_second);
        forces.set_torque(-angular * per_second);
        forces.set_motor_force(motor * per_second);
    }
}
