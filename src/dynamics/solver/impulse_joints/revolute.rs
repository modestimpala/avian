use core::f32::consts::{PI, TAU};

#[cfg(feature = "3d")]
use super::solve2;
use super::{ImpulseJoint, Pass, PointImpulses, Turning, turn};
use crate::{
    dynamics::{
        joints::MotorModel,
        solver::{
            softness_parameters::SoftnessParameters,
            solver_body::{SolverBody, SolverBodyInertia},
        },
    },
    prelude::*,
};
use bevy::prelude::*;

/// What the solver keeps for a [`RevoluteJoint`].
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Reflect)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[reflect(Component, Debug, PartialEq)]
pub struct RevoluteJointImpulses {
    /// The anchors, held together.
    pub point: PointImpulses,
    /// The angle of the second body's joint frame from the first's, as the step began.
    #[cfg(feature = "2d")]
    pub rotation_difference: f32,
    /// The hinge axis on the first body, in the world, as the step began.
    #[cfg(feature = "3d")]
    pub a1: Vector,
    /// The hinge axis on the second body, in the world, as the step began.
    #[cfg(feature = "3d")]
    pub a2: Vector,
    /// A direction across the hinge axis on the first body, in the world, as the step began.
    #[cfg(feature = "3d")]
    pub b1: Vector,
    /// The same direction on the second body.
    #[cfg(feature = "3d")]
    pub b2: Vector,
    /// The angular impulse that keeps the two hinge axes in line. It lies across them.
    #[cfg(feature = "3d")]
    pub align_impulse: Vector,
    /// The impulse about the hinge axis that keeps the angle from under its least.
    pub lower_impulse: f32,
    /// The impulse against the hinge axis that keeps the angle from over its most.
    pub upper_impulse: f32,
    /// The motor's impulse about the hinge axis.
    pub motor_impulse: f32,
    /// The angular impulses of the step's substeps so far, added up.
    pub total_angular_impulse: AngularVector,
    /// The motor's impulses of the step's substeps so far, added up.
    pub total_motor_impulse: f32,
}

/// The hinge, as the bodies stand within a substep.
fn hinge(
    data: &RevoluteJointImpulses,
    [body1, body2]: [&SolverBody; 2],
    inertias: [&SolverBodyInertia; 2],
) -> Turning {
    #[cfg(feature = "2d")]
    {
        Turning::new(
            data.rotation_difference + body1.delta_rotation.angle_to(body2.delta_rotation),
            inertias,
        )
    }
    #[cfg(feature = "3d")]
    {
        let axis = body1.delta_rotation * data.a1;
        let b1 = body1.delta_rotation * data.b1;
        let b2 = body2.delta_rotation * data.b2;
        Turning::new(axis, b1.cross(b2).dot(axis).atan2(b1.dot(b2)), inertias)
    }
}

impl ImpulseJoint for RevoluteJoint {
    type Impulses = RevoluteJointImpulses;

    fn prepare(&self, bodies: [&RigidBodyQueryReadOnlyItem; 2], data: &mut RevoluteJointImpulses) {
        data.total_angular_impulse = AngularVector::default();
        data.total_motor_impulse = 0.0;
        let (Some(anchor1), Some(anchor2), Some(basis1), Some(basis2)) = (
            self.local_anchor1(),
            self.local_anchor2(),
            self.local_basis1(),
            self.local_basis2(),
        ) else {
            return;
        };
        data.point.prepare(bodies, anchor1, anchor2);

        let frame1 = Rot::from(*bodies[0].rotation) * basis1;
        let frame2 = Rot::from(*bodies[1].rotation) * basis2;
        #[cfg(feature = "2d")]
        {
            data.rotation_difference = frame1.angle_to(frame2);
        }
        #[cfg(feature = "3d")]
        {
            let across = self.hinge_axis.any_orthonormal_vector();
            data.a1 = frame1 * self.hinge_axis;
            data.a2 = frame2 * self.hinge_axis;
            data.b1 = frame1 * across;
            data.b2 = frame2 * across;
        }
    }

    fn warm_start(
        &self,
        [body1, body2]: [&mut SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
        data: &mut RevoluteJointImpulses,
        coefficient: f32,
    ) {
        // Only a sprung motor holds an impulse: the others push by their own laws, anew
        // each substep.
        let sprung = matches!(self.motor.motor_model, MotorModel::SpringDamper { .. });
        if !self.motor.enabled || !sprung {
            data.motor_impulse = 0.0;
        }
        if self.angle_limit.is_none() {
            data.lower_impulse = 0.0;
            data.upper_impulse = 0.0;
        }
        data.lower_impulse *= coefficient;
        data.upper_impulse *= coefficient;
        data.motor_impulse *= coefficient;

        let hinge = hinge(data, [&*body1, &*body2], inertias);
        let about = data.motor_impulse + data.lower_impulse - data.upper_impulse;
        #[cfg(feature = "2d")]
        let angular = hinge.about(about);
        #[cfg(feature = "3d")]
        let angular = {
            // What held the axes in line lies across them as they are now.
            data.align_impulse = coefficient
                * (data.align_impulse - hinge.axis * data.align_impulse.dot(hinge.axis));
            data.align_impulse + hinge.about(about)
        };
        turn([&mut *body1, &mut *body2], inertias, angular);
        data.point.warm_start([body1, body2], inertias, coefficient);
    }

    fn solve(
        &self,
        [body1, body2]: [&mut SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
        data: &mut RevoluteJointImpulses,
        pass: &Pass,
    ) {
        #[cfg(feature = "3d")]
        self.align([&mut *body1, &mut *body2], inertias, data, pass);

        // Motors before limits: what is solved later is kept more exactly.
        self.drive([&mut *body1, &mut *body2], inertias, data, pass);
        self.limit([&mut *body1, &mut *body2], inertias, data, pass);

        data.point.solve(
            [&mut *body1, &mut *body2],
            inertias,
            self.point_compliance,
            pass,
        );

        if !pass.use_bias {
            // The substep's impulses are settled.
            data.point.settle();
            let hinge = hinge(data, [&*body1, &*body2], inertias);
            let about = data.motor_impulse + data.lower_impulse - data.upper_impulse;
            data.total_motor_impulse += data.motor_impulse;
            #[cfg(feature = "2d")]
            {
                data.total_angular_impulse += hinge.about(about);
            }
            #[cfg(feature = "3d")]
            {
                let angular = data.align_impulse + hinge.about(about);
                data.total_angular_impulse += angular;
            }
        }
    }

    fn totals(data: &RevoluteJointImpulses) -> (Vector, AngularVector, f32) {
        (
            data.point.total,
            data.total_angular_impulse,
            #[cfg(feature = "2d")]
            data.total_motor_impulse,
            #[cfg(feature = "3d")]
            data.total_motor_impulse.abs(),
        )
    }
}

impl RevoluteJoint {
    /// Keeps the two bodies' hinge axes in line, leaving the turn about them free.
    #[cfg(feature = "3d")]
    fn align(
        &self,
        [body1, body2]: [&mut SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
        data: &mut RevoluteJointImpulses,
        pass: &Pass,
    ) {
        let a1 = body1.delta_rotation * data.a1;
        let a2 = body2.delta_rotation * data.a2;
        // Two directions across the axis. Along it nothing is asked.
        let u = body1.delta_rotation * data.b1;
        let v = a1.cross(u);
        let across = |vector: Vector| Vec2::new(vector.dot(u), vector.dot(v));

        let k = inertias[0].effective_inv_angular_inertia()
            + inertias[1].effective_inv_angular_inertia();
        let (ku, kv) = (k * u, k * v);
        let k = Mat2::from_cols(across(ku), across(kv));

        let given = across(data.align_impulse);
        let impulse = pass.step(
            |give, rhs| solve2(k + Mat2::from_diagonal(Vec2::splat(give)), rhs),
            self.align_compliance,
            across(body2.angular_velocity - body1.angular_velocity),
            // How the second axis is turned from the first.
            across(a1.cross(a2)),
            given,
        );
        data.align_impulse = u * (given.x + impulse.x) + v * (given.y + impulse.y);
        turn([body1, body2], inertias, u * impulse.x + v * impulse.y);
    }

    /// Keeps the angle within its limits.
    fn limit(
        &self,
        [body1, body2]: [&mut SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
        data: &mut RevoluteJointImpulses,
        pass: &Pass,
    ) {
        let Some(limit) = self.angle_limit else {
            return;
        };
        hinge(data, [&*body1, &*body2], inertias).limit(
            [body1, body2],
            inertias,
            limit,
            self.limit_compliance,
            &mut data.lower_impulse,
            &mut data.upper_impulse,
            pass,
        );
    }

    /// Drives the angle and its speed towards the motor's targets.
    fn drive(
        &self,
        [body1, body2]: [&mut SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
        data: &mut RevoluteJointImpulses,
        pass: &Pass,
    ) {
        let motor = &self.motor;
        if !motor.enabled {
            return;
        }
        let hinge = hinge(data, [&*body1, &*body2], inertias);
        if hinge.k <= f32::EPSILON {
            return;
        }

        let velocity_error = motor.target_velocity - hinge.speed([&*body1, &*body2]);
        // The shortest way round.
        let position_error = (motor.target_position - hinge.angle + PI).rem_euclid(TAU) - PI;
        // Zero is no torque at all, not no limit: a spent brake must let go.
        let most = if motor.max_torque < f32::MAX {
            motor.max_torque.max(0.0) * pass.delta_secs
        } else {
            f32::INFINITY
        };

        let held = data.motor_impulse;
        data.motor_impulse = match motor.motor_model {
            // A spring and damper, integrated implicitly: a soft constraint, solved in
            // both passes against the impulse it holds.
            MotorModel::SpringDamper {
                frequency,
                damping_ratio,
            } => {
                let soft = SoftnessParameters::new(damping_ratio, frequency)
                    .compute_coefficients(pass.delta_secs);
                let impulse = soft.mass_scale * (velocity_error + soft.bias * position_error)
                    / hinge.k
                    - soft.impulse_scale * held;
                (held + impulse).clamp(-most, most)
            }
            // The other models are laws of force. They push once in a substep.
            _ if !pass.use_bias => held,
            MotorModel::AccelerationBased { stiffness, damping } => {
                let acceleration = stiffness * position_error + damping * velocity_error;
                (acceleration * pass.delta_secs / hinge.k).clamp(-most, most)
            }
            MotorModel::ForceBased { stiffness, damping } => {
                let torque = stiffness * position_error + damping * velocity_error;
                (torque * pass.delta_secs).clamp(-most, most)
            }
        };
        turn(
            [body1, body2],
            inertias,
            hinge.about(data.motor_impulse - held),
        );
    }
}
