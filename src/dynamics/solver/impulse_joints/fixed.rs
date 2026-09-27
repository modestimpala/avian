#[cfg(feature = "3d")]
use super::solve3;
use super::{ImpulseJoint, Pass, PointImpulses, turn};
use crate::{
    dynamics::solver::solver_body::{SolverBody, SolverBodyInertia},
    prelude::*,
};
use bevy::prelude::*;

/// What the solver keeps for a [`FixedJoint`].
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Reflect)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[reflect(Component, Debug, PartialEq)]
pub struct FixedJointImpulses {
    /// The anchors, held together.
    pub point: PointImpulses,
    /// The angle of the second body's joint frame from the first's, as the step began.
    #[cfg(feature = "2d")]
    pub rotation_difference: f32,
    /// The rotation taking the second body's joint frame to the first's, as the step began.
    #[cfg(feature = "3d")]
    pub rotation_difference: Quat,
    /// The angular compliance for each direction of rotation in the world, if it differs
    /// by direction.
    #[cfg(feature = "3d")]
    pub angle_compliance: Option<SymmetricTensor>,
    /// The angular impulse a substep gives the second body, and takes from the first.
    /// Kept from one substep and step to the next for warm starting.
    pub angular_impulse: AngularVector,
    /// The angular impulses of the step's substeps so far, added up.
    pub total_angular_impulse: AngularVector,
}

impl ImpulseJoint for FixedJoint {
    type Impulses = FixedJointImpulses;

    fn prepare(&self, bodies: [&RigidBodyQueryReadOnlyItem; 2], data: &mut FixedJointImpulses) {
        data.total_angular_impulse = AngularVector::default();
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
            data.rotation_difference = frame1 * frame2.inverse();
            data.angle_compliance = self.angle_compliance_tensor.map(|compliance| {
                let basis = Mat3::from_quat(frame1);
                SymmetricTensor::from_mat3_unchecked(
                    basis * compliance.to_mat3() * basis.transpose(),
                )
            });
        }
    }

    fn warm_start(
        &self,
        [body1, body2]: [&mut SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
        data: &mut FixedJointImpulses,
        coefficient: f32,
    ) {
        data.angular_impulse *= coefficient;
        turn([&mut *body1, &mut *body2], inertias, data.angular_impulse);
        data.point.warm_start([body1, body2], inertias, coefficient);
    }

    fn solve(
        &self,
        [body1, body2]: [&mut SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
        data: &mut FixedJointImpulses,
        pass: &Pass,
    ) {
        // The angle first: the anchors then meet about the turned bodies.
        let k = inertias[0].effective_inv_angular_inertia()
            + inertias[1].effective_inv_angular_inertia();
        let speed = body2.angular_velocity - body1.angular_velocity;
        #[cfg(feature = "2d")]
        let impulse = {
            let error =
                data.rotation_difference + body1.delta_rotation.angle_to(body2.delta_rotation);
            pass.step(
                |give, rhs| (k + give).recip_or_zero() * rhs,
                self.angle_compliance,
                speed,
                error,
                data.angular_impulse,
            )
        };
        #[cfg(feature = "3d")]
        let impulse = {
            // The turn takes the second frame to the first.
            let turn =
                body1.delta_rotation * data.rotation_difference * body2.delta_rotation.inverse();
            let error = if turn.w < 0.0 { 2.0 } else { -2.0 } * turn.xyz();
            match data.angle_compliance {
                // A spring, softer one way than another.
                Some(compliance) => {
                    let give = compliance / (pass.delta_secs * pass.delta_secs);
                    -solve3(
                        k + give,
                        speed + error / pass.delta_secs + give * data.angular_impulse,
                    )
                }
                None => pass.step(
                    |give, rhs| solve3(k + SymmetricTensor::from_diagonal(Vec3::splat(give)), rhs),
                    self.angle_compliance,
                    speed,
                    error,
                    data.angular_impulse,
                ),
            }
        };
        data.angular_impulse += impulse;
        turn([&mut *body1, &mut *body2], inertias, impulse);

        data.point
            .solve([body1, body2], inertias, self.point_compliance, pass);

        if !pass.use_bias {
            // The substep's impulses are settled.
            data.point.settle();
            let angular = data.angular_impulse;
            data.total_angular_impulse += angular;
        }
    }

    fn totals(data: &FixedJointImpulses) -> (Vector, AngularVector, f32) {
        (data.point.total, data.total_angular_impulse, 0.0)
    }
}
