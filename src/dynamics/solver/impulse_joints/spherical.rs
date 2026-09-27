use super::{ImpulseJoint, Pass, PointImpulses, Turning, turn};
use crate::{
    dynamics::solver::solver_body::{SolverBody, SolverBodyInertia},
    prelude::*,
};
use bevy::prelude::*;

/// What the solver keeps for a [`SphericalJoint`].
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Reflect)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[reflect(Component, Debug, PartialEq)]
pub struct SphericalJointImpulses {
    /// The anchors, held together.
    pub point: PointImpulses,
    /// The axis the swing is measured between, on the first body, in the world, as the
    /// step began.
    pub swing_axis1: Vector,
    /// The same axis on the second body.
    pub swing_axis2: Vector,
    /// The axis the twist is measured between, on the first body, in the world, as the
    /// step began.
    pub twist_axis1: Vector,
    /// The same axis on the second body.
    pub twist_axis2: Vector,
    /// The impulses that keep the swing from under its least and over its most.
    pub swing_impulses: [f32; 2],
    /// The impulses that keep the twist from under its least and over its most.
    pub twist_impulses: [f32; 2],
    /// The angular impulses of the step's substeps so far, added up.
    pub total_angular_impulse: Vector,
}

impl SphericalJointImpulses {
    /// The swing, as the bodies stand within a substep: the angle between the two bodies'
    /// swing axes, about the direction across both.
    fn swing(
        &self,
        [body1, body2]: [&SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
    ) -> Option<Turning> {
        let a1 = body1.delta_rotation * self.swing_axis1;
        let a2 = body2.delta_rotation * self.swing_axis2;
        let across = a1.cross(a2);
        let sine = across.length();
        (sine > f32::EPSILON).then(|| Turning::new(across / sine, sine.atan2(a1.dot(a2)), inertias))
    }

    /// The twist, as the bodies stand within a substep: the angle between the two bodies'
    /// twist axes, about the direction midway between their swing axes.
    fn twist(
        &self,
        [body1, body2]: [&SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
    ) -> Option<Turning> {
        let a1 = body1.delta_rotation * self.swing_axis1;
        let a2 = body2.delta_rotation * self.swing_axis2;
        // Swung right over, there is no telling twist from swing.
        let n = (a1 + a2).try_normalize().filter(|_| a1.dot(a2) > -0.5)?;
        let b1 = body1.delta_rotation * self.twist_axis1;
        let b2 = body2.delta_rotation * self.twist_axis2;
        let n1 = (b1 - n * n.dot(b1)).try_normalize()?;
        let n2 = (b2 - n * n.dot(b2)).try_normalize()?;
        Some(Turning::new(
            n,
            n1.cross(n2).dot(n).atan2(n1.dot(n2)),
            inertias,
        ))
    }

    /// The angular impulse the limits hold with, as the second body is given it.
    fn angular(&self, bodies: [&SolverBody; 2], inertias: [&SolverBodyInertia; 2]) -> Vector {
        let about = |turning: Option<Turning>, [lower, upper]: [f32; 2]| {
            turning.map_or(Vector::ZERO, |turning| turning.about(lower - upper))
        };
        about(self.swing(bodies, inertias), self.swing_impulses)
            + about(self.twist(bodies, inertias), self.twist_impulses)
    }
}

impl ImpulseJoint for SphericalJoint {
    type Impulses = SphericalJointImpulses;

    fn prepare(&self, bodies: [&RigidBodyQueryReadOnlyItem; 2], data: &mut SphericalJointImpulses) {
        data.total_angular_impulse = Vector::ZERO;
        let (Some(anchor1), Some(anchor2), Some(basis1), Some(basis2)) = (
            self.local_anchor1(),
            self.local_anchor2(),
            self.local_basis1(),
            self.local_basis2(),
        ) else {
            return;
        };
        data.point.prepare(bodies, anchor1, anchor2);

        let frame1 = bodies[0].rotation.0 * basis1;
        let frame2 = bodies[1].rotation.0 * basis2;
        let swing_axis = self.twist_axis.any_orthonormal_vector();
        data.swing_axis1 = frame1 * swing_axis;
        data.swing_axis2 = frame2 * swing_axis;
        data.twist_axis1 = frame1 * self.twist_axis;
        data.twist_axis2 = frame2 * self.twist_axis;
    }

    fn warm_start(
        &self,
        [body1, body2]: [&mut SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
        data: &mut SphericalJointImpulses,
        coefficient: f32,
    ) {
        if self.swing_limit.is_none() {
            data.swing_impulses = [0.0; 2];
        }
        if self.twist_limit.is_none() {
            data.twist_impulses = [0.0; 2];
        }
        data.swing_impulses = data.swing_impulses.map(|impulse| impulse * coefficient);
        data.twist_impulses = data.twist_impulses.map(|impulse| impulse * coefficient);
        let angular = data.angular([&*body1, &*body2], inertias);
        turn([&mut *body1, &mut *body2], inertias, angular);
        data.point.warm_start([body1, body2], inertias, coefficient);
    }

    fn solve(
        &self,
        [body1, body2]: [&mut SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
        data: &mut SphericalJointImpulses,
        pass: &Pass,
    ) {
        if let Some(limit) = self.swing_limit
            && let Some(swing) = data.swing([&*body1, &*body2], inertias)
        {
            let [lower, upper] = &mut data.swing_impulses;
            swing.limit(
                [&mut *body1, &mut *body2],
                inertias,
                limit,
                self.swing_compliance,
                lower,
                upper,
                pass,
            );
        }
        if let Some(limit) = self.twist_limit
            && let Some(twist) = data.twist([&*body1, &*body2], inertias)
        {
            let [lower, upper] = &mut data.twist_impulses;
            twist.limit(
                [&mut *body1, &mut *body2],
                inertias,
                limit,
                self.twist_compliance,
                lower,
                upper,
                pass,
            );
        }

        data.point.solve(
            [&mut *body1, &mut *body2],
            inertias,
            self.point_compliance,
            pass,
        );

        if !pass.use_bias {
            // The substep's impulses are settled.
            data.point.settle();
            let angular = data.angular([&*body1, &*body2], inertias);
            data.total_angular_impulse += angular;
        }
    }

    fn totals(data: &SphericalJointImpulses) -> (Vector, AngularVector, f32) {
        (data.point.total, data.total_angular_impulse, 0.0)
    }
}
