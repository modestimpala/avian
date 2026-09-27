use super::{ImpulseJoint, Pass};
use crate::{
    dynamics::solver::solver_body::{SolverBody, SolverBodyInertia},
    prelude::*,
};
use bevy::prelude::*;

/// What the solver keeps for a [`DistanceJoint`].
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Reflect)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[reflect(Component, Debug, PartialEq)]
pub struct DistanceJointImpulses {
    /// The anchor from the first body's center of mass, in the world, as the step began.
    pub r1: Vector,
    /// The anchor from the second body's center of mass, in the world, as the step began.
    pub r2: Vector,
    /// From the first body's center of mass to the second's, as the step began.
    pub center_difference: Vector,
    /// The impulse that keeps the anchors from nearer than their least distance, pushing
    /// the second from the first.
    pub lower_impulse: f32,
    /// The impulse that keeps the anchors from further than their most, drawing the
    /// second to the first.
    pub upper_impulse: f32,
    /// The impulses of the step's substeps so far, added up, as the second body was
    /// given them.
    pub total: Vector,
}

/// The line between the anchors, as the bodies stand within a substep.
struct Line {
    /// The anchors from the centers of mass.
    arms: [Vector; 2],
    /// From the first anchor towards the second.
    direction: Vector,
    /// How far apart they are.
    distance: f32,
    /// How fast an impulse along the line parts the anchors: its inverse effective mass.
    k: f32,
}

impl Line {
    fn of(
        data: &DistanceJointImpulses,
        [body1, body2]: [&SolverBody; 2],
        [inertia1, inertia2]: [&SolverBodyInertia; 2],
    ) -> Option<Self> {
        let r1 = body1.delta_rotation * data.r1;
        let r2 = body2.delta_rotation * data.r2;
        let separation =
            (body2.delta_position - body1.delta_position) + (r2 - r1) + data.center_difference;
        let distance = separation.length();
        if distance <= f32::EPSILON {
            return None;
        }
        let direction = separation / distance;

        let w = inertia1.effective_inv_mass() + inertia2.effective_inv_mass();
        let i1 = inertia1.effective_inv_angular_inertia();
        let i2 = inertia2.effective_inv_angular_inertia();
        let (across1, across2) = (cross(r1, direction), cross(r2, direction));
        #[cfg(feature = "2d")]
        let turning = i1 * across1 * across1 + i2 * across2 * across2;
        #[cfg(feature = "3d")]
        let turning = across1.dot(i1 * across1) + across2.dot(i2 * across2);

        Some(Self {
            arms: [r1, r2],
            direction,
            distance,
            k: direction.dot(w * direction) + turning,
        })
    }

    /// How fast the anchors part.
    fn speed(&self, [body1, body2]: [&SolverBody; 2]) -> f32 {
        (body2.velocity_at_point(self.arms[1]) - body1.velocity_at_point(self.arms[0]))
            .dot(self.direction)
    }

    /// Gives the second body an impulse along the line at its anchor, and takes it from
    /// the first.
    fn push(
        &self,
        [body1, body2]: [&mut SolverBody; 2],
        [inertia1, inertia2]: [&SolverBodyInertia; 2],
        impulse: f32,
    ) {
        let impulse = self.direction * impulse;
        let [r1, r2] = self.arms;
        body1.linear_velocity -= inertia1.effective_inv_mass() * impulse;
        body1.angular_velocity -= inertia1.effective_inv_angular_inertia() * cross(r1, impulse);
        body2.linear_velocity += inertia2.effective_inv_mass() * impulse;
        body2.angular_velocity += inertia2.effective_inv_angular_inertia() * cross(r2, impulse);
    }
}

impl ImpulseJoint for DistanceJoint {
    type Impulses = DistanceJointImpulses;

    fn prepare(
        &self,
        [body1, body2]: [&RigidBodyQueryReadOnlyItem; 2],
        data: &mut DistanceJointImpulses,
    ) {
        data.total = Vector::ZERO;
        let (JointAnchor::Local(anchor1), JointAnchor::Local(anchor2)) =
            (self.anchor1, self.anchor2)
        else {
            return;
        };
        data.r1 = body1.rotation * (anchor1 - body1.center_of_mass.0);
        data.r2 = body2.rotation * (anchor2 - body2.center_of_mass.0);
        data.center_difference = (body2.position.0 - body1.position.0).f32()
            + (body2.rotation * body2.center_of_mass.0 - body1.rotation * body1.center_of_mass.0);
    }

    fn warm_start(
        &self,
        [body1, body2]: [&mut SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
        data: &mut DistanceJointImpulses,
        coefficient: f32,
    ) {
        data.lower_impulse *= coefficient;
        data.upper_impulse *= coefficient;
        if let Some(line) = Line::of(data, [&*body1, &*body2], inertias) {
            line.push(
                [body1, body2],
                inertias,
                data.lower_impulse - data.upper_impulse,
            );
        }
    }

    fn solve(
        &self,
        [body1, body2]: [&mut SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
        data: &mut DistanceJointImpulses,
        pass: &Pass,
    ) {
        let Some(line) = Line::of(data, [&*body1, &*body2], inertias) else {
            return;
        };
        if line.k > f32::EPSILON {
            let lower = pass.limit(
                line.k,
                self.compliance,
                line.distance - self.limits.min,
                line.speed([&*body1, &*body2]),
                data.lower_impulse,
            );
            line.push(
                [&mut *body1, &mut *body2],
                inertias,
                lower - data.lower_impulse,
            );
            data.lower_impulse = lower;

            let upper = pass.limit(
                line.k,
                self.compliance,
                self.limits.max - line.distance,
                -line.speed([&*body1, &*body2]),
                data.upper_impulse,
            );
            line.push([body1, body2], inertias, data.upper_impulse - upper);
            data.upper_impulse = upper;
        }

        if !pass.use_bias {
            // The substep's impulses are settled.
            data.total += line.direction * (data.lower_impulse - data.upper_impulse);
        }
    }

    fn totals(data: &DistanceJointImpulses) -> (Vector, AngularVector, f32) {
        (data.total, AngularVector::default(), 0.0)
    }
}
