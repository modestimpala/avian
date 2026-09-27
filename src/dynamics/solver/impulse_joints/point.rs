use super::Pass;
#[cfg(feature = "2d")]
use super::{size2, solve2};
#[cfg(feature = "3d")]
use super::{size3, solve3};
use crate::{
    dynamics::solver::solver_body::{SolverBody, SolverBodyInertia},
    prelude::*,
};
use bevy::prelude::*;

/// A point on one body held to a point on another.
#[derive(Clone, Copy, Debug, Default, PartialEq, Reflect)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[reflect(Debug, PartialEq)]
pub struct PointImpulses {
    /// The anchor from the first body's center of mass, in the world, as the step began.
    pub r1: Vector,
    /// The anchor from the second body's center of mass, in the world, as the step began.
    pub r2: Vector,
    /// From the first body's center of mass to the second's, as the step began.
    pub center_difference: Vector,
    /// The impulse a substep gives the second body at the anchor, and takes from the first.
    /// Kept from one substep and step to the next for warm starting.
    pub impulse: Vector,
    /// The impulses of the step's substeps so far, added up.
    pub total: Vector,
}

impl PointImpulses {
    /// Records where the anchors stand as the step begins.
    pub fn prepare(
        &mut self,
        [body1, body2]: [&RigidBodyQueryReadOnlyItem; 2],
        anchor1: Vector,
        anchor2: Vector,
    ) {
        self.total = Vector::ZERO;
        self.r1 = body1.rotation * (anchor1 - body1.center_of_mass.0);
        self.r2 = body2.rotation * (anchor2 - body2.center_of_mass.0);
        self.center_difference = (body2.position.0 - body1.position.0).f32()
            + (body2.rotation * body2.center_of_mass.0 - body1.rotation * body1.center_of_mass.0);
    }

    /// Applies the impulse the point was held with last.
    pub fn warm_start(
        &mut self,
        bodies: [&mut SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
        coefficient: f32,
    ) {
        self.impulse *= coefficient;
        let arms = self.arms(&bodies);
        push(bodies, inertias, arms, self.impulse);
    }

    /// Solves for the anchors meeting.
    pub fn solve(
        &mut self,
        [body1, body2]: [&mut SolverBody; 2],
        [inertia1, inertia2]: [&SolverBodyInertia; 2],
        compliance: f32,
        pass: &Pass,
    ) {
        let [r1, r2] = self.arms(&[&mut *body1, &mut *body2]);
        let error =
            (body2.delta_position - body1.delta_position) + (r2 - r1) + self.center_difference;
        let speed = body2.velocity_at_point(r2) - body1.velocity_at_point(r1);

        let w = inertia1.effective_inv_mass() + inertia2.effective_inv_mass();
        let i1 = inertia1.effective_inv_angular_inertia();
        let i2 = inertia2.effective_inv_angular_inertia();

        // K = J M^-1 J^T for J = [-1, -skew(r1)^T, 1, skew(r2)^T], as for a contact's
        // normal, in every direction at once.
        #[cfg(feature = "2d")]
        let k = {
            let across = -i1 * r1.x * r1.y - i2 * r2.x * r2.y;
            Mat2::from_cols(
                Vec2::new(w.x + i1 * r1.y * r1.y + i2 * r2.y * r2.y, across),
                Vec2::new(across, w.y + i1 * r1.x * r1.x + i2 * r2.x * r2.x),
            )
        };
        #[cfg(feature = "3d")]
        let k = SymmetricTensor::from_diagonal(w) + i1.skew(r1) + i2.skew(r2);

        let impulse = pass.step(
            |give, rhs| {
                #[cfg(feature = "2d")]
                return solve2(k + Mat2::from_diagonal(Vec2::splat(give)), rhs);
                #[cfg(feature = "3d")]
                return solve3(k + SymmetricTensor::from_diagonal(Vec3::splat(give)), rhs);
            },
            compliance,
            #[cfg(feature = "2d")]
            size2(k),
            #[cfg(feature = "3d")]
            size3(k),
            speed,
            error,
            self.impulse,
        );
        self.impulse += impulse;
        push([body1, body2], [inertia1, inertia2], [r1, r2], impulse);
    }

    /// Adds the substep's impulse to the step's.
    pub fn settle(&mut self) {
        self.total += self.impulse;
    }

    /// The anchors from the centers of mass, as the bodies have turned since.
    fn arms(&self, [body1, body2]: &[&mut SolverBody; 2]) -> [Vector; 2] {
        [
            body1.delta_rotation * self.r1,
            body2.delta_rotation * self.r2,
        ]
    }
}

/// Gives the second body an impulse at the anchor, and takes it from the first.
fn push(
    [body1, body2]: [&mut SolverBody; 2],
    [inertia1, inertia2]: [&SolverBodyInertia; 2],
    [r1, r2]: [Vector; 2],
    impulse: Vector,
) {
    body1.linear_velocity -= inertia1.effective_inv_mass() * impulse;
    body1.angular_velocity -= inertia1.effective_inv_angular_inertia() * cross(r1, impulse);
    body2.linear_velocity += inertia2.effective_inv_mass() * impulse;
    body2.angular_velocity += inertia2.effective_inv_angular_inertia() * cross(r2, impulse);
}
