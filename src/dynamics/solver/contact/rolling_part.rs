use crate::prelude::*;
use bevy::reflect::Reflect;
#[cfg(feature = "serialize")]
use bevy::reflect::{ReflectDeserialize, ReflectSerialize};

/// How much of the torque a patch's friction could give about its edge it gives about its
/// middle: that of a disc pressed evenly all over.
#[cfg(feature = "3d")]
const TWIST: f32 = 2.0 / 3.0;

/// The part of a [`ContactConstraint`](super::ContactConstraint) that resists the bodies
/// rolling over and twisting on one another, for bodies that touch over a
/// [`ContactPatch`] and not at points. It is as strong as the patch is wide and the
/// bodies are pressed together.
#[derive(Clone, Debug, PartialEq, Reflect)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[reflect(Debug, PartialEq)]
pub struct ContactRollingPart {
    /// The radius of the patch the bodies touch over.
    pub radius: f32,

    /// The angular impulse given to the second body, and taken from the first.
    pub impulse: AngularVector,

    /// How the bodies' turning against one another answers an angular impulse, inverted.
    #[cfg(feature = "2d")]
    pub effective_mass: f32,
    /// How the bodies' turning against one another answers an angular impulse, inverted.
    #[cfg(feature = "3d")]
    pub effective_mass: bevy::math::Mat3,
}

impl ContactRollingPart {
    /// Generates a new [`ContactRollingPart`].
    pub fn generate(
        inverse_angular_inertia1: &SymmetricTensor,
        inverse_angular_inertia2: &SymmetricTensor,
        radius: f32,
        warm_start_impulse: Option<AngularVector>,
    ) -> Self {
        // The bodies' angular velocities are to be the same: J = [0, -1, 0, 1], and
        // K = J M^-1 J^T is the sum of their inverse angular inertias.
        #[cfg(feature = "2d")]
        let effective_mass = (inverse_angular_inertia1 + inverse_angular_inertia2).recip_or_zero();
        #[cfg(feature = "3d")]
        let effective_mass = {
            use crate::dynamics::solver::impulse_joints::solve3;
            let k = *inverse_angular_inertia1 + *inverse_angular_inertia2;
            bevy::math::Mat3::from_cols(
                solve3(k, Vector::X),
                solve3(k, Vector::Y),
                solve3(k, Vector::Z),
            )
        };
        Self {
            radius,
            impulse: warm_start_impulse.unwrap_or_default(),
            effective_mass,
        }
    }

    /// Solves for the bodies turning as one, as far as the patch can make them, updating
    /// the impulse in `self` and returning the angular impulse to give the second body
    /// and take from the first.
    ///
    /// `relative` is the second body's angular velocity less the first's, and
    /// `normal_impulse` what all the manifold's points are pressed together with.
    #[allow(unused_variables)]
    pub fn solve_impulse(
        &mut self,
        relative: AngularVector,
        normal: Vector,
        friction: f32,
        normal_impulse: f32,
    ) -> AngularVector {
        // Pressed together with a force N over a patch of radius a, the force can stand
        // as far as a from the middle of it, and resists rolling with a torque up to a N.
        let rolling = self.radius * normal_impulse;
        #[cfg(feature = "2d")]
        let impulse = (self.impulse - self.effective_mass * relative).clamp(-rolling, rolling);
        #[cfg(feature = "3d")]
        let impulse = {
            let wanted = self.impulse - self.effective_mass * relative;
            // Twisting is resisted by the friction over the patch.
            let twisting = TWIST * friction * rolling;
            let twist = wanted.dot(normal);
            normal * twist.clamp(-twisting, twisting)
                + (wanted - normal * twist).clamp_length_max(rolling)
        };
        let given = impulse - self.impulse;
        self.impulse = impulse;
        given
    }
}
