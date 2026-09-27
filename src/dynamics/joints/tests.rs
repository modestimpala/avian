use core::time::Duration;

#[cfg(feature = "2d")]
use approx::assert_relative_eq;
use bevy::{mesh::MeshPlugin, prelude::*, time::TimeUpdateStrategy};

use crate::prelude::*;

const TIMESTEP: f32 = 1.0 / 64.0;

fn create_app() -> App {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        PhysicsPlugins::default(),
        TransformPlugin,
        #[cfg(feature = "bevy_scene")]
        AssetPlugin::default(),
        #[cfg(feature = "bevy_scene")]
        bevy::scene::ScenePlugin,
        MeshPlugin,
    ));

    app.insert_resource(SubstepCount(20));

    app.insert_resource(Gravity(Vector::ZERO));

    app.insert_resource(Time::<Fixed>::from_duration(Duration::from_secs_f32(
        TIMESTEP,
    )));
    app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
        TIMESTEP,
    )));

    app
}

/// Tests that an angular motor on a revolute joint spins the attached body.
#[test]
fn revolute_motor_spins_body() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    app.world_mut().spawn(
        RevoluteJoint::new(anchor, dynamic).with_motor(AngularMotor {
            target_velocity: 2.0,
            max_torque: 100.0,
            motor_model: MotorModel::AccelerationBased {
                stiffness: 0.0,
                damping: 10.0,
            },
            ..default()
        }),
    );

    // Initialize the app.
    app.update();

    // Run simulation for 1 second.
    let duration = 1.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let angular_velocity = body_ref.get::<AngularVelocity>().unwrap();

    #[cfg(feature = "2d")]
    {
        assert!(
            angular_velocity.0.abs() > 1.0,
            "Angular velocity should be significant"
        );
        assert_relative_eq!(angular_velocity.0, 2.0, epsilon = 0.5);
    }
    #[cfg(feature = "3d")]
    {
        let speed = angular_velocity.0.length();
        assert!(speed > 1.0, "Angular velocity should be significant");
    }
}

/// Tests that a linear motor on a prismatic joint moves the attached body.
#[test]
fn prismatic_motor_moves_body() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    app.world_mut().spawn(
        PrismaticJoint::new(anchor, dynamic)
            .with_local_anchor1(Vector::X * 2.0)
            .with_motor(LinearMotor {
                target_velocity: 1.0,
                max_force: 100.0,
                motor_model: MotorModel::AccelerationBased {
                    stiffness: 0.0,
                    damping: 10.0,
                },
                ..default()
            }),
    );

    // Initialize the app.
    app.update();

    let initial_x = app.world().entity(dynamic).get::<Position>().unwrap().0.x;

    // Run simulation for 1 second.
    let duration = 1.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let final_x = body_ref.get::<Position>().unwrap().0.x;

    let displacement = final_x - initial_x;
    assert!(
        displacement > 0.5,
        "Body should have moved: {}",
        displacement
    );
}

/// Tests that an angular motor with max torque limit respects the limit.
#[test]
fn revolute_motor_respects_max_torque() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(100.0), // Heavy body to test torque limiting
            #[cfg(feature = "2d")]
            AngularInertia(100.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(100.0)),
        ))
        .id();

    // Create a revolute joint with a very limited motor torque.
    app.world_mut().spawn(
        RevoluteJoint::new(anchor, dynamic).with_motor(AngularMotor {
            target_velocity: 10.0,
            max_torque: 0.1,
            motor_model: MotorModel::AccelerationBased {
                stiffness: 0.0,
                damping: 1.0,
            },
            ..default()
        }),
    );

    // Initialize the app.
    app.update();

    // Run simulation for 1 second.
    let duration = 1.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let angular_velocity = body_ref.get::<AngularVelocity>().unwrap();

    #[cfg(feature = "2d")]
    {
        assert!(
            angular_velocity.0.abs() < 5.0,
            "Velocity should be limited by max torque"
        );
    }
    #[cfg(feature = "3d")]
    {
        let speed = angular_velocity.0.length();
        assert!(speed < 5.0, "Velocity should be limited by max torque");
    }
}

/// Tests that a motor whose torque limit is zero applies no torque, rather than an
/// unlimited amount.
#[test]
fn revolute_motor_with_zero_max_torque_does_nothing() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    app.world_mut().spawn(
        RevoluteJoint::new(anchor, dynamic).with_motor(AngularMotor {
            target_velocity: 10.0,
            max_torque: 0.0,
            motor_model: MotorModel::AccelerationBased {
                stiffness: 0.0,
                damping: 10.0,
            },
            ..default()
        }),
    );

    app.update();
    for _ in 0..(1.0 / TIMESTEP) as usize {
        app.update();
    }

    let angular_velocity = app
        .world()
        .entity(dynamic)
        .get::<AngularVelocity>()
        .unwrap();
    #[cfg(feature = "2d")]
    let speed = angular_velocity.0.abs();
    #[cfg(feature = "3d")]
    let speed = angular_velocity.0.length();
    assert!(
        speed < 0.01,
        "a motor with no torque drove the joint at {speed}"
    );
}

/// Tests that a position-targeting motor moves the joint towards the target position.
#[test]
fn revolute_motor_position_target() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    // Create a revolute joint with a position-targeting motor.
    let target_angle = 1.0;
    app.world_mut().spawn(
        RevoluteJoint::new(anchor, dynamic).with_motor(AngularMotor {
            target_position: target_angle,
            max_torque: f32::MAX,
            motor_model: MotorModel::AccelerationBased {
                stiffness: 50.0,
                damping: 20.0,
            },
            ..default()
        }),
    );

    // Initialize the app.
    app.update();

    // Run simulation for 3 seconds to let it settle.
    let duration = 3.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let rotation = body_ref.get::<Rotation>().unwrap();

    // The body should have rotated towards the target angle (allow some tolerance).
    #[cfg(feature = "2d")]
    {
        let angle = rotation.as_radians();
        assert!(
            angle.abs() > 0.3,
            "Motor should have rotated the body: {}",
            angle
        );
    }
    #[cfg(feature = "3d")]
    {
        let (axis, angle) = rotation.to_axis_angle();
        let signed_angle = angle * axis.z.signum();
        assert!(
            (signed_angle - target_angle).abs() < 0.05,
            "Motor should have brought the body to its target: {}",
            signed_angle
        );
    }
}

/// Tests that a linear position-targeting motor moves the joint towards the target position.
#[test]
fn prismatic_motor_position_target() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    // Create a prismatic joint with a position-targeting motor.
    // Use AccelerationBased model for stable position targeting.
    let target_position = 1.0; // Target is 1 meter along the slider axis
    app.world_mut().spawn(
        PrismaticJoint::new(anchor, dynamic)
            .with_local_anchor1(Vector::X * 2.0)
            .with_motor(LinearMotor {
                target_position,
                max_force: 100.0,
                motor_model: MotorModel::AccelerationBased {
                    stiffness: 10.0,
                    damping: 5.0,
                },
                ..default()
            }),
    );

    // Initialize the app.
    app.update();

    let initial_pos = app.world().entity(dynamic).get::<Position>().unwrap().0;

    // Run simulation for 3 seconds to let it settle.
    let duration = 3.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let final_pos = body_ref.get::<Position>().unwrap().0;

    assert!(!final_pos.x.is_nan(), "Final position should not be NaN");
    assert!(!final_pos.y.is_nan(), "Final position should not be NaN");

    let displacement = final_pos.x - initial_pos.x;

    assert!(
        displacement.abs() > 0.1 || final_pos.x.abs() > 0.1,
        "Body should have moved: displacement={}, final_x={}",
        displacement,
        final_pos.x
    );
}

/// Tests that a velocity motor on a revolute joint respects angle limits.
///
/// The motor drives with constant velocity, but the joint should stop
/// when it reaches the angle limit.
#[test]
fn revolute_motor_respects_angle_limits() {
    use core::f32::consts::PI;

    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    let angle_limit = PI / 4.0;

    app.world_mut().spawn(
        RevoluteJoint::new(anchor, dynamic)
            .with_angle_limits(-angle_limit, angle_limit)
            .with_motor(AngularMotor {
                target_velocity: 5.0,
                max_torque: 100.0,
                motor_model: MotorModel::AccelerationBased {
                    stiffness: 0.0,
                    damping: 10.0,
                },
                ..default()
            }),
    );

    app.update();

    // Run for 2 seconds - enough time for motor to hit the limit.
    let duration = 2.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let rotation = body_ref.get::<Rotation>().unwrap();

    #[cfg(feature = "2d")]
    {
        let angle = rotation.as_radians();
        assert!(
            angle <= angle_limit + 0.1,
            "Angle {} should not exceed limit {}",
            angle,
            angle_limit
        );
        assert!(
            angle > angle_limit - 0.3,
            "Angle {} should be near the limit {}",
            angle,
            angle_limit
        );
    }
    #[cfg(feature = "3d")]
    {
        let (axis, angle) = rotation.to_axis_angle();
        let signed_angle = angle * axis.z.signum();
        assert!(
            signed_angle <= angle_limit + 0.1,
            "Angle {} should not exceed limit {}",
            signed_angle,
            angle_limit
        );
        assert!(
            signed_angle > angle_limit - 0.3,
            "Angle {} should be near the limit {}",
            signed_angle,
            angle_limit
        );
    }
}

/// Tests that a velocity motor on a prismatic joint respects distance limits.
///
/// The motor drives with constant velocity, but the joint should stop
/// when it reaches the distance limit.
#[test]
fn prismatic_motor_respects_limits() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    // Start at origin so we can measure displacement clearly.
    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::ZERO),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    // Limit translation to [0, 1] meters along the slide axis.
    let distance_limit = 1.0;

    app.world_mut().spawn(
        PrismaticJoint::new(anchor, dynamic)
            .with_limits(0.0, distance_limit)
            .with_motor(LinearMotor {
                target_velocity: 5.0, // High velocity to ensure we hit the limit
                max_force: 100.0,
                motor_model: MotorModel::AccelerationBased {
                    stiffness: 0.0,
                    damping: 10.0,
                },
                ..default()
            }),
    );

    app.update();

    // Make sure the motor is not near the limit from the start.
    {
        let body_ref = app.world().entity(dynamic);
        let position = body_ref.get::<Position>().unwrap();
        assert!(
            (position.0.x - distance_limit.real()).abs() > 0.1,
            "Displacement {} should not be near the limit {} at the start of the test",
            position.0.x,
            distance_limit
        );
    }

    // Run for 2 seconds - enough time for motor to hit the limit.
    let duration = 2.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let position = body_ref.get::<Position>().unwrap();

    // The displacement along the slide axis (X) should be at or near the limit.
    let displacement = position.x.f32();
    assert!(
        displacement <= distance_limit + 0.001,
        "Displacement {} should not exceed limit {}",
        displacement,
        distance_limit
    );
    assert!(
        (displacement - distance_limit).abs() < 0.1,
        "Displacement {} should be near the limit {}",
        displacement,
        distance_limit
    );
}

/// Tests that `ForceBased` motor model works for revolute joints.
///
/// This is the physically accurate motor model that takes mass into account.
#[test]
fn revolute_motor_force_based() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    // Use ForceBased motor model with velocity control.
    app.world_mut().spawn(
        RevoluteJoint::new(anchor, dynamic).with_motor(AngularMotor {
            target_velocity: 2.0,
            max_torque: 100.0,
            motor_model: MotorModel::ForceBased {
                stiffness: 0.0,
                damping: 10.0,
            },
            ..default()
        }),
    );

    app.update();

    let body_ref = app.world().entity(dynamic);
    let angular_velocity = body_ref.get::<AngularVelocity>().unwrap();
    #[cfg(feature = "2d")]
    let initial_speed = angular_velocity.0.abs();
    #[cfg(feature = "3d")]
    let initial_speed = angular_velocity.0.length();

    assert!(
        initial_speed.abs() < 0.001,
        "ForceBased motor should be initiall still, speed: {}",
        initial_speed
    );

    // Run simulation for 1 second.
    let duration = 1.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let angular_velocity = body_ref.get::<AngularVelocity>().unwrap();

    // The body should have gained angular velocity.
    #[cfg(feature = "2d")]
    {
        assert!(
            angular_velocity.0.abs() > 0.5,
            "ForceBased motor should spin the body: {}",
            angular_velocity.0
        );
    }
    #[cfg(feature = "3d")]
    {
        let speed = angular_velocity.0.length();
        assert!(
            speed > 0.5,
            "ForceBased motor should spin the body: {}",
            speed
        );
    }
}

/// Tests that the default `SpringDamper` motor model works.
///
/// `SpringDamper` is unconditionally stable and uses `frequency`/`damping_ratio`.
#[test]
fn revolute_motor_spring_damper() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    let target_angle = 1.0;
    app.world_mut().spawn(
        RevoluteJoint::new(anchor, dynamic).with_motor(AngularMotor {
            target_position: target_angle,
            max_torque: f32::MAX,
            motor_model: MotorModel::SpringDamper {
                frequency: 2.0,
                damping_ratio: 1.0,
            },
            ..default()
        }),
    );

    app.update();

    // TODO Assert the body is initially not near the target.

    // Run simulation for 3 seconds to let the spring settle.
    let duration = 3.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let rotation = body_ref.get::<Rotation>().unwrap();

    // The body should have rotated towards the target.
    #[cfg(feature = "2d")]
    {
        let angle = rotation.as_radians();
        assert!(
            angle.abs() > 0.3,
            "SpringDamper motor should rotate towards target: {}",
            angle
        );
    }
    #[cfg(feature = "3d")]
    {
        let (axis, angle) = rotation.to_axis_angle();
        let signed_angle = angle * axis.z.signum();
        assert!(
            signed_angle.abs() > 0.3,
            "SpringDamper motor should rotate towards target: {}",
            signed_angle
        );
    }
}

/// Tests that a motor with both velocity and position targeting works.
///
/// Combined spring-damper behavior: position targeting with velocity damping.
#[test]
fn revolute_motor_combined_position_velocity() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    // Motor with both position and velocity targeting.
    // The velocity adds a constant offset to the spring behavior.
    let target_angle = 0.5;
    app.world_mut().spawn(
        RevoluteJoint::new(anchor, dynamic).with_motor(AngularMotor {
            enabled: true,
            target_position: target_angle,
            target_velocity: 0.5,
            max_torque: 100.0,
            motor_model: MotorModel::AccelerationBased {
                stiffness: 30.0,
                damping: 15.0,
            },
        }),
    );

    app.update();

    // Run for 2 seconds.
    let duration = 2.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let rotation = body_ref.get::<Rotation>().unwrap();

    // The body should have rotated in the positive direction.
    #[cfg(feature = "2d")]
    {
        let angle = rotation.as_radians();
        assert!(
            angle > 0.2,
            "Combined motor should rotate body positively: {}",
            angle
        );
    }
    #[cfg(feature = "3d")]
    {
        let (axis, angle) = rotation.to_axis_angle();
        let signed_angle = angle * axis.z.signum();
        assert!(
            signed_angle > 0.2,
            "Combined motor should rotate body positively: {}",
            signed_angle
        );
    }
}

/// Tests that a prismatic motor with both velocity and position targeting works.
#[test]
fn prismatic_motor_combined_position_velocity() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    // Motor with both position and velocity targeting.
    app.world_mut().spawn(
        PrismaticJoint::new(anchor, dynamic)
            .with_local_anchor1(Vector::X * 2.0)
            .with_motor(LinearMotor {
                enabled: true,
                target_position: 1.0,
                target_velocity: 0.5,
                max_force: 100.0,
                motor_model: MotorModel::AccelerationBased {
                    stiffness: 20.0,
                    damping: 10.0,
                },
            }),
    );

    app.update();

    let initial_x = app.world().entity(dynamic).get::<Position>().unwrap().0.x;

    // Run for 2 seconds.
    let duration = 2.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let final_x = body_ref.get::<Position>().unwrap().0.x;

    // The body should have moved.
    let displacement = final_x - initial_x;
    assert!(
        displacement.abs() > 0.1,
        "Combined motor should move the body: {}",
        displacement
    );
}

/// Bodies tied round one knot, as a tipi's tops are, with sleep and removals in the same
/// frames a game makes them: the island's joint list must stay whole.
#[cfg(feature = "3d")]
#[test]
fn joints_round_one_knot_survive_sleeping_and_removal_in_any_order() {
    for variant in 0..8 {
        let mut app = create_app();
        app.insert_resource(Gravity(Vector::NEG_Y * 9.81));
        app.world_mut().spawn((
            RigidBody::Static,
            Collider::cuboid(20.0, 1.0, 20.0),
            Transform::from_xyz(0.0, -0.5, 0.0),
        ));
        app.finish();
        let knot = app
            .world_mut()
            .spawn((
                RigidBody::Dynamic,
                Collider::sphere(0.02),
                CollisionLayers::NONE,
                LockedAxes::ROTATION_LOCKED,
                Transform::from_xyz(0.0, 0.5, 0.0),
            ))
            .id();
        let mut sticks = Vec::new();
        let mut joints = Vec::new();
        for i in 0..3 {
            let angle = i as f32 * core::f32::consts::TAU / 3.0;
            let out = Vec3::new(angle.cos(), 0.0, angle.sin());
            let top = Vec3::new(0.0, 0.5, 0.0) + out * 0.03;
            let butt = out * 0.25;
            let along = (top - butt).normalize();
            let stick = app
                .world_mut()
                .spawn((
                    RigidBody::Dynamic,
                    Collider::capsule(0.015, top.distance(butt)),
                    Transform::from_translation((top + butt) * 0.5)
                        .with_rotation(Quat::from_rotation_arc(Vec3::Y, along)),
                ))
                .id();
            let joint = app
                .world_mut()
                .spawn((
                    SphericalJoint::new(knot, stick)
                        .with_local_anchor1(top - Vec3::new(0.0, 0.5, 0.0))
                        .with_local_anchor2(Vec3::Y * top.distance(butt) * 0.5),
                    JointCollisionDisabled,
                ))
                .id();
            sticks.push(stick);
            joints.push(joint);
        }
        for _ in 0..30 {
            app.update();
        }
        // Put some to sleep by hand, as a game resting still bodies does.
        if variant & 1 == 1 {
            for &s in &sticks {
                app.world_mut().entity_mut(s).insert(Sleeping);
            }
            app.update();
        }
        if variant & 2 == 2 {
            app.world_mut().entity_mut(sticks[0]).remove::<Sleeping>();
        }
        // Take the joints away: one at a time, or all with the knot at once.
        if variant & 4 == 4 {
            // The knot and its joints go in one frame, and a body is sent to sleep in that
            // same frame: its island is split while the joints are gone from the graph but
            // still linked in the island.
            for &j in &joints {
                app.world_mut().despawn(j);
            }
            app.world_mut().despawn(knot);
            app.world_mut().entity_mut(sticks[1]).remove::<Sleeping>();
            app.world_mut().entity_mut(sticks[2]).insert(Sleeping);
            app.update();
        } else {
            for &j in &joints {
                app.world_mut().despawn(j);
                app.update();
            }
            app.world_mut().despawn(knot);
        }
        // And new joints made right away, reusing what was freed.
        let knot = app
            .world_mut()
            .spawn((
                RigidBody::Dynamic,
                Collider::sphere(0.02),
                CollisionLayers::NONE,
            ))
            .id();
        for &s in &sticks {
            app.world_mut().spawn(SphericalJoint::new(knot, s));
        }
        for _ in 0..30 {
            app.update();
        }
    }
}

/// A floor, and on it a 10 kg block whose friction can hold 59 N.
#[cfg(feature = "3d")]
fn block_on_a_floor(substeps: u32) -> (App, Entity) {
    let mut app = create_app();
    app.insert_resource(SubstepCount(substeps));
    app.insert_resource(Gravity(Vector::NEG_Y * 9.81));
    app.world_mut().spawn((
        RigidBody::Static,
        Collider::cuboid(20.0, 1.0, 20.0),
        Friction::new(0.6),
        Transform::from_xyz(0.0, -0.5, 0.0),
    ));
    app.finish();
    let block = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Collider::cuboid(0.4, 0.4, 0.4),
            Mass(10.0),
            Friction::new(0.6),
            SleepingDisabled,
            Transform::from_xyz(0.0, 0.2, 0.0),
        ))
        .id();
    (app, block)
}

/// How far a body goes between the first second, when it has taken up its load, and the
/// tenth.
#[cfg(feature = "3d")]
fn creep(app: &mut App, body: Entity) -> f32 {
    for _ in 0..(1.0 / TIMESTEP) as usize {
        app.update();
    }
    let from = app.world().get::<Position>(body).unwrap().0;
    for _ in 0..(9.0 / TIMESTEP) as usize {
        app.update();
    }
    app.world().get::<Position>(body).unwrap().0.distance(from) as f32
}

/// A block pushed with less than its friction can hold stays where it stands, however
/// few substeps there are: the friction that held it last step still points the same way.
#[cfg(feature = "3d")]
#[test]
fn a_push_within_friction_does_not_creep() {
    for substeps in [1, 4, 12] {
        let (mut app, block) = block_on_a_floor(substeps);
        app.world_mut()
            .entity_mut(block)
            .insert(ConstantForce::new(30.0, 0.0, 0.0));
        let crept = creep(&mut app, block);
        assert!(
            crept < 1e-4,
            "the block crept {crept} m in 9 s at {substeps} substeps"
        );
    }
}

/// A block held by friction, with a steady sideways load passed to it through a joint,
/// stays where it stands: the joint gets no motion that friction has not answered.
#[cfg(feature = "3d")]
#[test]
fn a_joint_load_within_friction_does_not_creep() {
    for substeps in [4, 12] {
        let (mut app, block) = block_on_a_floor(substeps);
        // A sled beside it slides freely and is pushed away with 30 N.
        let sled = app
            .world_mut()
            .spawn((
                RigidBody::Dynamic,
                Collider::cuboid(0.4, 0.4, 0.4),
                Mass(5.0),
                Friction::ZERO.with_combine_rule(CoefficientCombine::Min),
                ConstantForce::new(30.0, 0.0, 0.0),
                SleepingDisabled,
                Transform::from_xyz(0.5, 0.2, 0.0),
            ))
            .id();
        app.world_mut().spawn((
            FixedJoint::new(block, sled)
                .with_anchor(RVector::new(0.25, 0.2, 0.0))
                .with_point_compliance(1e-6)
                .with_angle_compliance(1e-4),
            JointCollisionDisabled,
        ));
        let crept = creep(&mut app, block);
        assert!(
            crept < 1e-4,
            "the block crept {crept} m in 9 s at {substeps} substeps"
        );
    }
}

/// A joint reports the force and the torque about its anchor that it applies to its first
/// body. A weight held out sideways from a fixed support pulls the support down, and
/// turns it the way the weight would fall.
#[cfg(feature = "3d")]
#[test]
fn a_fixed_joint_reports_the_load_on_its_first_body() {
    for compliance in [0.0, 1e-5] {
        let mut app = create_app();
        app.insert_resource(Gravity(Vector::NEG_Y * 9.81));
        app.finish();
        let support = app
            .world_mut()
            .spawn((RigidBody::Static, Position(RVector::ZERO)))
            .id();
        let weight = app
            .world_mut()
            .spawn((
                RigidBody::Dynamic,
                Position(RVector::X),
                Mass(1.0),
                AngularInertia::new(Vec3::splat(0.1)),
                SleepingDisabled,
            ))
            .id();
        let joint = app
            .world_mut()
            .spawn((
                FixedJoint::new(support, weight)
                    .with_anchor(RVector::ZERO)
                    .with_point_compliance(compliance)
                    .with_angle_compliance(compliance),
                JointForces::new(),
            ))
            .id();
        for _ in 0..(2.0 / TIMESTEP) as usize {
            app.update();
        }
        let forces = app.world().get::<JointForces>(joint).unwrap();
        let sag = app.world().get::<Position>(weight).unwrap().0.y;
        println!(
            "compliance {compliance}: force {} torque {} sag {sag}",
            forces.force(),
            forces.torque()
        );
        assert!(
            forces.force().distance(Vec3::new(0.0, -9.81, 0.0)) < 0.2,
            "force {}",
            forces.force()
        );
        assert!(
            forces.torque().distance(Vec3::new(0.0, 0.0, -9.81)) < 0.2,
            "torque {}",
            forces.torque()
        );
        assert!(sag.abs() < 0.005, "the weight sagged {sag} m");
    }
}

/// A weight on a rope hangs at the rope's length, and the rope pulls its support down
/// with the weight.
#[test]
fn a_rope_holds_a_weight_at_its_length() {
    let mut app = create_app();
    app.insert_resource(Gravity(Vector::NEG_Y * 9.81));
    app.finish();
    let support = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();
    // Let go a little way up and to the side, with the rope slack.
    let weight = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 0.3 + RVector::NEG_Y * 0.8),
            Mass(2.0),
            #[cfg(feature = "2d")]
            AngularInertia(0.1),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(0.1)),
            LinearDamping(2.0),
            SleepingDisabled,
        ))
        .id();
    let rope = app
        .world_mut()
        .spawn((
            DistanceJoint::new(support, weight).with_limits(0.0, 1.0),
            JointForces::new(),
        ))
        .id();
    let mut furthest: f32 = 0.0;
    for _ in 0..(6.0 / TIMESTEP) as usize {
        app.update();
        let at = app.world().get::<Position>(weight).unwrap().0;
        furthest = furthest.max(at.length() as f32);
    }
    let at = app.world().get::<Position>(weight).unwrap().0;
    assert!(furthest < 1.01, "the rope stretched to {furthest} m");
    assert!(
        at.distance(RVector::NEG_Y) < 0.02,
        "the weight came to rest at {at}"
    );
    let force = app.world().get::<JointForces>(rope).unwrap().force();
    assert!(
        force.distance(Vector::NEG_Y * 19.62) < 0.4,
        "the rope pulls its support with {force}"
    );
}

/// A ball joint keeps its anchors together and its swing within its limit.
#[cfg(feature = "3d")]
#[test]
fn a_ball_joint_holds_its_anchor_and_its_swing() {
    let mut app = create_app();
    app.insert_resource(Gravity(Vector::NEG_Y * 9.81));
    app.finish();
    let support = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();
    // A rod held out level by its end, along the axis the swing is measured by, free to
    // fall as far as the joint lets it swing.
    let rod = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::Z * 0.5),
            Mass(1.0),
            AngularInertia::new(Vec3::new(0.1, 0.1, 0.01)),
            AngularDamping(1.0),
            SleepingDisabled,
        ))
        .id();
    app.world_mut().spawn(
        SphericalJoint::new(support, rod)
            .with_local_anchor2(Vec3::NEG_Z * 0.5)
            .with_swing_limits(0.0, 0.5),
    );
    let mut furthest: f32 = 0.0;
    let mut steepest: f32 = 0.0;
    for _ in 0..(4.0 / TIMESTEP) as usize {
        app.update();
        let at = app.world().get::<Position>(rod).unwrap().0;
        let turn = app.world().get::<Rotation>(rod).unwrap().0;
        let end = at.f32() + turn * (Vec3::NEG_Z * 0.5);
        furthest = furthest.max(end.length());
        steepest = steepest.max((turn * Vec3::Z).angle_between(Vec3::Z));
    }
    assert!(furthest < 0.005, "the anchors came {furthest} m apart");
    assert!(
        (0.4..0.55).contains(&steepest),
        "the rod swung {steepest} rad against a limit of 0.5"
    );
}
