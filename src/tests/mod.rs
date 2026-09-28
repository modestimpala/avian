#[cfg(all(feature = "3d", feature = "default-collider"))]
use crate::math::Real;
use crate::prelude::*;
#[cfg(all(
    feature = "default-collider",
    any(feature = "parry-f32", feature = "parry-f64")
))]
use approx::assert_relative_eq;
use bevy::{
    ecs::schedule::{LogLevel, ScheduleBuildSettings, ScheduleLabel},
    prelude::*,
    time::TimeUpdateStrategy,
};
use core::time::Duration;

#[cfg(all(feature = "2d", feature = "enhanced-determinism"))]
mod determinism_2d;

fn create_app() -> App {
    let mut app = App::new();

    app.add_plugins((
        MinimalPlugins,
        TransformPlugin,
        PhysicsPlugins::default(),
        bevy::asset::AssetPlugin::default(),
        #[cfg(all(feature = "collider-from-mesh", feature = "default-collider"))]
        bevy::mesh::MeshPlugin,
        #[cfg(feature = "bevy_scene")]
        bevy::scene::ScenePlugin,
    ))
    .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
        1.0 / 60.0,
    )));

    app.finish();

    app
}

fn tick_app(app: &mut App, timestep: f64) {
    let strategy = TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(timestep));

    if let Some(mut update_strategy) = app.world_mut().get_resource_mut::<TimeUpdateStrategy>() {
        *update_strategy = strategy;
    } else {
        app.insert_resource(strategy);
    }

    app.update();
}

#[cfg(all(feature = "3d", feature = "default-collider"))]
fn setup_cubes_simulation(mut commands: Commands) {
    let mut next_id = 0;
    // copied from "cubes" example
    let floor_size = Vec3::new(80.0, 1.0, 80.0);
    commands.spawn((
        RigidBody::Static,
        Position(RVector::NEG_Y),
        Collider::cuboid(floor_size.x, floor_size.y, floor_size.z),
    ));

    let radius = 1.0;
    let count_x = 4;
    let count_y = 4;
    let count_z = 4;
    for y in 0..count_y {
        for x in 0..count_x {
            for z in 0..count_z {
                let pos = RVector::new(
                    (x as Real - count_x as Real * 0.5) * 2.1 * radius as Real,
                    10.0 * radius as Real * y as Real,
                    (z as Real - count_z as Real * 0.5) * 2.1 * radius as Real,
                );
                commands.spawn((
                    Transform::default(),
                    RigidBody::Dynamic,
                    Position(pos + RVector::Y * 5.0),
                    Collider::cuboid(radius * 2.0, radius * 2.0, radius * 2.0),
                    Id(next_id),
                ));
                next_id += 1;
            }
        }
    }
}

#[test]
fn it_loads_plugin_without_errors() -> Result<(), Box<dyn core::error::Error>> {
    let mut app = create_app();

    for _ in 0..500 {
        tick_app(&mut app, 1.0 / 60.0);
    }

    Ok(())
}

#[test]
#[cfg(all(
    feature = "default-collider",
    any(feature = "parry-f32", feature = "parry-f64")
))]
fn body_with_velocity_moves() {
    let mut app = create_app();

    app.insert_resource(Gravity::ZERO);

    app.add_systems(Startup, |mut commands: Commands| {
        // move right at 1 unit per second
        commands.spawn((
            Transform::default(),
            RigidBody::Dynamic,
            LinearVelocity(Vector::X),
            #[cfg(feature = "2d")]
            MassPropertiesBundle::from_shape(&Circle::new(0.5), 1.0),
            #[cfg(feature = "3d")]
            MassPropertiesBundle::from_shape(&Sphere::new(0.5), 1.0),
        ));
    });

    // Run startup systems
    app.update();

    const UPDATES: usize = 500;

    for _ in 0..UPDATES {
        tick_app(&mut app, 1.0 / 60.0);
    }

    let mut app_query = app.world_mut().query::<(&Transform, &RigidBody)>();

    let (transform, _body) = app_query.single(app.world()).unwrap();

    assert_relative_eq!(transform.translation.y, 0.);
    assert_relative_eq!(transform.translation.z, 0.);

    // make sure we end up in the expected position
    assert_relative_eq!(
        transform.translation.x,
        1. * UPDATES as f32 * 1. / 60.,
        epsilon = 0.03 // allow some leeway, as we might be one frame off
    );
}

#[derive(Component, Clone, Copy, Debug, PartialEq, PartialOrd, Eq, Ord)]
#[cfg(all(feature = "3d", feature = "default-collider"))]
struct Id(usize);

#[cfg(all(feature = "3d", feature = "default-collider"))]
#[test]
fn cubes_simulation_is_locally_deterministic() {
    use itertools::Itertools;

    fn run_cubes() -> Vec<(Id, Transform)> {
        let mut app = create_app();

        app.add_systems(Startup, setup_cubes_simulation);

        // Run startup systems
        app.update();

        const SECONDS: usize = 5;
        const UPDATES: usize = 60 * SECONDS;

        for _ in 0..UPDATES {
            tick_app(&mut app, 1.0 / 60.0);
        }

        let mut app_query = app.world_mut().query::<(&Id, &Transform)>();

        let mut bodies: Vec<(Id, Transform)> = app_query
            .iter(app.world())
            .map(|(id, transform)| (*id, *transform))
            .collect();
        bodies.sort_by_key(|b| b.0);
        bodies
    }

    // run simulation and check that results are equal each time
    for (a, b) in (0..4).map(|_| run_cubes()).tuple_windows() {
        assert_eq!(a, b);
    }
}

#[test]
fn no_ambiguity_errors() {
    #[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
    struct DeterministicSchedule;

    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        PhysicsPlugins::new(DeterministicSchedule)
            .build()
            .disable::<ColliderHierarchyPlugin>(),
        bevy::asset::AssetPlugin::default(),
        #[cfg(feature = "bevy_scene")]
        bevy::scene::ScenePlugin,
        #[cfg(all(feature = "collider-from-mesh", feature = "default-collider"))]
        bevy::mesh::MeshPlugin,
    ))
    .edit_schedule(DeterministicSchedule, |s| {
        s.set_build_settings(ScheduleBuildSettings {
            ambiguity_detection: LogLevel::Error,
            ..default()
        });
    })
    .add_systems(Update, |w: &mut World| {
        w.run_schedule(DeterministicSchedule);
    })
    .finish();
    app.update();
}

/// A body spawned already asleep has an island of its own, asleep too: without one, the
/// first contact made with it has no island to merge into, and a body landing on it wakes it.
#[test]
#[cfg(all(feature = "3d", feature = "default-collider"))]
fn a_body_spawned_asleep_has_an_island() {
    use crate::dynamics::solver::islands::{BodyIslandNode, PhysicsIslands};

    let mut app = create_app();
    app.world_mut().spawn((
        RigidBody::Static,
        Collider::cuboid(10.0, 1.0, 10.0),
        Transform::from_xyz(0.0, -0.5, 0.0),
    ));
    let resting = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Collider::cuboid(1.0, 1.0, 1.0),
            Transform::from_xyz(0.0, 0.5, 0.0),
            Sleeping,
        ))
        .id();
    app.update();
    let island = app
        .world()
        .get::<BodyIslandNode>(resting)
        .expect("a body spawned asleep is in an island")
        .island_id();
    assert!(
        app.world()
            .resource::<PhysicsIslands>()
            .get(island)
            .is_some_and(|island| island.is_sleeping()),
        "and its island is asleep"
    );
    let falling = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Collider::cuboid(1.0, 1.0, 1.0),
            Transform::from_xyz(0.0, 2.0, 0.0),
        ))
        .id();
    for _ in 0..120 {
        tick_app(&mut app, 1.0 / 60.0);
    }
    let landed = app.world().get::<Position>(falling).unwrap().y;
    assert!(
        landed > 1.2,
        "the falling body came to rest on the sleeping one, at {landed}"
    );
    assert!(app.world().get::<BodyIslandNode>(resting).is_some());
}

/// A heavy slab laid across light rods on the ground: whether it has come to rest, asleep,
/// after `seconds`, and how fast anything was still going over the last second.
#[cfg(all(feature = "3d", feature = "default-collider"))]
fn slab_on_rods(max_mass_ratio: f32, seconds: f32) -> (bool, f32) {
    let mut app = create_app();
    app.insert_resource(crate::dynamics::solver::SolverConfig {
        max_mass_ratio,
        ..default()
    });
    app.world_mut().spawn((
        RigidBody::Static,
        Collider::cuboid(10.0, 1.0, 10.0),
        Transform::from_xyz(0.0, -0.5, 0.0),
    ));
    let mut bodies = Vec::new();
    // Rods 3 cm through and 1.2 m long, near a kilogram each, crossing under the slab.
    for (i, x) in [-0.35, 0.0, 0.35].into_iter().enumerate() {
        let turn = Quat::from_rotation_y(0.2 * i as f32 - 0.2);
        bodies.push(
            app.world_mut()
                .spawn((
                    RigidBody::Dynamic,
                    Collider::cuboid(0.03, 0.03, 1.2),
                    Transform::from_xyz(x, 0.015, 0.0).with_rotation(turn),
                ))
                .id(),
        );
    }
    // A slab of 360 kg dropped a hand onto them.
    bodies.push(
        app.world_mut()
            .spawn((
                RigidBody::Dynamic,
                Collider::cuboid(1.0, 0.3, 1.2),
                Transform::from_xyz(0.02, 0.25, 0.0).with_rotation(Quat::from_rotation_z(0.05)),
            ))
            .id(),
    );
    let steps = (seconds * 60.0) as usize;
    let mut fastest = 0.0f32;
    for step in 0..steps {
        tick_app(&mut app, 1.0 / 60.0);
        if step + 60 >= steps {
            for &b in &bodies {
                let v = app.world().get::<LinearVelocity>(b).unwrap();
                let w = app.world().get::<AngularVelocity>(b).unwrap();
                #[allow(
                    clippy::unnecessary_cast,
                    reason = "velocities are f64 with that feature"
                )]
                let (v, w) = (v.length() as f32, w.length() as f32);
                fastest = fastest.max(v).max(w * 0.1);
            }
        }
    }
    let asleep = bodies
        .iter()
        .all(|&b| app.world().get::<Sleeping>(b).is_some());
    (asleep, fastest)
}

/// Weight passed down through a much lighter body: with the mass ratio limited, a slab over
/// light rods settles and sleeps.
#[test]
#[cfg(all(feature = "3d", feature = "default-collider"))]
fn a_heavy_slab_on_light_rods_settles_with_the_mass_ratio_limited() {
    let (asleep, fastest) = slab_on_rods(5.0, 8.0);
    assert!(asleep, "never slept; still going {fastest} m/s");
    if std::env::var("AVIAN_RATIO_REFERENCE").is_ok() {
        let unlimited = slab_on_rods(f32::INFINITY, 8.0);
        eprintln!("limited: {:?}, unlimited: {unlimited:?}", (asleep, fastest));
    }
}

/// A log a fifth of a metre through laid across a slope of `slope` radians, touching over
/// a patch of radius `patch`: how far down the slope it has gone after three seconds, m.
#[cfg(all(feature = "3d", feature = "default-collider"))]
fn log_on_a_slope(slope: f32, patch: f32) -> f32 {
    let mut app = create_app();
    let tilt = Quat::from_rotation_z(-slope);
    app.world_mut().spawn((
        RigidBody::Static,
        Collider::cuboid(40.0, 1.0, 10.0),
        Transform::from_translation(tilt * Vec3::new(0.0, -0.5, 0.0)).with_rotation(tilt),
        Friction::new(0.8),
    ));
    // Lying along z, so it rolls down x.
    let log = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Collider::cylinder(0.1, 1.0),
            Transform::from_translation(tilt * Vec3::new(0.0, 0.1, 0.0))
                .with_rotation(tilt * Quat::from_rotation_x(core::f32::consts::FRAC_PI_2)),
            Friction::new(0.8),
            ContactPatch(patch),
        ))
        .id();
    for _ in 0..180 {
        tick_app(&mut app, 1.0 / 60.0);
    }
    let at = app.world().get::<Transform>(log).unwrap().translation;
    (tilt.inverse() * at).x
}

/// A log lies still on a slope no steeper than its patch is wide for its girth, and rolls
/// down a steeper one, and down any slope on a point.
#[test]
#[cfg(all(feature = "3d", feature = "default-collider"))]
fn a_log_lies_on_a_slope_its_patch_can_hold() {
    // A patch of 15 mm on a log of radius 100 mm holds a slope of up to 0.15.
    let held = log_on_a_slope(0.08, 0.015);
    assert!(held.abs() < 0.002, "rolled {held} m with a patch");
    // Rolling, a log gathers speed at two thirds of what the slope gives it, less what
    // the patch holds: 2/3 g (sin 0.25 - 0.15 cos 0.25), 3.0 m in three seconds.
    let steep = log_on_a_slope(0.25, 0.015);
    assert!(
        (steep - 3.0).abs() < 0.2,
        "rolled {steep} m down a steep slope"
    );
    let point = log_on_a_slope(0.08, 0.0);
    assert!((point - 2.35).abs() < 0.2, "rolled {point} m on a point");
}

/// A ball set spinning on the spot, touching over a patch of radius `patch`: how fast it
/// still spins after two seconds, rad/s.
#[cfg(all(feature = "3d", feature = "default-collider"))]
fn spun_ball(patch: f32) -> f32 {
    let mut app = create_app();
    app.world_mut().spawn((
        RigidBody::Static,
        Collider::cuboid(10.0, 1.0, 10.0),
        Transform::from_xyz(0.0, -0.5, 0.0),
        Friction::new(0.5),
    ));
    let ball = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Collider::sphere(0.1),
            Transform::from_xyz(0.0, 0.1, 0.0),
            Friction::new(0.5),
            ContactPatch(patch),
            AngularVelocity(Vec3::Y * 10.0),
        ))
        .id();
    for _ in 0..120 {
        tick_app(&mut app, 1.0 / 60.0);
    }
    app.world().get::<AngularVelocity>(ball).unwrap().y
}

/// Twisting on the spot is stopped by the friction over the patch, as fast as that
/// friction can: and not at all on a point.
#[test]
#[cfg(all(feature = "3d", feature = "default-collider"))]
fn a_spun_ball_is_stopped_by_its_patch() {
    let point = spun_ball(0.0);
    assert!(point > 9.0, "spinning at {point} rad/s on a point");
    // A ball of radius r spun at w has 2/5 m r^2 w to lose, and a patch of radius a takes
    // 2/3 mu a m g of it a second: all of it in 3 r^2 w / (5 mu a g), here 0.61 s.
    let stopped = spun_ball(0.02);
    assert!(
        stopped.abs() < 0.01,
        "spinning at {stopped} rad/s on a patch"
    );
    // A patch a tenth as wide has taken a fifth of it in two seconds.
    let narrow = spun_ball(0.002);
    assert!(
        (narrow - 6.73).abs() < 0.3,
        "spinning at {narrow} rad/s on a narrow patch"
    );
}

/// A wheel on a slope of `slope` radians, touching over a patch of radius `patch`: how far
/// down the slope it has gone after three seconds, m.
#[cfg(all(feature = "2d", feature = "default-collider"))]
fn wheel_on_a_slope(slope: f32, patch: f32) -> f32 {
    let mut app = create_app();
    let tilt = Quat::from_rotation_z(-slope);
    app.world_mut().spawn((
        RigidBody::Static,
        Collider::rectangle(4000.0, 100.0),
        Transform::from_translation(tilt * Vec3::new(0.0, -50.0, 0.0)).with_rotation(tilt),
        Friction::new(0.8),
    ));
    let wheel = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Collider::circle(10.0),
            Transform::from_translation(tilt * Vec3::new(0.0, 10.0, 0.0)),
            Friction::new(0.8),
            ContactPatch(patch),
        ))
        .id();
    for _ in 0..180 {
        tick_app(&mut app, 1.0 / 60.0);
    }
    let at = app.world().get::<Transform>(wheel).unwrap().translation;
    (tilt.inverse() * at).x
}

/// A wheel stands on a slope no steeper than its patch is wide for its size, and rolls
/// down a steeper one, and down any slope on a point.
#[test]
#[cfg(all(feature = "2d", feature = "default-collider"))]
fn a_wheel_stands_on_a_slope_its_patch_can_hold() {
    // A patch of 1.5 on a wheel of radius 10 holds a slope of up to 0.15.
    let held = wheel_on_a_slope(0.08, 1.5);
    assert!(held.abs() < 0.02, "rolled {held} with a patch");
    // Rolling, a disc gathers speed at two thirds of what the slope gives it, less what
    // the patch holds: 2/3 g (sin 0.25 - 0.15 cos 0.25), 3.0 in three seconds.
    let steep = wheel_on_a_slope(0.25, 1.5);
    assert!(
        (steep - 3.0).abs() < 0.2,
        "rolled {steep} down a steep slope"
    );
    let point = wheel_on_a_slope(0.08, 0.0);
    assert!((point - 2.35).abs() < 0.2, "rolled {point} on a point");
}

/// A body at rest presses on what it rests on with its weight: the impulses its contacts
/// report over a step, divided by the step, come to it.
#[test]
#[cfg(all(feature = "3d", feature = "default-collider"))]
fn a_resting_body_presses_with_its_weight() {
    let mut app = create_app();
    app.world_mut().spawn((
        RigidBody::Static,
        Collider::cuboid(10.0, 1.0, 10.0),
        Transform::from_xyz(0.0, -0.5, 0.0),
    ));
    let block = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Collider::cuboid(1.0, 0.5, 1.0),
            Mass(40.0),
            Transform::from_xyz(0.0, 0.25, 0.0),
            SleepingDisabled,
        ))
        .id();
    for _ in 0..120 {
        tick_app(&mut app, 1.0 / 60.0);
    }
    let step = app
        .world()
        .resource::<Time<Fixed>>()
        .timestep()
        .as_secs_f32();
    let pressed: f32 = app
        .world()
        .resource::<ContactGraph>()
        .contact_pairs_with(block)
        .flat_map(|pair| &pair.manifolds)
        .map(|manifold| manifold.total_normal_impulse())
        .sum();
    let weight = 40.0 * 9.81;
    assert!(
        (pressed / step - weight).abs() < 0.02 * weight,
        "pressing with {} N of {weight} N",
        pressed / step
    );
}

/// A collider given to another body rests on the ground as that body's: the body it was
/// taken from falls through nothing, and nothing panics over contacts that name it.
#[test]
#[cfg(all(feature = "3d", feature = "default-collider"))]
fn a_collider_given_to_another_body_touches_as_that_bodys() {
    let mut app = create_app();
    app.world_mut().spawn((
        RigidBody::Static,
        Collider::cuboid(10.0, 1.0, 10.0),
        Transform::from_xyz(0.0, -0.5, 0.0),
    ));
    let first = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Mass(5.0),
            Transform::from_xyz(0.0, 0.25, 0.0),
            SleepingDisabled,
        ))
        .id();
    let collider = app
        .world_mut()
        .spawn((
            ChildOf(first),
            Collider::cuboid(1.0, 0.5, 1.0),
            Transform::IDENTITY,
        ))
        .id();
    for _ in 0..30 {
        tick_app(&mut app, 1.0 / 60.0);
    }
    assert!(app.world().get::<Transform>(first).unwrap().translation.y > 0.2);
    // Another body takes the collider, where it is.
    let second = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Mass(5.0),
            Transform::from_xyz(0.5, 0.25, 0.0),
            SleepingDisabled,
        ))
        .id();
    app.world_mut()
        .entity_mut(collider)
        .insert((ChildOf(second), Transform::from_xyz(-0.5, 0.0, 0.0)));
    app.world_mut().entity_mut(first).despawn();
    for _ in 0..60 {
        tick_app(&mut app, 1.0 / 60.0);
    }
    let rests = app.world().get::<Transform>(second).unwrap().translation;
    assert!(
        (rests.y - 0.25).abs() < 0.01,
        "the second body is at {rests}"
    );
    assert_eq!(
        app.world().get::<ColliderOf>(collider).unwrap().body,
        second
    );
}

/// Level ground made of triangles, 0.5 m to a cell, and a plank lying on it.
#[cfg(all(feature = "3d", feature = "default-collider"))]
fn plank_on_ground(heights: impl Fn(f32) -> f32, reduce: bool) -> (App, Entity) {
    let mut app = create_app();
    if !reduce {
        app.world_mut()
            .resource_mut::<NarrowPhaseConfig>()
            .manifold_reduction_angle = 0.0;
    }
    // Rows run along x and columns along z: the ground's height varies across the plank.
    let ground: Vec<Vec<Real>> = (0..=20)
        .map(|_| (0..=20).map(|z| heights((z - 10) as f32 * 0.5)).collect())
        .collect();
    app.world_mut().spawn((
        RigidBody::Static,
        Collider::heightfield(ground, Vec3::new(10.0, 1.0, 10.0)),
        Transform::default(),
    ));
    let plank = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Collider::cuboid(5.4, 0.05, 0.26),
            Mass(30.0),
            Transform::from_xyz(0.13, 0.025 + heights(0.0), 0.0),
            SleepingDisabled,
        ))
        .id();
    for _ in 0..120 {
        tick_app(&mut app, 1.0 / 60.0);
    }
    (app, plank)
}

/// How many manifolds and points a body touches with, and what it presses with, N.
#[cfg(all(feature = "3d", feature = "default-collider"))]
fn touching(app: &App, body: Entity) -> (usize, usize, f32) {
    let step = app
        .world()
        .resource::<Time<Fixed>>()
        .timestep()
        .as_secs_f32();
    let manifolds: Vec<&ContactManifold> = app
        .world()
        .resource::<ContactGraph>()
        .contact_pairs_with(body)
        .flat_map(|pair| &pair.manifolds)
        .collect();
    (
        manifolds.len(),
        manifolds.iter().map(|manifold| manifold.points.len()).sum(),
        manifolds
            .iter()
            .map(|manifold| manifold.total_normal_impulse())
            .sum::<f32>()
            / step,
    )
}

/// A plank on level ground made of triangles lies on every triangle under it the same
/// way. It touches the ground with one manifold of four points, not one for each triangle,
/// and lies as still and presses as hard as it did with them all.
#[test]
#[cfg(all(feature = "3d", feature = "default-collider"))]
fn a_plank_on_level_ground_of_triangles_touches_it_once() {
    let weight = 30.0 * 9.81;
    let (app, plank) = plank_on_ground(|_| 0.0, false);
    let (manifolds, _, pressed) = touching(&app, plank);
    assert!(manifolds > 10, "{manifolds} manifolds with none made one");
    assert!((pressed - weight).abs() < 0.02 * weight, "{pressed} N");

    let (app, plank) = plank_on_ground(|_| 0.0, true);
    let (manifolds, points, pressed) = touching(&app, plank);
    assert_eq!((manifolds, points), (1, 4));
    assert!((pressed - weight).abs() < 0.02 * weight, "{pressed} N");
    let lies = app.world().get::<Transform>(plank).unwrap();
    assert!(
        (lies.translation - Vec3::new(0.13, 0.025, 0.0)).length() < 0.002,
        "{}",
        lies.translation
    );
    assert!(lies.rotation.angle_between(Quat::IDENTITY) < 0.002);
    let moving = app.world().get::<LinearVelocity>(plank).unwrap();
    assert!(moving.length() < 0.002, "moving at {}", moving.0);
}

/// A plank in a trough lies on both its sides, which face different ways: the manifolds
/// of one side are made one, and those of the two sides are not.
#[test]
#[cfg(all(feature = "3d", feature = "default-collider"))]
fn a_plank_in_a_trough_touches_each_side_of_it() {
    let (app, plank) = plank_on_ground(|across| across.abs() * 0.4, true);
    let (manifolds, points, pressed) = touching(&app, plank);
    let faces: Vec<(Vec3, Vec<(Vec3, f32)>)> = app
        .world()
        .resource::<ContactGraph>()
        .contact_pairs_with(plank)
        .flat_map(|pair| &pair.manifolds)
        .map(|manifold| {
            (
                manifold.normal,
                manifold
                    .points
                    .iter()
                    .map(|point| (point.anchor1, point.penetration))
                    .collect(),
            )
        })
        .collect();
    // One manifold of four points for each side, facing as the side does; and no more
    // than the plank's ends add, which bear nothing.
    let sides: Vec<f32> = faces
        .iter()
        .filter(|(_, points)| points.iter().any(|(_, penetration)| *penetration > -0.005))
        .map(|(normal, _)| normal.z)
        .collect();
    assert_eq!(sides.len(), 2, "{points} points: {faces:?}");
    assert!(sides[0] * sides[1] < -0.1, "{faces:?}");
    assert!(manifolds <= 4 && points <= 12, "{points} points: {faces:?}");
    // The sides bear the weight between them, by pressing and by friction.
    let weight = 30.0 * 9.81;
    assert!(
        pressed > 0.8 * weight && pressed < 1.1 * weight,
        "{pressed} N"
    );
    let moving = app.world().get::<LinearVelocity>(plank).unwrap();
    assert!(moving.length() < 0.005, "moving at {}", moving.0);
}

/// Manifolds that lie in one plane are made one, of the four points that span the most.
/// Manifolds that face the same way from another level, or another way, are left.
#[test]
#[cfg(feature = "3d")]
fn manifolds_in_one_plane_are_made_one() {
    let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
    let at = |(x, z): (f32, f32), reach: f32, level: f32| {
        let on_first = Vec3::new(x * reach, level, z * reach);
        let on_second = on_first - Vec3::Y * 0.001;
        ContactPoint::new(on_first, on_second, on_first.into(), 0.001)
    };
    let mut manifolds = vec![
        ContactManifold::new(corners.map(|corner| at(corner, 0.5, 0.0)), Vec3::Y),
        ContactManifold::new(corners.map(|corner| at(corner, 3.0, 0.001)), Vec3::Y),
        ContactManifold::new(
            corners.map(|corner| at(corner, 1.5, -0.001)),
            Vec3::new(0.0003, 1.0, 0.0).normalize(),
        ),
        ContactManifold::new(corners.map(|corner| at(corner, 4.0, 0.05)), Vec3::Y),
        ContactManifold::new([at((0.0, 0.0), 1.0, 0.0)], Vec3::X),
    ];
    let made = ContactManifold::reduce(&mut manifolds, 0.05f32.cos(), 0.002);
    assert_eq!(made, vec![0]);
    assert_eq!(manifolds.len(), 3);
    assert_eq!(manifolds[0].points.len(), 4);
    for point in &manifolds[0].points {
        assert!(
            point.anchor1.x.abs() == 3.0 && point.anchor1.z.abs() == 3.0,
            "kept a point at {}",
            point.anchor1
        );
    }
    let levels: Vec<f32> = manifolds[1..]
        .iter()
        .map(|manifold| manifold.points[0].anchor1.y)
        .collect();
    assert!(
        levels.contains(&0.05) && levels.contains(&0.0),
        "{levels:?}"
    );
}

/// What is made of several manifolds is warm started from the points that were nearest.
#[test]
#[cfg(feature = "3d")]
fn points_with_no_features_are_matched_by_the_nearest() {
    let at = |x: f32, impulse: f32| {
        let mut point = ContactPoint::new(
            Vec3::new(x, 0.0, 0.0),
            Vec3::new(x, -1.0, 0.0),
            Vec3::new(x, 0.0, 0.0).into(),
            0.0,
        );
        point.warm_start_normal_impulse = impulse;
        point
    };
    let before = [
        ContactManifold::new([at(0.0, 1.0), at(0.06, 2.0), at(0.5, 3.0)], Vec3::Y),
        ContactManifold::new([at(0.052, 9.0)], Vec3::X),
    ];
    let mut now = ContactManifold::new([at(0.05, 0.0), at(0.49, 0.0), at(0.9, 0.0)], Vec3::Y);
    now.match_contacts_by_place(&before, 0.1);
    let impulses: Vec<f32> = now
        .points
        .iter()
        .map(|point| point.warm_start_normal_impulse)
        .collect();
    assert_eq!(impulses, vec![2.0, 3.0, 0.0]);
}

/// A stone of several boxes on hummocked ground made of triangles, and how fast it still
/// turns after some seconds, rad/s.
#[cfg(all(feature = "3d", feature = "default-collider"))]
fn stone_on_hummocks(reduce: bool, seed: f32) -> (f32, f32, usize) {
    let mut app = create_app();
    app.insert_resource(SubstepCount(12));
    if !reduce {
        app.world_mut()
            .resource_mut::<NarrowPhaseConfig>()
            .manifold_reduction_angle = 0.0;
    }
    let ground: Vec<Vec<Real>> = (0..=20)
        .map(|x| {
            (0..=20)
                .map(|z| {
                    let (x, z) = (x as f32 * 0.5 + seed, z as f32 * 0.5 - seed);
                    0.08 * (x * 1.9).sin() * (z * 2.3).cos() + 0.04 * (x * 4.1 + z * 3.3).sin()
                })
                .collect()
        })
        .collect();
    app.world_mut().spawn((
        RigidBody::Static,
        Collider::heightfield(ground, Vec3::new(10.0, 1.0, 10.0)),
        Transform::default(),
    ));
    // A stone as a sculpted one is collided: boxes of voxels, laid in courses.
    let mut boxes = Vec::new();
    for (course, (across, along)) in [(0.30, 0.22), (0.36, 0.28), (0.34, 0.24), (0.24, 0.16)]
        .into_iter()
        .enumerate()
    {
        boxes.push((
            Position::from(Vec3::new(
                0.01 * course as f32,
                0.05 * course as f32,
                -0.01 * course as f32,
            )),
            Rotation::default(),
            Collider::cuboid(across, 0.05, along),
        ));
    }
    let stone = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Collider::compound(boxes),
            ColliderDensity(2600.0),
            Friction::new(0.7),
            Transform::from_xyz(0.2 + seed, 0.4, -0.3).with_rotation(Quat::from_euler(
                EulerRot::YXZ,
                0.4,
                0.1,
                -0.2,
            )),
        ))
        .id();
    let (mut turning, mut moving): (f32, f32) = (0.0, 0.0);
    for step in 0..480 {
        tick_app(&mut app, 1.0 / 60.0);
        if step >= 360 {
            if app.world().get::<Sleeping>(stone).is_some() {
                continue;
            }
            turning = turning.max(app.world().get::<AngularVelocity>(stone).unwrap().length());
            moving = moving.max(app.world().get::<LinearVelocity>(stone).unwrap().length());
        }
    }
    let manifolds = app
        .world()
        .resource::<ContactGraph>()
        .contact_pairs_with(stone)
        .flat_map(|pair| &pair.manifolds)
        .count();
    (turning, moving, manifolds)
}

/// A stone on hummocked ground comes to rest, and sleeps.
#[test]
#[cfg(all(feature = "3d", feature = "default-collider"))]
fn a_stone_on_hummocks_comes_to_rest() {
    for seed in [0.0, 1.3, 3.6] {
        let (turning, moving, manifolds) = stone_on_hummocks(true, seed);
        assert!(manifolds > 0, "seed {seed}: the stone touches nothing");
        assert!(
            turning == 0.0 && moving == 0.0,
            "seed {seed}: turning at {turning} rad/s and moving at {moving} m/s"
        );
    }
}
