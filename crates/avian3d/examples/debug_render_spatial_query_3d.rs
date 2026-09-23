use avian3d::{math::RVector, prelude::*};
use bevy::prelude::*;

fn main() {
    App::default()
        .add_plugins((
            DefaultPlugins,
            PhysicsPlugins::default(),
            PhysicsDebugPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, cast)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(10.0, 10.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(8.0, 1.0, 1.0))),
        MeshMaterial3d(materials.add(Color::WHITE)),
        RigidBody::Kinematic,
        Collider::cuboid(8.0, 1.0, 1.0),
        Transform::from_xyz(0.0, 0.0, -6.0),
        AngularVelocity(Vec3::new(0.0, 0.5, 0.0)),
    ));
}

fn cast(space: SpatialQuery) {
    let filter = SpatialQueryFilter::default();
    let shape = Collider::sphere(0.5);

    space.cast_ray(
        RVector::new(-2.0, 0.0, 4.0),
        Dir3::NEG_Z,
        100.0,
        false,
        &filter,
    );
    space.cast_shape(
        &shape,
        RVector::new(0.0, 0.0, 4.0),
        Quat::IDENTITY,
        Dir3::NEG_Z,
        &Default::default(),
        &filter,
    );
    space.project_point(RVector::new(2.0, 0.0, 4.0), false, &filter);
    space.shape_intersections(
        &shape,
        RVector::new(-3.0, 0.0, -5.0),
        Quat::IDENTITY,
        &filter,
    );
}
