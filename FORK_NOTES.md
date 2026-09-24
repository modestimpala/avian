# Fork notes

Known problems and local changes in this fork (the `orbits` branch), found while building
the games in the workspace. Kept out of `README.md` so upstream merges stay clean.

## Open: cylinders jitter against flat faces, and bodies built from them never settle

**Symptom.** A dynamic body whose colliders are `Collider::cylinder` does not come to rest on
flat ground (a `Collider::cuboid`). It keeps rocking and rolling, and two such bodies meeting
end to end shove at each other indefinitely, even with a gap between them. The lighter body
wanders. `AngularDamping` slows it but does not stop it. It looks like something is pushing
the body, but nothing is: the contacts themselves inject energy.

**Measured** (shack, `world/timber/pieces.rs`, test `settling`): a pine log with 3 m round
slabs, each slab one `Collider::cylinder` in a compound of about 30, laid on a static cuboid
floor, 16 ms steps.

| Log's colliders                                  | After 8 s: speed | spin       |
|--------------------------------------------------|------------------|------------|
| cylinders, trunk only                            | 0.05 m/s         | 0.46 rad/s |
| cylinders, bucked in two, both left in place     | 1.5–2 m/s (butt) | 1.5 rad/s  |
| the same shapes as 12-sided `convex_hull` prisms | 0.000002 m/s     | 0.00002    |
| prisms, bucked in two, both left in place        | 0.00008 m/s      | 0.0005     |

The only contact between the two bucked pieces was cylinder end cap to cylinder end cap,
reported at about -0.015 m penetration, which means the caps weren't even touching.

**Likely cause.** Cylinder–cuboid and cylinder–cylinder contacts go through the general
GJK/EPA path, and their manifolds for flat cap faces and the curved side are poorly
conditioned (points jumping between frames). A body that is light and round sits on that
jitter and turns it into motion. Convex polyhedra get proper clipped face manifolds.

**Workaround used in shack.** A 12-sided prism (`Collider::convex_hull` of two rings)
instead of `Collider::cylinder`. It rests on a facet and still rolls when pushed. See
`shack/src/world/timber/trunk.rs` (`rolled`).

**To look into.** Reproduce with a single cylinder lying on a cuboid in an Avian test, then
look at how the cylinder contact manifolds are generated and whether caps get a stable
face manifold, or whether cylinders should go through a polyhedral approximation.

## Related: bodies that start buried keep moving

A dynamic body whose colliders start deep inside static geometry (for example a felled tree
whose thin branch colliders were driven into the ground) never settles either: depenetration
keeps pushing it. Dropped from a height onto the same ground, the same body settles to
rest. This is probably expected solver behaviour rather than a bug, but it is worth knowing
when bodies are spawned or reshaped in contact.

## Fixed: a zero motor torque or force limit meant "unlimited"

`AngularMotor::max_torque` and `LinearMotor::max_force` were only enforced when positive, so
a limit of `0.0` left the motor unclamped. A brake whose strength had run out, lowered to
zero, instead held its joint rigid. Fixed in the XPBD revolute and prismatic joint solvers:
zero now means no torque or force, and a negative limit is treated as zero. Test:
`revolute_motor_with_zero_max_torque_does_nothing`.

## Fixed: a body spawned already asleep crashed the first contact made with it

**Symptom.** Spawning a dynamic body with `Sleeping` in its bundle panicked with "Neither
body … nor … is in an island" as soon as anything touched it (shack's bench tools, spawned
asleep, crashed at startup). Once the panic was avoided, a body landing on it passed
straight through, and the sleeping body never woke.

**Cause.** Islands are joined through `BodyIslandNode`, which is required by
`SolverBodyIndex`, and a body only gets a solver body when it is awake. A body inserted
already asleep therefore had no island, so merging islands for its first contact found
neither body. And `WakeIslands` walked an island's bodies with a query requiring
`SleepTimer` (also required by `SolverBodyIndex`), so it stopped at such a body without
waking it or anything after it.

**Fix.** `IslandPlugin` gives a dynamic or kinematic body inserted with `Sleeping` an island
of its own, asleep. `WakeIslands` treats `SleepTimer` as optional. Regression test:
`tests::a_body_spawned_asleep_has_an_island`.

