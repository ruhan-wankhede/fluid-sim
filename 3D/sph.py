import warp as wp
import math

HEIGHT = 38.0
WIDTH = HEIGHT * 1.6
LENGTH = 25.0
NUM_PARTICLES = 600000

FLUID_WIDTH = WIDTH
FLUID_HEIGHT = HEIGHT
FLUID_DEPTH = LENGTH

# lague spawning
SPAWN_CENTRE = ((WIDTH + 20.0) / 4, HEIGHT / 3, LENGTH / 2)
SPAWN_SIZE = (22.0, 26.0, 20.0)
SPAWN_JITTER = 0.05

NEIGHBOR_RATIO = 2.0
FLUID_WIDTH, FLUID_HEIGHT, FLUID_DEPTH = (16.0, 16.0, 20.0)

PARTICLE_SPACING = (FLUID_WIDTH * FLUID_HEIGHT * FLUID_DEPTH / NUM_PARTICLES) ** (1.0/3.0)
SUPPORT_RADIUS = NEIGHBOR_RATIO * PARTICLE_SPACING
PARTICLE_RADIUS = 0.33 * PARTICLE_SPACING * 3
REST_DENSITY = 0.5 / PARTICLE_SPACING**3

GRAVITY = 9.81
MAX_COMPRESSION = 0.1
PRESSURE_K = GRAVITY * FLUID_HEIGHT / MAX_COMPRESSION
NEAR_PRESSURE_K = 0.02 * PRESSURE_K



@wp.kernel
def write_speed(particle_v: wp.array[wp.vec3], speed_scale: float, out: wp.array[float]):
    i = wp.tid()
    out[i] = wp.length(particle_v[i]) / speed_scale

@wp.func
def density_kernel(h: float, r: float, k_w: float):
    return k_w * (h - r) * (h - r)

@wp.func
def density_derivative(h: float, r: float, k_dw: float):
    return k_dw * (r - h)

@wp.func
def near_density(h: float, r: float, k_n: float):
    return k_n * (h - r) * (h - r) * (h - r)

@wp.func
def near_density_derivative(h: float, r: float, k_dn: float):
    return k_dn * (h - r) * (h - r) * -1.0


@wp.kernel
def compute_density(grid: wp.uint64, particle_x: wp.array[wp.vec3], particle_rho: wp.array[float], particle_near_rho: wp.array[float], h: float, k_w: float, k_n: float):
    tid = wp.tid()

    i = wp.hash_grid_point_id(grid, tid)
    x = particle_x[i]

    rho = float(0.0)
    near_rho = float(0.0)

    neighbours = wp.hash_grid_query(grid, x, h)

    for index in neighbours:
        distance = particle_x[index] - x
        r2 = wp.dot(distance, distance)
        if r2 >= h * h:
            continue
        r = wp.sqrt(r2)

        rho += density_kernel(h, r, k_w)
        near_rho += near_density(h, r, k_n)

    particle_rho[i] = rho
    particle_near_rho[i] = near_rho


@wp.kernel
def get_acceleration(grid: wp.uint64, particle_x: wp.array[wp.vec3], particle_v: wp.array[wp.vec3], particle_rho: wp.array[float], particle_near_rho: wp.array[float], particle_a: wp.array[wp.vec3], h: float, rho0: float, pressure_k: float, near_pressure_k: float, k_dw: float, k_dn: float, gravity: wp.vec3):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    x = particle_x[i]
    v = particle_v[i]
    rho = particle_rho[i]
    near_rho = particle_near_rho[i]
    pressure = pressure_k * (rho - rho0)
    near_pressure = near_pressure_k * near_rho

    h2 = h * h

    neighbours = wp.hash_grid_query(grid, x, h)

    pressure_force = wp.vec3(0.0, 0.0, 0.0)

    for index in neighbours:
        if index == i:
            continue
        r_vec = particle_x[index] - x
        r2 = wp.dot(r_vec, r_vec)
        if r2 > h2:
            continue

        r = wp.sqrt(r2)
        direction = wp.vec3(0.0, 1.0, 0.0)
        if r > 1.0e-6:
            direction = r_vec / r

        neighbour_rho = particle_rho[index]
        neighbour_near_rho = wp.max(particle_near_rho[index], 1.0e-6)
        neighbour_pressure = pressure_k * (neighbour_rho - rho0)
        neighbour_near_pressure = near_pressure_k * particle_near_rho[index]

        shared = (pressure + neighbour_pressure) / 2.0
        shared_near = (near_pressure + neighbour_near_pressure) / 2.0

        pressure_force += direction * density_derivative(h, r, k_dw) * shared / neighbour_rho
        pressure_force += direction * near_density_derivative(h, r, k_dn) * shared_near / neighbour_near_rho

    particle_a[i] = pressure_force / rho + gravity

@wp.kernel
def kick(particle_v: wp.array[wp.vec3], particle_a: wp.array[wp.vec3], dt: float):
    tid = wp.tid()
    v = particle_v[tid]
    particle_v[tid] = v + particle_a[tid] * dt


@wp.kernel
def drift(particle_x: wp.array[wp.vec3], particle_v: wp.array[wp.vec3], dt: float):
    tid = wp.tid()
    x = particle_x[tid]
    particle_x[tid] = x + particle_v[tid] * dt

@wp.kernel
def apply_bounds(
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    damping_coef: float,
    width: float,
    height: float,
    length: float,
    radius: float
):
    tid = wp.tid()

    # get pos and velocity
    x = particle_x[tid]
    v = particle_v[tid]
    # clamp x left
    if x[0] - radius < 0.0:
        x = wp.vec3(radius, x[1], x[2])
        v = wp.vec3(-1.0 * v[0] * damping_coef, v[1], v[2])

    # clamp x right
    if x[0] + radius > width:
        x = wp.vec3(width - radius, x[1], x[2])
        v = wp.vec3(-1.0 * v[0] * damping_coef, v[1], v[2])

    # clamp y bot
    if x[1] - radius < 0.0:
        x = wp.vec3(x[0], radius, x[2])
        v = wp.vec3(v[0], -1.0 * v[1] * damping_coef, v[2])

    if x[1] > height - radius:
        x = wp.vec3(x[0], height - radius, x[2])
        v = wp.vec3(v[0], -1.0 * v[1] * damping_coef, v[2])

    if x[2] - radius < 0.0:
        x = wp.vec3(x[0], x[1], radius)
        v = wp.vec3(v[0], v[1], -1.0 * v[2] * damping_coef)

    if x[2] > length - radius:
        x = wp.vec3(x[0], x[1], length - radius)
        v = wp.vec3(v[0], v[1], -1.0 * v[2] * damping_coef)

    # apply clamps
    particle_x[tid] = x
    particle_v[tid] = v

@wp.kernel
def spawn_random(x: wp.array[wp.vec3], v: wp.array[wp.vec3], width: float, height: float, seed: wp.int32, radius: float):
    i = wp.tid()
    state = wp.rand_init(seed, i)
    x[i] = wp.vec3(wp.randf(state) * (width - radius), wp.randf(state) * (height - radius), 0.0)
    v[i] = wp.vec3(0.0, 0.0, 0.0)

@wp.kernel
def spawn_grid(x: wp.array[wp.vec3], v: wp.array[wp.vec3], centre: wp.vec3, size: wp.vec3, nx: int, ny: int, nz: int, jitter: float, seed: int):
    # lague grid spawning do not ask me to explain this
    # y is the slowest index so a partially filled last layer ends up on top

    i = wp.tid()
    tx = float(i % nx) / float(wp.max(nx - 1, 1))
    tz = float((i // nx) % nz) / float(wp.max(nz - 1, 1))
    ty = float(i // (nx * nz)) / float(wp.max(ny - 1, 1))
    state = wp.rand_init(seed, i)
    offset = wp.sample_unit_sphere(state) * jitter
    x[i] = centre + wp.cw_mul(wp.vec3(tx, ty, tz) - wp.vec3(0.5, 0.5, 0.5), size) + offset
    v[i] = wp.vec3(0.0, 0.0, 0.0)

class Solver:
    def __init__(self, n=NUM_PARTICLES, width=WIDTH, height=HEIGHT, length=LENGTH, seed=67, gravity=(0.0, -9.81, 0.0), radius=PARTICLE_RADIUS, support_radius=SUPPORT_RADIUS):
        self.n = n
        self.width = width
        self.height = height
        self.length = length
        self.gravity = wp.vec3(*gravity)
        self.radius = radius
        self.support_radius = support_radius

        self.rest_density = REST_DENSITY
        self.pressure_k = PRESSURE_K
        self.near_pressure_k = NEAR_PRESSURE_K
        h = support_radius
        self.k_w = 15.0 / (2 * wp.PI * h ** 5)
        self.k_n = 15.0 / (wp.PI * h ** 6)
        self.k_dw = 2.0 * self.k_w
        self.k_dn = 3.0 * self.k_n

        self.x = wp.zeros(n, dtype=wp.vec3)
        self.v = wp.zeros(n, dtype=wp.vec3)
        self.a = wp.zeros(n, dtype=wp.vec3)

        self.rho = wp.zeros(n, dtype=float)
        self.near_rho = wp.zeros(n, dtype=float)

        w, h, d = SPAWN_SIZE
        spawn_spacing = (w * h * d / n) ** (1.0 / 3.0)
        self.nx, self.ny, self.nz = (math.ceil(w / spawn_spacing), math.ceil(h / spawn_spacing), math.ceil(d / spawn_spacing))

        self.grid = wp.HashGrid(math.ceil(width / support_radius), math.ceil(height / support_radius), math.ceil(length / support_radius),)


        wp.launch(spawn_grid, dim=n,
                  inputs=[self.x, self.v, wp.vec3(*SPAWN_CENTRE), wp.vec3(*SPAWN_SIZE), self.nx, self.ny, self.nz,
                          SPAWN_JITTER, seed])

        #wp.launch(spawn_random, dim=n, inputs=[self.x, self.v, self.width, self.height, seed, self.radius])

    def step(self, dt: float):
        x_star = self.x + self.v * 1.0 / 120.0
        self.grid.build(x_star, self.support_radius)

        wp.launch(compute_density, dim=self.n, inputs=[
            self.grid.id, x_star, self.rho, self.near_rho,
            self.support_radius, self.k_w, self.k_n,
        ])

        wp.launch(get_acceleration, dim=self.n, inputs=[
            self.grid.id, x_star, self.v, self.rho, self.near_rho, self.a,
            self.support_radius, self.rest_density, self.pressure_k, self.near_pressure_k,
            self.k_dw, self.k_dn, self.gravity,
        ])

        # kick
        wp.launch(kernel=kick, dim=self.n, inputs=[self.v, self.a, dt])

        # drift
        wp.launch(kernel=drift, dim=self.n, inputs=[self.x, self.v, dt])

        #bounds
        wp.launch(apply_bounds, dim=self.n, inputs=[self.x, self.v, 0.2, self.width, self.height, self.length, self.radius])

def log(solver):
    x = solver.x.numpy()
    return {
        "n": solver.n,
        "x_range": (float(x[:, 0].min()), float(x[:, 0].max())),
        "y_range": (float(x[:, 1].min()), float(x[:, 1].max())),
        "z_max_abs": float(abs(x[:, 2]).max()),
    }