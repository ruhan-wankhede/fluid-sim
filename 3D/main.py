

import math
from pathlib import Path
from typing import Any

import moderngl
import moderngl_window as mglw
import warp as wp
from moderngl_window.context.base import KeyModifiers
from moderngl_window.scene.camera import KeyboardCamera
from moderngl_window import geometry

import sph


SIM_DT = 1.0 / 60.0
MAX_FRAME_TIME = 0.25

TIME_SCALE = 1.0

CFL = 0.4
SUBSTEPS = max(1, math.ceil(SIM_DT * math.sqrt(sph.PRESSURE_K) / (CFL * sph.SUPPORT_RADIUS)))

SPEED_SCALE = 10.0

SHADER_DIR = Path(__file__).parent / "shaders"

def read_shader(name: str) -> str:
    path = SHADER_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")

    raise SystemExit("path does not exist")


class Viewer(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "3D SPH"
    WIDTH = 2560
    HEIGHT = 1600
    WIDTH *= 0.96
    HEIGHT *= 0.96
    window_size = (WIDTH, HEIGHT)
    resizable = True
    vsync = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.blur_fbo_b = None
        self.blur_tex_b = None
        self.blur_fbo_a = None
        self.blur_tex_a = None
        self.depth_fbo = None
        self.depth_tex = None
        wp.init()
        self.solver = sph.Solver()
        self.paused = True
        print(sph.log(self.solver))
        print("substeps per frame:", SUBSTEPS)
        print("GL renderer:", self.ctx.info["GL_RENDERER"])

        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE | moderngl.DEPTH_TEST)
        self.prog = self.ctx.program(
            vertex_shader=read_shader("points.vert"),
            fragment_shader=read_shader("points.frag"),
        )

        self.quad = geometry.quad_fs()
        self.blur_prog = self.ctx.program(
            vertex_shader=read_shader("fullscreen.vert"),
            fragment_shader=read_shader("blur.frag"),
        )
        self.display_prog = self.ctx.program(
            vertex_shader=read_shader("fullscreen.vert"),
            fragment_shader=read_shader("display.frag"),
        )
        self.display_prog["u_depth"].value = 0
        self.ctx.gc_mode = "auto"
        self.create_targets()

        n = self.solver.n
        self.vbo_pos = self.ctx.buffer(reserve=n * 3 * 4, dynamic=True)
        self.vbo_scalar = self.ctx.buffer(reserve=n * 4, dynamic=True)
        attributes = [(self.vbo_pos, "3f", "in_pos")]
        if "in_scalar" in self.prog:
            attributes.append((self.vbo_scalar, "1f", "in_scalar"))
        self.vao = self.ctx.vertex_array(self.prog, attributes)

        self.sky_prog = self.ctx.program(vertex_shader=read_shader("skybox.vert"), fragment_shader=read_shader("skybox.frag"))
        self.sky_cube = geometry.cube(size=(2.0, 2.0, 2.0), normals=False, uvs=False)

        SKY_DIR = Path(__file__).parent / "skybox"

        self.sky_tex = self.load_texture_cube(
            pos_x=str(SKY_DIR / "skybox_posx.png"), neg_x=str(SKY_DIR / "skybox_negx.png"),
            pos_y=str(SKY_DIR / "skybox_posy.png"), neg_y=str(SKY_DIR / "skybox_negy.png"),
            pos_z=str(SKY_DIR / "skybox_posz.png"), neg_z=str(SKY_DIR / "skybox_negz.png"),
        )

        self.gl_pos = wp.RegisteredGLBuffer(self.vbo_pos.glo, flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
                                            fallback_to_copy=False)
        self.gl_scalar = wp.RegisteredGLBuffer(self.vbo_scalar.glo, flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
                                               fallback_to_copy=False)

        self.accumulator = 0.0
        self.fps_frames = 0
        self.fps_steps = 0
        self.fps_elapsed = 0.0

        self.camera = KeyboardCamera(self.wnd.keys, fov=60.0, aspect_ratio=self.wnd.aspect_ratio, near=0.1, far=500.0)
        self.camera.set_position(sph.WIDTH / 2, sph.HEIGHT / 2, sph.LENGTH + 40.0)
        self.camera.velocity = 20.0
        self.camera.mouse_sensitivity = -0.1

    def create_targets(self) -> None:
        size = self.wnd.buffer_size
        self.depth_tex = self.ctx.texture(size, 1, dtype="f4")
        self.depth_fbo = self.ctx.framebuffer(
            color_attachments=[self.depth_tex],
            depth_attachment=self.ctx.depth_renderbuffer(size),
        )
        self.blur_tex_a = self.ctx.texture(size, 1, dtype="f4")
        self.blur_fbo_a = self.ctx.framebuffer(color_attachments=[self.blur_tex_a])
        self.blur_tex_b = self.ctx.texture(size, 1, dtype="f4")
        self.blur_fbo_b = self.ctx.framebuffer(color_attachments=[self.blur_tex_b])

    def on_mouse_drag_event(self, x: int, y: int, dx: int, dy: int) -> None:
        self.camera.rot_state(dx, dy)

    def on_mouse_scroll_event(self, x_offset: float, y_offset: float) -> None:
        self.camera.velocity = max(1.0, self.camera.velocity * 1.2 ** y_offset)

    def on_resize(self, width: int, height: int) -> None:
        self.camera.projection.update(aspect_ratio=self.wnd.aspect_ratio)
        self.create_targets()

    def on_key_event(self, key: Any, action: Any, modifiers: KeyModifiers) -> None:
        keys = self.wnd.keys
        self.camera.key_input(key, action, modifiers)

        if action != keys.ACTION_PRESS:
            return
        if key == keys.SPACE:
            self.paused = not self.paused
        if key == keys.RIGHT and self.paused:
            self.step_frame()

    def step_frame(self) -> None:
        for _ in range(SUBSTEPS):
            self.solver.step(SIM_DT / SUBSTEPS)
        self.fps_steps += 1

    def on_render(self, time: float, frame_time: float):
        if not self.paused:
            self.accumulator += min(frame_time, MAX_FRAME_TIME) * TIME_SCALE
            while self.accumulator >= SIM_DT:
                self.step_frame()
                self.accumulator -= SIM_DT

        pos = self.gl_pos.map(dtype=wp.vec3, shape=(self.solver.n,))
        scalar = self.gl_scalar.map(dtype=float, shape=(self.solver.n,))
        wp.copy(pos, self.solver.x)
        wp.launch(sph.write_speed, dim=self.solver.n, inputs=[self.solver.v, SPEED_SCALE, scalar])
        self.gl_pos.unmap()
        self.gl_scalar.unmap()

        self.depth_fbo.use()
        self.depth_fbo.clear(red=1e6, depth=1.0)

        self.prog["u_view"].write(self.camera.matrix)
        self.prog["u_proj"].write(self.camera.projection.matrix)
        self.prog["u_radius"].value = sph.PARTICLE_RADIUS

        self.prog["u_viewport_h"].value = self.wnd.buffer_size[1]

        self.vao.render(moderngl.POINTS)

        self.blur_prog["u_radius"].value = 20
        self.blur_prog["u_sigma"].value = 10.0
        self.blur_prog["u_depth"].value = 0

        self.blur_fbo_a.use()
        self.depth_tex.use(location=0)
        self.blur_prog["u_direction"].value = (1, 0)
        self.quad.render(self.blur_prog)

        self.blur_fbo_b.use()
        self.blur_tex_a.use(location=0)
        self.blur_prog["u_direction"].value = (0, 1)
        self.quad.render(self.blur_prog)

        self.wnd.use()
        self.ctx.clear(0.0, 0.0, 0.0)

        self.sky_prog["view"].write(self.camera.matrix)
        self.sky_prog["proj"].write(self.camera.projection.matrix)
        self.sky_tex.use(location=0)
        self.sky_prog["skybox"].value = 0

        self.ctx.depth_func = "<="
        self.sky_cube.render(self.sky_prog)
        self.ctx.depth_func = "<"

        self.blur_tex_b.use(location=0)
        self.display_prog["u_proj"].write(self.camera.projection.matrix)
        self.quad.render(self.display_prog)

        wp.synchronize()

        self.fps_frames += 1
        self.fps_elapsed += frame_time
        if self.fps_elapsed >= 0.5:
            fps = self.fps_frames / self.fps_elapsed
            sim_rate = self.fps_steps / self.fps_elapsed
            self.wnd.title = (
                f"3D SPH  |  {fps:.0f} fps  |  {sim_rate:.0f} sim frames/s (target {60 * TIME_SCALE:.0f})"
                f"  |  {SUBSTEPS} substeps  |  {self.solver.n:,} particles"
                f"{'  |  PAUSED' if self.paused else ''}"
            )
            self.fps_frames = 0
            self.fps_steps = 0
            self.fps_elapsed = 0.0


if __name__ == "__main__":
    mglw.run_window_config(Viewer)
