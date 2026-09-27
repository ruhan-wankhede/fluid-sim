

import math
from pathlib import Path
from typing import Any

import moderngl
import moderngl_window as mglw
import numpy as np
import warp as wp
from moderngl_window.context.base import KeyModifiers

import sph


SIM_DT = 1.0 / 60.0
MAX_FRAME_TIME = 0.25

# Simulated seconds per real second. Scales how much time is fed to the
# accumulator, not the step size, so stability is unaffected -- only cost.
TIME_SCALE = 1.0

# Max fraction of the support radius a pressure wave may cross in one substep.
# Blows up above ~0.6; Lague's scenes sit around 0.35.
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
    title = "2D SPH"
    WIDTH = 2560
    HEIGHT = 1600
    WIDTH *= 0.96
    HEIGHT *= 0.96
    window_size = (WIDTH, HEIGHT)
    resizable = True
    vsync = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        wp.init()
        self.solver = sph.Solver()
        self.paused = True
        print(sph.log(self.solver))
        print("substeps per frame:", SUBSTEPS)

        # Hybrid-graphics check. Warp computes on the 4090, so if GL lands on the
        # Intel iGPU every frame crosses between devices and wp.RegisteredGLBuffer
        # interop is impossible. Force it via Windows Graphics Settings if needed.
        print("GL renderer:", self.ctx.info["GL_RENDERER"])

        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.prog = self.ctx.program(
            vertex_shader=read_shader("points.vert"),
            fragment_shader=read_shader("points.frag"),
        )

        n = self.solver.n
        self.vbo_pos = self.ctx.buffer(reserve=n * 3 * 4, dynamic=True)
        self.vbo_scalar = self.ctx.buffer(reserve=n * 4, dynamic=True)
        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.vbo_pos, "3f", "in_pos"),
                (self.vbo_scalar, "1f", "in_scalar"),
            ],
        )

        self.accumulator = 0.0

    def letterbox(self) -> tuple[int, int, int, int]:
        """Largest domain-aspect rectangle centred in the window, in pixels.

        Keeps point sprites circular regardless of how the window is resized.
        """
        width_px, height_px = self.wnd.buffer_size
        aspect = self.solver.width / self.solver.height

        if width_px / height_px > aspect:
            visible_height = height_px
            visible_width = int(round(height_px * aspect))
        else:
            visible_width = width_px
            visible_height = int(round(width_px / aspect))

        return (
            (width_px - visible_width) // 2,
            (height_px - visible_height) // 2,
            visible_width,
            visible_height,
        )

    def on_key_event(self, key: Any, action: Any, modifiers: KeyModifiers) -> None:
        keys = self.wnd.keys

        if action != keys.ACTION_PRESS:
            return
        if key == keys.SPACE:
            self.paused = not self.paused
        if key == keys.RIGHT and self.paused:
            self.step_frame()

    def step_frame(self) -> None:
        for _ in range(SUBSTEPS):
            self.solver.step(SIM_DT / SUBSTEPS)

    def on_render(self, time: float, frame_time: float):
        if not self.paused:
            self.accumulator += min(frame_time, MAX_FRAME_TIME) * TIME_SCALE
            while self.accumulator >= SIM_DT:
                self.step_frame()
                self.accumulator -= SIM_DT

        positions = self.solver.x.numpy()
        speed = np.linalg.norm(self.solver.v.numpy(), axis=1) / SPEED_SCALE

        self.vbo_pos.write(positions.astype("f4"))
        self.vbo_scalar.write(speed.astype("f4"))

        self.ctx.clear(0.0, 0.0, 0.0)
        viewport = self.letterbox()
        self.ctx.viewport = viewport

        self.prog["u_domain"].value = (self.solver.width, self.solver.height)
        self.prog["u_diameter_px"].value = (
                2.0 * sph.PARTICLE_RADIUS * viewport[3] / self.solver.height
        )

        self.vao.render(moderngl.POINTS)


if __name__ == "__main__":
    mglw.run_window_config(Viewer)
