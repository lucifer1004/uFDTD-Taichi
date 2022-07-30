import taichi as ti
import numpy as np

ti.init(arch=ti.cuda)

size = 200
ez = ti.field(dtype=ti.f64, shape=(size))
hy = ti.field(dtype=ti.f64, shape=(size-1))
ceze = ti.field(dtype=ti.f64, shape=(size))
cezh = ti.field(dtype=ti.f64, shape=(size))
imp0 = 377.0
loss = 0.01


@ti.kernel
def init():
    for i in ceze:
        if i < size // 2:
            ceze[i] = 1.0
            cezh[i] = imp0
        else:
            ceze[i] = (1.0 - loss) / (1.0 + loss)
            cezh[i] = imp0 / 9.0 / (1.0 + loss)


@ti.kernel
def update(t: int):  # do time stepping
    # update magnetic field
    for mm in ti.static(range(size - 1)):
        hy[mm] += (ez[mm + 1] - ez[mm]) / imp0

    # correction for Hy adjacent to TFSF boundary
    hy[49] -= ti.exp(-(t - 30) ** 2 / 100) / imp0

    # simple ABC for ez[0]
    ez[0] = ez[1]

    # update electric field
    for mm in ti.static(range(1, size)):
        ez[mm] = ceze[mm] * ez[mm] + cezh[mm] * (hy[mm] - hy[mm - 1])

    # correction for Ez adjacent to TFSF boundary
    ez[50] += ti.exp(-(t + 0.5 - (-0.5) - 30) ** 2 / 100)


def color(v: float):
    x = min(255, int(v * 256))
    if x < 0:
        return (-x) * 256 * 256  # Red
    return x * 256  # Green


width = 1280
height = 320
gui = ti.GUI("1D Total Field / Scattered Field", res=(width, height))
t = 0
init()

vm = ti.tools.VideoManager(
    output_dir='gif', framerate=24, automatic_build=True)
while gui.running:
    t += 1
    update(t)
    colors = np.array([color(ei) for ei in ez.to_numpy()])
    gui.text('Ez', (0.1, 0.1))
    gui.circles(pos=np.array([((i + 0.5) / size, 0.3)
                for i in range(size)]), radius=3, color=colors)

    # multiply by imp0 so that the value falls between -1 and 1
    gui.text('Hy', (0.1, 0.6))
    colors = np.array([color(hi) for hi in hy.to_numpy() * imp0])
    gui.circles(pos=np.array([((i + 0.5) / size, 0.8)
                              for i in range(size-1)]), radius=3, color=colors)

    vm.write_frame(gui.get_image())
    gui.show()
