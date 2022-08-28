import taichi as ti
import numpy as np

ti.init(arch=ti.cuda)

size = 200
ez = ti.field(dtype=ti.f64, shape=(size))
hy = ti.field(dtype=ti.f64, shape=(size - 1))
ceze = ti.field(dtype=ti.f64, shape=(size))
cezh = ti.field(dtype=ti.f64, shape=(size))
chyh = ti.field(dtype=ti.f64, shape=(size - 1))
chye = ti.field(dtype=ti.f64, shape=(size - 1))
abc_coef_left = ti.field(dtype=ti.f64, shape=(3))
abc_coef_right = ti.field(dtype=ti.f64, shape=(3))
ez_old_left1 = ti.field(dtype=ti.f64, shape=(3))
ez_old_left2 = ti.field(dtype=ti.f64, shape=(3))
ez_old_right1 = ti.field(dtype=ti.f64, shape=(3))
ez_old_right2 = ti.field(dtype=ti.f64, shape=(3))
imp0 = 377.0
cdtds = 1.0  # Courant number
delay = 30.0
width = 10.0
tfsf_boundary = 49


@ti.func
def ez_inc(time: int, location: float) -> float:
    return ti.exp(-((time - delay - location / cdtds) / width) ** 2)


@ti.kernel
def init_grid():
    for i in ceze:
        ceze[i] = 1.0
        if i < size // 2:
            cezh[i] = imp0
        else:
            cezh[i] = imp0 / 9.0
    for i in chyh:
        chyh[i] = 1.0
        chye[i] = 1.0 / imp0


@ti.kernel
def init_abc():
    tmp = ti.sqrt(cezh[0] * chye[0])
    tmp2 = 1.0 / tmp + 2.0 + tmp
    abc_coef_left[0] = -(1.0 / tmp - 2.0 + tmp) / tmp2
    abc_coef_left[1] = -2.0 * (tmp - 1.0 / tmp) / tmp2
    abc_coef_left[2] = 4.0 * (tmp + 1.0 / tmp) / tmp2
    tmp = ti.sqrt(cezh[size - 1] * chye[size - 2])
    tmp2 = 1.0 / tmp + 2.0 + tmp
    abc_coef_right[0] = -(1.0 / tmp - 2.0 + tmp) / tmp2
    abc_coef_right[1] = -2.0 * (tmp - 1.0 / tmp) / tmp2
    abc_coef_right[2] = 4.0 * (tmp + 1.0 / tmp) / tmp2


@ti.kernel
def update_abc():
    # ABC for left side of grid
    ez[0] = abc_coef_left[0] * (ez[2] + ez_old_left2[0]) + \
        abc_coef_left[1] * (ez_old_left1[0] +
                            ez_old_left1[2] - ez[1] - ez_old_left2[1]) + \
        abc_coef_left[2] * ez_old_left1[1] - ez_old_left2[2]

    # ABC for right side of grid
    ez[size - 1] = abc_coef_right[0] * (ez[size - 3] + ez_old_right2[0]) + \
        abc_coef_right[1] * (
        ez_old_right1[0] + ez_old_right1[2] - ez[size - 2] - ez_old_right2[1]) + \
        abc_coef_right[2] * ez_old_right1[1] - ez_old_right2[2]
    for mm in ez_old_left1:
        ez_old_left2[mm] = ez_old_left1[mm]
        ez_old_left1[mm] = ez[mm]
        ez_old_right2[mm] = ez_old_right1[mm]
        ez_old_right1[mm] = ez[size - 1 - mm]


@ti.kernel
def update_H():  # update magnetic field
    for mm in ti.static(range(size - 1)):
        hy[mm] = chyh[mm] * hy[mm] + chye[mm] * (ez[mm + 1] - ez[mm])


@ti.kernel
def update_E():  # update electric field
    for mm in ti.static(range(1, size - 1)):
        ez[mm] = ceze[mm] * ez[mm] + cezh[mm] * (hy[mm] - hy[mm - 1])


@ti.kernel
def update_tfsf(time: int):
    # correct Hy adjacent to TFSF boundary
    hy[tfsf_boundary] -= ez_inc(time, 0.0) * chye[tfsf_boundary]

    # correct Ez adjacent to TFSF boundary
    ez[tfsf_boundary + 1] += ez_inc(time + 0.5, -0.5)


def color(v: float):
    x = min(255, int(v * 256))
    if x < 0:
        return (-x) * 256 * 256  # Red
    return x * 256  # Green


window_size = (1280, 320)
gui = ti.GUI("First Order ABC", res=window_size)
t = 0

init_grid()
init_abc()
while gui.running:
    update_H()
    update_tfsf(t)
    update_E()
    update_abc()
    t += 1

    colors = np.array([color(ei) for ei in ez.to_numpy()])
    gui.text('Ez', (0.1, 0.1))
    gui.circles(pos=np.array([((i + 0.5) / size, 0.3)
                for i in range(size)]), radius=3, color=colors)

    # multiply by imp0 so that the value falls between -1 and 1
    gui.text('Hy', (0.1, 0.6))
    colors = np.array([color(hi) for hi in hy.to_numpy() * imp0])
    gui.circles(pos=np.array([((i + 0.5) / size, 0.8)
                              for i in range(size-1)]), radius=3, color=colors)
    gui.show()
