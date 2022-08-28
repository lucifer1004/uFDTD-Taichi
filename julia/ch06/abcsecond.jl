using Taichi
using Plots

let
    ti.init(; arch=ti.cuda)
    size = 200
    ez = ti.field(dtype=ti.f64, shape=(size,))
    hy = ti.field(dtype=ti.f64, shape=(size - 1,))
    ceze = ti.field(dtype=ti.f64, shape=(size,))
    cezh = ti.field(dtype=ti.f64, shape=(size,))
    chyh = ti.field(dtype=ti.f64, shape=(size - 1,))
    chye = ti.field(dtype=ti.f64, shape=(size - 1,))
    abc_coef_left = ti.field(dtype=ti.f64, shape=(3,))
    abc_coef_right = ti.field(dtype=ti.f64, shape=(3,))
    ez_old_left1 = ti.field(dtype=ti.f64, shape=(3,))
    ez_old_left2 = ti.field(dtype=ti.f64, shape=(3,))
    ez_old_right1 = ti.field(dtype=ti.f64, shape=(3,))
    ez_old_right2 = ti.field(dtype=ti.f64, shape=(3,))
    imp0 = 377.0
    cdtds = 1.0 # Courant number
    delay = 30.0
    width = 10.0
    tfsf_boundary = 49

    ez_inc = @ti_func (time::Int, location::Float64)::Float64 ->
        ti.exp(-((time - delay - location / cdtds) / width)^2)

    init_grid = @ti_kernel () -> begin
        for i in ceze
            ceze[i] = 1.0

            if i < size ÷ 2
                cezh[i] = imp0
            else
                cezh[i] = imp0 / 9.0
            end
        end

        for i in chyh
            chyh[i] = 1.0
            chye[i] = 1.0 / imp0
        end
    end

    init_abc = @ti_kernel () -> begin
        tmp = ti.sqrt(cezh[0] * chye[0])
        tmp2 = 1.0 / tmp + 2.0 + tmp
        abc_coef_left[0] = -(1.0 / tmp - 2.0 + tmp) / tmp2
        abc_coef_left[1] = -2.0 * (tmp - 1.0 / tmp) / tmp2
        abc_coef_left[2] = 4.0 * (tmp + 1.0 / tmp) / tmp2

        tmp = ti.sqrt(cezh[size-1] * chye[size-2])
        tmp2 = 1.0 / tmp + 2.0 + tmp
        abc_coef_right[0] = -(1.0 / tmp - 2.0 + tmp) / tmp2
        abc_coef_right[1] = -2.0 * (tmp - 1.0 / tmp) / tmp2
        abc_coef_right[2] = 4.0 * (tmp + 1.0 / tmp) / tmp2

        return nothing
    end

    update_abc = @ti_kernel () -> begin
        # ABC for left side of grid
        ez[0] = abc_coef_left[0] * (ez[2] + ez_old_left2[0]) +
                abc_coef_left[1] * (ez_old_left1[0] + ez_old_left1[2] - ez[1] - ez_old_left2[1]) +
                abc_coef_left[2] * ez_old_left1[1] - ez_old_left2[2]

        # ABC for right side of grid
        ez[size-1] = abc_coef_right[0] * (ez[size-3] + ez_old_right2[0]) +
                     abc_coef_right[1] * (ez_old_right1[0] + ez_old_right1[2] - ez[size-2] - ez_old_right2[1]) +
                     abc_coef_right[2] * ez_old_right1[1] - ez_old_right2[2]

        for mm in ez_old_left1
            ez_old_left2[mm] = ez_old_left1[mm]
            ez_old_left1[mm] = ez[mm]

            ez_old_right2[mm] = ez_old_right1[mm]
            ez_old_right1[mm] = ez[size-1-mm]
        end

        return nothing
    end

    # update magnetic field
    update_H = @ti_kernel () -> for mm in ti.static(0:size-2)
        hy[mm] = chyh[mm] * hy[mm] + chye[mm] * (ez[mm+1] - ez[mm])
    end

    # update electric field
    update_E = @ti_kernel () -> for mm in ti.static(1:size-2)
        ez[mm] = ceze[mm] * ez[mm] + cezh[mm] * (hy[mm] - hy[mm-1])
    end

    update_tfsf = @ti_kernel (time::Int) -> begin
        # correct Hy adjacent to TFSF boundary
        hy[tfsf_boundary] -= ez_inc(time, 0.0) * chye[tfsf_boundary]

        # correct Ez adjacent to TFSF boundary
        ez[tfsf_boundary+1] += ez_inc(time + 0.5, -0.5)

        return nothing
    end

    init_grid()
    init_abc()
    anim = @animate for t in 0:800
        # do time stepping
        update_H()
        update_tfsf(t)
        update_E()
        update_abc()

        Ez = pyconvert(Array, ez.to_numpy())
        Hy = pyconvert(Array, hy.to_numpy()) .* imp0
        p1 = plot(0:size-1, Ez; color=:blue, ylims=(-1, 1), legend=false)
        p2 = plot(0:size-2, Hy; color=:red, ylims=(-1, 1), legend=false)
        if t % 10 == 0
            plot(p1, p2, layout=(2, 1), title=["Ez (t=$t)" "Hy (t=$t)"])
        end
    end every 10

    gif(anim, joinpath(@__DIR__, "..", "gif", "abcsecond.gif"), fps=15)
end
