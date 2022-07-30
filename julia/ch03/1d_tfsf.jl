using Taichi
using Plots

let
    ti.init(; arch=ti.cuda)
    size = 200
    ez = ti.field(dtype=ti.f64, shape=(size,))
    hy = ti.field(dtype=ti.f64, shape=(size,))
    imp0 = 377.0

    update = @ti_kernel t::Int -> begin
        # do time stepping

        # simple ABC for hy[size - 1]
        hy[size-1] = hy[size-2]

        # update magnetic field
        for mm in ti.static(0:size-2)
            hy[mm] += (ez[mm+1] - ez[mm]) / imp0
        end

        # correction for Hy adjacent to TFSF boundary
        hy[49] -= ti.exp(-(t - 30)^2 / 100) / imp0

        # simple ABC for ez[0]
        ez[0] = ez[1]

        # update electric field
        for mm in ti.static(1:size-1)
            ez[mm] += (hy[mm] - hy[mm-1]) * imp0
        end

        # correction for Ez adjacent to TFSF boundary
        ez[50] += ti.exp(-(t + 0.5 - (-0.5) - 30)^2 / 100)

        return nothing
    end

    anim = @animate for t in 0:300
        update(t)
        Ez = pyconvert(Array, ez.to_numpy())
        Hy = pyconvert(Array, hy.to_numpy()) .* imp0
        p1 = plot(0:size-1, Ez; color=:blue, ylims=(-1, 1), legend=false)
        p2 = plot(0:size-1, Hy; color=:red, ylims=(-1, 1), legend=false)
        if t % 10 == 0
            plot(p1, p2, layout=(2, 1), title=["Ez (t=$t)" "Hy (t=$t)"])
        end
    end every 10

    gif(anim, joinpath(@__DIR__, "..", "gif", "1d_tfsf.gif"), fps=15)
end
