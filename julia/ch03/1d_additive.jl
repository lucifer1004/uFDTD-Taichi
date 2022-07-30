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

        # update magnetic field
        for mm in ti.static(0:size-2)
            hy[mm] += (ez[mm+1] - ez[mm]) / imp0
        end

        # update electric field
        for mm in ti.static(1:size-1)
            ez[mm] += (hy[mm] - hy[mm-1]) * imp0
        end

        # use additive source at node 50
        ez[50] += ti.exp(-(t - 30)^2 / 100)

        return nothing
    end

    anim = @animate for t in 0:1000
        update(t)
        Ez = pyconvert(Array, ez.to_numpy())
        Hy = pyconvert(Array, hy.to_numpy()) .* imp0
        p1 = plot(0:size-1, Ez; color=:blue, ylims=(-1, 1), legend=false)
        p2 = plot(0:size-1, Hy; color=:red, ylims=(-1, 1), legend=false)
        if t % 10 == 0
            plot(p1, p2, layout=(2, 1), title=["Ez (t=$t)" "Hy (t=$t)"])
        end
    end every 10

    gif(anim, joinpath(@__DIR__, "..", "gif", "1d_additive.gif"), fps=15)
end
