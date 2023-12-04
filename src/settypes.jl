
struct ComplexBounds
    ∂::Tuple{Complex,Complex}
    n::UInt16  
    function ComplexBounds(min::Complex, max::Complex, n::Integer)::ComplexBounds
        new((min,max),n) 
    end
    ComplexBounds(∂::Tuple{Complex,Complex}, n::Integer)::ComplexBounds = ComplexBounds(∂...,n)
end

Base.size(B::ComplexBounds) = (round(UInt16,(B.∂[2].im-B.∂[1].im)/(B.∂[2].re-B.∂[1].re)*B.n),B.n) 

function genplane(B::ComplexBounds)::Matrix{ComplexF64}
    xn,yn = size(B)
    (x,y) = (range(B.∂[1].re,stop=B.∂[2].re,length=xn), range(B.∂[1].im,stop=B.∂[2].im,length=yn))
    return x' .+ y*im
end

parsetofunc(expr,args::Union{Vector,Tuple}) = eval(:(($(args...),)->$expr))

struct SetParams
    E::Expr
    F::Function
    Q::Function
    C::Function
    Ω::ComplexBounds
    N::UInt16
    ϵ::Float64
    z_init::ComplexF64
    function SetParams(E;Q=:(abs2(z)),C=:((angle(z)/(2π))*n^p),∂=(-2.0-2.0im,2.0+2.0im),n::Integer=128,N::Integer=35,ϵ=4.0,z_init::ComplexF64=0.0+0.0im)
        F=parsetofunc(E,(:z, :c))
        Q=parsetofunc(Q,(:z, :c))
        C=parsetofunc(C,(:z, :n, :p))
        Ω=ComplexBounds(∂,n)
        new(E,F,Q,C,Ω,N,ϵ,z_init)
    end
end

function ComputeSet(p::SetParams)
    yn,xn = size(p.Ω)
    plane = genplane(p.Ω)
    (trace_out_iters, tract_out_arg) = (Matrix{UInt16}(undef, yn, xn), Matrix{ComplexF64}(undef, yn, xn))
    Threads.@threads for i = 1:yn; for j = 1:xn
        (trace_out_iters[i,j], tract_out_arg[i,j]) = orbit(p, plane[i,j])::Tuple{UInt16, ComplexF64}
    end; end
    return (trace_out_iters, tract_out_arg)
end

function orbit(p::SetParams, z0::ComplexF64)
    z = p.z_init
    zn = 0x0000
    while (p.Q(z,z0)::Float64<p.ϵ)::Bool && p.N>zn
        z=p.F(z,z0)
        zn+=0x0001
    end
    return (zn::UInt16, z::ComplexF64)
end

function color(p::SetParams, results::Tuple{Matrix{UInt16},Matrix{ComplexF64}})::Matrix{Float16}
    return p.C.(results[2],results[1] .|> float, 0)
end


