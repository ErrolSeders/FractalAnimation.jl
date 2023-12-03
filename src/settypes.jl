begin 
    using Pkg

    Pkg.activate(".")
    Pkg.instantiate()

end

abstract type AbstractSpace end

struct BoundedRectangle <: AbstractSpace
    ∂::Tuple{Number, Number}
end

struct BoundedComplexRectangle <: AbstractSpace
    ∂::Tuple{Complex, Complex}
    n::UInt16 #resolution
end



parsetofunc(expr,args::Union{Vector,Tuple}) = eval(:(($(args...),)->$expr))

struct SetParams
    E::Expr
    F::Function
    Q::Function
    C::Function
    Ω::BoundedComplexRectangle
    N::UInt16
    z_init::ComplexF64
    function SetParams(E;Q=:(abs2(z)),C=:((angle(z)/(2π))*n^p),∂=(-2.0-2.0im,2.0+2.0im),n::Integer=128,N::Integer=35,z_init::ComplexF64=0.0+0.0im) where {FT,QT,CT}
        F=parsetofunc(E,(:z, :c))
        Q=parsetofunc(Q,(:z, :c))
        C=parsetofunc(C,(:z, :c))
        Ω=BoundedComplexRectangle(∂,n)
        new(E,F,Q,C,Ω,N,z_init)
    end
end

function ComputeSet(p::SetParams)
    yn,xn = size(p.Ω)
    (trace_out_iters, tract_out_arg) = (Matrix{UInt16}(undef, yn, xn), Matrix{ComplexF64}(undef, yn, xn))
    Threads.@threads for i = 1:yn; for j = 1:xn
        (trace_out_iters[i,j], tract_out_arg[i,j]) = orbit(p, p.Ω[i,j])::Tuple{UInt16, ComplexF64}
    end; end
    return (trace_out_iters, tract_out_arg)
end

function orbit(p::SetParams, z0::ComplexF64)
    z = p.z_init
    zn = 0x0000
    while (N ? (p.Q(z,z0)::Float64>K.ϵ)::Bool : (p.Q(z,z0)::Float64<K.ϵ)) && K.N>zn
        z=p.F(z,z0)
        zn+=0x0001
    end
    return (zn::UInt16, z::ComplexF64)
end


