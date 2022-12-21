################################################################################
# Utility functions for elliptical and Mixture Copulas

@inline function filldiag!(mat::AbstractMatrix, val::T) where {T<:Real}
    @inbounds for idx in CartesianIndices(mat)
        if idx[1] != idx[2]
            mat[idx] = val
        end
    end
    return nothing
end
function ρToΣ(ρᵥ::AbstractVector{T}) where {T<:Real}
    Σ = [ones(eltype(ρᵥ), 2, 2) for _ in Base.OneTo( size(ρᵥ, 1) ) ]
    for iter in eachindex(Σ)
        filldiag!(Σ[iter], ρᵥ[iter])
    end
    return Σ
end
function ρToΣ(ρ::T) where {T<:Real}
    Σ = ones(eltype(ρ), 2, 2)
    filldiag!(Σ, ρ)
    return Σ
end

################################################################################
#Include
include("mixturequantile.jl")

################################################################################
#export
