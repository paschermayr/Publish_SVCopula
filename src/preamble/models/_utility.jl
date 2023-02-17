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
# Reflection for archimedean Copulas

function rotatecopula(reflection::Reflection0, u::AbstractVector{T}) where {T<:Real}
    return u
end
function rotatecopula(reflection::Reflection90, u::AbstractVector{T}) where {T<:Real}
    u_rotated = zeros(eltype(u), size(u))
    u_rotated[1] = 1 - u[2]
    u_rotated[2] = u[1]
    return u_rotated
end
function rotatecopula(reflection::Reflection180, u::AbstractVector{T}) where {T<:Real}
    u_rotated = zeros(eltype(u), size(u))
    u_rotated[1] = 1 - u[1]
    u_rotated[2] = 1 - u[2]
    return u_rotated
end
function rotatecopula(reflection::Reflection270, u::AbstractVector{T}) where {T<:Real}
    u_rotated = zeros(eltype(u), size(u))
    u_rotated[1] = u[2]
    u_rotated[2] = 1 - u[1]
    return u_rotated
end
function unrotatecopula(reflection::Reflection0, u::AbstractVector{T}) where {T<:Real}
    return u
end
function unrotatecopula(reflection::Reflection90, u_rotated::AbstractVector{T}) where {T<:Real}
    u = zeros(eltype(u_rotated), size(u_rotated))
    u[1] = u_rotated[2]
    u[2] = 1 - u_rotated[1]
    return u
end
function unrotatecopula(reflection::Reflection180, u_rotated::AbstractVector{T}) where {T<:Real}
    u = zeros(eltype(u_rotated), size(u_rotated))
    u[1] = 1 - u_rotated[1]
    u[2] = 1 - u_rotated[2]
    return u
end
function unrotatecopula(reflection::Reflection270, u_rotated::AbstractVector{T}) where {T<:Real}
    u = zeros(eltype(u_rotated), size(u_rotated))
    u[1] = 1 - u_rotated[2]
    u[2] = u_rotated[1]
    return u
end

######################################
function rotatecopula(reflection::Reflection0, u::AbstractMatrix{T}) where {T<:Real}
    return u
end
function rotatecopula(reflection::Reflection90, u::AbstractMatrix{T}) where {T<:Real}
    u_rotated = zeros(eltype(u), size(u))
    for iter in Base.OneTo(size(u, 2))
        u_rotated[1, iter] = 1 - u[2, iter]
        u_rotated[2, iter] = u[1, iter]
    end
    return u_rotated
end
function rotatecopula(reflection::Reflection180, u::AbstractMatrix{T}) where {T<:Real}
    u_rotated = zeros(eltype(u), size(u))
    for iter in Base.OneTo(size(u, 2))
        u_rotated[1, iter] = 1 - u[1, iter]
        u_rotated[2, iter] = 1 - u[2, iter]
    end
    return u_rotated
end
function rotatecopula(reflection::Reflection270, u::AbstractMatrix{T}) where {T<:Real}
    u_rotated = zeros(eltype(u), size(u))
    for iter in Base.OneTo(size(u, 2))
        u_rotated[1, iter] = u[2, iter]
        u_rotated[2, iter] = 1 - u[1, iter]
    end
    return u_rotated
end

function unrotatecopula(reflection::Reflection0, u::AbstractMatrix{T}) where {T<:Real}
    return u
end
function unrotatecopula(reflection::Reflection90, u_rotated::AbstractMatrix{T}) where {T<:Real}
    u = zeros(eltype(u_rotated), size(u_rotated))
    for iter in Base.OneTo(size(u, 2))
        u[1, iter] = u_rotated[2, iter]
        u[2, iter] = 1 - u_rotated[1, iter]
    end
    return u
end
function unrotatecopula(reflection::Reflection180, u_rotated::AbstractMatrix{T}) where {T<:Real}
    u = zeros(eltype(u_rotated), size(u_rotated))
    for iter in Base.OneTo(size(u, 2))
        u[1, iter] = 1 - u_rotated[1, iter]
        u[2, iter] = 1 - u_rotated[2, iter]
    end
    return u
end
function unrotatecopula(reflection::Reflection270, u_rotated::AbstractMatrix{T}) where {T<:Real}
    u = zeros(eltype(u_rotated), size(u_rotated))
    for iter in Base.OneTo(size(u, 2))
        u[1, iter] = 1 - u_rotated[2, iter]
        u[2, iter] = u_rotated[1, iter]
    end
    return u
end

#=
u1 = [.1, .2]
u2 = [1 2 3 4 5 ; 6 7 8 9 10]

u1_rotated = rotatecopula(Reflection90(), u1)
u2_rotated = rotatecopula(Reflection90(), u2)

u1_orig = unrotatecopula(Reflection90(), u1_rotated)
u2_orig = unrotatecopula(Reflection90(), u2_rotated)

sum( abs.(u1 .- u1_orig))
sum( abs.(u2 .- u2_orig))

u1_rotated = rotatecopula(Reflection180(), u1)
u2_rotated = rotatecopula(Reflection180(), u2)

u1_orig = unrotatecopula(Reflection180(), u1_rotated)
u2_orig = unrotatecopula(Reflection180(), u2_rotated)

sum( abs.(u1 .- u1_orig))
sum( abs.(u2 .- u2_orig))

u1_rotated = rotatecopula(Reflection270(), u1)
u2_rotated = rotatecopula(Reflection270(), u2)

u1_orig = unrotatecopula(Reflection270(), u1_rotated)
u2_orig = unrotatecopula(Reflection270(), u2_rotated)

sum( abs.(u1 .- u1_orig))
sum( abs.(u2 .- u2_orig))


################################################################################
# Proof of work concept:

#Current workflow
data = rand(2, 10)
reflection = Reflection90()
copula = frankcopula

u = rand(2)
u_unrotated = unrotatecopula(Reflection90(), u)
u_rotated = rotatecopula(Reflection90(), u)

#1 Assume rotated data as input: Unrorate data, then evaluate:
ℓur = ℓlikelihood(Frank(), (;α = 5.), u_unrotated)

# Assume unrotated data as input: Rotate data, then evaluate
ℓr = ℓlikelihood(Frank(), (;α = 5.), u_rotated)

ℓur ≈ ℓr

=#

################################################################################
# Utility function to to name copulas
function get_rotation_name(copula::AbstractCopulas, reflection)
    return ""
end
function get_rotation_name(copula::Archimedean, reflection::Reflection90)
    return "1-reflected "
end
function get_rotation_name(copula::Archimedean, reflection::Reflection180)
    return "Survival "
end
function get_rotation_name(copula::Archimedean, reflection::Reflection270)
    return "2-reflected "
end

function get_copula_name(copula::AbstractCopulas, reflection::ArchimedeanReflection)
    return string(get_rotation_name(copula, reflection), typeof(copula))
end
