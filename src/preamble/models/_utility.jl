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
# Rotation for archimedean Copulas

function rotatecopula(rotation::Rotation0, u::AbstractVector{T}) where {T<:Real}
    return u
end
function rotatecopula(rotation::Rotation90, u::AbstractVector{T}) where {T<:Real}
    u_rotated = zeros(eltype(u), size(u))
    u_rotated[1] = 1 - u[2]
    u_rotated[2] = u[1]
    return u_rotated
end
function rotatecopula(rotation::Rotation180, u::AbstractVector{T}) where {T<:Real}
    u_rotated = zeros(eltype(u), size(u))
    u_rotated[1] = 1 - u[1]
    u_rotated[2] = 1 - u[2]
    return u_rotated
end
function rotatecopula(rotation::Rotation270, u::AbstractVector{T}) where {T<:Real}
    u_rotated = zeros(eltype(u), size(u))
    u_rotated[1] = u[2]
    u_rotated[2] = 1 - u[1]
    return u_rotated
end
function unrotatecopula(rotation::Rotation0, u::AbstractVector{T}) where {T<:Real}
    return u
end
function unrotatecopula(rotation::Rotation90, u_rotated::AbstractVector{T}) where {T<:Real}
    u = zeros(eltype(u_rotated), size(u_rotated))
    u[1] = u_rotated[2]
    u[2] = 1 - u_rotated[1]
    return u
end
function unrotatecopula(rotation::Rotation180, u_rotated::AbstractVector{T}) where {T<:Real}
    u = zeros(eltype(u_rotated), size(u_rotated))
    u[1] = 1 - u_rotated[1]
    u[2] = 1 - u_rotated[2]
    return u
end
function unrotatecopula(rotation::Rotation270, u_rotated::AbstractVector{T}) where {T<:Real}
    u = zeros(eltype(u_rotated), size(u_rotated))
    u[1] = 1 - u_rotated[2]
    u[2] = u_rotated[1]
    return u
end

######################################
function rotatecopula(rotation::Rotation0, u::AbstractMatrix{T}) where {T<:Real}
    return u
end
function rotatecopula(rotation::Rotation90, u::AbstractMatrix{T}) where {T<:Real}
    u_rotated = zeros(eltype(u), size(u))
    for iter in Base.OneTo(size(u, 2))
        u_rotated[1, iter] = 1 - u[2, iter]
        u_rotated[2, iter] = u[1, iter]
    end
    return u_rotated
end
function rotatecopula(rotation::Rotation180, u::AbstractMatrix{T}) where {T<:Real}
    u_rotated = zeros(eltype(u), size(u))
    for iter in Base.OneTo(size(u, 2))
        u_rotated[1, iter] = 1 - u[1, iter]
        u_rotated[2, iter] = 1 - u[2, iter]
    end
    return u_rotated
end
function rotatecopula(rotation::Rotation270, u::AbstractMatrix{T}) where {T<:Real}
    u_rotated = zeros(eltype(u), size(u))
    for iter in Base.OneTo(size(u, 2))
        u_rotated[1, iter] = u[2, iter]
        u_rotated[2, iter] = 1 - u[1, iter]
    end
    return u_rotated
end

function unrotatecopula(rotation::Rotation0, u::AbstractMatrix{T}) where {T<:Real}
    return u
end
function unrotatecopula(rotation::Rotation90, u_rotated::AbstractMatrix{T}) where {T<:Real}
    u = zeros(eltype(u_rotated), size(u_rotated))
    for iter in Base.OneTo(size(u, 2))
        u[1, iter] = u_rotated[2, iter]
        u[2, iter] = 1 - u_rotated[1, iter]
    end
    return u
end
function unrotatecopula(rotation::Rotation180, u_rotated::AbstractMatrix{T}) where {T<:Real}
    u = zeros(eltype(u_rotated), size(u_rotated))
    for iter in Base.OneTo(size(u, 2))
        u[1, iter] = 1 - u_rotated[1, iter]
        u[2, iter] = 1 - u_rotated[2, iter]
    end
    return u
end
function unrotatecopula(rotation::Rotation270, u_rotated::AbstractMatrix{T}) where {T<:Real}
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

u1_rotated = rotatecopula(Rotation90(), u1)
u2_rotated = rotatecopula(Rotation90(), u2)

u1_orig = unrotatecopula(Rotation90(), u1_rotated)
u2_orig = unrotatecopula(Rotation90(), u2_rotated)

sum( abs.(u1 .- u1_orig))
sum( abs.(u2 .- u2_orig))

u1_rotated = rotatecopula(Rotation180(), u1)
u2_rotated = rotatecopula(Rotation180(), u2)

u1_orig = unrotatecopula(Rotation180(), u1_rotated)
u2_orig = unrotatecopula(Rotation180(), u2_rotated)

sum( abs.(u1 .- u1_orig))
sum( abs.(u2 .- u2_orig))

u1_rotated = rotatecopula(Rotation270(), u1)
u2_rotated = rotatecopula(Rotation270(), u2)

u1_orig = unrotatecopula(Rotation270(), u1_rotated)
u2_orig = unrotatecopula(Rotation270(), u2_rotated)

sum( abs.(u1 .- u1_orig))
sum( abs.(u2 .- u2_orig))
=#
