module DeltaArrays

using LinearAlgebra
import Base: similar, copyto!, ndims, size, getindex, parent, real, imag
import Base: -, +, ==
import LinearAlgebra: diagzero, ishermitian, issymmetric, isposdef, factorize, Array

export DeltaArray, delta, deltaind

function deltaind(A::AbstractArray)
    Base.require_one_based_indexing(A)
    deltaind(size(A)[1:end-1]...)
end

deltaind(n::Integer...) = range(1, step=sum(cumprod(n), init=1), length=minimum(n))

delta(i::Integer...) = allequal(i)

delta(A::AbstractArray) = A[deltaind(A)]

struct DeltaArray{T,N,V<:AbstractVector{T}} <: AbstractArray{T,N}
    data::V

    function DeltaArray{T,N,V}(values) where {T,N,V<:AbstractVector{T}}
        Base.require_one_based_indexing(values)
        new{T,N,V}(values)
    end
end

DeltaArray{T,N,V}(D::DeltaArray) where {T,N,V<:AbstractVector{T}} = DeltaArray{T,N,V}(D.data)

function Base.promote_rule(A::Type{<:DeltaArray{<:Any,N,V}}, B::Type{<:DeltaArray{<:Any,N,W}}) where {N,V,W}
    X = promote_type(V, W)
    T = eltype(X)
    isconcretetype(T) && return DeltaArray{T,N,X}
    return typejoin(A, B)
end

"""
    DeltaArray(V::AbstractVector)

Construct a matrix with `V` as its diagonal.

See also [`delta`](@ref).

# Examples
```jldoctest
julia> DeltaArray([1, 10, 100])
3×3 DeltaArray{$Int, 2, Vector{$Int}}:
 1   0    0
 0  10    0
 0   0  100
```
"""
Diagonal(V::AbstractVector)

# `N`=2 by default, equivalent to diagonal
DeltaArray(v::AbstractVector{T}) where {T} = DeltaArray{T,2,typeof(v)}(v)
DeltaArray(d::Diagonal) = DeltaArray(diag(v))
DeltaArray{N}(v::AbstractVector{T}) where {T,N} = DeltaArray{T,N,typeof(v)}(v)
# TODO maybe add `DeltaArray{N}(d::Diagonal)?`

"""
    DeltaArray(M::AbstractMatrix)

Constructs a matrix from the diagonal of `M`.

# Note
The resulting `DeltaArray` should in a similar way to a `Diagonal` object.

# Examples
```jldoctest
julia> A = permutedims(reshape(1:15, 5, 3))
3×5 Matrix{$Int}:
  1   2   3   4   5
  6   7   8   9  10
 11  12  13  14  15

julia> DeltaArray(A)
3×3 DeltaArray{$Int, 2, Vector{$Int}}:
 1  0   0
 0  7   0
 0  0  13

julia> delta(A)
3-element Vector{$Int}:
  1
  7
 13
"""
DeltaArray(M::AbstractMatrix) = DeltaArray(diag(M))

"""
    DeltaArray(A::AbstractArray)

Constructs an array from the diagonal of `A`.

# Examples
```jldoctest
julia> A = reshape(1:16, 2, 2, 2, 2)
2×2×2×2 reshape(::UnitRange{$Int}, 2, 2, 2, 2) with eltype $Int:
[:, :, 1, 1] =
 1  3
 2  4

[:, :, 2, 1] =
 5  7
 6  8

[:, :, 1, 2] =
  9  11
 10  12

[:, :, 2, 2] =
 13  15
 14  16

julia> DeltaArray(A)
2×2 DeltaArray{$Int, 2, Vector{Int64}}:
 1   0
 0  16

julia> delta(A)
2-element Vector{$Int}:
  1
 16
"""
DeltaArray(A::AbstractArray) = DeltaArray{ndims(A)}(delta(A))

DeltaArray(D::DeltaArray) = D
DeltaArray{T}(D::DeltaArray) where {T} = D
DeltaArray{T,N}(D::DeltaArray) where {T,N} = DeltaArray{T,N}(D.data)

AbstractArray{T}(D::DeltaArray) where {T} = DeltaArray{T}(D)
AbstractArray{T,N}(D::DeltaArray) where {T,N} = DeltaArray{T,N}(D)
Array(D::DeltaArray{T,N}) where {T,N} = Array{promote_type(T, typeof(zero(T)))}(D)
function Array{T,N}(D::DeltaArray) where {T,N}
    n = size(D, 1)
    B = zeros(T, ntuple(_ -> n, N))
    @inbounds for i in 1:n
        # TODO revise if `ntuple` is performance optimal
        # alternative could be to use `B[delta(B)] .= D.data`
        B[ntuple(_ -> i, N)...] = D.data[i]
    end
    return B
end

"""
    DeltaArray{T,N}(undef,n)

Construct an uninitialized `DeltaArray{T,N}` of order `N` and length `n`.
"""
DeltaArray{T,N}(::UndefInitializer, n::Integer) where {T,N} = DeltaArray{N}(Vector{T}(undef, n))

similar(D::DeltaArray, ::Type{T}) where {T} = DeltaArray(similar(D.data, T))
similar(::DeltaArray, ::Type{T}, dims) where {T} = zeros(T, dims...)

copyto!(D1::DeltaArray, D2::DeltaArray) = (copyto!(D1.data, D2.data); D1)

__nvalues(D::DeltaArray) = length(D.data)

ndims(D::DeltaArray{T,N}) where {T,N} = N
size(D::DeltaArray) = ntuple(_ -> __nvalues(D), ndims(D))

# TODO put type to i... to be `Integer`?
@inline function getindex(D::DeltaArray, i::Integer...)
    @boundscheck checkbounds(D, i...)
    if allequal(i)
        @inbounds r = D.data[first(i)]
    else
        r = diagzero(D, i...)
    end
    r
end
diagzero(::DeltaArray{T}, i...) where {T} = zero(T)
diagzero(D::DeltaArray{<:AbstractArray{T,N}}, i...) where {T,N} = zeros(T, (size(D.data[j], n) for (j, n) in zip(i, 1:N))...)

# TODO put type to i... to be `Integer`?
function setindex!(D::DeltaArray, v, i...)
    @boundscheck checkbounds(D, i...)
    if allequal(i...)
        @inbounds D.data[first(i)] = v
    elseif !iszero(v)
        throw(ArgumentError("cannot set off-diagonal entry ($(i...)) to a nonzero value ($v)"))
    end
    return v
end

## TODO structured matrix methods
# function Base.replace_in_print_matrix(D::DeltaArray, i..., s)

parent(D::DeltaArray) = D.data

# TODO ishermitian, issymmetric, isposdef, factorize for DeltaArray{T,2}

real(D::DeltaArray) = DeltaArray{ndims(D)}(real(D.data))
imag(D::DeltaArray) = DeltaArray{ndims(D)}(imag(D.data))

(==)(Da::DeltaArray, Db::DeltaArray) = DeltaArray() # TODO
(-)(D::DeltaArray) = DeltaArray{ndims(D)}(-D.data)
(+)(Da::DeltaArray, Db::DeltaArray) = DeltaArray{ndims(D)}() # TODO
# TODO ...

end