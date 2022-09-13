module DeltaArrays

using LinearAlgebra
import Base: similar, copyto!, ndims, size, getindex, parent, real, imag
import LinearAlgebra: diagzero, ishermitian, issymmetric, isposdef, factorize

export DeltaArray

struct DeltaArray{T,N,V<:AbstractVector{T}} <: AbstractArray{T,N}
    data::V

    function DeltaArray{T,N,V}(values) where {T,N,V<:AbstractVector{T}}
        Base.require_one_based_indexing(values)
        new{T,N,V}(values)
    end
end

DeltaArray{T,N,V}(D::DeltaArray) where {T,N,V<:AbstractVector{T}} = DeltaArray{T,N,V}(D.data)

# `N`=2 by default, equivalent to diagonal
DeltaArray(v::AbstractVector{T}) where {T} = DeltaArray{T,2,typeof(v)}(v)
DeltaArray(d::Diagonal) = DeltaArray(diag(v))
DeltaArray{N}(v::AbstractVector{T}) where {T,N} = DeltaArray{T,N,typeof(v)}(v)
# TODO maybe add `DeltaArray{N}(d::Diagonal)?`

DeltaArray(M::AbstractMatrix) = DeltaArray(diag(M))

DeltaArray(D::DeltaArray) = D

# ...

"""
    DeltaArray{T,N}(undef,n)
Construct an uninitialized `DeltaArray{T,N}` of order `N` and length `n`.
"""
DeltaArray{T,N}(::UndefInitializer, n::Integer) where {T,N} = DeltaArray{N}(Vector{T}(undef, n))

similar(D::DeltaArray, ::Type{T}) where {T} = DeltaArray(similar(D.data, T))
# similar(::DeltaArray, ::Type{T}, dims)

copyto!(D1::DeltaArray, D2::DeltaArray) = (copyto!(D1.data, D2.data); D1)

__nvalues(D::DeltaArray) = length(D.data)

ndims(D::DeltaArray{T,N}) where {T,N} = N
size(D::DeltaArray) = ntuple(_ -> __nvalues(D), ndims(D))

# TODO put type to i... to be `Integer`?
@inline function getindex(D::DeltaArray, i...)
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