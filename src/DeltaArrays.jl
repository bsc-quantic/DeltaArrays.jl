module DeltaArrays

using LinearAlgebra
using LinearAlgebra: sym_uplo
import Core: Array
import Base: similar, copyto!, size, getindex, setindex!, parent, real, imag, iszero, isone, conj, adjoint, transpose, permutedims, inv, sum
import Base: -, +, ==, *, /, \, ^
import LinearAlgebra: ishermitian, issymmetric, isposdef, factorize, isdiag, tr, det, logdet, logabsdet, pinv, eigvals, eigvecs, eigen, svdvals, svd

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
DeltaArray(V::AbstractVector)

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
DeltaArray(A::AbstractArray{<:Any,N}) where {N} = DeltaArray{N}(delta(A))

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

size(D::DeltaArray{<:Any,N}) where {N} = ntuple(_ -> __nvalues(D), N)

# TODO put type to i... to be `Integer`?
@inline function getindex(D::DeltaArray, i::Int...)
    @boundscheck checkbounds(D, i...)
    if allequal(i)
        @inbounds r = D.data[first(i)]
    else
        r = deltazero(D, i...)
    end
    r
end
deltazero(::DeltaArray{T}, i...) where {T} = zero(T)
deltazero(D::DeltaArray{<:AbstractArray{T,N}}, i...) where {T,N} = zeros(T, (size(D.data[j], n) for (j, n) in zip(i, 1:N))...)

# TODO put type to i... to be `Integer`?
function setindex!(D::DeltaArray, v, i::Int...)
    @boundscheck checkbounds(D, i...)
    if allequal(i)
        @inbounds D.data[first(i)] = v
    elseif !iszero(v)
        throw(ArgumentError("cannot set off-diagonal entry ($(i...)) to a nonzero value ($v)"))
    end
    return v
end

# NOTE not working/used currently
## structured matrix methods ##
# function Base.replace_in_print_matrix(D::DeltaArray{<:Any,2}, i::Integer, j::Integer, s::AbstractString)
#     allequal(i) ? s : Base.replace_with_centered_mark(s)
# end

parent(D::DeltaArray) = D.data

# NOTE `DeltaArrays` are always symmetric because they are invariant under permutations of its dims
ishermitian(D::DeltaArray{<:Real}) = true
ishermitian(D::DeltaArray{<:Number}) = isreal(D.data)
ishermitian(D::DeltaArray) = all(ishermitian, D.data)
issymmetric(D::DeltaArray{<:Number}) = true
issymmetric(D::DeltaArray) = all(issymmetric, D.data)
isposdef(D::DeltaArray) = all(isposdef, D.data)

factorize(D::DeltaArray) = D

real(D::DeltaArray{<:Any,N}) where {N} = DeltaArray{N}(real(D.data))
imag(D::DeltaArray{<:Any,N}) where {N} = DeltaArray{N}(imag(D.data))

iszero(D::DeltaArray) = all(iszero, D.data)
isone(D::DeltaArray) = all(isone, D.data)
isdiag(D::DeltaArray) = all(isdiag, D.data)
isdiag(D::DeltaArray{<:Number}) = true
# TODO istriu, istril, triu!, tril!

# NOTE the following method is not well defined and is susceptible for change
function (==)(Da::DeltaArray, Db::DeltaArray)
    @boundscheck ndims(Da) != ndims(Db) && throw(DimensionMismatch("a has dims $(ndims(Da)) and b has $(ndims(Db))"))
    return Da.data == Db.data
end

(-)(D::DeltaArray{<:Any,N}) where {N} = DeltaArray{N}(-D.data)

# NOTE the following method is not well defined and is susceptible for change
function (+)(Da::DeltaArray, Db::DeltaArray)
    @boundscheck ndims(Da) != ndims(Db) && throw(DimensionMismatch("a has dims $(ndims(Da)) and b has $(ndims(Db))"))
    return DeltaArray{ndims(Da)}(Da.data + Db.data)
end

# NOTE the following method is not well defined and is susceptible for change
function (-)(Da::DeltaArray, Db::DeltaArray)
    @boundscheck ndims(Da) != ndims(Db) && throw(DimensionMismatch("a has dims $(ndims(Da)) and b has $(ndims(Db))"))
    return DeltaArray{ndims(Da)}(Da.data - Db.data)
end

for f in (:+, :-)
    @eval function $f(D::DeltaArray{<:Any,2}, S::Symmetric)
        return Symmetric($f(D, S.data), sym_uplo(S.uplo))
    end
    @eval function $f(S::Symmetric, D::DeltaArray{<:Any,2})
        return Symmetric($f(S.data, D), sym_uplo(S.uplo))
    end
    @eval function $f(D::DeltaArray{<:Real,2}, H::Hermitian)
        return Hermitian($f(D, H.data), sym_uplo(H.uplo))
    end
    @eval function $f(H::Hermitian, D::DeltaArray{<:Real,2})
        return Hermitian($f(H.data, D), sym_uplo(H.uplo))
    end
end

(*)(x::Number, D::DeltaArray{<:Any,N}) where {N} = DeltaArray{N}(x * D.data)
(*)(D::DeltaArray{<:Any,N}, x::Number) where {N} = DeltaArray{N}(D.data * x)
(/)(D::DeltaArray{<:Any,N}, x::Number) where {N} = DeltaArray{N}(D.data / x)
(\)(x::Number, D::DeltaArray{<:Any,N}) where {N} = DeltaArray{N}(x \ D.data)
(^)(D::DeltaArray{<:Any,N}, a::Number) where {N} = DeltaArray{N}(D.data .^ a)
(^)(D::DeltaArray{<:Any,N}, a::Real) where {N} = DeltaArray{N}(D.data .^ a) # for disambiguation
(^)(D::DeltaArray{<:Any,N}, a::Integer) where {N} = DeltaArray{N}(D.data .^ a) # for disambiguation
Base.literal_pow(::typeof(^), D::DeltaArray{<:Any,N}, valp::Val) where {N} = DeltaArray{N}(Base.literal_pow.(^, D.data, valp)) # for speed
Base.literal_pow(::typeof(^), D::DeltaArray, valp::Val{-1}) = inv(D) # for disambiguation

# TODO ...

conj(D::DeltaArray{<:Any,N}) where {N} = DeltaArray{N}(conj(D.data))
transpose(D::DeltaArray{<:Number}) = D
transpose(D::DeltaArray{<:Any,N}) where {N} = DeltaArray{N}(transpose.(D.data))
adjoint(D::DeltaArray{<:Number}) = conj(D)
adjoint(D::DeltaArray{<:Any,N}) where {N} = DeltaArray{N}(adjoint.(D.data))
permutedims(D::DeltaArray) = D
permutedims(D::DeltaArray, perm) = (Base.checkdims_perm(D, D, perm); D)

# TODO diag

tr(D::DeltaArray) = sum(tr, D.data)
det(D::DeltaArray) = prod(det, D.data)
function logdet(D::DeltaArray{<:Complex})
    z = sum(log, D.data)
    complex(real(z), rem2pi(imag(z), RoundNearest))
end
function logabsdet(A::DeltaArray)
    mapreduce(x -> (log(abs(x)), sign(x)), ((d1, s1), (d2, s2)) -> (d1 + d2, s1 * s2), A.data)
end

for f in (:exp, :cis, :log, :sqrt,
    :cos, :sin, :tan, :csc, :sec, :cot,
    :cosh, :sinh, :tanh, :csch, :sech, :coth,
    :acos, :asin, :atan, :acsc, :asec, :acot,
    :acosh, :asinh, :atanh, :acsch, :asech, :acoth)
    @eval Base.$f(D::DeltaArray{<:Any,N}) where {N} = DeltaArray{N}($f.(D.data))
end

function inv(D::DeltaArray{T,N}) where {T,N}
    Di = similar(D.data, typeof(inv(oneunit(T))))
    for i = 1:length(D.data)
        if iszero(D.data[i])
            throw(SingularException(i))
        end
        Di[i] = inv(D.data[i])
    end
    DeltaArray{N}(Di)
end

function pinv(D::DeltaArray{T,N}) where {T,N}
    Di = similar(D.data, typeof(inv(oneunit(T))))
    for i = 1:length(D.data)
        if !iszero(D.data[i])
            invD = inv(D.data[i])
            if isfinite(invD)
                Di[i] = invD
                continue
            end
        end
        # fallback
        Di[i] = zero(T)
    end
    DeltaArray{N}(Di)
end

function pinv(D::DeltaArray{T,N}, tol::Real) where {T,N}
    Di = similar(D.data, typeof(inv(oneunit(T))))
    if !isempty(D.data)
        maxabsD = maximum(abs, D.data)
        for i = 1:length(D.data)
            if abs(D.data[i]) > tol * maxabsD
                invD = inv(D.data[i])
                if isfinite(invD)
                    Di[i] = invD
                    continue
                end
            end
            # fallback
            Di[i] = zero(T)
        end
    end
    DeltaArray{N}(Di)
end

# Eigensystem
eigvals(D::DeltaArray{<:Number,2}; permute::Bool=true, scale::Bool=true) = copy(D.data)
eigvals(D::DeltaArray{<:Any,2}; permute::Bool=true, scale::Bool=true) = eigvals.(D.data)
eigvecs(D::DeltaArray{<:Any,2}) = Matrix{eltype(D)}(I, size(D))

function eigen(D::DeltaArray{<:Any,2}; permute::Bool=true, scale::Bool=true, sortby::Union{Function,Nothing}=nothing)
    if any(!isfinite, D.data)
        throw(ArgumentError("matrix contains Infs or NaNs"))
    end

    Td = Base.promote_op(/, eltype(D), eltype(D))
    λ = eigvals(D)
    if !isnothing(sortby)
        p = sortperm(λ; alg=QuickSort, by=sortby)
        λ = λ[p]
    end
    evecs = Matrix{Td}(I, size(D))
    Eigen(λ, evecs)
end

function eigen(Da::DeltaArray{<:Any,2}, Db::DeltaArray{<:Any,2}; sortby::Union{Function,Nothing}=nothing)
    if any(!isfinite, Da.data) || any(!isfinite, Db.data)
        throw(ArgumentError("matrices contain Infs or NaNs"))
    end
    if any(iszero, Db.data)
        throw(ArgumentError("right-hand side diagonal matrix is singular"))
    end
    return GeneralizedEigen(eigen(Db \ Da; sortby)...)
end

function eigen(A::AbstractMatrix, D::DeltaArray{<:Any,2}; sortby::Union{Function,Nothing}=nothing)
    if any(iszero, D.data)
        throw(ArgumentError("right-hand side diagonal matrix is singular"))
    end
    if size(A, 1) == size(A, 2) && isdiag(A)
        return eigen(DeltaArray(A), D; sortby)
    elseif ishermitian(A)
        S = promote_type(LinearAlgebra.eigtype(eltype(A)), eltype(D))
        return eigen!(eigencopy_oftype(Hermitian(A), S), DeltaArray{S,2}(D); sortby)
    else
        S = promote_type(LinearAlgebra.eigtype(eltype(A)), eltype(D))
        return eigen!(eigencopy_oftype(A, S), DeltaArray{S,2}(D); sortby)
    end
end

# Singular system
svdvals(D::DeltaArray{<:Number,2}) = sort!(abs.(D.data), rev=true)
svdvals(D::DeltaArray{<:Any,2}) = [svdvals(v) for v in D.data]

function svd(D::DeltaArray{T,2}) where {T<:Number}
    d = D.data
    s = abs.(d)
    piv = sortperm(s, rev=true)
    S = s[piv]
    Td = typeof(oneunit(T) / oneunit(T))
    U = zeros(Td, size(D))
    Vt = copy(U)
    for i in 1:length(d)
        j = piv[i]
        U[j, i] = d[j] / S[i]
        Vt[i, j] = one(Td)
    end
    return SVD(U, S, Vt)
end

sum(A::DeltaArray) = sum(A.data)
sum(A::DeltaArray{<:Any,N}, dims::Integer) where {N} = N <= 1 ? sum(A.data) : DeltaArray{N - 1}(A.data)



end