module DeltaArrays

using LinearAlgebra: LinearAlgebra, sym_uplo, AdjointAbsVec, TransposeAbsVec, AbstractTriangular, AbstractVecOrMat, HermOrSym, QRCompactWYQ, QRPackedQ, Diagonal, Symmetric, Hermitian, Tridiagonal, AdjOrTransAbsMat, Adjoint, Transpose, SymTridiagonal, UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular
import Core: Array
import Base: similar, copyto!, size, getindex, setindex!, parent, real, imag, iszero, isone, conj, conj!, adjoint, transpose, permutedims, inv, sum, kron, kron!, @propagate_inbounds, @invoke
import Base: -, +, ==, *, /, \, ^
import LinearAlgebra: ishermitian, issymmetric, isposdef, factorize, isdiag, diag, tr, det, logdet, logabsdet, pinv, eigvals, eigvecs, eigen, svdvals, svd, istriu, istril, triu!, tril!, lmul!, rmul!, ldiv!, rdiv!

export DeltaArray, delta, deltaind

function deltaind(A::AbstractArray)
    Base.require_one_based_indexing(A)
    deltaind(size(A)[1:end-1]...)
end

deltaind(n::Integer...) = range(1, step=sum(cumprod(n), init=1), length=minimum(n))

delta(i::Integer...) = allequal(i)

delta(A::AbstractArray) = A[deltaind(A)]
delta(D::DeltaArray) = D.data

struct DeltaArray{T,N,V<:AbstractVector{T}} <: AbstractArray{T,N}
    data::V

    function DeltaArray{T,N,V}(values) where {T,N,V<:AbstractVector{T}}
        Base.require_one_based_indexing(values)
        new{T,N,V}(values)
    end
end

DeltaArray{T,N,V}(D::DeltaArray) where {T,N,V<:AbstractVector{T}} = DeltaArray{T,N,V}(D.data)

delta(D::DeltaArray) = D.data

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
istriu(D::DeltaArray{<:Any,2}, k::Integer=0) = k <= 0 || iszero(D.data) ? true : false
istril(D::DeltaArray{<:Any,2}, k::Integer=0) = k >= 0 || iszero(D.data) ? true : false

function triu!(D::DeltaArray{T,2}, k::Integer=0) where {T}
    n = size(D, 1)
    if !(-n + 1 <= k <= n + 1)
        throw(ArgumentError("the requested diagonal, $k, must be at least $(-n + 1) and at most $(n + 1) in an $n-by-$n matrix"))
    elseif k > 0
        fill!(D.data, zero(T))
    end
    return D
end

function tril!(D::DeltaArray{T,2}, k::Integer=0) where {T}
    n = size(D, 1)
    if !(-n + 1 <= k <= n + 1)
        throw(ArgumentError("the requested diagonal, $k, must be at least $(-n + 1) and at most $(n + 1) in an $n-by-$n matrix"))
    elseif k < 0
        fill!(D.data, zero(T))
    end
    return D
end

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

function checkmulsize(A, B)
    nA = size(A, 2)
    mB = size(B, 1)
    nA == mB || throw(DimensionMismatch("second dimension of A, $nA, does not match first dimension of B, $mB"))
    return nothing
end

checksizeout(C, ::DeltaArray{<:Any,2}, A) = checksizeout(C, A)
checksizeout(C, A, ::DeltaArray{<:Any,2}) = checksizeout(C, A)
checksizeout(C, A::DeltaArray{<:Any,2}, ::DeltaArray{<:Any,2}) = checksizeout(C, A)
function checksizeout(C, A)
    szA = size(A)
    szC = size(C)
    szA == szC || throw(DimensionMismatch("output matrix has size: $szC, but should have size $szA"))
    return nothing
end

checkmulsize(C, A, B) = (checkmulsize(A, B); checksizeout(C, A, B))

function (*)(Da::DeltaArray{<:Any,2}, Db::DeltaArray{<:Any,2})
    checkmulsize(Da, Db)
    return DeltaArray{2}(Da.data .* Db.data)
end

function (*)(Da::DeltaArray{<:Any,2}, V::AbstractVector)
    checkmulsize(Da, V)
    return D.data .* V
end

(*)(A::AbstractMatrix, D::DeltaArray) = mul!(similar(A, Base.promote_op(*, eltype(A), eltype(D)), size(A)), A, D)
(*)(D::DeltaArray, A::AbstractMatrix) = mul!(similar(A, promote_op(*, eltype(A), eltype(D)), size(A)), D, A)

rmul!(A::AbstractMatrix, D::DeltaArray{<:Any,2}) = @inline mul!(A, A, D)
lmul!(D::DeltaArray{<:Any,2}, B::AbstractVecOrMat) = @inline mul!(B, D, B)

function *(A::AdjOrTransAbsMat, D::DeltaArray{<:Any,2})
    Ac = LinearAlgebra.copy_similar(A, Base.promote_op(*, eltype(A), eltype(D.data)))
    rmul!(Ac, D)
end

*(D::DeltaArray{<:Any,2}, adjQ::Adjoint{<:Any,<:Union{QRCompactWYQ,QRPackedQ}}) =
    rmul!(Array{promote_type(eltype(D), eltype(adjQ))}(D), adjQ)

function *(D::DeltaArray{<:Any,2}, A::AdjOrTransAbsMat)
    Ac = LinearAlgebra.copy_similar(A, Base.promote_op(*, eltype(A), eltype(D.data)))
    lmul!(D, Ac)
end

@inline function __muldiag!(out, D::DeltaArray{<:Any,2}, B, alpha, beta)
    Base.require_one_based_indexing(B)
    Base.require_one_based_indexing(out)
    if iszero(alpha)
        LinearAlgebra._rmul_or_fill!(out, beta)
    else
        if iszero(beta)
            @inbounds for j in axes(B, 2)
                @simd for i in axes(B, 1)
                    out[i, j] = D.data[i] * B[i, j] * alpha
                end
            end
        else
            @inbounds for j in axes(B, 2)
                @simd for i in axes(B, 1)
                    out[i, j] = D.data[i] * B[i, j] * alpha + out[i, j] * beta
                end
            end
        end
    end
    return out
end
@inline function __muldiag!(out, A, D::DeltaArray{<:Any,2}, alpha, beta)
    Base.require_one_based_indexing(A)
    Base.require_one_based_indexing(out)
    if iszero(alpha)
        LinearAlgebra._rmul_or_fill!(out, beta)
    else
        if iszero(beta)
            @inbounds for j in axes(A, 2)
                dja = D.data[j] * alpha
                @simd for i in axes(A, 1)
                    out[i, j] = A[i, j] * dja
                end
            end
        else
            @inbounds for j in axes(A, 2)
                dja = D.data[j] * alpha
                @simd for i in axes(A, 1)
                    out[i, j] = A[i, j] * dja + out[i, j] * beta
                end
            end
        end
    end
    return out
end
@inline function __muldiag!(out::DeltaArray{<:Any,2}, D1::DeltaArray{<:Any,2}, D2::DeltaArray{<:Any,2}, alpha, beta)
    d1 = D1.data
    d2 = D2.data
    if iszero(alpha)
        LinearAlgebra._rmul_or_fill!(out.data, beta)
    else
        if iszero(beta)
            @inbounds @simd for i in eachindex(out.data)
                out.data[i] = d1[i] * d2[i] * alpha
            end
        else
            @inbounds @simd for i in eachindex(out.data)
                out.data[i] = d1[i] * d2[i] * alpha + out.data[i] * beta
            end
        end
    end
    return out
end
@inline function __muldiag!(out, D1::DeltaArray{<:Any,2}, D2::DeltaArray{<:Any,2}, alpha, beta)
    Base.require_one_based_indexing(out)
    mA = size(D1, 1)
    d1 = D1.data
    d2 = D2.data
    LinearAlgebra._rmul_or_fill!(out, beta)
    if !iszero(alpha)
        @inbounds @simd for i in 1:mA
            out[i, i] += d1[i] * d2[i] * alpha
        end
    end
    return out
end

@inline function _muldiag!(out, A, B, alpha, beta)
    checksizeout(out, A, B)
    __muldiag!(out, A, B, alpha, beta)
    return out
end

function (*)(Da::DeltaArray{<:Any,2}, A::AbstractMatrix, Db::DeltaArray{<:Any,2})
    return broadcast(*, Da.data, A, permutedims(Db.data))
end

# Get ambiguous method if try to unify AbstractVector/AbstractMatrix here using AbstractVecOrMat
@inline mul!(out::AbstractVector, D::DeltaArray{<:Any,2}, V::AbstractVector, alpha::Number, beta::Number) = _muldiag!(out, D, V, alpha, beta)
@inline mul!(out::AbstractMatrix, D::DeltaArray{<:Any,2}, B::AbstractMatrix, alpha::Number, beta::Number) = _muldiag!(out, D, B, alpha, beta)
@inline mul!(out::AbstractMatrix, D::DeltaArray{<:Any,2}, B::Adjoint{<:Any,<:AbstractVecOrMat}, alpha::Number, beta::Number) = _muldiag!(out, D, B, alpha, beta)
@inline mul!(out::AbstractMatrix, D::DeltaArray{<:Any,2}, B::Transpose{<:Any,<:AbstractVecOrMat}, alpha::Number, beta::Number) = _muldiag!(out, D, B, alpha, beta)

@inline mul!(out::AbstractMatrix, A::AbstractMatrix, D::DeltaArray{<:Any,2}, alpha::Number, beta::Number) = _muldiag!(out, A, D, alpha, beta)
@inline mul!(out::AbstractMatrix, A::Adjoint{<:Any,<:AbstractVecOrMat}, D::DeltaArray{<:Any,2}, alpha::Number, beta::Number) = _muldiag!(out, A, D, alpha, beta)
@inline mul!(out::AbstractMatrix, A::Transpose{<:Any,<:AbstractVecOrMat}, D::DeltaArray{<:Any,2}, alpha::Number, beta::Number) = _muldiag!(out, A, D, alpha, beta)
@inline mul!(C::DeltaArray{<:Any,2}, Da::DeltaArray{<:Any,2}, Db::DeltaArray{<:Any,2}, alpha::Number, beta::Number) = _muldiag!(C, Da, Db, alpha, beta)

mul!(C::AbstractMatrix, Da::DeltaArray{<:Any,2}, Db::DeltaArray{<:Any,2}, alpha::Number, beta::Number) = _muldiag!(C, Da, Db, alpha, beta)

/(A::AbstractVecOrMat, D::DeltaArray{<:Any,2}) = _rdiv!(similar(A, LinearAlgebra._init_eltype(/, eltype(A), eltype(D))), A, D)
/(A::HermOrSym, D::DeltaArray{<:Any,2}) = _rdiv!(similar(A, LinearAlgebra._init_eltype(/, eltype(A), eltype(D)), size(A)), A, D)
rdiv!(A::AbstractVecOrMat, D::DeltaArray{<:Any,2}) = @inline _rdiv!(A, A, D)

# avoid copy when possible via internal 3-arg backend
function _rdiv!(B::AbstractVecOrMat, A::AbstractVecOrMat, D::DeltaArray{<:Any,2})
    require_one_based_indexing(A)
    dd = D.data
    m, n = size(A, 1), size(A, 2)
    if (k = length(dd)) != n
        throw(DimensionMismatch("left hand side has $n columns but D is $k by $k"))
    end
    @inbounds for j in 1:n
        ddj = dd[j]
        iszero(ddj) && throw(SingularException(j))
        for i in 1:m
            B[i, j] = A[i, j] / ddj
        end
    end
    B
end

function (\)(D::DeltaArray{<:Any,2}, B::AbstractVector)
    j = findfirst(iszero, D.data)
    isnothing(j) || throw(SingularException(j))
    return D.data .\ B
end

\(D::DeltaArray{<:Any,2}, B::AbstractMatrix) = ldiv!(similar(B, LinearAlgebra._init_eltype(\, eltype(D), eltype(B))), D, B)
\(D::DeltaArray{<:Any,2}, B::HermOrSym) = ldiv!(similar(B, LinearAlgebra._init_eltype(\, eltype(D), eltype(B)), size(B)), D, B)

ldiv!(D::DeltaArray{<:Any,2}, B::AbstractVecOrMat) = @inline ldiv!(B, D, B)
function ldiv!(B::AbstractVecOrMat, D::DeltaArray{<:Any,2}, A::AbstractVecOrMat)
    require_one_based_indexing(A, B)
    dd = D.data
    d = length(dd)
    m, n = size(A, 1), size(A, 2)
    m′, n′ = size(B, 1), size(B, 2)
    m == d || throw(DimensionMismatch("right hand side has $m rows but D is $d by $d"))
    (m, n) == (m′, n′) || throw(DimensionMismatch("expect output to be $m by $n, but got $m′ by $n′"))
    j = findfirst(iszero, D.data)
    isnothing(j) || throw(SingularException(j))
    @inbounds for j = 1:n, i = 1:m
        B[i, j] = dd[i] \ A[i, j]
    end
    B
end

# Optimizations for \, / between DeltaArrays
\(D::DeltaArray{<:Any,2}, B::DeltaArray{<:Any,2}) = ldiv!(similar(B, Base.promote_op(\, eltype(D), eltype(B))), D, B)
/(A::DeltaArray{<:Any,2}, D::DeltaArray{<:Any,2}) = _rdiv!(similar(A, Base.promote_op(/, eltype(A), eltype(D))), A, D)
function _rdiv!(Dc::DeltaArray{<:Any,2}, Db::DeltaArray{<:Any,2}, Da::DeltaArray{<:Any,2})
    n, k = length(Db.data), length(Da.data)
    n == k || throw(DimensionMismatch("left hand side has $n columns but D is $k by $k"))
    j = findfirst(iszero, Da.data)
    isnothing(j) || throw(SingularException(j))
    Dc.data .= Db.data ./ Da.data
    Dc
end
ldiv!(Dc::DeltaArray{<:Any,2}, Da::DeltaArray{<:Any,2}, Db::DeltaArray{<:Any,2}) = DeltaArray{2}(ldiv!(Dc.data, Da, Db.data))

# optimizations for (Sym)Tridiagonal and DeltaArray
@propagate_inbounds _getudiag(T::Tridiagonal, i) = T.du[i]
@propagate_inbounds _getudiag(S::SymTridiagonal, i) = S.ev[i]
@propagate_inbounds _getdiag(T::Tridiagonal, i) = T.d[i]
@propagate_inbounds _getdiag(S::SymTridiagonal, i) = symmetric(S.dv[i], :U)::LinearAlgebra.symmetric_type(eltype(S.dv))
@propagate_inbounds _getldiag(T::Tridiagonal, i) = T.dl[i]
@propagate_inbounds _getldiag(S::SymTridiagonal, i) = transpose(S.ev[i])

function (\)(D::DeltaArray{<:Any,2}, S::SymTridiagonal)
    T = Base.promote_op(\, eltype(D), eltype(S))
    du = similar(S.ev, T, max(length(S.dv) - 1, 0))
    d = similar(S.dv, T, length(S.dv))
    dl = similar(S.ev, T, max(length(S.dv) - 1, 0))
    ldiv!(Tridiagonal(dl, d, du), D, S)
end
(\)(D::DeltaArray{<:Any,2}, T::Tridiagonal) = ldiv!(similar(T, Base.promote_op(\, eltype(D), eltype(T))), D, T)
function ldiv!(T::Tridiagonal, D::DeltaArray{<:Any,2}, S::Union{SymTridiagonal,Tridiagonal})
    m = size(S, 1)
    dd = D.data
    if (k = length(dd)) != m
        throw(DimensionMismatch("diagonal matrix is $k by $k but right hand side has $m rows"))
    end
    if length(T.d) != m
        throw(DimensionMismatch("target matrix size $(size(T)) does not match input matrix size $(size(S))"))
    end
    m == 0 && return T
    j = findfirst(iszero, dd)
    isnothing(j) || throw(SingularException(j))
    ddj = dd[1]
    T.d[1] = ddj \ _getdiag(S, 1)
    @inbounds if m > 1
        T.du[1] = ddj \ _getudiag(S, 1)
        for j in 2:m-1
            ddj = dd[j]
            T.dl[j-1] = ddj \ _getldiag(S, j - 1)
            T.d[j] = ddj \ _getdiag(S, j)
            T.du[j] = ddj \ _getudiag(S, j)
        end
        ddj = dd[m]
        T.dl[m-1] = ddj \ _getldiag(S, m - 1)
        T.d[m] = ddj \ _getdiag(S, m)
    end
    return T
end

function (/)(S::SymTridiagonal, D::DeltaArray{<:Any,2})
    T = Base.promote_op(\, eltype(D), eltype(S))
    du = similar(S.ev, T, max(length(S.dv) - 1, 0))
    d = similar(S.dv, T, length(S.dv))
    dl = similar(S.ev, T, max(length(S.dv) - 1, 0))
    _rdiv!(Tridiagonal(dl, d, du), S, D)
end
(/)(T::Tridiagonal, D::DeltaArray{<:Any,2}) = _rdiv!(similar(T, Base.promote_op(/, eltype(T), eltype(D))), T, D)
function _rdiv!(T::Tridiagonal, S::Union{SymTridiagonal,Tridiagonal}, D::DeltaArray{<:Any,2})
    n = size(S, 2)
    dd = D.data
    if (k = length(dd)) != n
        throw(DimensionMismatch("left hand side has $n columns but D is $k by $k"))
    end
    if length(T.d) != n
        throw(DimensionMismatch("target matrix size $(size(T)) does not match input matrix size $(size(S))"))
    end
    n == 0 && return T
    j = findfirst(iszero, dd)
    isnothing(j) || throw(SingularException(j))
    ddj = dd[1]
    T.d[1] = _getdiag(S, 1) / ddj
    @inbounds if n > 1
        T.dl[1] = _getldiag(S, 1) / ddj
        for j in 2:n-1
            ddj = dd[j]
            T.dl[j] = _getldiag(S, j) / ddj
            T.d[j] = _getdiag(S, j) / ddj
            T.du[j-1] = _getudiag(S, j - 1) / ddj
        end
        ddj = dd[n]
        T.d[n] = _getdiag(S, n) / ddj
        T.du[n-1] = _getudiag(S, n - 1) / ddj
    end
    return T
end

# Optimizations for [l/r]mul!, l/rdiv!, *, / and \ between Triangular and DeltaArray.
# These functions are generally more efficient if we calculate the whole data field.
# The following code implements them in a unified pattern to avoid missing.
@inline function _setdiag!(data, f, diag, diag′=nothing)
    @inbounds for i in 1:length(diag)
        data[i, i] = isnothing(diag′) ? f(diag[i]) : f(diag[i], diag′[i])
    end
    data
end
for Tri in (:UpperTriangular, :LowerTriangular)
    UTri = Symbol(:Unit, Tri)
    # 2 args
    for (fun, f) in zip((:*, :rmul!, :rdiv!, :/), (:identity, :identity, :inv, :inv))
        @eval $fun(A::$Tri, D::DeltaArray{<:Any,2}) = $Tri($fun(A.data, D))
        @eval $fun(A::$UTri, D::DeltaArray{<:Any,2}) = $Tri(_setdiag!($fun(A.data, D), $f, D.data))
    end
    for (fun, f) in zip((:*, :lmul!, :ldiv!, :\), (:identity, :identity, :inv, :inv))
        @eval $fun(D::DeltaArray{<:Any,2}, A::$Tri) = $Tri($fun(D, A.data))
        @eval $fun(D::DeltaArray{<:Any,2}, A::$UTri) = $Tri(_setdiag!($fun(D, A.data), $f, D.data))
    end
    # 3-arg ldiv!
    @eval ldiv!(C::$Tri, D::DeltaArray{<:Any,2}, A::$Tri) = $Tri(ldiv!(C.data, D, A.data))
    @eval ldiv!(C::$Tri, D::DeltaArray{<:Any,2}, A::$UTri) = $Tri(_setdiag!(ldiv!(C.data, D, A.data), inv, D.data))
    # 3-arg mul!: invoke 5-arg mul! rather than lmul!
    @eval mul!(C::$Tri, A::Union{$Tri,$UTri}, D::DeltaArray{<:Any,2}) = mul!(C, A, D, true, false)
    # 5-arg mul!
    @eval @inline mul!(C::$Tri, D::DeltaArray{<:Any,2}, A::$Tri, α::Number, β::Number) = $Tri(mul!(C.data, D, A.data, α, β))
    @eval @inline function mul!(C::$Tri, D::DeltaArray{<:Any,2}, A::$UTri, α::Number, β::Number)
        iszero(α) && return LinearAlgebra._rmul_or_fill!(C, β)
        diag′ = iszero(β) ? nothing : diag(C)
        data = mul!(C.data, D, A.data, α, β)
        $Tri(_setdiag!(data, MulAddMul(α, β), D.data, diag′))
    end
    @eval @inline mul!(C::$Tri, A::$Tri, D::DeltaArray{<:Any,2}, α::Number, β::Number) = $Tri(mul!(C.data, A.data, D, α, β))
    @eval @inline function mul!(C::$Tri, A::$UTri, D::DeltaArray{<:Any,2}, α::Number, β::Number)
        iszero(α) && return LinearAlgebra._rmul_or_fill!(C, β)
        diag′ = iszero(β) ? nothing : diag(C)
        data = mul!(C.data, A.data, D, α, β)
        $Tri(_setdiag!(data, MulAddMul(α, β), D.data, diag′))
    end
end

kron(A::DeltaArray{<:Any,N}, B::DeltaArray{<:Any,M}) where {N,M} = DeltaArray{N + M}(kron(A.data, B.data))

function kron(A::DeltaArray{<:Any,2}, B::SymTridiagonal)
    kdv = kron(delta(A), B.dv)
    # We don't need to drop the last element
    kev = kron(delta(A), LinearAlgebra._pushzero(LinearAlgebra._evview(B)))
    SymTridiagonal(kdv, kev)
end

function kron(A::DeltaArray{<:Any,2}, B::Tridiagonal)
    # `_droplast!` is only guaranteed to work with `Vector`
    kd = LinearAlgebra._makevector(kron(delta(A), B.d))
    kdl = LinearAlgebra._droplast!(LinearAlgebra._makevector(kron(delta(A), LinearAlgebra._pushzero(B.dl))))
    kdu = LinearAlgebra._droplast!(LinearAlgebra._makevector(kron(delta(A), LinearAlgebra._pushzero(B.du))))
    Tridiagonal(kdl, kd, kdu)
end

@inline function kron!(C::AbstractMatrix, A::DeltaArray{<:Any,2}, B::AbstractMatrix)
    require_one_based_indexing(B)
    (mA, nA) = size(A)
    (mB, nB) = size(B)
    (mC, nC) = size(C)
    @boundscheck (mC, nC) == (mA * mB, nA * nB) ||
                 throw(DimensionMismatch("expect C to be a $(mA * mB)x$(nA * nB) matrix, got size $(mC)x$(nC)"))
    isempty(A) || isempty(B) || fill!(C, zero(A[1, 1] * B[1, 1]))
    m = 1
    @inbounds for j = 1:nA
        A_jj = A[j, j]
        for k = 1:nB
            for l = 1:mB
                C[m] = A_jj * B[l, k]
                m += 1
            end
            m += (nA - 1) * mB
        end
        m += mB
    end
    return C
end

@inline function kron!(C::AbstractMatrix, A::AbstractMatrix, B::DeltaArray{<:Any})
    require_one_based_indexing(A)
    (mA, nA) = size(A)
    (mB, nB) = size(B)
    (mC, nC) = size(C)
    @boundscheck (mC, nC) == (mA * mB, nA * nB) ||
                 throw(DimensionMismatch("expect C to be a $(mA * mB)x$(nA * nB) matrix, got size $(mC)x$(nC)"))
    isempty(A) || isempty(B) || fill!(C, zero(A[1, 1] * B[1, 1]))
    m = 1
    @inbounds for j = 1:nA
        for l = 1:mB
            Bll = B[l, l]
            for k = 1:mA
                C[m] = A[k, j] * Bll
                m += nB
            end
            m += 1
        end
        m -= nB
    end
    return C
end

conj(D::DeltaArray{<:Any,N}) where {N} = DeltaArray{N}(conj(D.data))
conj!(D::DeltaArray) = conj!(D.data)
transpose(D::DeltaArray{<:Number}) = D
transpose(D::DeltaArray{<:Any,N}) where {N} = DeltaArray{N}(transpose.(D.data))
adjoint(D::DeltaArray{<:Number}) = conj(D)
adjoint(D::DeltaArray{<:Any,N}) where {N} = DeltaArray{N}(adjoint.(D.data))
permutedims(D::DeltaArray) = D
permutedims(D::DeltaArray, perm) = (Base.checkdims_perm(D, D, perm); D)

function diag(D::DeltaArray{T,2}, k::Integer=0) where {T}
    # every branch call similar(..., ::Int) to make sure the
    # same vector type is returned independent of k
    if k == 0
        return copyto!(similar(D.data, length(D.data)), D.data)
    elseif -size(D, 1) <= k <= size(D, 1)
        return fill!(similar(D.data, size(D, 1) - abs(k)), zero(T))
    else
        throw(ArgumentError("requested diagonal, $k, must be at least $(-size(D, 1)) and at most $(size(D, 2)) for an $(size(D, 1))-by-$(size(D, 2)) matrix"))
    end
end

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
    for i in 1:length(D.data)
        if iszero(D.data[i])
            throw(SingularException(i))
        end
        Di[i] = inv(D.data[i])
    end
    DeltaArray{N}(Di)
end

function pinv(D::DeltaArray{T,N}) where {T,N}
    Di = similar(D.data, typeof(inv(oneunit(T))))
    for i in 1:length(D.data)
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
        for i in 1:length(D.data)
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

# disambiguation methods: * and / of DeltaArray{<:Any,2} and Adj/Trans AbsVec
*(u::AdjointAbsVec, D::DeltaArray{<:Any,2}) = (D'u')'
*(u::TransposeAbsVec, D::DeltaArray{<:Any,2}) = transpose(transpose(D) * transpose(u))
*(x::AdjointAbsVec, D::DeltaArray{<:Any,2}, y::AbstractVector) = _mapreduce_prod(*, x, D, y)
*(x::TransposeAbsVec, D::DeltaArray{<:Any,2}, y::AbstractVector) = _mapreduce_prod(*, x, D, y)
/(u::AdjointAbsVec, D::DeltaArray{<:Any,2}) = (D' \ u')'
/(u::TransposeAbsVec, D::DeltaArray{<:Any,2}) = transpose(transpose(D) \ transpose(u))
# disambiguation methods: Call unoptimized version for user defined AbstractTriangular.
*(A::AbstractTriangular, D::DeltaArray{<:Any,2}) = @invoke *(A::AbstractMatrix, D::DeltaArray{<:Any,2})
*(D::DeltaArray{<:Any,2}, A::AbstractTriangular) = @invoke *(D::DeltaArray{<:Any,2}, A::AbstractMatrix)

dot(A::DeltaArray{<:Any,2}, B::DeltaArray{<:Any,2}) = dot(A.data, B.data)
function dot(D::DeltaArray{<:Any,2}, B::AbstractMatrix)
    size(D) == size(B) || throw(DimensionMismatch("Matrix sizes $(size(D)) and $(size(B)) differ"))
    return dot(D.data, view(B, deltaind(B)))
end
dot(A::AbstractMatrix, B::DeltaArray{<:Any,2}) = conj(dot(B, A))

function _mapreduce_prod(f, x, D::DeltaArray{<:Any,2}, y)
    if !(length(x) == length(D.data) == length(y))
        throw(DimensionMismatch("x has length $(length(x)), D has size $(size(D)), and y has $(length(y))"))
    end
    if isempty(x) && isempty(D) && isempty(y)
        return zero(Base.promote_op(f, eltype(x), eltype(D), eltype(y)))
    else
        return mapreduce(t -> f(t[1], t[2], t[3]), +, zip(x, D.data, y))
    end
end

# TODO cholesky

sum(A::DeltaArray) = sum(A.data)
sum(A::DeltaArray{<:Any,N}, dims::Integer) where {N} = N <= 1 ? sum(A.data) : DeltaArray{N - 1}(A.data)


function Base.muladd(A::DeltaArray{<:Any,2}, B::DeltaArray{<:Any,2}, z::DeltaArray{<:Any,2})
    DeltaArray{2}(A.data .* B.data .+ z.data)
end

end