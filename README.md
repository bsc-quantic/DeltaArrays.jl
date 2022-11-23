# DeltaArrays.jl

This Julia library provides `DeltaArray`, an efficient N-dimensional `Diagonal` array type. If your array $A$ is of the form

$$
A = a_i \delta_{i \dots j} = \begin{cases}
	a_i, &\text{if} ~~ i=\dots=j \\
	0, &\text{otherwise}
\end{cases}
$$

then it can be represented by a `DeltaArray`.

For compatibility, `DeltaArrays{T,2}` should just behave like `Diagonal{T}`.