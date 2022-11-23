using DeltaArrays

@testset "isinteger and isreal" begin
    @test all(isinteger, DeltaArray(rand(1:5, 5)))
    @test isreal(DeltaArray(rand(5)))
end

@testset "unary ops" begin
    let A = DeltaArray(rand(1:5, 5))
        @test +(A) == A
        @test *(A) == A
    end
end

@testset "reverse dim on empty" begin
    @test reverse(DeltaArray([]), dims=1) == DeltaArray([])
end

@testset "ndims and friends" begin
    @test ndims(DeltaArray(rand(1:5, 5))) == 2
    @test_skip ndims(DeltaArray{Float64}) == 2
    @test_skip ndims(DeltaArray) == 2
end