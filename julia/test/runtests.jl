# runtests.jl -- Test suite for SU2FFT.jl.
#
# Mirrors the C-side test in tests/test_fft.c: the gold standard is that the
# fast O(N^4) path agrees with the O(N^6) direct reference to ~1e-10 on a
# random ComplexF64 input. Plus layout invariants from DESIGN.md §6.

using SU2FFT
using Test
using LinearAlgebra
using Random

@testset "SU2FFT" begin

    @testset "coefficient counts" begin
        # total_coeffs(N) = sum_{l=0..N-1} (2l+1)^2
        for N in (1, 2, 4, 8)
            expected = sum((2l + 1)^2 for l in 0:N-1)
            @test SU2FFT.total_coeffs(N) == expected
        end
        # Hand-computed: N=4 -> 1 + 9 + 25 + 49 = 84.
        @test SU2FFT.total_coeffs(4) == 84
    end

    @testset "coeff_offset consistency" begin
        @test SU2FFT.coeff_offset(0) == 0
        # coeff_offset(l+1) - coeff_offset(l) == (2l+1)^2
        for l in 0:4
            @test SU2FFT.coeff_offset(l + 1) - SU2FFT.coeff_offset(l) == (2l + 1)^2
        end
    end

    @testset "wigner_d invariants" begin
        # P^0_{0,0}(cos theta) = 1 for all theta (degree-0 trivial rep).
        for theta in (0.0, 0.1, 0.5, 1.5, pi - 0.1)
            @test SU2FFT.wigner_d(0, 0, 0, theta) ≈ 1.0 + 0.0im atol=1e-12
        end
        # P^l_{n,m}(cos 0) = delta_{n,m} (identity rotation).
        for l in 0:3, n in -l:l, m in -l:l
            expected = (n == m) ? complex(1.0) : complex(0.0)
            @test SU2FFT.wigner_d(l, n, m, 0.0) ≈ expected atol=1e-12
        end
    end

    @testset "fft on zero input" begin
        N = 4
        f = zeros(ComplexF64, N, N, N)
        fhat = SU2FFT.fft(f)
        @test length(fhat) == SU2FFT.total_coeffs(N)
        @test maximum(abs, fhat) == 0.0
    end

    @testset "fft matches ft_direct on random input (gold-standard)" begin
        # Mirror of tests/test_fft.c::test_fft_matches_direct_random.
        # The two implementations compute the SAME discrete sum and must agree
        # to floating-point noise propagated through O(N^3) ops -- 1e-10
        # tolerance per the C test.
        Random.seed!(20260526)
        for N in (6, 8)
            f = (rand(ComplexF64, N, N, N) .- (0.5 + 0.5im))
            fhat_fast   = SU2FFT.fft(f)
            fhat_direct = SU2FFT.ft_direct(f)
            @test length(fhat_fast) == length(fhat_direct) == SU2FFT.total_coeffs(N)
            # Element-wise tolerance, matching the C-side ASSERT_CNEAR.
            @test maximum(abs, fhat_fast .- fhat_direct) < 1e-10
        end
    end

    @testset "fhat_at accessor" begin
        N = 4
        Random.seed!(1)
        f = rand(ComplexF64, N, N, N)
        fhat = SU2FFT.fft(f)
        # fhat_at(fhat, l, m, n) must equal the raw flat-index value at
        # coeff_offset(l) + (m+l)*(2l+1) + (n+l) + 1.
        for l in 0:N-1, m in -l:l, n in -l:l
            off = SU2FFT.coeff_offset(l)
            expected = fhat[off + (m + l) * (2l + 1) + (n + l) + 1]
            @test SU2FFT.fhat_at(fhat, l, m, n) == expected
        end
        # Bounds checks.
        @test_throws ArgumentError SU2FFT.fhat_at(fhat, 1, 2, 0)
        @test_throws ArgumentError SU2FFT.fhat_at(fhat, 1, 0, -2)
    end

    @testset "fhat_block view" begin
        N = 4
        Random.seed!(2)
        f = rand(ComplexF64, N, N, N)
        fhat = SU2FFT.fft(f)
        # Block view dimensions.
        for l in 0:N-1
            blk = SU2FFT.fhat_block(fhat, l)
            @test size(blk) == (2l + 1, 2l + 1)
        end
        # Modifying the underlying vector through the view should be a no-op
        # since reshape(view(...)) shares storage. Sanity check.
        blk0 = SU2FFT.fhat_block(fhat, 0)
        @test blk0[1, 1] == fhat[1]
    end

    @testset "input validation" begin
        # Non-cubic input rejected.
        @test_throws ArgumentError SU2FFT.fft(zeros(ComplexF64, 4, 4, 5))
        # N < 2 rejected.
        @test_throws ArgumentError SU2FFT.fft(zeros(ComplexF64, 1, 1, 1))
    end

end
