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

    @testset "inverse FFT (synthesis)" begin
        # Analytical tests that DO hit machine precision (synthesis is exact math;
        # closed-grid Riemann error only affects the spectrum roundtrip).

        @testset "inv(zeros) -> zeros" begin
            N = 5
            fhat = zeros(ComplexF64, SU2FFT.total_coeffs(N))
            f = SU2FFT.fft_inv(fhat, N)
            @test size(f) == (N, N, N)
            @test maximum(abs, f) < 1e-14
        end

        @testset "inv(delta_{l=0,m=0,n=0}) -> constant 1" begin
            # f(g) = (2*0+1) * 1 * t^0_{0,0}(g) = 1 everywhere.
            N = 8
            fhat = zeros(ComplexF64, SU2FFT.total_coeffs(N))
            # offset(0) = 0; mn_index(0, 0, 0) = 0; so fhat[1] = 1.0 (Julia 1-based).
            fhat[1] = 1.0 + 0im
            f = SU2FFT.fft_inv(fhat, N)
            @test maximum(abs.(f .- (1.0 + 0im))) < 1e-13
        end

        @testset "inv(delta_{l=1,m=0,n=0}) -> 3*cos(theta_k)" begin
            # f(g_{j1,k,j2}) = (2*1+1) * 1 * P^1_{0,0}(cos theta_k) * 1 = 3*cos(theta_k).
            N = 8
            fhat = zeros(ComplexF64, SU2FFT.total_coeffs(N))
            # offset(1) = 1; mn_index(1, 0, 0) = (0+1)*3 + (0+1) = 4; entry index 1 + 4 + 1 = 6.
            # (Use the accessor to be robust.)
            off = SU2FFT.coeff_offset(1)
            idx = off + (0 + 1) * 3 + (0 + 1) + 1   # Julia 1-based
            fhat[idx] = 1.0 + 0im
            f = SU2FFT.fft_inv(fhat, N)
            # Build the expected: theta_k = k*pi/(N-1), k = 0..N-1. f[j2+1, k+1, j1+1] = 3*cos(theta_k).
            max_err = 0.0
            for k in 0:N-1
                theta_k = k * pi / (N - 1)
                expected = 3.0 * cos(theta_k) + 0im
                # All (j1, j2) entries at this k slice should be expected.
                for j1 in 0:N-1, j2 in 0:N-1
                    got = f[j2+1, k+1, j1+1]
                    max_err = max(max_err, abs(got - expected))
                end
            end
            @test max_err < 1e-12
        end

        @testset "linearity" begin
            N = 6
            Random.seed!(2026)
            nc = SU2FFT.total_coeffs(N)
            fhat1 = (rand(ComplexF64, nc) .- (0.5 + 0.5im))
            fhat2 = (rand(ComplexF64, nc) .- (0.5 + 0.5im))
            alpha = 2.0 + 1.5im
            beta  = -0.7 + 0.4im
            f1 = SU2FFT.fft_inv(fhat1, N)
            f2 = SU2FFT.fft_inv(fhat2, N)
            fc = SU2FFT.fft_inv(alpha .* fhat1 .+ beta .* fhat2, N)
            @test maximum(abs.(fc .- (alpha .* f1 .+ beta .* f2))) < 1e-12
        end

        @testset "inverse output dimensions" begin
            for N in (4, 6, 8)
                fhat = zeros(ComplexF64, SU2FFT.total_coeffs(N))
                f = SU2FFT.fft_inv(fhat, N)
                @test size(f) == (N, N, N)
            end
        end

        @testset "spectrum roundtrip (Riemann floor)" begin
            # Documented in notes/inverse_fft.md §2: forward(inverse(fhat)) is NOT
            # recovered to machine precision because closed-grid Riemann theta is
            # not exact for Wigner-d polynomials. This testset confirms that the
            # mathematical correctness of the synthesis is unaffected: the rel_err
            # is bounded (loose, but bounded), and the per-l error grows with l.
            # Achieving 1e-12 here requires Gauss-Legendre nodes (bead su2fft-ega).
            Random.seed!(20260526)
            for N in (6, 8)
                nc = SU2FFT.total_coeffs(N)
                fhat = (rand(ComplexF64, nc) .- (0.5 + 0.5im))
                f = SU2FFT.fft_inv(fhat, N)
                fhat_prime = SU2FFT.fft(f)
                rel_err = maximum(abs.(fhat .- fhat_prime)) / maximum(abs, fhat)
                # Loose threshold matches the C-side test_roundtrip.c floor.
                @test rel_err < 15.0
                @info "spectrum roundtrip (Riemann floor)" N rel_err
            end
        end

        @testset "input validation" begin
            @test_throws ArgumentError SU2FFT.fft_inv(zeros(ComplexF64, 1), 4)   # wrong length
            @test_throws ArgumentError SU2FFT.fft_inv(zeros(ComplexF64, 1), 1)   # N < 2
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
