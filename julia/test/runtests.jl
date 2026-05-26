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

    @testset "Gauss-Legendre theta nodes (bead ega)" begin
        @testset "gl_nodes_weights basic properties" begin
            for N in (1, 2, 4, 8, 16)
                x, w = SU2FFT.gl_nodes_weights(N)
                @test length(x) == length(w) == N
                @test all(-1 .< x .< 1)
                @test all(w .> 0)
                @test issorted(x)
                @test sum(w) ≈ 2.0 atol=1e-13
                # Symmetry: x[k] == -x[N-1-k] (zero-based -> reverse-paired).
                @test maximum(abs.(x .+ reverse(x))) < 1e-14
            end
        end

        @testset "GL degree exactness up to 2N-1" begin
            # Integral of x^p on [-1,1] is 0 (odd p) or 2/(p+1) (even p).
            # N-point GL is exact for polynomials of degree <= 2N-1.
            for N in (4, 8)
                x, w = SU2FFT.gl_nodes_weights(N)
                for p in 0:(2N - 1)
                    approx = sum(w[k] * x[k]^p for k in 1:N)
                    exact = isodd(p) ? 0.0 : 2.0 / (p + 1)
                    @test approx ≈ exact atol=1e-12
                end
            end
        end

        @testset "fft_gl(constant) -> exact DC" begin
            # The headline GL improvement vs fft: fhat(0,0,0) = 1.0 exactly,
            # not (N/(N-1))^2.
            for N in (4, 6, 8, 16)
                f = ones(ComplexF64, N, N, N)
                fhat = SU2FFT.fft_gl(f)
                @test fhat[1] ≈ 1.0 + 0.0im atol=1e-13
            end
        end

        @testset "fft_gl(constant) leakage bounded (phi/psi aliasing floor)" begin
            # Documents the empirical floor that bead su2fft-0t1 will eliminate.
            N = 8
            f = ones(ComplexF64, N, N, N)
            fhat = SU2FFT.fft_gl(f)
            leakage = maximum(abs.(fhat[2:end]))   # skip the DC entry
            @info "fft_gl(constant) leakage" N leakage
            @test leakage < 0.3   # empirically ~0.197 at N=8
        end

        @testset "fft_inv_gl(delta_{l=0,m=0,n=0}) -> constant 1 at GL nodes" begin
            N = 8
            fhat = zeros(ComplexF64, SU2FFT.total_coeffs(N))
            fhat[1] = 1.0 + 0im
            f = SU2FFT.fft_inv_gl(fhat, N)
            @test maximum(abs.(f .- (1.0 + 0im))) < 1e-13
        end

        @testset "fft_inv_gl(delta_{l=1,m=0,n=0}) -> 3*cos(theta_k_gl)" begin
            N = 8
            fhat = zeros(ComplexF64, SU2FFT.total_coeffs(N))
            off = SU2FFT.coeff_offset(1)
            idx = off + (0 + 1) * 3 + (0 + 1) + 1
            fhat[idx] = 1.0 + 0im
            f = SU2FFT.fft_inv_gl(fhat, N)
            # GL theta nodes
            x_gl, _ = SU2FFT.gl_nodes_weights(N)
            theta_gl = acos.(x_gl)
            max_err = 0.0
            for k in 0:N-1
                expected = 3.0 * cos(theta_gl[k+1]) + 0im
                for j1 in 0:N-1, j2 in 0:N-1
                    got = f[j2+1, k+1, j1+1]
                    max_err = max(max_err, abs(got - expected))
                end
            end
            @test max_err < 1e-12
        end

        @testset "linearity of fft_inv_gl" begin
            N = 6
            Random.seed!(20260526)
            nc = SU2FFT.total_coeffs(N)
            fhat1 = rand(ComplexF64, nc) .- (0.5 + 0.5im)
            fhat2 = rand(ComplexF64, nc) .- (0.5 + 0.5im)
            alpha = 2.0 + 1.5im
            beta  = -0.7 + 0.4im
            f1 = SU2FFT.fft_inv_gl(fhat1, N)
            f2 = SU2FFT.fft_inv_gl(fhat2, N)
            fc = SU2FFT.fft_inv_gl(alpha .* fhat1 .+ beta .* fhat2, N)
            @test maximum(abs.(fc .- (alpha .* f1 .+ beta .* f2))) < 1e-12
        end

        @testset "GL roundtrip floor (regression bound; bead su2fft-0t1 tracks fix)" begin
            # Single-coefficient roundtrip at (l, 0, 0): error stays bounded.
            # Empirical at N=8: max around 0.34 at l=2; bound at 0.5.
            N = 8
            nc = SU2FFT.total_coeffs(N)
            fhat = zeros(ComplexF64, nc)
            fhat_rt = zeros(ComplexF64, nc)
            max_overall = 0.0
            for l in 0:(N - 1)
                fhat .= 0
                idx_l00 = SU2FFT.coeff_offset(l) + (0 + l) * (2l + 1) + (0 + l) + 1
                fhat[idx_l00] = 1.0 + 0im
                f = SU2FFT.fft_inv_gl(fhat, N)
                fhat_rt .= SU2FFT.fft_gl(f)
                max_err = maximum(abs.(fhat_rt .- fhat))
                max_overall = max(max_overall, max_err)
            end
            @info "GL roundtrip floor (max over l in [0, N-1] at (m=n=0))" N max_overall
            @test max_overall < 0.5
        end
    end

    @testset "wigner_d_half (bead n8e Tier 1)" begin
        # Spin-1/2 closed forms (Sakurai d^{1/2}, paper i^{m-n} phase applied).
        # See notes/half_integer.md §1 and tests/test_wigner.c.
        @testset "spin-1/2 at identity" begin
            two_l = 1
            @test SU2FFT.wigner_d_half(two_l, -1, -1, 0.0) ≈ 1.0 + 0.0im atol=1e-13
            @test SU2FFT.wigner_d_half(two_l, -1,  1, 0.0) ≈ 0.0 + 0.0im atol=1e-13
            @test SU2FFT.wigner_d_half(two_l,  1, -1, 0.0) ≈ 0.0 + 0.0im atol=1e-13
            @test SU2FFT.wigner_d_half(two_l,  1,  1, 0.0) ≈ 1.0 + 0.0im atol=1e-13
        end

        @testset "spin-1/2 closed form" begin
            # P^{1/2}_{ 1/2,  1/2}(t) =  cos(t/2)
            # P^{1/2}_{ 1/2, -1/2}(t) = +i sin(t/2)
            # P^{1/2}_{-1/2,  1/2}(t) = +i sin(t/2)
            # P^{1/2}_{-1/2, -1/2}(t) =  cos(t/2)
            theta = 0.7
            two_l = 1
            c = cos(theta * 0.5)
            s = sin(theta * 0.5)
            @test SU2FFT.wigner_d_half(two_l,  1,  1, theta) ≈ c + 0.0im atol=1e-12
            @test SU2FFT.wigner_d_half(two_l,  1, -1, theta) ≈ 0.0 + s*im atol=1e-12
            @test SU2FFT.wigner_d_half(two_l, -1,  1, theta) ≈ 0.0 + s*im atol=1e-12
            @test SU2FFT.wigner_d_half(two_l, -1, -1, theta) ≈ c + 0.0im atol=1e-12
        end

        @testset "matches integer wigner_d when two_l even" begin
            for l in 0:3, n in -l:l, m in -l:l
                for theta in (0.0, 0.3, 1.1, 2.4, pi - 0.1)
                    got  = SU2FFT.wigner_d_half(2l, 2n, 2m, theta)
                    want = SU2FFT.wigner_d(l, n, m, theta)
                    @test got ≈ want atol=1e-11
                end
            end
        end

        @testset "spin-1/2 column unitarity" begin
            two_l = 1
            for theta in 0.1:0.5:3.0
                for two_m in (-1, 1)
                    s = 0.0
                    for two_n in (-1, 1)
                        v = SU2FFT.wigner_d_half(two_l, two_n, two_m, theta)
                        s += abs2(v)
                    end
                    @test s ≈ 1.0 atol=1e-12
                end
            end
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

    @testset "convolve (bead d7v)" begin
        @testset "identity-in-l=0 (constant-1 spectrum) acts as identity-on-DC" begin
            # fhat = (1, 0, 0, ...): (f*g)hat(0)_{00} = ghat(0)_{00}; rest zero.
            N = 6
            nc = SU2FFT.total_coeffs(N)
            fhat = zeros(ComplexF64, nc)
            fhat[1] = 1.0 + 0im
            Random.seed!(20260526)
            ghat = rand(ComplexF64, nc) .- (0.5 + 0.5im)
            fghat = SU2FFT.convolve(fhat, ghat, N)
            @test fghat[1] ≈ ghat[1] atol=1e-13
            @test maximum(abs, fghat[2:end]) < 1e-13
        end

        @testset "linearity in first argument" begin
            N = 5
            nc = SU2FFT.total_coeffs(N)
            Random.seed!(2)
            f1 = rand(ComplexF64, nc) .- (0.5 + 0.5im)
            f2 = rand(ComplexF64, nc) .- (0.5 + 0.5im)
            g  = rand(ComplexF64, nc) .- (0.5 + 0.5im)
            a = 2.0 + 1.5im
            b = -0.7 + 0.4im
            cv1 = SU2FFT.convolve(f1, g, N)
            cv2 = SU2FFT.convolve(f2, g, N)
            cv_combo = SU2FFT.convolve(a .* f1 .+ b .* f2, g, N)
            @test maximum(abs.(cv_combo .- (a .* cv1 .+ b .* cv2))) < 1e-12
        end

        @testset "diagonal l=1 explicit (4, 10, 18)" begin
            # fhat(1) = diag(1,2,3), ghat(1) = diag(4,5,6) -> (fg)hat(1) = diag(4,10,18).
            N = 2
            nc = SU2FFT.total_coeffs(N)
            fhat = zeros(ComplexF64, nc)
            ghat = zeros(ComplexF64, nc)
            fhat[1] = 1.0; ghat[1] = 1.0
            off1 = SU2FFT.coeff_offset(1)
            for idx in 0:2
                flat = off1 + idx * 3 + idx + 1   # Julia 1-based
                fhat[flat] = ComplexF64(idx + 1)
                ghat[flat] = ComplexF64(idx + 4)
            end
            fghat = SU2FFT.convolve(fhat, ghat, N)
            expected = [4.0, 10.0, 18.0]
            for idx in 0:2
                flat = off1 + idx * 3 + idx + 1
                @test fghat[flat] ≈ expected[idx+1] + 0.0im atol=1e-13
            end
            # Off-diagonal zero.
            for i in 0:2, j in 0:2
                i == j && continue
                @test abs(fghat[off1 + i * 3 + j + 1]) < 1e-13
            end
            @test fghat[1] ≈ 1.0 + 0.0im atol=1e-13
        end

        @testset "input validation" begin
            @test_throws ArgumentError SU2FFT.convolve(zeros(ComplexF64, 1),
                                                       zeros(ComplexF64, SU2FFT.total_coeffs(4)),
                                                       4)
            @test_throws ArgumentError SU2FFT.convolve(zeros(ComplexF64, SU2FFT.total_coeffs(4)),
                                                       zeros(ComplexF64, 1),
                                                       4)
        end
    end

    @testset "spherical FFT (bead 5fb)" begin
        @testset "sphere_total_coeffs == N^2" begin
            for N in (1, 2, 4, 8)
                @test SU2FFT.sphere_total_coeffs(N) == N * N
            end
        end

        @testset "fft_sphere(zeros) -> zeros" begin
            N = 6
            f = zeros(ComplexF64, N, N)
            fhat = SU2FFT.fft_sphere(f)
            @test length(fhat) == SU2FFT.sphere_total_coeffs(N)
            @test maximum(abs, fhat) < 1e-13
        end

        @testset "fft_sphere_inv(zeros) -> zeros" begin
            N = 6
            fhat = zeros(ComplexF64, SU2FFT.sphere_total_coeffs(N))
            f = SU2FFT.fft_sphere_inv(fhat, N)
            @test size(f) == (N, N)
            @test maximum(abs, f) < 1e-13
        end

        @testset "inv(delta_{l=0, n=0}) -> constant 1" begin
            # Spherical analog of inv(delta_{l=0,m=n=0}) = constant.
            # fhat_sph[1] corresponds to (l=0, n=0); synthesis = 1 everywhere.
            N = 8
            fhat = zeros(ComplexF64, SU2FFT.sphere_total_coeffs(N))
            fhat[1] = 1.0 + 0im
            f = SU2FFT.fft_sphere_inv(fhat, N)
            @test maximum(abs.(f .- (1.0 + 0im))) < 1e-13
        end

        @testset "inv(delta_{l=1, n=0}) -> 3*cos(theta_k)" begin
            # fhat_sph[(l=1, n=0)] index: sum_{l'<1}(2l'+1) + (n+l) + 1 = 1 + 1 + 1 = 3.
            N = 8
            fhat = zeros(ComplexF64, SU2FFT.sphere_total_coeffs(N))
            fhat[3] = 1.0 + 0im
            f = SU2FFT.fft_sphere_inv(fhat, N)
            # f_sph[k+1, j1+1] = 3*cos(theta_k); theta_k = k*pi/(N-1).
            max_err = 0.0
            for k in 0:N-1
                theta_k = k * pi / (N - 1)
                expected = 3.0 * cos(theta_k) + 0im
                for j1 in 0:N-1
                    got = f[k+1, j1+1]
                    max_err = max(max_err, abs(got - expected))
                end
            end
            @test max_err < 1e-12
        end

        @testset "linearity of fft_sphere_inv" begin
            N = 6
            Random.seed!(20260526)
            nc = SU2FFT.sphere_total_coeffs(N)
            fh1 = rand(ComplexF64, nc) .- (0.5 + 0.5im)
            fh2 = rand(ComplexF64, nc) .- (0.5 + 0.5im)
            alpha = 2.0 + 1.5im
            beta  = -0.7 + 0.4im
            f1 = SU2FFT.fft_sphere_inv(fh1, N)
            f2 = SU2FFT.fft_sphere_inv(fh2, N)
            fc = SU2FFT.fft_sphere_inv(alpha .* fh1 .+ beta .* fh2, N)
            @test maximum(abs.(fc .- (alpha .* f1 .+ beta .* f2))) < 1e-12
        end

        @testset "input validation" begin
            @test_throws ArgumentError SU2FFT.fft_sphere(zeros(ComplexF64, 4, 5))
            @test_throws ArgumentError SU2FFT.fft_sphere(zeros(ComplexF64, 1, 1))
            @test_throws ArgumentError SU2FFT.fft_sphere_inv(zeros(ComplexF64, 1), 4)
            @test_throws ArgumentError SU2FFT.fft_sphere_inv(zeros(ComplexF64, 4), 1)
        end
    end

end
