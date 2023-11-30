using SpecialMatrices
using LinearAlgebra

include("methods.jl")

function special_hilbert(n)
    return Hilbert(n)
end

function H(n)
    A = ones(Rational{BigInt}, n, n)
    for i = 1:n
        for j = 1:n
            A[i, j] = 1 // (i + j - 1)
        end
    end
    return A
end

function B(n)
    return ones(BigInt, n)
end

function inverse_H(n)
    A = ones(Rational{BigInt}, n, n)
    for j = 1:n
        for i = 1:n
            A[i, j] = (i + j - 1)
        end
    end
    return A * 1 // n
end

function condicionamento(Matrix, p=2)
    inversa_matrix = inv(Matrix)
    norma = norm(Matrix, p)
    norma_inversa = norm(inversa_matrix, p)

    return abs(norma) * abs(norma_inversa)
end

n = 100
A = H(n)
b = B(n)

x0 = zeros(BigFloat, n)
tol = 1e-5
max_iter = 100000
ω = 0.5
#x_cg = Resolve_Cholesky(A, b)

x_sor = SOR(A, b, ω, tol, max_iter)
x_jor = JOR(A, b, ω, tol, max_iter)
x_md = maxima_descida(A, b, x0, tol, max_iter)
x_gc = grad_conj(A, b, x0, tol, max_iter)
x_ch = Resolve_CHolensky(A, b)
x_lu = Resolve_LU(A, b)


x_sol = A \ b
#println("Erro Cholensky:", norm(x_cg - x_sol))
function er(x_pred, x_ot)
    return norm(x_ot - x_pred)
end

println("Erro SOR:", er(x_sor, x_sol))
println("Erro JOR:", er(x_jor, x_sol))
println("Erro Max Des:", er(x_md, x_sol))
println("Erro Grad:", er(x_gc, x_sol))
println("Erro Cholensky:", er(x_ch, x_sol))
println("Erro LU:", er(x_lu, x_sol))
