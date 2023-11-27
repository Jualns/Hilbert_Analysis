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

#x_cg = Resolve_Cholesky(A, b)
x_gd = Resolve_LU(A, b)


x_sol = A \ b
#println("Erro Cholensky:", norm(x_cg - x_sol))
println("Erro LU:", norm(x_gd - x_sol))
