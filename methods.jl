using LinearAlgebra: I, norm, dot, isposdef

function lu(A)
    n = size(A, 1)
    L = Matrix{Rational{BigInt}}(I, n, n)
    U = copy(A)

    for k in 1:n-1
        for i in k+1:n
            L[i, k] = U[i, k] // U[k, k]
            for j in k:n
                U[i, j] -= L[i, k] * U[k, j]
            end
        end
    end

    return L, U
end

function Cholensky(A)
    n = size(A, 1)
    L = zeros(BigFloat, n, n)

    for j in 1:n
        for i in 1:j
            if i == j
                L[i, j] = sqrt(A[i, i] - sum(L[i, k]^2 for k in 1:i-1))
            else
                L[i, j] = (A[i, j] - sum(L[i, k] * L[j, k] for k in 1:i-1)) / L[i, i]
            end
        end
    end

    return L
end

function SOR(A, b, omega, tol, max_iter)
    n = length(b)
    x = zeros(n)
    x_new = copy(x)
    iter = 0

    while iter < max_iter
        for i in 1:n
            sigma = 0.0
            for j in 1:n
                if j != i
                    sigma += A[i, j] * x_new[j]
                end
            end
            x_new[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (b[i] - sigma)
        end

        if norm(x_new - x) < tol
            return x_new
        end

        copyto!(x, x_new)
        iter += 1
    end

    return x
end

function JOR(A, b, omega, tol, max_iter)
    n = length(b)
    x = zeros(n)
    x_new = copy(x)
    iter = 0

    while iter < max_iter
        for i in 1:n
            sigma = 0.0
            for j in 1:n
                if j != i
                    sigma += A[i, j] * x[j]
                end
            end
            x_new[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (b[i] - sigma)
        end

        if norm(x_new - x) < tol
            return x_new
        end

        copyto!(x, x_new)
        iter += 1
    end

    return x
end

function maxima_descida(A, b, x0, tol, max_iter)
    n = length(b)
    x = copy(x0)
    r = b - A * x
    for iter in 1:max_iter
        Ap = A * r
        alpha = dot(r, r) / dot(r, Ap)
        x += alpha * r
        r = b - A * x
        if norm(r) < tol
            return x
        end
    end
    return x
end


function grad_conj(A, b, x0, tol, max_iter)
    n = length(b)
    x = copy(x0)
    r = b - A * x
    p = copy(r)
    rsold = dot(r, r)

    for iter in 1:max_iter
        Ap = A * p
        alpha = rsold / dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = dot(r, r)

        if sqrt(rsnew) < tol
            return x
        end

        p = r + (rsnew / rsold) * p
        rsold = rsnew
    end

    return x
end

# Função para resolver um sistema Ax = b com a decomposição de Cholesky
function Resolve_Cholesky(A, b)
    if true #isposdef(A)
        L = Cholensky(A).L  # Decomposição de Cholesky
        y = L \ b  # Resolvendo Ly = b
        x = L' \ y  # Resolvendo L'x = y (usando a transposta de L)
        return x
    else
        throw(ArgumentError("A matriz não é simétrica definida positiva."))
    end
end

# Função para resolver um sistema Ax = b com a decomposição LU
function Resolve_LU(A, b)
    n = size(A, 1)
    L, U = LU(A)
    y = zeros(BigFloat, n)
    x = zeros(BigFloat, n)

    # Resolvendo Ly = b
    for i in 1:n
        y[i] = b[i]
        for j in 1:i-1
            y[i] -= L[i, j] * y[j]
        end
        y[i] /= L[i, i]
    end

    # Resolvendo Ux = y
    for i in n:-1:1
        x[i] = y[i]
        for j in i+1:n
            x[i] -= U[i, j] * x[j]
        end
        x[i] /= U[i, i]
    end

    return x
end