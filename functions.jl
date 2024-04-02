#####################
##### FUNCTIONS #####
#####################

# using Pkg, LinearAlgebra, Plots, Distributions, Random, StatsKit, Roots 
# using BlackBoxOptimizationBenchmarking, NLopt, LatexPrint, RegressionTables, LaTeXStrings, Polynomials, Optim, Interpolations 
# NLsolve, FastGaussQuadrature, QuadGK, QuantEcon



# function to calculate the state grid and the transition porobability by the Tauchen's method
function Tauchen(rho, N, sigma, mu = 0.0, m = 3) 

    theta_sup = mu + m*sqrt(sigma^2/(1-rho^2)) # sigma is the sd of the error
    theta_inf = mu - m*sqrt(sigma^2/(1-rho^2))
    theta_grid = collect(LinRange(theta_inf, theta_sup, N))
    delta_t = (theta_sup - theta_inf)/(2*(N-1))

    P = zeros(N,N)
    x = Normal(0,1)

    for i in 1:N
        for j in 2:(N-1)
            P[i,1] = cdf(x, (theta_inf + delta_t - rho*theta_grid[i]-(1-rho)*mu)/sigma)
            P[i,N] = 1 - cdf(x, (theta_sup - delta_t - rho*theta_grid[i]-(1-rho)*mu)/sigma)
            P[i,j] = cdf(x, (theta_grid[j] + delta_t - rho*theta_grid[i]-(1-rho)*mu)/sigma) - 
                     cdf(x, (theta_grid[j] - delta_t - rho*theta_grid[i]-(1-rho)*mu)/sigma)
        end 
    end            

    return [P, theta_grid] # using collect to create a vector

end

# function to calculate the consumption in the basic neoclassical model
function consump(k, z, kk, alpha, delta) 

    return((z*(k)^alpha + (1-delta)*k - kk))

end

# function to calculate the CRRA utility function
function util(c, gamma) 
    if c < 0
        u = - Inf
    else
    u = (c^(1-gamma)-1)/(1-gamma)
    end
    return u
end

# function to calculate the derivative of the utility function
function deriv_util(c, gamma =2)
    d_u = 1/(c^gamma)
end

# function to calculate the inverse function of the derivative
function inv_deriv_util(c, gamma = 2)
    i_d_u = (1/c)^(1/gamma)
end


#### Euler Equation Error (EEE)

function EEE(c0, z_grid, markov_matrix, k_grid, capital, alpha = 1/3, beta = 0.987, delta = 0.012, gamma = 2)
    nk = length(k_grid);
    nz = length(z_grid);
    eee = zeros(nk, nz);

    Threads.@threads for iz in 1:nz # calculating the EEE
        for ik in 1:nk
            kk = argmin(abs.(k_grid .- capital[ik, iz]));    
            cc = deriv_util.(c0[kk,:], gamma);
            konst = z_grid[iz]*alpha*(k_grid[kk])^(alpha-1) + (1-delta)
            eee[ik, iz] = log10(abs(1 - (1/c0[ik,iz])*inv_deriv_util((
                                beta * markov_matrix[iz,:]' * cc * konst), gamma)))
        end
    end
    return eee
end

#### Euler Equation Error for the chebyshev polynomial(EEE)

function EEE_cheby(c0, z_grid, markov_matrix, k_grid, capital, theta, d = 3,  alpha = 1/3, beta = 0.987, delta = 0.012, gamma = 2)
    nk = length(k_grid);
    nz = length(z_grid);
    eee = zeros(nk, nz);

    Threads.@threads for iz in 1:nz # calculating the EEE
        for ik in 1:nk
            ve = 0 
            for izz in 1:nz
                c_new = c_hat(theta[:, izz], capital[ik, iz], k_grid[1] , k_grid[end], d)
                cc = deriv_util.(c_new, gamma);
                ve = ve + cc*markov_matrix[iz, izz]
            end
            konst = z_grid[iz]*alpha*(capital[ik, iz])^(alpha-1) + (1-delta)
            eee[ik, iz] = log10(abs(1 - (1/c0[ik,iz])*inv_deriv_util((
                                beta * ve * konst), gamma)))
        end
    end
    return eee
end

function EEE_FE(c0, z_grid, markov_matrix, k_grid, capital, theta, n_ki = 10,  alpha = 1/3, beta = 0.987, delta = 0.012, gamma = 2)
    nk = length(k_grid);
    nz = length(z_grid);
    eee = zeros(nk, nz);
    intervals = FE_intervals(k_grid, n_ki); 

    Threads.@threads for iz in 1:nz # calculating the EEE
        for ik in 1:nk
            ve = 0 
            for izz in 1:nz
                c_new = c_hat_FE(theta[:, izz], capital[ik, iz], intervals, n_ki)
                cc = deriv_util.(c_new, gamma);
                ve = ve + cc*markov_matrix[iz, izz]
            end
            konst = z_grid[iz]*alpha*(capital[ik, iz])^(alpha-1) + (1-delta)
            eee[ik, iz] = log10(abs(1 - (1/c0[ik,iz])*inv_deriv_util((
                                beta * ve * konst), gamma)))
        end
    end
    return eee
end

function EEE_H(c0, r0, z_grid, markov_matrix, a_grid, capital, beta = 0.987, gamma = 2)
    nk = length(a_grid);
    nz = length(z_grid);
    eee = zeros(nk, nz);
    
    Threads.@threads for iz in 1:nz # calculating the EEE
        for ik in 1:nk
            ve = 0 
            for izz in 1:nz
                aa = argmin(abs.(a_grid .- capital[ik, iz]));    
                c_new = exp(z_grid[izz]) + (1+r0)*a_grid[aa] - capital[aa, izz]
                cc = deriv_util.(c_new, gamma);
                ve = ve + cc*markov_matrix[iz, izz]
            end
            eee[ik, iz] = log10(abs(1 - (1/c0[ik,iz])*inv_deriv_util((
                                beta * (1+r0) * ve), gamma)))
        end
    end
    return eee
end

