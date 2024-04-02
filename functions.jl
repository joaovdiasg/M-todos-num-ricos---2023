#####################
##### FUNCTIONS #####
#####################

using Pkg, LinearAlgebra, Plots, Distributions, Random, StatsKit, Interpolations, Roots
using BlackBoxOptimizationBenchmarking, NLopt, LatexPrint, RegressionTables, LaTeXStrings 


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
