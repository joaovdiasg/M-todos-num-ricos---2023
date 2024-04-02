#####################
##### FUNCTIONS #####
#####################

using Pkg, LinearAlgebra, Plots, Distributions, Random, StatsKit, NLsolve, FastGaussQuadrature, QuadGK
# using BlackBoxOptimizationBenchmarking, NLopt, LatexPrint, RegressionTables, LaTeXStrings, Polynomials, Optim, Interpolations, Roots 


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

# brute force method exploring the concavity and the monotonicity of the value function
function concave_mon(k_grid, z_grid, markov_matrix, v0 = 0, alpha = 1/3, beta = 0.987, delta = 0.012, gamma = 2, inertia = 0, maxiter = 10000, toler = 10^-5)

    nk = length(k_grid);
    nz = length(z_grid);
    capital = zeros(nk, nz);
    v_new = zeros(nk, nz);
    consumption = zeros(nk, nz);
    it = 0;
    if v0 == 0
        v_old = zeros(nk, nz);
    else
        v_old = copy(v0);
    end
        
    @time begin
        for j in 1:maxiter
            Threads.@threads for iz in 1:nz # state
                mon = 1; # monotonicity
                for ik in 1:nk # capital
                    h_0 = 0 # initial value to use concavity, using zero as in v_old
                    for im in mon:nk # next capital
                        h_temp = util(consump(k_grid[ik], z_grid[iz], k_grid[im], alpha, delta), gamma) + beta * markov_matrix[iz,:]' * v_old[im,:]

                        if h_temp < h_0 # testing because v is concave
                            mon = im-1 # update the monotonicity index
                            capital[ik, iz] = k_grid[im-1] # saving the k'
                            v_new[ik, iz] = h_0 # updating the value function
                            break

                        else # keep going
                            capital[ik, iz] = k_grid[im] 
                            h_0 = copy(h_temp) # updating the last value for concavity
                            v_new[ik,iz] = h_temp
                        end # concavity
                    end # k' and monotonicity
                end # capital
            end # state

            if maximum(abs.(v_new - v_old)) > toler
                v_old = inertia .* v_old .+ (1-inertia) .* copy(v_new)
                it += 1 # iterations counter
            else
                break
            end # tolerance
        end # iterations
        
        Threads.@threads for iz in 1:nz # calculating the political function
            for ik in 1:nk
                consumption[ik, iz] = consump(k_grid[ik], z_grid[iz], capital[ik,iz], alpha, delta);
            end
        end
    end #time
    return v_new, capital, consumption, println("
    Iterações: ", it, ". O primeiro elemento dessa função retorna a função valor, o segundo a sequência de capitais escolhidos e o terceiro retorna a função política do consumo.
    ")
end # function

# concavity and the monotonicity with accelerator
function concave_mon_acc(k_grid, z_grid, markov_matrix, accelerator = 0, v0 = 0, alpha = 1/3, beta = 0.987, delta = 0.012, gamma = 2, inertia = 0, maxiter = 10000, toler = 10^-5)

    nk = length(k_grid);
    nz = length(z_grid);
    capital = zeros(nk, nz);
    v_new = zeros(nk, nz);
    consumption = zeros(nk, nz);
    it = 0;
    if v0 == 0
        v_old = zeros(nk, nz);
    else
        v_old = copy(v0);
    end
    if accelerator == 0 # when not using the accelerator
        acc = collect(range(1,maxiter, length=maxiter))
    elseif accelerator == 1 # a default accelerator for maximizating 10% of the iterations
        n_acc = trunc(Int,round(0.05*maxiter))-1
        final_size = trunc(Int, round(0.05*(maxiter-length(n_acc)) + 1));
        acc = [collect(range(1, n_acc, length= n_acc)); round.(collect(range(n_acc+1, maxiter, final_size)))]
    else # to choose another vector of 'acceleration'
        acc = copy(accelerator)
    end
        
    @time begin
        for j in 1:maxiter
            if j in acc
                Threads.@threads for iz in 1:nz # state
                    mon = 1; # monotonicity
                    for ik in 1:nk # capital
                        h_0 = 0 # initial value to use concavity, using zero as in v_old
                        for im in mon:nk # next capital
                            
                            h_temp = util(consump(k_grid[ik], z_grid[iz], k_grid[im], alpha, delta), gamma) + beta * markov_matrix[iz,:]' * v_old[im,:]

                            if h_temp < h_0 # testing because v is concave
                                mon = im-1 # update the monotonicity index
                                capital[ik, iz] = k_grid[im-1] # saving the k'
                                v_new[ik, iz] = h_0 # updating the value function
                                break

                            else # keep going
                                capital[ik, iz] = k_grid[im] 
                                h_0 = copy(h_temp) # updating the last value for concavity
                                v_new[ik,iz] = h_temp
                            end # concavity
                        end # k' and monotonicity
                    end # capital
                end # state

                if maximum(abs.(v_new - v_old)) > toler
                                
                    v_old = inertia .* v_old .+ (1-inertia) .* copy(v_new)
                    it += 1 # iterations counter

                else

                    break
                    
                end # tolerance
            else # accelerator
                Threads.@threads for iz in 1:nz
                    for ik in 1:nk
                        kk = findfirst(x -> x == capital[ik, iz], k_grid)
                        h_temp = util(consump(k_grid[ik], z_grid[iz], k_grid[kk], alpha, delta), gamma) + beta * markov_matrix[iz,:]' * v_old[kk,:]
                        v_new[ik, iz] = h_temp; # value function
                    end # capital
                end # state

                if maximum(abs.(v_new - v_old)) > toler
                    
                    v_old = inertia .* v_old .+ (1-inertia) .* copy(v_new)
                    it += 1 # iterations counter

                else

                    break
                    
                end # tolerance
            end # accelerator
        end # iteration
        
        Threads.@threads for iz in 1:nz # calculating the political function
            for ik in 1:nk
                consumption[ik, iz] = consump(k_grid[ik], z_grid[iz], capital[ik,iz], alpha, delta);
            end
        end

    end #time
    return v_new, capital, consumption, println("
    Iterações: ", it, ". O primeiro elemento dessa função retorna a função valor, o segundo a sequência de capitais escolhidos e o terceiro retorna a função política do consumo.
    ")
end # function


# concavity and the monotonicity with multigrid
function concave_mon_mult(k_grid, z_grid, markov_matrix, accelerator = 0, v0 = 0, alpha = 1/3, beta = 0.987, delta = 0.012, inertia = 0, maxiter = 10000, toler = 10^-5) 

    NK = length(k_grid);
    nz = length(z_grid);
    n1 = trunc(Int,length(k_grid[1]));
    acc = copy(accelerator);
    v_old = copy(v0);   

    for mult in 1:NK

        nk = trunc(Int,length(k_grid[mult]));
        
        global result =  concave_mon_acc(k_grid[mult], z_grid, markov_matrix, acc, v_old)
        
        if mult < NK
            v_0 = zeros(trunc(Int, length(k_grid[mult+1])), nz)
            v_i = copy(result[1])
            for iz in 1:nz
            itp = linear_interpolation(GRID[mult],v_i[:, iz])
            v_0[:, iz] = itp(GRID[mult+1])
            end
            v_old = copy(v_0)
        end
    end
    return result
end


### Endogenous grid method
function endog_grid(k_grid, z_grid, markov_matrix, gamma = 2, alpha = 1/3, beta = 0.987, delta = 0.012, inertia = 0, maxiter = 10000, toler = 10^-5) 

    nk = length(k_grid);
    nz = length(z_grid);
    c_old = zeros(nk, nz);
    c_new = zeros(nk, nz);
    inv_c = zeros(nk, nz);
    capital_temp = zeros(nk, nz);
    capital = zeros(nk, nz);
    it = 0;

    @time begin
        Threads.@threads for iz in 1:nz # creating a guess for the initial c(k,z). I opted to use k = k'
            for ik in 1:nk
            c_old[ik, iz] = consump(k_grid[ik], z_grid[iz], k_grid[ik], alpha, delta);
            end
        end

        for j in 1:maxiter
            Threads.@threads for iz in 1:nz 
                for ik in 1:nk
                    cc = deriv_util.(c_old[ik,:], gamma);
                    konst = z_grid[iz]*alpha*(k_grid[ik])^(alpha-1) + (1-delta)
                    inv_c[ik, iz] = inv_deriv_util((beta * markov_matrix[iz,:]' * cc * konst), gamma) # inverse of the marginal utility
                    capital_temp[ik, iz] = find_zero(x -> consump(x, z_grid[iz], k_grid[ik], alpha, delta)-inv_c[ik, iz], 0) # 0 is the intial guess for the function 
                end # capital
                itp = linear_interpolation(capital_temp[:, iz], k_grid, extrapolation_bc = Line());
                capital[:, iz] = itp.(k_grid)
            end # state

            Threads.@threads for iz in 1:nz # creating the new consumption matrix
                for ik in 1:nk
                    c_new[ik,iz] = consump(k_grid[ik], z_grid[iz], capital[ik, iz], alpha, delta)
                end # capital   
            end # state

            if maximum(abs.(c_new - c_old)) > toler
                                
                c_old = inertia .* c_old .+ (1-inertia) .* copy(c_new)
                it += 1 # iterations counter
            else
                break
            end # tolerance
        end # iterations
    end # time
   
    return  capital, c_new, println("
    Iterações: ", it, ". O primeiro elemento dessa função retorna a sequência de capitais escolhidos e o segundo retorna a função política do consumo.
    ")
end
