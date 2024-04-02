##########################################
#### Lista 3 - Métodos Numéricos 2023 ####
##########################################

include("functions.jl") # I included the functions from ps1 and ps2 in this file 

const rho = 0.95 # persistence
const sigma = 0.007 # sd of the error
const nz = 7 # number of states
const nk500 = 500 # number of grid points
const nk100 = 100
const nk5000 = 5000
const m = 3 # number of sds
const mu = 0 # mean of the ar process
const beta = 0.987 # discount rate
const delta = 0.012 # depreciation 
const gamma = 2 # relative risk aversion
const alpha = 1/3 # relative participation of capital
const inertia = 0 # inertia for the value function iteration
const toler = 10^-5 # tolerance for the value function iteration
const maxiter = 10000 # maximum number of value function iterations

Random.seed!(7)

const k_ss = (beta*alpha/(1-beta*(1-delta)))^(1/(1-alpha));

k_grid500 = range(start = 0.75*k_ss, stop = 1.25*k_ss, length = 500);
k_grid5000 = range(start = 0.75*k_ss, stop = 1.25*k_ss, length = 5000);

z = Tauchen(rho, nz, sigma);
# z[1] is the markov transiction matrix and z[2] is the state grid

######### Question 1

# chebyshev function of degree d
function cheby(x, d)
    t = cos(d*acos(x))
end

# function to calculate chebyshev zeros
function cheby_zeros(d) 
    z = zeros(d)
    for i in 1:d
    z[i] = -cos((2*i-1)*pi/(2*d))
    end
    return z
end

# chebyshev polynomials for the approximation
# k here is in the 'normal' grid, a and b are the bounds of the grid
function c_hat(theta, k, a ,b, d = 3)
    x = 2*((k-a)/(b-a))-1 # transforming so x is in [-1, 1]
    c = 0
    for i in 1:d+1
        c = c + theta[i]*cheby(x, i-1) # creating the sum of chebyshev polynomials 
    end
    return c    
end

# function to calculate the residuals
function cheby_resid(theta, k0, k_grid, markov_matrix, z_grid, d = 3, alpha = 1/3, beta = 0.987, delta = 0.012, gamma = 2)
    a = k_grid[1];
    b = k_grid[end];
    nz = length(z_grid);
    R = zeros(nz);
    
    k00 = ((b-a)/2)*k0 + (a+b)/2 # transforming k0 in [-1, 1] to the capital 'normal' size

    for iz in 1:nz
        c0 = c_hat(theta[:,iz], k00, a, b, d);
        k1 = z_grid[iz]*k00^alpha+(1-delta)*k00 - c0;
        c1 = zeros(nz);
        for izz in 1:nz
            c1[izz] = c_hat(theta[:,izz], k1, a, b, d); # vector os tomorrow cunsumptions based on each state 
        end
        konst = alpha*z_grid[iz]*k1^(alpha-1)+(1-delta)

        R[iz] = (c0^(-gamma))^(-1)*beta*(markov_matrix[iz,:]'*(c1).^(-gamma)*konst)-1 # Euler Equation
    end
    return R'
end

# function to construct the linear system
function final_f(theta, k_grid, markov_matrix, z_grid, d = 3, alpha = 1/3, beta = 0.987, delta = 0.012, gamma = 2)    
    k0 = cheby_zeros(d+1);
    nz = length(z_grid);
    F = zeros(d+1, nz);

    for id in 1:d+1
        F[id, :] = cheby_resid(theta, k0[id], k_grid, markov_matrix, z_grid, d, alpha, beta, delta, gamma) # cheby_resid returns a vector of size nz
    end
    return F
end

initial = [1 1 1 1 1 1 1; 0.05 0.05 0.05 0.05 0.05 0.05 0.05] # initial guess

# this function solve the linear system and calculate the politcal functions for the consumption and capital and the value function
function solve_system(k_grid, markov_matrix, z_grid, theta0, d = 3, alpha = 1/3, beta = 0.987, delta = 0.012, gamma = 2)
    nz = length(z_grid);
    nk = length(k_grid);

    @time begin # counting time just for the method
        x = nlsolve(theta -> final_f(theta, k_grid, markov_matrix, z_grid, 1), theta0); # d = 1
        
        # using the previous theta solutions as initial guess
        if d > 1
            for id in 2:d
                initial = [x.zero; zeros(nz)'];
                x = nlsolve(theta -> final_f(theta, k_grid, markov_matrix, z_grid, id), initial)
            end
        end
    end    

    consumption = zeros(nk, nz);
    capital = zeros(nk, nz);
    value_function = zeros(nk, nz);
    v_old = zeros(nk, nz);

    # calculating the politicals functions
    Threads.@threads for iz in 1:nz
        for ik in 1:nk
            consumption[ik, iz] = c_hat(x.zero[:, iz], k_grid[ik], k_grid[1] , k_grid[end], d)
            capital[ik, iz] = z_grid[iz]*k_grid[ik]^alpha+(1-delta)*k_grid[ik] - consumption[ik, iz];
#            kk = argmin(abs.(k_grid .- capital[ik, iz]))
#            capital[ik, iz] = k_grid[kk]; # the capital in the value function is not in the grid, so I find the closest
        end # capital
    end # state

    # calculating the value function
    for j in 1:maxiter
        Threads.@threads for iz in 1:nz
            for ik in 1:nk
                kk = argmin(abs.(k_grid .- capital[ik, iz]));
#                kk = findfirst(x -> x == capital[ik, iz], k_grid); # finding the index of the next capital
                value_function[ik, iz] = util(consump(k_grid[ik], z_grid[iz], k_grid[kk], alpha, delta), gamma) + beta * markov_matrix[iz,:]' * v_old[kk,:];
            end # capital
        end # state    
        
        if maximum(abs.(value_function - v_old)) > toler                    
            v_old = inertia .* v_old .+ (1-inertia) .* copy(value_function)
        else
            break
        end # tolerance
    end # iterations

    return x.zero, consumption, capital, value_function
end

# using a 3 degree chebyshev polynomial, so I have a 4x7 theta matrix
final_cheby_3 = @time solve_system(k_grid500, z[1], exp.(z[2]), initial, 3)

# Euler Equation Error (function from the previous problem set)
eee_cheby_3 = EEE_cheby(final_cheby_3[2], exp.(z[2]), z[1], k_grid500, final_cheby_3[3], final_cheby_3[1], 3)

mean(eee_cheby_3)

plot(k_grid500,final_cheby_3[4],labels=false,xlabel="capital",title="Value Function \n Chebyshev third degree polynomial")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps3//plot_1.png")
plot(k_grid500,final_cheby_3[2],labels=false,xlabel="capital", ylabel = "consumption", title="Consumption Function \n Chebyshev third degree polynomial")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps3//plot_2.png")
plot(k_grid500,final_cheby_3[3],labels=false,xlabel="capital", ylabel="capital",title="Capital Function \n Chebyshev third degree polynomial")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps3//plot_3.png")
plot(k_grid500,eee_cheby_3, labels=false,xlabel="capital", ylabel="errors",title="Euler Erros \n Chebyshev third degree polynomial")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps3//plot_4.png")


######### Question 2

# The problem is similar to the last one, changing the polynomial and the collocation points
# I will try to use the same nomenclature from the previous question

# function to define the intervals
# we have n_ki - 1 intervals
function FE_intervals(k_grid, n_ki = 10)

    dif = trunc(Int, round(length(k_grid)/(n_ki-1)));
    k_limits = zeros(n_ki);
    k_limits[1] = k_grid[1];
    k_limits[end] = k_grid[end];

    for ik in 1:n_ki-2
        k_limits[ik+1] = k_grid[ik*dif + 1] 
    end
    return k_limits
end

# phi_i functions
function phi_i(k, i, intervals)
    if i == 1 #phi_1
        if k < intervals[1]
            p = 1
        elseif k >= intervals[i] && k <= intervals[i+1]
            p = (intervals[i+1] - k)/(intervals[i+1] - intervals[i])
        else 
            p = 0
        end
    elseif i == length(intervals) #phi_n
        if k > intervals[end]
            p = 1
        elseif k >= intervals[i-1] && k <= intervals[i]
            p = (k - intervals[i-1])/(intervals[i] - intervals[i-1])
        else 
            p = 0
        end
    else #phi_i
        if k >= intervals[i-1] && k <= intervals[i]
            p = (k - intervals[i-1])/(intervals[i] - intervals[i-1])
        elseif k >= intervals[i] && k <= intervals[i+1]
            p = (intervals[i+1] - k)/(intervals[i+1] - intervals[i])
        else 
            p = 0
        end
    end
    return p
end


# new consumption function using the phi_i function
function c_hat_FE(theta, k, intervals, n_ki = 10)
    c = 0
    for ik in 1:n_ki
        c = c + theta[ik]*phi_i(k, ik, intervals);
    end
    return c
end


function resid_FE(theta, k0, k_grid, markov_matrix, z_grid, n_ki = 10, alpha = 1/3, beta = 0.987, delta = 0.012, gamma = 2)
    
    nz = length(z_grid);
    R = zeros(nz);
    intervals = FE_intervals(k_grid, n_ki); 
    
    for iz in 1:nz
        c0 = c_hat_FE(theta[:,iz], k0, intervals, n_ki);
        k1 = z_grid[iz]*k0^alpha+(1-delta)*k0 - c0;
        c1 = zeros(nz); 
        for izz in 1:nz
            c1[izz] = c_hat_FE(theta[:,izz], k1, intervals, n_ki); # vector of tomorrow cunsumptions based on each state 
        end
    konst = alpha*z_grid[iz]*k1^(alpha-1)+(1-delta)

    R[iz] = (c0^(-gamma))^(-1)*beta*(markov_matrix[iz,:]'*(c1).^(-gamma)*konst)-1 # Euler Equation
    end
    return R'
end

function final_f_FE(theta, k_grid, markov_matrix, z_grid, n_ki = 10, alpha = 1/3, beta = 0.987, delta = 0.012, gamma = 2)    
    k0 = FE_intervals(k_grid, n_ki);
    nz = length(z_grid);
    F = zeros(n_ki, nz); 

    for id in 1:n_ki
        F[id, :] = resid_FE(theta, k0[id], k_grid, markov_matrix, z_grid, n_ki, alpha, beta, delta, gamma) # cheby_resid returns a vector of size nz
    end
    return F
end

# we have a n_ki x nz theta matrix, here i'm using n_ki = 10, so I have 9 intervals 
ini1 = collect(range(start = 1, stop = 10, length = 10))

initial1 = [ini1 ini1 ini1 ini1 ini1 ini1 ini1]

# testing a different initial guess
ini2 = collect(range(start = 3, stop = 5, length = 10))

initial2 = [ini2 ini2 ini2 ini2 ini2 ini2 ini2]


function solve_system_FE(k_grid, markov_matrix, z_grid, theta0, n_ki = 10, alpha = 1/3, beta = 0.987, delta = 0.012, gamma = 2)
    nz = length(z_grid);
    nk = length(k_grid);
    intervals = FE_intervals(k_grid, n_ki);
    
    @time begin # counting time just for the method
        x = nlsolve(theta -> final_f_FE(theta, k_grid500, markov_matrix, z_grid, n_ki), theta0)
    end    

    consumption = zeros(nk, nz);
    capital = zeros(nk, nz);
    value_function = zeros(nk, nz);
    v_old = zeros(nk, nz);

    # calculating the politicals functions
    Threads.@threads for iz in 1:nz
        for ik in 1:nk
            consumption[ik, iz] = c_hat_FE(x.zero[:, iz], k_grid[ik], intervals, n_ki)
            capital[ik, iz] = z_grid[iz]*k_grid[ik]^alpha+(1-delta)*k_grid[ik] - consumption[ik, iz];
#            kk = argmin(abs.(k_grid .- capital[ik, iz]))
#            capital[ik, iz] = k_grid[kk]; # the capital in the value function is not in the grid, so I find the closest
        end # capital
    end # state

    # calculating the value function
    for j in 1:maxiter
        Threads.@threads for iz in 1:nz
            for ik in 1:nk
                kk = argmin(abs.(k_grid .- capital[ik, iz])) # finding the index of the next capital
                value_function[ik, iz] = util(consump(k_grid[ik], z_grid[iz], k_grid[kk], alpha, delta), gamma) + beta * markov_matrix[iz,:]' * v_old[kk,:];
            end # capital
        end # state    
        
        if maximum(abs.(value_function - v_old)) > toler                    
            v_old = inertia .* v_old .+ (1-inertia) .* copy(value_function)
        else
            break
        end # tolerance
    end # iterations

    return x.zero, consumption, capital, value_function
end

final_fe_10 = @time solve_system_FE(k_grid500, z[1], exp.(z[2]), initial1, 10)

final_fe2_10 = @time solve_system_FE(k_grid500, z[1], exp.(z[2]), initial2, 10)

final_fe5000 = @time solve_system_FE(k_grid5000, z[1], exp.(z[2]), initial1, 10)

eee_fe_10 = EEE_FE(final_fe_10[2], exp.(z[2]), z[1], k_grid500, final_fe_10[3], final_fe_10[1], 10)
mean(eee_fe_10)

eee_fe2_10 = EEE_FE(final_fe2_10[2], exp.(z[2]), z[1], k_grid500, final_fe2_10[3], final_fe2_10[1], 10)
mean(eee_fe2_10)

#eee_fe5000 = EEE_FE(final_fe5000[2], exp.(z[2]), z[1], k_grid500, final_fe5000[3], final_fe5000[1], 10)
#mean(eee_fe5000)

plot(k_grid500, final_fe_10[4],labels=false,xlabel="capital",title="Value Function \n FE Collocation (9 intervals)")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps3//plot_5.png")
plot(k_grid500, final_fe_10[2],labels=false,xlabel="capital", ylabel = "consumption", title="Consumption Function \n FE Collocation (9 intervals)")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps3//plot_6.png")
plot(k_grid500, final_fe_10[3],labels=false,xlabel="capital", ylabel="capital",title="Capital Function \n FE Collocation (9 intervals)")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps3//plot_7.png")
plot(k_grid500,eee_fe_10, labels=false,xlabel="capital", ylabel="errors",title="Euler Erros \n FE Collocation (9 intervals)")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps3//plot_8.png")

# Now I will resolve the problem using the Galerkin method

function E_i(theta, k_grid, markov_matrix, z_grid, i, intervals, n = 10, n_ki = 10, alpha = 1/3, beta = 0.987, delta = 0.012, gamma = 2)    
    (k0, weights) = gausslegendre(n)
    if i == 1 #E_1
        a = intervals[i]
        b = intervals[i+1];
        x = k0.*(b-a)./2 .+ (a+b)/2
        R = zeros(n, nz)
        for iin in 1:n 
            p = (b - x[iin])/(b - a) 
            R[iin,:] = p.*resid_FE(theta, x[iin], k_grid, markov_matrix, z_grid, n_ki, alpha, beta, delta, gamma)
        end
        E = (b-a)/2 .* weights' * R
    
    elseif i == n_ki #E_n_ki
        a = intervals[i-1];
        b = intervals[i];
        x = k0.*(b-a)./2 .+ (a+b)/2
        R = zeros(n, nz)
        for iin in 1:n 
            p = (x[iin]-a)/(b - a) 
            R[iin,:] = p.*resid_FE(theta, x[iin], k_grid, markov_matrix, z_grid, n_ki, alpha, beta, delta, gamma)
        end
        E = (b-a)/2 .* weights' * R
     
    else #E_i
        a = intervals[i-1]
        b = intervals[i]
        c = intervals[i+1];
        x_i = k0.*(b-a)./2 .+ (a+b)/2
        x_s = k0.*(c-b)./2 .+ (b+c)/2
        R_i = zeros(n, nz)
        for iin in 1:n 
            p = (x_i[iin] - a)/(b - a) 
            R_i[iin,:] = p.*resid_FE(theta, x_i[iin], k_grid, markov_matrix, z_grid, n_ki, alpha, beta, delta, gamma)
        end
        E_i = (b-a)/2 .* weights' * R_i
        R_s = zeros(n, nz)
        for iin in 1:n 
            p = (c - x_s[iin])/(c - b) 
            R_s[iin,:] = p.*resid_FE(theta, x_s[iin], k_grid, markov_matrix, z_grid, n_ki, alpha, beta, delta, gamma)
        end
        E_s = (c-b)/2 .* weights' * R_s
        E = E_i + E_s
    end
    return E'
end

function final_f_ga(theta, k_grid, markov_matrix, z_grid, n = 10, n_ki = 10, alpha = 1/3, beta = 0.987, delta = 0.012, gamma = 2)    
    nz = length(z_grid);
    F = zeros(n_ki, nz); 
    intervals = FE_intervals(k_grid, n_ki)

    for id in 1:n_ki
        F[id, :] = E_i(theta, k_grid, markov_matrix, z_grid, id, intervals, n, n_ki, alpha, beta, delta, gamma)
    end
    return F
end


function solve_system_ga(k_grid, markov_matrix, z_grid, theta0, n = 10, n_ki = 10, alpha = 1/3, beta = 0.987, delta = 0.012, gamma = 2)
    nz = length(z_grid);
    nk = length(k_grid);
    intervals = FE_intervals(k_grid, n_ki);
    
    @time begin # counting time just for the method
        x = nlsolve(theta -> final_f_ga(theta, k_grid500, markov_matrix, z_grid, n, n_ki), theta0)
    end    

    consumption = zeros(nk, nz);
    capital = zeros(nk, nz);
    value_function = zeros(nk, nz);
    v_old = zeros(nk, nz);

    # calculating the politicals functions
    Threads.@threads for iz in 1:nz
        for ik in 1:nk
            consumption[ik, iz] = c_hat_FE(x.zero[:, iz], k_grid[ik], intervals, n_ki)
            capital[ik, iz] = z_grid[iz]*k_grid[ik]^alpha+(1-delta)*k_grid[ik] - consumption[ik, iz];
#            kk = argmin(abs.(k_grid .- capital[ik, iz]))
#            capital[ik, iz] = k_grid[kk]; # the capital in the value function is not in the grid, so I find the closest
        end # capital
    end # state

    # calculating the value function
    for j in 1:maxiter
        Threads.@threads for iz in 1:nz
            for ik in 1:nk
                kk = argmin(abs.(k_grid .- capital[ik, iz])) # finding the index of the next capital
                value_function[ik, iz] = util(consump(k_grid[ik], z_grid[iz], k_grid[kk], alpha, delta), gamma) + beta * markov_matrix[iz,:]' * v_old[kk,:];
            end # capital
        end # state    
        
        if maximum(abs.(value_function - v_old)) > toler                    
            v_old = inertia .* v_old .+ (1-inertia) .* copy(value_function)
        else
            break
        end # tolerance
    end # iterations

    return x.zero, consumption, capital, value_function
end


f_ga_10_10 = @time solve_system_ga(k_grid500, z[1], exp.(z[2]), initial1, 10, 10)

eee_ga_10_10 = EEE_FE(f_ga_10_10[2], exp.(z[2]), z[1], k_grid500, f_ga_10_10[3], f_ga_10_10[1], 10)
mean(eee_ga_10_10)

plot(k_grid500, f_ga_10_10[4],labels=false,xlabel="capital",title="Value Function \n FE Galerkin (9 intervals and 10 points quadrature)")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps3//plot_9.png")
plot(k_grid500, f_ga_10_10[2],labels=false,xlabel="capital", ylabel = "consumption", title="Consumption Function \n FE Galerkin (9 intervals and 10 points quadrature)")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps3//plot_10.png")
plot(k_grid500, f_ga_10_10[3],labels=false,xlabel="capital", ylabel="capital",title="Capital Function \n FE Galerkin (9 intervals and 10 points quadrature)")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps3//plot_11.png")
plot(k_grid500, eee_ga_10_10, labels=false,xlabel="capital", ylabel="errors",title="Euler Erros \n FE Galerkin (9 intervals and 10 points quadrature)")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps3//plot_12.png")

f_ga2 = @time solve_system_ga(k_grid500, z[1], exp.(z[2]), initial2, 10, 10)

plot(k_grid500, f_ga2[2])

plot(k_grid500, f_ga2[3])

plot(k_grid500, f_ga2[4])


