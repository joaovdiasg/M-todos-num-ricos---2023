##########################################
#### Lista 4 - Métodos Numéricos 2023 ####
##########################################

include("functions.jl") # I opted to put all secondary functions in an attached file
using Pkg, LinearAlgebra, Plots, Distributions, Random, StatsKit, Roots, SparseArrays

using(Latexify)

const nz = 9 # number of states
const nk = 500 # number of grid points
const m = 3 # number of sds
const mu = 0 # mean of the ar process
const beta = 0.96 # discount rate
const delta = 0.012 # depreciation 
const alpha = 1/3 # relative participation of capital
const inertia = 0 # inertia for the value function iteration
const toler = 10^-5 # tolerance for the value function iteration
const maxiter = 10000 # maximum number of value function iterations

Random.seed!(7)

########### Question 1

## Limite natural: a' >= - min_s{y_s}/r
## Hugget: E_λ[a'] = 0

const rho1 = 0.9 # persistence
const sigma1 = 0.01 # sd of the error
const gamma1 = 1.0001 # relative risk aversion

z = Tauchen(rho1, nz, sigma1);
latexify(round.(z[2], digits=2))
latexify(round.(z[1], digits=2))
# z[1] is the markov trasition probabilty matrix and z[2] the state grid 

# since we need an ad hoc debt limit I used a modified natural limit, I changed r by beta/1-beta, using the complete markets result
# eps is a very small number so the exact natural limit is not available to the agent
eps = 0.001
phi = beta*minimum(exp.(z[2]))/(1-beta) - eps

a_grid500 = collect(range(start = -phi, stop = phi, length = nk));

# solving a hugget model
function hugget_vfi(a_grid, z_grid, markov_matrix, r0, v0 = 0, beta = 0.987, gamma = 2, inertia = 0, maxiter = 10000, toler = 10^-5)

    nk = length(a_grid);
    nz = length(z_grid);
    ns = nk*nz;
    capital = zeros(nk, nz);
    v_new = zeros(nk, nz);
    itvfi = 0;
    if v0 == 0 # to facilitate the input in the function
        v_old = zeros(nk, nz);
        for iz in 1:nz
            for ia in 1:nk
                v_old[ia,iz] = util((exp(z_grid[iz]) + r0*a_grid[ia])/(1-beta), gamma) # Ben Moll's initial guess (a = a')
            end
        end
    else
        v_old = copy(v0);
    end
        
    @time begin
        for max in 1:maxiter
            # the function will maximize in the 100 first iterations and after that from 10 to 10 iterations only
            if mod(max,10) == 0 || max < 100 
                Threads.@threads for iz in 1:nz # state
                    mon = 1; # monotonicity
                    for ia in 1:nk # capital
                        h_0 = -Inf # initial value to use concavity, using infinity because now we have very nagative value functions
                        for iaa in mon:nk # next capital
                            h_temp = util((exp(z_grid[iz]) + (1+r0)*a_grid[ia] - a_grid[iaa]), gamma) + beta * markov_matrix[iz,:]' * v_old[iaa,:]
                            if h_temp < h_0 # testing because v is concave
                                mon = iaa-1 # update the monotonicity index
                                capital[ia, iz] = a_grid[iaa-1] # saving the k'
                                v_new[ia, iz] = h_0 # updating the value function
                                break
                            else # keep going
                                capital[ia, iz] = a_grid[iaa] 
                                h_0 = copy(h_temp) # updating the last value for concavity
                                v_new[ia,iz] = h_temp
                            end # concavity
                        end # k' and monotonicity
                    end # capital
                end # state

            else # accelerator
                # when the function is not maximized
                Threads.@threads for iz in 1:nz
                    for ia in 1:nk
                        aa = findfirst(x -> x == capital[ia, iz], a_grid)
                        h_temp = util((exp(z_grid[iz]) + (1+r0)*a_grid[ia] - a_grid[aa]), gamma) + beta * markov_matrix[iz,:]' * v_old[aa,:]
                        v_new[ia, iz] = h_temp; # value function
                    end # capital
                end # state
            end # accelerator
            
            if maximum(abs.(v_new - v_old)) > toler
                v_old = inertia .* v_old .+ (1-inertia) .* copy(v_new)
                itvfi += 1 # iterations counter
            else
                break
            end # tolerance

        end # iterations

    #calculating the stationary distribution
    # reshaping the a' grid
    grid_aa = kron(ones(1,nz), a_grid')

    # reshaping the capital police
    grid_capital = kron(capital, ones(1,nk))
    grid_capital = kron(ones(nz,1), grid_capital)

    A = spzeros(ns,ns);
    for s in 1:ns
        A[s,:] = grid_aa .== grid_capital[s,:]'; # puts a 1 if g(a,s)=a' 
    end        

    PP = kron(markov_matrix, ones(nk,nk))
    Q = A .* PP # transition markov chain

    # solving the linear system to find the eigenvector associated to the unitary eigenvalue
    v = [1; (I-Q[2:end, 2:end]')\Vector(Q'[2:end, 1])] 
    v = v./sum(v) # normalizing to sum one
    v = reshape(v, nk, nz)

    end #time
 return v_new, capital, v
end # function

function final_f(a_grid, z_grid, markov_matrix, r0, v0 = 0, beta = 0.987, gamma = 2, inertia = 0, maxiter = 10000, toler = 10^-5)

    nk = length(a_grid);
    nz = length(z_grid);

    # Now let us see the total demand for capital and adjust the interest rate
    result = hugget_vfi(a_grid, z_grid, markov_matrix, r0, v0, beta, gamma, inertia, maxiter, toler)
    lambda = result[3] # stationary distribution
    capital = result[2] # capital police function
    
    agg_demand = 0
    for iz in 1:nz # z
        for ia in 1:nk # a
            agg_demand = agg_demand + lambda[ia, iz]*capital[ia,iz]
        end # a
    end # z
    return agg_demand
end


function solve_hugget(a_grid, z_grid, markov_matrix, initial, v0 = 0, beta = 0.987, gamma = 2, inertia = 0, maxiter = 10000, toler = 10^-5)
    
    nk = length(a_grid);
    nz = length(z_grid);

    r = @time find_zero(r0 -> final_f(a_grid, z_grid, markov_matrix, r0, v0, beta, gamma, inertia, maxiter, toler),
                                     initial, Bisection(), verbose = true)

    result = hugget_vfi(a_grid, z_grid, markov_matrix, r, v0, beta, gamma, inertia, maxiter, toler)
    value_f = result[1];
    capital = result[2];
    lambda = result[3];

    consumption = zeros(nk, nz);

    Threads.@threads for iz in 1:nz # calculating the political function
        for ia in 1:nk
            consumption[ia, iz] = exp(z_grid[iz]) + (1+r)*a_grid[ia] - capital[ia, iz];
        end
    end
    return r, value_f, capital, consumption, lambda
end


###########################################################################
# B)

r_complete = 1/beta -1

grid_r = collect(range(start = 0.035, stop = r_complete, length = 50))

function demand_curve(a_grid, z_grid, markov_matrix, grid_r, v0 = 0, beta = 0.987, gamma = 2, inertia = 0, maxiter = 10000, toler = 10^-5)

    nr = length(grid_r)
    demand = zeros(nr)

    for ir in 1:nr
        demand[ir] = final_f(a_grid, z_grid, markov_matrix, grid_r[ir], v0, beta, gamma, inertia, maxiter, toler)
    end
    return demand
end

demand_f = @time demand_curve(a_grid500, z[2], z[1], grid_r, 0, beta, gamma1)

plot(demand_f,grid_r, xlims=(-25,5), labels=false,xlabel="demand",ylabel="rate",title="Excess demand of bonds")
hline!([final_result[1]], linestyle=:dash, labels="Complete markets rate", legend=:bottomleft)
vline!([0], linestyle=:dash, lc=:black, labels=false)
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_1.png")


initial = (0.96*r_complete, r_complete)

item_b = hugget_vfi(a_grid500, z[2], z[1], 0.04, 0, beta, gamma1)
e_demand_b = final_f(a_grid500, z[2], z[1], 0.04, 0, beta, gamma1)

consumption_b = zeros(nk, nz);

Threads.@threads for iz in 1:nz # calculating the political function
    for ia in 1:nk
        consumption_b[ia, iz] = exp(z[1][iz]) + (1+0.04)*a_grid500[ia] - item_b[2][ia, iz];
    end
end

plot(a_grid500, item_b[1], labels=false,xlabel="capital",title="Value Function")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_2.png")
plot(a_grid500, item_b[2], labels=false,xlabel="capital",title="Capital police")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_3.png")
plot(a_grid500, consumption_b, labels=false,xlabel="capital",title="Consumption police")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_4.png")

###########################################################################
# C)

# 3.5 minutes
final_result = @time solve_hugget(a_grid500, z[2], z[1], initial, 0, beta, gamma1)

eee_1 = EEE_H(final_result[4], final_result[1], z[2], z[1], a_grid500, final_result[3], beta, gamma1)
mean(eee_1)

final_result[1]
plot(a_grid500, final_result[2], labels=false,xlabel="capital",title="Value Function")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_5.png")
plot(a_grid500, final_result[3], labels=false,xlabel="capital",title="Capital police")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_6.png")
plot(a_grid500, final_result[4], labels=false,xlabel="capital",title="Consumption police")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_7.png")
plot(a_grid500, final_result[5], labels=false,xlabel="capital",title="Stationary distribution")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_8.png")
plot(a_grid500, sum(final_result[5], dims = 2), labels=false,xlabel="capital",title="PDF stationary distribution")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_9.png")
plot(a_grid500, eee_1, labels=false,xlabel="capital",title="EEE")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_10.png")

###########################################################################
# D)

# just changing the persistence parameter as asked

rho2 = 0.97

z_2 = Tauchen(rho2, nz, sigma1);
# z[1] is the markov trasition probabilty matrix and z[2] the state grid 

phi2 = beta*minimum(exp.(z_2[2]))/(1-beta) - eps

a_grid500_2 = collect(range(start = -phi2, stop = phi2, length = nk))

# 3.5 minutes
final_result2 = @time solve_hugget(a_grid500_2, z_2[2], z_2[1], initial, 0, beta, gamma1)

eee_2 = EEE_H(final_result2[4], final_result2[1], z_2[2], z_2[1], a_grid500_2, final_result2[3], beta, gamma1)
mean(eee_2)

final_result2[1]
plot(a_grid500_2, final_result2[2], labels=false,xlabel="capital",title="Value Function")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_11.png")
plot(a_grid500_2, final_result2[3], labels=false,xlabel="capital",title="Capital police")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_12.png")
plot(a_grid500_2, final_result2[4], labels=false,xlabel="capital",title="Consumption police")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_13.png")
plot(a_grid500_2, final_result2[5], labels=false,xlabel="capital",title="Stationary distribution")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_14.png")
plot(a_grid500_2, sum(final_result2[5], dims = 2), labels=false,xlabel="capital",title="PDF stationary distribution")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_15.png")
plot(a_grid500, eee_2, labels=false,xlabel="capital",title="EEE")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_16.png")


###########################################################################
# E)

# just changing the risk aversion parameter

gamma2 = 5

# 4 minutes
final_result3 = @time solve_hugget(a_grid500, z[2], z[1], initial, 0, beta, gamma2)

eee_3 = EEE_H(final_result3[4], final_result3[1], z[2], z[1], a_grid500, final_result3[3], beta, gamma1)
mean(eee_3)

final_result3[1]
plot(a_grid500[100:500], final_result3[2][100:500,:], labels=false,xlabel="capital",title="Value Function")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_17.png")
plot(a_grid500, final_result3[3], labels=false,xlabel="capital",title="Capital police")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_18.png")
plot(a_grid500, final_result3[4], labels=false,xlabel="capital",title="Consumption police")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_19.png")
plot(a_grid500, final_result3[5], labels=false,xlabel="capital",title="Stationary distribution")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_20.png")
plot(a_grid500, sum(final_result3[5], dims = 2), labels=false,xlabel="capital",title="PDF stationary distribution")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_21.png")
plot(a_grid500, eee_3, labels=false,xlabel="capital",title="EEE")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_22.png")

###########################################################################
# F)

# just changing the volatility fo the shock

sigma2 = 0.05

z_3 = Tauchen(rho1, nz, sigma2);
# z[1] is the markov trasition probabilty matrix and z[2] the state grid 

phi3 = beta*minimum(exp.(z_3[2]))/(1-beta) - eps

a_grid500_3 = collect(range(start = -phi3, stop = phi3, length = nk))

# 4 minuts
final_result4 = @time solve_hugget(a_grid500_3, z_3[2], z_3[1], initial, 0, beta, gamma1)

eee_4 = EEE_H(final_result4[4], final_result4[1], z_3[2], z_3[1], a_grid500_3, final_result4[3], beta, gamma1)
mean(eee_4)

final_result4[1]
plot(a_grid500_3, final_result4[2], labels=false,xlabel="capital",title="Value Function")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_23.png")
plot(a_grid500_3, final_result4[3], labels=false,xlabel="capital",title="Capital police")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_24.png")
plot(a_grid500_3, final_result4[4], labels=false,xlabel="capital",title="Consumption police")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_25.png")
plot(a_grid500_3, final_result4[5], labels=false,xlabel="capital",title="Stationary distribution")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_26.png")
plot(a_grid500_3, sum(final_result4[5], dims = 2), labels=false,xlabel="capital",title="PDF stationary distribution")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_27.png")
plot(a_grid500, eee_4, labels=false,xlabel="capital",title="EEE")
savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps4//plot_28.png")

###########################################################################

# constructing a demand demand curve for the assets

inf = collect(range(start = 0.039, stop = final_result[1], length = 20))
sup = collect(range(start = final_result[1], stop = r_complete, length = 20))
grid_r2 = vcat(inf, sup[2:end]);

demand_f2 = @time demand_curve(a_grid500, z[2], z[1], grid_r, 0, beta, gamma1)

plot(demand_f2,grid_r, xlims=(-25,10))
hline!([final_result[1]], linestyle=:dash)
vline!([0], linestyle=:dash)