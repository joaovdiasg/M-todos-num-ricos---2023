##########################################
#### Lista 2 - Métodos Numéricos 2023 ####
##########################################

include("functions.jl") # I opted to put all secondary functions in an attached file

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

########### Question 1

const k_ss = (beta*alpha/(1-beta*(1-delta)))^(1/(1-alpha));

k_grid500 = range(start = 0.75*k_ss, stop = 1.25*k_ss, length = nk500);

z = Tauchen(rho, nz, sigma);
# z[1] is the markov trasition probabilty matrix and z[2] the state grid 
# we need to exponensiate the state grid before using, once we have a log normal process

#lap(round.(z[1], digits=5))
#lap(round.(z[2], digits=5))

utility = zeros(nk500, nz, nk500); # creating a 3d array

# Creating a 3d array for all the possible combinations of k, k' and z. Filling it whit tha values applied to the utility function. 
for ik in 1:nk500 # for k
    for iz in 1:nz # for k'
        for ikk in 1:nk500 # for z
            utility[ik, iz, ikk] = util(consump(k_grid500[ik], exp(z[2][iz]), k_grid500[ikk], alpha, delta), gamma);
        end
    end
end


#### Euler Equation Error (EEE)

function EEE(c, z_grid, markov_matrix, k_grid, capital, alpha = 1/3, beta = 0.987, delta = 0.012, gamma = 2)
    nk = length(k_grid);
    nz = length(z_grid);
    eee = zeros(nk, nz);

    Threads.@threads for iz in 1:nz # calculating the EEE
        for ik in 1:nk    
            kk = findfirst(x -> x == capital[ik, iz], k_grid)
            cc = deriv_util.(c[kk,:], gamma);
            konst = z_grid[iz]*alpha*(k_grid[kk])^(alpha-1) + (1-delta)
            eee[ik, iz] = log10(abs(1 - (1/c[ik,iz])*inv_deriv_util((
                                beta * markov_matrix[iz,:]' * cc * konst), gamma)))
        end
    end
    return eee
end

# brute force method
function brute_force(utility, k_grid, z_grid, markov_matrix, v0 = 0, alpha = 1/3, beta = 0.987, delta = 0.012, inertia = 0, maxiter = 10000, toler = 10^-5) 

    nk = length(k_grid);
    nz = length(z_grid);
    v_temp = zeros(nk, nz, nk);
    v_new = zeros(nk, nz);
    consumption = zeros(nk, nz);
    capital = zeros(nk, nz);
    it = 0;
    if v0 == 0
        v_old = zeros(nk, nz);
    else
        v_old = copy(v0);
    end

    @time begin
        for j in 1:maxiter
            for iz in 1:nz
                for ik in 1:nk
                    v_temp[ik, iz, :] = utility[ik, iz, :] + beta .* (markov_matrix[iz, :]' * v_old')'; # maximization of the Bellman equation
                    kk = argmax(v_temp[ik, iz, :]) # doing that way to avoid using max function twice
                    capital[ik, iz] = k_grid[kk]; # chosen capital for tomorrow
                    v_new[ik, iz] = v_temp[ik, iz, kk]; # value function
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
    end # time

    return  v_new, capital, consumption, println("
    Iterações: ", it, ". O primeiro elemento dessa função retorna a função valor, o segundo a sequência de capitais escolhidos e o terceiro retorna a função política do consumo.
    ")
end

# because I used vectorized the problem, I couldn't parallelize the code

bf = brute_force(utility, k_grid500, exp.(z[2]), z[1])

eee_bf = EEE(bf[3], exp.(z[2]), z[1], k_grid500, bf[2])
mean(eee_cm)

# brute force method exploring the concavity and the monotonicity of the value function
function concave_mon(k_grid, z_grid, markov_matrix, v0 = 0, alpha = 1/3, beta = 0.987, delta = 0.012, gamma = 2, inertia = 0, maxiter = 10000, toler = 10^-5)

    nk = length(k_grid);
    nz = length(z_grid);
    capital = zeros(nk, nz);
    v_new = zeros(nk, nz);
    consumption = zeros(nk, nz);
    it = 0;
    if v0 == 0
        v_new = zeros(nk, nz);
        for iz in 1:nz
            for ik in 1:nk
                v_old[ik,iz] = util(consump(k_grid[ik], z_grid[iz], k_grid[ik], alpha, delta)/(1-beta), gamma); # Ben Moll's initial guess (k = k')
            end
        end
    else
        v_old = copy(v0);
    end
        
    @time begin
        for j in 1:maxiter
            Threads.@threads for iz in 1:nz # state
                mon = 1; # monotonicity
                for ik in 1:nk # capital
                    h_0 = 0 # initial value to use concavity
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

cm = concave_mon(k_grid500, exp.(z[2]), z[1])


eee_cm = EEE(cm[3], exp.(z[2]), z[1], k_grid500, cm[2])
mean(eee_cm)

plot(k_grid500,bf[1],title="Função Valor cm")
plot(k_grid500,bf[3],title="Função Consumo cm")
plot(k_grid500,bf[2],title="Função Política cm")

plot(k_grid500,cm[1],labels=false,xlabel="capital",title="Value Function - grid_500 (brute force)")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps2//plot_1.png")
plot(k_grid500,cm[3],labels=false,xlabel="capital", ylabel = "consumption", title="Consumption Function - grid_500 (brute force)")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps2//plot_2.png")
plot(k_grid500,cm[2],labels=false,xlabel="capital", ylabel="capital",title="Capital Function - grid_500 (brute force)")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps2//plot_3.png")
plot(k_grid500,eee_cm, labels=false,xlabel="capital", ylabel="errors",title="Euler Erros - grid_500 (brute force)")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps2//plot_4.png")

### Accelerator

# brute force method
# this functions accepts both using the accelerator and the normal version
function brute_force_acc(utility, k_grid, z_grid, markov_matrix, accelerator = 0, v0 = 0, alpha = 1/3, beta = 0.987, delta = 0.012, inertia = 0, maxiter = 10000, toler = 10^-5) 

    nk = length(k_grid);
    nz = length(z_grid);
    v_temp = zeros(nk, nz, nk);
    v_new = zeros(nk, nz);
    consumption = zeros(nk, nz);
    capital = zeros(nk, nz);
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
                for iz in 1:nz
                    for ik in 1:nk
                        v_temp[ik, iz, :] = utility[ik, iz, :] + beta .* (markov_matrix[iz, :]' * v_old')'; # maximization of the Bellman equation
                        kk = argmax(v_temp[ik, iz, :]) # doing that way to avoid using max function twice
                        capital[ik, iz] = k_grid[kk]; # chosen capital for tomorrow
                        v_new[ik, iz] = v_temp[ik, iz, kk]; # value function
                    end # capital
                end # state

                if maximum(abs.(v_new - v_old)) > toler
                    
                    v_old = inertia .* v_old .+ (1-inertia) .* copy(v_new)
                    it += 1 # iterations counter

                else

                    break
                    
                end # tolerance
            else
                for iz in 1:nz
                    for ik in 1:nk
                        v_temp[ik, iz, :] = utility[ik, iz, :] + beta .* (markov_matrix[iz, :]' * v_old')'; # maximization of the Bellman equation
                        kk = findfirst(x -> x == capital[ik, iz], k_grid)
                        v_new[ik, iz] = v_temp[ik, iz, kk]; # value function
                    end # capital
                end # state

                if maximum(abs.(v_new - v_old)) > toler
                    
                    v_old = inertia .* v_old .+ (1-inertia) .* copy(v_new)
                    it += 1 # iterations counter

                else

                    break
                    
                end # tolerance
            end # accelerator
        end # iterations

        Threads.@threads for iz in 1:nz # calculating the political function
            for ik in 1:nk
                consumption[ik, iz] = consump(k_grid[ik], z_grid[iz], capital[ik,iz], alpha, delta);
            end
        end
    end # time

    return  v_new, capital, consumption, println("
    Iterações: ", it, ". O primeiro elemento dessa função retorna a função valor, o segundo a sequência de capitais escolhidos e o terceiro retorna a função política do consumo.
    ")
end

bf_acc = brute_force_acc(utility, k_grid500, exp.(z[2]), z[1], 1)

eee_bf_acc = EEE(bf_acc[3], exp.(z[2]), z[1], k_grid500, bf_acc[2])
mean(eee_bf_acc)

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

cm_acc = concave_mon_acc(k_grid500, exp.(z[2]), z[1], 1)

eee_cm_acc = EEE(cm_acc[3], exp.(z[2]), z[1], k_grid500, cm_acc[2])
mean(eee_cm_acc)

plot(k_grid500,bf_acc[1],title="Função Valor")
plot(k_grid500,bf_acc[3],title="Função Consumo")
plot(k_grid500,bf_acc[2],title="Função Política")

plot(k_grid500,cm_acc[1],labels=false,xlabel="capital",title="Value Function - grid_500 (accelerator)")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps2//plot_5.png")
plot(k_grid500,cm_acc[3],labels=false,xlabel="capital", ylabel = "consumption", title="Consumption Function - grid_500 (accelerator)")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps2//plot_6.png")
plot(k_grid500,cm_acc[2],labels=false,xlabel="capital", ylabel="capital",title="Capital Function - grid_500 (accelerator)")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps2//plot_7.png")
plot(k_grid500,eee_cm_acc, labels=false,xlabel="capital", ylabel="errors",title="Euler Erros - grid_500 (accelerator)")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps2//plot_8.png")

### Multigrid


k_grid100 = range(start = 0.75*k_ss, stop = 1.25*k_ss, length = nk100);
k_grid5000 = range(start = 0.75*k_ss, stop = 1.25*k_ss, length = nk5000);

function brute_force_mult(k_grid, z_grid, markov_matrix, accelerator = 0, v0 = 0, alpha = 1/3, beta = 0.987, delta = 0.012, inertia = 0, maxiter = 10000, toler = 10^-5) 

    NK = length(k_grid);
    nz = length(z_grid);
    n1 = trunc(Int,length(k_grid[1]));
    acc = copy(accelerator);
    v_old = copy(v0);   
    
    for mult in 1:NK

        nk = trunc(Int,length(k_grid[mult]));
        utility = zeros(nk, nz, nk); 

        # Creating a 3d array for all the possible combinations of k, k' and z. Filling it whit tha values applied to the utility function. 
        for ik in 1:nk # for k
            for iz in 1:nz # for k'
                for ikk in 1:nk # for z
                    
                    utility[ik, iz, ikk] = util(consump(k_grid[mult][ik], z_grid[iz], k_grid[mult][ikk], alpha, delta), gamma);

                end
            end
        end

        result =  brute_force_acc(utility, k_grid[mult], z_grid, markov_matrix, acc, v_old)
        
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

GRID = [k_grid100, k_grid500, k_grid5000]

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

cm_mult_5000 = concave_mon_mult(GRID, exp.(z[2]), z[1]) 
eee_mult_5000 = EEE(cm_mult_5000[3], exp.(z[2]), z[1], k_grid5000, cm_mult_5000[2])
mean(eee_mult_5000)

cm_mult_acc_5000 = concave_mon_mult(GRID, exp.(z[2]), z[1], 1) 
eee_mult_acc_5000 = EEE(cm_mult_acc_5000[3], exp.(z[2]), z[1], k_grid5000, cm_mult_acc_5000[2])
mean(eee_mult_acc_5000)


plot(k_grid5000,cm_mult[1],labels=false,xlabel="capital",title="Value Function - grid_5000 (multigrid)")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps2//plot_9.png")
plot(k_grid5000,cm_mult[3],labels=false,xlabel="capital", ylabel = "consumption", title="Consumption Function - grid_5000 (multigrid)")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps2//plot_10.png")
plot(k_grid5000,cm_mult[2],labels=false,xlabel="capital", ylabel="capital",title="Capital Function - grid_5000 (multigrid)")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps2//plot_11.png")
plot(k_grid5000,eee_mult, labels=false,xlabel="capital", ylabel="errors",title="Euler Erros - grid_5000 (multigrid)")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps2//plot_12.png")

plot(k_grid5000,cm_mult_acc[1],labels=false,xlabel="capital",title="Value Function - grid_5000 (multigrid and accelerator)")
plot(k_grid5000,cm_mult_acc[3],labels=false,xlabel="capital", ylabel = "consumption", title="Consumption Function - grid_5000 (multigrid and accelerator)")
plot(k_grid5000,cm_mult_acc[2],labels=false,xlabel="capital", ylabel="capital",title="Capital Function - grid_5000 (multigrid and accelerator)")
plot(k_grid5000,eee_mult_acc, labels=false,xlabel="capital", ylabel="errors",title="Euler Erros - grid_5000 (multigrid and accelerator)")

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

e_grid_500 = endog_grid(k_grid500, exp.(z[2]), z[1])

e_grid_5000 = endog_grid(k_grid5000, exp.(z[2]), z[1])

# to calculate the Euler Error I need to transform the output of the function, so the capital function is on my capital exogenous grid

# eee_acc1 = EEE(e_grid_500[2], exp.(z[2]), z[1], k_grid500, e_grid_500[1])

plot(k_grid500,e_grid_500[2],labels=false,xlabel="capital", ylabel = "consumption", title="Consumption Function - grid_500 (endogenous grid)")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps2//plot_13.png")
plot(k_grid500,e_grid_500[1],labels=false,xlabel="capital", ylabel="capital",title="Capital Function - grid_500 (endogenous grid)")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps2//plot_14.png")

plot(k_grid5000,e_grid_5000[2],labels=false,xlabel="capital", ylabel = "consumption", title="Consumption Function - grid_5000 (endogenous grid)")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps2//plot_15.png")
plot(k_grid5000,e_grid_5000[1],labels=false,xlabel="capital", ylabel="capital",title="Capital Function - grid_5000 (endogenous grid)")
#savefig("G://Meu Drive//JOAO//Mestrado//Segundo ano//Metodos//plots_ps2//plot_16.png")

plot(k_grid5000,eee_mult, labels=false,xlabel="capital", ylabel="errors",title="Euler Erros - grid_5000 (multigrid)")
