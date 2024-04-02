##########################################
#### Lista 1 - Métodos Numéricos 2023 ####
##########################################

using Pkg, LinearAlgebra, Statistics, Plots, LaTeXStrings, Distributions, GLM, Random, DataFrames, LatexPrint, TexTables, RegressionTables

rho = 0.95 # persistence
sigma = 0.007 # sd of the error
N = 9 # number of states
n = 10000 # number of periodos
m = 3 # number of sds
mu = 0 # mean of the ar process

Random.seed!(7)

########### Question 1

## Tauchen

function Tauchen(rho, N, sigma, mu = 0.0, m = 3)

    theta_sup = m*sqrt(sigma^2/(1-rho^2)) # sigma is the sd of the error
    theta_grid = range(start = -theta_sup, stop = theta_sup, length = N)
    delta = (theta_grid[2]-theta_grid[1])/2

    P = zeros(N,N)
    x = Normal(0,1)

    for i in 1:N
        for j in 2:N-1
            P[i,1] = cdf(x, (-theta_sup + delta - rho*theta_grid[i]-(1-rho)*mu)/sigma)
            P[i,N] = 1 - cdf(x, (theta_sup - delta - rho*theta_grid[i]-(1-rho)*mu)/sigma)
            P[i,j] = cdf(x, (theta_grid[j] + delta - rho*theta_grid[i]-(1-rho)*mu)/sigma) - 
                     cdf(x, (theta_grid[j] - delta - rho*theta_grid[i]-(1-rho)*mu)/sigma)
        end 
    end            

    return [P, collect(theta_grid)] # using collect to create a vector

end

G = Rouwen(rho, N, sigma)
GG = cumsum(G[1], dims = 2)


lap(round.(Tauchen(rho, N, sigma)[1], digits=5)) # export matrix to latex
lap(round.(Tauchen(rho, N, sigma)[2], digits=5))

########### Question 2

## Rouwenhorst

function Rouwen(rho, N, sigma, mu=0.0)

    theta_sup = mu + sigma*sqrt((N-1)/(1-rho^2)) # sigma is the sd of the error
    theta_inf = mu - sigma*sqrt((N-1)/(1-rho^2))
    theta_grid = range(start = theta_inf, stop = theta_sup, length = N)

    p = (1+rho)/2

    P2 = [p 1-p 
        1-p p]

    if N == 2
    
    P_new = P2

    else
    
    P_old = P2

        for i in 3:N
            P_new = p*[P_old zeros(i-1,1); zeros(1, i)]+
                (1-p)*[zeros(i-1,1) P_old; zeros(1, i)]+
                (1-p)*[zeros(1,i); P_old zeros(i-1, 1)]+
                p*[zeros(1,i); zeros(i-1, 1) P_old]
            P_old = P_new
        end

        P_new = P_new .*(1 ./ sum(P_new, dims = 2)) #normalizing the rows to sum 1
    end

    return [P_new, collect(theta_grid)]   

end

lap(round.(Rouwen(rho, N, sigma)[1], digits=5)) 
lap(round.(Rouwen(rho, N, sigma)[2], digits=5))

########### Question 3

## Simulating the discrete process

z_cont_95 = zeros(n)

x = Normal(mu, sigma)

e = rand(x, n)

for i in 2:n
    z_cont_95[i] = rho*z_cont_95[i-1] + e[i]
end

function Disc_ar(rho, N, sigma, n, e, z0 = "median", mu=0, m=3, tauchen::Bool = true)

    if tauchen    
        PP = Tauchen(rho, N, sigma, mu, m)
    else    
        PP = Rouwen(rho, N, sigma, mu)
    end

    P_acum = cumsum(PP[1], dims = 2)

    z_disc = zeros(n)

    cdf_e = zeros(n)

    cdf_e[1] = cdf(x, e[1])

    if z0 == "median" # z0 is the initial state
        P_e =  trunc(Int, median(1:N)) # trunc is to mutate from float to int
    else 
        P_e =  trunc(Int, findfirst(x -> x <= z0, PP[2]))
    end 

    z_disc[1] = PP[2][P_e] 

    for i in 2:n
        cdf_e[i] = cdf(x, e[i])

        P_e_n = trunc(Int, findfirst(x -> x >= cdf_e[i], P_acum[P_e,:]))
    
        z_disc[i] = PP[2][P_e_n]

        P_e = P_e_n
    end

    return(z_disc)
end

Disc_ar_t_95 = Disc_ar(rho, N, sigma, n, e, "median", mu, m)
Disc_ar_r_95 = Disc_ar(rho, N, sigma, n, e, "median", mu, m, false)

savefig(plot([z_cont_95 Disc_ar_t_95], title = "rho = 0.95", label=["Continuous process" "Tauchen discrete process"]),
        "G:\\Meu Drive\\JOAO\\Mestrado\\Segundo ano\\Metodos\\plots_ps1\\plot_1")

savefig(plot([z_cont_95 Disc_ar_r_95], title = "rho = 0.95", label=["Continuous process" "Rouwenhorst discrete process"]),
        "G:\\Meu Drive\\JOAO\\Mestrado\\Segundo ano\\Metodos\\plots_ps1\\plot_2")        


########### question 4

est_t_95 = zeros(n)

for i in 2:n
    est_t_95[i] = Disc_ar_t_95[i-1]
end

est_r_95 = zeros(n)

for i in 2:n
    est_r_95[i] = Disc_ar_r_95[i-1]
end


data_t_95 = DataFrame(y = Disc_ar_t_95, x = est_t_95)
data_r_95 = DataFrame(y = Disc_ar_r_95, x = est_r_95)

reg_t_95 = lm(@formula(y ~ 0 + x), data_t_95)
reg_r_95 = lm(@formula(y ~ 0 + x), data_r_95)

regtable(reg_t_95, reg_r_95; renderSettings = latexOutput()) # export latex table

########### question 5

z_cont_99 = zeros(n)

for i in 2:n
    z_cont_99[i] = 0.99*z_cont_99[i-1] + e[i]
end


Disc_ar_t_99 = Disc_ar(0.99, N, sigma, n, e, "median", mu, m)
Disc_ar_r_99 = Disc_ar(0.99, N, sigma, n, e, "median", mu, m, false)

est_t_99 = zeros(n)

for i in 2:n
    est_t_99[i] = Disc_ar_t_99[i-1]
end

est_r_99 = zeros(n)

for i in 2:n
    est_r_99[i] = Disc_ar_r_99[i-1]
end


data_t_99 = DataFrame(y = Disc_ar_t_99, x = est_t_99)
data_r_99 = DataFrame(y = Disc_ar_r_99, x = est_r_99)

reg_t_99 = lm(@formula(y ~ 0 + x), data_t_99)
reg_r_99 = lm(@formula(y ~ 0 + x), data_r_99)

savefig(plot([z_cont_99 Disc_ar_t_99], title = "rho = 0.99", label=["Continuous process" "Tauchen discrete process"]),
        "G:\\Meu Drive\\JOAO\\Mestrado\\Segundo ano\\Metodos\\plots_ps1\\plot_5")

savefig(plot([z_cont_99 Disc_ar_r_99], title = "rho = 0.99", label=["Continuous process" "Rouwenhorst discrete process"]),
        "G:\\Meu Drive\\JOAO\\Mestrado\\Segundo ano\\Metodos\\plots_ps1\\plot_6")





        function Disc_ar(rho, N, sigma, n, e, mu=0, m=3)
   
            PP = Tauchen(rho, N, sigma, mu, m)
    
            P_acum = cumsum(PP[1], dims = 2)
        
            z_disc = zeros(n)
        
            cdf_e = zeros(n)
        
            cdf_e[1] = cdf(x, e[1])
            
            P_e =  trunc(Int, median(1:N)) # trunc is to mutate from float to int
            
            z_disc[1] = PP[2][P_e] 
        
            for i in 2:n
                cdf_e[i] = cdf(x, e[i])
        
                P_e_n = trunc(Int, findfirst(x -> x >= cdf_e[i], P_acum[P_e,:]))
            
                z_disc[i] = PP[2][P_e_n]
        
                P_e = P_e_n
            end
        
            return(z_disc)
        end