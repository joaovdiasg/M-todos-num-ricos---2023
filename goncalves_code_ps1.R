##########################################
#### Lista 1 - Métodos Numéricos 2023 ####
##########################################

library(tidyverse)
library(xtable)

rm(list = ls())

rho <- 0.95 # persistence
sigma <- 0.007 # sd of the error
N <- 9 # number of states
n <- 10000 # number of periodos
m <- 3 # number of sds
mu <- 0 # mean of the ar process

set.seed(7)

########### Question 1

## Tauchen

Tauchen <- function(rho, N, sigma, mu = 0.0, m = 3){

  theta_sup <- m*sqrt(sigma^2/(1-rho^2)) # sigma is the sd of the error
  theta_grid <- seq(from = -theta_sup, to = theta_sup, length.out = N)
  delta <- (theta_grid[2]-theta_grid[1])/2
  
  P <- matrix(0, nrow = N, ncol = N)
  
  for(i in 1:N){
    for(j in 2:N-1){
      
      P[i,1] <- pnorm((-theta_sup + delta - rho*theta_grid[i]-(1-rho)*mu)/sigma)
      P[i,N] <- 1 - pnorm((theta_sup - delta - rho*theta_grid[i]-(1-rho)*mu)/sigma)
      P[i,j] <- pnorm((theta_grid[j] + delta - rho*theta_grid[i]-(1-rho)*mu)/sigma) -
                pnorm((theta_grid[j] - delta - rho*theta_grid[i]-(1-rho)*mu)/sigma)
    } 
  }            
  
  return(list(P, theta_grid))
}


########### Question 2

## Rouwenhorst

Rouwen <- function(rho, N, sigma, mu=0.0){

  theta_sup <- mu + sigma*sqrt((N-1)/(1-rho^2)) # sigma is the sd of the error
  theta_inf <- mu - sigma*sqrt((N-1)/(1-rho^2))
  theta_grid <- seq(from = theta_inf, to = theta_sup, length.out = N)
  
  p <- (1+rho)/2
  
  P2 <- matrix(c(p,1-p,1-p,p), nrow = 2)
  
  if(N == 2){
  
  P_new <- P2
  
  }else{
    
    P_old <- P2
  
    for(i in 3:N){
      P_new <- p*cbind(rbind(
                             P_old, 
                             matrix(0, nrow = 1, ncol = i-1)),
                             matrix(0, nrow = i, ncol = 1)) +
               (1-p)*cbind(rbind(
                             matrix(0, nrow = 1, ncol = i-1),
                             P_old),
                             matrix(0, nrow = i, ncol = 1)) +
               (1-p)*rbind(cbind(
                             matrix(0, nrow = i-1, ncol = 1),
                             P_old),
                             matrix(0, nrow = 1,ncol = i)) +
               p*rbind(matrix(0, nrow = 1, ncol = i),
                             cbind(matrix(0, nrow = i-1,ncol = 1),
                             P_old))
      P_old <- P_new
    }
    
  P_new <- P_new/rowSums(P_new) # normalizing the rows to sum 1
  }
  
  return(list(P_new, theta_grid)) 
  
}


########### Question 3

## Simulating the discrete process

z_cont_95 <- rep(0,n)

e <- rnorm(n, mean = mu, sd = sigma)

for(i in 2:n){
  z_cont_95[i] = rho*z_cont_95[i-1] + e[i]
}

Disc_ar <- function(rho, N, sigma, n, e, z0 = "median", mu=0, m=3, tauchen = TRUE){

  if(tauchen){    
  PP <- Tauchen(rho, N, sigma, mu, m)
  }else{    
  PP <- Rouwen(rho, N, sigma, mu)
  }

  P_acum <- t(apply(PP[[1]], 1, cumsum))

  z_disc <- rep(0,n)

  cdf_e <- rep(0,n)

  cdf_e[1] <- pnorm(e[1], mean = mu, sd = sigma)

  if(z0 == "median"){ # z0 is the initial state
    P_e <- median(1:N)
  }else{
    P_e <- which(PP[[2]] >= z0)
  }
  
  z_disc[1] <- PP[[2]][P_e]

  for(i in 2:n){
  cdf_e[i] <- pnorm(e[i], mean = mu, sd = sigma)

  P_e_n <- which(P_acum[P_e[1],] >= cdf_e[i])

  z_disc[i] <- PP[[2]][P_e_n[1]]

  P_e <- P_e_n[1]
  }

  return(z_disc)
}

Disc_ar_t_95 <- Disc_ar(rho, N, sigma, n, e, "median", mu, m, TRUE)

Disc_ar_r_95 <- Disc_ar(rho, N, sigma, n, e, "median", mu, m, FALSE)

plot(z_cont_95, main = "rho = 0.95", xlab = '', ylab = '', col = "navyblue", type = 'l', lwd = 1)
lines(Disc_ar_t_95, col="grey", type = 'l', lwd = 1)
legend("bottomright",legend = c("Continuous process", "Tauchen discrete process"),
       col=c("navyblue", "grey"), lty = c(1, 1), cex = 0.7)


plot(z_cont_95, main = "rho = 0.95", xlab = '', ylab = '', col = "navyblue", type = 'l', lwd = 1)
lines(Disc_ar_r_95, col="yellow", type = 'l', lwd = 1)
legend("bottomright",legend = c("Continuous process", "Rouwenhorst discrete process"),
       col=c("navyblue", "yellow"), lty = c(1, 1), cex = 0.7)


########### question 4

est_t_95 <- rep(0,n)

for(i in 2:n){
  est_t_95[i] = Disc_ar_t_95[i-1]
}

est_r_95 <- rep(0,n)

for(i in 2:n){
  est_r_95[i] = Disc_ar_r_95[i-1]
}

reg_t_95 <- lm(Disc_ar_t_95 ~ est_t_95 -1)
summary(reg_t_95)

reg_r_95 <- lm(Disc_ar_r_95 ~ est_r_95 -1)
summary(reg_r_95)


########### question 5

z_cont_99 <- rep(0,n)

for(i in 2:n){
  z_cont_99[i] = 0.99*z_cont_99[i-1] + e[i]
}

Disc_ar_t_99 <- Disc_ar(0.99, N, sigma, n, e, "median", mu, m, TRUE)

Disc_ar_r_99 <- Disc_ar(0.99, N, sigma, n, e, "median", mu, m, FALSE)

plot(z_cont_99, main = "rho = 0.99", xlab = '', ylab = '', col = "navyblue", type = 'l', lwd = 1)
lines(Disc_ar_t_99, col="grey", type = 'l', lwd = 1)
legend("bottomright",legend = c("Continuous process", "Tauchen discrete process"),
       col=c("navyblue", "grey"), lty = c(1, 1), cex = 0.7)


plot(z_cont_99, main = "rho = 0.99", xlab = '', ylab = '', col = "navyblue", type = 'l', lwd = 1)
lines(Disc_ar_r_99, col="yellow", type = 'l', lwd = 1)
legend("bottomright",legend = c("Continuous process", "Rouwenhorst discrete process"),
       col=c("navyblue", "yellow"), lty = c(1, 1), cex = 0.7)


est_t_99 <- rep(0,n)

for(i in 2:n){
  est_t_99[i] = Disc_ar_t_99[i-1]
}

est_r_99 <- rep(0,n)

for(i in 2:n){
  est_r_99[i] = Disc_ar_r_99[i-1]
}

reg_t_99 <- lm(Disc_ar_t_99 ~ est_t_99 -1)
summary(reg_t_99)

reg_r_99 <- lm(Disc_ar_r_99 ~ est_r_99 -1)
summary(reg_r_99)
