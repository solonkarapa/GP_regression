
library(plgp)

# Consider a toy 1d example in where the response (y) is a simple sinusoid 
# measured at n=5 equally spaced x locations. 
n <- 5
X <- matrix(seq(0, 3 * pi, length = n), ncol = 1)
y <- sin(X)
D <- distance(X) # pairwise squared distances between x locations
eps <- sqrt(.Machine$double.eps) 
Sigma <- exp(-D) + diag(eps, ncol(D)) # covariance matrix

XX <- matrix(seq(-0.5, 3 * pi + 0.8, length = 50), ncol = 1) # testing design matrix
DXX <- distance(XX) # distances between testing locations
SXX <- exp(-DXX) + diag(eps, ncol(DXX))

DX <- distance(XX, X) # distances between testing and training locations
SX <- exp(-DX)

Si <- solve(Sigma)
mup <- SX %*% Si %*% y # mean of the predictive distribution
Sigmap <- SXX - SX %*% Si %*% t(SX) # variance of the predictive distribution

YY <- rmvnorm(100, mup, Sigmap) # posterior/predictive distribution  

q1 <- mup + qnorm(0.025, 0, sqrt(diag(Sigmap))) # pointwise quantile-based error-bars
q2 <- mup + qnorm(0.975, 0, sqrt(diag(Sigmap))) # pointwise quantile-based error-bars

### plot
par(mfrow = c(2, 2))
matplot(XX, t(YY), type = "l", col = "white", lty = 1, xlab = "x", ylab = "y", 
        main = "Gaussian Process Regression",
        sub = "Observed data (black points)")
points(X, y, pch = 20, cex = 1.5)

matplot(XX, t(YY), type = "l", col = "white", lty = 1, xlab = "x", ylab = "y", 
        main = "True mean outcome (blue)",
        sub = "Unobserved mean; to be estimated given the observed data")
points(X, y, pch = 20, cex = 1.5)
lines(XX, sin(XX), col = "blue")

matplot(XX, t(YY), type = "l", col = "white", lty = 1, xlab = "x", ylab = "y", 
        main = "Estimated mean (solid black)", sub = "95% probability intervals (dashed black)")
points(X, y, pch = 20, cex = 1.5)
lines(XX, sin(XX), col = "blue")
lines(XX, mup, lwd = 2)
lines(XX, q1, lwd = 2, lty = 2)
lines(XX, q2, lwd = 2, lty = 2)

matplot(XX, t(YY), type = "l", col = "gray", lty = 1, xlab = "x", ylab = "y", 
        main = "Samples from the predictive distribution (grey)",
        sub = "95% of the grey lines fall within the dashed lines - as expected")
points(X, y, pch = 20, cex = 1.5)
lines(XX, sin(XX), col = "blue")
lines(XX, mup, lwd = 2)
lines(XX, q1, lwd = 2, lty = 2)
lines(XX, q2, lwd = 2, lty = 2)

