library(copula)
library(lme4)

## copula
# Dimension
p = 5
nsim = 10000

# Clayton copula with theta=2
set.seed(123)
cop = claytonCopula(param = 2, dim = p)
u_cop = rCopula(nsim, cop)

# Gamma marginal distribution
micdf = function(u) qgamma(u, shape = 2, scale = 1) - 2
u = micdf(u_cop)

# Simulate GLMM data
set.seed(123)
x = 0.2 * matrix(rnorm(nsim * p), nsim, p)
w = 0.2 * matrix(rnorm(nsim * p), nsim, p)
beta = runif(p, -2, 2)
linear = c(x %*% beta) + rowSums(w * u)
y = matrix(rpois(nsim, exp(linear)) + 0.0, ncol = 1)

data = data.frame(x = x, w = w, y = y, u = u)
write.csv(data, 'copula.csv')

## VerbAgg
data(VerbAgg)
write.csv(VerbAgg, 'VerbAgg.csv')