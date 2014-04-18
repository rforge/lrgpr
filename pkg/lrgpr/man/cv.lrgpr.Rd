\name{cv.lrgpr}
\alias{cv.lrgpr}
\title{Cross-validation for \code{\link{lrgpr}}}
\usage{
cv.lrgpr(formula, features, order, nfolds = 10, rank = c(seq(0, 10), seq(20,
  100, by = 10), seq(200, 1000, by = 100)), nthreads = 1)
}
\arguments{
  \item{formula}{standard linear modeling syntax as used in
  `lm'}

  \item{features}{matrix from which the SVD is performed}

  \item{order}{sorted indices of features.  When rank is
  10, decomp = svd(X[,order[1:10]])}

  \item{nfolds}{number of training sets}

  \item{rank}{array with elements indicating the number of
  confounding covariates to be used in the random effect.}

  \item{nthreads}{number of threads to be used}
}
\description{
Fit cross-validation for multiple ranks of
\code{\link{lrgpr}}
}
\examples{
n = 300
p = 5000
X = matrix(sample(0:2, n*p, replace=TRUE), nrow=n)

dcmp = svd(X)

# simulate response
h_sq = .8
eta = dcmp$u[,1:2] \%*\% rgamma(2, 2, 1)
error_var = (1-h_sq) / h_sq  * var(eta)
y = eta + rnorm(n, sd=sqrt(error_var))

# Get ordering based on marginal correlation
i = order(cor(y, X)^2, decreasing=TRUE)

# Fit cross-validation
fit = cv.lrgpr( y ~ 1, features=X, order=i)

plot(fit)
}

