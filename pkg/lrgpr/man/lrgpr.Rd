\name{lrgpr}
\alias{lrgpr}
\title{Fit a Low Rank Gaussian Process Regression (LRGPR) / Linear Mixed Model (LMM)}
\usage{
  lrgpr(formula, decomp,
    rank = max(length(decomp$d), length(decomp$values)),
    delta = NULL, nthreads = detectCores(logical = TRUE),
    W_til = NULL)
}
\arguments{
  \item{formula}{standard linear modeling syntax as used in
  'lm'}

  \item{decomp}{eigen-decomposition produced from eigen(K),
  where K is the covariance matrix.  Or singular value
  decomposition svd(X[,1:100]) based on a subset of
  markers}

  \item{rank}{decomposition is truncated to the first rank
  eigen-vectors}

  \item{delta}{ratio of variance components governing the
  fit of the model.  This should be estimated from a
  previous evaluation of 'lm' on the same response and
  eigen-decomposition}

  \item{nthreads}{number of threads to use for parallel
  execution}

  \item{W_til}{markers used to construct decomp that should
  now be remvoe from costruction of decomp.  This is the
  proximal contamination term of Listgarten, et al. (2012)}
}
\value{
  \item{coefficients}{regression coefficients for each
  covariate} \item{p.values}{p-values from Wald test of
  each coefficient} \item{sd}{standard deviation of each
  coefficient estimate} \item{sigSq_e}{variance component
  \deqn{\sigma^2_e} corresponding to the residual error}
  \item{sigSq_a}{variance component \deqn{\sigma^2_a}
  corresponding the scale of the covariance, K}
  \item{delta}{ratio of variance components:
  \deqn{\sigma^2_e / \sigma^2_a}} \item{rank}{the rank of
  the random effect} \item{logLik}{log-likelihood of the
  model fit} \item{fitted.values}{estimated response
  values: y_hat} \item{alpha}{BLUP of the random effect}
  \item{Sigma}{variance-covariate matrix of estimate of
  beta} \item{hii}{diagonals of the matrix H such that
  y_hat = Hy} \item{y}{responses} \item{x}{design matrix}
  \item{df}{effective degrees of freedom: trace(H) based on
  Hoffman (2013)} \item{residuals}{residuals of model fit:
  y - y_hat} \item{AIC}{Akaike information criterion}
  \item{BIC}{Bayesian information criterion}
  \item{GCV}{generalized cross-validation}
  \item{eigenVectors}{eigen-vectors in decomp}
  \item{eigenValues}{eigen-values in decomp}
  \item{df.residual}{n - ncol(X)} \item{rank}{rank of
  decomposition used, where only non-negative
  eigen/singular values are considered}
}
\description{
  `lrgpr' is used to fit LRGPR/LMM models that account for
  covariance in response values, but where the scale of the
  covariance is unknown.  Standard linear modeling syntax
  is used for the model specification in addition to a
  covariance matrix or its eigen-decomposition.
}
\section{Details}{
  `lrgpr' fits the model: \deqn{ y = X \beta + \alpha +
  \epsilon}

  \deqn{\alpha \sim N(0, K \sigma^2_a)}

  \deqn{\epsilon \sim N(0, \sigma^2_e)}

  where \deqn{\delta = \sigma^2_e / \sigma^2_a}

  In practice the eigen-decomposition of K, and not K
  itself is required.  The rank can be set to use only
  eigen-vectors 1:rank in the model.

  This package allows hypothesis tests of single
  coefficients using fit$p.values which fits a Wald test.
  Composite hypothesis tests of multiple coefficients are
  performed with wald(fit, terms=1:3).

  Note that likelihood ratio tests with linear mixed models
  do not perform well and the resulting p-values often do
  not follow a uniform distribution under the null
  (Pinheiro and Bates, 2000).  We strongly advise against
  using it with this model.

  `lrgpr' uses the algorithm of Lippert, et al. (2011).

  See Hoffman (2013) for an interpretation of the linear
  mixed model.
}
\examples{
# Generate random data
set.seed(1)
n <- 200
y <- rnorm(n)
K <- crossprod( matrix(rnorm(n*1000), ncol=n) )
age <- rpois(n, 50)
sex <- as.factor(sample(1:2, n, replace=TRUE))
decomp <- eigen(K)

# Fit the model
fit <- lrgpr( y ~ sex + age, decomp)

# Print results
fit

# Print more detailed results
summary(fit)

# P-values for each covariate
fit$p.values

# Visualize fit of the model like for 'lm'
par(mfrow=c(2,2))
plot(fit)

# Composite hypothesis test using Wald's test
# Joint test of coefficients 2:3
wald( fit, terms=2:3)
}
\references{
  Kang, H. M., et al. (2010) Variance component model to
  account for sample structure in genome-wide association
  studies. _Nature Genetics_ 42, 348-54

  Lippert, C., et al. (2011) FaST linear mixed models for
  genome-wide association studies. _Nature Methods_ 9,
  525-26

  Listgarten, J., et al. (2012) Improved linear mixed
  models for genome-wide association studies. _Nature
  Methods_ 8, 833-5

  Rasmussen, C. E. and Williams, C. K. I. (2006) Gaussian
  processes for machine learning. MIT Press

  Pinheiro, J. C. and Bates, D. M. (2000) Mixed-Effects
  Models in S and S-PLUS. Springer, New York

  Hoffman, G. E. (2013) Correcting for Population Structure
  and Kinship Using the Linear Mixed Model: Theory and
  Extensions. _PLoS ONE_ 8(10):e75707
}
\seealso{
  'wald', 'lrgprApply'
}

