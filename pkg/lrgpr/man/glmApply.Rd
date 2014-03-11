\name{glmApply}
\alias{glmApply}
\title{Fit standard linear or logistic model for many markers}
\usage{
  glmApply(formula, features, terms = NULL,
    family = gaussian(), useMean = TRUE,
    nthreads = detectCores(logical = TRUE),
    univariateTest = TRUE, multivariateTest = FALSE,
    verbose = FALSE, progress = TRUE, cincl = c(),
    cexcl = c())
}
\arguments{
  \item{formula}{standard linear modeling syntax as used in
  `lm'.  SNP is a place holder for the each successive
  column of features}

  \item{features}{a matrix where the statistical model is
  evaluated with SNP if formula replace by each column
  successively}

  \item{terms}{indices of the coefficients to be tested.
  The indices corresponding to SNP are used if terms is not
  specified}

  \item{family}{gaussian() for a continuous response, and
  binomial() to fit a logit model for a binary response}

  \item{useMean}{if TRUE, replace missing entries with
  column mean.  Otherwise, do not evaluate the model for
  that column}

  \item{nthreads}{number of to use for parallel execution}

  \item{univariateTest}{perform univariate hypothesis test
  for each response for each feature in the loop variable}

  \item{multivariateTest}{perform multivariate hypothesis
  test for each response (if more than one) for each
  feature.  Note that the runtime is cubic in the number of
  response variables}

  \item{verbose}{print additional information}

  \item{progress}{show progress bar}

  \item{cincl}{column indeces of features to include for
  analysis}

  \item{cexcl}{column indeces of features to exclude for
  analysis}
}
\description{
  Analogous to \code{\link{lrgprApply}}, but fits standard
  linear or logistic models for many markers
}
\examples{
# Generate data
n = 100
p = 500
X = matrix(sample(0:2, n*p, replace=TRUE), nrow=n)
y = rnorm(n)
sex = as.factor(sample(1:2, n, replace=TRUE))

# Fit model for all markers
pValues = glmApply( y ~ sex + sex:SNP, features=X, terms=c(3,4))


# Multivariate model
n = 100
p = 1000
m = 10

Y = matrix(rnorm(n*m), nrow=n, ncol=m)
X = matrix(rnorm(n*p), nrow=n, ncol=p)

res = glmApply( Y ~ SNP, features = X, terms=2, multivariateTest=TRUE)

# p-values for univariate hypothesis test of each feature against
# 	each response
res$pValues

# p-values for multivariate hypothesis test of each feature against
# 	all responses are the same time
# returns the results of the Hotelling and Pillai tests
res$pValues_mv

# The multivariate test for X[,1]
res$pValues_mv[1,]

# The result is the same as the standard tests in R
fit = manova( Y ~ X[,1])

summary(fit, test="Hotelling-Lawley")
summary(fit, test="Pillai")
}

