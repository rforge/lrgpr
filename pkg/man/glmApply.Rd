\name{glmApply}
\alias{glmApply}
\title{Fit standard linear or logitic model for many markers}
\usage{
  glmApply(formula, features, terms, family = gaussian(),
    useMean = FALSE,
    nthreads = detectCores(logical = TRUE),
    univariate = TRUE, multivariate = TRUE)
}
\arguments{
  \item{formula}{standard linear modeling syntax as used in
  'lm'.  SNP is a place holder for the eash successive
  column of features}

  \item{features}{a matrix were the statistical model is
  evaluated with SNP if formula replace by each column
  successively}

  \item{terms}{indeces of the coefficients to be tested}

  \item{family}{gaussian() for a continuous response, and
  binomial() to fit a logit model for a binay response}

  \item{useMean}{if TRUE, replace missing entries with
  column mean.  Otherwise, do not evaluate the model for
  that column}

  \item{nthreads}{number of to use for parallel execution}

  \item{univariate}{perform univariate hypothesis test for
  each response for each feature in the loop variable}

  \item{multivariate}{perform multivariate hypothesis test
  for each response (if more than one) for each feature}
}
\description{
  'glmApply' is analogous to 'lrgprApply', but fits
  standard linear or logitic models for many markers
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
}

