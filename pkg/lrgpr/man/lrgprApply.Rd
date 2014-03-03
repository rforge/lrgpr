\name{lrgprApply}
\alias{lrgprApply}
\title{Fit a Low Rank Gaussian Process Regression (LRGPR) / Linear Mixed Model (LMM) for many markers}
\usage{
  lrgprApply(formula, features, decomp, terms = NULL,
    rank = max(ncol(decomp$u), ncol(decomp$vectors)),
    map = NULL, distance = NULL, dcmp_features = NULL,
    W_til = NULL, scale = TRUE, delta = NULL,
    reEstimateDelta = FALSE,
    nthreads = detectCores(logical = TRUE),
    verbose = FALSE, progress = TRUE, cincl = c(),
    cexcl = c())
}
\arguments{
  \item{formula}{standard linear modeling syntax as used in
  'lm'.  SNP is a place holder for the each successive
  column of features}

  \item{features}{a matrix where the statistical model is
  evaluated with SNP if formula replace by each column
  successively}

  \item{decomp}{eigen-decomposition produced from eigen(K),
  where K is the covariance matrix.  Or singular value
  decomposition svd(features[,1:100]) based on a subset of
  markers}

  \item{terms}{indices of the coefficients to be tested.
  The indices corresponding to SNP are used if terms is not
  specified}

  \item{rank}{decomposition is truncated to the first rank
  eigen-vectors}

  \item{map}{p x 2 matrix where each entry corresponds to a
  marker in features.  First column is the marker names,
  second columns is the genetic or physical location}

  \item{distance}{size of the proximal contamination window
  in units specifed by map.}

  \item{dcmp_features}{the indices in features of the
  markers used to construct dcmp}

  \item{W_til}{markers used to construct decomp that should
  now be removed from costruction of decomp.  This is the
  proximal contamination term of Listgarten, et al. (2012)}

  \item{scale}{should W_til be scaled and centered}

  \item{delta}{ratio of variance components governing the
  fit of the model.  This should be estimated from a
  previous evaluation of 'lm' on the same response and
  eigen-decomposition}

  \item{reEstimateDelta}{should delta be re-estimated for
  every marker. Note: reEstimateDelta=TRUE is much slower}

  \item{nthreads}{number of to use for parallel execution}

  \item{verbose}{print extra information}

  \item{progress}{show progress bar}

  \item{cincl}{column indeces of features to include for
  analysis}

  \item{cexcl}{column indeces of features to exclude for
  analysis}
}
\description{
  `lrgprApply' is used to fit LRGPR/LMM models that account
  for covariance in response values, but where the scale of
  the covariance is unknown.  It returns p-values
  equivalent to the results of lrgpr() and wald(), but is
  designed to analyze thousands of markers in a single
  function call.
}
\examples{
# Generate data
n = 100
p = 500
X = matrix(sample(0:2, n*p, replace=TRUE), nrow=n)
y = rnorm(n)
sex = as.factor(sample(1:2, n, replace=TRUE))

K = tcrossprod(matrix(rnorm(n*n*3), nrow=n))
decomp = eigen(K, symmetric=TRUE)

# Fit null model
fit = lrgpr( y ~ sex, decomp)

# Fit model for all markers
pValues = lrgprApply( y ~ sex + sex:SNP, features=X, decomp, terms=c(3,4), delta=fit$delta)
}

