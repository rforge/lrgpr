\name{wald}
\alias{wald}
\title{Composite hypothesis test of multiple coefficients}
\usage{
  wald(fit, terms)
}
\arguments{
  \item{fit}{result of fitting with \code{\link{lrgpr}}}

  \item{terms}{indices of the coefficients to be tested}
}
\description{
  Performs a multi-dimensional Wald test against H0:
  beta_i...beta_j = 0 using the estimated coefficients and
  their variance-covariance matrix
}
\section{Details}{
  The Wald statistic is \deqn{\beta_h^T \Sigma_h^{-1}
  \beta_h \sim \chi^2_{|h|}} where \deqn{h} specifies the
  coefficients being tested and \deqn{|h|} is the number of
  entries
}
\seealso{
  \code{\link{lrgpr}}
}

