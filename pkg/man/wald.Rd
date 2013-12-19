\name{wald}
\alias{wald}
\title{Composite hypothesis test of multiple coefficients}
\usage{
  wald(fit, terms)
}
\arguments{
  \item{fit}{result of fitting with 'lrgpr'}

  \item{terms}{indeces of the coefficients to be tested}
}
\description{
  'wald' performs a multi-dimensional Wald test against H0:
  beta_i...beta_j = 0 using the estimated coefficients and
  their variance-covariance matrix
}
\section{Details}{
  The Wald statistic is \deqn{\beta^T \Sigma_\beta \beta
  \sim \chi^2_k} where k is the dimension of beta
}
\seealso{
  'lrgpr'
}

