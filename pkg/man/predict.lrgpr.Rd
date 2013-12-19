\name{predict.lrgpr}
\alias{predict.lrgpr}
\title{Predict response}
\usage{
  predict.lrgpr(object, X_test = NULL, K_test = NULL, ...)
}
\arguments{
  \item{object}{model fit from \code{\link{lrgpr}} on
  training samples}

  \item{X_test}{design matrix of covariates for test
  samples}

  \item{K_test}{covariance matrix between samples in the
  test set and training set}

  \item{...}{other arguments}
}
\description{
  Predict response values after training with
  \code{\link{lrgpr}}.  Leaving X_test amd K_test as NULL
  returns the fitted values on the training set
}

