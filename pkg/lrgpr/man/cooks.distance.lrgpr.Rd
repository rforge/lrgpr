\name{cooks.distance.lrgpr}
\alias{cooks.distance.lrgpr}
\title{Regression Deletion Diagnostics}
\usage{
  cooks.distance.lrgpr(model,
    infl = lm.influence(model, do.coef = FALSE),
    res = weighted.residuals(model),
    sd = sqrt(deviance(model)/df.residual(model)),
    hat = infl$hat, ...)
}
\arguments{
  \item{model}{model fit with \code{\link{lrgpr}}}

  \item{infl}{influence structure as returned by
  \code{\link{lm.influence}}}

  \item{res}{residuals}

  \item{sd}{standard deviation to use}

  \item{hat}{hat values}

  \item{...}{other arguments}
}
\description{
  Basic quantities for regression deletion diagnostics from
  fit of \code{\link{lrgpr}}
}

