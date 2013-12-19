\name{plot.cv.lrgpr}
\alias{plot.cv.lrgpr}
\title{Plot Results of Cross-validation}
\usage{
  plot.cv.lrgpr(x,
    ylim = c(min(x$cve - x$cvse), max(x$cve + x$cvse)),
    xlim = range(x$rank), pch = 20, col = "red",
    main = "Cross validation",
    xlab = "# of confounding markers used",
    ylab = "Cross validation error", ...)
}
\arguments{
  \item{x}{result of cv.lrgpr}

  \item{ylim}{limits of y-axis}

  \item{xlim}{limits of x-axis}

  \item{pch}{}

  \item{col}{}

  \item{main}{}

  \item{xlab}{}

  \item{ylab}{}

  \item{...}{other parameters fed to plot()}
}
\description{
  Plot results of 'cv.lrgpr', which fits cross-validation
  for multiple ranks of the LRGPR
}

