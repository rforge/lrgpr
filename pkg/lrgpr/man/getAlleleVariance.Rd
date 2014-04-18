\name{getAlleleVariance}
\alias{getAlleleVariance}
\title{Evaluate variance for each column}
\usage{
getAlleleVariance(X, nthreads = detectCores(logical = TRUE),
  progress = TRUE)
}
\arguments{
  \item{X}{matrix where each column is a marker}

  \item{nthreads}{number of threads to use}

  \item{progress}{show progress bar}
}
\description{
Evaluate variance for each column
}

