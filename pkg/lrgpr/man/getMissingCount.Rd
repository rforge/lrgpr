\name{getMissingCount}
\alias{getMissingCount}
\title{Count missing values}
\usage{
  getMissingCount(X,
    nthreads = detectCores(logical = TRUE),
    progress = TRUE)
}
\arguments{
  \item{X}{matrix where each column is a marker}

  \item{nthreads}{number of threads to use}

  \item{progress}{show progress bar}
}
\description{
  Count missing values
}

