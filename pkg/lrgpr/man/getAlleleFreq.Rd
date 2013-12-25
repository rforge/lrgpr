\name{getAlleleFreq}
\alias{getAlleleFreq}
\title{Calculate allele frequency}
\usage{
  getAlleleFreq(X, nthreads = detectCores(logical = TRUE))
}
\arguments{
  \item{X}{matrix where each column is a marker coded 0,1,2
  or with dosage values in this range}

  \item{nthreads}{number of threads to use}
}
\description{
  Calculate allele frequency
}

