\name{read.gen}
\alias{read.gen}
\title{Read plink GEN file in OXFORD format}
\usage{
  read.gen(file, missing = NA, start_line = 1,
    nlines = Inf, quiet = FALSE, markerNames = NULL)
}
\arguments{
  \item{file}{location of GEN file}

  \item{missing}{instruction for how to deal with missing
  data: missing='NA' sets all missing values to NA,
  missing='mean' sets values to the mean value for that
  marker}

  \item{start_line}{index of the first line to read, where
  the indexing starts at 1}

  \item{nlines}{number of markers to read after and
  including start_line.  If nlines=Inf, read entire file
  starting at start_line.}

  \item{quiet}{print information about data processing}

  \item{markerNames}{array of marker names to be read from
  the file}
}
\description{
  Read GEN file in OXFORD format into R as a matrix.  A
  subset of consecutive markers can be read with start_line
  and nlines
}

