\name{QQ_plot}
\alias{QQ_plot}
\title{QQ plot}
\usage{
  QQ_plot(p_values,
    col = rainbow(min(length(p_values), ncol(p_values))),
    main = "", pch = 20, errors = TRUE, lambda = TRUE,
    p_thresh = 1e-06, showNames = FALSE, ylim = NULL,
    xlim = NULL, plot = TRUE, new = TRUE,
    box.lty = par("lty"), collapse = FALSE, ...)
}
\arguments{
  \item{p_values}{vector, matrix or list of p-values}

  \item{col}{colors corresponding to the number of columns
  in matrix, or entries in the list}

  \item{main}{title}

  \item{pch}{pch}

  \item{errors}{show 95\% confidence interval}

  \item{lambda}{calculate and show genomic control lambda.
  Lambda_GC is calcualted using the 'median' method on
  p-values > p_thresh.}

  \item{p_thresh}{Lambda_GC is calcualted using the
  'median' method on p-values > p_thresh.}

  \item{showNames}{show column names or list keys in the
  legend}

  \item{ylim}{ylim}

  \item{xlim}{xlim}

  \item{method}{method}

  \item{plot}{make a plot.  If FALSE, returns lamda_GC
  values without making plot}

  \item{new}{make a new plot.  If FALSE, overlays QQ over
  current plot}

  \item{box.lty}{box line type}

  \item{collapse}{combine entries in matrix or list into a
  single vector}

  \item{...}{other arguments}
}
\description{
  QQ plot and lambda_GC optimizd for large datasets.
}
\examples{
p = runif(1e6)
QQ_plot(p)

# get lambda_GC values without making plot
lambda = QQ_plot(p, plot=FALSE)
}

