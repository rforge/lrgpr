\name{plot.lrgpr}
\alias{plot.lrgpr}
\title{Plot Diagnostics for an \code{\link{lrgpr}} Object}
\usage{
  plot.lrgpr(x, which = c(1L:3L, 5L),
    caption = list("Residuals vs Fitted", "Normal Q-Q", "Scale-Location", "Cook's distance", "Residuals vs Leverage", expression("Cook's dist vs Leverage  " * h[ii]/(1 - h[ii]))),
    panel = if (add.smooth) panel.smooth else points,
    sub.caption = NULL, main = "",
    ask = prod(par("mfcol")) < length(which) && dev.interactive(),
    ..., id.n = 3, labels.id = names(residuals(x)),
    cex.id = 0.75, qqline = TRUE, cook.levels = c(0.5, 1),
    add.smooth = getOption("add.smooth"),
    label.pos = c(4, 2), cex.caption = 1)
}
\arguments{
  \item{x}{\code{\link{lrgpr}} object.}

  \item{which}{if a subset of the plots is required,
  specify a subset of the numbers \"1:6\".}

  \item{caption}{captions to appear above the plots;
  \"character\" vector or \"list\" of valid graphics
  annotations, see \"as.graphicsAnnot\". Can be set to
  \"""\" or \"NA\" to suppress all captions.}

  \item{panel}{panel function.  The useful alternative to
  \"points\", \"panel.smooth\" can be chosen by
  \"add.smooth = TRUE\".}

  \item{sub.caption}{common title-above the figures if
  there are more than one; used as \"sub\" (s.\"title\")
  otherwise.  If \"NULL\", as by default, a possible
  abbreviated version of \"deparse(x$call)\" is used.}

  \item{main}{title to each plot-in addition to
  \"caption\".}

  \item{ask}{logical; if \"TRUE\", the user is _ask_ed
  before each plot, see \"par(ask=.)\".}

  \item{...}{other parameters to be passed through to
  plotting functions.}

  \item{id.n}{number of points to be labelled in each plot,
  starting with the most extreme.}

  \item{labels.id}{vector of labels, from which the labels
  for extreme points will be chosen.  \"NULL\" uses
  observation numbers.}

  \item{cex.id}{magnification of point labels.}

  \item{qqline}{logical indicating if a \"qqline()\" should
  be added to the normal Q-Q plot.}

  \item{cook.levels}{levels of Cook's distance at which to
  draw contours.}

  \item{add.smooth}{logical indicating if a smoother should
  be added to most plots; see also \"panel\" above.}

  \item{label.pos}{positioning of labels, for the left half
  and right half of the graph respectively, for plots 1-3.}

  \item{cex.caption}{controls the size of \"caption\".}
}
\description{
  Six plots (selectable by \"which\") are currently
  available: a plot of residuals against fitted values, a
  Scale-Location plot of sqrt(| residuals |) against fitted
  values, a Normal Q-Q plot, a plot of Cook's distances
  versus row labels, a plot of residuals against leverages,
  and a plot of Cook's distances against
  leverage/(1-leverage).  By default, the first three and
  \"5\" are provided.
}
\seealso{
  \code{plot.lm}
}

