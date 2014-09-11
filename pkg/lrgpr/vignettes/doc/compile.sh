#! /usr/bin/env Rscript
library(knitr)
Sweave2knitr('lrgpr.Rnw')
knit('lrgpr-knitr.Rnw')
system("sed -i 's/\\hlopt{~}/\\mytilde/g' lrgpr-knitr.tex")
#system("sed -i 's/~/\\$\\\\mytilde\\$/g' lrgpr-knitr.tex")
system("sed -i 's/\\hlopt{^}/\\mycaret/g' lrgpr-knitr.tex")
tools::texi2dvi('lrgpr-knitr.tex', pdf=TRUE, clean=TRUE)
q()