

#' Print Values
#'
#' Print details for fit from \code{\link{lrgpr}}
#' 
#' @param x model fit from \code{\link{lrgpr}}
#' @param ... other arguments
#' @S3method print lrgpr
#' @export
print.lrgpr <- function( x,...){
	
	cat( "\nCall:\n" )
	print( x$call )

	cat( "\nCoefficients:\n")
	print( x$coefficients )

	cat( "\nVariance components:\n")
	print( c(x$sigSq_a, x$sigSq_e, x$delta) )
}

#' Object Summaries
#'
#' Print summary for fit from \code{\link{lrgpr}}
#' 
#' @param x model fit from \code{\link{lrgpr}}
#' @param ... other arguments
#' @export
print.summary.lrgpr <- function( x,... ){

	cat( "\nCall:\n" )
	print( x$call )

	cat( "\nResiduals:\n" )
	R <- format(quantile( x$residuals, probs=c( 0, .25, .5, .75, 1) ), digits=4)
	names(R) <- c('Min', '1Q', 'Median', '3Q', 'Max') 
	print(R, quote=F)

	cat( "\nCoefficients:\n")	

	pvalues = x$coefficients[,4]

	signif_dots <- rep( '', length(pvalues) )
	for( i in 1:length(pvalues) ){
		if( pvalues[i] < .1 )   signif_dots[i] = '.          '
		if( pvalues[i] < .05 )  signif_dots[i] = '*          '
		if( pvalues[i] < .01 )  signif_dots[i] = '**         '
		if( pvalues[i] < .001 ) signif_dots[i] = '***        '
	}

    res = cbind(x$coefficients, signif_dots)
    names(res)[5] = ''
    
	print( res, digits=4)
	
	cat("---\nSignif. codes:  0 \"***\" 0.001 \"**\" 0.01 \"*\" 0.05 \".\" 0.1 \" \" 1\n\n")	

	n <- length(x$y)
	resid_sq_error <- sqrt( var(x$residuals) * ((n-1) / (n-x$df) ) )

	cat("Residual standard error:", format(x$sigma, digits=4), "on", format(x$df[2], digits=4), "degrees of freedom\n")

	cat("Multiple R-squared:", format(x$r.squared, digits=4), ",\tAdjusted R-squared:", format(x$adj.r.squared, digits=4), "\n")
	
	# Don't show F-statistic since it is not valid with a variance component
	if( attr(x$terms,"intercept") ){
		p1 <- 1
	}else{ p1 = 0 }
	p2 <- x$df[1]

	pValue = pf( x$fstatistic, p2-p1, x$df[2], lower.tail=FALSE)

	#cat("F-statistic:", format(x$fstatistic, digits=4), "on", format(p2-p1, digits=6), "and", format(x$df[2],digits=4), "DF,  p-value:", format(pValue, digits=4))

	cat( "\n\nVariance components:\n")
	print( x$variance_components )

}

#' Summarizing LRGPR / Linear Mixed Model Fits
#'
#' Print summary for fit from \code{\link{lrgpr}}
#' 
#' @param object model fit from \code{\link{lrgpr}}
#' @param ... other arguments
#' @export
summary.lrgpr <- function( object,... ){

	res = list()
	res$call = object$call
	res$terms = object$terms
	res$residuals = object$residuals

    std_dev = sqrt(diag(object$Sigma))

	M <- cbind( object$coefficients, std_dev, object$coefficients/std_dev, object$p.values)	
	rownames(M) <- names(object$coefficients)
	colnames(M) <- c('Estimate', 'Std. Error', 't value', 'Pr(>|t|)')
	res$coefficients <- as.data.frame(M)

	n <- length(object$y)
	res$sigma <- sqrt( var(object$residuals) * ((n-1) / (n-object$df) ) )

	res$r.squared <- cor(object$y, object$fitted.values)^2

	# Compared empirical adjusted Rsq with lm()
	res$adj.r.squared <- res$r.squared - (1-res$r.squared)  * object$df / (n-object$df)

	if( attr(object$terms,"intercept") ){
		p1 <- 1
		RSS1 <- sum( (object$fitted.values - mean(object$fitted.values))^2)
	}else{ 
		p1 <- 0 
		RSS1 <- sum( object$fitted.values^2)
	}

	p2 <- object$df

	res$df <- c(object$df, n - p2, object$df)

	resvar <- sum(object$residuals^2)/(n-p1)
	
	res$fstatistic <- (RSS1/ (p2-p1) ) /resvar	

	res$variance_components = c(object$sigSq_a, object$sigSq_e, object$delta)

	class(res) = "summary.lrgpr"

	return( res )
}




# summary(obj)

# a <- predict(fit, matrix(1, nrow(K_test)), K_test)
# sqrt(mean((f(X_test) - a)^2))

# cor(predict(fitLow), predict(fitLow, mu, K))

# Predicton for full rank is TERRIBLE when spectrum is long

#' Predict response
#'
#' Predict response values after training with \code{\link{lrgpr}}.  Leaving X_test amd K_test as NULL returns the fitted values on the training set
#' 
#' @param object model fit from \code{\link{lrgpr}} on training samples
#' @param X_test design matrix of covariates for test samples
#' @param K_test covariance matrix between samples in the test set and training set
#' @param ... other arguments
#' @export
predict.lrgpr <- function( object, X_test=NULL, K_test=NULL,... ){

	if( is.null(X_test) && is.null(K_test) ){
		y_hat <- object$fitted.values
	}else{

		K_test_U <- K_test %*% object$eigenVectors
		resid <- object$y - object$x %*% object$coefficients
		ru <- crossprod(object$eigenVectors, resid)

        y_hat <- K_test_U %*% diag(1/(object$eigenValues+object$delta), length(object$eigenValues)) %*% ru + X_test %*% object$coefficients #+ (K_test %*% resid - K_test_U %*% ru)/ obj$delta


      # ( K_test %*% ( diag(1, nrow(X_test)) - tcrossprod(obj$eigenVectors) ) %*% resid )  / obj$delta 

      # I = diag(1, nrow(X_test)) 
      # G = I - tcrossprod(obj$eigenVectors)  

      #  crossprod(G %*% K_test, G %*% resid) / obj$delta 

	
	}
	
	return( y_hat)
}

#' Extract Model Residuals
#'
#' Residuals fitted with \code{\link{lrgpr}}
#' 
#' @param object model fit with \code{\link{lrgpr}}
#' @param type the type of residual, but there is only one option here
#' @param ... other arguments
#' @export
residuals.lrgpr <- function( object, type="working",...){
	object$residuals
}

#' Extract Model Coefficients
#'
#' Coefficients estimated with \code{\link{lrgpr}}
#' 
#' @param object model fit with \code{\link{lrgpr}}
#' @param ... other arguments
#' @export
coefficients.lrgpr <- function( object ){
	object$coefficients
}

#' Residual Degrees-of-Freedom
#'
#' Residual df from fit of \code{\link{lrgpr}}
#' 
#' @param object model fit with \code{\link{lrgpr}}
#' @param ... other arguments
#' @export
df.residual.lrgpr <- function( object,... ){
	# OLS
	#length( obj$y) - length(obj$coefficients)
	
	# including effective defgrees of freedom
	length( object$y) - object$df
}

#' Regression Diagnostics
#'
#' Basic quantities for regression diagnostics from fit of \code{\link{lrgpr}}
#' 
#' @param object model fit with \code{\link{lrgpr}}
#' @param ... other arguments
#' @export
lm.influence.lrgpr <- function( object,...){
	list(hat=object$hii)
}

#' Regression Diagnostics
#'
#' Basic quantities for regression diagnostics from fit of \code{\link{lrgpr}}
#' 
#' @param model model fit with \code{\link{lrgpr}}
#' @param ... other arguments
#' @export
influence.lrgpr <- function( model,... ){
	list(hat=model$hii)
}

# Regression diagnostics based on 
# http://people.stern.nyu.edu/jsimonof/classes/2301/pdf/diagnost.pdf

#' Regression Deletion Diagnostics
#'
#' Basic quantities for regression deletion diagnostics from fit of \code{\link{lrgpr}}
#' 
#' @param model model fit with \code{\link{lrgpr}}
#' @param ... other arguments
#' @export
rstandard.lrgpr <- function( model,... ){
	
	#sig <- sqrt(object$sigSq_e / df.residual(object) )
	sig <- sqrt(deviance(model)/df.residual(model))

	(model$y - model$fitted.values) / ( sig*sqrt(1 - model$hii) )
}

# Leverage values are diag(H)

#' Regression Diagnostics
#'
#' Basic quantities for regression diagnostics from fit of \code{\link{lrgpr}}
#' 
#' @param object model fit with \code{\link{lrgpr}}
#' @export
leverage.lrgpr <- function( object ){
	object$hii
}

# Cook's distance

#' Regression Deletion Diagnostics
#'
#' Basic quantities for regression deletion diagnostics from fit of \code{\link{lrgpr}}
#' 
#' @param model model fit with \code{\link{lrgpr}}
#' @param infl influence structure as returned by \"lm.influence\"
#' @param res residuals
#' @param sd standard deviation to use
#' @param hat hat values
#' @param ... other arguments
#' @S3method cooks.distance lrgpr
#' @export
cooks.distance.lrgpr <- function( model, infl = lm.influence(model, do.coef = FALSE), res=weighted.residuals(model), sd=sqrt(deviance(model)/df.residual(model)), hat = infl$hat,... ){	
		
	res^2 / (model$df*sd^2) * model$hii/(1-model$hii)^2 

	#rstandard( obj )^2 * obj$hii / ( obj$df * (1-obj$hii ) )
}

#' Calculate Variance-Covariance Matrix for a \code{\link{lrgpr}} Object
#'
#' Returns the variance-covariance matrix of the main parameters of a fitted model object 
#' 
#' @param object model fit with \code{\link{lrgpr}}
#' @param ... other arguments
#' @export
vcov.lrgpr <- function( object,...){
	object$Sigma
}

# This code is not exported

plot.lrgpr_Test <- function( obj,...){
	
	par(mfrow=c(1,4))
	plot(predict(obj), residuals(obj), main="Residuals vs Fitted", xlab="", ylab="Residuals")
	
	cex=.7
	mtext( "Fitted values", 1, line=2.5, cex=cex)
	mtext( obj$call, 1, line=4, cex=cex)
	abline(h=0, col="grey", lty=2)	
	
	residuals_standardized <- rstandard(obj)

	qqnorm( residuals_standardized )
	abline( 0, 1 )
	mtext( obj$call, 1, line=4.3, cex=cex)
	qqline( residuals_standardized, lty=3, col="gray50")
		
	ylab <- expression(paste(sqrt(abs(Standardized~~residuals))))
	plot( predict(obj), sqrt(abs(residuals_standardized)), xlab='', main="Scale-Location", ylab=ylab)
	mtext( "Fitted values", 1, line=2.5, cex=cex)
	mtext( obj$call, 1, line=4, cex=.7)
	
	ylab <- "Standardized residuals"
	plot( leverage.lrgpr(obj), residuals_standardized, xlab='', main="Residuals vs Leverage", ylab=ylab, xlim=c(0, max(leverage.lrgpr(obj))) )
	mtext( "Leverage", 1, line=2.5, cex=cex)
	mtext( obj$call, 1, line=4, cex=cex)	
	abline( h=0, lty=3, col="gray50")
	abline( v=0, lty=3, col="gray50")


}


#' Plot Diagnostics for an \code{\link{lrgpr}} Object
#'
#' Six plots (selectable by \"which\") are currently available: a plot of residuals against fitted values, a Scale-Location plot of sqrt(| residuals |) against fitted values, a Normal Q-Q plot, a plot of Cook's distances versus row labels, a plot of residuals against leverages, and a plot of Cook's distances against leverage/(1-leverage).  By default, the first three and \"5\" are provided.
#'  @param x \code{\link{lrgpr}} object.
#' @param which if a subset of the plots is required, specify a subset of the numbers \"1:6\".
#' @param caption captions to appear above the plots; \"character\" vector or \"list\" of valid graphics annotations, see \"as.graphicsAnnot\". Can be set to \"""\" or \"NA\" to suppress all captions.
#' @param panel panel function.  The useful alternative to \"points\", \"panel.smooth\" can be chosen by \"add.smooth = TRUE\".
#' @param sub.caption common title-above the figures if there are more than one; used as \"sub\" (s.\"title\") otherwise.  If \"NULL\", as by default, a possible abbreviated version of \"deparse(x$call)\" is used.
#' @param main title to each plot-in addition to \"caption\".
#' @param ask logical; if \"TRUE\", the user is _ask_ed before each plot, see \"par(ask=.)\".
#' @param ... other parameters to be passed through to plotting functions.
#' @param id.n number of points to be labelled in each plot, starting with the most extreme.
#' @param labels.id vector of labels, from which the labels for extreme points will be chosen.  \"NULL\" uses observation numbers.
#' @param cex.id magnification of point labels.
#' @param qqline logical indicating if a \"qqline()\" should be added to the normal Q-Q plot.
#' @param cook.levels levels of Cook's distance at which to draw contours.
#' @param add.smooth logical indicating if a smoother should be added to most plots; see also \"panel\" above.
#' @param label.pos positioning of labels, for the left half and right half of the graph respectively, for plots 1-3.
#' @param cex.caption controls the size of \"caption\".
#' @seealso \code{plot.lm}
#' @export
plot.lrgpr <- function (x, which=c(1L:3L, 5L), caption=list("Residuals vs Fitted", 
    "Normal Q-Q", "Scale-Location", "Cook's distance", "Residuals vs Leverage", 
    expression("Cook's dist vs Leverage  " * h[ii]/(1 - h[ii]))), 
    panel=if (add.smooth) panel.smooth else points, sub.caption=NULL, 
    main ="", ask=prod(par("mfcol")) < length(which) && dev.interactive(), 
    ..., id.n=3, labels.id=names(residuals(x)), cex.id=0.75, 
    qqline=TRUE, cook.levels=c(0.5, 1), add.smooth=getOption("add.smooth"), 
    label.pos=c(4, 2), cex.caption=1) 
{
    dropInf <- function(x, h) {
        if (any(isInf <- h >= 1)) {
            warning("Not plotting observations with leverage one:\n  ", 
                paste(which(isInf), collapse= ", "), call.= FALSE)
            x[isInf] <- NaN
        }
        x
    }
    if (!inherits(x, "lm")) 
        stop("use only with \"lm\" objects")
    if (!is.numeric(which) || any(which < 1) || any(which > 6)) 
        stop("'which' must be in 1:6")
    isGlm <- inherits(x, "glm")
    show <- rep(FALSE, 6)
    show[which] <- TRUE
    r <- residuals(x)
    yh <- predict(x)
    w <- weights(x)
    if (!is.null(w)) {
        wind <- w != 0
        r <- r[wind]
        yh <- yh[wind]
        w <- w[wind]
        labels.id <- labels.id[wind]
    }
    n <- length(r)
    if (any(show[2L:6L])) {
        s <- if (inherits(x, "rlm")) 
            x$s
        else if (isGlm) 
            sqrt(summary(x)$dispersion)
        else sqrt(deviance(x)/df.residual(x))
        hii <- lm.influence.lrgpr(x, do.coef=FALSE)$hat
        if (any(show[4L:6L])) {
            cook <- if (isGlm) 
                cooks.distance(x)
            else cooks.distance(x, sd=s, res=r)
        }
    }
    if (any(show[2L:3L])) {
        ylab23 <- if (isGlm) 
            "Std. deviance resid."
        else "Standardized residuals"
        r.w <- if (is.null(w)) 
            r
        else sqrt(w) * r
        rs <- dropInf(r.w/(s * sqrt(1 - hii)), hii)
    }
    if (any(show[5L:6L])) {
        r.hat <- range(hii, na.rm=TRUE)
        isConst.hat <- all(r.hat == 0) || diff(r.hat) < 1e-10 * 
            mean(hii, na.rm=TRUE)
    }
    if (any(show[c(1L, 3L)])) 
        l.fit <- if (isGlm) 
            "Predicted values"
        else "Fitted values"
    if (is.null(id.n)) {
        id.n <- 0
    }else {
        id.n <- as.integer(id.n)
        if (id.n < 0L || id.n > n) 
            stop(gettextf("'id.n' must be in {1,..,%d}", n), domain=NA)
    }
    if (id.n > 0L) {
        if (is.null(labels.id)) 
            labels.id <- paste(1L:n)
        iid <- 1L:id.n
        show.r <- sort.list(abs(r), decreasing=TRUE)[iid]
        if (any(show[2L:3L])) 
            show.rs <- sort.list(abs(rs), decreasing=TRUE)[iid]
        text.id <- function(x, y, ind, adj.x=TRUE) {
            labpos <- if (adj.x) 
                label.pos[1 + as.numeric(x > mean(range(x)))]
            else 3
            text(x, y, labels.id[ind], cex= cex.id, xpd=TRUE, 
                pos=labpos, offset=0.25)
        }
    }
    getCaption <- function(k) if (length(caption) < k) 
        NA_character_
    else as.graphicsAnnot(caption[[k]])
    if (is.null(sub.caption)) {
        cal <- x$call
        if (!is.na(m.f <- match("formula", names(cal)))) {
            cal <- cal[c(1, m.f)]
            names(cal)[2L] <- ""
        }
        cc <- deparse(cal, 80)
        nc <- nchar(cc[1L], "c")
        abbr <- length(cc) > 1 || nc > 75
        sub.caption <- if (abbr) 
            paste(substr(cc[1L], 1L, min(75L, nc)), "...")
        else cc[1L]
    }
    one.fig <- prod(par("mfcol")) == 1
    if (ask) {
        oask <- devAskNewPage(TRUE)
        on.exit(devAskNewPage(oask))
    }
    if (show[1L]) {
        ylim <- range(r, na.rm <- TRUE)
        if (id.n > 0) 
            ylim <- extendrange(r=ylim, f=0.08)
        dev.hold()
        plot(yh, r, xlab=l.fit, ylab="Residuals", main=main, 
            ylim=ylim, type="n", ...)
        panel(yh, r, ...)
        if (one.fig) 
            title(sub=sub.caption, ...)
        mtext(getCaption(1), 3, 0.25, cex=cex.caption)
        if (id.n > 0) {
            y.id <- r[show.r]
            y.id[y.id < 0] <- y.id[y.id < 0] - strheight(" ")/3
            text.id(yh[show.r], y.id, show.r)
        }
        abline(h=0, lty=3, col="gray")
        dev.flush()
    }
    if (show[2L]) {
        ylim <- range(rs, na.rm=TRUE)
        ylim[2L] <- ylim[2L] + diff(ylim) * 0.075
        dev.hold()
        qq <- qqnorm(rs, main=main, ylab=ylab23, ylim=ylim,...)
        if (qqline) 
            qqline(rs, lty=3, col="gray50")
        if (one.fig) 
            title(sub=sub.caption, ...)
        mtext(getCaption(2), 3, 0.25, cex=cex.caption)
        if (id.n > 0) 
            text.id(qq$x[show.rs], qq$y[show.rs], show.rs)
        dev.flush()
    }
    if (show[3L]) {
        sqrtabsr <- sqrt(abs(rs))
        ylim <- c(0, max(sqrtabsr, na.rm=TRUE))
        yl <- as.expression(substitute(sqrt(abs(YL)), list(YL=as.name(ylab23))))
        yhn0 <- if (is.null(w)) 
            yh
        else yh[w != 0]
        dev.hold()
        plot(yhn0, sqrtabsr, xlab=l.fit, ylab=yl, main=main, 
            ylim=ylim, type="n", ...)
        panel(yhn0, sqrtabsr, ...)
        if (one.fig) 
            title(sub=sub.caption, ...)
        mtext(getCaption(3), 3, 0.25, cex=cex.caption)
        if (id.n > 0) 
            text.id(yhn0[show.rs], sqrtabsr[show.rs], show.rs)
        dev.flush()
    }
    if (show[4L]) {
        if (id.n > 0) {
            show.r <- order(-cook)[iid]
            ymx <- cook[show.r[1L]] * 1.075
        }
        else ymx <- max(cook, na.rm=TRUE)
        dev.hold()
        plot(cook, type="h", ylim=c(0, ymx), main=main, 
            xlab="Obs. number", ylab="Cook's distance", ...)
        if (one.fig) 
            title(sub=sub.caption, ...)
        mtext(getCaption(4), 3, 0.25, cex=cex.caption)
        if (id.n > 0) 
            text.id(show.r, cook[show.r], show.r, adj.x=FALSE)
        dev.flush()
    }
    if (show[5L]) {
        ylab5 <- if (isGlm) 
            "Std. Pearson resid."
        else "Standardized residuals"
        r.w <- residuals(x, "pearson")
        if (!is.null(w)) 
            r.w <- r.w[wind]
        rsp <- dropInf(r.w/(s * sqrt(1 - hii)), hii)
        ylim <- range(rsp, na.rm=TRUE)
        if (id.n > 0) {
            ylim <- extendrange(r=ylim, f=0.08)
            show.rsp <- order(-cook)[iid]
        }
        do.plot <- TRUE
        if (isConst.hat) {
            if (missing(caption)) 
                caption[[5L]] <- "Constant Leverage:\n Residuals vs Factor Levels"
            aterms <- attributes(terms(x))
            dcl <- aterms$dataClasses[-aterms$response]
            facvars <- names(dcl)[dcl %in% c("factor", "ordered")]
            mf <- model.frame(x)[facvars]
            if (ncol(mf) > 0) {
                effM <- mf
                for (j in seq_len(ncol(mf))) effM[, j] <- sapply(split(yh, 
                  mf[, j]), mean)[mf[, j]]
                ord <- do.call(order, effM)
                dm <- data.matrix(mf)[ord, , drop=FALSE]
                nf <- length(nlev <- unlist(unname(lapply(x$xlevels,length))))
                ff <- if (nf == 1) 
                  1
                else rev(cumprod(c(1, nlev[nf:2])))
                facval <- (dm - 1) %*% ff
                xx <- facval
                dev.hold()
                plot(facval, rsp, xlim=c(-1/2, sum((nlev - 
                  1) * ff) + 1/2), ylim=ylim, xaxt="n", main=main, 
                  xlab="Factor Level Combinations", ylab=ylab5, 
                  type="n", ...)
                grp_means <- sapply(split(yh, mf[, 1L]), mean)
                axis(1, at=ff[1L] * (1L:nlev[1L] - 1/2) - 1/2, 
                  labels=x$xlevels[[1L]][order(grp_means)])
                mtext(paste(facvars[1L], ":"), side=1, line=0.25, 
                  adj=-0.05)
                abline(v=ff[1L] * (0:nlev[1L]) - 1/2, col="gray", 
                  lty="F4")
                panel(facval, rsp, ...)
                abline(h=0, lty=3, col="gray")
                dev.flush()
            }
            else {
                message("hat values (leverages) are all=", 
                  format(mean(r.hat)), "\n and there are no factor predictors; no plot no. 5")
                frame()
                do.plot <- FALSE
            }
        }
        else {
            xx <- hii
            xx[xx >= 1] <- NA
            dev.hold()
            plot(xx, rsp, xlim=c(0, max(xx, na.rm=TRUE)), 
                ylim=ylim, main=main, xlab="Leverage", 
                ylab=ylab5, type="n", ...)
            panel(xx, rsp, ...)
            abline(h=0, v=0, lty=3, col="gray")
            if (one.fig) 
                title(sub=sub.caption, ...)
            if (length(cook.levels)) {
                #p <- length(coef(x))
		p <- x$df
                usr <- par("usr")
                hh <- seq.int(min(r.hat[1L], r.hat[2L]/100), 
                  usr[2L], length.out=101)
                for (crit in cook.levels) {
                  #suppressWarnings( cl.h=sqrt(crit * p * (1 - hh)/hh) ) # this causes warnings for lrgpr, so suppress
		cl.h=sqrt(crit * p * (1 - hh)/hh)
                  lines(hh, cl.h, lty=2, col=2)
                  lines(hh, -cl.h, lty=2, col=2)
                }
                legend("bottomleft", legend="Cook's distance", 
                  lty=2, col=2, bty="n")
                xmax <- min(0.99, usr[2L])
                ymult <- sqrt(p * (1 - xmax)/xmax)
                aty <- c(-sqrt(rev(cook.levels)) * ymult, sqrt(cook.levels) * 
                  ymult)
                axis(4, at=aty, labels=paste(c(rev(cook.levels), 
                  cook.levels)), mgp=c(0.25, 0.25, 0), las=2, 
                  tck=0, cex.axis=cex.id, col.axis=2)
            }
            dev.flush()
        }
        if (do.plot) {
            mtext(getCaption(5), 3, 0.25, cex=cex.caption)
            if (id.n > 0) {
                y.id <- rsp[show.rsp]
                y.id[y.id < 0] <- y.id[y.id < 0] - strheight(" ")/3
                text.id(xx[show.rsp], y.id, show.rsp)
            }
        }
    }
    if (show[6L]) {
        g <- dropInf(hii/(1 - hii), hii)
        ymx <- max(cook, na.rm=TRUE) * 1.025
        dev.hold()
        plot(g, cook, xlim=c(0, max(g, na.rm=TRUE)), ylim=c(0, 
            ymx), main=main, ylab="Cook's distance", xlab=expression("Leverage  " * 
            h[ii]), xaxt="n", type="n", ...)
        panel(g, cook, ...)
        athat <- pretty(hii)
        axis(1, at=athat/(1 - athat), labels=paste(athat))
        if (one.fig) 
            title(sub=sub.caption, ...)
       	#p <- length(coef(x))
	p <- x$df
        bval <- pretty(sqrt(p * cook/g), 5)
        usr <- par("usr")
        xmax <- usr[2L]
        ymax <- usr[4L]
        for (i in seq_along(bval)) {
            bi2 <- bval[i]^2
            if (ymax > bi2 * xmax) {
                xi <- xmax + strwidth(" ")/3
                yi <- bi2 * xi
                abline(0, bi2, lty=2)
                text(xi, yi, paste(bval[i]), adj=0, xpd=TRUE)
            }
            else {
                yi <- ymax - 1.5 * strheight(" ")
                xi <- yi/bi2
                lines(c(0, xi), c(0, yi), lty=2)
                text(xi, ymax - 0.8 * strheight(" "), paste(bval[i]), 
                  adj=0.5, xpd=TRUE)
            }
        }
        mtext(getCaption(6), 3, 0.25, cex=cex.caption)
        if (id.n > 0) {
            show.r <- order(-cook)[iid]
            text.id(g[show.r], cook[show.r], show.r)
        }
        dev.flush()
    }
    if (!one.fig && par("oma")[3L] >= 1) 
        mtext(sub.caption, outer=TRUE, cex=1.25)
    invisible()
}

plot.lrgpr_rankSearch <- function(obj){

	par(mfrow=c(2,3), mar=c(4,4,2,1))
	
	plot( obj$evalues / sum(obj$evalues), xlab="index", ylab="fraction of variance", main="Eigen Spectrum")

	plot( obj$logLik, xlab="index", ylab="log-likelihood", main="Log-likelihood")

	plot( obj$rank, obj$df, xlab="rank", ylab="degrees of freedom", main="Degrees of Freedom")
	abline(obj$df[1], 1, lty=2, col="grey")

	plot( obj$rank, obj$AIC, xlab="rank",  ylab="AIC", main="Akaike information criterion")
	i <- which.min(obj$AIC)
	points( obj$rank[i], obj$AIC[i], col="red", pch=20, cex=1.3)

	plot( obj$rank, obj$BIC, xlab="rank",  ylab="BIC", main="Bayes information criterion")
	i <- which.min(obj$BIC)
	points( obj$rank[i], obj$BIC[i], col="red", pch=20, cex=1.3)

	plot( obj$rank, obj$GCV, xlab="rank",  ylab="GCV", main="Generalized cross-validation")
	i <- which.min(obj$GCV)
	points( obj$rank[i], obj$GCV[i], col="red", pch=20, cex=1.3)

}


#' Plot Error Bars
#'
#' Plot error bars for a confidence interval
#' @param x x-axis position
#' @param y y-axis position
#' @param upper height of bar above y
#' @param lower height of bar below y
#' @param length horizontal length of the error bar
#' @param ... arguments for arrows(...)
#' @export
error.bar <- function(x, y, upper, lower=upper, length=0.1,...){
    if(length(x) != length(y) | length(y) !=length(lower) | length(lower) != length(upper))
    stop("vectors must be same length")
    arrows(x,y+upper, x, y-lower, angle=90, code=3, length=length, ...)
}


