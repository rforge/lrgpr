## lrgpr.R: 'lrgpr' is used to fit LRGPR/LMM models that account for covariance in response values
##
## Copyright (C) 2013 Gabriel E. Hoffman
##
##
## RcppGSL is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 2 of the License, or
## (at your option) any later version.
##
## RcppGSL is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with RcppGSL.  If not, see <http://www.gnu.org/licenses/>.

# trace(crossprod(A,B))
#
# Efficiently evaluate the trace of the matrix product by computing only the diagonals of the product
# 
# @param A matrix
# @param B matrix
# @export
crossprodTrace <- function(A, B){
	obj <- .Call("R_crossprod_trace", A, B, package="lrgpr")
	return( obj$trace )
}

#' Replace Missing Values with Mean
#'
#' For each column, replace NA values with the column mean
#'
#' @param A matrix
#' 
#' @export
set_missing_to_mean = function(A){

	if( ! is.matrix(A) || ncol(A) == 1){
		cm = mean( A, na.rm=TRUE )

		idx = which(is.na(A))

		if( length(idx) > 0){
			A[idx] = cm
		}

		return(A) 
	}

	cm = colMeans( A, na.rm=TRUE )

	idx = sapply( 1:ncol(A), function(i){which(is.na(A[,i]))})

	for(i in 1:ncol(A) ){
		if( length(idx[[i]]) > 0 ){
			A[idx[[i]],i] = cm[i]
		}
	}

	return( A)
}

.is_supported_lrgpr = function(X){

	supported = FALSE

	if( is.matrix(X) ){
		supported = TRUE
	}else if( is.big.matrix(X) && ! is.sub.big.matrix(X) ){
		supported = TRUE	
	}
	
	return( supported )
}

#' Fit a Low Rank Gaussian Process Regression (LRGPR) / Linear Mixed Model (LMM)
#'
#' `lrgpr' is used to fit LRGPR/LMM models that account for covariance in response values, but where the scale of the covariance is unknown.  Standard linear modeling syntax is used for the model specification in addition to a covariance matrix or its eigen-decomposition.
#'
#' @param formula standard linear modeling syntax as used in 'lm'
#' @param decomp eigen-decomposition produced from eigen(K), where K is the covariance matrix.  Or singular value decomposition svd(X[,1:100]) based on a subset of markers
#' @param rank decomposition is truncated to the first rank eigen-vectors
#' @param delta ratio of variance components governing the fit of the model.  This should be estimated from a previous evaluation of 'lm' on the same response and eigen-decomposition  
#' @param nthreads number of threads to use for parallel execution
#' @param W_til markers used to construct decomp that should now be removed from costruction of decomp.  This is the proximal contamination term of Listgarten, et al. (2012)
#' @param scale should W_til be scaled and centered
#' @return
#' \item{coefficients}{regression coefficients for each covariate}
#' \item{p.values}{p-values from Wald test of each coefficient}
#' \item{sd}{standard deviation of each coefficient estimate}
#' \item{sigSq_e}{variance component \deqn{\sigma^2_e} corresponding to the residual error}
#' \item{sigSq_a}{variance component \deqn{\sigma^2_a} corresponding the scale of the covariance, K}
#' \item{delta}{ratio of variance components: \deqn{\sigma^2_e / \sigma^2_a}}
#' \item{rank}{the rank of the random effect}
#' \item{logLik}{log-likelihood of the model fit}
#' \item{fitted.values}{estimated response values: y_hat}
#' \item{alpha}{BLUP of the random effect}
#' \item{Sigma}{variance-covariate matrix of estimate of beta}
#' \item{hii}{diagonals of the matrix H such that y_hat = Hy}
#' \item{y}{responses}
#' \item{x}{design matrix}
#' \item{df}{effective degrees of freedom: trace(H) based on Hoffman (2013)}
#' \item{residuals}{residuals of model fit: y - y_hat}
#' \item{AIC}{Akaike information criterion}
#' \item{BIC}{Bayesian information criterion}
#' \item{GCV}{generalized cross-validation}
#' \item{eigenVectors}{eigen-vectors in decomp}
#' \item{eigenValues}{eigen-values in decomp}
#' \item{df.residual}{n - ncol(X)}
#' \item{rank}{rank of decomposition used, where only non-negative eigen/singular values are considered}
#' @section Details:
#' `lrgpr' fits the model:
#
#' \deqn{ y =  X \beta + \alpha +  \epsilon}
#'
#' \deqn{\alpha \sim N(0,  K \sigma^2_a)}
#'
#' \deqn{\epsilon \sim N(0, \sigma^2_e)}
#'
#' where \deqn{\delta = \sigma^2_e / \sigma^2_a}
#'
#' In practice the eigen-decomposition of K, and not K itself is required.  The rank can be set to use only eigen-vectors 1:rank in the model.
#'
#' This package allows hypothesis tests of single coefficients using fit$p.values which fits a Wald test.  Composite hypothesis tests of multiple coefficients are performed with wald(fit, terms=1:3).
#' 
#' Note that likelihood ratio tests with linear mixed models do not perform well and the resulting p-values often do not follow a uniform distribution under the null (Pinheiro and Bates, 2000).  We strongly advise against using it with this model.
#'
#' `lrgpr' uses the algorithm of Lippert, et al. (2011). 
#'
#' See Hoffman (2013) for an interpretation of the linear mixed model.
#'
#' @seealso 'wald', 'lrgprApply'
#' @references
#'
#' Kang, H. M., et al. (2010) Variance component model to account for sample structure in genome-wide association studies. _Nature Genetics_ 42, 348-54
#' 
#' Lippert, C., et al. (2011) FaST linear mixed models for genome-wide association studies. _Nature Methods_ 9, 525-26
#'
#' Listgarten, J., et al. (2012) Improved linear mixed models for genome-wide association studies. _Nature Methods_ 8, 833-5
#'
#' Rasmussen, C. E. and Williams, C. K. I. (2006) Gaussian processes for machine learning. MIT Press
#'
#' Pinheiro, J. C. and Bates, D. M. (2000) Mixed-Effects Models in S and S-PLUS. Springer, New York
#'
#' Hoffman, G. E. (2013) Correcting for Population Structure and Kinship Using the Linear Mixed Model: Theory and Extensions. _PLoS ONE_ 8(10):e75707
#' 
#' Note that degrees freedom and some diagnostic statistics are not currently calculated with W_til is specified.
#'
#' @useDynLib lrgpr
#' @examples 
#' # Generate random data
#' set.seed(1)
#' n <- 200
#' y <- rnorm(n)
#' K <- crossprod( matrix(rnorm(n*1000), ncol=n) )
#' age <- rpois(n, 50)
#' sex <- as.factor(sample(1:2, n, replace=TRUE))
#' decomp <- eigen(K)
#'
#' # Fit the model
#' fit <- lrgpr( y ~ sex + age, decomp) 
#'
#' # Print results
#' fit
#'
#' # Print more detailed results 
#' summary(fit)
#'
#' # P-values for each covariate
#' fit$p.values
#'
#' # Visualize fit of the model like for 'lm'
#' par(mfrow=c(2,2))
#' plot(fit)
#'
#' # Composite hypothesis test using Wald's test
#' # Joint test of coefficients 2:3
#' wald( fit, terms=2:3)
#' @export
lrgpr <- function( formula, decomp, rank=max(ncol(decomp$u), ncol(decomp$vectors)), delta=NULL, nthreads=detectCores(logical=TRUE), W_til=NULL, scale=TRUE){

	#chisq=FALSE
	rdf=FALSE # Was false before

	if( ! is.na(nthreads) && nthreads < 1 ){
		stop("nthreads must be positive")
	}

	result = tryCatch({
		# save status from current state
		na.status = options("na.action")$na.action
		# Cause failure if y or X contain NA
		options(na.action="na.fail")

	    mf <- match.call(expand.dots=FALSE)
		m <- match(c("formula", "data"), names(mf), 0L)
		#covariateName <- mf[["X"]]
		mf <- mf[c(1L, m)]
		mf$drop.unused.levels <- TRUE
		mf[[1L]] <- as.name("model.frame")
		mf <- eval(mf, parent.frame())
		mt <- attr(mf, "terms")
		y <- model.response(mf, "numeric")
		X <- model.matrix(mt, mf, contrasts)
	}, warning 	= function(w) { warning(w)
	}, error 	= function(e) { stop(e)
	}, finally 	= {
	    # restore status to previous state
		options(na.action=na.status)
	})	

	functionCallString <- match.call()	

	##############
	# check data #
	##############

	if( !is.null(delta) && delta <= 0 ){
		stop("delta must be positive")
	} 	
	if( is.null(delta) ){
		delta <- NaN
	}
	if( length(which(is.na(X))) > 0 ){
		stop("Cannot accept missing data in covariates")
	}
	if( length(which(is.na(y))) > 0 ){
		stop("Cannot accept missing data in response")
	}

	# if decomp is result of eigen()
	if( is.eigen_decomp(decomp) ){
		
		# if the number of samples does not match
		if( length(y) != nrow(decomp$vectors) ){ 
			stop( paste("decomp$vectors must have the same number of samples as the response variable.\nnrow(decomp$vectors):", nrow(decomp$vectors), "\nlength(y):", length(y), "\nThis may be due to R automatically removing samples with missing values from the response or design matrix."))
		}

		# set rank to not exceed the number of positive eigen-values
		rank <- min( rank, length(which(decomp$values > 0)) )
		rank <- min (rank, ncol(decomp$vectors))

		# truncate eigen-spectrum
		decomp$values = decomp$values[1:rank]
		decomp$vectors = decomp$vectors[,1:rank,drop=FALSE]	
	}

	# if decomp is result of svd()
	if( is.svd_decomp(decomp) ){
		
		# if the number of samples does not match
		if( length(y) != nrow(decomp$u) ){ 
			stop( paste("decomp$u must have the same number of samples as the response variable.\nnrow(decomp$vectors):", nrow(decomp$u), "\nlength(y):", length(y), "\nThis may be due to R automatically removing samples with missing values from the response or design matrix."))
		}		

		# set rank to not exceed the number of positive eigen-values
		rank <- min( rank, length(which(decomp$d > 0)) )
		rank <- min (rank, ncol(decomp$u))

		# truncate eigen-spectrum
		decomp$d = decomp$d[1:rank]
		decomp$u = decomp$u[,1:rank,drop=FALSE]	
	}

	a = gc()

	if( rank < 1 ){
		stop("rank < 1 is not valid")
	}

	n <- length(y)
	p <- ncol(X)

	if( is.null(W_til) ) W_til = matrix(0)
	if( ! is.matrix(W_til) ) W_til = as.matrix(W_til)

	if( ! is.null(W_til) && scale ) W_til = scale(W_til)

	# if decomp is result of eigen()
	if( is.eigen_decomp(decomp) ){			
		obj <- .Call("R_lrgpr", y, X, decomp$vectors, decomp$values, delta, nthreads, W_til, package="lrgpr")
	}

	# if decomp is result of svd()
	if( is.svd_decomp(decomp) ){		
		# if decomp is an svd, then the eigen-values are decomp$d^2
		obj <- .Call("R_lrgpr", y, X, decomp$u, decomp$d^2, delta, nthreads, W_til, package="lrgpr")
	}

	gc()	

	# rename coefficients
	names(obj$coefficients) <- colnames(X)

	# rename variance components
	names(obj$sigSq_a) <- "sigSq_a"
	names(obj$sigSq_e) <- "sigSq_e"
	names(obj$delta)   <- "delta"

	class( obj ) <- c("lrgpr", "lm")
	#class( obj ) <- c("lrgpr")
	#inherits( obj, "lm", TRUE)
	obj$y <- y
	obj$x <- X
	obj$call <- functionCallString
	obj$df <- sum( obj$hii)

	obj$na.action <- attr(mf, "na.action")
	obj$offset <- offset
	obj$contrasts <- model.matrix(mt, mf, contrasts)
	obj$xlevels <- .getXlevels(mt, mf)
	obj$terms <- mt

	obj$residuals <- obj$y - obj$fitted.values

	# Criteria
	# Following Wood, 2006, p129
	# n*SSE / (tr(I-H))^2  except use df of fit, instead of rdf of residuals
	obj$AIC <- -2*obj$logLik + obj$df*2
	obj$BIC <- -2*obj$logLik + obj$df*log(n)	
	obj$GCV <- n*sum(obj$residuals^2) / (n-obj$df)^2  	

	names(obj$p.values) <- colnames(X)

	obj$df.residual <- n - ncol(X)

	if( is.eigen_decomp(decomp) ){	
		obj$eigenVectors = decomp$vectors
		obj$eigenValues = decomp$values
	}else{
		obj$eigenVectors = decomp$u
		obj$eigenValues = decomp$d^2
	}

	obj$rank = rank

	# if W_til is defined, then df and hii are not computed correctly
	if( length(W_til) != 1){
		obj$hii = NA
		obj$df = NA
	}

	return( obj )
}

setClass("lrgpr")

#' Composite hypothesis test of multiple coefficients 
#'
#' `wald' performs a multi-dimensional Wald test against H0: beta_i...beta_j = 0 using the estimated coefficients and their variance-covariance matrix
#'
#' @param fit result of fitting with 'lrgpr'
#' @param terms indices of the coefficients to be tested
#'
#' @section Details:
#' The Wald statistic is \deqn{\beta_h^T \Sigma_h^{-1} \beta_h \sim \chi^2_{|h|}}
#' where \deqn{h} specifies the coefficients being tested and \deqn{h} is the number of entries
#'
#' @seealso 'lrgpr'
#' @export
wald <- function(fit, terms){

	if( is.null(terms) || is.na(terms) || length(terms)==0  ){
		stop("Must specify terms")
	}	
	if( length(which( !( terms %in% 1:length(fit$coefficients) ) )) ){
		stop("Element in terms is not valid or is out of range")
	}	

	stat <- tcrossprod(fit$coefficients[terms], solve(fit$Sigma[terms,terms])) %*% fit$coefficients[terms]
	
	df <- length(terms)

	res <- c(stat, df, pchisq( stat, df, lower.tail=FALSE))

	names(res) <- c("chisq", "df", "p.value")
	
	res <- as.data.frame(t(res))
	rownames(res) <- ""

	return(res)
}

# define local function to extract response with a single function call
# this is called by C++ code in R_fastGPR_batch
#' @export
.mm_get_response <- function(a, env){
	#cat(as.character(a), "\n")
	as.matrix(model.response(model.frame(a, env)))
}

# Get column names of design matrix
# this is called by C++ code in R_fastGPR_batch
#' @export
.mm_get_colnames <- function(a){
	colnames(model.matrix.default(a))
}

#' @export
.mm_get_terms <- function(form, query, env){

	nameArray = colnames(model.matrix.default(form, env))

	# Find query matching nameArray
	# convert from R to C indexing

	#pat1 = "([:*]*)SNP$"
	#pat2 = "^SNP([:*]*)"
	pat1 = paste("([:*]*)", query, "$", sep='')
	pat2 = paste("^", query, "([:*]*)", sep='')

	idx = unique( c( grep(pat1, nameArray), grep(pat2, nameArray)))

	return( idx -1 )
}

# remove any terms involving SNP from formula
#' @export



.mm_replace_query <- function(form, query){

	form = as.character(form)

	# remove spaces
	form = gsub(" ", "", form)

	# insert spaces at ~, +, -
	form = gsub("~", " ~ ", form)
	form = gsub("\\+", " \\+ ", form)
	form = gsub("\\-", " \\- ", form)

	form = gsub(paste(query, "\\*", sep=''), "", form)
	form = gsub(paste(query, "\\:", sep=''), "", form)
	form = gsub(paste(query, " \\+", sep=''), "", form)
	form = gsub(paste(query, " \\-", sep=''), "", form)

	form = gsub(paste("\\*", query, sep=''), "", form)
	form = gsub(paste("\\:", query, sep=''), "", form)
	form = gsub(paste("\\+ ", query, sep=''), "", form)
	form = gsub(paste("\\- ", query, sep=''), "", form)

	form = gsub(paste(" ", query, " ", sep=''), "1", form, perl=TRUE)
	form = gsub(paste(" ", query, "$", sep=''), "1", form, perl=TRUE)

	return( form )
}

# check .mm_replace_query
if(1){

.mm_replace_query( Y ~ SNP213123, "SNP")

.mm_replace_query( Y ~ SNP:sex + 3, "SNP")

.mm_replace_query( Y ~ SNP*sex, "SNP")

.mm_replace_query( Y ~ SNP + sex, "SNP")

.mm_replace_query( Y ~ sex + SNP, "SNP")

.mm_replace_query( Y ~ SNP + SNP*sex, "SNP")

.mm_replace_query( Y ~ SNP, "SNP")
}


.mm_replace_query_with <- function(form, query, repl){

	form = as.character(form)

	# remove spaces
	form = gsub(" ", "", form)

	# insert spaces at ~, +, -
	form = gsub("~", " ~ ", form)
	form = gsub("\\+", " \\+ ", form)
	form = gsub("\\-", " \\- ", form)

	form = gsub(paste(query, "\\*", sep=''),  paste(repl, "\\*", sep=''), form)
	form = gsub(paste(query, "\\:", sep=''),  paste(repl, "\\:", sep=''), form)
	form = gsub(paste(query, " \\+", sep=''), paste(repl, "\\+", sep=''), form)
	form = gsub(paste(query, " \\-", sep=''), paste(repl, "\\-", sep=''), form)

	form = gsub(paste("\\*", query, sep=''),  paste("\\*", repl, sep=''), form)
	form = gsub(paste("\\:", query, sep=''),  paste("\\:", repl, sep=''), form)
	form = gsub(paste("\\+ ", query, sep=''), paste("\\+", repl, sep=''), form)
	form = gsub(paste("\\- ", query, sep=''), paste("\\-", repl, sep=''), form)

	form = gsub(paste(" ", query, " ", sep=''), repl, form, perl=TRUE)
	form = gsub(paste(" ", query, "$", sep=''), repl, form, perl=TRUE)

	return( form )
}



# check .mm_replace_query
if(1){
.mm_replace_query_with( Y ~ SNP213123, "SNP", "X")

.mm_replace_query_with( Y ~ SNP:sex + 3, "SNP", "X")

.mm_replace_query_with( Y ~ SNP*sex, "SNP", "X")

.mm_replace_query_with( Y ~ SNP + sex, "SNP", "X")

.mm_replace_query_with( Y ~ sex + SNP, "SNP", "X")

.mm_replace_query_with( Y ~ SNP + SNP*sex, "SNP", "X")

.mm_replace_query_with( Y ~ SNP, "SNP", "X")

}


# Check if decomp is the result of eigen()
is.eigen_decomp <- function( decomp ){
	
	check = (! is.null(decomp$vectors) ) && (! is.null(decomp$values))
	
	#if( check ){
	#	check = (length(decomp$values) == nrow(decomp$vectors))
	#}

	return( check )
}

# Check if svd is the result of svd()
is.svd_decomp <- function( decomp ){
	
	check = (! is.null(decomp$u) ) && (! is.null(decomp$d) ) && (! is.null(decomp$v))
	
	# This is not satisifed if dcmp is low rank
	#if( check ){
	#	check = (length(decomp$d) == ncol(decomp$u))
	#}

	return( check )
}

# Check if svd is the result of svd() of a symmetric matrix
is.svd_decomp_symmetric <- function( decomp ){
	
	check = is.svd_decomp( decomp )
	
	if( check ){
		# if U and V are the same size
		check = (ncol(decomp$u) == ncol(decomp$v)) && (nrow(decomp$u) == nrow(decomp$v))

		if( check ){
			# if U and V have the same entries, at least for the first 10 PC's
			check = ( cor( as.vector(abs(decomp$v[,1:10])), as.vector(abs(decomp$u[,1:10]))) > .999)
		}
	}

	return( check )
}

#' Fit a Low Rank Gaussian Process Regression (LRGPR) / Linear Mixed Model (LMM) for many markers
#'
#' `lrgprApply' is used to fit LRGPR/LMM models that account for covariance in response values, but where the scale of the covariance is unknown.  It returns p-values equivalent to the results of lrgpr() and wald(), but is designed to analyze thousands of markers in a single function call.
#'
#' @param formula standard linear modeling syntax as used in 'lm'.  SNP is a place holder for the each successive column of features
#' @param features a matrix where the statistical model is evaluated with SNP if formula replace by each column successively
#' @param decomp eigen-decomposition produced from eigen(K), where K is the covariance matrix.  Or singular value decomposition svd(features[,1:100]) based on a subset of markers
#' @param terms indices of the coefficients to be tested. The indices corresponding to SNP are used if terms is not specified
#' @param rank decomposition is truncated to the first rank eigen-vectors
#' @param map p x 2 matrix where each entry corresponds to a marker in features.  First column is the marker names, second columns is the genetic or physical location
#' @param distance size of the proximal contamination window in units specifed by map. 
#' @param dcmp_features the indices in features of the markers used to construct dcmp
#' @param W_til markers used to construct decomp that should now be removed from costruction of decomp.  This is the proximal contamination term of Listgarten, et al. (2012)
#' @param scale should W_til be scaled and centered 
#' @param delta ratio of variance components governing the fit of the model.  This should be estimated from a previous evaluation of 'lm' on the same response and eigen-decomposition  
#' @param reEstimateDelta should delta be re-estimated for every marker. Note: reEstimateDelta=TRUE is much slower
#' @param nthreads number of to use for parallel execution
#' @param verbose print extra information
#' @param progress show progress bar 
#'
#' @examples 
#' 
#' # Generate data
#' n = 100
#' p = 500
#' X = matrix(sample(0:2, n*p, replace=TRUE), nrow=n)
#' y = rnorm(n)
#' sex = as.factor(sample(1:2, n, replace=TRUE))
#'
#' K = tcrossprod(matrix(rnorm(n*n*3), nrow=n))
#' decomp = eigen(K, symmetric=TRUE)
#' 
#' # Fit null model
#' fit = lrgpr( y ~ sex, decomp)
#' 
#' # Fit model for all markers 
#' pValues = lrgprApply( y ~ sex + sex:SNP, features=X, decomp, terms=c(3,4), delta=fit$delta)
#'
#' @export
lrgprApply <- function( formula, features, decomp, terms=NULL, rank=max(ncol(decomp$u), ncol(decomp$vectors)), map=NULL, distance=NULL, dcmp_features=NULL, W_til=NULL, scale=TRUE, delta=NULL, reEstimateDelta=FALSE, nthreads=detectCores(logical = TRUE), verbose=FALSE, progress=TRUE ){

	env = parent.frame()

	# convert formula to character string
	formChar = as.character.formula( formula )

	# Save value to restore later
	# Note that if there is a global SNP variable, AND lrgprApply is called in a subthread, AND another threads accesses SNP, then there will be a RACE CONDITION
	# If varable exists
	#if( length(grep( "^SNP$", ls(env))) > 1) SNP_tmp = SNP

	# Need to use:
	# formChar = as.formula( .mm_replace_query_with( formChar, "SNP", SNPrandom) )
	# where SNPrandom is a random string that doesn't occur in ls()
	# Then pass this name into R_lrgprApply to be replace for each marker

	##################################
	# Argument checking from lrgpr() #
	##################################

	if( ! is.matrix(features) && ! is.big.matrix(features) ){
		features = as.matrix(features)
	}

	if( ! .is_supported_lrgpr(features) ){
		stop("Unsupported data type for features.\nSupported types are matrix and big.matrix.\nNote that sub.big.matrix is not currently supported")
	}	

	##############
	# check data #
	##############
	
	result = tryCatch({
		# save status from current state
		na.status = options("na.action")$na.action

		# Cause failure if y or X contain NA
		 options(na.action="na.fail")
		#options(na.action="na.omit")

	   	# Get response from formula to check identity or logistic link
		#++++++++++++++++++++++

		# do not fail if featres[,1] has missing entry
		.lrgpr_tmp = set_missing_to_mean(features[,1])
		idx = which(is.na(.lrgpr_tmp))

		if( length(idx) > 0 ){
			.lrgpr_tmp[idx] = rnorm(length(idx))
		}

		assign(".lrgpr_tmp", .lrgpr_tmp, envir = env)
		
		form_mod = as.formula( .mm_replace_query_with( formChar, "SNP", ".lrgpr_tmp") )
		.y = .mm_get_response( form_mod, env)

		# identify indeces that contain the SNP variable
		terms = grep(".lrgpr_tmp", colnames(model.matrix(form_mod, env)))

		n_indivs = nrow(as.matrix(.y))
		#.X = model.matrix.default( form_mod )

		#if( ! is.numeric(.y) && ! is.factor(.y)){
		#	stop("Response is not numeric")
		#}

	}, warning 	= function(w) { warning(w)
	}, error 	= function(e) { stop(e)
	}, finally 	= {
	    # restore status to previous state
		options(na.action=na.status)
	})	

	if( !is.null(delta) && delta <= 0 ){
		stop("delta must be positive")
	} 	
	if( is.null(delta) ){
		delta <- NA
	}
	if( ! is.na(nthreads) && nthreads < 1 ){
		stop("nthreads must be positive")
	}
	if( length(terms) == 0 ){
		stop("Must specify terms")
	}
	if( length(which( terms <= 0 )) > 0 ){
		stop(paste("Invalid values for terms:", paste(terms, collapse=', ')))
	}
	if( ! is.eigen_decomp(decomp) && ! is.svd_decomp(decomp)  ){
		stop("Must specify decomp as the result of eigen() or svd()")
	}
	if( ! is.null(map) && ncol(features) != nrow(map) ){
		stop("ncol(features) must be the same as nrow(map)")
	}
	if( ! is.null(map) && ncol(map) != 2 ){
		stop("map, if specified, must have 2 columns: chromesome, location")
	}
	if( ! is.null(map) && ! is.numeric(distance) ){
		stop("distance must be numeric")
	}
	if( ! is.null(map) && length(dcmp_features) == 0 ){
		stop("Must specify dcmp_features")
	}
	if( is.null(map) && length(dcmp_features) != 0 ){
		stop("Argument dcmp_features is not used when map argument is not set")
	}

	if( verbose ){
		cat("terms:", terms, "\n")
	}

	# if decomp is result of eigen()
	if( is.eigen_decomp(decomp) ){
		
		# if the number of samples does not match
		if( length(.y) != nrow(decomp$vectors) ){ 
			stop( paste("decomp$vectors must have the same number of samples as the response variable.\nnrow(decomp$vectors):", nrow(decomp$vectors), "\nn_indivs:", n_indivs, "\nThis may be due to R automatically removing samples with missing values from the response or design matrix."))
		}

		# set rank to not exceed the number of positive eigen-values
		rank <- min( rank, length(which(decomp$values > 0)) )
		rank <- min (rank, ncol(decomp$vectors))

		# truncate eigen-spectrum
		decomp$values = decomp$values[1:rank]
		decomp$vectors = decomp$vectors[,1:rank,drop=FALSE]		
	}

	# if decomp is result of eigen()
	if( is.svd_decomp(decomp) ){
		
		# if the number of samples does not match
		if( length(.y) != nrow(decomp$u) ){ 
			stop( paste("decomp$u must have the same number of samples as the response variable.\nnrow(decomp$vectors):", nrow(decomp$u), "\nn_indivs:", n_indivs, "\nThis may be due to R automatically removing samples with missing values from the response or design matrix."))
		}

		# set rank to not exceed the number of positive eigen-values
		rank <- min( rank, length(which(decomp$d > 0)) )
		rank <- min (rank, ncol(decomp$u))

		# truncate eigen-spectrum
		decomp$d = decomp$d[1:rank]
		decomp$u = decomp$u[,1:rank,drop=FALSE]	
	}
	
	a = gc()

	if( rank < 1 ){
		stop("rank < 1 is not valid")
	}

	if( is.null(W_til) ) W_til = matrix(0)
	if( ! is.matrix(W_til) ) W_til = as.matrix(W_til)

	if( is.big.matrix(features) ){ 
		ptr = features@address 
	}else{
		ptr = 0
	}

	if( ! is.null(map) ){
		chrom = as.character(map[,1])
		location = as.double(map[,2])

		# only keep entries in dcmp_features that are M= ncol(features)
		dcmp_features = dcmp_features[which(dcmp_features <= ncol(features))]

		if( length(dcmp_features) == 0){
			dcmp_features = c(-1)
		}

	}else{
		chrom = c("0")
		location = c(0)
		distance = 1
		dcmp_features = c(0)
	}	

	# if decomp is result of eigen()
	if( is.svd_decomp(decomp) ){
		pValues	<- .Call("R_lrgprApply", as.character(form_mod), features, ptr, env, terms-1, decomp$u, decomp$d^2, W_til, as.integer(rank), chrom, location, as.double(distance), dcmp_features-1, scale, as.numeric(delta), as.integer(reEstimateDelta), as.integer(nthreads), ! progress, package="lrgpr")
	}else{
		# if decomp is result of eigen()
		pValues	<- .Call("R_lrgprApply", as.character(form_mod), features, ptr, env, terms-1, decomp$vectors, decomp$values, W_til, as.integer(rank), chrom, location, as.double(distance), dcmp_features-1, scale, as.numeric(delta), as.integer(reEstimateDelta), as.integer(nthreads), ! progress, package="lrgpr")
	}

	gc()	

	# If varable exists
	# Restore value	
	#if( length(grep( "^SNP_tmp$", ls())) > 1) env[["SNP"]] = SNP_tmp

	# If there was not error generating p-values
	if( length(pValues) == ncol(features)){

		# assign marker names
		names(pValues) = colnames(features)

		return( pValues )
	}
} 



#' Fit standard linear or logistic model for many markers
#'
#' `glmApply' is analogous to 'lrgprApply', but fits standard linear or logistic models for many markers

#' @param formula standard linear modeling syntax as used in 'lm'.  SNP is a place holder for the each successive column of features
#' @param features a matrix where the statistical model is evaluated with SNP if formula replace by each column successively
#' @param terms indices of the coefficients to be tested.  The indices corresponding to SNP are used if terms is not specified
#' @param family gaussian() for a continuous response, and binomial() to fit a logit model for a binary response 
#' @param useMean if TRUE, replace missing entries with column mean.  Otherwise, do not evaluate the model for that column
#' @param nthreads number of to use for parallel execution
#' @param univariateTest perform univariate hypothesis test for each response for each feature in the loop variable
#' @param multivariateTest perform multivariate hypothesis test for each response (if more than one) for each feature.  Note that the runtime is cubic in the number of response variables
#' @param verbose print additional information
#' @param progress show progress bar
#'
#' @examples 
#' 
#' # Generate data
#' n = 100
#' p = 500
#' X = matrix(sample(0:2, n*p, replace=TRUE), nrow=n)
#' y = rnorm(n)
#' sex = as.factor(sample(1:2, n, replace=TRUE))
#'
#' # Fit model for all markers 
#' pValues = glmApply( y ~ sex + sex:SNP, features=X, terms=c(3,4))
#'
#'
#' # Multivariate model
#' n = 100
#' p = 1000
#' m = 10
#' 
#' Y = matrix(rnorm(n*m), nrow=n, ncol=m)
#' X = matrix(rnorm(n*p), nrow=n, ncol=p)
#' 
#' res = glmApply( Y ~ SNP, features = X, terms=2, multivariateTest=TRUE)
#' 
#' # p-values for univariate hypothesis test of each feature against 
#' # 	each response
#' res$pValues
#' 
#' # p-values for multivariate hypothesis test of each feature against 
#' # 	all responses are the same time
#' # returns the results of the Hotelling and Pillai tests
#' res$pValues_mv
#' 
#' # The multivariate test for X[,1]
#' res$pValues_mv[1,]
#' 
#' # The result is the same as the standard tests in R
#' fit = manova( Y ~ X[,1])
#' 
#' summary(fit, test="Hotelling-Lawley")
#' summary(fit, test="Pillai")
#' 
#' @export
glmApply <- function( formula, features, terms=NULL, family=gaussian(), useMean=TRUE, nthreads=detectCores(logical = TRUE), univariateTest=TRUE, multivariateTest=FALSE, verbose=FALSE, progress=TRUE ){

	env = parent.frame()

	formula = as.formula(formula)
	# convert formula to character string
	formChar = as.character.formula( formula )
	
	# Save value to restore later
	# Note that if there is a global SNP variable, AND lrgprApply is called in a subthread, AND another threads accesses SNP, then there will be a RACE CONDITION
	# If varable exists
	#if( length(grep( "^SNP$", ls(env))) > 1) SNP_tmp = SNP

	#####################
	# Argument checking #
	#####################

	##############
	# check data #
	##############
	
	if( ! is.na(nthreads) && nthreads < 1 ){
		stop("nthreads must be positive")
	}

	if( ! is.matrix(features) && ! is.big.matrix(features) ){
		features = as.matrix(features)
	}

	if( ! .is_supported_lrgpr(features) ){
		stop("Unsupported data type for features.\nSupported types are matrix and big.matrix.\nNote that sub.big.matrix is not currently supported")
	}	

	result = tryCatch({
		# save status from current state
		na.status = options("na.action")$na.action

		# Cause failure if y or X contain NA
		 options(na.action="na.fail")
		#options(na.action="na.omit")

	   	# Get response from formula to check identity or logistic link
		#++++++++++++++++++++++

		# do not fail if features[,1] has missing entry
		.lrgpr_tmp = set_missing_to_mean(features[,1])
		idx = which(is.na(.lrgpr_tmp))

		if( length(idx) > 0 ){
			.lrgpr_tmp[idx] = rnorm(length(idx))
		}
		
		assign(".lrgpr_tmp", .lrgpr_tmp, envir = env)

		form_mod = as.formula( .mm_replace_query_with( formChar, "SNP", ".lrgpr_tmp") )
		.y = .mm_get_response( form_mod, env)
		#.X = model.matrix.default( form_mod )

		# identify indeces that contain the SNP variable
		terms = grep(".lrgpr_tmp", colnames(model.matrix(form_mod, env)))

		#if( ! is.numeric(.y) && ! is.factor(.y)){
		#	stop("Response is not numeric")
		#}

	}, warning 	= function(w) { warning(w)
	}, error 	= function(e) { stop(e)
	}, finally 	= {
	    # restore status to previous state
		options(na.action=na.status)
	})	

	if( length(terms) == 0 ){
		stop("Must specify terms, cannot be determined from formula")
	}
	if( length(which( terms <= 0 )) > 0 ){
		stop(paste("Invalid values for terms:", paste(terms, collapse=', ')))
	}	

	if( verbose ){
		cat("terms:", terms, "\n")
	}

	if( family[2]$link == "logit" && sum(!(.y %in% c(0,1))) > 0 ){
		stop("A logistic link cannot be applied to this reponse.  Only 0/1 values are allowed")
	}
	if( sum( ! (family[2]$link %in% c("identity", "logit") ) ) > 0 ){
		stop("Only family values gaussian() and binomial() with canonical links are allowed")
	}

	if( is.big.matrix(features) ){ 
		ptr = features@address 
	}else{
		ptr = 0
	}

	# run regressions
	pValList <- .Call("R_glmApply", as.character(form_mod), features, ptr, env, terms-1, as.integer(nthreads), useMean, sum(family[2]$link == "identity"), univariateTest, multivariateTest, ! progress, package="lrgpr")

	gc()

	if( length(pValList) > 1){

		# If there was no error generating univariate p-values
		if( length(pValList$pValues) > 0){
			# assign marker names
			rownames(pValList$pValues) = colnames(features)
			#colnames(pValList$pValues) = colnames(.y)
		}else{
			pValList$pValues = NULL
		}

		# If varable exists
		# Restore value	
		#if( length(grep( "^SNP_tmp$", ls())) > 1) env$SNP = SNP_tmp

		# If there was no error generating multivariate p-values
		if( length(pValList$pValues_mv) > 0){
			# assign marker names
			rownames(pValList$pValues_mv) = colnames(features)
			colnames(pValList$pValues_mv) = c("Hotelling", "Pillai");
		}else{
			pValList$pValues_mv = NULL
		}

		return( pValList )
	}		
} 

#' Like glmApply, by linear instead of quadratic as a function of the number of covariates
#' @export
glmApply2 <- function( formula, features, terms=NULL, family=gaussian(), useMean=TRUE, nthreads=detectCores(logical = TRUE), univariateTest=TRUE, multivariateTest=FALSE, verbose=FALSE, progress=TRUE ){

	env = parent.frame()

	formula = as.formula(formula)
	# convert formula to character string
	formChar = as.character.formula( formula )
	
	# Save value to restore later
	# Note that if there is a global SNP variable, AND lrgprApply is called in a subthread, AND another threads accesses SNP, then there will be a RACE CONDITION
	# If varable exists
	#if( length(grep( "^SNP$", ls(env))) > 1) SNP_tmp = SNP

	#####################
	# Argument checking #
	#####################

	##############
	# check data #
	##############
	
	if( ! is.na(nthreads) && nthreads < 1 ){
		stop("nthreads must be positive")
	}

	if( ! is.matrix(features) && ! is.big.matrix(features) ){
		features = as.matrix(features)
	}

	if( ! .is_supported_lrgpr(features) ){
		stop("Unsupported data type for features.\nSupported types are matrix and big.matrix.\nNote that sub.big.matrix is not currently supported")
	}	

	result = tryCatch({
		# save status from current state
		na.status = options("na.action")$na.action

		# Cause failure if y or X contain NA
		 options(na.action="na.fail")
		#options(na.action="na.omit")

	   	# Get response from formula to check identity or logistic link
		#++++++++++++++++++++++

		# do not fail if features[,1] has missing entry
		.lrgpr_tmp = set_missing_to_mean(features[,1])
		idx = which(is.na(.lrgpr_tmp))

		if( length(idx) > 0 ){
			.lrgpr_tmp[idx] = rnorm(length(idx))
		}
		
		assign(".lrgpr_tmp", .lrgpr_tmp, envir = env)

		form_mod = as.formula( .mm_replace_query_with( formChar, "SNP", ".lrgpr_tmp") )
		.y = .mm_get_response( form_mod, env)
		#.X = model.matrix.default( form_mod )

		# identify indeces that contain the SNP variable
		terms = grep(".lrgpr_tmp", colnames(model.matrix(form_mod, env)))

		#if( ! is.numeric(.y) && ! is.factor(.y)){
		#	stop("Response is not numeric")
		#}

	}, warning 	= function(w) { warning(w)
	}, error 	= function(e) { stop(e)
	}, finally 	= {
	    # restore status to previous state
		options(na.action=na.status)
	})	

	if( length(terms) == 0 ){
		stop("Must specify terms, cannot be determined from formula")
	}
	if( length(which( terms <= 0 )) > 0 ){
		stop(paste("Invalid values for terms:", paste(terms, collapse=', ')))
	}	

	if( verbose ){
		cat("terms:", terms, "\n")
	}

	if( family[2]$link == "logit" && sum(!(.y %in% c(0,1))) > 0 ){
		stop("A logistic link cannot be applied to this reponse.  Only 0/1 values are allowed")
	}
	if( sum( ! (family[2]$link %in% c("identity", "logit") ) ) > 0 ){
		stop("Only family values gaussian() and binomial() with canonical links are allowed")
	}

	if( is.big.matrix(features) ){ 
		ptr = features@address 
	}else{
		ptr = 0
	}

	# run regressions
	pValList <- .Call("R_glmApply2", as.character(form_mod), features, ptr, env, terms-1, as.integer(nthreads), useMean, sum(family[2]$link == "identity"), univariateTest, multivariateTest, ! progress, package="lrgpr")

	gc()

	if( length(pValList) > 1){

		# If there was no error generating univariate p-values
		if( length(pValList$pValues) > 0){
			# assign marker names
			rownames(pValList$pValues) = colnames(features)
			#colnames(pValList$pValues) = colnames(.y)
		}else{
			pValList$pValues = NULL
		}

		# If varable exists
		# Restore value	
		#if( length(grep( "^SNP_tmp$", ls())) > 1) env$SNP = SNP_tmp

		# If there was no error generating multivariate p-values
		if( length(pValList$pValues_mv) > 0){
			# assign marker names
			rownames(pValList$pValues_mv) = colnames(features)
			colnames(pValList$pValues_mv) = c("Hotelling", "Pillai");
		}else{
			pValList$pValues_mv = NULL
		}

		return( pValList )
	}		
} 



#' Compute AIC/BIC/GCV for lrgpr() model as rank changes
#'
#' `criterion.lrgpr' evaluate information criteria to select an optimal rank
#'
#' @param formula standard linear modeling syntax as used in 'lm'
#' @param features matrix from which the SVD is performed
#' @param order sorted indices of features.  When rank is 10, decomp = svd(X[,order[1:10]]) 
#' @param rank array with elements indicating the number of confounding covariates to be used in the random effect.
#'
#' @seealso plot.criterion.lrgpr, cv.lrgpr
#' 
#'#' @examples
#' n = 300
#' p = 5000
#' X = matrix(sample(0:2, n*p, replace=TRUE), nrow=n)
#' 
#' dcmp = svd(X)
#' 
#' # simulate response
#' h_sq = .8
#' eta = dcmp$u[,1:2] %*% rgamma(2, 2, 1)
#' error_var = (1-h_sq) / h_sq  * var(eta)
#' y = eta + rnorm(n, sd=sqrt(error_var))
#' 
#' # Get ordering based on marginal correlation
#' i = order(cor(y, X)^2, decreasing=TRUE) 
#' 
#' # Fit AIC / BIC / GCV based on degrees of freedom
#' fit = criterion.lrgpr( y ~ 1, features=X, order=i)
#' 
#' plot.criterion.lrgpr(fit)
#' 
#' @export
criterion.lrgpr = function( formula, features, order, rank = c(seq(1, 10), seq(20, 100, by=10), seq(200, 1000, by=100)) ){

	if( length(rank) < 2){
		stop("rank must have at least 2 elements")		
	}

	if( ! is.matrix(features) && ! is.big.matrix(features) ){
		features = as.matrix(features)
	}

	if( ! .is_supported_lrgpr(features) ){
		stop("Unsupported data type for features.\nSupported types are matrix and big.matrix.\nNote that sub.big.matrix is not currently supported")
	}	

	# sort
	rank = sort(rank)

	# discard numbers that are larger than the number of features in the confounder matrix
	rank = rank[which( rank <= ncol(features) )]

	crit_fxn = function( ncon ){
		dcmp = svd( scale(set_missing_to_mean(features[,order[1:ncon]])) )

		fit = lrgpr( formula, dcmp)

		return( c(fit$AIC, fit$BIC, fit$GCV, fit$logLik, fit$df) )
	}

	result = matrix(0, nrow=length(rank), ncol=5)

	for(i in 1:length(rank)){
		result[i,] = crit_fxn( rank[i] )
	}

	#registerDoParallel(cores=nthreads)
	#result = foreach( ncon=rank, .combine='cbind') %do% crit_fxn( ncon )
	#result = t(result)

	colnames(result) =  c("AIC", "BIC", "GCV", "logLik", "df")
	rownames(result) = rank
	result = as.data.frame(result)

	best = rank[apply(result[,1:3], 2, which.min)]
	names(best) = c("AIC", "BIC", "GCV")
	best = as.data.frame(t(best))
	rownames(best) = c()

	result$rank = as.numeric(rownames(result))

	return( list(criteria=result[,1:3], best=best, df = result$df, logLik = result$logLik, rank=result$rank ) )
}


#' Plot AIC/BIC/GCV values for lrgpr() model as rank changes
#'
#' `plot.criterion.lrgpr' plots the criteria returned by 'criterion.lrgpr' 
#'
#' @param x list returned by 'criterion.lrgpr'
#' @param col array of 3 colors 
#' @param ... other arguments 
#'
#' @seealso criterion.lrgpr
#'
#' @export
plot.criterion.lrgpr = function( x, col=rainbow(3),...){
	ylim = range(c(x$criteria$AIC, x$criteria$BIC))	

	xvals = as.numeric(rownames(x$criteria))

	par(mar=c(4,4,2,4))
	plot(1, type='n', xlim=range(xvals), xlab="# of markers used", ylim=ylim, ylab="AIC/BIC scale",  main=bquote(bold(paste("Criterion scores using ", df[e], sep=" "))),...)

	points( xvals, x$criteria$AIC, col=col[1], pch=20)
	points(xvals, x$criteria$BIC, col=col[2], pch=20 )
	legend( 0.10*max(xvals), ylim[2], c("AIC", "BIC", "GCV"), fill=rainbow(3), bty='n')

	axis(side=4, at=seq(ylim[1], ylim[2], length.out=5), labels=format(seq(min(x$criteria$GCV), max(x$criteria$GCV), length.out=5), digits=2) )

	par(new=T)
	loc = plot(as.numeric(rownames(x$criteria)), x$criteria$GCV, col=col[3], pch=20, xlab='', ylab='', xaxt='n', yaxt='n')

	mtext("GCV scale", side=4, line=3)
}


#' Plot Results of Cross-validation
#'
#' Plot results of 'cv.lrgpr', which fits cross-validation for multiple ranks of the LRGPR
#'
#' @param x result of cv.lrgpr
#' @param ylim limits of y-axis
#' @param xlim limits of x-axis
#' @param pch pch
#' @param col col
#' @param main main
#' @param xlab xlab  
#' @param ylab ylab 
#' @param ... other parameters fed to plot()
#' 
#' @export
plot.cv.lrgpr = function( x, ylim=c(min(x$cve - x$cvse), max(x$cve + x$cvse)), xlim=range(x$rank), pch=20, col="red", main="Cross validation", xlab="# of markers used", ylab = "Cross validation error",...){

	par(mar=c(4,4,2,4))
	plot(x$rank, x$cve, ylim=ylim, xlim=xlim, main=bquote(bold( .(main))), xlab=xlab, ylab=ylab, col=col, pch=pch,...)

	error.bar( x$rank, x$cve, x$cvse)
}

#' Loss function
#'
#' Compare observed and fitted response under some loss function
#'
#' @param y observed response
#' @param yhat fitted response
#' @param family "gaussian" or "binomial"
#' 
#' @export
loss.lrgpr <- function(y, yhat, family){
	n <- length(y)
	if (family=="gaussian") val <- (y-yhat)^2
	if (family=="binomial") {
		val <- matrix(NA, nrow=nrow(yhat), ncol=ncol(yhat))
		val[y==1,] <- -2*log(yhat[y==1, , drop=FALSE])
		val[y==0,] <- -2*log(1-yhat[y==0, , drop=FALSE])
	}
	val
}

#' Cross-validation for LRGPR
#'
#' `cv.lrgpr' fits cross-validation for multiple ranks of the LRGPR
#'
#' @param formula standard linear modeling syntax as used in 'lm'
#' @param features matrix from which the SVD is performed
#' @param order sorted indices of features.  When rank is 10, decomp = svd(X[,order[1:10]]) 
#' @param nfolds number of training sets
#' @param rank array with elements indicating the number of confounding covariates to be used in the random effect.
#' @param nthreads number of threads to be used
#
#
#' @examples
#' n = 300
#' p = 5000
#' X = matrix(sample(0:2, n*p, replace=TRUE), nrow=n)
#' 
#' dcmp = svd(X)
#' 
#' # simulate response
#' h_sq = .8
#' eta = dcmp$u[,1:2] %*% rgamma(2, 2, 1)
#' error_var = (1-h_sq) / h_sq  * var(eta)
#' y = eta + rnorm(n, sd=sqrt(error_var))
#' 
#' # Get ordering based on marginal correlation
#' i = order(cor(y, X)^2, decreasing=TRUE) 
#' 
#' # Fit cross-validation
#' fit = cv.lrgpr( y ~ 1, features=X, order=i)
#' 
#' plot.cv.lrgpr(fit)
#' 
#' 
#' @export
cv.lrgpr <- function( formula, features, order, nfolds=10, rank = c(seq(0, 10), seq(20, 100, by=10), seq(200, 1000, by=100)), nthreads=1 ){
  
	if( ! is.matrix(features) && ! is.big.matrix(features) ){
		features = as.matrix(features)
	}

	if( ! .is_supported_lrgpr(features) ){
		stop("Unsupported data type for features.\nSupported types are matrix and big.matrix.\nNote that sub.big.matrix is not currently supported")
	}	

  	mf <- match.call(expand.dots=FALSE)
	m <- match(c("formula", "data"), names(mf), 0L)
	mf <- mf[c(1L, m)]
	mf$drop.unused.levels <- TRUE
	mf[[1L]] <- as.name("model.frame")
	mf <- eval(mf, parent.frame())
	mt <- attr(mf, "terms")
	y <- model.response(mf, "numeric")
	X <- model.matrix(mt, mf, contrasts)

	if( length(rank) < 2){
		stop("rank must have at least 2 elements")		
	}
	if( nfolds < 2){
		stop("nfolds must be >=2")		
	}

	# sort
	rank = sort(rank)

	# discard numbers that are larger than the number of features in the confounder matrix
	rank = rank[which( rank <= ncol(features) )]

	features = scale(features[,order[1:max(rank)]])

	E <- matrix(NA, nrow=length(y), ncol=length(rank))

	n <- length(y)
	cv.ind <- ceiling(sample(1:n)/n*nfolds)

	cv_fxn = function(ncon){
		# if rank is zero, fit linear model
		if( ncon == 0){
			fit.i <- lm(y[i_train] ~ X[i_train,]-1)
			yhat <- X[i_test,,drop=FALSE] %*% fit.i$coefficients
		}else{			

			# decomp of X of training set for first ncon markers
			# dcmp = svd(X_confounders_sorted[i_train,1:ncon])
			dcmp = svd( set_missing_to_mean(features[i_train,1:ncon]) ) 
			
			# RRM of samples in the test set
			# K_test = tcrossprod(X_confounders_sorted[i_test,1:ncon], X_confounders_sorted[i_train,1:ncon])
			K_test = tcrossprod( set_missing_to_mean(features[i_test,1:ncon]), set_missing_to_mean(features[i_train,1:ncon]) )

			#fit.i <- lrgpr_rankSearch(y[i_train] ~ X[i_train,]-1, decomp=dcmp, maxRank=100)
			fit.i <- lrgpr(y[i_train] ~ X[i_train,]-1, decomp=dcmp)

			# In-sample and out-of-sample prediction code gives same results
			# cbind(predict(fit.i), predict.lrgpr(fit.i, X[i_train,,drop=FALSE], RRM[i_train,i_train]))
			yhat <- predict.lrgpr(fit.i, X[i_test,,drop=FALSE], K_test)
		}

		# return the MSE between observed and prediected response
		return( loss.lrgpr(y[i_test], yhat, family="gaussian") )
	}

	#registerDoParallel(cores=nthreads)
	#mcoptions <- list(preschedule=FALSE)

	#unlistfn <- function(z){ matrix(unlist(z), ncol=length(rank), byrow=FALSE) }

	for(i in 1:nfolds){

		cat("\rStarting CV fold #", i,"/", nfolds)

		# arrays of train/test indices
		i_train <- which(cv.ind!=i)
		i_test <- which(cv.ind==i)

		# mclapply can stall on when it is invoked a second time
		#E[cv.ind==i,] = unlistfn( mclapply( rank, cv_fxn, mc.cores=nthreads) )	

		# foreach parallelization seems to be faster than mclapply
		E[cv.ind==i,] = foreach(ncon=rank, .combine='cbind') %do% cv_fxn( ncon )
	}
 
 	cat("\n")

	## Eliminate saturated lambda values, if any
	ind <- which(apply(is.finite(E), 2, all))
	E <- E[,ind]

	cve <- apply(E, 2, mean)
	cvse <- apply(E, 2, sd) / sqrt(n)
	min <- which.min(cve)
	
	structure(list(cve=cve, cvse=cvse, rank=rank, min=min, best=min, class="cv.lrgpr"))
}

#' Convert ASCII to binary file
#'
#' `convertToBinary' converts TPED/DOSAGE/GEN files to binary format
#'
#' @param filename file to be converted
#' @param filenameOut name of binary file produced
#' @param format specify 'TPED', 'DOSAGE' or 'GEN'
#'
#' @details
#' \itemize{
#' \item{TPED:	}{plink file can be in either --recode or --recode12 format}
#' \item{DOSAGE:	}{file follows plink format: http://pngu.mgh.harvard.edu/~purcell/plink/dosage.shtml}
#' }
#'
#' Example:
#'
#'     SNP    A1  A2   F1   I1     F2   I2     F3   I3
#'
#'   rs0001   A   C   0.98 0.02   1.00 0.00    0.00 0.01
#'
#'   rs0002   G   A   0.00 1.00   0.00 0.00    0.99 0.01
#'
#' where the F* values correspond to the dosage values
#'
#' \itemize{
#' \item{GEN:	}{file follow OXFORD format}
#' }
#'
#' @export
convertToBinary = function( filename, filenameOut, format, nthreads=detectCores(logical=TRUE) ){

	if( path.expand(filename) == path.expand(filenameOut) ){
		stop("Cannot read and write to the same file")
	}

	if( ! file.exists(filename) ){		
		stop("File does not exist: ", filename)
	}

	isZipFile = as.integer(length(grep(".gz$", filename)) > 0 )

	if( isZipFile ){
		cat("gzip file detected...\n")
	}

	if( ! (format %in% c("TPED", "GEN", "DOSAGE")) ){		
		stop("Invalid format: ", format)
	}

	res = .Call("R_convertToBinary", filename, filenameOut, format, isZipFile, nthreads, package="lrgpr")

	if( ! res$success ){
		stop("File is not properly formatted as ", format)
	}

	if( ! is.list(res) ){
		stop("File is not correctly formatted")
	}

	ret = list(sharedType	= 'FileBacked',
               filename 	= basename( filenameOut ),
               totalRows	= res$nrow,
               totalCols 	= res$ncol,
               rowOffset 	= c(0, res$nrow),
               colOffset 	= c(0, res$ncol),
               nrow 		= res$nrow,
               ncol 		= res$ncol,
               rowNames 	= NULL, 
               colNames 	= res$colNames, 
               type			= "double", 
               separated = FALSE)

	# Save structure in ASCII format
	#dput( new("big.matrix.descriptor", description = ret ), file=paste(filenameOut, "_text", sep=''))

	# Save structure in BINARY format
	saveRDS( new("big.matrix.descriptor", description = ret ), file=paste(filenameOut, "_descr", sep=''))
}

readBinary = function( filename, N ){

	res = .Call("R_readBinary", filename, N, package="lrgpr")

	return( res)
}

#' Calculate allele frequency
#' 
#' @param X matrix where each column is a marker coded 0,1,2 or with dosage values in this range
#' @param nthreads number of threads to use
#' @param progress show progress bar 
#' @export 
getAlleleFreq = function( X, nthreads=detectCores(logical=TRUE), progress=TRUE){

	if( ! is.matrix(X) && ! is.big.matrix(X) ){
		X = as.matrix(X)
	}

	if( ! .is_supported_lrgpr(X) ){
		stop("Unsupported data type for features.\nSupported types are matrix and big.matrix.\nNote that sub.big.matrix is not currently supported")
	}	

	if( is.big.matrix(X) ){ 
		ptr = X@address 
	}else{
		ptr = 0
	}

	# run allele frequency calculations
	allelefreq <- .Call("R_getAlleleFreq", X, ptr, as.integer(nthreads), !progress, package="lrgpr")

	gc()

	return( allelefreq )
}	

#' @export 
getAlleleFreq2 = function( X, nthreads=detectCores(logical=TRUE), progress=TRUE){

	if( ! is.matrix(X) && ! is.big.matrix(X) ){
		X = as.matrix(X)
	}

	if( ! .is_supported_lrgpr(X) ){
		stop("Unsupported data type for features.\nSupported types are matrix and big.matrix.\nNote that sub.big.matrix is not currently supported")
	}	

	if( is.big.matrix(X) ){ 
		ptr = X@address 
	}else{
		ptr = 0
	}

	# run allele frequency calculations
	allelefreq <- .Call("R_getAlleleFreq2", X, ptr, as.integer(nthreads), !progress, package="lrgpr")

	gc()

	return( allelefreq )
}	





#' Count missing values
#' 
#' @param X matrix where each column is a marker
#' @param nthreads number of threads to use
#' @param progress show progress bar 
#' @export 
getMissingCount = function( X, nthreads=detectCores(logical=TRUE), progress=TRUE){

	if( ! is.matrix(X) && ! is.big.matrix(X) ){
		X = as.matrix(X)
	}

	if( ! .is_supported_lrgpr(X) ){
		stop("Unsupported data type for features.\nSupported types are matrix and big.matrix.\nNote that sub.big.matrix is not currently supported")
	}	

	if( is.big.matrix(X) ){ 
		ptr = X@address 
	}else{
		ptr = 0
	}

	# count missing entries for each column
	missingCount <- .Call("R_getMissingCount", X, ptr, as.integer(nthreads), !progress, package="lrgpr")

	gc()
	
	return( missingCount )
}	

#' EValuate variance for each column
#' 
#' @param X matrix where each column is a marker
#' @param nthreads number of threads to use
#' @param progress show progress bar 
#' @export 
getAlleleVariance = function( X, nthreads=detectCores(logical=TRUE), progress=TRUE){

	if( ! is.matrix(X) && ! is.big.matrix(X) ){
		X = as.matrix(X)
	}

	if( ! .is_supported_lrgpr(X) ){
		stop("Unsupported data type for features.\nSupported types are matrix and big.matrix.\nNote that sub.big.matrix is not currently supported")
	}	

	if( is.big.matrix(X) ){ 
		ptr = X@address 
	}else{
		ptr = 0
	}

	# count missing entries for each column
	alleleVar <- .Call("R_getAlleleVariance", X, ptr, as.integer(nthreads), !progress, package="lrgpr")

	gc()
	
	return( alleleVar )
}	
