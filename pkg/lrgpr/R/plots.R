

#' QQ plot
#' 
#' QQ plot and lambda_GC optimized for large datasets.
#'
#' @param p_values vector, matrix or list of p-values
#' @param col colors corresponding to the number of columns in matrix, or entries in the list
#' @param main title
#' @param pch pch
#' @param errors show 95\% confidence interval
#' @param lambda calculate and show genomic control lambda.  Lambda_GC is calcualted using the 'median' method on p-values > p_thresh.
#' @param p_thresh Lambda_GC is calcualted using the 'median' method on p-values > p_thresh.
#' @param showNames show column names or list keys in the legend
#' @param ylim ylim 
#' @param xlim xlim
#' @param plot make a plot.  If FALSE, returns lamda_GC values without making plot
#' @param new make a new plot.  If FALSE, overlays QQ over current plot
#' @param box.lty box line type
#' @param collapse combine entries in matrix or list into a single vector
#' @param ... other arguments
#' 
#' @examples 
#' 
#' p = runif(1e6)
#' QQ_plot(p)
#' 
#' # get lambda_GC values without making plot
#' lambda = QQ_plot(p, plot=FALSE)
#'
#' @export 
QQ_plot = function(p_values, col=rainbow(min(length(p_values), ncol(p_values))), main="", pch=20, errors=TRUE, lambda=TRUE, p_thresh = 1e-6, showNames=FALSE, ylim=NULL, xlim=NULL, plot=TRUE,new=TRUE, box.lty=par("lty"), collapse=FALSE,...){

	if( collapse ){
		p_values = as.vector(unlist(p_values, use.names=FALSE))
	}

	# convert array, vector or matrix into list
	if( ! is.list(p_values) ){

		names(p_values) = c()

		keys = colnames(p_values)

		# if there is am empty name
		if( "" %in% keys ){
			keys[which("" %in% keys)] = "NULL"
		}

		p_values = as.matrix(p_values)

		p_values_list = list()

		for(i in 1:ncol(p_values) ){
			p_values_list[[i]] = p_values[,i]
		}

		names(p_values_list) = keys
	
		p_values = p_values_list
		rm( p_values_list )
	}		

	rge = range(p_values, na.rm=TRUE)

	if( rge[1] < 0 || rge[2] > 1 ){
		stop("p_values outside of range [0,1]")
	}

	# assign names to list entries if they don't exist
	if( is.null( names( p_values ) ) ){
		names( p_values ) = 1:length( p_values )
	}

	# set pch values if not defined
	if( is.null( pch ) ){
		pch = rep(20, length(p_values))
	}
	if( length(pch) == 1){
		pch = rep(pch, length(p_values))
	}

	p_values = as.list(p_values)

	# Set the x and y ranges of the plot to the largest such values 
	#	encountered in the data 
	ry = 0; rx = 0

	for( key in names( p_values ) ){

		# remove NA values
		p_values[[key]] = p_values[[key]][which( ! is.na( p_values[[key]] ))]

		j = which(p_values[[key]] == 0)

		if( length(j) > 0){
			p_values[[key]][j] = min(p_values[[key]][-j])
		}
		
		#ry = max(ry,range( -log10(p_values[[key]]), finite=TRUE) )
		ry = max(ry, -log10(min(p_values[[key]])) )

		rx = max(rx, -log10( 1/(length(p_values[[key]])+1)  ))		
	}

	if( ! is.null(ylim) ){
		ry = max(ylim)
	}
	if( ! is.null(xlim) ){
		rx = max(xlim)
	}

	r = max(rx, ry)

	xlab = expression(-log[10]('expected p-value'))
	ylab = expression(-log[10]('observed p-value'))

	if( plot && new ){
		# make initial plot with proper ranges
		plot(1, type='n', las=1, pch=20, xlim=c(0, rx), ylim=c(0, ry), xlab=xlab, ylab=ylab, main=main,...)
		abline(0, 1)		
	}

	lambda_values = c()
	se = c()

	# Plots points for each p-value set
	# Since 90% of p-values should be < .1 and 99% < 0.01, plotting all of these points is
	# time consuming and takes up space, even though all the points appear on top of each other
	# Therefore, thin the p-values at the beginning and increases density for smaller p-values
	# This allows saving QQ plots as a PDF.....with out thinning, a PDF can be VERY large
	i = 1
	for( key in names( p_values ) ){

		observed = sort(p_values[[key]], decreasing=TRUE)

		# Calculated lambda_GC directly from sorted observed values
		# This is MUCH faster than using estlambda() from GenABEL
		# Compute only for p-values < 1e-6
		obs = sort(p_values[[key]][which(p_values[[key]] > p_thresh)], decreasing=TRUE)

		lambda_values[i] = qchisq(median( obs), 1, lower.tail = FALSE) / qchisq(0.5, 1)

		if( plot ){
			# if a p-value is exactly zero, set it equal to the smallest nonzero p-value
			j = which( observed == 0)

			if( length(j) > 0){
				observed[j] = min(observed[-j])
			}
			
			expected = (length(observed):1) / (length(observed)+1)		

			p = length(expected)

			if( p < 1e6 ){
				# Thin p-values near 1, and increase density for smaller p-values 
				intervals = ceiling(c(1, 0.2*p, 0.4*p, 0.7*p, 0.9*p, 0.95*p, 0.99*p, p))
				scaling = c(28, 200, 800, 1000, 3000, 5000, p)
			}else{
				intervals = ceiling(c(1, 0.2*p, 0.4*p, 0.7*p, 0.9*p, 0.95*p, 0.99*p, 0.995*p, 0.999*p, p))
				scaling = c(28, 200, 800, 1000, 3000, 5000, 10000, 50000, p)
			}

			for( j in 1:length(scaling) ){
				k = seq(intervals[j], intervals[j+1], intervals[j+1]/scaling[j])
				points(-log10(expected[k]), -log10(observed[k]),  col=col[i], pch=pch[i])#,...)
			}
		}

		i = i + 1
	}
	
	# Plot lambda values, if desired
	if( plot && lambda ){
		# Calculated lambda using estlambda() from GenABLE
		# As of Sept24, 2013, I calculated it much faster using the sorted observed p-values

		#result = sapply( p_values, estlambda, plot=FALSE,method=method)
		#result = matrix(unlist(result), ncol=2, byrow=TRUE)
		#lambda_values = result[,1]
		#se = result[,2]

		if( ! showNames ){
			namesStrings = rep('', length(names( p_values )) )
		}else{
			namesStrings = paste( names( p_values ), ": ", sep='')
		}

		legend("topleft", legend = paste(namesStrings, format(lambda_values, digits = 4, nsmall = 3)), col=col, pch=15, pt.cex=1.5, title=expression(lambda['GC']), box.lty=box.lty)
	}

	# Standard errors 
	# From Supplementary information from Listgarten et al. 2010. PNAS	
	if( plot && errors ){
		error_quantile = 0.95

		# Using M:1 plots too many points near zero that are not discernable 
		# 	Reduce the number of points near zero 
		
		#plot(1, type='n', las=1, pch=20, xlim=c(0, rx), ylim=c(0, ry))

		M = length(p_values[[key]])
		#alpha = M:1
			
		alpha = seq(M, M/10 + 1, length.out=1000)

		if( M/10 > 1) alpha = append(alpha, seq(M/10, M/100 + 1, length.out=1000))
		if( M/100 > 1) alpha = append(alpha, seq(M/100, M/1000 + 1, length.out=1000))
		if( M/1000 > 1) alpha = append(alpha, seq(M/1000, M/10000 + 1, length.out=10000))
		alpha = append(alpha, seq(min(alpha), 1, length.out=10000))

		alpha = round(alpha)
		beta = M - alpha + 1

		x_top = qbeta(error_quantile, alpha, beta)
		x_bot = qbeta(1-error_quantile, alpha, beta)

		lines( -log10(alpha/(M+1)), -log10(x_top))
		lines( -log10(alpha/(M+1)), -log10(x_bot))
	}

	return( invisible(lambda_values) )
}