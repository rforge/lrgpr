

#' Read plink FAM/TFAM files
#'
#' Read FAM/TFAM file into a dataframe.  This function is the same as read.fam
#' 
#' @param file location of FAM/TFAM file
#' @export
read.tfam = function( file ){

	if( ! file.exists( file) ){
		stop("File does not exist: ", file)		
	}

	if( length(grep(".tfam|.fam", file)) == 0 ){
		stop(paste("File does not have '.fam' or '.tfam' suffix:", file));
	}

	FAM = data.frame(read.table( file, stringsAsFactors=FALSE))
	colnames( FAM ) = c("familyID", "ID", "pID", "mID", "sex", "phenotype")

	FAM$sex = as.factor(FAM$sex)

	i = which( FAM$phenotype == -9 )

	if( length(i) > 0 ){
		FAM$phenotype[i] = NA
	}

	return( FAM )
}


#' Read plink FAM/TFAM files
#'
#' Read FAM/TFAM file into a dataframe.  This function is the same as read.tfam
#' 
#' @param file location of FAM/TFAM file
#' @export
read.fam = read.tfam

#' @export
.pkg.env <- new.env()

# Read plink TPED file
#
# Read TPED in 0/1 format into R as a matrix.  A subset of consecutive markers can be read with start_line and nlines    
# 
# @param file location of TPED file
# @param missing instruction for how to deal with missing data: missing='NA' sets all missing values to NA, missing='mean' sets values to the mean value for that marker
# @param start_line index of the first line to read, where the indexing starts at 1 
# @param nlines number of markers to read after and including start_line.  If nlines=Inf, read entire file starting at start_line. 
# @param quiet print information about data processing
# @param markerNames array of marker names to be read from the file
# @export
read.tped = function( file, missing=NA, start_line=1, nlines=Inf, quiet=FALSE, markerNames=NULL){
	
	useTPED = TRUE;
	# determine if code has been called trough read.gen
	if( regexpr("^read.gen", match.call())[1] == 1){
		useTPED = FALSE
	}	

	file = path.expand( file )

	if( useTPED && length(grep(".tped", file)) == 0 ){
		stop(paste("File does not have '.tped' suffix:", file));
	}
	if( ! useTPED && length(grep(".gen", file)) == 0 ){
		stop(paste("File does not have '.gen' suffix:", file));
	}
	if( ! file.exists(file) ){
		stop(paste("File does not exist:", file));
	}
	if( start_line <= 0 ){
		stop(paste("start_line must be greater than zero:", start_line));
	}
	if( nlines <= 0 ){
		stop(paste("nlines must be positive:", nlines));
	}
	if( ! is.na(missing) && missing != 'NA' && missing != "mean"){
		stop(paste("Invalid value for missing:", missing))
	}
	
	if( is.infinite(nlines) ) nlines = -1;

	if( is.null(.pkg.env$plink_GEH_files) ){		
		.pkg.env$plink_GEH_files = list()
	}

	key = paste(file, file.info(file)$size, file.info(file)$mtime)

	if( length(markerNames) == 0 ){
		markerNames = c("NULL_place_holder")
	}

	# if .pkg.env$plink_GEH_files[[key]] has not been initialized
	# 	and markerNames is (effectively) empty
	if( ! is.null( .pkg.env$plink_GEH_files[[key]] ) && markerNames[1] == "NULL_place_holder" ){

		n_tokens = .pkg.env$plink_GEH_files[[key]]$n_tokens
		n_lines_total = .pkg.env$plink_GEH_files[[key]]$n_lines_total

		# indicates that location cannot be based on previous file access
		byte_loc = -1		
	
		# if start_line corresponds to the next line from the last access
		if( .pkg.env$plink_GEH_files[[key]]$next_line == start_line ){
			byte_loc = .pkg.env$plink_GEH_files[[key]]$byte_loc
		}
		#cat("byte_loc: ", byte_loc, "\n")

		res <- .Call("read_tped_gen", file, start_line-1, nlines, n_tokens, n_lines_total, byte_loc, markerNames, useTPED, package="plink")
	}else{

		markerNames = sub(" ", "", markerNames)

		# this command will be used if
		#   markerNames contains at least one entry, 
		#   or of the entire TPED file should be read
		res <- .Call("read_tped_gen", file, start_line-1, nlines, -1, -1, -1, markerNames, useTPED, package="plink")
	}		

	# If no markers are read
	if( is.null(res$X) ){
		res$X = matrix(0,0,0)		
		
		info = as.data.frame(matrix(0,nrow=0, ncol=6))
		colnames(info) = c("name","chrom", "genetic_pos", "position", "allele1", "allele2") 

		return(list(X=res$X, info=info))
	}
	
	if( nlines == -1 ) nlines = Inf;

	if( length(res$X) == 0){
		warning(paste("Values for either start_line or nlines are not valid for the specified genotype file:\n file = ", file, "\n start_line = ",  start_line, "\n nlines = ", nlines, sep=''))
	}

	# Remove markers that are all missing
	frq = getAlleleFreq( res )

	i = which(is.na(frq))

	if( length(i) > 0 ){
		res$X 			= res$X[,-i]
		res$name 		= res$name[-i]
		res$chromosome 	= res$chromosome[-i] 
		res$pos_genetic = res$pos_genetic[-i]
		res$pos_physical= res$pos_physical[-i]
		res$allele1 	= res$allele1[-i]
		res$allele2 	= res$allele2[-i]
		res$imputed 	= res$imputed[-i]
	}

	if( ! is.na(missing) && missing == 'mean'){
		if( ! quiet ) cat("Setting missing to marker means...\n")
		mu = colMeans(res$X, na.rm=TRUE)
	
		.Call("set_missing_values", res$X, mu, package="plink")	
	}
	
	.pkg.env$plink_GEH_files[[key]] = list()
	.pkg.env$plink_GEH_files[[key]]$n_tokens = res$n_tokens
	.pkg.env$plink_GEH_files[[key]]$n_lines_total = res$n_lines_total
	.pkg.env$plink_GEH_files[[key]]$byte_loc = res$byte_loc
	.pkg.env$plink_GEH_files[[key]]$next_line = start_line + ncol(res$X)

	gc()

	if( length(unique(res$name)) != length(res$name) ){
		warning("Marker names are not unique in the genotype file")
	}

	if( ncol(res$X) != length(res$name) ){
		warning("Not all marker names were found in genotype file")
	}

	res$X = res$X[,1:length(res$name),drop=FALSE] 

	colnames(res$X) = res$name

	info = as.data.frame(cbind(res$name, res$chromosome, res$pos_genetic, res$pos_physical, res$allele1, res$allele2, res$imputed), stringsAsFactors=FALSE)
	colnames(info) = c("name","chrom", "genetic_pos", "position", "allele1", "allele2", "imputed") 
	info$chrom = as.integer(info$chrom)
	info$genetic_pos = as.numeric(info$genetic_pos)
	info$position = as.integer(info$position)
	info$imputed = as.logical(info$imputed)

	#info$allele1[which(is.numeric(info$allele1))] = NA
	#info$allele2[which(is.numeric(info$allele2))] = NA

	return(list(X=res$X, info=info))
}

# Read plink GEN file in OXFORD format
#
# Read GEN file in OXFORD format into R as a matrix.  A subset of consecutive markers can be read with start_line and nlines    
# 
# @param file location of GEN file
# @param missing instruction for how to deal with missing data: missing='NA' sets all missing values to NA, missing='mean' sets values to the mean value for that marker
# @param start_line index of the first line to read, where the indexing starts at 1 
# @param nlines number of markers to read after and including start_line.  If nlines=Inf, read entire file starting at start_line. 
# @param quiet print information about data processing
# @param markerNames array of marker names to be read from the file
# @export
read.gen = read.tped

