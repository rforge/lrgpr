# Write to plink TPED format
#
# Write R matrix to plink-TPED format 
# 
# @param genoObj object produced by read.tped() or read.gen()
# @param file output file
# @param prefix if colnames(X) == "", the markers will be called snp1, snp2, etc.
# @export
write.tped = function( genoObj, file, prefix="snp"){

	useTPED = TRUE;
	# determine if code has been called trough read.gen
	if( regexpr("^write.gen", match.call())[1] == 1){
		useTPED = FALSE
	}	

	file = path.expand( file )

	if( useTPED && length(grep(".tped", file)) == 0 ){
		stop(paste("File does not have '.tped' suffix:", file));
	}

	if( length(colnames(genoObj$X)) == 0){		
		colnames(genoObj$X) = paste(prefix, 1:p, sep='')
	}

	a <- .Call("write_tped", genoObj$X, genoObj$info$allele1, genoObj$info$allele2, as.integer(genoObj$info$chrom), as.numeric(genoObj$info$genetic_pos), as.integer(genoObj$info$position), file, colnames(genoObj$X), useTPED, package="plink")
}

# Write to OXFORD GEN format
#
# Write R matrix to GEN format 
# 
# @param genoObj object produced by read.tped() or read.gen()
# @param file output file
# @param prefix if colnames(X) == "", the markers will be called snp1, snp2, etc.
# @export
write.gen = write.tped


# Calculate allele frequency
#
# Calculate allele frequency with respect to the first allele
# 
# @export
alleleFreq = function( geno ){
	n = nrow(geno$X)
	(colSums(geno$X) + n)/(2*n)
}

# Screen markers by maf
#
# @export
mafScreen = function(geno, mafCutoff){

	freq = alleleFreq(geno)

	maf = apply(cbind(freq, 1-freq), 1, min)

	i = which( maf <= mafCutoff)

	if( length(i) > 0){
		geno$X = geno$X[,-i]
		geno$info = geno$info[-i,]
	}

	return( geno )
}


# Merge two genetic marker sets from the same individuals
#
# Add new markers to an existing genotype data structure
# 
# @param geno1 primary genotype dataset whose markers take precedence 
# @param geno2 secondar genotype dataset whose markers are inclded only of the marker names do not appear in geno1
# @return genotype datasets with augmented X and info components
# @export
merge.geno = function( geno1, geno2 ){

	i = match(geno1$info$name, geno2$info$name)
	i = i[which(!is.na(i))]

	geno = list()
	geno$X = cbind(geno1$X, geno2$X[,-i])
	geno$info = rbind(geno1$info, geno2$info[-i,])

	if( length(geno$info$name[duplicated(geno$info$name)]) != 0){
		stop("Merge produced duplicated names")
	}

	return(geno)
}

# @export
hwe.test = function( X ){

	res <- .Call("get_allele_table_multi", X, package="plink")

	#apply( res, 1, function(a){pchisq(HWChisq( a )$chisq, 1, lower.tail=FALSE)} )
}

# gives same results as hwe.test except when a dosage value is +/- 0.5
#	so that is it equi-distant from two classes
hwe.test_slow = function( x ){

	# convert dosages to their closest allele values
	#	then determine which class they fall in
	res = apply(cbind( (x- -1)^2, x^2, (x-1)^2), 1, which.min)

	# construct counts for 3 classes
	counts = c(length(which(res==1)), length(which(res==2)), length(which(res==3)))

	# report p-value
	#pchisq(HWChisq( counts )$chisq, 1, lower.tail=FALSE)
}




#read.grm = function( file, zipped=FALSE){
#	file = path.expand( file )
#	.Call("read_grm", file, zipped, package="plink")
#}
