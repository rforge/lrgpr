#ifndef __PLINK__
#define __PLINK__

#include <Rcpp.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <stdio.h>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include "misc_functions.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

using namespace std;
using namespace Rcpp;

/*extern "C" RcppExport SEXP read_tped( SEXP TPED ){
	Rcpp::NumericMatrix xcpp(TPED);
	int nrow = xcpp.nrow();
	int ncol = xcpp.ncol();
	Rcpp::NumericMatrix X(ncol/2,nrow);
	for(int i=0; i<nrow; i++){
		for(int j=0; j<ncol; j=j+2){
			X(j/2,i) = xcpp(i,j) + xcpp(i,j+1) - 2;
		}
	}
	return(X);
}*/

extern "C" RcppExport SEXP write_tped( SEXP X_in, SEXP allele1_, SEXP allele2_, SEXP chromosome_, SEXP genetic_pos_, SEXP position_, SEXP file_in, SEXP markerNames_, SEXP _isTPED){
	
	Rcpp::NumericMatrix X(X_in);
	Rcpp::CharacterVector allele1(allele1_);
	Rcpp::CharacterVector allele2(allele2_);
	Rcpp::IntegerVector chromosome(chromosome_);
	Rcpp::NumericVector genetic_pos(genetic_pos_);
	Rcpp::IntegerVector position(position_);
	string file = Rcpp::as<string>(file_in);
	Rcpp::CharacterVector markerNames( markerNames_ );	
	bool isTPED = (bool) Rcpp::as<long>( _isTPED );

	int nrow = X.nrow();
	int ncol = X.ncol();
	ofstream out(file.c_str());  

	if( isTPED ){
		for(int i=0; i<ncol; i++){
			out << chromosome[i] << " " << markerNames[i] << " " << genetic_pos[i] << " " << position[i];
			for(int j=0; j<nrow; j++){
				if( X(j,i) == -1 ){
					out << " " << allele2[i] << " " << allele2[i] ;
				}else if( X(j,i) == 0 ){
					out << " " << allele1[i] << " " << allele2[i] ;
				}else if( X(j,i) == 1 ){
					out << " " << allele1[i] << " " << allele1[i] ;
				}else{
					out << " 0 0";
				}
			}
			out << endl;
		}
	}else{
		for(int i=0; i<ncol; i++){
			out << markerNames[i] << " " << markerNames[i] << " " << position[i] << " " << allele1[i] << " " << allele2[i];
			for(int j=0; j<nrow; j++){
				if( X(j,i) == -1 ){
					out << " 0 0 1";
				}else if( X(j,i) == 0 ){
					out << " 0 1 0";
				}else if( X(j,i) == 1 ){
					out << " 1 0 0";
				}else{
					out << " 0 0 0";
				}
			}
			out << endl;
		}
	}
	out.close();
	
	Rcpp::NumericVector r(1);
	return r;
}


class SNPinfo {
public:
	vector<string> chromosome;
	vector<string> name;
	vector<double> position_genetic;
	vector<double> position_physical;
	vector<string> allele1, allele2;
	vector<bool> imputed;

	SNPinfo(){}

	~SNPinfo(){}

	void clear(){
		chromosome.clear();
		name.clear();
		position_genetic.clear();
		position_physical.clear();
		allele1.clear();
		allele2.clear();
		imputed.clear();
	}
};

/*
long count_tokens(const char *str_in, const char *delimiters){

	//cout << "str_in: " << str_in << endl;

	// copy str_in
	size_t n = strlen( str_in );

	// need to add one to account for terminating \0 at end of character string
	char *str = (char *) malloc( sizeof(char) * n + 1);

	strcpy(str, str_in);

	// Delimiter is either space or tab
	char *pch = strtok( str, delimiters);

	long count = 0;

	// Tokenize until end is reached
	while ( pch != NULL )	{
		pch = strtok (NULL, delimiters);

		count++;
	}

	free( str );

	return count;
}*/




bool parse_TPED( Rcpp::NumericMatrix *X, const long line_number, const string &line, const char *delimiters, long *n_tokens, SNPinfo *markerInfo, bool *foundMissingData, const vector<string> &markerNames){

	char *pch;

	int n_indivs;
	double value;
	 
	*foundMissingData = false;

	// if the number of tokens has not been determined
	if( *n_tokens == 0 ){

		*n_tokens = count_tokens( line.c_str(), delimiters);

		//cout << "new n_tokens: " << *n_tokens << endl;

		n_indivs = (*n_tokens - 4) / 2;

		// If number of tokens is not even
		// Parsing Error type 1
		if( *n_tokens % 2 != 0 || *n_tokens < 0){
				throw "parsing error 1";
		}
		//cout << "n_tokens: " << n_tokens << endl;
		//cout << "n_indivs: " << n_indivs << endl;
	}else{
		n_indivs = (*n_tokens - 4) / 2;
	}

	// Initialize tokenizer
	// const_cast can be used because the string is not accessed again until a new line is read
	pch = strtok( const_cast<char *>(line.c_str()), delimiters);

	/////////////////////////
	// Get first 4 columns //
	/////////////////////////

	string chrom, name;

	chrom = string( pch );
	pch = strtok (NULL, delimiters);
	
	name = string( pch );
	pch = strtok (NULL, delimiters);

	// If markerNames is empty, then continue
	// or if the name is found in the list, then continue
	if( markerNames.size() == 0 || is_in_vector( markerNames, name) ){
	
		markerInfo->imputed.push_back( false );
		markerInfo->chromosome.push_back( chrom );
		markerInfo->name.push_back( name );	

		markerInfo->position_genetic.push_back( atof( pch ) );
		pch = strtok (NULL, delimiters);

		markerInfo->position_physical.push_back( atof( pch ) );
		
		char *alleleArray = (char *) malloc( *n_tokens * sizeof(char));
		if( alleleArray == NULL){	
			return false;
		}

		///////////////////////////////////
		// Tokenize until end is reached //
		///////////////////////////////////

		// Read single line into alleleArray
		for(int i=0; i<2*n_indivs; i++){		

			pch = strtok (NULL, delimiters);

			// Parsing Error type 2
			if( pch == NULL ){
				return false;
			}
			alleleArray[i] = pch[0]; // Since pch will be a single character
		}

		// if there are remaining tokens on this line
		// Parsing Error type 3
		if( strtok (NULL, delimiters) != NULL ){	
			free( alleleArray );		
			return false;
		}

		// Identify first and second alleles
		char allele1 = '0';
		char allele2 = '0';

		for(int j=0; j<*n_tokens; j++){
			// If first allele has not been found
			if( allele1 == '0' && alleleArray[j] != '0' ){
				allele1 = alleleArray[j];
				continue;
			}
			// If second allele has not been found
			if( allele1 != '0' && allele2 == '0' && alleleArray[j] != '0' && alleleArray[j] != allele1){
				allele2 = alleleArray[j];
				break;
			}
		}

		markerInfo->allele1.push_back( stringify(allele1) );
		markerInfo->allele2.push_back( stringify(allele2) );

		// Assign coding
		int j=0;
		for(int i=0; i<n_indivs; i++){
			// 1 1
			if( alleleArray[j] == allele1 && alleleArray[j+1] == allele1){
				value = 1;

			// 2 2
			}else if( alleleArray[j] == allele2 && alleleArray[j+1] == allele2){
				value = -1;

			// 	either is zero
			}else if( alleleArray[j] == '0' || alleleArray[j+1] == '0'){
				value = NA_REAL;				
				*foundMissingData = true;
			// 1 2	
			}else{
				value = 0;
			}
			(*X)(i,line_number) = value;
			j = j + 2;
		}	

		free( alleleArray );
		
	}else{
		// send pch the end of the current line
		char delim =  '\n';
		pch = strtok (NULL, &delim);
		pch -= 1;
	}

	return true;
}

bool parse_GEN( Rcpp::NumericMatrix *X, const long line_number, const string &line, const char *delimiters, long *n_tokens, SNPinfo *markerInfo, bool *foundMissingData, const vector<string> &markerNames){

	char *pch;

	int n_indivs;
	double value;
	 
	*foundMissingData = false;

	// if the number of tokens has not been determined
	if( *n_tokens == 0 ){

		*n_tokens = count_tokens( line.c_str(), delimiters);

		//cout << "new n_tokens: " << *n_tokens << endl;

		n_indivs = (*n_tokens - 5) / 3;

		// If number of tokens is not even
		// Parsing Error type 1
		/*(if( *n_tokens % 2 != 0 || *n_tokens < 0){
				throw "parsing error 1";
		}*/
		//cout << "n_tokens: " << n_tokens << endl;
		//cout << "n_indivs: " << n_indivs << endl;
	}else{
		n_indivs = (*n_tokens - 5) / 3;
	}

	// Initialize tokenizer
	// const_cast can be used because the string is not accessed again until a new line is read
	pch = strtok( const_cast<char *>(line.c_str()), delimiters);

	/////////////////////////
	// Get first 4 columns //
	/////////////////////////

	string name, chrom;

	chrom = string( pch );
	pch = strtok (NULL, delimiters);
	
	name = string( pch );
	pch = strtok (NULL, delimiters);

	// If markerNames is empty, then continue
	// or if the name is found in the list, then continue
	if( markerNames.size() == 0 || is_in_vector( markerNames, name) ){
	
		if( chrom == "---") markerInfo->imputed.push_back( true );
		else  				markerInfo->imputed.push_back( false );

		markerInfo->chromosome.push_back( stringify(NA_REAL) );
		markerInfo->name.push_back( name );	
		markerInfo->position_genetic.push_back( NA_REAL );

		markerInfo->position_physical.push_back( atof( pch ) );
		pch = strtok (NULL, delimiters);

		markerInfo->allele1.push_back( stringify(pch) );
		pch = strtok (NULL, delimiters);

		markerInfo->allele2.push_back( stringify(pch) );
		
		float *alleleArray = (float *) malloc( 3*n_indivs*sizeof(float));
		if( alleleArray == NULL){
			return false;
		}

		///////////////////////////////////
		// Tokenize until end is reached //
		///////////////////////////////////

		// Read single line into alleleArray
		for(int i=0; i<3*n_indivs; i++){		

			pch = strtok (NULL, delimiters);

			// Parsing Error type 2
			if( pch == NULL ){
				free( alleleArray );
				return false;
			}
			alleleArray[i] = atof(pch);
		}

		// if there are remaining tokens on this line
		// Parsing Error type 3
		if( strtok (NULL, delimiters) != NULL ){
			free( alleleArray );
			return false;
		}

		// Assign coding 1, 0, -1
		int j=0;
		for(int i=0; i<n_indivs; i++){
			(*X)(i,line_number) = 2*alleleArray[j] + alleleArray[j+1] - 1;

			// If entry is 0 0 0 
			if( alleleArray[j] == 0 && alleleArray[j+1] == 0 && alleleArray[j+2] == 0 ){
				(*X)(i,line_number) = NAN;
			}

			j = j + 3;
		}	

		free( alleleArray );
		
	}else{
		// send pch the end of the current line
		char delim =  '\n';
		pch = strtok (NULL, &delim);
		pch -= 1;
	}

	return true;
}


// return true on success, false on failure
bool parse_DOSAGE( Rcpp::NumericMatrix *X, const long line_number, const string &line, const char *delimiters, long *n_tokens, SNPinfo *markerInfo, bool *foundMissingData, const vector<string> &markerNames){

	char *pch;

	int n_indivs;
	double value;
	 
	*foundMissingData = false;

	// if the number of tokens has not been determined
	if( *n_tokens == 0 ){

		*n_tokens = count_tokens( line.c_str(), delimiters);

		n_indivs = (*n_tokens - 4) / 2;

		// If number of tokens is not even
		// Parsing Error type 1
		if( *n_tokens % 2 != 0 || *n_tokens < 0){
				throw "parsing error 1";
		}
		//cout << "n_tokens: " << n_tokens << endl;
		//cout << "n_indivs: " << n_indivs << endl;
	}else{
		n_indivs = (*n_tokens - 4) / 2;
	}

	// Initialize tokenizer
	// const_cast can be used because the string is not accessed again until a new line is read
	pch = strtok( const_cast<char *>(line.c_str()), delimiters);
	if( pch == NULL) return false;

	/////////////////////////
	// Get first 3 columns //
	/////////////////////////

	string name = string( pch );
	markerInfo->name.push_back( name );

	pch = strtok (NULL, delimiters);	
	if( pch == NULL) return false;	
	markerInfo->allele1.push_back( string( pch ) );

	pch = strtok (NULL, delimiters);
	if( pch == NULL) return false;	
	markerInfo->allele2.push_back( string( pch ) );

	// If markerNames is empty, then continue
	// or if the name is found in the list, then continue
	if( markerNames.size() == 0 || is_in_vector( markerNames, name) ){
	
		float *alleleArray = (float *) malloc( n_indivs*sizeof(float));

		///////////////////////////////////
		// Tokenize until end is reached //
		///////////////////////////////////

		int k = 0;
		// Read single line into alleleArray
		// INCREMENT BY 2, because other values reflects quality and is always (?) zero
		for(int i=0; i<=2*n_indivs; i=i+2){		

			pch = strtok (NULL, delimiters);

			//cout << pch << " ";

			// Parsing Error type 2
			if( pch == NULL ){
				free( alleleArray );
				return false;
			}
			alleleArray[k++] = atof(pch); 

			// burn next value
			pch = strtok (NULL, delimiters);
		}

		pch = strtok (NULL, delimiters);

		//cout << "\npch: " << (pch) << endl;

		// if there are remaining tokens on this line
		// Parsing Error type 3
		if( pch != NULL ){
			free( alleleArray );
			return false;
		}

		// Assign values to X
		for(int i=0; i<n_indivs; i++){			
			(*X)(i,line_number) = 2*alleleArray[i] ;
		}

		free( alleleArray );
		
	}else{
		// send pch the end of the current line
		char delim =  '\n';
		pch = strtok (NULL, &delim);
		pch -= 1;
	}

	return true;
}

RcppExport SEXP read_tped_gen( SEXP fileName_in, SEXP start_line_in, SEXP nlines_in, SEXP n_tokens_, SEXP n_lines_total_, SEXP byte_loc_, SEXP markerNames_, SEXP _isTPED){

	string fileName = Rcpp::as<string>( fileName_in );
	int start_line = Rcpp::as<int>( start_line_in );
	long nlines = Rcpp::as<long>( nlines_in );
	long n_tokens = Rcpp::as<long>( n_tokens_ );
	long n_lines_total = Rcpp::as<long>( n_lines_total_ );
	long byte_loc = Rcpp::as<long>( byte_loc_ );
	vector<string> markerNames = Rcpp::as<vector<string> >( markerNames_) ;
	bool isTPED = (bool) Rcpp::as<long>( _isTPED );

	// Since Rcpp::as<vector<string> >( markerNames_) crashes on an empty vector, 
	// 		denote am empty vector bu making the first entry "NULL_place_holder"
	// Thus if markerNames_ is (effectively) empty, set markerNames to be empty
	if( markerNames[0] == "NULL_place_holder"){
		markerNames.clear();
	}

	if( nlines == -1 ) nlines = std::numeric_limits<long>::max(); 

	string line;	

	// If these terms have not been initialized
	if( n_tokens == -1  && n_lines_total == -1){

		ifstream file( fileName.c_str() );

		getline(file, line);
	
		// Count individuals
		n_tokens = count_tokens( line.c_str(), " ");		

		// Initialize count to 1 since the first time was already accessed
		n_lines_total = 1;

		while( getline(file, line) ){
			n_lines_total++; 
		}
		file.close();
	}

	long n_indivs; 

	if( isTPED ) 	n_indivs = ( n_tokens - 4)/2; // TPED 
	else 			n_indivs = ( n_tokens - 5)/3; // GEN  / OXFORD
	
	nlines = MIN( nlines, n_lines_total - start_line );

	// if markerNames is not empty, set the nlines based on it
	// set the byte_loc to the beginning, and start_line to the default (which is then set to 0?)
	if( markerNames.size() > 0 ){
		nlines = markerNames.size();
		byte_loc = 0;
		start_line = -1;
	}

	// If no markers can be read, because the start is beyond the end of the file
	if( nlines <= 0 ){
		List retEmpty;
		return retEmpty;
	}

	Rcpp::NumericMatrix X(n_indivs, nlines);

	bool foundMissingData = false;
	SNPinfo *markerInfo = new SNPinfo();

	ifstream file( fileName.c_str());

	if(  byte_loc == -1 ){

		byte_loc = 0;

		// find requested start line
		long line_number = 0;
		while( getline(file, line) ){
			if( line_number == start_line ) break;
			line_number++;
			byte_loc += line.size() + 1;
		}
	}

	// start reading file at location byte_loc
	file.clear();
	file.seekg( byte_loc, ios::beg); 

	// If markerNames is empty read nlines to to end of file
	if( markerNames.size() == 0 ){
		for(long count=0; count<nlines; count++){
			if( ! getline(file, line) ) break;
			
			if( isTPED ) parse_TPED( &X, count, line, " ", &n_tokens, markerInfo, &foundMissingData, markerNames);
			else 		 parse_GEN ( &X, count, line, " ", &n_tokens, markerInfo, &foundMissingData, markerNames);

			// if the number of markers read into R is the same as the number requested, then break 
			if( markerNames.size() > 0 && markerInfo->name.size() == markerNames.size()) break;

			byte_loc += line.size() + 1;
		}
	}else{
		while( getline(file, line) ){

			if( isTPED )
			 parse_TPED( &X, markerInfo->name.size(), line, " ", &n_tokens, markerInfo, &foundMissingData, markerNames);
			else 		 
			 parse_GEN ( &X, markerInfo->name.size(), line, " ", &n_tokens, markerInfo, &foundMissingData, markerNames);

			// if the number of markers read into R is the same as the number requested, then break 
			if( markerNames.size() > 0 && markerInfo->name.size() == markerNames.size()) break;

			byte_loc += line.size() + 1;
		}
	}

	file.close();

	/*try{
		} catch( std::exception &ex ) {
		forward_exception_to_r( ex );
	} catch(...) { 
		::Rf_error( "c++ exception (unknown reason)" ); 
	}*/

	List ret; 
	ret["X"] 			= X; 
	ret["name"] 		= markerInfo->name;
	ret["chromosome"] 	= markerInfo->chromosome;
	ret["pos_genetic"] 	= markerInfo->position_genetic;
	ret["pos_physical"] = markerInfo->position_physical;
	ret["allele1"] 		= markerInfo->allele1;
	ret["allele2"] 		= markerInfo->allele2;
	ret["imputed"] 		= markerInfo->imputed;
	ret["n_tokens"] 	= n_tokens;
	ret["n_lines_total"] = n_lines_total;
	ret["byte_loc"]		= byte_loc;

	delete markerInfo;

	return ret;
}

RcppExport SEXP set_missing_values( SEXP X_in, SEXP mu_in){
	
	Rcpp::NumericMatrix X(X_in);
	Rcpp::NumericVector mu(mu_in);

	for(int j=0; j<X.ncol(); j++){		
		for(int i=0; i<X.nrow(); i++){
			if( R_IsNA( X(i,j) ) ){
				X(i,j) = mu(j);
			}
		}
	}
	return wrap(0);
}


// Create table vector with 3 entries to of counts for each allele of 1 marker
NumericVector get_allele_table( NumericVector &x ){

	NumericVector res1 = abs(x + 1.0);
	NumericVector res2 = abs(x);
	NumericVector res3 = abs(x - 1.0);

	NumericVector pmin1 = pmin(res2, res3);
	NumericVector pmin2 = pmin(res1, res3);

	NumericVector counts(3);
	for( int j=0; j<x.size(); j++){
		if( res1(j) < pmin1(j) ) counts(0)++;
		else if( res2(j) < pmin2(j) ) counts(1)++;
		else counts(2)++;
	} 
	return( counts );
}

// Create allele table for each marker in genotype matrix using get_allele_table
RcppExport SEXP get_allele_table_multi( SEXP X_ ){

	NumericMatrix X(X_);
	NumericMatrix counts(X.ncol(),3);

	for( int j=0; j<X.ncol(); j++){

		NumericVector xcol = X(_,j);

		counts(j,_) = get_allele_table( xcol );
	} 

	return( wrap( counts) );
}

// if string starts with "SNP A1 A2" return true, else false
bool is_dosage_header( const string &line){

	bool result;

	stringstream ss( line );

	string a,b,c;

	ss >> a;

	// if they are equal
	if( a == "SNP" ){

		ss >> b;
		ss >> c;

		// if both are equal
		if( b == "A1" && c == "A2") result = true;
		else result = false;

	}else result = false;

	return result;
}

RcppExport SEXP R_convertToBinary( SEXP fileName_, SEXP fileNameOut_, SEXP format_, SEXP isZipFile_){

	string fileName = Rcpp::as<string>( fileName_ );
	string fileNameOut = Rcpp::as<string>( fileNameOut_ );
	string format = Rcpp::as<string>( format_ );
	bool isZipFile = (bool) as<int>( isZipFile_ );

	string line;	

	// Get ASCII file dimensions
	////////////////////////////

	// open file
	ifstream file( fileName.c_str() );

	// Boost stream
	boost::iostreams::filtering_istream ifstrm, ifstrm2;

	// set gzip filter to stream
	if( isZipFile ) ifstrm.push(boost::iostreams::gzip_decompressor());
       
    // run file through the filter stream   
    ifstrm.push(file);

	bool res = std::getline(ifstrm, line);

	if( ! res ){
		return( wrap(1) );
	}

	// Count individuals
	long n_tokens = count_tokens( line.c_str(), " ");	

	//cout << "n_tokens: " << n_tokens << endl;

	// Initialize count to 1 since the first time was already accessed
	long n_lines = 0;

	do{
		n_lines++; 

		// if is identifier line
		if( format == "DOSAGE" && is_dosage_header(line) ){
			n_lines--;
		}

		if( n_lines % 10000 == 0 )
			Rcpp::Rcout <<"\rReading line " << n_lines << "             " << flush;
		
	}while( getline(ifstrm, line) );

	Rcpp::Rcout <<"\rReading line " << n_lines << endl;

	long n_indivs; 

	if( format == "TPED" ) 			n_indivs = ( n_tokens - 4)/2; // TPED 
	else if( format == "GEN" )		n_indivs = ( n_tokens - 5)/3; // GEN  / OXFORD
	else if( format == "DOSAGE" )	n_indivs = ( n_tokens - 3)/2; // dosage,gz  / MACH
	
	Rcpp::Rcout << "n_indivs: " << n_indivs << endl;

	//cout << "malloc: " << n_indivs * n_lines * sizeof(double) / (double) 1e9 << endl;

	Rcpp::Rcout << "Write file..." << endl;

	// Set to beginning of file
	//file.seekg(0, ios_base::beg);

	file.close();
	file.open( fileName.c_str() );

	if( isZipFile ) ifstrm2.push(boost::iostreams::gzip_decompressor());
       
    // run file through the filter stream   
    ifstrm2.push(file);

	int byte_loc = 0;

	Rcpp::NumericMatrix X(n_indivs, 1);

	bool foundMissingData = false;
	SNPinfo *markerInfo = new SNPinfo();

	vector<string> markerNames;

	double *array = (double *) malloc(n_indivs*sizeof(double));

	FILE *fp = fopen( fileNameOut.c_str(), "wb");

	// int ftruncate(int fd, off_t length);
	int result = ftruncate( fileno(fp), n_indivs*n_lines*sizeof(double) );

	time_t start_time;
	time(&start_time);

	bool success = true;

	long count=0;

	while( getline(ifstrm2, line) && success ){	

		if( format == "DOSAGE" && is_dosage_header(line) ) continue;

		//for(int k=0; k<40; k++) cout << line[k];
		//cout << endl;
		
		if( format == "TPED" ){
			
			success = parse_TPED( &X, 0, line, " \t", &n_tokens, markerInfo, &foundMissingData, markerNames);

			// Note: parse_TPED returns -1, 0, 1
			//		so add 1 to concert to 0,1,2 
			X(_, 0) = X(_, 0) + 1;

		}else if( format == "GEN" ){

			success = parse_GEN ( &X, 0, line, " \t", &n_tokens, markerInfo, &foundMissingData, markerNames);


			// Note: parse_GEN returns -1, 0, 1
			//		so add 1 to concert to 0,1,2 
			X(_, 0) = X(_, 0) + 1;
		
		}else if( format == "DOSAGE" ){

			success = parse_DOSAGE ( &X, 0, line, " \t", &n_tokens, markerInfo, &foundMissingData, markerNames);
		}

		// If line was parsed sucessfully
		if( success ){

			//  Convert to array 
			for(int i=0; i<n_indivs; i++){
				array[i] = X(i, 0);
			}

			// int fseek(FILE *stream, long offset, int whence);
			fseek( fp, count*n_indivs*sizeof(double), 0);

			// size_t fwrite(const void *ptr, size_t size_of_elements, size_t number_of_elements, FILE *a_file);
			fwrite( array, sizeof(double), n_indivs, fp);

			if( count % 1000 == 0 ) 
				print_progress( count, n_lines, 25, start_time);

			count++;
		}
	}
	
	if( success ){ 
		Rcpp::Rcout << print_progress( n_lines, n_lines, 25, start_time);
		Rcpp::Rcout << endl;
	}

	file.close();
	fclose(fp);
	free( array );

	Rcpp::List ret = Rcpp::List::create(				
				Rcpp::Named("success")  	= success, 
				Rcpp::Named("nrow")  		= n_indivs, 
				Rcpp::Named("ncol")  		= n_lines, 
				Rcpp::Named("colNames") 	= wrap(markerInfo->name) );

	delete markerInfo;

	return( ret );
}


RcppExport SEXP R_readBinary( SEXP fileName_, SEXP N_){

	string fileName = Rcpp::as<string>( fileName_ );
	long N = Rcpp::as<long>( N_ );

	double *array = (double *) malloc(N*sizeof(double));

	FILE *fp = fopen( fileName.c_str(), "rb");

	// size_t fread(void *ptr, size_t size_of_elements, size_t number_of_elements, FILE *a_file);
	int res = fread( array, sizeof(double), N, fp);

	fclose(fp);

	Rcpp::NumericVector ret(N);

	for(int i=0; i<N; i++){
		ret[i] = array[i];
	}
	return( wrap(ret) );
}

/*
RcppExport SEXP read_grm( SEXP sfileName, SEXP szipped ){

	string fileName = Rcpp::as<string>(sfileName);
	bool zipped = (Rcpp::as<int>(szipped) != 0);

	igzstream file( fileName.c_str() );

	string line;	
	
	int n, n_lines = 1;

	while( file >> line ){ n_lines++; }
	file.close();

	n_lines = n_lines/4.0;
	//cout << "n_lines: " << n_lines << endl;
	
	int count = n_lines;
	int i=0;

	while( count > 0 ){
		count -= i;
		i++;
	}
	if( count < 0){ 
		cout << "Read GRM error" << endl;
	} 

	int n_indivs = i-1;	

	//cout << "count: " << count << endl;
	cout << "n_indivs: " << n_indivs << endl;
	
	Rcpp::NumericMatrix K(n_indivs, n_indivs);

	int idx1, idx2;
	double value;

	igzstream file2( fileName.c_str() );
	for( int j=0; j<n_lines; j++){
		
		file2 >> idx1;
		file2 >> idx2;
		file2 >> value;
		file2 >> value;
		
		// subtract 1 to convert from R to C indexing
		K( idx1-1, idx2-1 ) = value;
		K( idx2-1, idx1-1 ) = value;
	}
	file2.close(); 

	return K;	
}*/



#endif