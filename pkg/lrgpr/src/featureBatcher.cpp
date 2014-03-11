#define MAX(a,b) a>b?a:b
#define MIN(a,b) a<b?a:b

#include "featureBatcher.h"

#include <exception>

#include "misc_functions.h"
#include "gsl_additions.h"

featureBatcher::featureBatcher(){ 
	init( std::vector<int>(0) );
}

// Constructor based on path to binary big.matrix
featureBatcher::featureBatcher( const string &filename_, long nrow_, long ncol_, long batchSize_, const std::vector<int> & cincl_){

	init( cincl_ );

	init_big_matrix(filename_, nrow_, ncol_, batchSize_);
} 

// Decide whether to use data_ NumericMatrix or pBigMat_ big.matrix based on the value of data_
featureBatcher::featureBatcher( SEXP data_, SEXP pBigMat_, const long batchSize_, const std::vector<int> & cincl_ ) {

	init( cincl_ );

	bool standardMatrix;

	// if matrix
	if( TYPEOF(data_) == REALSXP || TYPEOF(data_) == INTSXP ){
		standardMatrix = true;

	// If big.matrix	
	}else if( TYPEOF(data_) == S4SXP ){	
		standardMatrix = false;
	}else{		
		throw invalid_argument("Invalid type for features");
	}

	if( standardMatrix)	init_numeric_matrix( data_ );
	else				init_big_matrix( pBigMat_, batchSize_ );
}

// generic
void featureBatcher::init( const std::vector<int> & cincl ){

	// initialize to false
	useBigMatrix = useNumericMatrix = false;

	X_chunk = NULL;

	// Insert elements into unordered_set
	for(unsigned int i=0; i<cincl.size(); i++){
		idxObj.insert( cincl[i] );
	}

}

// Initialize based on pointer to big.matrix location
void featureBatcher::init_big_matrix( SEXP pBigMat_, long batchSize_ ){

	// Convert external pointer to BigMatrix
	BigMatrix *pBigMat = Rcpp::XPtr<BigMatrix>( pBigMat_ );

	X_loop = NumericMatrix( pBigMat->nrow(), 1 );

	n_features = pBigMat->ncol();
	n_indivs = pBigMat->nrow();

	FileBackedBigMatrix *pfbbm = (FileBackedBigMatrix*) pBigMat;

	string filename = pfbbm->file_name(); 
	string filepath = pfbbm->file_path();

	filename = filepath + "/" + filename;

	init_big_matrix( filename, n_indivs, n_features, batchSize_);

	useBigMatrix = true;
}

// Initialize based on text string
void featureBatcher::init_big_matrix( const string &filename_, long nrow_, long ncol_, long batchSize_){
	filename 	= filename_;
	nrow 		= nrow_;
	ncol 		= ncol_;
	batchSize 	= batchSize_;

	/*cout << "filename: " << filename << endl;
	cout << "nrow: " << nrow << endl;
	cout << "ncol: " << ncol << endl;
	cout << "batchSize: " << batchSize << endl;*/

	fd = fopen(filename.c_str(), "rb");

	if( fd == NULL ){
		throw runtime_error( "File failure: " + filename );
	}

	data = (double *) malloc(batchSize*nrow*sizeof(double)); 
	X_chunk = gsl_matrix_alloc( batchSize, nrow );

	//cout << "X_chunk: " << X_chunk->size1 << " x " << X_chunk->size2 << endl;

	// count the number of features read
	runningTotal = 0;
} 


void featureBatcher::init_numeric_matrix( SEXP data_ ){

	// copy by reference
	X_loop = NumericMatrix( data_ );

	n_features = X_loop.ncol();
	n_indivs = X_loop.nrow();

	batchSize = n_features;

	useNumericMatrix = true;
}

featureBatcher::~featureBatcher(){
	//cout << "featureBatcher destructor begin..." << endl;
	if( useBigMatrix ){
		fclose(fd);
		gsl_matrix_free(X_chunk);
		free(data);
	}		
	//cout << "featureBatcher destructor end..." << endl;
}

// If useBigMatrix
// 		if conatains_element, read next batch of batchSize features into X_chunk, return true
// 		if ! conatains_element, advance file pointer, return false
// 	or do noting if NumericMatrix is used
bool featureBatcher::loadNextChunk(){

	// Set to true so that then analysis is run if !useBigMatrix
	bool useThisChunk = true;

	if( useBigMatrix ){

		//cout << endl;
		//cout << "start: " << runningTotal << endl;
		//cout << "end: " << runningTotal + batchSize - 1 << endl << endl;

		// If include list is empty, or index is in include list
		if( idxObj.size() == 0 || contains_element( runningTotal, runningTotal + batchSize - 1  ) ){

			//cout << "Contains..." <<endl;
			// Read data and advance file pointer, fd
			int res = fread( (void*) X_chunk->data, sizeof(double), batchSize*nrow, fd);	

			useThisChunk = true;
		}else{
			// Just advance file pointer, fd
			fseek( fd, batchSize*nrow*sizeof(double), SEEK_CUR );

			useThisChunk = false;
		}

		// If this is the final batch, set batchSize to the number of features in this final batch
		if( runningTotal + batchSize >= n_features ){

			batchSize = n_features - runningTotal;

			runningTotal = n_features;
		}else{
			runningTotal += batchSize;
		}
	}

	return useThisChunk;
}

// Set marker_j to the jth feature in the current chunk
void featureBatcher::getFeautureInChunk( const long j, gsl_vector *marker_j ){

	if( j > batchSize ){
		throw range_error( "j > batchSize; j = " + stringify(j) + ", j = " + stringify(batchSize));
	}

	if( useNumericMatrix ){
		for(int h=0; h<marker_j->size; h++){
			gsl_vector_set(marker_j, h, X_loop(h,j));
		}	
	}else{		
		for(int h=0; h<marker_j->size; h++){
			//cout << "\rj: " << j << " h: " << h << flush;
			gsl_vector_set(marker_j, h, gsl_matrix_get(X_chunk, j, h) );
		}							
	}			
}

/*
gsl_matrix *featureBatcher::getNextChunk(){

	int res = fread( (void*) data, sizeof(double), batchSize*nrow, fd);	

	runningTotal += batchSize;

	gsl_matrix *X = gsl_matrix_attach_array( data, batchSize, nrow);

	return X;
}*/

// return X[,idx]
gsl_matrix *featureBatcher::getFeatures( const vector<int> &idx){

	if( idx.size() == 0 ) return NULL;  

	gsl_matrix *X;

	if( useBigMatrix){

		double *X_idx = (double *) malloc(idx.size()*nrow*sizeof(double)); 

		FILE *fd_tmp = fopen(filename.c_str(), "rb");

		int res;

		for(int i=0; i<idx.size(); i++){

			// set file pointer to element idx[i]*nrow
			res = fseek( fd_tmp, idx[i]*nrow*sizeof(double), SEEK_SET );

			if( res != 0 ) throw runtime_error("fseek error");

			// set X_idx pointer to element idx[i]*nrow
			res = fread( (void*) (X_idx + i*nrow), sizeof(double), nrow, fd_tmp);

			if( res != nrow ) throw runtime_error("fread error");	
		}
		
		fclose(fd_tmp);

		gsl_matrix *t_X = gsl_matrix_attach_array( X_idx, idx.size(), nrow);

		X = gsl_matrix_alloc( t_X->size2, t_X->size1 );
		gsl_matrix_transpose_memcpy( X, t_X);

		gsl_matrix_free( t_X );
		free(X_idx);

		//gsl_matrix_print(X);	
	}else{
		X = gsl_matrix_alloc( nrow, idx.size());

		for(int a=0; a<nrow; a++){
			for(int b=0; b<idx.size(); b++){
				gsl_matrix_set(X, a, b, X_loop(a,idx[b]));
			}
		}
	}

	return X;
}

// return true if value is in idxObj
bool featureBatcher::contains_element( const int value ){
	return ( idxObj.count(value) != 0);
}

// return true if an element in values is in idxObj
bool featureBatcher::contains_element( const std::vector<int> &values ){

	bool found = false;

	for(unsigned int i=0; i<values.size(); i++){
		if( idxObj.count(values[i]) != 0){
			found = true;
			break;
		}
	}

	return found;
}

// return true if value in interval (start, end) is in idxObj
bool featureBatcher::contains_element( const int start, const int end ){

	bool found = false;

	for(unsigned int i=start; i<=end; i++){
		if( idxObj.count(i) != 0){
			found = true;
			break;
		}
	}

	return found;
}

