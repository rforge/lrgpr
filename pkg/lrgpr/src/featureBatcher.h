#define MAX(a,b) a>b?a:b
#define MIN(a,b) a<b?a:b

#include <Rcpp.h>

class featureBatcher {

public:
	
	featureBatcher(){ 
		init();
	}

	// Constructor based on path to binary big.matrix
	featureBatcher( const string &filename_, long nrow_, long ncol_, long batchSize_){

		init();

		init_big_matrix(filename_, nrow_, ncol_, batchSize_);
	} 

	// Decide whether to use data_ NumericMatrix or pBigMat_ big.matrix based on the value of data_
	featureBatcher( SEXP data_, SEXP pBigMat_, const long batchSize_) {

		init();

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
	void init(){

		// initialize to false
		useBigMatrix = useNumericMatrix = false;

		X_chunk = NULL;

	}

	// Initialize based on pointer to big.matrix location
	void init_big_matrix( SEXP pBigMat_, long batchSize_ ){

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
	void init_big_matrix( const string &filename_, long nrow_, long ncol_, long batchSize_){
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
	

	void init_numeric_matrix( SEXP data_ ){

		// copy by reference
		X_loop = NumericMatrix( data_ );

		n_features = X_loop.ncol();
		n_indivs = X_loop.nrow();

		batchSize = n_features;

		useNumericMatrix = true;
	}

	~featureBatcher(){
		//cout << "featureBatcher destructor begin..." << endl;
		if( useBigMatrix ){
			fclose(fd);
			gsl_matrix_free(X_chunk);
			free(data);
		}		
		//cout << "featureBatcher destructor end..." << endl;
	}

	// If useBigMatrix, read next batch of batchSize features into X_chunk
	// 	or do noting if NumericMatrix is used
	void loadNextChunk(){

		if( useBigMatrix ){
			//cout << "Load chunk" << endl;
			int res = fread( (void*) X_chunk->data, sizeof(double), batchSize*nrow, fd);	

			// If this is the final batch, set batchSize to the number of features in this final batch
			if( runningTotal + batchSize >= n_features ){

				batchSize = n_features - runningTotal;

				runningTotal = n_features;
			}else{
				runningTotal = runningTotal + batchSize;
			}

		}
	}

	// Set marker_j to the jth feature in the current chunk
	void getFeautureInChunk( const long j, gsl_vector *marker_j ){

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

	inline long getBatchSize(){
		return batchSize;
	}

	inline long get_N_features(){
		return n_features;
	}

	inline long get_N_indivs(){
		return n_indivs;
	}

	gsl_matrix *getNextChunk(){

		int res = fread( (void*) data, sizeof(double), batchSize*nrow, fd);	

		runningTotal += batchSize;

		gsl_matrix *X = gsl_matrix_attach_array( data, batchSize, nrow);

		return X;
	}

	// return X[,idx]
	gsl_matrix *getFeatures( const vector<int> &idx){

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

private:
	string filename;
	FILE *fd;
	double *data;
	long batchSize;
	long nrow, ncol;
	long runningTotal;

	NumericMatrix X_loop;
	gsl_matrix *X_chunk;
	long n_indivs, n_features;

	bool useNumericMatrix, useBigMatrix;
};
