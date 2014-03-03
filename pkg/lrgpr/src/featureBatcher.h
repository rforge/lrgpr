#define MAX(a,b) a>b?a:b
#define MIN(a,b) a<b?a:b

#include <Rcpp.h>

#include <string>
#include <set>
#include <vector>

#include <gsl/gsl_matrix.h>

#include <bigmemory/MatrixAccessor.hpp>

using namespace Rcpp;
using namespace std;

class featureBatcher {

public:
	
	featureBatcher();

	// Constructor based on path to binary big.matrix
	featureBatcher( const string &filename_, long nrow_, long ncol_, long batchSize_, const std::vector<int> & cincl_ = std::vector<int>(0));

	// Decide whether to use data_ NumericMatrix or pBigMat_ big.matrix based on the value of data_
	featureBatcher( SEXP data_, SEXP pBigMat_, const long batchSize_, const std::vector<int> & cincl_ = std::vector<int>(0));

	// generic
	void init( const std::vector<int> & cincl );

	// Initialize based on pointer to big.matrix location
	void init_big_matrix( SEXP pBigMat_, long batchSize_ );

	// Initialize based on text string
	void init_big_matrix( const string &filename_, long nrow_, long ncol_, long batchSize_);
	
	void init_numeric_matrix( SEXP data_ );

	// Destructor
	~featureBatcher();

	// If useBigMatrix
	// 		if conatains_element, read next batch of batchSize features into X_chunk, return true
	// 		if ! conatains_element, advance file pointer, return false
	// 	or do noting if NumericMatrix is used
	bool loadNextChunk();

	// Set marker_j to the jth feature in the current chunk
	void getFeautureInChunk( const long j, gsl_vector *marker_j );

	gsl_matrix *getNextChunk();

	// return X[,idx]
	gsl_matrix *getFeatures( const vector<int> &idx);

	// return true if value is in idxObj
	bool contains_element( const int value );

	// return true if an element in values is in idxObj
	bool contains_element( const std::vector<int> &values );

	// return true if value in interval (start, end) is in idxObj
	bool contains_element( const int start, const int end );


	inline long getBatchSize(){
		return batchSize;
	}

	inline long get_N_features(){
		return n_features;
	}

	inline long get_N_features_included(){
		return idxObj.size();
	}

	inline long get_N_indivs(){
		return n_indivs;
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

	std::set<int> idxObj;

	bool useNumericMatrix, useBigMatrix;
};
