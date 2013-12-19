
#define MAX(a,b) a>b?a:b
#define MIN(a,b) a<b?a:b
#include <RcppGSL.h>
#include <Rcpp.h>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cdf.h>

#include <omp.h>

#include <iostream>
#include <string>
#include <math.h>

#include <bigmemory/MatrixAccessor.hpp>

using namespace std;
using namespace Rcpp;

#include "lrgpr.h"
#include "gsl_additions.h"
#include "gsl_lapack.h"
#include "GLM.h"
#include "misc_functions.h"


#ifdef INTEL
#include "mkl_service.h"
#endif


class featureSet {

public:
	
	featureSet(){ initialized = false;}

	featureSet( const string &filename_, long nrow_, long ncol_, long setSize_){
		init(filename_, nrow_, ncol_, setSize_);
	} 

	void init( const string &filename_, long nrow_, long ncol_, long setSize_){
		filename 	= filename_;
		nrow 		= nrow_;
		ncol 		= ncol_;
		setSize 	= setSize_;

		cout << "filename: " << filename << endl;
		/*cout << "nrow: " << nrow << endl;
		cout << "ncol: " << ncol << endl;
		cout << "setSize: " << setSize << endl;*/

		fd = fopen(filename.c_str(), "rb");

		if( fd == NULL ){
			cerr << "File failure: " << filename << endl << endl;
		}

		data = (double *) malloc(setSize*nrow*sizeof(double)); 

		// count the number of features read
		runningTotal = 0;

		initialized = true;
	} 

	~featureSet(){
		if( initialized ){
			fclose(fd);
			free(data);
		}
	}

	gsl_matrix *getNextChunk(){

		if( ! initialized ) return NULL;

		int res = fread( (void*) data, sizeof(double), setSize*nrow, fd);	

		runningTotal += setSize;

		gsl_matrix *X = gsl_matrix_attach_array(  data, setSize, nrow);

		/*gsl_matrix *X = gsl_matrix_attach_array(  data, setSize, nrow);
		gsl_matrix_save(X, "X_set.txt");


		gsl_matrix *X2 = gsl_matrix_attach_array(  data, nrow, setSize );
		gsl_matrix_save(X2, "X_set2.txt");

		exit(1);*/


		return X;
	}

private:
	string filename;
	FILE *fd;
	double *data;
	long setSize;
	long nrow, ncol;
	long runningTotal;
	bool initialized;
};


static Rcpp::Function asMatrix("as.matrix");

class Rexpress {

	private:
	string expression, query;
	Environment env;
	NumericMatrix X_data;
	NumericVector One;
	Formula form;

	public:
	
	Rexpress( const string &expression_, const NumericMatrix &X_data_, Environment &env_){
		
		expression 	= expression_;
		X_data 		= X_data_;
		env 		= env_;
		query = "SNP";
		form = Formula(expression);

		// One = rep(1, nrow(X))
		One = NumericVector( X_data.nrow() );
		std::fill(One.begin(), One.end(), 1);
	}

	RcppGSL::matrix<double> get_model_matrix( const int j ){

		// SNP = X[,j]
		env["SNP"] = X_data(_,j);

		Language call( "model.matrix.default", form);	
		return call.eval( env );
	}

	RcppGSL::matrix<double> get_model_matrix_clean(){

		env["SNP"] = One;

		Language call( "model.matrix.default", form);	
		return call.eval( env );
	}

	// Get the matrix for the null model
	RcppGSL::matrix<double> get_model_matrix_null(){

		// replace query SNP with nothing
		Language call1( ".mm_replace_query", form, query);	
		string expressionLocal = as<string>( call1.eval( env ) );

		// evaluate formula
		Formula form2( expressionLocal );
		Language call2( "model.matrix.default", form2);	
		return call2.eval( env );
	}

	RcppGSL::vector<double> get_response(){	

		env["SNP"] = One;

		Language call( ".mm_get_response", form);
		return call.eval( env ); 
	}

	RcppGSL::matrix<double> get_response_m(){
		
		env["SNP"] = One;
		
		Language call( ".mm_get_response", form);
		return call.eval( env ); 
	}

	vector<int> get_loop_terms(){		

		Language call( ".mm_get_terms", form, query);
		NumericVector loopIndex = call.eval( env ); 

		return as<vector<int> >(loopIndex);
	}

	string get_expression(){
		return expression;
	}

};


RcppExport SEXP R_lrgpr( SEXP Y_, SEXP X_, SEXP eigenVectors_, SEXP eigenValues_, SEXP delta_, SEXP nthreads_, SEXP Wtilde_){

	try {
	
		RcppGSL::matrix<double> X = X_; 
		RcppGSL::vector<double> y = Y_; 	
		RcppGSL::matrix<double> eigenVectors = eigenVectors_; 	
		RcppGSL::vector<double> eigenValues = eigenValues_; 
		RcppGSL::matrix<double> Wtilde = Wtilde_; 

		// If # of samples in W and y is the same
		bool useProxCon = ( Wtilde->size1 == y->size );
		
		double delta = Rcpp::as<double>( delta_ );
		double nthreads = Rcpp::as<double>( nthreads_ );

		if( ! R_IsNA( nthreads ) ){
			omp_set_num_threads( 1 );
			// Intel paralellism
			#ifdef INTEL
			//mkl_set_num_threads( nthreads );	
			mkl_set_num_threads( 1 );			
			#endif
			// disable nested OpenMP parallelism
			omp_set_nested(0);
		}
		
		// Make sure all eigen values are non-negative
		for(unsigned int i=0; i<eigenValues->size; i++){
			if( gsl_vector_get( eigenValues, i) < 0 )  gsl_vector_set( eigenValues, i, 0);
		}

		LRGPR *lrgpr = new LRGPR( y, eigenVectors, eigenValues, X->size2, useProxCon ? Wtilde->size2 : 0);

		// Must update W before X
		if( useProxCon ) lrgpr->update_Wtilde( Wtilde );

		// set missing to mean
		gsl_vector_view col_view;

		for( unsigned int j=0; j<X->size2; j++){
			col_view = gsl_matrix_column( (gsl_matrix*)(X), j );
			gsl_vector_set_missing_mean( &col_view.vector );
		}		

		lrgpr->update_X( X );

		double log_L, sig_g, sig_e;

		if( isnan( delta ) ){
			// Estimate delta
			lrgpr->fit_mle( &log_L, &sig_g, &sig_e);

			delta = sig_e / sig_g;
		}else{
			// Use pre-specified delta value
			lrgpr->fit_fixed_delta( delta, &log_L, &sig_g, &sig_e );
		}

		RcppGSL::vector<double> alpha = gsl_vector_alloc( y->size );

		RcppGSL::vector<double> Hii = lrgpr->get_hat_matrix_diag();
		RcppGSL::vector<double> Y_hat = lrgpr->get_fitted_response( alpha );
		RcppGSL::matrix<double> Sigma = lrgpr->coeff_covariance();
		RcppGSL::vector<double> pValues = lrgpr->wald_test_all();
		RcppGSL::vector<double> beta = lrgpr->get_beta();		

		Rcpp::List res = Rcpp::List::create(Rcpp::Named("coefficients") = beta,
											Rcpp::Named("p.values") 	= pValues,
											Rcpp::Named("sigSq_e") 		= sig_e,
											Rcpp::Named("sigSq_a") 		= sig_g,
											Rcpp::Named("delta") 		= delta, 
											Rcpp::Named("rank") 		= eigenValues->size,
											Rcpp::Named("logLik")		= log_L,
											Rcpp::Named("fitted.values")= Y_hat,
											Rcpp::Named("alpha")		= alpha,
											Rcpp::Named("Sigma")		= Sigma,			
											Rcpp::Named("hii")			= Hii  );
		
		X.free(); 
		y.free();
		eigenVectors.free();
		eigenValues.free();
		beta.free();
		Wtilde.free();

		delete lrgpr;

		return res; // return the result list to R

	} catch( std::exception &ex ) {
		forward_exception_to_r( ex );
	} catch(...) { 
		::Rf_error( "c++ exception (unknown reason)" ); 
	}
	return R_NilValue; // -Wall
}

RcppExport SEXP R_glmApply( SEXP expression_, SEXP data_, SEXP pBigMat_, SEXP env_, SEXP terms_, SEXP nthreads_, SEXP useMean_, SEXP useIdentityLink_, SEXP univariate_, SEXP multivariate_){

	try {
		CharacterVector expression( expression_ );
		Environment env( env_ );
		std::vector<int> terms = as<std::vector<int> >( terms_ ); 
		int nthreads = as<int>( nthreads_ );
		bool useMean = (bool) as<int>( useMean_ );
		bool useIdentityLink = as<int>( useIdentityLink_ );
		bool univariate = as<int>( univariate_ );
		bool multivariate = as<int>( multivariate_ );
			
		regressionType regressType = useIdentityLink ? LINEAR : LOGISTIC;

		bool standardMatrix;

		// if matrix
		if( TYPEOF(data_) == REALSXP || TYPEOF(data_) == INTSXP ){
			standardMatrix = true;

		// If big.matrix	
		}else if( TYPEOF(data_) == S4SXP ){	
			standardMatrix = false;
		}else{		
			throw "Invalid type for features";
		}

		// Import design matrix
		NumericMatrix X_loop;
		featureSet fset;

		long n_markers;

		long setSize = 10000;

		if( standardMatrix){

			// copy by reference
			X_loop = NumericMatrix( data_ );

			n_markers = X_loop.ncol();

			setSize = n_markers;

		}else{	

			// Convert external pointer to BigMatrix
			BigMatrix *pBigMat = Rcpp::XPtr<BigMatrix>( pBigMat_ );

			X_loop = NumericMatrix( pBigMat->nrow(), 1 );

			n_markers = pBigMat->ncol();

			FileBackedBigMatrix *pfbbm = (FileBackedBigMatrix*) pBigMat;
 			string filename = pfbbm->file_name(); 

 			string filepath = pfbbm->file_path();
			filename = filepath + "/" + filename;
			cout << "File: " << filename << endl;

			fset.init( filename, pBigMat->nrow(), pBigMat->ncol(), setSize);
		}

		// Set threads to 1
		omp_set_num_threads( nthreads );
		// Intel paralellism
		#ifdef INTEL
		mkl_set_num_threads( 1 );
		#endif
		// disable nested OpenMP parallelism
		omp_set_nested(0);

		// Process exression, X_loop and env
		Rexpress expr( as<string>( expression ), X_loop, env );			

		RcppGSL::matrix<double> Y = expr.get_response_m(); 

		int n_indivs = Y->size1;
		int n_pheno = Y->size2;

		if( n_pheno > 1 && regressType == LOGISTIC ){
			Y.free();
		
			throw "Cannot do multivariate logistic regression\n";
		}

		RcppGSL::matrix<double> Xn = expr.get_model_matrix_null();

		if( n_pheno == 1 ) multivariate = false;

		// Check that sizes of and X_loop match
		if( n_indivs != X_loop.nrow() ){
			Y.free();
			Xn.free();
		
			throw "Dimensions of response and design matrix do not match\n";
		}
		
		NumericMatrix pValues;
		NumericMatrix pValues_multivariate;

		// Define p-values matrix to be returned, only of the corresponding test is performed
		if( univariate )  	pValues = NumericMatrix(n_markers, n_pheno );
		if( multivariate )  pValues_multivariate = NumericMatrix(n_markers, 2 );

		// X_clean = model.matrix.default( y ~ sex:One + age )
		// Replace marker with 1's so that design matrix for marker j can be created by multiplcation
		RcppGSL::matrix<double> X_clean = expr.get_model_matrix_clean(); 

		// If trying to access index beyond range
		// Exit gracefully
		if( terms[which_max(terms)] >= X_clean->size2 ){
			Y.free();
			Xn.free();
			X_clean.free();
		
			throw "Element in \"terms\" is out of range";
		}

		// get indeces of columns that depend on X[,j]
		vector<int> loopIndex = expr.get_loop_terms();
		//vector<string> colNames = expr.get_terms();		

		//cout << "loopIndex: " << endl;
		//print_vector(loopIndex);

		//n_markers = MIN(n_markers, 100000);

		//cout << "n_indivs: " << n_indivs << endl;
		//cout << "n_markers: " << n_markers << endl;
		
		long batch_size = MAX( 1, setSize/100.0 );

		long tests_completed = 0;

		time_t start_time;
		time(&start_time);

		for(int i_set=0; i_set<n_markers; i_set+=setSize){

			gsl_matrix *X_set = fset.getNextChunk();

			#pragma omp parallel
			{
				// Variables local to each thread
				gsl_matrix *X = gsl_matrix_alloc( X_clean->size1, X_clean->size2 );

				gsl_vector_view col_view, col_view_y;

				gsl_vector *marker_j = gsl_vector_alloc( n_indivs );

				GLM_MV_workspace *workmv;
				GLM_IRLS_workspace *irls_workspace;

				gsl_vector *y, *beta;

				double sig_e, log_L;
				int rank;

				if( regressType == LINEAR ){
					workmv = GLM_MV_workspace_alloc(Y->size1, X->size2, Y->size2, true, false);
				}else{
					y = gsl_vector_attach_array( Y->data, Y->size1 );
					beta = gsl_vector_calloc( X->size2 );

					irls_workspace = GLM_IRLS_workspace_alloc( X->size1, X->size2, true);
				}

				double Hotelling, Pillai;

				bool isMissing;

				#pragma omp for schedule(static, batch_size)
				for(int j=0; j<setSize; j++){		

					// if index exceeds n_markers, continue to next interation
					if(  j+i_set >= n_markers ) continue;

					#pragma omp critical
					tests_completed++;

					// Copy feature data from X_loop to marker_j in a way that does not use
					// 	Rcpp or RcppGSL objects.  This avoid the Rcpp thread-safety issues.
					// In summary, Rcpp objects, even if they declared in their own thread, 
					// 	CANNOT be written to safely
					// Here I do the copying manually, instead of using the nice Rcpp methods
					if( standardMatrix ){
						for(int h=0; h<marker_j->size; h++){
							gsl_vector_set(marker_j, h, X_loop(h,j));
						}	
					}else{		
						for(int h=0; h<marker_j->size; h++){
							gsl_vector_set(marker_j, h, gsl_matrix_get(X_set, j, h) );
						}							
					}			

					if( useMean )
						isMissing = gsl_vector_set_missing_mean( marker_j );					
					else 
						isMissing = any_is_na( marker_j->data, marker_j->size );

					// Check if X has an NA value 
					// If it does, set the p-value to 
					if( isMissing && ! useMean){
						if( univariate ){
							#pragma omp critical
							for(int k=0; k<n_pheno; k++){
								pValues(j+i_set,k) = NA_REAL;
							}
						}
						#pragma omp critical
						if( multivariate ){  
							pValues_multivariate(j+i_set,0) = NA_REAL;
							pValues_multivariate(j+i_set,1) = NA_REAL;
						}
						continue;
					}

					// Copy clean version of matrices to active variables
					gsl_matrix_memcpy( X, X_clean );

					for(int k=0; k<loopIndex.size(); k++){			
						// X_design[,loopIndex[k]] = X_design[,loopIndex[k]] * marker_j
						col_view = gsl_matrix_column( (gsl_matrix*)(X), loopIndex[k] );
						gsl_vector_mul( &col_view.vector, marker_j );
					}		

					// Evaluate regression
					if( regressType == LINEAR ){
						GLM_regression( Y, X, regressType, workmv);

						if( multivariate ){
							GLM_HotellingPillai_test( X, workmv, terms, &Hotelling, &Pillai );

							#pragma omp critical
							{
								pValues_multivariate(j+i_set,0) = Hotelling;
								pValues_multivariate(j+i_set,1) = Pillai;
							}							
						}

						if( univariate ){
							// Perform Wald test
							gsl_vector *pvals = GLM_wald_test( workmv, terms ) ; 

							#pragma omp critical
			 				for(int k=0; k<n_pheno; k++){
								pValues(j+i_set,k) = gsl_vector_get(pvals, k);
							}
							gsl_vector_free( pvals );
						}

					}else{						

						GLM_unpenalized_regression_IRLS(y, X, beta, &sig_e, &log_L, &rank, regressType, false, irls_workspace);

						#pragma omp critical
						pValues(j+i_set,0) = GLM_wald_test( beta, irls_workspace->Sigma, X->size1, terms, regressType, irls_workspace->design_is_singular);
					}

				} // END for
							
				gsl_matrix_free( X );
				gsl_vector_free( marker_j );

				if( regressType == LINEAR ){
					GLM_MV_workspace_free( workmv );
				}else{
					GLM_IRLS_workspace_free( irls_workspace );

					gsl_vector_free(y);
					gsl_vector_free(beta);
				}

			} // End parallel

			print_progress( tests_completed, n_markers, 25, start_time);

			// X_set is NULL if fset is empty
			if( X_set != NULL ) gsl_matrix_free( X_set );
		} // End set loop

		cout << endl;

		Y.free();
		Xn.free();
		X_clean.free();

		Rcpp::List res = Rcpp::List::create(Rcpp::Named("pValues") 	= pValues,
											Rcpp::Named("pValues_mv") = pValues_multivariate);

		return( res );

	} catch( const char* msg ){
		 Rcpp::Rcout << "glm error: " << msg << endl;
	}catch( std::exception &ex ) {
		Rcpp::Rcout << "exception:  " << endl;
		forward_exception_to_r( ex );
	} catch(...) { 
		::Rf_error( "c++ exception (unknown reason)" ); 
	}
	return R_NilValue; // -Wall
}

RcppExport SEXP R_lrgprApply( SEXP expression_, SEXP data_, SEXP pBigMat_, SEXP env_, SEXP terms_, SEXP EigenVectors_, SEXP EigenValues_, SEXP Wtilde_, SEXP rank_, SEXP delta_, SEXP reEstimateDelta_, SEXP nthreads_){

	try {
		CharacterVector expression( expression_ );
		Environment env( env_ );
		std::vector<int> terms = as<std::vector<int> >( terms_ ); 
		RcppGSL::matrix<double> eigenVectors = EigenVectors_;
		RcppGSL::vector<double> eigenValues = EigenValues_; 
		RcppGSL::matrix<double> Wtilde = Wtilde_; 
		int rank = as<int>( rank_ );
		double delta_global = as<double>( delta_ );
		bool reEstimateDelta = Rcpp::as<int>( reEstimateDelta_ );
		int nthreads = as<int>( nthreads_ );

		// Make sure all eigen values are non-negative
		for(unsigned int i=0; i<eigenValues->size; i++){
			if( gsl_vector_get( eigenValues, i) < 0 )  gsl_vector_set( eigenValues, i, 0);
		}

		bool standardMatrix;

		// if matrix
		if( TYPEOF(data_) == REALSXP || TYPEOF(data_) == INTSXP ){
			standardMatrix = true;

		// If big.matrix	
		}else if( TYPEOF(data_) == S4SXP ){	
			standardMatrix = false;
		}else{		
			throw "Invalid type for features";
		}

		// Import design matrix
		NumericMatrix X_loop;
		featureSet fset;

		long n_markers;

		long setSize = 10000;

		if( standardMatrix){

			// copy by reference
			X_loop = NumericMatrix( data_ );

			n_markers = X_loop.ncol();

			setSize = n_markers;

		}else{	

			// Convert external pointer to BigMatrix
			BigMatrix *pBigMat = Rcpp::XPtr<BigMatrix>( pBigMat_ );

			X_loop = NumericMatrix( pBigMat->nrow(), 1 );

			n_markers = pBigMat->ncol();

			FileBackedBigMatrix *pfbbm = (FileBackedBigMatrix*) pBigMat;
 			string filename = pfbbm->file_name(); 	

 			string filepath = pfbbm->file_path();
			filename = filepath + "/" + filename;

			fset.init( filename, pBigMat->nrow(), pBigMat->ncol(), setSize);
		}

		// Set threads to 1
		omp_set_num_threads( nthreads );
		// Intel paralellism
		#ifdef INTEL
		mkl_set_num_threads( 1 );
		#endif
		// disable nested OpenMP parallelism
		omp_set_nested(0);

		// Process exression, X_loop and env
		Rexpress expr( as<string>( expression ), X_loop, env );			

		RcppGSL::vector<double> y = expr.get_response(); 

		int n_indivs = y->size;

		// Check that sizes of and X_loop match
		if( n_indivs != X_loop.nrow() ){
			y.free();
		
			throw "Dimensions of response and design matrix do not match\n";
		}

		// If # of samples in W and y is the same
		bool useProxCon = ( Wtilde->size1 == y->size );

		// X_clean = model.matrix.default( y ~ sex:One + age )
		// Replace marker with 1's so that design matrix for marker j can be created by multiplcation
		RcppGSL::matrix<double> X_clean = expr.get_model_matrix_clean(); 

		// If trying to access index beyond range
		// Exit gracefully
		if( terms[which_max(terms)] >= X_clean->size2 ){
			y.free();
			X_clean.free();
		
			throw "Element in \"terms\" is out of range";
		}

		gsl_matrix *Xu_clean = gsl_matrix_alloc( eigenVectors->size1, X_clean->size2 );

		// Xu_clean = crossprod( decomp$vectors, X_clean)
		gsl_blas_dgemm( CblasNoTrans, CblasNoTrans, 1.0, eigenVectors, X_clean, 0.0, Xu_clean );

		// get indeces of columns that depend on X[,j]
		vector<int> loopIndex = expr.get_loop_terms();

		long batch_size = MAX( 1, setSize/100.0 );

		// Fit null model if necessary
		if( R_IsNA( delta_global ) && ! reEstimateDelta ){
				
				double log_L, sig_g, sig_e;

				RcppGSL::matrix<double> X_null = expr.get_model_matrix_null(); 			

				LRGPR *lrgpr = new LRGPR( y, eigenVectors, eigenValues, X_null->size2, useProxCon ? Wtilde->size2 : 0);

				// Must update W before X
				if( useProxCon ) lrgpr->update_Wtilde( Wtilde );
				lrgpr->update_X( X_null );

				// Estimate delta
				lrgpr->fit_mle( &log_L, &sig_g, &sig_e);
				delta_global = sig_e / sig_g;

				X_null.free();
				delete lrgpr;
		}

		long tests_completed = 0;

		std::vector<double> pValues( n_markers );

		time_t start_time;
		time(&start_time);

		for(int i_set=0; i_set<n_markers; i_set+=setSize){

			gsl_matrix *X_set = fset.getNextChunk();

			#pragma omp parallel
			{
				// Variables local to each thread		
				LRGPR *lrgpr = new LRGPR( y, eigenVectors, eigenValues, X_clean->size2, useProxCon ? Wtilde->size2 : 0);

				// Must update W before X
				if( useProxCon ) lrgpr->update_Wtilde( Wtilde );

				gsl_matrix *X = gsl_matrix_alloc( X_clean->size1, X_clean->size2 );
				gsl_matrix *Xu = gsl_matrix_alloc( eigenVectors->size1, X_clean->size2 );

				gsl_vector_view col_view, col_view2;

				double log_L, sig_g, sig_e;
				double delta;

				gsl_vector *marker_j = gsl_vector_alloc( y->size ); 

				#pragma omp for schedule(static, batch_size)
				for(int j=0; j<setSize; j++){		

					// if index exceeds n_markers, continue to next interation
					if(  j+i_set >= n_markers ) continue;

					#pragma omp critical
					tests_completed++;

					// Copy feature data from X_loop to marker_j in a way that does not use
					// 	Rcpp or RcppGSL objects.  This avoid the Rcpp thread-safety issues.
					// In summary, Rcpp objects, even if they declared in their own thread, 
					// 	CANNOT be written to safely
					// Here I do the copying manually, instead of using the nice Rcpp methods
					if( standardMatrix ){
						for(int h=0; h<marker_j->size; h++){
							gsl_vector_set(marker_j, h, X_loop(h,j));
						}	
					}else{		
						for(int h=0; h<marker_j->size; h++){
							gsl_vector_set(marker_j, h, gsl_matrix_get(X_set, j, h) );
						}							
					}			

					// replace missings values with the mean
					gsl_vector_set_missing_mean( marker_j );					
					
					// Copy clean version of matrices to active variables
					gsl_matrix_memcpy( X, X_clean );
					gsl_matrix_memcpy( Xu, Xu_clean );	

					for(int k=0; k<loopIndex.size(); k++){			
						// X_design[,loopIndex[k]] = X_design[,loopIndex[k]] * marker_j
						col_view = gsl_matrix_column( (gsl_matrix*)(X), loopIndex[k] );
						gsl_vector_mul( &col_view.vector, marker_j );

						// Xu[,loopIndex[k]] = crossprod( U, X_design[,loopIndex[k]] * marker_j)
						col_view2 = gsl_matrix_column( (gsl_matrix*)(Xu), loopIndex[k] );
						gsl_blas_dgemv( CblasNoTrans, 1.0, eigenVectors, &col_view.vector, 0.0, &col_view2.vector );
					}

					//lrgpr->update_X( X, eigen );
					lrgpr->update_Xu( X, Xu );						

					if( reEstimateDelta ){
						// Estimate delta
						lrgpr->fit_mle( &log_L, &sig_g, &sig_e);
					
						delta = sig_e / sig_g;
					}else{
						// Use pre-specified delta value
						lrgpr->fit_fixed_delta( delta_global, &log_L, &sig_g, &sig_e );
						delta = delta_global;
					}

					#pragma omp critical
					pValues[j+i_set] = lrgpr->wald_test( terms );	

				} // END for
							
				delete lrgpr;			
				gsl_matrix_free( X );
				gsl_matrix_free( Xu );
				gsl_vector_free( marker_j );

			} // End parallel

			print_progress( tests_completed, n_markers, 25, start_time);

			// X_set is NULL if fset is empty
			if( X_set != NULL ) gsl_matrix_free( X_set );

		} // End set loop

		cout << endl;

		y.free();
		X_clean.free();
		gsl_matrix_free(Xu_clean);

		eigenVectors.free();
		eigenValues.free();
		Wtilde.free();

		return( wrap(pValues) );

	} catch( const char* msg ){
		 Rcpp::Rcout << "glm error: " << msg << endl;
	}catch( std::exception &ex ) {
		Rcpp::Rcout << "exception:  " << endl;
		forward_exception_to_r( ex );
	} catch(...) { 
		::Rf_error( "c++ exception (unknown reason)" ); 
	}
	return R_NilValue; // -Wall
}
