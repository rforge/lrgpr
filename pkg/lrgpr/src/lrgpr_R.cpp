
#define MAX(a,b) a>b?a:b
#define MIN(a,b) a<b?a:b
#include <RcppGSL.h>
#include <Rcpp.h>

#define error Rf_error
#include <progress.hpp>

#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_statistics.h>

#include <omp.h>

#include <iostream>
#include <algorithm>
#include <string>
#include <math.h>

#include <bigmemory/MatrixAccessor.hpp>


#include "lrgpr.h"
#include "gsl_additions.h"
#include "gsl_lapack.h"
#include "GLM.h"
#include "misc_functions.h"
#include "featureBatcher.h"
#include "R_interface.h"

#ifdef INTEL
#include "mkl_service.h"
#endif

using namespace std;
using namespace Rcpp;

RcppExport SEXP R_lrgpr( SEXP Y_, SEXP X_, SEXP eigenVectors_, SEXP eigenValues_, SEXP delta_, SEXP nthreads_, SEXP Wtilde_){

BEGIN_RCPP
	
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

	gsl_matrix_set_missing_mean_col( X );

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

END_RCPP
}


RcppExport SEXP R_glmApply( SEXP expression_, SEXP data_, SEXP pBigMat_, SEXP env_, SEXP terms_, SEXP nthreads_, SEXP useMean_, SEXP useIdentityLink_, SEXP univariate_, SEXP multivariate_, SEXP quiet_){

BEGIN_RCPP
		
	CharacterVector expression( expression_ );
	Environment env( env_ );
	std::vector<int> terms = as<std::vector<int> >( terms_ ); 
	int nthreads = as<int>( nthreads_ );
	bool useMean = (bool) as<int>( useMean_ );
	bool useIdentityLink = as<int>( useIdentityLink_ );
	bool univariate = as<int>( univariate_ );
	bool multivariate = as<int>( multivariate_ );
	bool quiet = as<int>( quiet_ );
		
	regressionType regressType = useIdentityLink ? LINEAR : LOGISTIC;

	featureBatcher fbatch( data_, pBigMat_, 10000);

	// Set threads to 1
	omp_set_num_threads( nthreads );
	// Intel paralellism
	#ifdef INTEL
	mkl_set_num_threads( 1 );
	#endif
	// disable nested OpenMP parallelism
	omp_set_nested(0);

	//cout << "expression: " << as<string>( expression ) << endl;
	// Process exression, X_loop and env
	Rexpress expr( as<string>( expression ), fbatch.get_N_indivs(), env );			

	RcppGSL::matrix<double> Y = expr.get_response_m(); 

	int n_indivs = Y->size1;
	int n_pheno = Y->size2;

	if( n_pheno > 1 && regressType == LOGISTIC ){
		Y.free();
	
		throw invalid_argument("Cannot do multivariate logistic regression\n");
	}

	RcppGSL::matrix<double> Xn = expr.get_model_matrix_null();

	if( n_pheno == 1 ) multivariate = false;

	// Check that sizes of and X_loop match
	if( n_indivs != fbatch.get_N_indivs() ){
		Y.free();
		Xn.free();
	
		throw invalid_argument("Dimensions of response and design matrix do not match\n");
	}
	
	NumericMatrix pValues;
	NumericMatrix pValues_multivariate;

	// Define p-values matrix to be returned, only of the corresponding test is performed
	if( univariate )  	pValues = NumericMatrix( fbatch.get_N_features(), n_pheno );
	if( multivariate )  pValues_multivariate = NumericMatrix( fbatch.get_N_features(), 2 );

	// X_clean = model.matrix.default( y ~ sex:One + age )
	// Replace marker with 1's so that design matrix for marker j can be created by multiplcation
	RcppGSL::matrix<double> X_clean = expr.get_model_matrix_clean(); 

	// If trying to access index beyond range
	// Exit gracefully
	if( terms[which_max(terms)] >= X_clean->size2 ){
		Y.free();
		Xn.free();
		X_clean.free();
	
		throw range_error("Element in \"terms\" is out of range");		
	}

	// get indeces of columns that depend on X[,j]
	vector<int> loopIndex = expr.get_loop_terms();
	//vector<string> colNames = expr.get_terms();		

	cout << "loopIndex: " << endl;
	print_vector(loopIndex);

	//n_markers = MIN(n_markers, 100000);

	//cout << "n_indivs: " << n_indivs << endl;
	//cout << "n_markers: " << n_markers << endl;
	
	long batch_size = MAX( 1, fbatch.getBatchSize()/100.0 ); 

	long tests_completed = 0;
	Progress p(0, false);

	time_t start_time;
	time(&start_time);

	for(int i_set=0; i_set<fbatch.get_N_features(); i_set+=fbatch.getBatchSize()){

		// Load data from binary matrix (or do noting if NumericMatrix is used)
		fbatch.loadNextChunk();

		#pragma omp parallel
		{
			// Variables local to each thread
			gsl_matrix *X = gsl_matrix_alloc( X_clean->size1, X_clean->size2 );

			gsl_vector_view col_view, col_view_y;

			gsl_vector *marker_j = gsl_vector_alloc( fbatch.get_N_indivs() ); 

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
			for(int j=0; j<fbatch.getBatchSize(); j++){		

				#pragma omp critical
				tests_completed++;

				fbatch.getFeautureInChunk( j, marker_j );

				if( useMean )	isMissing = gsl_vector_set_missing_mean( marker_j );					
				else 			isMissing = any_is_na( marker_j->data, marker_j->size );

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
						gsl_vector *pvals = GLM_wald_test( workmv, terms );

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

		if( ! quiet )  Rcpp::Rcout << print_progress( tests_completed, fbatch.get_N_features(), 25, start_time);

		if( Progress::check_abort() ){
			Y.free();
			Xn.free();
			X_clean.free();

			return wrap(0);
		}
	} // End set loop

	if( ! quiet ) Rcpp::Rcout << endl;

	Y.free();
	Xn.free();
	X_clean.free();

	Rcpp::List res = Rcpp::List::create(Rcpp::Named("pValues") 	= pValues,
										Rcpp::Named("pValues_mv") = pValues_multivariate);

	return( res );

END_RCPP
}

vector<int> get_markers_in_window( const string chr_j, const double loc_j, const vector<string> &chrom, const vector<double> &location, const double distance){

	// # get markers on the same chromosome
	// idx = which( chrom == chr_j )
	vector<int> idx = which( chrom,  chr_j );

	//print_vector(idx);

	// sort indeces
	sort(idx.begin(), idx.end());

	//print_vector(idx);

	// # get markers within the window 
	// idx2 = which( abs(location[idx] - loc_j) <= distance )
	// return idx[idx2]

	// the vector names are mixed up here due to the differences in C++ vs R
	vector<int> idx2;

	for(int i=0; i<idx.size(); i++){
		if( fabs(location[idx[i]] - loc_j) <= distance ){
			idx2.push_back(idx[i]);
		}
	}

	return idx2;
}



RcppExport SEXP R_lrgprApply( SEXP expression_, SEXP data_, SEXP pBigMat_, SEXP env_, SEXP terms_, SEXP EigenVectors_, SEXP EigenValues_, SEXP Wtilde_, SEXP rank_,  SEXP chromosome_, SEXP location_, SEXP distance_, SEXP dcmp_features_, SEXP scale_, SEXP delta_, SEXP reEstimateDelta_, SEXP nthreads_, SEXP quiet_){

//BEGIN_RCPP
		
	CharacterVector expression( expression_ );
	Environment env( env_ );
	std::vector<int> terms = as<std::vector<int> >( terms_ ); 
	RcppGSL::matrix<double> eigenVectors = EigenVectors_;
	RcppGSL::vector<double> eigenValues = EigenValues_; 
	RcppGSL::matrix<double> Wtilde = Wtilde_; 
	int rank = as<int>( rank_ );
	std::vector<string> chromosome = as<std::vector<string> >(chromosome_);
	std::vector<double> location = as<std::vector<double> >(location_);
	double distance = as<double>( distance_ );
	std::vector<int> dcmp_features = as<std::vector<int> >( dcmp_features_ );
	bool scale = (bool) Rcpp::as<int>( scale_ );
	double delta_global = as<double>( delta_ );
	bool reEstimateDelta = Rcpp::as<int>( reEstimateDelta_ );
	int nthreads = as<int>( nthreads_ );
	bool quiet = Rcpp::as<int>( quiet_ );

	// Make sure all eigen values are non-negative
	for(unsigned int i=0; i<eigenValues->size; i++){
		if( gsl_vector_get( eigenValues, i) < 0 )  gsl_vector_set( eigenValues, i, 0);
	}
	
	featureBatcher fbatch( data_, pBigMat_, 10000);

	// Set threads to 1
	omp_set_num_threads( nthreads );
	// Intel paralellism
	#ifdef INTEL
	mkl_set_num_threads( 1 );
	#endif
	// disable nested OpenMP parallelism
	omp_set_nested(0);

	// Process exression, X_loop and env
	Rexpress expr( as<string>( expression ), fbatch.get_N_indivs(), env );			

	RcppGSL::vector<double> y = expr.get_response(); 

	int n_indivs = y->size;

	// Check that sizes of and X_loop match
	if( n_indivs != fbatch.get_N_indivs() ){
		y.free();
	
		throw invalid_argument("Dimensions of response and design matrix do not match\n");
	}

	// If # of samples in W and y is the same
	bool useProxCon = ( Wtilde->size1 == y->size );

	if( useProxCon && scale) gsl_matrix_center_scale(Wtilde);
							
	// X_clean = model.matrix.default( y ~ sex:One + age )
	// Replace marker with 1's so that design matrix for marker j can be created by multiplcation
	RcppGSL::matrix<double> X_clean = expr.get_model_matrix_clean(); 

	// If trying to access index beyond range
	// Exit gracefully
	if( terms[which_max(terms)] >= X_clean->size2 ){
		y.free();
		X_clean.free();
	
		throw range_error("Element in \"terms\" is out of range");
	}

	gsl_matrix *Xu_clean = gsl_matrix_alloc( eigenVectors->size2, X_clean->size2 );

	// Xu_clean = crossprod( decomp$vectors, X_clean)
	gsl_blas_dgemm( CblasTrans, CblasNoTrans, 1.0, eigenVectors, X_clean, 0.0, Xu_clean );

	// get indeces of columns that depend on X[,j]
	vector<int> loopIndex = expr.get_loop_terms();

	long batch_size = MAX( 1, fbatch.getBatchSize()/100.0 ); 

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
	Progress p(0, false);

	std::vector<double> pValues( fbatch.get_N_features() );

	// Get map for marker in dcmp_features
	// map_local = map[dcmp_features,]
	vector<string> chromosome_local = get_values( chromosome, dcmp_features);
	vector<double> location_local = get_values( location, dcmp_features); 

	time_t start_time;
	time(&start_time);

	gsl_matrix *X_set = NULL;

	for(int i_set=0; i_set<fbatch.get_N_features(); i_set+=fbatch.getBatchSize()){

		// Load data from binary matrix (or do noting if NumericMatrix is used)
		fbatch.loadNextChunk();

		#pragma omp parallel
		{
			// Variables local to each thread		
			LRGPR *lrgpr = new LRGPR( y, eigenVectors, eigenValues, X_clean->size2, useProxCon ? Wtilde->size2 : 0);

			gsl_matrix *X = gsl_matrix_alloc( X_clean->size1, X_clean->size2 );
			gsl_matrix *Xu = gsl_matrix_alloc( eigenVectors->size2, X_clean->size2 );

			gsl_vector_view col_view, col_view2;

			double log_L, sig_g, sig_e;
			double delta;

			gsl_vector *marker_j = gsl_vector_alloc( y->size ); 

			gsl_matrix *Wtilde_local = NULL;
			vector<int> exclude_prev; 

			// Initialize with meaningless data
			exclude_prev.push_back(-10);

			#pragma omp for schedule(static, batch_size)		
			for(int j=0; j<fbatch.getBatchSize(); j++){		

				#pragma omp critical
				tests_completed++;

				fbatch.getFeautureInChunk( j, marker_j );	

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
					gsl_blas_dgemv( CblasTrans, 1.0, eigenVectors, &col_view.vector, 0.0, &col_view2.vector );
				}

				// Proximal contamination
				//////////////////////////

				// Must update W before X
				if( useProxCon ) lrgpr->update_Wtilde( Wtilde );

				// If a MAP was specified
				if( chromosome.size() > 1 ){

					vector<int> idx = get_markers_in_window( (string) chromosome[j+i_set], location[j+i_set], chromosome_local, location_local, distance);

					vector<int> exclude = get_values( dcmp_features, idx);

					/*cout << j+i_set <<": " << endl;
					
					cout << "idx: " << endl;
					print_vector(idx);

					cout << "exclude: " << endl;
					print_vector(exclude);*/

					// If exclude != exclude_prev, Wtilde_local should be updated
					if( ! equal(exclude.begin(), exclude.end(), exclude_prev.begin()) ){

						if( exclude.size() > 0 ){

							// Wtilde_local = X[,exclude]
							///////////////////////////////

							if( Wtilde_local != NULL ) gsl_matrix_free( Wtilde_local );
	
							Wtilde_local = fbatch.getFeatures( exclude );

							gsl_matrix_set_missing_mean_col( Wtilde_local );

							if( scale ){
								gsl_matrix_center_scale(Wtilde_local);
							}

						}else{
							if( Wtilde_local != NULL ) gsl_matrix_free( Wtilde_local );
							Wtilde_local = NULL;
						}

						// Update W_tilde based on marker window
						// if argument is NULL, proximal contamination is not used
						///////////////////////////////////

						// If a global Wtilde is given
						if( useProxCon ){
							if( Wtilde_local == NULL ){
								lrgpr->update_Wtilde( Wtilde );
							}
							if( Wtilde_local != NULL ){

								gsl_matrix *Wcbind = gsl_matrix_alloc( Wtilde->size1, Wtilde_local->size2 + Wtilde->size2);
								gsl_matrix_cbind( Wtilde, Wtilde_local, Wcbind);

								lrgpr->update_Wtilde( Wcbind );

								gsl_matrix_free(Wcbind);
							}
						}else{						
							lrgpr->update_Wtilde( Wtilde_local );
						}
					}

					// save exclude for next marker
					exclude_prev = exclude;
				}

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
			if( Wtilde_local != NULL ) gsl_matrix_free( Wtilde_local );

		} // End parallel

		if( ! quiet )  Rcpp::Rcout << print_progress( tests_completed, fbatch.get_N_features(), 25, start_time);

		if( Progress::check_abort() ){

			y.free();
			X_clean.free();
			gsl_matrix_free(Xu_clean);

			eigenVectors.free();
			eigenValues.free();
			Wtilde.free();
			return wrap(0);
		}

	} // End set loop

	if( ! quiet ) Rcpp::Rcout << endl;

	y.free();
	X_clean.free();
	gsl_matrix_free(Xu_clean);

	eigenVectors.free();
	eigenValues.free();
	Wtilde.free();

	return( wrap(pValues) );

//END_RCPP
}

RcppExport SEXP R_glmApply2( SEXP expression_, SEXP data_, SEXP pBigMat_, SEXP env_, SEXP terms_, SEXP nthreads_, SEXP useMean_, SEXP useIdentityLink_, SEXP univariate_, SEXP multivariate_, SEXP quiet_){

BEGIN_RCPP
		
	CharacterVector expression( expression_ );
	Environment env( env_ );
	std::vector<int> terms = as<std::vector<int> >( terms_ ); 
	int nthreads = as<int>( nthreads_ );
	bool useMean = (bool) as<int>( useMean_ );
	bool useIdentityLink = as<int>( useIdentityLink_ );
	bool univariate = as<int>( univariate_ );
	bool multivariate = as<int>( multivariate_ );
	bool quiet = as<int>( quiet_ );
		
	regressionType regressType = useIdentityLink ? LINEAR : LOGISTIC;

	featureBatcher fbatch( data_, pBigMat_, 10000);

	// Set threads to 1
	omp_set_num_threads( nthreads );
	// Intel paralellism
	#ifdef INTEL
	mkl_set_num_threads( 1 );
	#endif
	// disable nested OpenMP parallelism
	omp_set_nested(0);

	// Process exression, X_loop and env
	Rexpress expr( as<string>( expression ), fbatch.get_N_indivs(), env );			

	RcppGSL::matrix<double> Y = expr.get_response_m(); 

	int n_indivs = Y->size1;
	int n_pheno = Y->size2;

	if( n_pheno > 1 && regressType == LOGISTIC ){
		Y.free();
	
		throw invalid_argument("Cannot do multivariate logistic regression\n");
	}

	RcppGSL::matrix<double> Xn = expr.get_model_matrix_null();

	if( n_pheno == 1 ) multivariate = false;

	// Check that sizes of and X_loop match
	if( n_indivs != fbatch.get_N_indivs() ){
		Y.free();
		Xn.free();
	
		throw invalid_argument("Dimensions of response and design matrix do not match\n");
	}
	
	NumericMatrix pValues;
	NumericMatrix pValues_multivariate;

	// Define p-values matrix to be returned, only of the corresponding test is performed
	if( univariate )  	pValues = NumericMatrix( fbatch.get_N_features(), n_pheno );
	if( multivariate )  pValues_multivariate = NumericMatrix( fbatch.get_N_features(), 2 );

	// X_clean = model.matrix.default( y ~ sex:One + age )
	// Replace marker with 1's so that design matrix for marker j can be created by multiplcation
	RcppGSL::matrix<double> X_clean = expr.get_model_matrix_clean(); 

	// If trying to access index beyond range
	// Exit gracefully
	if( terms[which_max(terms)] >= X_clean->size2 ){
		Y.free();
		Xn.free();
		X_clean.free();
	
		throw range_error("Element in \"terms\" is out of range");		
	}

	// get indeces of columns that depend on SNP
	vector<int> loopIndex = expr.get_loop_terms();

	// Indeces that don't depend on SNP, i,e. are complementary
	vector<int> loopCompl;

	for(int i=0; i<X_clean->size2; i++){
		if( ! is_in_vector(loopIndex, i) ){
			loopCompl.push_back(i);
		}
	}

	// Extract covariate terms
	gsl_matrix *X_clean_cov = gsl_matrix_alloc(X_clean->size1, loopCompl.size());
	gsl_matrix_sub_col(X_clean, loopCompl, X_clean_cov);

	LM_preproc *preproc = LM_preproc_alloc( Y, X_clean_cov);

	gsl_matrix_free(X_clean_cov);
	
	long batch_size = MAX( 1, fbatch.getBatchSize()/100.0 ); 

	long tests_completed = 0;
	Progress p(0, false);

	time_t start_time;
	time(&start_time);

	for(int i_set=0; i_set<fbatch.get_N_features(); i_set+=fbatch.getBatchSize()){

		// Load data from binary matrix (or do noting if NumericMatrix is used)
		fbatch.loadNextChunk();

		#pragma omp parallel
		{
			Preproc_workspace *proc_work = Preproc_workspace_alloc( Y, loopIndex.size(), preproc);

			// Variables local to each thread
			gsl_matrix *X = gsl_matrix_alloc( X_clean->size1, X_clean->size2 );

			gsl_vector_view col_view, col_view_y;

			gsl_vector *marker_j = gsl_vector_alloc( fbatch.get_N_indivs() ); 

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
			for(int j=0; j<fbatch.getBatchSize(); j++){	

				#pragma omp critical
				tests_completed++;

				fbatch.getFeautureInChunk( j, marker_j );	

				if( useMean ) 	isMissing = gsl_vector_set_missing_mean( marker_j );					
				else 			isMissing = any_is_na( marker_j->data, marker_j->size );

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

					// Extract marker terms
					gsl_matrix *X_markers = gsl_matrix_alloc(X->size1, loopIndex.size());
					gsl_matrix_sub_col(X, loopIndex, X_markers);

					gsl_vector *pvals = GLM_regression_preproc( Y, X_markers, workmv, preproc, proc_work);

					gsl_matrix_free(X_markers);

					#pragma omp critical
	 				for(int k=0; k<n_pheno; k++){
						pValues(j+i_set,k) = gsl_vector_get(pvals, k);
					}
					gsl_vector_free( pvals );


					/*if( multivariate ){
						GLM_HotellingPillai_test( X, workmv, terms, &Hotelling, &Pillai );

						#pragma omp critical
						{
							pValues_multivariate(j+i_set,0) = Hotelling;
							pValues_multivariate(j+i_set,1) = Pillai;
						}							
					}*/

					/*if( univariate ){
						// Perform Wald test
						gsl_vector *pvals = GLM_wald_test( workmv, terms );

						#pragma omp critical
		 				for(int k=0; k<n_pheno; k++){
							pValues(j+i_set,k) = gsl_vector_get(pvals, k);
						}
						gsl_vector_free( pvals );
					}*/

				}else{						

					// LOGISTIC
					GLM_unpenalized_regression_IRLS(y, X, beta, &sig_e, &log_L, &rank, regressType, false, irls_workspace);

					#pragma omp critical
					pValues(j+i_set,0) = GLM_wald_test( beta, irls_workspace->Sigma, X->size1, terms, regressType, irls_workspace->design_is_singular);
				}

			} // END for
						
			Preproc_workspace_free(proc_work);
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

		if( ! quiet )  Rcpp::Rcout << print_progress( tests_completed, fbatch.get_N_features(), 25, start_time);

		if( Progress::check_abort() ){
			Y.free();
			Xn.free();
			X_clean.free();

			return wrap(0);
		}
	} // End set loop

	if( ! quiet ) Rcpp::Rcout << endl;

	Y.free();
	Xn.free();
	X_clean.free();
	LM_preproc_free( preproc );

	Rcpp::List res = Rcpp::List::create(Rcpp::Named("pValues") 	= pValues,
										Rcpp::Named("pValues_mv") = pValues_multivariate);

	return( res );

END_RCPP
}


RcppExport SEXP R_getAlleleFreq( SEXP data_, SEXP pBigMat_, SEXP nthreads_, SEXP quiet_){

BEGIN_RCPP

	int nthreads = as<int>( nthreads_ );	
	bool quiet = as<int>( quiet_ );
	
	featureBatcher fbatch( data_, pBigMat_, 10000);

	// Set threads to 1
	omp_set_num_threads( nthreads );
	// Intel paralellism
	#ifdef INTEL
	mkl_set_num_threads( 1 );
	#endif
	// disable nested OpenMP parallelism
	omp_set_nested(0);

	long batch_size = MAX( 1, fbatch.getBatchSize()/100.0 ); 

	long tests_completed = 0;
	Progress p(0, false);

	time_t start_time;
	time(&start_time); 

	vector<double> alleleFreq( fbatch.get_N_features() );

	for(int i_set=0; i_set<fbatch.get_N_features(); i_set+=fbatch.getBatchSize()){

		// Load data from binary matrix (or do noting if NumericMatrix is used)
		fbatch.loadNextChunk();

		#pragma omp parallel
		{		
			gsl_vector *marker_j = gsl_vector_alloc( fbatch.get_N_indivs() ); 

			double N_present; 

			#pragma omp for schedule(static, batch_size)
			for(int j=0; j<fbatch.getBatchSize(); j++){		

				#pragma omp critical
				tests_completed++;

				fbatch.getFeautureInChunk( j, marker_j );	

				N_present = fbatch.get_N_indivs() - gsl_vector_count_missing( marker_j );

				alleleFreq[j+i_set]	= gsl_vector_sum_elements_finite( marker_j ) / (2.0*N_present);	

			} // END for

			gsl_vector_free(marker_j);

		} // End parallel

		if( ! quiet )  Rcpp::Rcout << print_progress( tests_completed, fbatch.get_N_features(), 25, start_time);

		if( Progress::check_abort() ){
			return wrap(0);
		}

	} // End set loop

	if( ! quiet ) Rcpp::Rcout << endl;

	return( wrap(alleleFreq) );

END_RCPP
}


RcppExport SEXP R_getMissingCount( SEXP data_, SEXP pBigMat_, SEXP nthreads_, SEXP quiet_){

BEGIN_RCPP

	int nthreads = as<int>( nthreads_ );	
	bool quiet = as<int>( quiet_ );
	
	featureBatcher fbatch( data_, pBigMat_, 10000);

	// Set threads to 1
	omp_set_num_threads( nthreads );
	// Intel paralellism
	#ifdef INTEL
	mkl_set_num_threads( 1 );
	#endif
	// disable nested OpenMP parallelism
	omp_set_nested(0);

	long batch_size = MAX( 1, fbatch.getBatchSize()/100.0 ); 

	long tests_completed = 0;
	Progress p(0, false);

	time_t start_time;
	time(&start_time); 

	vector<double> missingCount( fbatch.get_N_features() );

	for(int i_set=0; i_set<fbatch.get_N_features(); i_set+=fbatch.getBatchSize()){

		// Load data from binary matrix (or do noting if NumericMatrix is used)
		fbatch.loadNextChunk();

		#pragma omp parallel
		{		
			gsl_vector *marker_j = gsl_vector_alloc( fbatch.get_N_indivs() ); 

			#pragma omp for schedule(static, batch_size)
			for(int j=0; j<fbatch.getBatchSize(); j++){		

				#pragma omp critical
				tests_completed++;

				fbatch.getFeautureInChunk( j, marker_j );	

				missingCount[j+i_set] = gsl_vector_count_missing( marker_j );

			} // END for

			gsl_vector_free(marker_j);

		} // End parallel

		if( ! quiet )  Rcpp::Rcout << print_progress( tests_completed, fbatch.get_N_features(), 25, start_time);

		if( Progress::check_abort() ){
			return wrap(0);
		}

	} // End set loop

	if( ! quiet ) Rcpp::Rcout << endl;

	return( wrap(missingCount) );

END_RCPP
}

RcppExport SEXP R_getAlleleVariance( SEXP data_, SEXP pBigMat_, SEXP nthreads_, SEXP quiet_){

BEGIN_RCPP

	int nthreads = as<int>( nthreads_ );
	bool quiet = as<int>( quiet_ );
	
	featureBatcher fbatch( data_, pBigMat_, 10000);

	// Set threads to 1
	omp_set_num_threads( nthreads );
	// Intel paralellism
	#ifdef INTEL
	mkl_set_num_threads( 1 );
	#endif
	// disable nested OpenMP parallelism
	omp_set_nested(0);

	long batch_size = MAX( 1, fbatch.getBatchSize()/100.0 ); 

	long tests_completed = 0;
	Progress p(0, false);

	time_t start_time;
	time(&start_time);

	vector<double> colVariance( fbatch.get_N_features() );

	for(int i_set=0; i_set<fbatch.get_N_features(); i_set+=fbatch.getBatchSize()){

		// Load data from binary matrix (or do noting if NumericMatrix is used)
		fbatch.loadNextChunk();

		#pragma omp parallel
		{		
			gsl_vector *marker_j = gsl_vector_alloc( fbatch.get_N_indivs() ); 

			#pragma omp for schedule(static, batch_size)			
			for(int j=0; j<fbatch.getBatchSize(); j++){			

				#pragma omp critical
				tests_completed++;

				// if( ! fbatch.excluded() )

				fbatch.getFeautureInChunk( j, marker_j );	

				// Compute varaince for non-missing entries
				gsl_vector *v = gsl_vector_get_nonmissing( marker_j );

				if( v == NULL ) colVariance[j+i_set] = -1;
				else colVariance[j+i_set] = gsl_stats_variance( v->data, v->stride, v->size);

				gsl_vector_free(v);

			} // END for

			gsl_vector_free(marker_j);

		} // End parallel

		if( ! quiet )  Rcpp::Rcout << print_progress( tests_completed, fbatch.get_N_features(), 25, start_time);

		if( Progress::check_abort() ){
			return wrap(0);
		}
	} // End set loop

	if( ! quiet ) Rcpp::Rcout << endl;

	return( wrap(colVariance) );

END_RCPP
}