// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 4 -*-
//
// colNorm.cpp: Rcpp and GSL based example of column norm
//              adapted from `Section 8.4.13 Example programs for matrices' 
//              of the GSL manual
//
// Copyright (C)  2010 Dirk Eddelbuettel and Romain Francois
//
// This file is part of RcppGSL.
//
// RcppGSL is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// RcppGSL is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with RcppGSL.  If not, see <http://www.gnu.org/licenses/>.

#define MAX(a,b) a>b?a:b
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

using namespace std;
using namespace Rcpp;

#include "fastGPR.h"
#include "ARD.h"
#include "pGLM.h"
#include "gsl_additions.h"
#include "gsl_lapack.h"
#include "quantGen.h"
#include "GLM.h"

#ifdef INTEL
#include "mkl_service.h"
#endif


static Rcpp::Function asMatrix("as.matrix");


extern "C" SEXP R_fastGPR( SEXP sY, SEXP sX, SEXP sEigenVectors, SEXP sEigenValues, SEXP srank, SEXP sdelta, SEXP schisq, SEXP snthreads){

	try {
	
		RcppGSL::matrix<double> X = sX; 
		RcppGSL::vector<double> y = sY; 	
		RcppGSL::matrix<double> eigenVectors = sEigenVectors; 	
		RcppGSL::vector<double> eigenValues = sEigenValues; 	

		int rank = Rcpp::as<int>(srank);
		double delta = Rcpp::as<double>(sdelta);
		bool useChisqLikelihood = ( Rcpp::as<int>(schisq) != 0 );
		double nthreads = Rcpp::as<double>(snthreads);

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

		// initialize eigenDecomp
		eigenDecomp *eigen = new eigenDecomp( eigenVectors, eigenValues, false, rank);

		eigenVectors.free();
		eigenValues.free();

		LMM_params *params = new LMM_params( y, X, eigen, rank, useChisqLikelihood);

		params->set_delta_grid( exp(-10), exp(10), 100);

		double log_L, sig_g, sig_e;

		gsl_vector *pValues = gsl_vector_alloc( X->size2 );
		gsl_vector *sd = gsl_vector_alloc( X->size2 );

		if( isnan( delta ) ){
			// Estimate delta
			LMM_fit_mle( params, &log_L, &sig_g, &sig_e, pValues, sd);

			delta = sig_e / sig_g;
		}else{
			// Use pre-specified delta value
			LMM_fit_fixed_delta( params, delta, &log_L, rank, true, pValues, sd );

			sig_g = params->sig_g;
			sig_e = delta * sig_g;
		}

		gsl_vector *alpha = gsl_vector_alloc( y->size );

		gsl_vector *Hii = LMM_get_hat_matrix_diag( delta, X, eigen, params );
		gsl_vector *Y_hat = LMM_get_fitted_response( y, delta, X, eigen, params, alpha);
		gsl_matrix *Sigma = LMM_coeff_covariance( params, delta);

		RcppGSL::vector<double> beta = params->beta;
		RcppGSL::vector<double> pValuesReturn = pValues;
		RcppGSL::vector<double> sdReturn = sd;
		RcppGSL::vector<double> HiiReturn = Hii;
		RcppGSL::vector<double> Y_hatReturn = Y_hat;
		RcppGSL::vector<double> alphaReturn = alpha;
		RcppGSL::matrix<double> Sigmareturn = Sigma;

		Rcpp::List res = Rcpp::List::create(Rcpp::Named("coefficients") = beta,
											Rcpp::Named("p.values") 	= pValuesReturn,
											Rcpp::Named("sd") 			= sdReturn,
											Rcpp::Named("sigSq_e") 		= sig_e,
											Rcpp::Named("sigSq_a") 		= sig_g,
											Rcpp::Named("delta") 		= delta, 
											Rcpp::Named("rank") 		= rank,
											Rcpp::Named("logLik")		= log_L,
											Rcpp::Named("fitted.values")= Y_hatReturn,
											Rcpp::Named("alpha")		= alphaReturn,
											Rcpp::Named("Sigma")		= Sigmareturn,				
											//Rcpp::Named("df")			= LMM_get_effective_df( delta, X, eigen, params ));//, // calculate df = sum(Hii) in R code
											Rcpp::Named("hii")			= HiiReturn );
		
		X.free(); 
		y.free();
		
		//beta.free(); beta owned by params_null and is free'd by the class destructor
		pValuesReturn.free();
		sdReturn.free();
		HiiReturn.free();
		Y_hatReturn.free();
		alphaReturn.free();
		Sigmareturn.free();

		delete params;
		delete eigen ;

		return res; // return the result list to R

	} catch( std::exception &ex ) {
		forward_exception_to_r( ex );
	} catch(...) { 
		::Rf_error( "c++ exception (unknown reason)" ); 
	}
	return R_NilValue; // -Wall
}


class Rexpression {

	private:
	string expression, query, loop, lcv;
	vector<int> index;
	Environment env;
	public:
	/*Rexpression( const string &expression_, const string &loop_, const string &lcv_, Environment &env_){
		
		expression 	= expression_;
		loop 		= loop_;
		lcv 		= lcv_;
		env 		= env_;

		query = loop + "[," + lcv + "]";

		// Find position of loop variable to replace
		// each instance is recorded in index
		int loc = -1;
		while(1){
			loc = expression.find( query, loc+1);
			if( loc == string::npos ) break;
			else index.push_back( loc );
		}
	}*/

	Rexpression( const string &expression_, const string &loop_, const string &lcv_, Environment &env_){
		
		expression 	= expression_;
		loop 		= loop_;
		lcv 		= lcv_;
		env 		= env_;

		query = loop + "[," + lcv + "]";

		// Find position of loop variable to replace
		// each instance is recorded in index
		int loc = -1;
		while(1){
			loc = expression.find( query, loc+1);
			if( loc == string::npos ) break;
			else index.push_back( loc );
		}
	}

	string update_expression( const int j ){

		string expressLocal = expression;

		// Replace generic loop variable with an actual variable name
		// DO NOT use unsigned int in decrementing loop, since comparing it to zero is not valid
		for(int i=index.size()-1; i>=0; i--){
			expressLocal.replace( index[i], query.size(), loop + "[," + stringify( j ) + "]");
		}

		return expressLocal;
	}

	RcppGSL::matrix<double> get_model_matrix( const int j ){

			Formula form( update_expression( j ) );
			Language call( "model.matrix.default", form);	
			return call.eval( env );
	}

	RcppGSL::matrix<double> get_model_matrix_clean(){

			Formula form( update_expression( 1 ) );
			Language call( ".model.matrix.clean_GEH", form, loop);	
			return call.eval( env );
	}

	// Get the matrix for the null model
	RcppGSL::matrix<double> get_model_matrix_null(){

			Formula form( update_expression( 1 ) );
			Language call( ".model.matrix.null_matrix_GEH", form, loop);	
			return call.eval( env );
	}

	RcppGSL::vector<double> get_response(){		
		Formula form( update_expression( 1 ) );
		Language call( ".mm_GEH", form);
		return call.eval( env ); 
	}

	RcppGSL::matrix<double> get_response_m(){		
		Formula form( update_expression( 1 ) );
		Language call( ".mm_GEH", form);
		return call.eval( env ); 
	}

	vector<string> get_terms(){		
		Formula form( update_expression( 1 ) );
		Language call( ".colnames_GEH", form);
		CharacterVector terms = call.eval( env ); 

		vector<string> ret;		
		for(int i=0; i<terms.size(); i++){
			ret.push_back( as<string>( terms(i) ) );
		}

		return ret;
	}

	vector<int> get_loop_terms(){		
		Formula form( update_expression( 1 ) );
		Language call( ".loop_terms_GEH", form, loop);
		NumericVector loopIndex = call.eval( env ); 

		vector<int> ret;		
		for(int i=0; i<loopIndex.size(); i++){
			ret.push_back( loopIndex(i) );
		}

		return ret;
	}

	string get_expression(){
		return expression;
	}

};

RcppExport SEXP R_fastGPR_batch( SEXP expression_, SEXP loop_, SEXP lcv_, SEXP nc_, SEXP env_, SEXP terms_, SEXP X_loop_, SEXP EigenVectors_, SEXP EigenValues_, SEXP rank_, SEXP delta_, SEXP reEstimateDelta_, SEXP nthreads_){

	try {
		CharacterVector expression( expression_ );
		string loop = as<string>( loop_ );
		string lcv = as<string>( lcv_ );
		int nc_total = as<int>( nc_ );
		Environment env( env_ );	
		std::vector<int> terms = as<std::vector<int> >( terms_ ); 
		RcppGSL::matrix<double> eigenVectors = EigenVectors_;
		RcppGSL::vector<double> eigenValues = EigenValues_; 
		int rank = as<int>( rank_ );
		double delta_global = as<double>( delta_ );
		bool reEstimateDelta = Rcpp::as<int>( reEstimateDelta_ );
		int nthreads = as<int>( nthreads_ );

		// Import design matrix
		// 	if it is a not numeric (i.e. it is a data.frame) convert it to a matrix
		//NumericMatrix X_loop( X_loop_ );

		NumericMatrix X_loop;

		if( TYPEOF(X_loop_) != REALSXP ){
			X_loop = asMatrix( X_loop_ );
		}else{
			X_loop = NumericMatrix( X_loop_ );
		}
			
		// Set threads to 1
		omp_set_num_threads( nthreads );
		// Intel paralellism
		#ifdef INTEL
		mkl_set_num_threads( 1 );
		#endif
		// disable nested OpenMP parallelism
		omp_set_nested(0);

		// initialize eigenDecomp
		eigenDecomp *eigen = new eigenDecomp( eigenVectors, eigenValues, false, rank);
	
		eigenVectors.free(); 
		eigenValues.free();

		// Process exression, loop and lcv
		Rexpression expr( as<string>( expression ), loop, lcv, env );

		std::vector<double> pValues( X_loop.ncol() );			

		RcppGSL::vector<double> y = expr.get_response(); 

		// Check that sizes of and X_loop match
		if( y->size != X_loop.nrow() ){
			y.free();
		
			throw "Dimensions of response and design matrix do not match\n";
		}

		// X_clean = model.matrix.default( y ~ sex:One + age )
		// Replace marker with 1's so that design matrix for marker j can be created by multiplcation
		RcppGSL::matrix<double> X_clean = expr.get_model_matrix_clean(); 

		// If trying to access index beyond range
		// Exit gracefully
		if( terms[which_max(terms)] >= X_clean->size2 ){
			y.free();
			X_clean.free();
			delete eigen;
		
			throw "Element in \"terms\" is out of range";
		}

		gsl_matrix *Xu_clean = gsl_matrix_alloc( eigen->t_U->size1, X_clean->size2 );

		// Xu_clean = crossprod( decomp$vectors, X_clean)
		gsl_blas_dgemm( CblasNoTrans, CblasNoTrans, 1.0, eigen->t_U, X_clean, 0.0, Xu_clean );

		// get indeces of columns that depend on X[,j]
		vector<int> loopIndex = expr.get_loop_terms();

		//vector<string> colNames = expr.get_terms();		
		
		long batch_size = MAX( 1, (long) X_loop.ncol()/100.0 );

		if( R_IsNA( delta_global ) && ! reEstimateDelta ){
				
				RcppGSL::matrix<double> X_null = expr.get_model_matrix_null(); 			

				LMM_params *params_master = new LMM_params( y, X_null, eigen, rank );
				params_master->set_delta_grid( exp(-10), exp(10), 100);
				params_master->hypothesisTest = false;

				double log_L, sig_g, sig_e;

				// Estimate delta
				LMM_fit_mle( params_master, &log_L, &sig_g, &sig_e, NULL, NULL);
				delta_global = sig_e / sig_g;

				X_null.free();
				delete params_master;
		}

		#pragma omp parallel
		{
			// Variables local to each thread
			gsl_matrix *Sigma_sub = gsl_matrix_alloc( terms.size(), terms.size() );			
			gsl_vector *beta_sub = gsl_vector_alloc( terms.size() );

			LMM_params *params = new LMM_params( y, eigen, rank );
			params->set_delta_grid( exp(-10), exp(10), 100);
			params->hypothesisTest = false;

			gsl_matrix *X = gsl_matrix_alloc( X_clean->size1, X_clean->size2 );
			gsl_matrix *Xu = gsl_matrix_alloc( eigen->t_U->size1, X_clean->size2 );

			// Initialize params
			params->update_X( X, eigen );

			gsl_vector_view col_view, col_view2;

			double log_L, sig_g, sig_e;
			double stat, delta;

			gsl_vector *marker_j = gsl_vector_alloc( y->size ); 

			#pragma omp for schedule(static, batch_size)
			for(int j=0; j<X_loop.ncol(); j++){		

				// Check if X has an NA value 
				// If it does, set the p-value to 
				if( is_true( any( is_na( X_loop(_,j) ) ) )){
					pValues[j] = NA_REAL;
					continue;
				}

				// Copy feature data from X_loop to marker_j in a way that does not use
				// 	Rcpp or RcppGSL objects.  This avoid the Rcpp thread-safety issues.
				// In summary, Rcpp objects, even if they declared in their own thread, 
				// 	CANNOT be written to safely
				// Here I do the copying manually, instead of using the nice Rcpp methods
				for(int k=0; k<marker_j->size; k++){
					gsl_vector_set(marker_j, k, X_loop(k,j));
				}				

				// Copy clean version of matrices to active variables
				gsl_matrix_memcpy( X, X_clean );
				gsl_matrix_memcpy( Xu, Xu_clean );	

				for(int k=0; k<loopIndex.size(); k++){			
					// X_design[,loopIndex[k]] = X_design[,loopIndex[k]] * marker_j
					col_view = gsl_matrix_column( (gsl_matrix*)(X), loopIndex[k] );
					gsl_vector_mul( &col_view.vector, marker_j );

					// Xu[,loopIndex[k]] = crossprod( U, X_design[,loopIndex[k]] * marker_j)
					col_view2 = gsl_matrix_column( (gsl_matrix*)(Xu), loopIndex[k] );
					gsl_blas_dgemv( CblasNoTrans, 1.0, eigen->t_U, &col_view.vector, 0.0, &col_view2.vector );
				}

				//params->update_X( X, eigen );
				params->update_Xu( X, Xu, rank );						

				if( reEstimateDelta ){
					// Estimate delta
					LMM_fit_mle( params, &log_L, &sig_g, &sig_e, NULL, NULL);
				
					delta = sig_e / sig_g;
				}else{
					// Use pre-specified delta value
					LMM_fit_fixed_delta( params, delta_global, &log_L, rank, true, NULL, NULL );
					delta = delta_global;
				}
		
				// gsl_matrix *Sigma = LMM_coeff_covariance( params, delta);
				gsl_matrix_scale( params->A, params->sig_g);
				char uplo = 'U';
				gsl_matrix_triangular_to_full(params->A, uplo);		
					
				gsl_matrix_sub_row_col( params->A, terms, Sigma_sub);
				gsl_vector_subset( params->beta, terms, beta_sub );
				gsl_lapack_chol_invert( Sigma_sub );			

				// tcrossprod(fit$coefficients[terms], solve(fit$Sigma[terms,terms])) %*% fit$coefficients[terms]
				stat = gsl_matrix_quadratic_form_sym( Sigma_sub, beta_sub );

				// pchisq( stat, df, lower.tail=FALSE)
				pValues[j] = gsl_cdf_chisq_Q( stat, terms.size() );
		
			} // END for
						
			gsl_matrix_free( Sigma_sub );
			gsl_vector_free( beta_sub );
			delete params;			
			gsl_matrix_free( X );
			gsl_matrix_free( Xu );
			gsl_vector_free( marker_j );

		} // End parallel

		delete eigen;
		y.free();
		X_clean.free();
		gsl_matrix_free(Xu_clean);

		return wrap( pValues ); // return the result list to R

	} catch( const char* msg ){
		 Rcpp::Rcout << "lrgpr error: " << msg << endl;
	}catch( std::exception &ex ) {
		Rcpp::Rcout << "exception:  " << endl;
		forward_exception_to_r( ex );
	} catch(...) { 
		::Rf_error( "c++ exception (unknown reason)" ); 
	}
	return R_NilValue; // -Wall
}

RcppExport SEXP R_glm_batch( SEXP expression_, SEXP loop_, SEXP lcv_, SEXP nc_, SEXP env_, SEXP terms_, SEXP X_loop_, SEXP nthreads_, SEXP useIdentityLink_, SEXP univariate_, SEXP multivariate_){

	try {
		CharacterVector expression( expression_ );
		string loop = as<string>( loop_ );
		string lcv = as<string>( lcv_ );
		int nc_total = as<int>( nc_ );
		Environment env( env_ );	
		std::vector<int> terms = as<std::vector<int> >( terms_ ); 
		int nthreads = as<int>( nthreads_ );
		bool useIdentityLink = as<int>( useIdentityLink_ );
		bool univariate = as<int>( univariate_ );
		bool multivariate = as<int>( multivariate_ );
			
		regressionType regressType = useIdentityLink ? LINEAR : LOGISTIC;

		// Import design matrix
		// 	if it is a not numeric (i.e. it is a data.frame) convert it to a matrix
		//NumericMatrix X_loop( X_loop_ );

		NumericMatrix X_loop;

		if( TYPEOF(X_loop_) != REALSXP ){
			X_loop = asMatrix( X_loop_ );
		}else{
			X_loop = NumericMatrix( X_loop_ );
		}

		// Set threads to 1
		omp_set_num_threads( nthreads );
		// Intel paralellism
		#ifdef INTEL
		mkl_set_num_threads( 1 );
		#endif
		// disable nested OpenMP parallelism
		omp_set_nested(0);

		// Process exression, loop and lcv
		Rexpression expr( as<string>( expression ), loop, lcv, env );			

		RcppGSL::matrix<double> Y = expr.get_response_m(); 

		int n_markers = X_loop.ncol();
		int n_indivs = Y->size1;
		int n_pheno = Y->size2;

		// Check that sizes of and X_loop match
		if( n_indivs != X_loop.nrow() ){
			Y.free();
		
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

		//gsl_matrix_print(X_clean);

		// If trying to access index beyond range
		// Exit gracefully
		if( terms[which_max(terms)] >= X_clean->size2 ){
			Y.free();
			X_clean.free();
		
			throw "Element in \"terms\" is out of range";
		}

		// get indeces of columns that depend on X[,j]
		vector<int> loopIndex = expr.get_loop_terms();
		//vector<string> colNames = expr.get_terms();		
		
		long batch_size = MAX( 1, (long) X_loop.ncol()/100.0 );

		#pragma omp parallel
		{
			// Variables local to each thread
			gsl_matrix *X = gsl_matrix_alloc( X_clean->size1, X_clean->size2 );

			gsl_vector_view col_view, col_view_y;

			gsl_vector *marker_j = gsl_vector_alloc( X->size1 ); 

			GLM_MV_workspace *workmv = GLM_MV_workspace_alloc(Y->size1, X->size2, Y->size2, true, false);

			double Hotelling, Pillai;

			#pragma omp for schedule(static, batch_size)
			for(int j=0; j<X_loop.ncol(); j++){		

				// Check if X has an NA value 
				// If it does, set the p-value to 
				if( is_true( any( is_na( X_loop(_,j) ) ) )){
					if( univariate ){
						for(int k=0; k<n_pheno; k++){
							pValues(j,k) = NA_REAL;
						}
					}

					if( multivariate ){  
						pValues_multivariate(j,0) = NA_REAL;
						pValues_multivariate(j,1) = NA_REAL;
					}

					continue;
				}

				// Copy feature data from X_loop to marker_j in a way that does not use
				// 	Rcpp or RcppGSL objects.  This avoid the Rcpp thread-safety issues.
				// In summary, Rcpp objects, even if they declared in their own thread, 
				// 	CANNOT be written to safely
				// Here I do the copying manually, instead of using the nice Rcpp methods
				for(int k=0; k<marker_j->size; k++){
					gsl_vector_set(marker_j, k, X_loop(k,j));
				}				

				// Copy clean version of matrices to active variables
				gsl_matrix_memcpy( X, X_clean );

				for(int k=0; k<loopIndex.size(); k++){			
					// X_design[,loopIndex[k]] = X_design[,loopIndex[k]] * marker_j
					col_view = gsl_matrix_column( (gsl_matrix*)(X), loopIndex[k] );
					gsl_vector_mul( &col_view.vector, marker_j );
				}					

				//print_vector(loopIndex);

				//gsl_vector_print(marker_j);
				//gsl_matrix_print(X_clean);

				// Evaluate regression
				GLM_regression( Y, X, regressType, workmv);

				if( multivariate ){
					GLM_HotellingPillai_test( X, workmv, terms, &Hotelling, &Pillai );
					pValues_multivariate(j,0) = Hotelling;
					pValues_multivariate(j,1) = Pillai;
				}


				if( univariate ){
					// Perform Wald test
					gsl_vector *pvals = GLM_wald_test( workmv, terms ) ; 

	 				for(int k=0; k<n_pheno; k++){
						pValues(j,k) = gsl_vector_get(pvals, k);
					}

					gsl_vector_free( pvals );
				}
		
			} // END for
						
			gsl_matrix_free( X );
			gsl_vector_free( marker_j );
			GLM_MV_workspace_free( workmv );

		} // End parallel

		Y.free();
		X_clean.free();

		//return wrap( pValues ); // return the result list to R
		//return ( pValues ); // return the result list to R

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


extern "C" SEXP R_LMM_rankSearch( SEXP sY, SEXP sX, SEXP sEigenVectors, SEXP sEigenValues, SEXP smaxRank, SEXP schisq, SEXP squiet){

	try {
	
		RcppGSL::matrix<double> X = sX; 
		RcppGSL::vector<double> y = sY; 	
		RcppGSL::matrix<double> eigenVectors = sEigenVectors; 	
		RcppGSL::vector<double> eigenValues = sEigenValues; 	
		bool useChisqLikelihood = ( Rcpp::as<int>(schisq) != 0 );

		int maxRank = INTEGER(smaxRank)[0];
		bool quiet = (bool) INTEGER(squiet)[0];

		// initialize eigenDecomp
		eigenDecomp *eigen = new eigenDecomp( eigenVectors, eigenValues, false, y->size);

		rankSearch rSeachResults = LMM_rankSearch( y, X, eigen, maxRank, useChisqLikelihood, quiet, true);

		gsl_matrix *coefficients = rSeachResults.get_coefficients();
		gsl_matrix *pValues = rSeachResults.get_pValues();

		RcppGSL::matrix<double> pValuesReturn = pValues;
		RcppGSL::matrix<double> coefficientsReturn = coefficients;

		Rcpp::List res = Rcpp::List::create(Rcpp::Named("rank") 	= rSeachResults.get_rank(),
											Rcpp::Named("df") 		= rSeachResults.get_df(),
											Rcpp::Named("delta") 	= rSeachResults.get_delta(),
											Rcpp::Named("sig_e") 	= rSeachResults.get_sig_e(),
											Rcpp::Named("logLik") 	= rSeachResults.get_logLik(),
											Rcpp::Named("SSE") 	    = rSeachResults.get_SSE(),
											Rcpp::Named("pValues") 	= pValuesReturn,
											Rcpp::Named("coefficients") 	= coefficientsReturn );
		
		X.free(); 
		y.free();
		eigenVectors.free();
		eigenValues.free();
		pValuesReturn.free();
		coefficientsReturn.free();
	
		delete eigen;

		return res; // return the result list to R


	} catch( std::exception &ex ) {
		forward_exception_to_r( ex );
	} catch(...) { 
		::Rf_error( "c++ exception (unknown reason)" ); 
	}
	return R_NilValue; // -Wall
}



extern "C" SEXP R_LMM_univariate_df( SEXP sY, SEXP sX, SEXP sEigenVectors, SEXP sEigenValues, SEXP schisq, SEXP quiet_){

	try {
	
		RcppGSL::matrix<double> X = sX; 
		RcppGSL::vector<double> y = sY; 	
		RcppGSL::matrix<double> eigenVectors = sEigenVectors; 	
		RcppGSL::vector<double> eigenValues = sEigenValues; 	
		bool useChisqLikelihood = ( Rcpp::as<int>(schisq) != 0 );
		bool quiet = ( Rcpp::as<int>(quiet_) != 0 );
		
		// initialize eigenDecomp
		// this copies eigenVectors and eigenValues by reference
		eigenDecomp *eigen = new eigenDecomp( eigenVectors, eigenValues, false, eigenValues->size);

		gsl_vector *df = LMM_univariate_df( y, X, eigen, useChisqLikelihood, quiet );

		eigenVectors.free();
		eigenValues.free();

		RcppGSL::vector<double> dfReturn = df;

		Rcpp::List res = Rcpp::List::create(Rcpp::Named("df") = dfReturn );
		
		X.free(); 
		y.free();
		dfReturn.free();
	
		delete eigen ;

		return res; // return the result list to R

	} catch( std::exception &ex ) {
		forward_exception_to_r( ex );
	} catch(...) { 
		::Rf_error( "c++ exception (unknown reason)" ); 
	}
	return R_NilValue; // -Wall
}


extern "C" SEXP R_crossprod_trace( SEXP sA, SEXP sB){

    try {

		RcppGSL::matrix<double> A = sA;
		RcppGSL::matrix<double> B = sB;

		double trace = gsl_matrix_product_trace( A, B);

		A.free();
		B.free();

		Rcpp::List res =  Rcpp::List::create(Rcpp::Named("trace") = trace);

		return res;

	} catch( std::exception &ex ) {
        forward_exception_to_r( ex );
    } catch(...) {
        ::Rf_error( "c++ exception (unknown reason)" );
    }
    return R_NilValue; // -Wall 
}

extern "C" SEXP R_ARD_search_targets( SEXP sY, SEXP sX, SEXP sEigenVectors, SEXP sEigenValues, SEXP sdfMax){

	try {

		RcppGSL::matrix<double> X = sX;
		RcppGSL::vector<double> y = sY;
		RcppGSL::matrix<double> U = sEigenVectors;
		RcppGSL::vector<double> eigenValues = sEigenValues;
		double dfMax = Rcpp::as<double>(sdfMax);

		// R = diag(lambda) %*% U
		gsl_matrix *R = gsl_matrix_alloc( U->size1, U->size2 );
		gsl_vector *singular_values = gsl_vector_alloc( eigenValues->size );

		gsl_vector_sqrt( eigenValues, singular_values);
		gsl_matrix_diagonal_multiply( U, singular_values, R, false);

		vector<double> logLikArray, dfArray, phiArray;

		// path search

		ARD_search_targets( y, X, R, eigenValues, U, 3, 10, dfMax, dfArray, phiArray, logLikArray );

		//gsl_vector *alpha = gsl_vector_alloc( y->size );

		//gsl_vector *Hii = LMM_get_hat_matrix_diag( delta, X, eigen, params );
		//gsl_vector *Y_hat = LMM_get_fitted_response( y, delta, X, eigen, params, alpha);

		/*RcppGSL::vector<double> beta = params->beta;
		RcppGSL::vector<double> pValuesReturn = pValues;
		RcppGSL::vector<double> sdReturn = sd;
		RcppGSL::vector<double> HiiReturn = Hii;
		RcppGSL::vector<double> Y_hatReturn = Y_hat;
		RcppGSL::vector<double> alphaReturn = alpha;*/

		/*Rcpp::NumericVector phiArrayReturn( phiArray );
		Rcpp::NumericVector dfArrayReturn = dfArray;
		Rcpp::NumericVector logLikArrayReturn = logLikArray;*/

		Rcpp::List res = Rcpp::List::create(Rcpp::Named("phi") 		= Rcpp::wrap( phiArray ),
											Rcpp::Named("df") 		= Rcpp::wrap( dfArray ),
											Rcpp::Named("logLik") 	= Rcpp::wrap( logLikArray ) );

		X.free();
		y.free();
		U.free();
		eigenValues.free();
		//beta.free(); beta owned by params_null and is free'd by the class destructor
		/*pValuesReturn.free();
		sdReturn.free();
		HiiReturn.free();
		Y_hatReturn.free();
		alphaReturn.free();*/

		return res; // return the result list to R


	} catch( std::exception &ex ) {
		forward_exception_to_r( ex );
	} catch(...) {
		::Rf_error( "c++ exception (unknown reason)" );
	}
	return R_NilValue; // -Wall
}

int count_nonzero( const gsl_vector *beta){
	int count = 0;

	for(int i=0; i<beta->size; i++){
		count += (gsl_vector_get(beta, i) != 0);
	}

	return count;
}

extern "C" SEXP R_GLM_penalized_regression( SEXP sY, SEXP sX, SEXP sX_cov, SEXP sn_features, SEXP sn_covariates, SEXP shp1, SEXP shp2, SEXP spenalty, SEXP sfamily, SEXP sweights, SEXP salpha, SEXP snlambda, SEXP slambda_min_ratio, SEXP sstandardize, SEXP sthresh, SEXP sdfmax, SEXP smaxit, SEXP smoreDense){

	try {

		RcppGSL::vector<double> y = sY;
		RcppGSL::matrix<double> X = sX;		
		RcppGSL::matrix<double> X_cov = sX_cov;
		int n_features = Rcpp::as<int>(sn_features);
		int n_covariates = Rcpp::as<int>(sn_covariates);
		Rcpp::NumericVector lambda_array(shp1);
		Rcpp::NumericVector hp2_array(shp2);
		string penalty = Rcpp::as<string>(spenalty);
		string family = Rcpp::as<string>(sfamily);
		RcppGSL::vector<double> weights = sweights;
		double alpha = Rcpp::as<double>(salpha);
		int nlambda = Rcpp::as<int>(snlambda);
		double lambda_min_ratio = Rcpp::as<double>(slambda_min_ratio);
		bool standardize = ( Rcpp::as<int>(sstandardize) != 0 );
		double thresh = Rcpp::as<double>(sthresh);
		int dfmax = Rcpp::as<int>(sdfmax);
		int maxit = Rcpp::as<int>(smaxit);
		bool moreDense = Rcpp::as<double>(smoreDense);

		int n_indivs = y->size;

		//gsl_matrix *t_X = gsl_matrix_attach_array( REAL(sX), n_features, n_indivs );

		gsl_matrix *t_X = gsl_matrix_alloc( X->size2, X->size1);
		gsl_matrix_transpose_memcpy( t_X, X);
		
		regressionType regression;
		if( family == "gaussian" || family == "Gaussian"){
			regression = LINEAR;
		}else if( family == "binomial" || family == "Binomial" || family == "logistic"){
			regression = LOGISTIC;
		}

		penaltyType penType;
		if( penalty == "Lasso" || penalty == "LASSO" ){
			penType = LASSO;
		}else if( penalty == "MCP" ){ 
			penType = MCP;
		}else if( penalty == "SCAD" ){ 
			penType = SCAD;
		}else if( penalty == "LOG" ){ 
			penType = LOG;
		}

		double log_L, max_unselected_beta;
		int n_iter;

		gsl_vector *feature_scales, *covariate_scales, *feature_weights;
		gsl_vector *beta, *beta_cov ;

		GLM_workspace *workspace = GLM_workspace_alloc(n_indivs);
		gsl_permutation *permutation = gsl_permutation_calloc( n_features );

		gsl_vector *feature_scales_orig, *feature_centers_orig;

		if( standardize ){
			feature_scales_orig = get_row_scales( t_X );
			feature_centers_orig = get_row_centers( t_X );

			standardize_rows( t_X, false);

			feature_scales = gsl_vector_alloc(  n_features );
			gsl_vector_set_all( feature_scales, n_indivs );
		}else{
			feature_scales = get_row_scales( t_X );
		}

		covariate_scales = get_column_scales( X_cov );

		feature_weights = gsl_vector_alloc(  n_features);
		gsl_vector_set_all( feature_weights, 1 );

		GLM_penRegInfo info;

		Rcpp::NumericVector loglikArray( nlambda );
		Rcpp::NumericVector n_iter_array( nlambda );
		Rcpp::NumericVector n_active( nlambda );

		Rcpp::NumericMatrix betaM( nlambda, n_features);
		Rcpp::NumericMatrix beta_covM( nlambda, X_cov->size2);

		// Alloc local memory for coefficients
		beta = gsl_vector_calloc(  n_features );
		beta_cov = gsl_vector_calloc( X_cov->size2 );

		bool initialized = false;
		int last_valid_active_index = 0, diff_active;

		///////////////////////
		// Initialize lambda //
		///////////////////////

		bool autoLearnLambda = isinf(lambda_array(0));

		if( autoLearnLambda ){

			GLM_penalized_regression( y, t_X, X_cov, beta, beta_cov, regression, penType, lambda_array(0), hp2_array(0), 					feature_scales, covariate_scales, feature_weights, workspace, permutation, maxit, thresh, &info, initialized);

			initialized = true;

			// Set initial lambda so that first beta contains all zeros
			lambda_array(0) = info.max_unselected_beta;
		}

		for(unsigned int i=0; i<nlambda; i++){	
				
			loglikArray(i) = GLM_penalized_regression( y, t_X, X_cov, beta, beta_cov, regression, penType, lambda_array(i), 					hp2_array(i), feature_scales, covariate_scales, feature_weights, workspace, permutation, maxit, thresh, 					&info, initialized);

			n_active(i) = count_nonzero(beta);			
			n_iter_array(i) = info.n_iter;

			// Copy results to returnable matrix
			for(unsigned int j=0; j<beta->size; j++) betaM(i,j) = gsl_vector_get(beta, j);
			for(unsigned int j=0; j<beta_cov->size; j++) beta_covM(i,j) = gsl_vector_get(beta_cov, j);

			//cout << i << " " << n_active(i) << " " << last_valid_active_index << " " << lambda_array(i) << endl;

			if( autoLearnLambda && i+1<nlambda ){

				diff_active = n_active(i) - n_active( last_valid_active_index );

				// if model size has been incremented by more than 1			
				if( diff_active > 1){	

					// increase the lambda value so produce an intermediate number of active features
					lambda_array(i+1) = (lambda_array(i) + lambda_array(last_valid_active_index)) / 2.0;

					// set beta equal to the coefficients from lambda = lambda_array(last_valid_active_index)
					// Especially non-convex penalties, this makes sure that the coefficients warms-start from lambda = zero
					for(unsigned int j=0; j<beta->size; j++) 
						gsl_vector_set(beta, j, betaM(last_valid_active_index,j) );
					for(unsigned int j=0; j<beta_cov->size; j++) 
						gsl_vector_set(beta_cov, j, beta_covM(last_valid_active_index,j) );	
			
				}else{
					if( diff_active == 1) last_valid_active_index = i;
					//if( moreDense )	lambda_array(i+1) = (info.mid_unselected_beta + 99*info.max_unselected_beta) / 100.0;
					//else 			lambda_array(i+1) = info.max_unselected_beta;
					//lambda_array(i+1) = 0.99*info.max_unselected_beta;

					lambda_array(i+1) = (info.mid_unselected_beta + info.max_unselected_beta)/2;

					if( lambda_array(i) - lambda_array(i+1) < 1e-7 )
					lambda_array(i+1) = lambda_array(i) - (lambda_array(i-1) - lambda_array(i));

					//if( lambda_array(i) - lambda_array(i+1) < 1e-5 ) lambda_array(i+1) = lambda_array(i)*.99;
				}
			}

			initialized = true;

			// Check degrees of freedom 
			if( n_active(i) > dfmax) break;
		}

		// free local memory
		gsl_vector_free( beta );
		gsl_vector_free( beta_cov );
	
		Rcpp::List res;

		if( standardize ){

			RcppGSL::vector<double> centers = feature_centers_orig;
			RcppGSL::vector<double> scales = feature_scales_orig;

			res = Rcpp::List::create(Rcpp::Named("beta") 	= betaM,
									Rcpp::Named("beta_cov") = beta_covM,
									Rcpp::Named("iter") 	= n_iter_array, 
									Rcpp::Named("lambda") 	= lambda_array,
									Rcpp::Named("hp2") 		= hp2_array,
									Rcpp::Named("center") 	= centers,
									Rcpp::Named("scale") 	= scales,
									Rcpp::Named("logLik") 	= loglikArray );
	
			centers.free();
			scales.free();

		}else{
			res = Rcpp::List::create(Rcpp::Named("beta") 	= betaM,
									Rcpp::Named("beta_cov") = beta_covM,
									Rcpp::Named("iter") 	= n_iter_array, 
									Rcpp::Named("lambda") 	= lambda_array,
									Rcpp::Named("hp2") 		= hp2_array,
									Rcpp::Named("logLik") 	= loglikArray );
		}

		y.free();
		gsl_matrix_free(t_X);
		X_cov.free();		
		X.free();

		return res; // return the result list to R


	} catch( std::exception &ex ) {
		forward_exception_to_r( ex );
	} catch(...) {
		::Rf_error( "c++ exception (unknown reason)" );
	}
	return R_NilValue; // -Wall
}




extern "C" SEXP R_LMM_penalized_regression( SEXP sY, SEXP sX, SEXP sX_cov, SEXP sn_features, SEXP sn_covariates, SEXP shp1, SEXP shp2, SEXP spenalty, SEXP sfamily, SEXP sweights, SEXP salpha, SEXP snlambda, SEXP slambda_min_ratio, SEXP sstandardize, SEXP sthresh, SEXP sdfmax, SEXP smaxit, SEXP smoreDense, SEXP seigenVectors, SEXP seigenValues){

	try {

		RcppGSL::vector<double> y = sY;
		RcppGSL::matrix<double> X = sX;		
		RcppGSL::matrix<double> X_cov = sX_cov;
		int n_features = Rcpp::as<int>(sn_features);
		int n_covariates = Rcpp::as<int>(sn_covariates);
		Rcpp::NumericVector lambda_array(shp1);
		Rcpp::NumericVector hp2_array(shp2);
		string penalty = Rcpp::as<string>(spenalty);
		string family = Rcpp::as<string>(sfamily);
		RcppGSL::vector<double> weights = sweights;
		double alpha = Rcpp::as<double>(salpha);
		int nlambda = Rcpp::as<int>(snlambda);
		double lambda_min_ratio = Rcpp::as<double>(slambda_min_ratio);
		bool standardize = ( Rcpp::as<int>(sstandardize) != 0 );
		double thresh = Rcpp::as<double>(sthresh);
		int dfmax = Rcpp::as<int>(sdfmax);
		int maxit = Rcpp::as<int>(smaxit);
		bool moreDense = Rcpp::as<double>(smoreDense);

		RcppGSL::matrix<double> eigenVectors = seigenVectors;	
		RcppGSL::vector<double> eigenValues = seigenValues;

		eigenDecomp *eigen = new eigenDecomp( eigenVectors, eigenValues, false, y->size);

		int n_indivs = y->size;

		//gsl_matrix *t_X = gsl_matrix_attach_array( REAL(sX), n_features, n_indivs );

		gsl_matrix *t_X = gsl_matrix_alloc( X->size2, X->size1);
		gsl_matrix_transpose_memcpy( t_X, X);
		
		regressionType regression;
		if( family == "gaussian" || family == "Gaussian"){
			regression = LINEAR;
		}else if( family == "binomial" || family == "Binomial" || family == "logistic"){
			regression = LOGISTIC;
		}

		penaltyType penType;
		if( penalty == "Lasso" || penalty == "LASSO" ){
			penType = LASSO;
		}else if( penalty == "MCP" ){ 
			penType = MCP;
		}else if( penalty == "SCAD" ){ 
			penType = SCAD;
		}else if( penalty == "LOG" ){ 
			penType = LOG;
		}

		double log_L, max_unselected_beta;
		int n_iter;

		gsl_vector *feature_scales, *covariate_scales, *feature_weights;
		gsl_vector *beta;

		GLM_workspace *workspace = GLM_workspace_alloc(n_indivs);
		gsl_permutation *permutation = gsl_permutation_calloc( n_features );

		gsl_vector *feature_scales_orig, *feature_centers_orig;

		if( standardize ){
			feature_scales_orig = get_row_scales( t_X );
			feature_centers_orig = get_row_centers( t_X );

			standardize_rows( t_X, false);

			feature_scales = gsl_vector_alloc(  n_features );
			gsl_vector_set_all( feature_scales, n_indivs );
		}else{
			feature_scales = get_row_scales( t_X );
		}

		covariate_scales = get_column_scales( X_cov );

		feature_weights = gsl_vector_alloc(  n_features);
		gsl_vector_set_all( feature_weights, 1 );

		GLM_penRegInfo info;

		Rcpp::NumericVector loglikArray( nlambda );
		Rcpp::NumericVector n_iter_array( nlambda );
		Rcpp::NumericVector deltaArray( nlambda );

		Rcpp::NumericMatrix betaM( nlambda, n_features);
		Rcpp::NumericMatrix beta_covM( nlambda, X_cov->size2);

		// Alloc local memory for coefficients
		beta = gsl_vector_calloc(  n_features );
		bool initialized = false;

		gsl_matrix *t_Xu = gsl_matrix_alloc( t_X->size1, t_X->size2 );
		gsl_blas_dgemm( CblasNoTrans, CblasNoTrans, 1.0, t_X, eigen->U, 0.0, t_Xu );
	
		// Initialize LMM
		LMM_params params( y, X_cov, eigen, y->size );

		///////////////////////
		// Initialize lambda //
		///////////////////////

		bool autoLearnLambda = isinf(lambda_array(0));

		if( autoLearnLambda ){

			LMM_penalized_regression( &params, t_Xu, beta, regression, penType, lambda_array(0), hp2_array(0), 						feature_scales, workspace, permutation, maxit, &info, initialized);

			initialized = true;

			// Set initial lambda so that first beta contains all zeros
			lambda_array(0) = info.max_unselected_beta;
			deltaArray(0)	= info.delta;
		}

		double delta_ratio;

		double weight = 20.0;

		for(unsigned int i=0; i<nlambda; i++){	
			//cout << "\r"<< lambda_array(i) << "\t" << i << " / " << nlambda << "           " << flush;

			loglikArray(i) = LMM_penalized_regression( &params, t_Xu, beta, regression, penType, lambda_array(i), hp2_array(i), 						feature_scales, workspace, permutation, maxit, &info, initialized);

			if( autoLearnLambda && i+1<nlambda ){
				lambda_array(i+1) = (info.mid_unselected_beta + (weight-1)*lambda_array(i)) / weight;			
	
				if( lambda_array(i) - lambda_array(i+1) < 1e-6 ) lambda_array(i+1) = lambda_array(i)*.995;
				if( lambda_array(i+1) >= lambda_array(i) ) lambda_array(i+1) = lambda_array(i)*.995;
			}

			initialized = true;

			// Copy results to returnable matrix
			for(unsigned int j=0; j<beta->size; j++) betaM(i,j) = gsl_vector_get(beta, j);
			for(unsigned int j=0; j<params.beta->size; j++) beta_covM(i,j) = gsl_vector_get(params.beta, j);

			n_iter_array(i) = info.n_iter;
			deltaArray(i)	= info.delta;

			if( i > 0 ){
				delta_ratio = deltaArray(i) / deltaArray(i-1);

				if( delta_ratio > 1.02 || delta_ratio < 1.0/1.02 ){
					weight = 1000.0;
				}else if( delta_ratio > 1.06 || delta_ratio < 1.0/1.06 ){
					weight = 10000.0;	
				}else{
					weight = 20.0;
				}
			}
	
			// Check degrees of freedom 
			if( info.n_beta_active > dfmax){
				//cout << info.n_beta_active << " " << dfmax << endl;
				break;
			}
		}

		// free local memory
		gsl_vector_free( beta );
		Rcpp::List res;

		if( standardize ){

			RcppGSL::vector<double> centers = feature_centers_orig;
			RcppGSL::vector<double> scales = feature_scales_orig;

			res = Rcpp::List::create(Rcpp::Named("beta") 	= betaM,
									Rcpp::Named("beta_cov") = beta_covM,
									Rcpp::Named("delta") 	= deltaArray, 
									Rcpp::Named("iter") 	= n_iter_array, 
									Rcpp::Named("lambda") 	= lambda_array,
									Rcpp::Named("hp2") 		= hp2_array,
									Rcpp::Named("center") 	= centers,
									Rcpp::Named("scale") 	= scales,
									Rcpp::Named("logLik") 	= loglikArray );
	
			centers.free();
			scales.free();

		}else{
			res = Rcpp::List::create(Rcpp::Named("beta") 	= betaM,
									Rcpp::Named("beta_cov") = beta_covM,
									Rcpp::Named("delta") 	= deltaArray, 
									Rcpp::Named("iter") 	= n_iter_array, 
									Rcpp::Named("lambda") 	= lambda_array,
									Rcpp::Named("hp2") 		= hp2_array,
									Rcpp::Named("logLik") 	= loglikArray );
		}

		y.free();
		gsl_matrix_free(t_X);
		X_cov.free();		
		X.free();

		return res; // return the result list to R


	} catch( std::exception &ex ) {
		forward_exception_to_r( ex );
	} catch(...) {
		::Rf_error( "c++ exception (unknown reason)" );
	}
	return R_NilValue; // -Wall
}





