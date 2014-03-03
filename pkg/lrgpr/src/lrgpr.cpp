
/*
 * lrgpr.cpp
 *
 *  Created on: November, 12. 2013
 *      Author: gh258
 */

#include "lrgpr.h"

#include <iostream>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <omp.h>

#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_cdf.h>
 #include <gsl/gsl_randist.h>

#include "gsl_additions.h"
#include "gsl_lapack.h"
#include "misc_functions.h"

#ifdef INTEL
#include "mkl_service.h"
#endif

using namespace std;

///////////
// LRGPR //
///////////

LRGPR::LRGPR( const gsl_vector *Y_, const gsl_matrix *U_, const gsl_vector *eigenValues, const int X_ncol_, const int W_ncol_){

	params = new LRGPR_params( Y_, U_, eigenValues, X_ncol_, W_ncol_);

	delta_grid = gsl_vector_seq( -10, 10, 100 );

	params->breakDueToSingularMatrix = false;
}

LRGPR::~LRGPR(){
	gsl_vector_free( delta_grid );	

	delete params;

}

void LRGPR::update_X( const gsl_matrix *X_in){

	params->update_X( X_in );
}


void LRGPR::update_Xu( const gsl_matrix *X_in, const gsl_matrix *Xu_in ){

	params->update_Xu( X_in, Xu_in );
}

void LRGPR::update_Y( const gsl_vector *Y_in ){

	params->update_Y( Y_in );	
}

void LRGPR::update_Wtilde( const gsl_matrix *W_til_ ){

	params->update_Wtilde( W_til_ );	
}


void LRGPR::set_delta_grid( const double left, const double right, const int n_intervals){

	gsl_vector_free( delta_grid );

	delta_grid = gsl_vector_seq( log(left), log(right), n_intervals);
}

void LRGPR::fit_mle( double *log_L, double *sig_g, double *sig_e ){

	params->breakDueToSingularMatrix = false;

	// minimization
	int status, iter, max_iter = 50;
	gsl_min_fminimizer *gsl_minimizer;

	// Assigned function and void struct pointer to gsl_function F
	gsl_function F;
	F.function = _LRGPR_fxn;
	F.params = this;

	// initialize minimizer
	gsl_minimizer = gsl_min_fminimizer_alloc( gsl_min_fminimizer_brent );

	double left, right, f_left, f_right, f_middle, middle, fraction;

	vector<double> log_L_values, delta_values;

	// for each interval in grid
	for( unsigned int i=1; i<this->delta_grid->size; i++){

		// evaluate the function at the bounds of the interval
		left = exp( gsl_vector_get( this->delta_grid, i-1 ) );
		right =  exp( gsl_vector_get( this->delta_grid, i ) );

		f_left = _LRGPR_fxn( left, this );
		f_right = _LRGPR_fxn( right, this );

		// Since the initial guess for the minimum must satisfy left < middle < right and f(left) > middle < f(right)
		// set the initial guess between the two values but towards the end with the smaller f(.) value
		// No all intervals will contain a maximum, so the second condition for middle will often not be met
		fraction = (f_left > f_right ) ? .95 : 0.05;

		middle = (right - left) * fraction + left;

		// save evaluations at interval since the maximum value in the interval may be on the boundary
		log_L_values.push_back( _LRGPR_fxn( left, this ) );
		delta_values.push_back( left );

		f_middle = _LRGPR_fxn( middle, this );

		if( params->breakDueToSingularMatrix ){
			break;
		}

		// If there is a maximum in the interval = If f(left) > middle < f(right)
		// this condition is not always satisfied: i.e. if the interval does not contain a maximum
		if( f_middle < f_left && f_middle < f_right ){

			// initialize minimizer with bounds and guess
			gsl_min_fminimizer_set (gsl_minimizer, &F, middle, left, right);

			iter = 0;
			// iterate minimizer until convergence to max iterations reached
			do{
			   	iter++;

			   	// update bounds and guess
			   	status = gsl_min_fminimizer_iterate( gsl_minimizer );

			  	// extract new bounds and guess
			  	middle = gsl_min_fminimizer_x_minimum( gsl_minimizer );
			   	left = gsl_min_fminimizer_x_lower( gsl_minimizer );
			   	right = gsl_min_fminimizer_x_upper( gsl_minimizer );

			   // cout << "iter: " << iter << endl;
			   // cout << left << " " << middle << " " << right << endl;

			   	// assess convergence based on desired error bounds
			 	status = gsl_min_test_interval (left, right, 0.00001, 0.0);

			   	if( status == GSL_SUCCESS  || iter == max_iter){

				   // save values from minimum
				   log_L_values.push_back( gsl_min_fminimizer_f_minimum( gsl_minimizer ) );
				   delta_values.push_back(  gsl_min_fminimizer_x_minimum( gsl_minimizer ) );
			   	}

			 } while (status == GSL_CONTINUE && iter < max_iter);
		}
	}
	gsl_min_fminimizer_free (gsl_minimizer);

	// evaluate function at last grid value
	left = exp( gsl_vector_get( this->delta_grid, this->delta_grid->size -1 ) );
	log_L_values.push_back( _LRGPR_fxn( left, this ) );
	delta_values.push_back( left );

	// get index of smallest functional value
	int i = which_min( log_L_values );

	// gsl uses a minimizer, but we want to maximize the log likelihood,
	// so take negative of smallest value
	*log_L = -log_L_values[i];

	// get delta value for largest log likelihood
	double delta = delta_values[i];

	//////////////////////////////
	// Evaluate sig_g and sig_e //
	//////////////////////////////

	*log_L = -1.0 * _LRGPR_fxn( delta, this );

	// Estimate beta, sig_g and delta
	params->delta = delta;
	*sig_g = params->sig_g;
	*sig_e = delta * (*sig_g);	
}

void LRGPR::fit_fixed_delta( const double delta, double *log_L, double *sig_g, double *sig_e ){

	*log_L = -1.0 * _LRGPR_fxn( delta, this );

	params->delta = delta;
	*sig_g = params->sig_g;
	*sig_e = delta * (*sig_g);
}

inline double LRGPR::logLikelihood( const double delta ){

	double sum1 = 0;

	for(int i=0; i<params->rank; i++){
		sum1 += log( gsl_vector_get( params->s, i ) + delta );
	}

	double n_indivs = params->n_indivs;

	// -n/2 * log(2*pi*env$obj$sig_g) - 1/2 * (sum( log(s + delta ) ) + (n-rank) * log(delta)) - n/2	
	double log_L = -n_indivs/2.0 * log(2*M_PI*params->sig_g) - 0.5 * (sum1 + (n_indivs - params->rank) * log(delta)) - n_indivs/2.0;

	if( params->W_ncol != 0){
		// log_L += - 1/2 * determinant( I_c - Q_WW(delta, env$obj))$modulus[1]
		log_L += -0.5 * params->Q_WW_1_logDet;
	}

	return log_L;
}

// env$obj$beta = solve( Omega_XX(delta, env$obj) ) %*% Omega_Xy(delta, env$obj)
void LRGPR::estimate_beta( const double delta ){

	this->Omega_XX( delta );
	this->Omega_Xy( delta );

	//gsl_matrix_print( params->Q_XX_value );
	//gsl_vector_print( params->Q_Xy_value );

	// Note: Invert multiply can be done faster with DSYSV or DSYSVX
	gsl_lapack_chol_invert( params->Q_XX_value );

	gsl_blas_dsymv(CblasLower, 1.0, params->Q_XX_value, params->Q_Xy_value, 0.0, params->beta);

	//gsl_matrix_print( params->Q_XX_value );
	//gsl_vector_print( params->Q_Xy_value );
	//gsl_vector_print( params->beta );
}

// crossprod(Xu, obj$inv_s_delta_Xu) + cp_X_low / delta
void LRGPR::Q_XX(const double delta){

	// crossprod(Xu, obj$inv_s_delta_Xu)
	gsl_blas_dgemm(CblasTrans,CblasNoTrans, 1.0, params->Xu, params->inv_s_delta_Xu, 0.0, params->Xu_inv_s_delta_Xu);

	gsl_matrix_memcpy( params->Q_XX_value, params->Xu_inv_s_delta_Xu);

	if( params->isLowRank){ 		
		// Q_XX_value += cp_X_low / delta
		gsl_matrix_daxpy( 1/delta,  params->cp_X_low, params->Q_XX_value );
	}
}

// crossprod(Xu, obj$inv_s_delta_Yu) + cp_X_low_Y_low / delta
inline void LRGPR::Q_Xy(const double delta){

	// crossprod(Xu, obj$inv_s_delta_Yu)
	gsl_blas_dgemv(CblasTrans,1.0, params->Xu, params->inv_s_delta_Yu, 0.0, params->Xu_inv_s_delta_Yu);

	gsl_vector_memcpy( params->Q_Xy_value, params->Xu_inv_s_delta_Yu );

	if( params->isLowRank){
		// Q_Xy_value += cp_X_low_Y_low / delta
		gsl_blas_daxpy( 1/delta,  params->cp_X_low_Y_low, params->Q_Xy_value );
	}
}

// crossprod(obj$ru, obj$inv_s_delta_ru) + (crossprod(obj$r)[1] - crossprod(obj$ru)[1])/ delta
inline void LRGPR::Q_rr(const double delta){

	// crossprod(obj$ru, obj$inv_s_delta_ru)
	gsl_blas_ddot(params->inv_s_delta_ru, params->ru, &(params->ru_inv_s_delta_ru) );

	params->Q_rr_value = params->ru_inv_s_delta_ru;

	if( params->isLowRank){
		// crossprod(obj$r)
		gsl_blas_ddot( params->r, params->r, &(params->SSE_r) );

		// crossprod(obj$ru)
		gsl_blas_ddot( params->ru, params->ru, &(params->SSE_ru) );

		// Combine temp values
		params->Q_rr_value += ( params->SSE_r - params->SSE_ru ) / delta;
	}
}

// t(crossprod(Xu, obj$inv_s_delta_Wu)) + cp_W_low_X_low / delta
inline void LRGPR::Q_XW(const double delta){

	// t(crossprod(Xu, obj$inv_s_delta_Wu))
	gsl_blas_dgemm(CblasTrans,CblasNoTrans, 1.0, params->Xu, params->inv_s_delta_Wu, 0.0, params->Xu_inv_s_delta_Wu);

	gsl_matrix_transpose_memcpy( params->Q_XW_value, params->Xu_inv_s_delta_Wu );

	if( params->isLowRank){
		// Q_XW_value += cp_W_low_X_low / delta
		gsl_matrix_daxpy( 1/delta,  params->cp_W_low_X_low, params->Q_XW_value );
	}
}

// crossprod(Wu, obj$inv_s_delta_Yu) + cp_W_low_Y_low / delta
inline void LRGPR::Q_Wy(const double delta){

	// crossprod(Wu, obj$inv_s_delta_Yu)
	gsl_blas_dgemv(CblasTrans, 1.0, params->Wu, params->inv_s_delta_Yu, 0.0, params->Wu_inv_s_delta_Yu);

	gsl_vector_memcpy( params->Q_Wy_value, params->Wu_inv_s_delta_Yu );

	if( params->isLowRank){
		// Q_Wy_value += cp_W_low_Y_low / delta
		gsl_blas_daxpy( 1/delta,  params->cp_W_low_Y_low, params->Q_Wy_value );
	}	
}

// cp_W_low_r = crossprod(W_til, obj$r) - crossprod(Wu, obj$ru)
// crossprod(Wu, obj$inv_s_delta_ru) + cp_W_low_r / delta
inline void LRGPR::Q_Wr(const double delta){

	// crossprod(Wu, obj$inv_s_delta_ru)
	gsl_blas_dgemv(CblasTrans, 1.0, params->Wu, params->inv_s_delta_ru, 0.0, params->Wu_inv_s_delta_ru);

	gsl_vector_memcpy( params->Q_Wr_value, params->Wu_inv_s_delta_ru );

	if( params->isLowRank){
		// cp_W_low_r = crossprod(W_til, obj$r)
		gsl_blas_dgemv(CblasTrans,1.0, params->W_til, params->r, 0.0, params->cp_W_til_r);

		// cp_W_til_r = cp_W_low_r - crossprod(Wu, obj$ru)
		gsl_blas_dgemv(CblasTrans,1.0, params->Wu, params->ru, -1.0, params->cp_W_til_r);

		// cp_W_low_r / delta
		/*gsl_vector_scale( params->cp_W_til_r, 1 / delta);

		// add	
		gsl_vector_memcpy( params->Q_Wr_value, params->Wu_inv_s_delta_ru);
		gsl_vector_add( params->Q_Wr_value, params->cp_W_til_r);*/

		// Q_Wr_value += cp_W_til_r / delta
		gsl_blas_daxpy( 1/delta,  params->cp_W_til_r, params->Q_Wr_value );
	}	
}

// crossprod(Wu, obj$inv_s_delta_Wu) + cp_W_low / delta
inline void LRGPR::Q_WW(const double delta){

	// crossprod(Wu, obj$inv_s_delta_Wu)
	gsl_blas_dgemm(CblasTrans,CblasNoTrans, 1.0, params->Wu, params->inv_s_delta_Wu, 0.0, params->Wu_inv_s_delta_Wu);

	gsl_matrix_memcpy( params->Q_WW_value, params->Wu_inv_s_delta_Wu );

	//gsl_matrix_print( params->Wu_inv_s_delta_Wu );
	//gsl_matrix_print( params->cp_W_low );

	if( params->isLowRank){
		// Q_WW_value += cp_W_low / delta
		gsl_matrix_daxpy( 1/delta,  params->cp_W_low, params->Q_WW_value );
	}
	
	//gsl_matrix_print( params->Q_WW_value );

}

/*
if( ! proximalContamination )
	Q_XX(delta, obj)
else
	Q_XX(delta, obj) + obj$eval_Q_XW_WW %*% obj$eval_Q_XW
*/
inline void LRGPR::Omega_XX(const double delta){

	this->Q_XX( delta );

	//gsl_matrix_print( params->Q_XX_value );

	if( params->proximalContamination ){		

		// Q_XW_WW_Q_Xw_value = eval_Q_XW_WW %*% eval_Q_XW
		gsl_blas_dgemm(CblasNoTrans,CblasNoTrans, 1.0, params->Q_XW_WW_value, params->Q_XW_value, 0.0, params->Q_XW_WW_Q_Xw_value);

		// Q_XX(delta, obj) + obj$eval_Q_XW_WW %*% obj$eval_Q_XW
		gsl_matrix_add( params->Q_XX_value, params->Q_XW_WW_Q_Xw_value );	
	}	

	/*gsl_matrix_print( params->Q_XW_WW_value );
	gsl_matrix_print( params->Q_XW_value );
	gsl_matrix_print( params->Q_XW_WW_Q_Xw_value );
	gsl_matrix_print( params->Q_XX_value );*/
}

/*
if( ! proximalContamination )
	Q_Xy(delta, obj)
else
	Q_Xy(delta, obj) + obj$eval_Q_XW_WW %*% Q_Wy(delta, obj)
*/
inline void LRGPR::Omega_Xy(const double delta){

	this->Q_Xy( delta );

	//gsl_vector_print( params->Q_Xy_value);

	if( params->proximalContamination ){
		// Q_Xy(delta, obj) + obj$eval_Q_XW_WW %*% Q_Wy(delta, obj)
		this->Q_Wy( delta );

		//gsl_vector_print( params->Q_Wy_value);

		gsl_blas_dgemv(CblasNoTrans, 1.0, params->Q_XW_WW_value, params->Q_Wy_value, 1.0, params->Q_Xy_value);
	}	
}

/*
if( ! proximalContamination )
	Q_rr(delta, obj)
else
	Q_rr(delta, obj) + crossprod(obj$eval_Q_Wr, obj$solve_I_Q_WW) %*% obj$eval_Q_Wr 
*/
inline void LRGPR::Omega_rr(const double delta){

	/*
	env$obj$ru 				= Yu - Xu %*% env$obj$beta
	env$obj$r 				= Y - X %*% env$obj$beta
	env$obj$inv_s_delta_ru 	= env$obj$inv_s_delta * env$obj$ru
	env$obj$eval_Q_Wr		= Q_Wr(delta, env$obj)
	*/

	// ru = Yu - Xu %*% beta
	gsl_vector_memcpy( params->ru, params->Yu );
	gsl_blas_dgemv (CblasNoTrans, -1.0, params->Xu, params->beta, 1.0, params->ru);

	// r = Y - X %*% beta
	gsl_vector_memcpy( params->r, params->Y );
	gsl_blas_dgemv (CblasNoTrans, -1.0, params->X, params->beta, 1.0, params->r);

	// inv_s_delta_ru 	= inv_s_delta * ru
	gsl_vector_memcpy( params->inv_s_delta_ru, params->ru );
	gsl_vector_mul( params->inv_s_delta_ru, params->inv_s_delta );

	if( ! params->proximalContamination ){
		this->Q_rr( delta );
	}else{

		this->Q_Wr( delta ); // this shoud (?) be evaluated before Q_rr(delta)

		// Q_rr(delta, obj) + crossprod(obj$eval_Q_Wr, obj$solve_I_Q_WW) %*% obj$eval_Q_Wr 
		this->Q_rr( delta );

		// += crossprod(obj$eval_Q_Wr, obj$solve_I_Q_WW) %*% obj$eval_Q_Wr 
		params->Q_rr_value += gsl_matrix_quadratic_form_sym( params->solve_I_Q_WW, params->Q_Wr_value  );
	}
}


double _LRGPR_fxn( const double delta, void *params_arg){

	//cout << "_LRGPR_fxn: " << delta << endl;

	LRGPR *obj = (LRGPR *) params_arg;

	LRGPR_params *params = obj->params;

	// Preprocess values
	/*
	env$obj$inv_s_delta 	= 1/(s+delta)
	env$obj$inv_s_delta_Yu 	= env$obj$inv_s_delta * Yu
	env$obj$inv_s_delta_Xu 	= env$obj$inv_s_delta * Xu
	env$obj$inv_s_delta_Wu 	= env$obj$inv_s_delta * Wu
	env$obj$solve_I_Q_WW 	= solve( I_c - Q_WW(delta, env$obj) )
	env$obj$eval_Q_XW 		= Q_XW(delta, env$obj)
	env$obj$eval_Q_XW_WW	= crossprod(env$obj$eval_Q_XW, env$obj$solve_I_Q_WW)
	*/

	// inv_s_delta = 1/(s+delta)
	for( unsigned int i=0; i<params->s->size; i++){
		gsl_vector_set( params->inv_s_delta, i, 1/(gsl_vector_get( params->s, i ) + delta) );
	}

	// inv_s_delta_Yu = inv_s_delta * Yu
	gsl_vector_memcpy( params->inv_s_delta_Yu, params->Yu );
	gsl_vector_mul( params->inv_s_delta_Yu, params->inv_s_delta );

	// inv_s_delta_Xu = inv_s_delta * Xu
	gsl_matrix_diagonal_multiply( params->Xu, params->inv_s_delta, params->inv_s_delta_Xu );

	if( params->proximalContamination ){

		// inv_s_delta_Wu = inv_s_delta * Wu
		gsl_matrix_diagonal_multiply( params->Wu, params->inv_s_delta, params->inv_s_delta_Wu );

		// crossprod(Wu, obj$inv_s_delta_Wu) + cp_W_low / delta
		obj->Q_WW( delta );

		// solve_I_Q_WW = solve( I_c - Q_WW(delta, env$obj) )
		gsl_matrix_memcpy( params->solve_I_Q_WW, params->Q_WW_value );
		gsl_matrix_scale( params->solve_I_Q_WW, -1.0);
		gsl_matrix_add_diag_scalar( params->solve_I_Q_WW, 1 );

		gsl_lapack_lu_invert_logdet(params->solve_I_Q_WW, &(params->Q_WW_1_logDet) );

		// Q_XW(delta, env$obj)
		obj->Q_XW( delta );

		// crossprod(env$obj$eval_Q_XW, env$obj$solve_I_Q_WW)
		gsl_blas_dgemm(CblasTrans,CblasNoTrans, 1.0, params->Q_XW_value, params->solve_I_Q_WW, 0.0, params->Q_XW_WW_value);
	}

	// Estimate beta
	obj->estimate_beta( delta );

	// Estimate sig_g
	obj->Omega_rr( delta );

	// sig_g = Omega_rr( delta ) / n
	params->sig_g = params->Q_rr_value / (double) params->n_indivs;

	//cout << "log_L: " << obj->logLikelihood( delta ) << endl;

	return -1.0 * obj->logLikelihood( delta );
}


gsl_vector *LRGPR::get_hat_matrix_diag( ){
	return params->get_hat_matrix_diag( params->delta );
}
void LRGPR::get_hat_matrix_diag( gsl_vector * res ){
	gsl_vector_memcpy( res, params->get_hat_matrix_diag( params->delta ) );
}

gsl_vector *LRGPR::get_fitted_response( gsl_vector *alpha){
	return params->get_fitted_response( params->delta, alpha);
}

gsl_matrix *LRGPR::coeff_covariance(){

	// V = solve(Omega_XX()) * sig_g
	//   = Q_XX_value * sig_g
	gsl_matrix *V = gsl_matrix_alloc( params->Q_XX_value->size1, params->Q_XX_value->size2 );

	coeff_covariance( V );

	return V;
}
void LRGPR::coeff_covariance( gsl_matrix * res){

	gsl_matrix_set_zero( res );

	// res = Q_XX_value * sig_g
	gsl_matrix_daxpy( params->sig_g, params->Q_XX_value, res );
}

gsl_vector *LRGPR::get_beta(){

	gsl_vector *beta_cpy = gsl_vector_alloc( params->beta->size );
	gsl_vector_memcpy( beta_cpy, params->beta);
	return beta_cpy;
}
void LRGPR::get_beta( gsl_vector *beta_ ){
	gsl_vector_memcpy( beta_, params->beta);
}


gsl_vector *LRGPR::wald_test_all(){

	// get beta, sd, z-value, p-value
	gsl_matrix *Sigma = coeff_covariance();		
	gsl_vector *p_values = gsl_vector_alloc( params->X_ncol );

	double z_stat;

	for(unsigned i=0; i<params->X_ncol; i++){

		// z = beta / sqrt(Sigma)
		z_stat = fabs(gsl_vector_get(params->beta, i) / sqrt(gsl_matrix_get(Sigma, i, i)));

		gsl_vector_set( p_values, i, 2*gsl_cdf_ugaussian_Q(z_stat) );
	}

	gsl_matrix_free(Sigma);

	return p_values;
}

double LRGPR::wald_test( vector<int> &terms ){

	double pValue; 

	// Sigma = Omega_XX(delta)
	gsl_matrix *Sigma = coeff_covariance();	
	gsl_matrix *Sigma_sub = gsl_matrix_alloc( terms.size(), terms.size() );
	gsl_vector *beta_sub = gsl_vector_alloc( terms.size() );

	gsl_matrix_sub_row_col( Sigma, terms, Sigma_sub);
	gsl_vector_subset( params->beta, terms, beta_sub );

	// solve(Sigma)
	int result = gsl_lapack_chol_invert( Sigma_sub );			

	if( result == GSL_SUCCESS ){
		// tcrossprod(fit$coefficients[terms], solve(fit$Sigma[terms,terms])) %*% fit$coefficients[terms]
		double stat = gsl_matrix_quadratic_form_sym( Sigma_sub, beta_sub );

		// pchisq( stat, df, lower.tail=FALSE)
		pValue = gsl_cdf_chisq_Q( stat, terms.size() );

	}else{	
		pValue = NAN;
	} 
	
	gsl_matrix_free( Sigma );
	gsl_matrix_free( Sigma_sub );
	gsl_vector_free( beta_sub );

	return pValue;
}

double LRGPR::get_effective_df(){
	return params->get_effective_df( params->delta);
}

double LRGPR::get_effective_df( const double delta){
	return params->get_effective_df(delta);
}