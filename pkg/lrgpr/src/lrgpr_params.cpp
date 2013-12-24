
#include "lrgpr_params.h"

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

#ifdef INTEL
#include "mkl_service.h"
#endif

// constructor
LRGPR_params::LRGPR_params( const gsl_vector *Y_, const gsl_matrix *t_U_, const gsl_vector *eigenValues, const int X_ncol_, const int W_ncol_ ){

	init_as_null();

	n_indivs = Y_->size;
	X_ncol 	= X_ncol_;
	W_ncol 	= W_ncol_;

	s = const_cast<gsl_vector*>( eigenValues );
	t_U = const_cast<gsl_matrix*>( t_U_ );

	rank = s->size;

	proximalContamination = false;

	alloc_Y( n_indivs );
	alloc_rank( rank );
	alloc_X( X_ncol ); 
	alloc_W( W_ncol ); 

	gsl_vector_memcpy( Y, Y_ );

	if( rank == n_indivs ){
		isLowRank = false;
	}else{
		isLowRank = true;
	}

	// Yu = t(U) %*% Y
	gsl_blas_dgemv( CblasNoTrans, 1.0, t_U, Y, 0.0, Yu );
}


void LRGPR_params::init_as_null(){

	init_variables_Y = false;
	init_variables_rank = false;
	init_variables_X = false;
	init_variables_W = false;

	Y = NULL;
	Yu = NULL;
	Xu = NULL;
	X = NULL;
	beta = NULL;
	inv_s_delta = NULL;
	W_til = NULL;
	Wu = NULL;
	r = NULL;
	ru = NULL;

	cp_X_low = NULL;
	cp_Xu = NULL;
	cp_X_low_Y_low = NULL;
	cp_Xu_Yu = NULL;
	cp_W_low = NULL;
	cp_Wu = NULL;
	//cp_W_low_X = NULL;
	cp_Wu_Xu = NULL;
	cp_W_low_Y_low = NULL;
	cp_Wu_Yu = NULL;
	cp_W_low_X_low = NULL;
	cp_W_til_r = NULL;

	Q_XX_value = NULL;
	Q_Xy_value = NULL;
	Q_XW_value = NULL;
	Q_Wy_value = NULL;
	Q_WW_value = NULL;
	Q_Wr_value = NULL;
	Q_XW_WW_value = NULL;
	inv_s_delta_Xu = NULL;
	inv_s_delta_Yu = NULL;
	inv_s_delta_ru = NULL;
	inv_s_delta_Wu = NULL;

	Xu_inv_s_delta_Xu = NULL;
	Xu_inv_s_delta_Yu = NULL;
	Wu_inv_s_delta_Wu = NULL;
	Wu_inv_s_delta_Yu = NULL;
	Wu_inv_s_delta_ru = NULL;	
	Xu_inv_s_delta_Wu = NULL;

	/*cp_X_low_delta = NULL;
	cp_X_low_Y_low_delta = NULL;
	cp_W_low_X_low_delta = NULL;
	cp_W_low_delta = NULL;
	cp_W_low_Y_low_delta = NULL;*/

	Q_XW_WW_Q_Xw_value = NULL;
	solve_I_Q_WW = NULL;

	/*Y = gsl_vector_alloc( n_indivs );
	residuals =  gsl_vector_alloc( n_indivs );
	r = gsl_vector_alloc( n_indivs );

	X = gsl_matrix_alloc( n_indivs, X_ncol );

	W_til = gsl_matrix_alloc( n_indivs, W_ncol );
	Wu = gsl_matrix_alloc( n_indivs, W_ncol );


	Yu = gsl_vector_alloc( rank );
	residuals_u = gsl_vector_alloc( rank );
	ru = gsl_vector_alloc( rank );	
	inv_s_delta = gsl_vector_alloc( rank );
	inv_s_delta_Yu = gsl_vector_alloc( rank );
	inv_s_delta_ru = gsl_vector_alloc( rank );

	inv_s_delta_Wu = gsl_matrix_alloc( rank, W_ncol );

	inv_s_delta_Xu = gsl_matrix_alloc( rank, X_ncol );
	Xu = gsl_matrix_alloc( rank, X_ncol );

	beta = gsl_vector_alloc( X_ncol );
	cp_X_low = gsl_matrix_alloc( X_ncol, X_ncol );
	cp_Xu = gsl_matrix_alloc( X_ncol, X_ncol );
	cp_X_low_Y_low = gsl_vector_alloc( X_ncol );
	cp_Xu_Yu = gsl_vector_alloc( X_ncol );
	Q_XX_value = gsl_matrix_alloc( X_ncol, X_ncol );
	Q_Xy_value = gsl_vector_alloc( X_ncol );
	Xu_inv_s_delta_Xu = gsl_matrix_alloc( X_ncol, X_ncol );
	Xu_inv_s_delta_Yu = gsl_vector_alloc( X_ncol );
	cp_X_low_delta = gsl_matrix_alloc( X_ncol, X_ncol );
	cp_X_low_Y_low_delta = gsl_vector_alloc( X_ncol );	
	Q_XW_WW_Q_Xw_value = gsl_matrix_alloc( X_ncol, X_ncol );

	
	cp_W_low_X = gsl_matrix_alloc( W_ncol, X_ncol );
	cp_Wu_Xu = gsl_matrix_alloc( W_ncol, X_ncol );
	cp_W_low_X_low = gsl_matrix_alloc( W_ncol, X_ncol );	
	Q_XW_value = gsl_matrix_alloc( W_ncol, X_ncol );
	Q_XW_WW_value = gsl_matrix_alloc( X_ncol, W_ncol );
	cp_W_low_X_low_delta = gsl_matrix_alloc( W_ncol, X_ncol );

	cp_W_low = gsl_matrix_alloc( W_ncol, W_ncol );
	cp_Wu = gsl_matrix_alloc( W_ncol, W_ncol );
	cp_W_til_r = gsl_vector_alloc( W_ncol );
	cp_W_low_Y_low = gsl_vector_alloc( W_ncol );
	cp_Wu_Yu = gsl_vector_alloc( W_ncol );
	Q_Wy_value = gsl_vector_alloc( W_ncol );
	Q_WW_value = gsl_matrix_alloc( W_ncol, W_ncol );
	Q_Wr_value = gsl_vector_alloc( W_ncol );
	Wu_inv_s_delta_Wu = gsl_matrix_alloc( W_ncol, W_ncol );
	Wu_inv_s_delta_Yu = gsl_vector_alloc( W_ncol );
	Wu_inv_s_delta_ru = gsl_vector_alloc( W_ncol );	
	cp_W_low_delta = gsl_matrix_alloc( W_ncol, W_ncol );
	cp_W_low_Y_low_delta = gsl_vector_alloc( W_ncol );
	solve_I_Q_WW = gsl_matrix_alloc( W_ncol, W_ncol );*/
}

void LRGPR_params::alloc_Y( const int n_indivs_ ){

	if( n_indivs != n_indivs_ || ! init_variables_Y ){

		n_indivs = n_indivs_;
		
		// free and alloc variables
		if( Y != NULL ) 		gsl_vector_free( Y );
		if( r != NULL ) 		gsl_vector_free( r );
		if( X != NULL ) 		gsl_matrix_free( X );
		if( W_til != NULL ) 	gsl_matrix_free( W_til );
		if( Wu != NULL ) 		gsl_matrix_free( Wu );

		Y = 			gsl_vector_alloc( n_indivs );
		r = 			gsl_vector_alloc( n_indivs );
		X = 			gsl_matrix_alloc( n_indivs, X_ncol );
		
		if( W_ncol > 0 ){
			W_til = 	gsl_matrix_alloc( n_indivs, W_ncol );			
			Wu = 		gsl_matrix_alloc( n_indivs, W_ncol );
		}

		init_variables_Y = true;
	}
}

void LRGPR_params::alloc_rank( const int rank_ ){

	if( rank != rank_ || ! init_variables_rank ){

		rank = rank_;

		// free and alloc variables
		if( Yu != NULL ) 				gsl_vector_free( Yu );
		if( ru != NULL ) 				gsl_vector_free( ru );
		if( inv_s_delta != NULL ) 		gsl_vector_free( inv_s_delta );
		if( inv_s_delta_Yu != NULL )	gsl_vector_free( inv_s_delta_Yu );
		if( inv_s_delta_ru != NULL ) 	gsl_vector_free( inv_s_delta_ru );
		if( inv_s_delta_Wu != NULL ) 	gsl_matrix_free( inv_s_delta_Wu );
		if( inv_s_delta_Xu != NULL ) 	gsl_matrix_free( inv_s_delta_Xu );
		if( Xu != NULL ) 				gsl_matrix_free( Xu );

		Yu = 				gsl_vector_alloc( rank );
		ru = 				gsl_vector_alloc( rank );	
		inv_s_delta = 		gsl_vector_alloc( rank );
		inv_s_delta_Yu = 	gsl_vector_alloc( rank );
		inv_s_delta_ru = 	gsl_vector_alloc( rank );

		if( W_ncol > 0 ){
			inv_s_delta_Wu = 	gsl_matrix_alloc( rank, W_ncol );
		}

		inv_s_delta_Xu = 	gsl_matrix_alloc( rank, X_ncol );
		Xu = 				gsl_matrix_alloc( rank, X_ncol );

		init_variables_rank = true;
	}
}

void LRGPR_params::alloc_X( const int X_ncol_ ){

	if( X_ncol != X_ncol_ || ! init_variables_X ){

		X_ncol = X_ncol_;

		// free and alloc variables
		if( X != NULL ) 				gsl_matrix_free( X );
		if( inv_s_delta_Xu != NULL ) 	gsl_matrix_free( inv_s_delta_Xu );
		if( Xu != NULL ) 				gsl_matrix_free( Xu );
		if( beta != NULL ) 				gsl_vector_free( beta );
		if( cp_X_low != NULL ) 			gsl_matrix_free( cp_X_low );
		if( cp_Xu != NULL ) 			gsl_matrix_free( cp_Xu );
		if( cp_X_low_Y_low != NULL ) 	gsl_vector_free( cp_X_low_Y_low );
		if( cp_Xu_Yu != NULL ) 			gsl_vector_free( cp_Xu_Yu );

		if( Q_XX_value != NULL ) 		gsl_matrix_free( Q_XX_value );
		if( Q_Xy_value != NULL ) 		gsl_vector_free( Q_Xy_value );
		if( Xu_inv_s_delta_Xu != NULL ) gsl_matrix_free( Xu_inv_s_delta_Xu );
		if( Xu_inv_s_delta_Yu != NULL ) gsl_vector_free( Xu_inv_s_delta_Yu );
		if( Xu_inv_s_delta_Wu != NULL ) gsl_matrix_free( Xu_inv_s_delta_Wu );

		//if( cp_X_low_delta != NULL ) 	gsl_matrix_free( cp_X_low_delta );
		//if( cp_X_low_Y_low_delta != NULL ) 	gsl_vector_free( cp_X_low_Y_low_delta );
		if( Q_XW_WW_Q_Xw_value != NULL ) 	gsl_matrix_free( Q_XW_WW_Q_Xw_value );

		//if( cp_W_low_X != NULL ) 			gsl_matrix_free( cp_W_low_X );
		if( cp_Wu_Xu != NULL ) 				gsl_matrix_free( cp_Wu_Xu );
		if( cp_W_low_X_low != NULL ) 		gsl_matrix_free( cp_W_low_X_low );
		if( Q_XW_value != NULL ) 			gsl_matrix_free( Q_XW_value );
		if( Q_XW_WW_value != NULL ) 		gsl_matrix_free( Q_XW_WW_value );
		//if( cp_W_low_X_low_delta != NULL ) 	gsl_matrix_free( cp_W_low_X_low_delta );

		X = 				gsl_matrix_alloc( n_indivs, X_ncol );
		inv_s_delta_Xu = 	gsl_matrix_alloc( rank, X_ncol );
		Xu = 				gsl_matrix_alloc( rank, X_ncol );

		beta = 				gsl_vector_alloc( X_ncol );
		cp_X_low = 			gsl_matrix_alloc( X_ncol, X_ncol );
		cp_Xu = 			gsl_matrix_alloc( X_ncol, X_ncol );
		cp_X_low_Y_low = 	gsl_vector_alloc( X_ncol );
		cp_Xu_Yu = 			gsl_vector_alloc( X_ncol );
		Q_XX_value = 		gsl_matrix_alloc( X_ncol, X_ncol );
		Q_Xy_value = 		gsl_vector_alloc( X_ncol );
		Xu_inv_s_delta_Xu = gsl_matrix_alloc( X_ncol, X_ncol );
		Xu_inv_s_delta_Yu = gsl_vector_alloc( X_ncol );
		//cp_X_low_delta = 	gsl_matrix_alloc( X_ncol, X_ncol );
		//cp_X_low_Y_low_delta = 	gsl_vector_alloc( X_ncol );	
		Q_XW_WW_Q_Xw_value = 	gsl_matrix_alloc( X_ncol, X_ncol );

		if( W_ncol > 0 ){
			//cp_W_low_X = 			gsl_matrix_alloc( W_ncol, X_ncol );
			cp_Wu_Xu = 				gsl_matrix_alloc( W_ncol, X_ncol );
			cp_W_low_X_low = 		gsl_matrix_alloc( W_ncol, X_ncol );	
			Q_XW_value = 			gsl_matrix_alloc( W_ncol, X_ncol );
			Q_XW_WW_value = 		gsl_matrix_alloc( X_ncol, W_ncol );
			Xu_inv_s_delta_Wu = 	gsl_matrix_alloc( X_ncol, W_ncol );
			//cp_W_low_X_low_delta = 	gsl_matrix_alloc( W_ncol, X_ncol );
		}

		init_variables_X = true;
	}
}

void LRGPR_params::alloc_W( const int W_ncol_ ){

	if( W_ncol != W_ncol_ || ! init_variables_W){

		W_ncol != W_ncol_;

		// free and alloc variables
		if( W_til != NULL ) 	gsl_matrix_free( W_til );

		if( Wu != NULL ) 	gsl_matrix_free( Wu );
		if( inv_s_delta_Wu != NULL ) 	gsl_matrix_free( inv_s_delta_Wu );

		//if( cp_W_low_X != NULL ) 			gsl_matrix_free( cp_W_low_X );
		if( cp_Wu_Xu != NULL ) 				gsl_matrix_free( cp_Wu_Xu );
		if( cp_W_low_X_low != NULL ) 		gsl_matrix_free( cp_W_low_X_low );
		if( Q_XW_value != NULL ) 			gsl_matrix_free( Q_XW_value );
		if( Q_XW_WW_value != NULL ) 		gsl_matrix_free( Q_XW_WW_value );
		//if( cp_W_low_X_low_delta != NULL ) 	gsl_matrix_free( cp_W_low_X_low_delta );

		if( cp_W_low != NULL ) 				gsl_matrix_free( cp_W_low );
		if( cp_Wu != NULL ) 				gsl_matrix_free( cp_Wu );
		if( cp_W_til_r != NULL ) 			gsl_vector_free( cp_W_til_r );
		if( cp_W_low_Y_low != NULL ) 		gsl_vector_free( cp_W_low_Y_low );
		if( cp_Wu_Yu != NULL ) 				gsl_vector_free( cp_Wu_Yu );
		if( Q_Wy_value != NULL ) 			gsl_vector_free( Q_Wy_value );
		if( Q_WW_value != NULL ) 			gsl_matrix_free( Q_WW_value );
		if( Q_Wr_value != NULL ) 			gsl_vector_free( Q_Wr_value );
		if( Wu_inv_s_delta_Wu != NULL ) 	gsl_matrix_free( Wu_inv_s_delta_Wu );
		if( Wu_inv_s_delta_Yu != NULL ) 	gsl_vector_free( Wu_inv_s_delta_Yu );
		if( Wu_inv_s_delta_ru ) 			gsl_vector_free( Wu_inv_s_delta_ru );
		if( Xu_inv_s_delta_Wu != NULL ) 	gsl_matrix_free( Xu_inv_s_delta_Wu );
		//if( cp_W_low_delta != NULL ) 		gsl_matrix_free( cp_W_low_delta );
		//if( cp_W_low_Y_low_delta != NULL ) 	gsl_vector_free( cp_W_low_Y_low_delta );
		if( solve_I_Q_WW != NULL ) 			gsl_matrix_free( solve_I_Q_WW );

		if( W_ncol > 0 ){
			W_til = gsl_matrix_alloc( n_indivs, W_ncol );
			Wu = gsl_matrix_alloc( rank, W_ncol );

			inv_s_delta_Wu = gsl_matrix_alloc( rank, W_ncol );

			//cp_W_low_X = gsl_matrix_alloc( W_ncol, X_ncol );
			cp_Wu_Xu = gsl_matrix_alloc( W_ncol, X_ncol );
			cp_W_low_X_low = gsl_matrix_alloc( W_ncol, X_ncol );	
			Q_XW_value = gsl_matrix_alloc( W_ncol, X_ncol );
			Q_XW_WW_value = gsl_matrix_alloc( X_ncol, W_ncol );
			//cp_W_low_X_low_delta = gsl_matrix_alloc( W_ncol, X_ncol );

			cp_W_low = gsl_matrix_alloc( W_ncol, W_ncol );
			cp_Wu = gsl_matrix_alloc( W_ncol, W_ncol );
			cp_W_til_r = gsl_vector_alloc( W_ncol );
			cp_W_low_Y_low = gsl_vector_alloc( W_ncol );
			cp_Wu_Yu = gsl_vector_alloc( W_ncol );
			Q_Wy_value = gsl_vector_alloc( W_ncol );
			Q_WW_value = gsl_matrix_alloc( W_ncol, W_ncol );
			Q_Wr_value = gsl_vector_alloc( W_ncol );
			Wu_inv_s_delta_Wu = gsl_matrix_alloc( W_ncol, W_ncol );
			Wu_inv_s_delta_Yu = gsl_vector_alloc( W_ncol );
			Wu_inv_s_delta_ru = gsl_vector_alloc( W_ncol );				
			Xu_inv_s_delta_Wu = 	gsl_matrix_alloc( X_ncol, W_ncol );
			//cp_W_low_delta = gsl_matrix_alloc( W_ncol, W_ncol );
			//cp_W_low_Y_low_delta = gsl_vector_alloc( W_ncol );
			solve_I_Q_WW = gsl_matrix_alloc( W_ncol, W_ncol );	
		}

		init_variables_W = true; 
	}

	if( W_ncol == 0 ) 	proximalContamination = false;
	else 				proximalContamination = true;	
}
 

// destructor
LRGPR_params::~LRGPR_params(){

	gsl_vector_free( Y );
	gsl_vector_free( Yu );
	gsl_matrix_free( Xu );
	gsl_matrix_free( X );
	gsl_vector_free( beta );
	//gsl_vector_free( s );
	gsl_vector_free( inv_s_delta );
	gsl_vector_free( r );
	gsl_vector_free( ru );

	gsl_matrix_free( cp_Xu );
	gsl_vector_free( cp_Xu_Yu );
	
	gsl_matrix_free( Q_XX_value );
	gsl_vector_free( Q_Xy_value );
	
	gsl_matrix_free( inv_s_delta_Xu );
	gsl_vector_free( inv_s_delta_Yu );
	gsl_vector_free( inv_s_delta_ru );

	gsl_matrix_free( Xu_inv_s_delta_Xu );
	gsl_vector_free( Xu_inv_s_delta_Yu );

	// low rank
	if( isLowRank ){
		gsl_matrix_free( cp_X_low );
		gsl_vector_free( cp_X_low_Y_low );

		//gsl_matrix_free( cp_X_low_delta );
		//gsl_vector_free( cp_X_low_Y_low_delta );
	}

	// prox con
	if( proximalContamination ){
		gsl_matrix_free( W_til );
		gsl_matrix_free( Wu );

		gsl_matrix_free( cp_W_low );
		gsl_matrix_free( cp_Wu );
		//gsl_matrix_free( cp_W_low_X );
		gsl_matrix_free( cp_Wu_Xu );
		gsl_vector_free( cp_W_low_Y_low );
		gsl_vector_free( cp_Wu_Yu );
		gsl_matrix_free( cp_W_low_X_low );
		gsl_vector_free( cp_W_til_r );

		gsl_matrix_free( Q_XW_value );
		gsl_vector_free( Q_Wy_value );
		gsl_matrix_free( Q_WW_value );
		gsl_vector_free( Q_Wr_value );
		gsl_matrix_free( Q_XW_WW_value );

		gsl_matrix_free( inv_s_delta_Wu );

		gsl_matrix_free( Wu_inv_s_delta_Wu );
		gsl_vector_free( Wu_inv_s_delta_Yu );
		gsl_vector_free( Wu_inv_s_delta_ru );	

		//gsl_matrix_free( cp_W_low_X_low_delta );
		//gsl_matrix_free( cp_W_low_delta );
		//gsl_vector_free( cp_W_low_Y_low_delta );

		gsl_matrix_free( Q_XW_WW_Q_Xw_value );
		gsl_matrix_free( solve_I_Q_WW );
	}
}

void LRGPR_params::update_X( const gsl_matrix *X_ ){

	alloc_X( X_->size2 );

	gsl_matrix_memcpy( X, X_);

	// Xu = t(U) %*% X
	//gsl_blas_dgemm( CblasTrans, CblasNoTrans, 1.0, eigen->U, X, 0.0, Xu );
	gsl_blas_dgemm( CblasNoTrans, CblasNoTrans, 1.0, t_U, X, 0.0, Xu );

	// If low rank model
	if( this->isLowRank ){

		// cp_X_low = crossprod(X) - crossprod(Xu)
		gsl_blas_dgemm( CblasTrans, CblasNoTrans, 1.0, X, X, 0.0, cp_X_low );
		gsl_blas_dgemm( CblasTrans, CblasNoTrans, 1.0, Xu, Xu, 0.0, cp_Xu );
		gsl_matrix_sub( cp_X_low, cp_Xu);

		// cp_X_low_Y_low = crossprod(X, Y) - crossprod(Xu, Yu)
		gsl_blas_dgemv( CblasTrans, 1.0, X, Y, 0.0, cp_X_low_Y_low );
		gsl_blas_dgemv( CblasTrans, 1.0, Xu, Yu, 0.0, cp_Xu_Yu );
		gsl_vector_sub( cp_X_low_Y_low, cp_Xu_Yu);

		if( this->proximalContamination ){

			gsl_vector_set_zero(cp_W_low_Y_low );
			gsl_matrix_set_zero(cp_W_low_X_low );
			gsl_matrix_set_zero(cp_W_low );

			// cp_W_low = crossprod(W_til) - crossprod(Wu)
			/*gsl_blas_dgemm( CblasTrans, CblasNoTrans, 1.0, W_til, W_til, 0.0, cp_W_low );
			gsl_blas_dgemm( CblasTrans, CblasNoTrans, 1.0, Wu, Wu, 0.0, cp_Wu );
			gsl_matrix_sub( cp_W_low, cp_Wu);

			//cp_W_low_X_low = crossprod(W_til, X) - crossprod(Wu, Xu)
			gsl_blas_dgemm( CblasTrans, CblasNoTrans, 1.0, W_til, X, 0.0, cp_W_low_X_low );
			gsl_blas_dgemm( CblasTrans, CblasNoTrans, 1.0, Wu, Xu, 0.0, cp_Wu_Xu );
			gsl_matrix_sub( cp_W_low_X_low, cp_Wu_Xu);

			// cp_W_low_Y_low = crossprod(W_til, Y) - crossprod(Wu, Yu)
			gsl_blas_dgemv( CblasTrans, 1.0, W_til, Y, 0.0, cp_W_low_Y_low );
			gsl_blas_dgemv( CblasTrans, 1.0, Wu, Yu, 0.0, cp_Wu_Yu );
			gsl_vector_sub( cp_W_low_Y_low, cp_Wu_Yu);*/			
		}	
	}
}


void LRGPR_params::update_Xu( const gsl_matrix *X_, const gsl_matrix *Xu_ ){

	alloc_X( X_->size2 );
	alloc_rank( Xu_->size1 );

	gsl_matrix_memcpy( X, X_);
	gsl_matrix_memcpy( Xu, Xu_);

	// If low rank model	
	if( this->isLowRank ){

		// cp_X_low = crossprod(X) - crossprod(Xu)
		gsl_blas_dgemm( CblasTrans, CblasNoTrans, 1.0, X, X, 0.0, cp_X_low );
		gsl_blas_dgemm( CblasTrans, CblasNoTrans, 1.0, Xu, Xu, 0.0, cp_Xu );
		gsl_matrix_sub( cp_X_low, cp_Xu);

		// cp_X_low_Y_low = crossprod(X, Y) - crossprod(Xu, Yu)
		gsl_blas_dgemv( CblasTrans, 1.0, X, Y, 0.0, cp_X_low_Y_low );
		gsl_blas_dgemv( CblasTrans, 1.0, Xu, Yu, 0.0, cp_Xu_Yu );
		gsl_vector_sub( cp_X_low_Y_low, cp_Xu_Yu);

		if( this->proximalContamination ){

			gsl_vector_set_zero(cp_W_low_Y_low );
			gsl_matrix_set_zero(cp_W_low_X_low );
			gsl_matrix_set_zero(cp_W_low );

			// cp_W_low_X_low = crossprod(W_til, X) - crossprod(Wu, Xu)
			/*gsl_blas_dgemm( CblasTrans, CblasNoTrans, 1.0, W_til, X, 0.0, cp_W_low_X_low );
			gsl_blas_dgemm( CblasTrans, CblasNoTrans, 1.0, Wu, Xu, 0.0, cp_Wu_Xu );
			gsl_matrix_sub( cp_W_low_X_low, cp_Wu_Xu);*/
		}	
	}
}

void LRGPR_params::update_Y( const gsl_vector *Y_ ){

	alloc_Y( Y_->size );

	gsl_vector_memcpy( Y, Y_);

	// Yu = t(U) %*% Y
	gsl_blas_dgemv( CblasNoTrans, 1.0, t_U, Y, 0.0, Yu );
}

void LRGPR_params::update_Wtilde( const gsl_matrix *W_til_ ){

	// If W_til_ is 1 x 1, do not consider it as a pron-con matrix
	// If W_til_ has at least 1 column and more than one row
	if( W_til_->size1 > 1 &&  W_til_->size2 >= 1){

		alloc_W( W_til_->size2 );

		gsl_matrix_memcpy( W_til, W_til_);

		// Wu = t(U) %*% W
		gsl_blas_dgemm( CblasNoTrans, CblasNoTrans, 1.0, t_U, W_til, 0.0, Wu );

		proximalContamination = true;
	}
}

double LRGPR_params::get_S_beta_trace(){

	return X->size2;	
}

gsl_vector *LRGPR_params::get_S_beta_diag(){

	// C = solve(crossprod(X))
	gsl_matrix *C = gsl_matrix_alloc( X->size2, X->size2 );
	gsl_matrix_crossprod( X, C);
	gsl_lapack_chol_invert( C );

	// Make C symmetric
	char uplo = 'U';
	gsl_matrix_triangular_to_full(C, uplo);

	// A = solve(crossprod(X)) %*% t(X)
	// A = C %*% t(X)
	gsl_matrix *A = gsl_matrix_alloc( C->size1, X->size1 );
	gsl_blas_dgemm( CblasNoTrans, CblasTrans, 1.0, C, X, 0.0, A );

	gsl_vector *S_beta_ii = gsl_matrix_product_diag( X, A);

	gsl_matrix_free( A );
	gsl_matrix_free( C );

	return S_beta_ii;	
}

double LRGPR_params::get_S_alpha_trace( const double delta ){

	double sum = 0;

	for(unsigned i=0; i<s->size; i++){
		sum += gsl_vector_get(s, i) / (gsl_vector_get(s, i) + delta);
	}

	return sum;	
}

gsl_vector *LRGPR_params::get_S_alpha_diag( const double delta ){

	gsl_vector *H_diag;

	if( ! proximalContamination ){

		// H = U %*% diag(s/(s+delta)) %*% t(U)
		//   = tcrossprod(U %*% diag(sqrt(s/(s+delta))))

		gsl_vector *a = gsl_vector_alloc( s->size );

		// a = sqrt(s/(s+delta))
		for( unsigned int i=0; i<s->size; i++){
			gsl_vector_set( a, i, sqrt(gsl_vector_get(s, i) / (gsl_vector_get(s, i) + delta) ) );
		}		

		// this method may be slow.  See LMM_get_S_alpha_diag() instead
		gsl_matrix *Ua = gsl_matrix_alloc( t_U->size2, t_U->size1 );
		gsl_matrix *t_Ua = gsl_matrix_alloc( t_U->size1, t_U->size2 );
		gsl_matrix *U = gsl_matrix_alloc( t_U->size2, t_U->size1 );

		gsl_matrix_transpose_memcpy( U, t_U);

		gsl_matrix_diagonal_multiply( U, a, Ua, false);

		gsl_matrix_transpose_memcpy( t_Ua, Ua);

		//gsl_matrix_print(U);
		//gsl_matrix_print(Ua);
		//gsl_vector_print(a);

		H_diag = gsl_matrix_product_diag( Ua, t_Ua);

		gsl_vector_free(a);
		gsl_matrix_free(Ua);
		gsl_matrix_free(t_Ua);
		gsl_matrix_free(U);

	}else{
		H_diag = gsl_vector_calloc( n_indivs );
	}	

	return H_diag;
}

double LRGPR_params::get_S_alpha_beta_trace( const double delta ){
	
	double trace;

	if( ! proximalContamination ){

		// diag( W ) = ss_delta = diag(s/(s+delta))
		gsl_vector *ss_delta = gsl_vector_alloc(s->size);
		for( unsigned int i=0; i<s->size; i++){
			gsl_vector_set( ss_delta, i, gsl_vector_get(s, i) / (gsl_vector_get(s, i) + delta) );
		}

		// A = t(Xu) %*% W %*% Xu
		// W_X = W %*% Xu
		gsl_matrix *W_X = gsl_matrix_alloc( t_U->size1, X_ncol );
		gsl_matrix *A = gsl_matrix_alloc( X_ncol, X_ncol );
		gsl_matrix_diagonal_quadratic_form( Xu, ss_delta, A, W_X);

		gsl_matrix *P = gsl_matrix_alloc( X->size2, X->size2 );
		gsl_matrix_crossprod( X, P);
		gsl_lapack_chol_invert( P );

		char uplo = 'U';
		gsl_matrix_triangular_to_full(P, uplo);

		// Get diag by quadratic multiplication
		trace = gsl_matrix_product_trace( A, P );

		gsl_matrix_free( P );
		gsl_matrix_free( W_X );
		gsl_matrix_free( A );
		gsl_vector_free(ss_delta);

	}else{
		trace = 0;
	}

	return trace;
}

gsl_vector *LRGPR_params::get_S_alpha_beta_diag( const double delta){

	gsl_vector *H_diag;

	if( ! proximalContamination ){

		/////////////
		// S_alpha //
		/////////////

		// diag( W ) = ss_delta = diag(s/(s+delta))
		gsl_vector *ss_delta = gsl_vector_alloc(s->size);
		for( unsigned int i=0; i<s->size; i++){
			gsl_vector_set( ss_delta, i, gsl_vector_get(s, i) / (gsl_vector_get(s, i) + delta) );
		}

		// M =  t(t_U) %*% W %*% t_U
		gsl_matrix *M = gsl_matrix_alloc( t_U->size2, t_U->size2);
		gsl_matrix_diagonal_quadratic_form( t_U, ss_delta, M);

		////////////
		// S_beta //
		////////////

		// C = solve(crossprod(X))
		gsl_matrix *C = gsl_matrix_alloc( X->size2, X->size2 );
		gsl_matrix_crossprod( X, C);
		gsl_lapack_chol_invert( C );

		// Make C symmetric
		char uplo = 'U';
		gsl_matrix_triangular_to_full(C, uplo);

		// A = X %*% solve(crossprod(X))
		// A = X %*% C
		gsl_matrix *A = gsl_matrix_alloc( X->size1, C->size2 );
		gsl_blas_dgemm( CblasNoTrans, CblasNoTrans, 1.0, X, C, 0.0, A );

		// B = X %*% solve(crossprod(X)) %*% t(X)
		// B = A %*% t(X)
		gsl_matrix *B = gsl_matrix_alloc( A->size1, X->size1 );
		gsl_blas_dgemm( CblasNoTrans, CblasTrans, 1.0, A, X, 0.0, B );

		// diag(S_beta) = diag( M %*% beta)
		H_diag = gsl_matrix_product_diag( M, B);

		gsl_matrix_free( A );
		gsl_matrix_free( B );
		gsl_matrix_free( C );
		gsl_matrix_free( M );
		gsl_vector_free(ss_delta);

	}else{
		H_diag = gsl_vector_calloc( n_indivs );		
	}	

	return H_diag;
}

double LRGPR_params::get_effective_df( const double delta ){

	double df;

	/*gsl_vector *S_beta_ii = get_S_beta_diag();
	gsl_vector *S_alpha_ii = get_S_alpha_diag( delta );
	double trace_S_alpha_beta = get_S_alpha_beta_trace( delta );

	 // double df = gsl_vector_sum_elements( S_beta_ii ) + gsl_vector_sum_elements( S_alpha_ii ) - trace_S_alpha_beta;

	gsl_vector_free( S_beta_ii );
	gsl_vector_free( S_alpha_ii );	*/

	df = get_S_beta_trace() + get_S_alpha_trace(delta) - get_S_alpha_beta_trace(delta);

	return df;
}

gsl_vector *LRGPR_params::get_hat_matrix_diag( const double delta ){
	
	gsl_vector *S_beta_ii = get_S_beta_diag();
	gsl_vector *S_alpha_ii = get_S_alpha_diag( delta );
	gsl_vector *S_alpha_beta_ii = get_S_alpha_beta_diag( delta );

	gsl_vector *Hii = gsl_vector_calloc( S_beta_ii->size );

	gsl_vector_add( Hii, S_beta_ii);
	gsl_vector_add( Hii, S_alpha_ii);
	gsl_vector_sub( Hii, S_alpha_beta_ii);

	gsl_vector_free( S_beta_ii );
	gsl_vector_free( S_alpha_ii );
	gsl_vector_free( S_alpha_beta_ii );

	return Hii;
}

// Can be done more efficientlly.  See get_S_alpha_diag
gsl_vector *LRGPR_params::get_fitted_response( const double delta, gsl_vector *alpha){
	
	//
	// alpha = decomp$vectors %*% diag(1/(1+delta/decomp$values), n) %*% crossprod(decomp$vectors, obj$y - X_beta)
	// Y_hat = alpha + X_beta

	// X_beta = obj$x %*% obj$coefficients
	gsl_vector *X_beta = gsl_vector_alloc( Y->size );
	gsl_blas_dgemv( CblasNoTrans, 1.0, X, beta, 0.0, X_beta );

	// Y_X_beta = Y - X_beta
	gsl_vector *Y_X_beta = gsl_vector_alloc( Y->size );
	gsl_vector_memcpy( Y_X_beta, Y);
	gsl_vector_sub( Y_X_beta, X_beta);

	// alpha = decomp$vectors %*% diag(1/(1+delta/decomp$values), n) %*% crossprod(decomp$vectors, obj$y - X_beta)
	////////////////////////////////////////////////////

	// diag( W ) = inv_s_delta = diag(s/(s+delta))
	for( unsigned int i=0; i<s->size; i++){
		gsl_vector_set( inv_s_delta, i, gsl_vector_get( s, i )/(gsl_vector_get( s, i ) + delta) );
	}
	// M =  t(t_U) %*% W %*% t_U
	gsl_matrix *M = gsl_matrix_alloc( t_U->size2, t_U->size2);
	gsl_matrix_diagonal_quadratic_form( t_U, inv_s_delta, M);

	//gsl_vector *alpha = gsl_vector_alloc( Y->size );
	gsl_blas_dgemv( CblasNoTrans, 1.0, M, Y_X_beta, 0.0, alpha );
	//gsl_blas_dgemv( CblasNoTrans, 1.0, M, Y, 0.0, alpha );

	gsl_vector *Y_hat = gsl_vector_calloc( Y->size );
	gsl_vector_add( Y_hat, X_beta);
	gsl_vector_add( Y_hat, alpha);

	//gsl_vector_print( alpha);

	gsl_vector_free( X_beta );
	gsl_vector_free( Y_X_beta );
	//gsl_vector_free( alpha );
	gsl_matrix_free( M );

	return Y_hat;
}

double LRGPR_params::get_SSE( const double delta ){

	gsl_vector *alpha = gsl_vector_alloc( Y->size );
	gsl_vector *Y_hat = get_fitted_response( delta, alpha );

	// Y_hat = Y_hat - Y
	gsl_vector_sub(Y_hat, Y);

	double SSE;
	gsl_blas_ddot(Y_hat, Y_hat, &SSE);

	gsl_vector_free( alpha );
	gsl_vector_free( Y_hat );

	return( SSE );	
}