#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_matrix_double.h>

class LRGPR_params {

public:

	// only the pointer is copied, and nothing new is malloc'ed, so don't free
	gsl_vector *s;
	gsl_matrix *t_U;

	int n_indivs, X_ncol, W_ncol, rank;

	double sig_g, delta, SSE_r, SSE_ru, Q_rr_value, ru_inv_s_delta_ru, Q_WW_1_logDet;

	bool isLowRank, proximalContamination, breakDueToSingularMatrix;

	gsl_vector *Y;
	gsl_vector *Yu;
	gsl_matrix *Xu;
	gsl_matrix *X;
	gsl_vector *beta;
	gsl_vector *inv_s_delta;
	gsl_matrix *W_til;
	gsl_matrix *Wu;
	gsl_vector *r;
	gsl_vector *ru;

	gsl_matrix *cp_X_low;
	gsl_matrix *cp_Xu;
	gsl_vector *cp_X_low_Y_low;
	gsl_vector *cp_Xu_Yu;
	gsl_matrix *cp_W_low;
	gsl_matrix *cp_Wu;
	gsl_matrix *cp_W_low_X;
	gsl_matrix *cp_Wu_Xu;
	gsl_vector *cp_W_low_Y_low;
	gsl_vector *cp_Wu_Yu;
	gsl_matrix *cp_W_low_X_low;
	gsl_vector *cp_W_til_r;

	gsl_matrix *Q_XX_value;
	gsl_vector *Q_Xy_value;
	gsl_matrix *Q_XW_value;
	gsl_vector *Q_Wy_value;
	gsl_matrix *Q_WW_value;
	gsl_vector *Q_Wr_value;
	gsl_matrix *Q_XW_WW_value;
	gsl_matrix *inv_s_delta_Xu;
	gsl_vector *inv_s_delta_Yu;
	gsl_vector *inv_s_delta_ru;
	gsl_matrix *inv_s_delta_Wu;

	gsl_matrix *Xu_inv_s_delta_Xu;
	gsl_vector *Xu_inv_s_delta_Yu;
	gsl_matrix *Wu_inv_s_delta_Wu;
	gsl_vector *Wu_inv_s_delta_Yu;
	gsl_vector *Wu_inv_s_delta_ru;	
	gsl_matrix *Xu_inv_s_delta_Wu;

	/*gsl_matrix *cp_X_low_delta;
	gsl_vector *cp_X_low_Y_low_delta;
	gsl_matrix *cp_W_low_X_low_delta;
	gsl_matrix *cp_W_low_delta;
	gsl_vector *cp_W_low_Y_low_delta;*/

	gsl_matrix *Q_XW_WW_Q_Xw_value;
	gsl_matrix *solve_I_Q_WW;

	// constructor
	LRGPR_params( const gsl_vector *Y_, const gsl_matrix *t_U_, const gsl_vector *eigenValues, const int X_ncol_, const int W_ncol_ = 0);

	void update_X( const gsl_matrix *X_ );
	void update_Xu( const gsl_matrix *X_, const gsl_matrix *Xu_ );	
	void update_Y( const gsl_vector *Y_ );
	void update_Wtilde( const gsl_matrix *W_til_ );

	void get_beta( gsl_vector *beta_ );

	gsl_vector *wald_test();

	/**
	 * \brief Compute standard deviation of regression coefficients to be used in the Wald test
	 */
	void standard_deviation( const double delta, gsl_vector *sd);

	/**
	 * \brief Return the variance-covariance matrix of \hat{beta} from which standard_deviation (should be) calculated
	 */
	gsl_matrix *coeff_covariance(const double delta);

	/*!
	 * \brief S_beta_ii = diag( X %*% crossprod(X) %*% t(X) )
	 */
	double get_S_beta_trace();
	gsl_vector *get_S_beta_diag();

	/*!
	 * \brief Compute S_alpha_ii =  s/(s+delta)
	 */
	double get_S_alpha_trace( const double delta );
	gsl_vector *get_S_alpha_diag( const double delta );

	/*!
	 * \brief trace(S_alpha_beta) = trace(S_alpha %*% S_beta) =
	 * tr( I + S^{-2}\delta)^{-1} U^T  X}  ( X^T X)^{-1} X^T  U )
	 *
	 * But note that the matrix inside is not S_alpha %*% S_beta due to trace() being a circular operator
	 * See LMM_correct_test.pdf for more details
	 */
	double get_S_alpha_beta_trace( const double delta );

	/*!
	 * \brief Evaluate the diag of S_alpha_beta for the full rank model using
	 * S_alpha_beta = S_alpha %*% S_beta = U ( I +  S^-2 delta)^-1 U^T  X ( X^T X)^{-1} X^T
	 */
	gsl_vector *get_S_alpha_beta_diag( const double delta);

	/*!
	 * \brief Get degrees of freedom = sum diagonals of smoother matrix
	 */
	double get_effective_df( const double delta );


	/*!
	 *  \brief  Compute diagonals complete smoother matrix
	 *  Since H = S_beta + S_alpha - S_alpha_beta, return diag(H), where only the diagonals were calculated
	 *
	 *   \hat{y} = X\hat{\beta} + \hat{R\gamma}
	 *  = S(delta) y
	 *  where
	 *  C = solve(crossprod(X)) %*% t(X)
	 *  Q = U(solve(I+ delta*S^-2)^-1 U^T
	 *
	 *  S = [ C + Q - QC]
	 *  df = trace( S ) = trace(C) + trace(Q) + trace(QC)
	 */
	gsl_vector *get_hat_matrix_diag( const double delta );

	/*
	 * X_beta = obj$x %*% obj$coefficients
	   alpha = decomp$vectors %*% diag(1/(1+delta/decomp$values), n) %*% crossprod(decomp$vectors, obj$y - X_beta)
	   Y_hat = alpha + X_beta

	   	  P_prep =
	 *
	 */
	gsl_vector *get_fitted_response( const double delta, gsl_vector *alpha);

	double get_SSE( const double delta );

	// destructor
	~LRGPR_params();

private:

	void init_as_null();

	// Update the dimensions of the temporary variables
	void alloc_Y( const int n_indivs_ );
	void alloc_rank( const int rank_ );
	void alloc_X( const int X_ncol_ ); 
	void alloc_W( const int W_ncol_ ); 

	bool init_variables_Y;
	bool init_variables_rank;
	bool init_variables_X;
	bool init_variables_W;
};
