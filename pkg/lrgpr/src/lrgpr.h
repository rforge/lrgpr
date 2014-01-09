
/*
 * lrgpr.h
 *
 *  Created on: November, 12. 2013
 *      Author: gh258
 */
#include <vector>

#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_matrix_double.h>

#include "lrgpr_params.h"

#include "gsl_lapack.h"

using namespace std;


class LRGPR {

public:	

	gsl_vector *delta_grid;

	// constructor
	LRGPR( const gsl_vector *Y_, const gsl_matrix *t_U_, const gsl_vector *eigenValues, const int X_ncol_, const int W_ncol_ = 0);

	// load new X, rotate is by t(U), and reallocate memory if necessary
	/*!
	 * \brief Preprocess new set of covariates to be used in next model.  X is transformed by t_U
	 * \param X new covariates for statistical model
	 * \param eigen spectral decomposition of kernal matrix used to transform X
	 */
	void update_X( const gsl_matrix *X_ );

	void update_Xu( const gsl_matrix *X_, const gsl_matrix *Xu_ );

	void update_Y( const gsl_vector *Y_ );

	void update_Wtilde( const gsl_matrix *W_til_ );

	/*!
	 * \brief Set the grid of delta intervals to be searched.  Values should be in actual space, not log-space
	 */
	void set_delta_grid( const double left, const double right, const int n_intervals);

	void fit_mle( double *log_L, double *sig_g, double *sig_e );

	void fit_fixed_delta( const double delta, double *log_L, double *sig_g, double *sig_e );

	gsl_vector *get_hat_matrix_diag();	
	void get_hat_matrix_diag( gsl_vector * res);

	gsl_vector *get_fitted_response( gsl_vector *alpha );

	gsl_matrix *coeff_covariance();
	void coeff_covariance( gsl_matrix * res);

	gsl_vector *get_beta();
	void get_beta( gsl_vector *res);

	gsl_vector *wald_test_all();

	double wald_test( vector<int> &terms );

	double get_effective_df();

	// destructor
	~LRGPR();

	friend double _LRGPR_fxn( const double delta, void *params_arg);

private:

	LRGPR_params *params;
	
	inline double logLikelihood_standard( const double delta );

	inline double logLikelihood_lowrank( const double delta );

	inline double logLikelihood( const double delta );

	/**
	 * \brief Estimate \beta in the LMM based on Y, X, U, delta and s using ordinary least squares as the used in the standard LMM model
	 *
	 * \param Yu adjusted responses U^T Y
	 * \param Xu adjusted covariates U^T X
	 * \param s return by reference the eigen values of K
	 * \param delta the ratio of variance represented by delta = sig_e / sig_a
	 * \param beta return to reference the regression coefficients of Yu ~ Xu
	 */
	void estimate_beta( const double delta);

	bool lowrank_preprocessed;

	inline void Q_XX(const double delta);
	inline void Q_Xy(const double delta);
	inline void Q_rr(const double delta);
	inline void Q_XW(const double delta);
	inline void Q_Wy(const double delta);
	inline void Q_Wr(const double delta);
	inline void Q_WW(const double delta);

	inline void Omega_XX(const double delta);
	inline void Omega_Xy(const double delta);
	inline void Omega_rr(const double delta);
};

/**
 * \brief Calls LMM_logLikelihood() but passes residuals and s in a struct labeled here as void pointer.
 * This functional for is need to cast it is a gsl_function in order to minimize the function
 */
double _LRGPR_fxn( const double delta, void *params_arg);


















