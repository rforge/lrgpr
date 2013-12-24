/*
 * pGLM.cpp
 *
 *  Created on: Feb 12, 2013
 *      Author: gh258
 */

#ifndef PGLM_H_
#define PGLM_H_

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_permutation.h>

#include "gsl_additions.h"

typedef enum {LASSO, ADAPTIVE_LASSO, ELASTIC_NET, ADAPTIVE_ELASTIC_NET, SCAD, MCP, NEG, BRIDGE, NONE, VBAY, LOG} penaltyType;

const double ACTIVE_MODEL_EFFECTIVELY_ZERO = 1e-5;
const double DELTA_BETA_TOLERANCE = 1e-5;
const double DELTA_LOG_LIKELIHOOD_TOLERANCE = 1e-9;

/** Define preallocated memory that can be reused by multiple calls to GLM_penalized_regression()
 *
 */
struct GLM_workspace{
	gsl_vector *mu;
	gsl_vector *eta;
	gsl_vector *v;
};

/**
 * Allocate reusable memory to be used by GLM_penalized_regression
 */
GLM_workspace *GLM_workspace_alloc(const int n_indivs);
GLM_workspace *GLM_workspace_calloc(const int n_indivs);

/**
 * Free reusable memory
 */
void GLM_workspace_free(GLM_workspace *workspace);

void check_hyperparameter_values( penaltyType penalty_enum, const double hyper_param1, const double hyper_param2 );

double eval_objective_function(const gsl_vector *Y, const gsl_vector *eta, const regressionType regress_type, int *exclude_indeces = NULL, int exclude_count = 0);

double eval_log_likelihood(const gsl_vector *Y, const gsl_vector *eta, const regressionType regress_type, int *exclude_indeces = NULL, int exclude_count = 0);

double soft_threashold(const double d, const double lambda);

/**
 * Evaluates the sum of the penalty on the coefficient vector given the penalty type and hyperparameters
 *
 * @param beta regression coefficients
 * @param coeff_weights weights for adaptive lasso and elastic_net
 * @param hyper_param1 \f$ \lambda_1 \f$ for all penalties
 * @param hyper_param2 \f$ \lambda_2 \f$ for elastic-net, \f$ a \f$ for MCP and SCAD, \f$ \gamma \f$ for NEG and \f$ q \f$ for Lq
 * @param penalty_enum specify the penalty type
 */
double penalty(const gsl_vector * __restrict__ beta, const double hyper_param1, const double hyper_param2, penaltyType penalty_enum,
		const gsl_vector * __restrict__ coeff_weights = NULL);


inline double mcp(const double beta, const double lambda, const double a);
inline double scad(const double beta, const double lambda, const double a);
inline double bridge(const double beta, const double lambda, const double gamma);
inline double log_penalty(const double beta, const double lambda, const double epsilon);

inline double mcp_derivative(const double beta, const double lambda, const double a);
inline double scad_derivative(const double beta, const double lambda, const double a);
double bridge_derivative(const double beta, const double lambda, const double gamma);

struct GLM_penRegInfo {
	vector<double> logLikArray;
	vector<double> nIterArray;
	 double max_unselected_beta, mid_unselected_beta, delta;
	 int n_iter, n_beta_active;
};
/*
 * if initialized == true, re-evaluate eta and mu
 */
double GLM_penalized_regression(const gsl_vector * Y, const gsl_matrix *t_X, const gsl_matrix *X_cov, gsl_vector *beta, gsl_vector *beta_cov,
		regressionType regress_enum, penaltyType penalty_enum, const double hyper_param1, const double hyper_param2,
		const gsl_vector *feature_scales, const gsl_vector *covariate_scales, const gsl_vector *feature_weights, GLM_workspace *workspace,
		const gsl_permutation *permutation, const int maxit, const double thresh, GLM_penRegInfo *info, const bool initialized = false);

void LMM_penalized_regression_on_residuals(gsl_vector *v, const gsl_matrix *t_X, gsl_vector *beta, penaltyType penalty_enum,
		const double hyper_param1, const double hyper_param2, const gsl_vector *feature_scales, const gsl_permutation *permutation,
		const int maxit, GLM_penRegInfo *info, const bool initialized, const gsl_vector *inv_s_delta);

#endif
