


#ifndef GLM_H_
#define GLM_H_

#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_permutation.h>

#include <vector>
#include <string>
#include <math.h>

#include "gsl_additions.h"

using namespace std;

/*
 * GLM_unpenalized_regression_IRLS is called hundreds of thousands of times on systems of the same size,
 * so this struct allows workspace variables to be alloc'ed only once
 */
struct GLM_IRLS_workspace{

	int n_indivs, n_markers;
	gsl_matrix *Xm_t_Xm;
	gsl_matrix *Xm_t_Xm_ginv_X;
	gsl_vector *z, *mu, *beta_update, *beta_old;
	gsl_vector *eta, *w;
	gsl_matrix *W_X;
	gsl_matrix *Sigma;
	bool evaluate_Sigma;
	bool design_is_singular;
};

GLM_IRLS_workspace *GLM_IRLS_workspace_alloc(const int n_indivs_in, const int n_markers_in, const bool evaluate_Sigma_in=false);

// Returns true iff the internal variables have been allocated to the dimensions given by n_indivs_in and n_markers_in and irls_workspace != NULL
bool GLM_IRLS_workspace_correct_size(const GLM_IRLS_workspace *irls_workspace, const int n_indivs_in, const int n_markers_in);

void GLM_IRLS_workspace_free(GLM_IRLS_workspace *irls_workspace);

/*
 * Fit the regression coefficients for Y ~ X using a LINEAR or LOGISTIC link
 * Calculate the standard deviations and return a gsl_vector of p-values
 * If the logistic system is separable, (i.e. mu's are very extreme) then the system is computationally singular
 * and the p_values are all set to 1
 */
gsl_vector *GLM_unpenalized_regression_IRLS(const gsl_vector * Y, const gsl_matrix *X, gsl_vector *beta, double *sig_e, double *log_L, int *rank,
		regressionType regress_enum, bool report_p_values = true, GLM_IRLS_workspace *irls_workspace = NULL);


struct GLM_MV_workspace {

	int n_indivs, n_col, n_resp;
	gsl_matrix *C;
	gsl_matrix *C_X;
	gsl_matrix *Beta;
	gsl_matrix *Eta;
	gsl_matrix *Residuals;
	gsl_vector *sigSq;
	gsl_vector *logLik;

	regressionType regress_enum;

	bool eval_cov, eval_logLik;

	bool design_is_singular;

	bool eval_mv_hypothesis;

	gsl_matrix *E, *Beta_sub, *H, *A, *EH;
	gsl_vector *lambda;
};


GLM_MV_workspace *GLM_MV_workspace_alloc(const int n_indivs, const int n_col, const int n_resp, const bool eval_cov=true, const bool eval_logLik=true);

GLM_MV_workspace *GLM_MV_workspace_alloc(const int n_indivs, const int n_col, const int n_resp, const bool eval_cov, const bool eval_logLik);

bool GMVLM_workspace_correct_size(const GLM_MV_workspace *work, const int n_indivs, const int n_col, const int n_resp);

void GLM_MV_workspace_free(GLM_MV_workspace *work);

struct LM_preproc {
	gsl_matrix *P;
	gsl_matrix *PY;
	gsl_matrix *cp_Y;
	gsl_matrix *cp_PY; 	
	gsl_matrix *cp_W; 
	gsl_matrix *W;
};

LM_preproc *LM_preproc_alloc( const gsl_matrix *Y, const gsl_matrix *W);

void LM_preproc_free( LM_preproc * preproc);

struct Preproc_workspace {
	gsl_matrix *Beta_sub;
	gsl_matrix *PX;
	gsl_matrix *cp_X;
	gsl_matrix *cp_YX;
	gsl_matrix *cp_PX;
	gsl_matrix *cp_PYPX;
	gsl_matrix *C;
	gsl_matrix *cp_YX_sub;
	gsl_matrix *SSE;
	gsl_matrix *beta_cp_X;
	gsl_matrix *beta_cp_PX;
	gsl_matrix *XW;
	gsl_matrix *Sigma;
	gsl_matrix *Sigma_sub;
};

Preproc_workspace *Preproc_workspace_alloc( const gsl_matrix *Y, const int n_snp_terms, LM_preproc * prep);

void Preproc_workspace_free( Preproc_workspace * preproc);





/*
Optimized to perform multivariate linear regression but preprocessing design matrix
	for use by multiple responses.
Currently only supports univariate hypothesis tests
*/
void GLM_regression( const gsl_matrix *Y, const gsl_matrix *X, regressionType regress_enum, GLM_MV_workspace *work=NULL);

/*
A very simple wrapper that takes a gsl_vector as the response
*/
void GLM_regression( const gsl_vector *y, const gsl_matrix *X, regressionType regress_enum, GLM_MV_workspace *work=NULL);

gsl_vector * GLM_regression_preproc( const gsl_matrix *Y, const gsl_matrix *X, GLM_MV_workspace *work, LM_preproc *preproc, Preproc_workspace *preproc_work);

/*
Performs Wald test optimized for multivariate regression,
	stat = t(beta[i]) %*% Sigma[i,i] %*% beta[i]

Letting Sigma = C * sigSq, calculate stat with C and then divide by sigSq afterward
	so that sigSq is involved in scaled rather than matrix multiplcation 
*/
gsl_vector * GLM_wald_test( GLM_MV_workspace *work, const vector<int> & terms);


double GLM_wald_test( const gsl_vector *beta, const gsl_matrix *Sigma, const int n_indivs, const vector<int> & terms, regressionType regress_enum, const bool design_is_singular);


/*
Performs multivariate test of association using the Hotelling and Pillai test
	based on

	 http://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_introreg_sect012.htm
	 http://www.math.wustl.edu/~sawyer/handouts/multivar.pdf
	 http://www.psych.yorku.ca/lab/psy6140/lectures/MultivariateRegression2x2.pdf
*/
void GLM_HotellingPillai_test(  const gsl_matrix *X, GLM_MV_workspace *work, const vector<int> & terms, double *Hotelling, double *Pillai);





#endif
