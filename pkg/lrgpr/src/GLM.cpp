#include "GLM.h"

#include <iostream>
#include <limits>

#include <gsl/gsl_cdf.h>
#include <gsl/gsl_linalg.h>

#include "gsl_additions.h"
#include "gsl_lapack.h"
#include "pGLM.h"

#include "misc_functions.h"

using namespace std;

GLM_IRLS_workspace *GLM_IRLS_workspace_alloc(const int n_indivs_in, const int n_markers_in, const bool evaluate_Sigma_in){

	GLM_IRLS_workspace *irls_workspace = (GLM_IRLS_workspace *) malloc(sizeof(GLM_IRLS_workspace));

	irls_workspace->n_indivs = n_indivs_in;
	irls_workspace->n_markers = n_markers_in;
	irls_workspace->Xm_t_Xm = gsl_matrix_alloc(n_markers_in, n_markers_in);
	irls_workspace->Xm_t_Xm_ginv_X = gsl_matrix_alloc(n_markers_in, n_indivs_in);
	irls_workspace->z = gsl_vector_alloc(n_indivs_in);
	irls_workspace->mu = gsl_vector_alloc(n_indivs_in);
	irls_workspace->beta_update = gsl_vector_alloc(n_markers_in);
	irls_workspace->beta_old = gsl_vector_alloc(n_markers_in);
	irls_workspace->eta = gsl_vector_alloc(n_indivs_in);
	irls_workspace->w = gsl_vector_alloc(n_indivs_in);
	irls_workspace->W_X = gsl_matrix_alloc(n_indivs_in, n_markers_in);

	irls_workspace->evaluate_Sigma = evaluate_Sigma_in;

	if( irls_workspace->evaluate_Sigma ){
		irls_workspace->Sigma = gsl_matrix_alloc( irls_workspace->n_markers, irls_workspace->n_markers );
	}else{
		irls_workspace->Sigma = NULL;
	}

	irls_workspace->design_is_singular = false;

	return irls_workspace;
}

bool GLM_IRLS_workspace_correct_size(const GLM_IRLS_workspace *irls_workspace, const int n_indivs_in, const int n_markers_in){

	return( irls_workspace != NULL && irls_workspace->n_indivs == n_indivs_in && irls_workspace->n_markers == n_markers_in);
}

void GLM_IRLS_workspace_free(GLM_IRLS_workspace *irls_workspace){
	gsl_matrix_free(irls_workspace->Xm_t_Xm);
	gsl_matrix_free(irls_workspace->Xm_t_Xm_ginv_X);
	gsl_vector_free(irls_workspace->z);
	gsl_vector_free(irls_workspace->mu);
	gsl_vector_free(irls_workspace->beta_update);
	gsl_vector_free(irls_workspace->beta_old);
	gsl_vector_free(irls_workspace->eta);
	gsl_vector_free(irls_workspace->w);
	gsl_matrix_free(irls_workspace->W_X);

	if( irls_workspace->Sigma != NULL){
		gsl_matrix_free( irls_workspace->Sigma );
	}

	free(irls_workspace);
}




GLM_MV_workspace *GLM_MV_workspace_alloc(const int n_indivs, const int n_col, const int n_resp, const bool eval_cov, const bool eval_logLik){

	GLM_MV_workspace *work = (GLM_MV_workspace *) malloc(sizeof(GLM_MV_workspace));

	work->n_indivs 	= n_indivs;
	work->n_col 	= n_col;
	work->n_resp 	= n_resp;
	work->C 		= gsl_matrix_alloc(n_col, n_col);
	work->C_X 		= gsl_matrix_alloc(n_col, n_indivs);
	work->Beta 		= gsl_matrix_alloc(n_col, n_resp);
	work->Eta 		= gsl_matrix_alloc(n_indivs, n_resp);
	work->Residuals = gsl_matrix_alloc(n_indivs, n_resp);

	work->regress_enum = LINEAR;

	work->eval_cov = eval_cov;
	work->eval_logLik = eval_logLik;

	work->logLik = NULL;
	work->sigSq =  NULL;

	if( eval_cov ) 		work->sigSq = gsl_vector_alloc( n_resp);
	if( eval_logLik ) 	work->logLik = gsl_vector_alloc( n_resp);

	work->design_is_singular = false;
	work->eval_mv_hypothesis = false;

	work->E = NULL;
	work->Beta_sub = NULL;
	work->H = NULL;
	work->A = NULL;
	work->EH = NULL;
	work->lambda = NULL;

	return work;
}

bool GLM_MV_workspace_correct_size(const GLM_MV_workspace *work, const int n_indivs, const int n_col, const int n_resp){

	return( (work != NULL) && (work->n_indivs == n_indivs) && (work->n_col == n_col) && (work->n_resp == n_resp));
}

void GLM_MV_workspace_free(GLM_MV_workspace *work){
	
	gsl_matrix_free(work->C);
	gsl_matrix_free(work->C_X);
	gsl_matrix_free(work->Beta);
	gsl_matrix_free(work->Eta);
	gsl_matrix_free(work->Residuals);

	if( work->eval_cov ) gsl_vector_free( work->sigSq);
	if( work->eval_logLik ) gsl_vector_free( work->logLik);

	if( work->eval_mv_hypothesis ){		
		gsl_matrix_free(work->E);
		gsl_matrix_free(work->Beta_sub);
		gsl_matrix_free(work->H);
		gsl_matrix_free(work->A);
		gsl_matrix_free(work->EH);
		gsl_vector_free(work->lambda);
	}

	free(work);
}


/*

Multiplying crossprod(X, Y) first is 2-4x faster

n = 1000
p = 500
c = 10

# matrix of responses
Y = matrix( rnorm(n*p), n,p)

# marker matrix
X = matrix( sample(0:2, n*c, replace=TRUE), n)


system.time( replicate(1000, tcrossprod(chol2inv(chol(crossprod(X))), X)%*%Y))

system.time( replicate(1000, chol2inv(chol(crossprod(X))) %*% crossprod(X, Y)))
*/


void GLM_regression( const gsl_vector *y, const gsl_matrix *X, regressionType regress_enum, GLM_MV_workspace *work){

	gsl_matrix *Y = gsl_matrix_attach_array( y->data, y->size, 1);

	GLM_regression( Y, X, regress_enum, work);

	// Since the attach method is used, this should free Y
	//	but leave y->data untouched
	gsl_matrix_free( Y );
}

void GLM_regression( const gsl_matrix *Y, const gsl_matrix *X, regressionType regress_enum, GLM_MV_workspace *work){

	work->regress_enum = regress_enum;

	unsigned int n_indivs = Y->size1;
	unsigned int n_response = Y->size2;
	unsigned int n_col = X->size2;
	
	double sigma_sq_hat, log_L;

	bool alloc_work_locally = false;

	if( ! GLM_MV_workspace_correct_size(work, n_indivs, n_col, n_response) ){
		
		work = GLM_MV_workspace_alloc(n_indivs, n_col, n_response, true, false);

		alloc_work_locally = true;
	}
	
	// Solve multivariate least squares
	////////////////////////////////////

	// C = crossprod(X)
	gsl_matrix_crossprod(X, work->C);

	// C = chol2inv(chol(C)) 
	int res = gsl_lapack_chol_invert( work->C );

	work->design_is_singular = false;
	if( res != GSL_SUCCESS ){
		work->design_is_singular = true;
		return;
	}

	// C %*% t(X) where C is symmetric and upper triangular
	//gsl_blas_dsymm(CblasRight, CblasUpper, 1.0, work->C, X, 0.0, work->C_X);

	// BLAS does not support symmetric and transpose of other matrix, so use dgemm
	gsl_matrix_triangular_to_full(work->C, 'U');

	// Calculating Beta these two ways is the same speed
	//if( 1 ){
		gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, work->C, X, 0.0, work->C_X);

		// Beta = C_X %*% Y = solve(crossprod(X)) %*% t(X) %*% Y
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, work->C_X, Y, 0.0, work->Beta);

	/*}else{

		gsl_matrix *XY = gsl_matrix_alloc( X->size2, Y->size2 );

		gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, X, Y, 0.0, XY);

		// Beta = C %*% XY = chol2inv(chol(crossprod(X))) %*% crossprod(X, Y))
		gsl_blas_dsymm(CblasLeft, CblasUpper, 1.0, work->C, XY, 0.0, work->Beta);

		gsl_matrix_free(XY);
	}*/

	// Eta = X %*% Beta
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, X, work->Beta, 0.0, work->Eta);

	// Residuals = Y - Eta
	gsl_matrix_memcpy( work->Residuals, Y);
	gsl_matrix_sub( work->Residuals, work->Eta);

	double SSE_sqrt, SSE;
	gsl_vector_view resid_k, resp_k, eta_k;

	// For each response, compute Residual Sum of Squares (SSE = RSS = sigSq)
	// Also compute log-likelihood
	for(unsigned int k=0; k<Y->size2; k++){

		// Get Y[,k]
		resid_k = gsl_matrix_column( work->Residuals, k );

		// sqrt(SSE)
		SSE_sqrt = gsl_blas_dnrm2( &resid_k.vector );

		SSE = SSE_sqrt * SSE_sqrt;

		// get residual variance
		// Use finite sample size correction
		sigma_sq_hat = SSE / (double) (n_indivs - n_col);
		
		gsl_vector_set( work->sigSq, k, sigma_sq_hat );	

		if( work->eval_logLik ){
			// Log-likelihood using pre-comptued values
			log_L = -n_indivs/2.0 * ( log(2*M_PI) + log(sigma_sq_hat) + 1);	

			gsl_vector_set( work->logLik, k, log_L );	
		}
	}

	if( alloc_work_locally ) GLM_MV_workspace_free( work );
}



gsl_vector *GLM_wald_test( GLM_MV_workspace *work, const vector<int> & terms){

	gsl_vector *pValues = gsl_vector_alloc( work->Beta->size2 );

	// If design matrix is  singular, set all p-values to NAN
	if( work->design_is_singular ){

		for(unsigned int k=0; k<work->Beta->size2; k++){
			gsl_vector_set(pValues, k, NAN);
		}

		return pValues;
	}

	// Sigma = C * sigSq

	// Get subset of beta and Sigma as determined by terms
	//////////////////////////////////////////////////////

	gsl_matrix *C_sub = gsl_matrix_alloc( terms.size(), terms.size() );
	gsl_vector *beta_sub = gsl_vector_alloc( terms.size() );

	// get var(beta) matrix	
	gsl_matrix_sub_row_col( work->C, terms, C_sub);
	gsl_lapack_chol_invert( C_sub );

	gsl_vector_view beta_view;

	double p_val;

	// for each response
	for(unsigned int k=0; k<work->Beta->size2; k++){

		beta_view = gsl_matrix_column( work->Beta, k );

		gsl_vector_subset( &beta_view.vector, terms, beta_sub );
		
		// stat = t(beta[i]) %*% Sigma[i,i] %*% beta[i]
		double stat = gsl_matrix_quadratic_form_sym( C_sub, beta_sub);

		// Since C is the unscaled covariance, now scale by sigSq (the residual error estimate)
		stat /= gsl_vector_get( work->sigSq, k);

		// For small samples sizes, the Cholesky decomposition 
		// 		doesn't fail when it is supposed to
		// But set p-value to NAN if the test statistical is close enoght to zero
		if( stat < 1e-14 ){
			p_val = NAN;
		}else if( work->regress_enum == LINEAR){
			// Student-t null for normal model
			// This p-value is exact for finite sample sizes	
			if( terms.size() == 1){
				p_val = 2*gsl_cdf_tdist_Q( sqrt(stat), work->n_indivs - work->Beta->size1);
			}else{
				// Else Use chisq null	
				// For a a high dimensional hypothesis test, this is 
				//	asymptotically correct			
				p_val = gsl_cdf_chisq_Q( stat, terms.size() );				
			}
		}else{		
			// Gaussuan null for logistic model
			// this is asymptotically correct			
			p_val = gsl_cdf_chisq_Q( stat, terms.size() );
		}

		gsl_vector_set(pValues, k, p_val);
	}

	gsl_matrix_free( C_sub );
	gsl_vector_free( beta_sub );

	return pValues;
} 

double GLM_wald_test( const gsl_vector *beta, const gsl_matrix *Sigma, const int n_indivs, const vector<int> & terms, regressionType regress_enum, const bool design_is_singular){

	// If design matrix is  singular, set p-value to NAN
	if( design_is_singular ){
		return NAN;
	}

	// Get subset of beta and Sigma as determined by terms
	//////////////////////////////////////////////////////

	gsl_matrix *C_sub = gsl_matrix_alloc( terms.size(), terms.size() );
	gsl_vector *beta_sub = gsl_vector_alloc( terms.size() );

	// get var(beta) matrix	
	gsl_matrix_sub_row_col( Sigma, terms, C_sub);
	gsl_lapack_chol_invert( C_sub );

	// get subset of beta
	gsl_vector_subset( beta, terms, beta_sub );
	
	// stat = t(beta[i]) %*% Sigma[i,i] %*% beta[i]
	double stat = gsl_matrix_quadratic_form_sym( C_sub, beta_sub);

	double p_val;

	// For small samples sizes, the Cholesky decomposition 
	// 		doesn't fail when it is supposed to
	// But set p-value to NAN if the test statistical is close enoght to zero
	if( stat < 1e-14 ){
		p_val = NAN;
	}else if( regress_enum == LINEAR){
		// Student-t null for normal model
		// This p-value is exact for finite sample sizes	
		if( terms.size() == 1){
			p_val = 2*gsl_cdf_tdist_Q( sqrt(stat), n_indivs - beta->size);
		}else{
			// Else Use chisq null	
			// For a a high dimensional hypothesis test, this is 
			//	asymptotically correct			
			p_val = gsl_cdf_chisq_Q( stat, terms.size() );				
		}
	}else{		
		// Gaussuan null for logistic model
		// this is asymptotically correct			
		p_val = gsl_cdf_chisq_Q( stat, terms.size() );
	}

	gsl_matrix_free( C_sub );
	gsl_vector_free( beta_sub );

	return p_val;
} 



/*
beta
eta
residuals
sig_e
log_L


C = chol2inv(chol(crossprod(X)))

# get coefficients for all phenotypes
Beta = tcrossprod(C, X ) %*% Y

# evaluate the residuals
R = Y - X%*%Beta

E = crossprod(R)

sqrt( crossprod(Beta[3,,drop=FALSE], solve(C[3,3])) %*% Beta[3,,drop=FALSE] / (sigSq) )
*/


/*

terms = 2

#E = crossprod(fit$residuals)

#Beta = as.matrix(coef(fit))

L = matrix(0, length(terms), nrow(Beta))
for(j in 1:length(terms)){
	L[j,terms[j]] = 1
}

#H = t(L%*%Beta) %*% solve(L %*% solve(crossprod(X)) %*% t(L)) %*% (L %*% Beta)

H = Beta[terms,] %*% solve(solve(crossprod(X))[terms,terms]) %*% Beta[terms,]


lambda = Re(eigen(solve(E) %*% H, only.values=TRUE)$values)

Wilks_Lambda = sum(1/(1+lambda))

p = nrow(H)
q = length(terms) #fit$rank

nu = nrow(as.matrix(fit$fitted.values)) - q

s = min(q,p)
m = (abs(q - p) -1)/2

n = (nu - p -1)/2

r = nu - (p - q +1)/2
u = (p*q -2)/4
t = ifelse( p^2 + q^2 -5 > 0, sqrt((p^2*q^2-4)/(p^2+q^2-5)), 1)

F = (1 - Wilks_Lambda^(1/t) ) / Wilks_Lambda^(1/t) * (r*t - 2*u) / (p*q)

Wilks = pf( F, p*q, r*t-2*u, lower.tail=FALSE)

V = sum(lambda/(1+lambda))

F = (2*n + s +1)/(2*m + s +1) * V / (s-V)

Pillai = pf( F, s*(2*m+s+1), s*(2*n+s+1), lower.tail=FALSE)


U = sum(lambda)

b = (p+2*n)*(q+2*n) /(2*(2*n+1)*(n-1))

c = (2+(p*q+2)/(b-1))/(2*n)

if( n > 0 ){
	F = (U/c) *((4+(p*q+2)/(b-1)) / (p*q))
}else{

	F = 2*(s*n+1)*U / (s^2 *(2*m+s+1))
}

Hotelling = pf( F, p*q, 4+(p*q+2)/(b-1), lower.tail=FALSE)


r = max(p,q)

F = lambda[1] * (nu - r + q) / r

Roy = pf( F, r, nu - r + q, lower.tail=FALSE)

*/

void GLM_HotellingPillai_test( const gsl_matrix *X, GLM_MV_workspace *work, const vector<int> & terms, double *Hotelling, double *Pillai){

	if( ! work->eval_mv_hypothesis ){
		work->E 		= gsl_matrix_alloc( work->n_resp, work->n_resp );
		work->Beta_sub 	= gsl_matrix_alloc( terms.size(), work->Beta->size2 );
		work->H 		= gsl_matrix_alloc( work->n_resp, work->n_resp );
		work->A 		= gsl_matrix_alloc( terms.size(), terms.size() );
		work->EH 		= gsl_matrix_alloc( work->n_resp, work->n_resp );
		work->lambda 	= gsl_vector_alloc( work->n_resp);
		work->eval_mv_hypothesis = true;
	}

	// E = crossprod(fit$residuals)
	gsl_matrix_crossprod( work->Residuals, work->E );

	// E = solve(E)
	int status = gsl_lapack_chol_invert( work->E );

	// Beta = coef(fit)[terms,]
	gsl_matrix_sub_row( work->Beta, terms, work->Beta_sub);

	// H = Beta[terms,] %*% solve(solve(crossprod(X))[terms,terms]) %*% Beta[terms,]
	// C = solve(crossprod(X))
	// A = solve(solve(crossprod(X))[terms,terms])
	gsl_matrix_sub_row_col( work->C, terms, work->A);
	gsl_lapack_chol_invert( work->A );

	gsl_matrix_quadratic_form_sym( work->A, work->Beta_sub, work->H);

	// lambda = Re(eigen(solve(E) %*% H, only.values=TRUE)$values)	

	// If E is singular, set p-values to NAN
	if( status != GSL_SUCCESS ){
		*Pillai = NAN;
		*Hotelling = NAN;
		return;
	}

	gsl_blas_dsymm(CblasRight,CblasLower, 1.0, work->E, work->H, 0.0, work->EH);

	status = gsl_lapack_eigenValues( work->EH, work->lambda);

	// If EH is singular, set p-values to NAN
	if( status != GSL_SUCCESS ){
		*Pillai = NAN;
		*Hotelling = NAN;
		return;
	}

	//gsl_vector_print(work->lambda);

	// Compute F statistics
	double F_stat;

	double p = work->n_resp;
	double q = terms.size();
	double nu = work->n_indivs - q;
	double s = min(q,p);
	double m = (fabs(q - p) -1) / 2.0;
	double n = (nu - p -1) / 2.0;
	double r = nu - (p - q +1) / 2.0;
	double u = (p*q -2) / 4.0;
	double t = (p*q + q*q -5 > 0) ? sqrt((p*p*q*q-4)/(p*p+q*q-5)) : 1;

	// V = sum(lambda/(1+lambda))
	double V = 0;
	for(unsigned int i=0; i<work->lambda->size; i++){
		V += gsl_vector_get(work->lambda, i) / ( 1 + gsl_vector_get(work->lambda, i) );
	}

	F_stat = (2*n + s +1)/(2*m + s +1) * V / (s-V);

	//cout << "F_stat: " << F_stat << endl;
	//cout << "s*(2*m+s+1): " << s*(2*m+s+1) << endl;
	//cout << "s*(2*n+s+1) : " << s*(2*n+s+1)  << endl <<endl;

	//Pillai = pf( F, s*(2*m+s+1), s*(2*n+s+1), lower.tail=FALSE)
	*Pillai = gsl_cdf_fdist_Q( F_stat,  s*(2*m+s+1), s*(2*n+s+1) );


	double U = gsl_vector_sum_elements(work->lambda);
	double b = (p+2*n)*(q+2*n) /(2*(2*n+1)*(n-1));
	double c = (2+(p*q+2)/(b-1))/(2*n);

	if( n > 0 ){
		F_stat = (U/c) *((4+(p*q+2)/(b-1)) / (p*q));
	}else{

		F_stat = 2*(s*n+1)*U / (s*s *(2*m+s+1));
	}

	//Hotelling = pf( F, p*q, 4+(p*q+2)/(b-1), lower.tail=FALSE)
	*Hotelling = gsl_cdf_fdist_Q( F_stat, p*q, 4+(p*q+2)/(b-1) );

	//cout << "F_stat: " << F_stat << endl;
	//cout << "p*q: " << p*q << endl;
	//cout << "4+(p*q+2)/(b-1): " << 4+(p*q+2)/(b-1)  << endl<<endl;

	// Clean up
}



gsl_vector* GLM_unpenalized_regression_IRLS(const gsl_vector * Y, const gsl_matrix *X, gsl_vector *beta, double *sig_e, double *log_L, int *rank,
		regressionType regress_enum, bool report_p_values, GLM_IRLS_workspace *irls_workspace){

	unsigned int n_indivs = Y->size;
	unsigned int n_markers = X->size2;

	// Set beta to zero because IRLS can fail if initial value is NAN
	gsl_vector_set_zero(beta);

	// fit least squares model with LAPACK
	if( regress_enum == LINEAR ){

		gsl_vector *p = gsl_lapack_fit_least_squares( Y, X, beta, rank, sig_e, log_L, report_p_values);

		if( GLM_IRLS_workspace_correct_size(irls_workspace, n_indivs, n_markers) ){

			// irls_workspace->eta = X_m %*% beta
			gsl_blas_dgemv(CblasNoTrans, 1.0, X, beta, 0.0, irls_workspace->eta );
		}
	}

	*rank = X->size2;

	if(beta->size != n_markers){
		//cout << "beta->size != n_params" << endl;
	}

	gsl_matrix *Xm_t_Xm, *Xm_t_Xm_ginv_X;

	if( GLM_IRLS_workspace_correct_size(irls_workspace, n_indivs, n_markers) ){
		Xm_t_Xm 		= irls_workspace->Xm_t_Xm;
		Xm_t_Xm_ginv_X 	= irls_workspace->Xm_t_Xm_ginv_X;
	}else{
		Xm_t_Xm 		= gsl_matrix_alloc(n_markers, n_markers);
		Xm_t_Xm_ginv_X 	= gsl_matrix_alloc(n_markers, n_indivs);
	}
	///////////////////////////////////
	// Common to LINEAR and LOGISTIC //
	///////////////////////////////////

	// crossprod(Xm)
	gsl_matrix_crossprod(X, Xm_t_Xm);

	int info = 0;
	char uplo = 'U';
	int n = Xm_t_Xm->size1;


	dpotrf_(&uplo, &n, Xm_t_Xm->data, &n, &info);

	// If designMatrix does not span full space
	// i.e. t(X) %*% X does not have a valid Cholesky decomposition
	if( info != 0 ){
		//cerr << "Cholesky decomposition of singular matrix failed" << endl;

		*sig_e = NAN;
		*log_L = NAN;
		*rank = X->size2 - 1;

		gsl_vector_set_all( beta, NAN);

		if( report_p_values ){
			gsl_vector *p = gsl_vector_alloc( beta->size );
			gsl_vector_set_all( p, NAN);
			return( p );
		}else{
			return NULL;
		}
	}

	dpotri_(&uplo, &n, Xm_t_Xm->data, &n, &info);
	gsl_matrix_triangular_to_full(Xm_t_Xm, uplo);
	

	// solve(crossprod(Xm)) %*% t(X)
	gsl_blas_dgemm(CblasNoTrans,CblasTrans, 1.0, Xm_t_Xm, X, 0.0, Xm_t_Xm_ginv_X);

	// declare variables for LOGISTIC regression
	gsl_vector *z, *mu, *beta_update, *beta_old, *eta;

	if( GLM_IRLS_workspace_correct_size(irls_workspace, n_indivs, n_markers) ){
		eta = irls_workspace->eta;
	}else{
		eta = gsl_vector_alloc(n_indivs);
	}

	double log_likelihood, log_likelihood_prev;
	int status;

	bool separable_logistic_system = false;

	int back_track;

	// IRLS
	switch(regress_enum){
		case LINEAR:

			// ginv(crossprod(Xm)) %*% t(X) %*% Y
			gsl_blas_dgemv(CblasNoTrans, 1.0, Xm_t_Xm_ginv_X, Y, 0.0, beta);

			break;

		case LINEAR_MIXED_MODEL:
			perror("Linear mixed model not handled by this function\n");
			break;

		case LOGISTIC:

			//////////////////
			// MM algorithm //
			//////////////////

			// allocate memory
			if( GLM_IRLS_workspace_correct_size(irls_workspace, n_indivs, n_markers) ){
				z 			= irls_workspace->z;
				mu 			= irls_workspace->mu;
				beta_update = irls_workspace->beta_update;
				beta_old 	= irls_workspace->beta_old;
			}else{
				z 			= gsl_vector_alloc(n_indivs);
				mu 			= gsl_vector_alloc(n_indivs);
				beta_update = gsl_vector_alloc(n_markers);
				beta_old 	= gsl_vector_alloc(n_markers);
			}

			// A = 4*ginv(crossprod(Xm)) %*% t(X)
			gsl_matrix_scale(Xm_t_Xm_ginv_X, 4);

			int iter = 0;

			//double MM_delta_beta_threshold = 1e-8;

			int max_iter_MM = 2;
			int max_iter_IRLS = 25;

			log_likelihood_prev = -numeric_limits<double>::max();

			// update beta until change in log likelihood is small enough
			while(1){

				// eta = X %*% beta
				gsl_blas_dgemv(CblasNoTrans, 1.0, X, beta, 0.0, eta);

				// mu = inverse_logit(eta)
				inverse_logit(mu, eta);

				// z = Y - mu
				gsl_vector_subtr(Y, mu, z);

				// beta_update = A %*% z
				gsl_blas_dgemv(CblasNoTrans, 1.0, Xm_t_Xm_ginv_X, z, 0.0, beta_update);

				// beta = beta + beta_update
				gsl_vector_add(beta, beta_update);

				log_likelihood = eval_log_likelihood(Y, eta, regress_enum, NULL, 0);
				//log_likelihood = logistic_log_likelihood(Y, eta);
				//log_likelihood = logistic_log_likelihood_sse(Y, eta, mu);

				// If the system is separable, the log likelihood can be -Inf
				if( ! isfinite(log_likelihood) ){
					separable_logistic_system = true;
					//cout << "Separable logistic system" << endl;
					break;
				}

				//cout << "MM: " << log_likelihood - log_likelihood_prev << endl << flush;
				//cout << "MM: " << log_likelihood << endl;

				if( log_likelihood - log_likelihood_prev < 1e3 || iter > max_iter_MM){
				// if change in beta is small enough
					break;
				}
				log_likelihood_prev = log_likelihood;

				iter++;
			}

			///////////////////
			// Standard IRLS //
			///////////////////

			gsl_matrix *W_X;
			gsl_vector *w;

			if( ! separable_logistic_system){
				double mu_i;

				iter = 0;

				//cout << "IRLS" << endl;

				if( GLM_IRLS_workspace_correct_size(irls_workspace, n_indivs, n_markers) ){
					W_X = irls_workspace->W_X;
					w 	= irls_workspace->w;
				}else{
					W_X = gsl_matrix_alloc(n_indivs, n_markers);
					w 	= gsl_vector_alloc(n_indivs);
				}

				log_likelihood_prev = eval_log_likelihood(Y, eta, regress_enum, NULL, 0);
				//log_likelihood = logistic_log_likelihood(Y, eta);
				//log_likelihood = logistic_log_likelihood_sse(Y, eta, mu);

				while(1){

					// save copy of beta
					gsl_vector_memcpy(beta_old, beta);

					// mu = inverse_logit(eta)
					inverse_logit(mu, eta);

					// Make solve(t(X) %*% W %*% X)
					///////////////////////////////

					// W = diag( mu * (1-mu) )
					for(unsigned int i=0; i<n_indivs; i++){
						mu_i = gsl_vector_get(mu, i);
						gsl_vector_set(w, i, mu_i * (1-mu_i));
					}

					gsl_matrix_memcpy(W_X, X);

					/*
					 * Use cblas directly to scale Xm[k,] by w[k]
					 * This is MUCH faster that evaluating W %*% X, and saves time in calloc'ing a huge W matrix
					 *
					 * Access row i and col j of a gsl_matrix directly: m->data[i * m->tda + j]
					 * So access begining of row k by: m->data + k*m->tda
					 * 		since accessing matrix elements is done by pointer addition
					 */
					// W_X = W %*% Xm
					for(unsigned int k=0; k<n_indivs; k++){
						cblas_dscal(n_markers, gsl_vector_get(w, k), W_X->data + k*W_X->tda, 1);
					}

					// Xm_t_Xm = t(Xm) %*% W %*% Xm
					gsl_blas_dgemm(CblasTrans,CblasNoTrans, 1.0, X, W_X, 0.0, Xm_t_Xm);

					// Xm_t_Xm = solve(Xm_t_Xm)
				
					n = Xm_t_Xm->size1;

					//gsl_matrix_print(Xm_t_Xm);

					dpotrf_(&uplo, &n, Xm_t_Xm->data, &n, &info);

					// check for non-positive definite error
					if( info != 0 ){
						separable_logistic_system = true;
						//cout << "Separable logistic system" << endl;
						break;
					}

					dpotri_(&uplo, &n, Xm_t_Xm->data, &n, &info);
					gsl_matrix_triangular_to_full(Xm_t_Xm, uplo);
											

					gsl_blas_dgemm(CblasNoTrans,CblasTrans, 1.0, Xm_t_Xm, X, 0.0, Xm_t_Xm_ginv_X);

					// z = Y - mu
					gsl_vector_subtr(Y, mu, z);

					// beta_update = A %*% z
					gsl_blas_dgemv(CblasNoTrans, 1.0, Xm_t_Xm_ginv_X, z, 0.0, beta_update);

					back_track = 0;

					// If likelihood does not increase, do a backtracking step
					while(1){
						// save copy
						gsl_vector_memcpy(beta_old, beta);

						// beta = beta + beta_update
						gsl_vector_add(beta, beta_update, beta);

						// eta = X %*% beta
						gsl_blas_dgemv(CblasNoTrans, 1.0, X, beta, 0.0, eta);

						log_likelihood = eval_log_likelihood(Y, eta, regress_enum, NULL, 0);
						//log_likelihood = logistic_log_likelihood(Y, eta);
						//log_likelihood = logistic_log_likelihood_sse(Y, eta, mu);

						if(log_likelihood < log_likelihood_prev){
							//gsl_vector_scale(beta_update, .5);
							gsl_blas_dscal(.5, beta_update);
							gsl_vector_memcpy(beta, beta_old);
							back_track++;
							//cout << '_' << flush;
						}else{
							break;
						}
						if( back_track > 20 ) break;
					}

					//cout << log_likelihood << endl;

					//if( log_likelihood - log_likelihood_prev < 1e-8){
					if( log_likelihood - log_likelihood_prev < 1e-8 || iter > max_iter_IRLS){
						break;
					}

					log_likelihood_prev = log_likelihood;

					iter++;
				}

				/*cout << log_likelihood - log_likelihood_prev << endl;
				cout << "iter: " << iter << endl;
				gsl_vector_print(beta);*/

				// if allocated locally
				if( ! GLM_IRLS_workspace_correct_size(irls_workspace, n_indivs, n_markers) ){
					gsl_vector_free(w);
					gsl_matrix_free(W_X);
				}
			}

			//gsl_vector_print(beta);

			// if allocated locally
			if( ! GLM_IRLS_workspace_correct_size(irls_workspace, n_indivs, n_markers) ){
				gsl_vector_free(z);
				gsl_vector_free(beta_update);
				gsl_vector_free(beta_old);
				gsl_vector_free(mu);
			}

			break;
	}

	if( separable_logistic_system ) irls_workspace->design_is_singular = true;
	else irls_workspace->design_is_singular = false;

	gsl_vector *p_values = NULL;

	///////////////////
	// get residuals //
	///////////////////

	double sigma_sq_hat, df;

	if(regress_enum == LINEAR){
		gsl_vector *r = gsl_vector_alloc(n_indivs);

		gsl_blas_dgemv(CblasNoTrans, 1.0, X, beta, 0.0, eta);

		gsl_vector_subtr_restrict(Y, eta, r);

		// get residual variance
		*sig_e = pow(gsl_blas_dnrm2(r), 2) / (double) n_indivs;

		sigma_sq_hat = (*sig_e) *n_indivs;

		df = n_indivs - beta->size;

		gsl_vector_free(r);
	}

	if(regress_enum == LINEAR){

		// Not sure if this part works ??

		// get log-likelihood
		*log_L = eval_objective_function(Y, eta, regress_enum);
	}else{
		*log_L = log_likelihood;
	}

	////////////////////////////////
	// Evaluate Sigma = var(beta) //
	////////////////////////////////

	if( irls_workspace != NULL && irls_workspace->evaluate_Sigma ){

		double mu_i;

		switch(regress_enum){
			case LINEAR:

				gsl_matrix_scale( Xm_t_Xm, sigma_sq_hat/df);
				gsl_matrix_memcpy( irls_workspace->Sigma, Xm_t_Xm );
			break;

			case LINEAR_MIXED_MODEL:
			perror("Linear mixed model not handled by this function\n");
			break;

			case LOGISTIC:
				mu = irls_workspace->mu;
				gsl_matrix *W_X = irls_workspace->W_X;
				gsl_vector *w 	= irls_workspace->w;

				// mu = inverse_logit(eta)
				// mu is evaluated again, after the final step of the IRLS
				// Therefore, the matrix XWX can become singular even if is was not aboves
				inverse_logit(mu, eta);

				// W = diag( mu * (1-mu) )
				for(unsigned int i=0; i<n_indivs; i++){
					mu_i = gsl_vector_get(mu, i);
					gsl_vector_set(w, i, mu_i * (1-mu_i));
				}

				gsl_matrix_memcpy(W_X, X);

				// W_X = W %*% Xm
				for(unsigned int k=0; k<n_indivs; k++){
					cblas_dscal(n_markers, gsl_vector_get(w, k), W_X->data + k*W_X->tda, 1);
				}

				// Xm_t_Xm = t(Xm) %*% W %*% Xm
				gsl_blas_dgemm(CblasTrans,CblasNoTrans, 1.0, X, W_X, 0.0, Xm_t_Xm);

				// Xm_t_Xm = solve(Xm_t_Xm)
				n = Xm_t_Xm->size1;

				dpotrf_(&uplo, &n, Xm_t_Xm->data, &n, &info);

				// check for non-positive definite error
				if( info != 0 ){
					separable_logistic_system = true;
					//cout << "Separable logistic system" << endl;
				}else{
					dpotri_(&uplo, &n, Xm_t_Xm->data, &n, &info);
					gsl_matrix_triangular_to_full(Xm_t_Xm, uplo);
				}

				gsl_matrix_memcpy( irls_workspace->Sigma, Xm_t_Xm );

			break;
		}
	}


	//////////////////////////////////////////////////////////////////////
	// Calculate p-values using t-test (LINEAR) or Wald test (LOGISTIC) //
	//////////////////////////////////////////////////////////////////////

	if(report_p_values && !separable_logistic_system){
		gsl_matrix *W_X;
		gsl_vector *w;

		p_values = gsl_vector_alloc(n_markers);
		gsl_vector *sd = gsl_vector_alloc(n_markers);
		double mu_i;

		////////////////////////////////////////////////
		// get standard deviations for each parameter //
		////////////////////////////////////////////////

		switch(regress_enum){
			case LINEAR:

				for(unsigned int i=0; i<n_markers; i++){
					gsl_vector_set(sd, i, sqrt( gsl_matrix_get(Xm_t_Xm, i, i) * sigma_sq_hat/df) );
				}

				for(unsigned int i=0; i<n_markers; i++){
					//cout << i << " " << gsl_vector_get(beta, i) << " " << gsl_vector_get(sd, i) << " " << df << flush;
							//cout << fabs(gsl_vector_get(beta, i)/ gsl_vector_get(sd, i)) << endl;

					if(df != 0){
						gsl_vector_set(p_values, i, gsl_cdf_tdist_Q( fabs(gsl_vector_get(beta, i)/ gsl_vector_get(sd, i)), df)*2 );
					}else{
						// df can be 0 so use Guassian rather than t-distribution
						gsl_vector_set(p_values, i, gsl_cdf_ugaussian_Q( fabs(gsl_vector_get(beta, i)/ gsl_vector_get(sd, i)))*2 );
					}
				}
				break;

			case LINEAR_MIXED_MODEL:
				perror("Linear mixed model not handled by this function\n");
				break;

			case LOGISTIC:

				if( GLM_IRLS_workspace_correct_size(irls_workspace, n_indivs, n_markers) ){
					mu = irls_workspace->mu;
					W_X = irls_workspace->W_X;
					w 	= irls_workspace->w;
				}else{
					mu = gsl_vector_alloc(n_indivs);
					W_X = gsl_matrix_alloc(n_indivs, n_markers);
					w 	= gsl_vector_alloc(n_indivs);
				}

				// mu = inverse_logit(eta)
				// mu is evaluated again, after the final step of the IRLS
				// Therefore, the matrix XWX can become singular even if is was not aboves
				inverse_logit(mu, eta);

				// W = diag( mu * (1-mu) )
				for(unsigned int i=0; i<n_indivs; i++){
					mu_i = gsl_vector_get(mu, i);
					gsl_vector_set(w, i, mu_i * (1-mu_i));
				}

				gsl_matrix_memcpy(W_X, X);

				// W_X = W %*% Xm
				for(unsigned int k=0; k<n_indivs; k++){
					cblas_dscal(n_markers, gsl_vector_get(w, k), W_X->data + k*W_X->tda, 1);
				}

				// Xm_t_Xm = t(Xm) %*% W %*% Xm
				gsl_blas_dgemm(CblasTrans,CblasNoTrans, 1.0, X, W_X, 0.0, Xm_t_Xm);


				// Xm_t_Xm = solve(Xm_t_Xm)
				
				n = Xm_t_Xm->size1;

				dpotrf_(&uplo, &n, Xm_t_Xm->data, &n, &info);

				// check for non-positive definite error
				if( info != 0 ){
					separable_logistic_system = true;
					//cout << "Separable logistic system" << endl;
				}else{

					dpotri_(&uplo, &n, Xm_t_Xm->data, &n, &info);
					gsl_matrix_triangular_to_full(Xm_t_Xm, uplo);
				}

				// if not separable logistic system
				// evaluate p-values
				if( ! separable_logistic_system ){

					for(unsigned int i=0; i<n_markers; i++){
						gsl_vector_set(sd, i, sqrt( gsl_matrix_get(Xm_t_Xm, i, i)) );
					}

					for(unsigned int i=0; i<n_markers; i++){
						gsl_vector_set(p_values, i, gsl_cdf_ugaussian_Q( fabs(gsl_vector_get(beta, i)/ gsl_vector_get(sd, i))) * 2 );

						if( isnan( gsl_vector_get(p_values, i)) ){
							//cout << "\t" << i <<  " "<< gsl_vector_get(beta, i) << " " << gsl_vector_get(sd, i) << endl;
						}
					}
				}

				// if allocated locally
				if( ! GLM_IRLS_workspace_correct_size(irls_workspace, n_indivs, n_markers) ){
					gsl_vector_free(w);
					gsl_vector_free(mu);
					gsl_matrix_free(W_X);
				}
				break;
		}
		gsl_vector_free(sd);
	}
	if(report_p_values && separable_logistic_system){
		p_values = gsl_vector_alloc(n_markers);
		gsl_vector_set_all(p_values, 1);
	}

	// if allocated locally
	if( ! GLM_IRLS_workspace_correct_size(irls_workspace, n_indivs, n_markers) ){
		gsl_vector_free(eta);
		gsl_matrix_free(Xm_t_Xm);
		gsl_matrix_free(Xm_t_Xm_ginv_X);
	}

	return p_values;
}