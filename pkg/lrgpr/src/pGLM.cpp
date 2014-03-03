/*
 * pGLM.cpp
 *
 *  Created on: Feb 12, 2013
 *      Author: gh258
 */

#include "pGLM.h"

#include <iostream>
#include <math.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_statistics.h>

#include "gsl_additions.h"
#include "gsl_additions_restrict.h"
#include "misc_functions.h"


using namespace std;

GLM_workspace *GLM_workspace_alloc(const int n_indivs){

	GLM_workspace *workspace = (GLM_workspace *) malloc(sizeof(GLM_workspace));

	workspace->eta = gsl_vector_alloc(n_indivs);
	workspace->mu = gsl_vector_alloc(n_indivs);
	workspace->v = gsl_vector_alloc(n_indivs);

	return workspace;
}

GLM_workspace *GLM_workspace_calloc(const int n_indivs){

	GLM_workspace *workspace = (GLM_workspace *) malloc(sizeof(GLM_workspace));

	workspace->eta = gsl_vector_calloc(n_indivs);
	workspace->mu = gsl_vector_calloc(n_indivs);
	workspace->v = gsl_vector_calloc(n_indivs);

	return workspace;
}

void GLM_workspace_free(GLM_workspace *workspace){

	gsl_vector_free(workspace->eta);
	gsl_vector_free(workspace->v);
	gsl_vector_free(workspace->mu);

	free(workspace);
}


void check_hyperparameter_values( penaltyType penalty_enum, const double hyper_param1, const double hyper_param2 ){

	switch(penalty_enum){
		case MCP:
			if(hyper_param2 <= 1){
				#ifdef PRINT_TO_CONSOLE
				cout << "MCP: a = " << hyper_param2 << " is not valid\n";
				exit(1);
				#endif
			}
			break;

		case SCAD:
			if(hyper_param2 <= 2){
				#ifdef PRINT_TO_CONSOLE
				cout << "SCAD: a = " << hyper_param2 << " is not valid\n";
				exit(1);
				#endif
			}
			break;

		case LASSO:
		case ADAPTIVE_LASSO:
			if(hyper_param2 != 0){
				#ifdef PRINT_TO_CONSOLE
				cout << "LASSO: lambda_2 = " << hyper_param2 << " is not valid\n";
				exit(1);
				#endif
			}
			break;

		case ELASTIC_NET:
		case ADAPTIVE_ELASTIC_NET:
			if(hyper_param2 < 0){
				#ifdef PRINT_TO_CONSOLE
				cout << "ELASTIC_NET: lambda_2 = " << hyper_param2 << " is not valid\n";
				exit(1);
				#endif
			}
			break;

		case BRIDGE:
			if(hyper_param2 < 0){
				#ifdef PRINT_TO_CONSOLE
				cout << "BRIDGE: gamma = " << hyper_param2 << " is not valid\n";
				exit(1);
				#endif
			}
			break;

		case NEG:
			if(hyper_param2 < 0){
				#ifdef PRINT_TO_CONSOLE
				cout << "NEG: gamma = " << hyper_param2 << " is not valid\n";
				exit(1);
				#endif
			}
			break;
		case LOG:
		case VBAY:
			break;

		case NONE:
			#ifdef PRINT_TO_CONSOLE
			perror("No penalty");
			#endif
			break;
	}
}



double eval_objective_function(const gsl_vector *Y, const gsl_vector *eta, const regressionType regress_type, int * __restrict__ exclude_indeces, int exclude_count){

	int n_indivs = Y->size;

	// the number of indiviudals that contribute the the likelihood
	int n_active_indivs = n_indivs - exclude_count;

	double result = 0;// sigma_sq;

	switch(regress_type){
		case LINEAR:
		case LINEAR_MIXED_MODEL:
			// r = Y - eta
			// log_L = sum(r^2) / (2 * n_indivs)
			for(int i=0; i<n_indivs; i++){
				result += pow(gsl_vector_get(Y, i) - gsl_vector_get(eta, i) , 2);
			}

			for(int i=0; i<exclude_count; i++){
				result -= pow(gsl_vector_get(Y, exclude_indeces[i]) - gsl_vector_get(eta, exclude_indeces[i]) , 2);
			}

			result /= (2 * n_active_indivs);

			//sigma_sq = result / (double) n_active_indivs;

			//result = -n_active_indivs / (double) 2 * ( log(2*pi) - log(sigma_sq) - 1);

			break;

		case LOGISTIC:
			// log_L = -1/n_indivs * sum((eta*Y - log(1+exp(eta)))) + penalty
			for(int i=0; i<n_indivs; i++){
				result += gsl_vector_get(eta, i) * gsl_vector_get(Y, i)  - log(1 + exp( gsl_vector_get(eta, i) ));
			}

			// subtract log-likelihood from individuals that are excluded from the model
			for(int i=0; i<exclude_count; i++){
				result -= gsl_vector_get(eta,  exclude_indeces[i]) * gsl_vector_get(Y,  exclude_indeces[i])  - log(1 + exp( gsl_vector_get(eta,  exclude_indeces[i]) ));
			}

			result /= -n_active_indivs;
			break;
	}

	return result;
}



double eval_log_likelihood(const gsl_vector *Y, const gsl_vector *eta, const regressionType regress_type, int * __restrict__ exclude_indeces, int exclude_count){

	int n_indivs = Y->size;

	// the number of individuals that contribute the the likelihood
	int n_active_indivs = n_indivs - exclude_count;

	double result = 0, sigma_sq;

	switch(regress_type){
		case LINEAR:
		case LINEAR_MIXED_MODEL:
			// -n/2*( log(2*pi) - log(sigma_sq) - 1)
			for(int i=0; i<n_indivs; i++){
				result += pow(gsl_vector_get(Y, i) - gsl_vector_get(eta, i) , 2);
			}

			for(int i=0; i<exclude_count; i++){
				result -= pow(gsl_vector_get(Y, exclude_indeces[i]) - gsl_vector_get(eta, exclude_indeces[i]) , 2);
			}

			sigma_sq = result / (double) n_active_indivs;

			result = -n_active_indivs / (double) 2 * ( log(2*M_PI) + log(sigma_sq) + 1);
			break;

		case LOGISTIC:

			// log_L = sum((eta*Y - log(1+exp(eta))))
			for(int i=0; i<n_indivs; i++){
				result += gsl_vector_get(eta, i) * gsl_vector_get(Y, i)  - log(1 + exp( gsl_vector_get(eta, i) ));
			}

			// subtract log-likelihood from individuals that are excluded from the model
			for(int i=0; i<exclude_count; i++){
				result -= gsl_vector_get(eta,  exclude_indeces[i]) * gsl_vector_get(Y,  exclude_indeces[i])  - log(1 + exp( gsl_vector_get(eta,  exclude_indeces[i]) ));
			}
			break;
	}

	return result;
}


double soft_threashold(const double d, const double lambda){
	if(lambda >= fabs(d)){
		return(0);
	}else if(d > 0){
		return (d - lambda);
	}else{
		return (d + lambda);
	}
}

double penalty(const gsl_vector * __restrict__ beta, const double hyper_param1, const double hyper_param2, penaltyType penalty_enum,
		const gsl_vector * __restrict__ coeff_weights){

	double result = 0;

	if(beta == NULL){
		result = 0;
	}else{
		switch(penalty_enum){
			case MCP:


				for(unsigned int j=0; j<beta->size; j++){
					if(gsl_vector_get(beta, j) != 0) result += mcp(fabs( gsl_vector_get(beta, j)), hyper_param1, hyper_param2);
				}
				break;

			case SCAD:
				for(unsigned int j=0; j<beta->size; j++){
					if(gsl_vector_get(beta, j) != 0) result += scad(fabs( gsl_vector_get(beta, j)), hyper_param1, hyper_param2);
				}
				break;

			case LASSO:
				for(unsigned int j=0; j<beta->size; j++){
					if(gsl_vector_get(beta, j) != 0) result += fabs( gsl_vector_get(beta, j));
				}
				result *= hyper_param1;
				break;

			case LOG:

				for(unsigned int j=0; j<beta->size; j++){
					if(gsl_vector_get(beta, j) != 0) log_penalty( gsl_vector_get(beta, j), hyper_param1, hyper_param2);
				}
				break;

			case ELASTIC_NET:
				for(unsigned int j=0; j<beta->size; j++){
					result += hyper_param1 * fabs( gsl_vector_get(beta, j)) + hyper_param2 * pow( gsl_vector_get(beta, j), 2) / (double) 2;
				}
				break;

			case ADAPTIVE_LASSO:
				for(unsigned int j=0; j<beta->size; j++){
					result +=  fabs( gsl_vector_get(beta, j)) * gsl_vector_get(coeff_weights, j);

					/*if(gsl_vector_get(coeff_weights, j) != 1000){
						cout << j << ": " << gsl_vector_get(coeff_weights, j) << endl;
					}*/
				}
				result *= hyper_param1;
				break;

			case ADAPTIVE_ELASTIC_NET:
				for(unsigned int j=0; j<beta->size; j++){
					result += hyper_param1 * fabs( gsl_vector_get(beta, j)) * gsl_vector_get(coeff_weights, j) + hyper_param2 * pow( gsl_vector_get(beta, j), 2) / (double) 2;
				}
				break;

			case BRIDGE:

				for(unsigned int j=0; j<beta->size; j++){
					result += pow( fabs(gsl_vector_get(beta, j)), hyper_param2);
				}
				result *= hyper_param1;
				break;

			case NEG:/*

				neg_at_zero = neg(0, hyper_param1, hyper_param2);

				for(unsigned int j=0; j<beta->size; j++){
					if(gsl_vector_get(beta, j) != 0){
						result += neg(gsl_vector_get(beta, j), hyper_param1, hyper_param2);
					}else{
						result += neg_at_zero;
					}
				}

				// Since the NEG value at 0 is nonzero, subtract the penalty for all betas being zero from the current penalty
				// Therefore the penalty for beta = 0 is now zero
				result -= beta->size * neg(0, hyper_param1, hyper_param2);
				break;*/

			case NONE:
			default:
				perror("No penalty");
				break;
		}
	}
	return result;
}

double GLM_penalized_regression(const gsl_vector *Y, const gsl_matrix *t_X, const gsl_matrix *X_cov, gsl_vector *beta, gsl_vector *beta_cov, regressionType regress_enum, penaltyType penalty_enum, const double hyper_param1, const double hyper_param2, const gsl_vector *feature_scales, const gsl_vector *covariate_scales, const gsl_vector *feature_weights, GLM_workspace *workspace, const gsl_permutation *permutation, const int maxit, const double thresh, GLM_penRegInfo *info, const bool initialized){

	// check validity of hyperparameter values for the penality selected
	check_hyperparameter_values( penalty_enum, hyper_param1, hyper_param2 );

	// remove excluded indeces from the count of individuals
	int n_indivs = Y->size;

	int n_markers = t_X->size1;
	int n_covariates;

	if( X_cov == NULL) 	n_covariates = 0;
	else 				n_covariates = X_cov->size2;

	bool alloc_local_workspace = false;

	if(workspace == NULL){
		workspace = GLM_workspace_alloc(n_indivs);
		alloc_local_workspace = true;
	}

	gsl_vector *eta =  workspace->eta;
	gsl_vector *mu =  workspace->mu;
	gsl_vector *v =  workspace->v;

	gsl_vector_view column_view, row_view;

	/*info->n_beta_active = 0;
	for(int j=0; j<n_markers; j++){
		if( fabs(gsl_vector_get(beta, j)) > ACTIVE_MODEL_EFFECTIVELY_ZERO ) info->n_beta_active++;
	}*/

	if( ! initialized ){

		gsl_vector_set_zero(eta);

		info->n_beta_active = 0;

		// eta = X %*% beta
		for(int j=0; j<n_markers; j++){

			// set very small values to exactly zero
			if(fabs(gsl_vector_get(beta, j)) < ACTIVE_MODEL_EFFECTIVELY_ZERO){
				gsl_vector_set(beta, j, 0);
			}else{
				//gsl_blas_daxpy( gsl_vector_get(beta, j), X->at(j), eta);
				row_view = gsl_matrix_row( (gsl_matrix*)(t_X), j);
				gsl_blas_daxpy( gsl_vector_get(beta, j), &row_view.vector, eta);

				info->n_beta_active++;
			}
		}

		for(int j=0; j<n_covariates; j++){
			//gsl_blas_daxpy( gsl_vector_get(beta_cov, j), X_cov->at(j), eta);
			column_view = gsl_matrix_column( (gsl_matrix*)(X_cov), j);
			gsl_blas_daxpy( gsl_vector_get(beta_cov, j), &column_view.vector, eta);
		}

		// apply the inverse link in the logistic case
		if(regress_enum == LOGISTIC){
			// mu = inv.logit(eta)
			inverse_logit(mu, eta);
		}
	}

	double log_likelihood, z;
	double log_likelihood_prev = eval_objective_function(Y, eta, regress_enum) + penalty(beta, hyper_param1, hyper_param2, penalty_enum, feature_weights);

	info->logLikArray.push_back( log_likelihood_prev );

	double beta_new, dot_prod, quad_form = 0, weight;

	switch(regress_enum){
		case LINEAR:
		case LINEAR_MIXED_MODEL:
			weight = 1;
			break;

		case LOGISTIC:
			weight = .25;
			break;
	}

	/* The standard dot product of a feature is n_indivs
	 Increasing the dot product by a factor causes the beta values to decrease by the sqrt of the factor
	 Therefore, is the ratio of the actual scale of the features to the standard scale
	 This value is used to effectively standardize beta for the penalty term
		but inflate/deflate beta for generating eta*/
	// scale = sqrt(sum(X[,1]^2)) / sqrt(n_indivs)
	// This is precalculated for each feature in feature_scales

	double scale;

	int j;

	switch(regress_enum){
		case LINEAR:
		case LINEAR_MIXED_MODEL:
			gsl_vector_subtr_restrict(Y, eta, v);
			break;

		case LOGISTIC:
			// v = Y_bin- mu
			gsl_vector_subtr_restrict(Y, mu, v);
			break;
	}

	// Initialize iteration count
	info->n_iter = 0;

	while(1){

		// Increment iteration count
		info->n_iter++;

		/////////////////////////////////////////////
		// cycle through each UNpenalized variable //
		/////////////////////////////////////////////

		for(int k=0; k<n_covariates; k++){

			////////////////////
			// with rescaling //
			////////////////////

			//ddot_wrapper(v, X_unpenalized->at(j), &dot_prod, exclude_indeces, exclude_count);
			column_view = gsl_matrix_column( (gsl_matrix*)(X_cov), k);
			gsl_blas_ddot( v, &column_view.vector, &dot_prod);

			// get the scale for feature j compared to the full dataset
			scale = gsl_vector_get(covariate_scales, k);

			quad_form = scale * weight;

			// set to the numerator of beta_new
			z = gsl_vector_get(beta_cov, k)*quad_form + dot_prod;

			beta_new = z / quad_form;

			if(fabs(beta_new - gsl_vector_get(beta_cov, k)) > DELTA_BETA_TOLERANCE){

				// eta = eta + X[,j] * ( beta_new - beta[j])
				//gsl_blas_daxpy( beta_new - gsl_vector_get(beta_cov, j), X_cov->at(j), eta);
				column_view = gsl_matrix_column( (gsl_matrix*)(X_cov), k);
				gsl_blas_daxpy( beta_new - gsl_vector_get(beta_cov, k),  &column_view.vector, eta);

				// beta[j] = beta_new
				gsl_vector_set(beta_cov, k, beta_new);

				// update residuals
				switch(regress_enum){
					case LINEAR:
					case LINEAR_MIXED_MODEL:
						gsl_vector_subtr_restrict(Y, eta, v);
						break;

					case LOGISTIC:
						// apply the inverse link in the logistic case
						// mu = inv.logit(eta)
						inverse_logit(mu, eta);

						// v = Y_bin- mu
						gsl_vector_subtr_restrict(Y, mu, v);
						break;
				}
			}
		}

		///////////////////////////////////////////
		// cycle through each PENALIZED variable //
		///////////////////////////////////////////

		// set to zero
		info->max_unselected_beta = 0;
		info->mid_unselected_beta = 0;

		for(int j_index=0; j_index<n_markers; j_index++){

			j = gsl_permutation_get(permutation, j_index);

			// In this case W is diag of a scalar and can be multiplied after
			// dot_prod = t(v) %*% X[,j]
			//gsl_blas_ddot(v, X->at(j), &dot_prod);
			row_view = gsl_matrix_row( (gsl_matrix*)(t_X), j);
			gsl_blas_ddot( v, &row_view.vector, &dot_prod);

			// get the scale for feature j compared to the full dataset
			scale = gsl_vector_get(feature_scales, j);

			quad_form = scale * weight;

			// set to the numerator of beta_new
			z = (gsl_vector_get(beta, j)*quad_form + dot_prod) / n_indivs;

			switch(penalty_enum){
				case LASSO:
					beta_new = soft_threashold(z, hyper_param1) / (quad_form/n_indivs);
					break;

				case MCP:
					if(fabs(z) <= hyper_param1*hyper_param2*quad_form/n_indivs){

						if(regress_enum == LINEAR){
							beta_new = soft_threashold(z, hyper_param1) / (quad_form/n_indivs - 1/hyper_param2);
						}else{
							// adaptive rescaling
							beta_new = soft_threashold(z, hyper_param1) / (quad_form/n_indivs *(1 - 1/hyper_param2));
						}
					}else{
						beta_new = z / (quad_form/n_indivs);
					}
					break;

				/*case SCAD:
					if(fabs(z) <= hyper_param1*(quad_form/n_indivs+1)){
						beta_new = soft_threashold(z, hyper_param1) / (quad_form/n_indivs);
					}else if( fabs(z) > hyper_param1*hyper_param2){
						beta_new = z / (quad_form/n_indivs);
					}else{
						beta_new = soft_threashold(z, hyper_param1*hyper_param2/(hyper_param2-1)) / (quad_form/n_indivs - 1/(hyper_param2-1));
					}
					break;

				case ELASTIC_NET:
					beta_new = soft_threashold(z, hyper_param1) / ((quad_form/n_indivs) + hyper_param2);
					break;*/

				case BRIDGE:
				case VBAY:
				case NONE:
					perror("No penalty / unsupported penalty");
					break;
			}// End switch

			// set very small values to exactly zero
			if( fabs(beta_new) < ACTIVE_MODEL_EFFECTIVELY_ZERO ){
				beta_new = 0;
				if( gsl_vector_get(beta, j) != 0 ) info->n_beta_active--;
			}

			// If beta should be updated
			if(fabs(beta_new - gsl_vector_get(beta, j)) > DELTA_BETA_TOLERANCE){

				// if previous value was zero
				if(gsl_vector_get(beta, j) == 0) info->n_beta_active++;

				// eta = eta + X[,j] * (beta_new - beta[j]) / scale
				//gsl_blas_daxpy( (beta_new - gsl_vector_get(beta, j)), X->at(j), eta);
				row_view = gsl_matrix_row( (gsl_matrix*)(t_X), j);
				gsl_blas_daxpy( beta_new - gsl_vector_get(beta, j),  &row_view.vector, eta);

				// beta[j] = beta_new
				gsl_vector_set(beta, j, beta_new);

				// update residuals
				switch(regress_enum){
					case LINEAR:
						gsl_vector_subtr_restrict(Y, eta, v);
						break;

					case LOGISTIC:
						// apply the inverse link
						inverse_logit(mu, eta);

						// v = Y_bin- mu
						gsl_vector_subtr_restrict(Y, mu, v);
						break;
				}
			}

			// If beta is zero and not updated, compare the value to the max value seen in this pass through the features
			if( gsl_vector_get(beta, j) == 0){
				info->mid_unselected_beta = max( info->mid_unselected_beta, fabs(z));
				if( info->max_unselected_beta < info->mid_unselected_beta ){
					swap( info->max_unselected_beta, info->mid_unselected_beta );
				}
			}
		} // End update of beta[j]


		log_likelihood = eval_objective_function(Y, eta, regress_enum) + penalty(beta, hyper_param1, hyper_param2, penalty_enum, feature_weights);

		info->logLikArray.push_back( log_likelihood );

		//cout << log_likelihood_prev - log_likelihood << endl;
		//if( log_likelihood_prev - log_likelihood < DELTA_LOG_LIKELIHOOD_TOLERANCE || info->n_iter >= maxit ){
		if( log_likelihood_prev - log_likelihood < thresh || info->n_iter >= maxit ){
			break;
		}

		if(log_likelihood_prev - log_likelihood < 0){
			#ifdef PRINT_TO_CONSOLE
			printf("log_L increased from %10.6f to %10.6f @ hyper_param1 =%10.6f, hyper_param2 =%10.6f\n", log_likelihood_prev, log_likelihood, hyper_param1, hyper_param2);
			#endif
			break;
		}

		// If lambda = Inf, then just testing for starting lambda value
		// Normal termination doesn't work because Inf messes up evaluation of log_likelihood
		if( isinf(hyper_param1) ) break;

		log_likelihood_prev = log_likelihood;
	}

	info->nIterArray.push_back( info->n_iter );

	log_likelihood = eval_log_likelihood(Y, eta, regress_enum) - penalty(beta, hyper_param1, hyper_param2, penalty_enum, feature_weights) / n_indivs;

	// Clean up if new workspace was alloc'd
	if(alloc_local_workspace) GLM_workspace_free(workspace);

	return log_likelihood;
}



inline double mcp(const double beta, const double lambda, const double a){

	if( beta <= a * lambda){
		return lambda * beta - beta*beta / (2 * a);
	}else{
		return a * lambda*lambda/ 2;
	}
}


inline double scad(const double beta, const double lambda, const double a){

	if( beta <= lambda){
		return beta * lambda;
	}else if(beta > a * lambda){
		//return pow(lambda, 2) * (pow(a, 2) -1 ) / (2 * (a-1));
		return (a + 1) * pow(lambda, 2)/ 2;
	}else{
		//return (a * lambda * beta - 0.5 * (pow(beta, 2) + pow(lambda, 2))) / (a-1);
		return (-pow(beta, 2) - pow(lambda, 2) + 2*a*lambda*beta)/ (2 * (a-1));
	}
}


inline double bridge(const double beta, const double lambda, const double gamma){

	return lambda * pow(fabs(beta), gamma);
}

inline double log_penalty(const double beta, const double lambda, const double epsilon){

	return lambda * log(1+fabs(beta)/ epsilon) / log(1+1/epsilon);
}

inline double mcp_derivative(const double beta, const double lambda, const double a){

	if( beta <= a * lambda){
		return lambda - beta / a;
	}else{
		return 0;
	}
}

inline double scad_derivative(const double beta, const double lambda, const double a){

	if( beta <= lambda){
		return lambda;
	}else if(beta > a * lambda){
		return 0;
	}else{
		return (a*lambda - beta) / (a-1);
	}
}

double bridge_derivative(const double beta, const double lambda, const double gamma){
	if(beta == 0){
		return lambda;
	}else{
		return lambda * gamma * pow(fabs(beta), gamma-1);
	}
}



void LMM_penalized_regression_on_residuals(gsl_vector *v, const gsl_matrix *t_X, gsl_vector *beta, penaltyType penalty_enum,
		const double hyper_param1, const double hyper_param2, const gsl_vector *feature_scales, const gsl_permutation *permutation,
		const int maxit, GLM_penRegInfo *info, const bool initialized, const gsl_vector *inv_s_delta){

	// check validity of hyperparameter values for the penality selected
	check_hyperparameter_values( penalty_enum, hyper_param1, hyper_param2 );


	int n_indivs = v->size;
	int n_markers = t_X->size1;

	gsl_vector_view row_view;

	double beta_new, dot_prod, quad_form = 0;
	double scale, z;
	int j;

	gsl_vector *v_inv_s_delta = gsl_vector_alloc( v->size );
	gsl_vector_memcpy( v_inv_s_delta, v );
	gsl_vector_mul(v_inv_s_delta, inv_s_delta);

	/*cout << "v" << endl;
	cout << gsl_vector_get(v, 0) << endl;
	cout << gsl_vector_get(v, 1) << endl;
	cout << gsl_vector_get(v, 2) << endl << endl;*/

	// Initialize iteration count
	info->n_iter = 0;

	info->n_beta_active = 0;

	for(unsigned int j=0; j<beta->size; j++){
		if( gsl_vector_get(beta, j) !=0 ) info->n_beta_active++;
	}

	while(1){

		// Increment iteration count
		info->n_iter++;

		///////////////////////////////////////////
		// cycle through each PENALIZED variable //
		///////////////////////////////////////////

		// set to zero
		info->max_unselected_beta = 0;
		info->mid_unselected_beta = 0;

		for(int j_index=0; j_index<n_markers; j_index++){

			j = gsl_permutation_get(permutation, j_index);

			// In this case W is diag of a scalar and can be multiplied after
			// dot_prod = t(v) %*% X[,j]
			//gsl_blas_ddot(v, X->at(j), &dot_prod);
			row_view = gsl_matrix_row( (gsl_matrix*)(t_X), j);
			gsl_blas_ddot( v_inv_s_delta, &row_view.vector, &dot_prod);

			// get the scale for feature j compared to the full dataset
			scale = gsl_vector_get(feature_scales, j);

			quad_form = scale;

			// set to the numerator of beta_new
			//z = (gsl_vector_get(beta, j)*quad_form + dot_prod) / n_indivs;
			z = gsl_vector_get(beta, j)*quad_form + dot_prod;

			/*cout << "z" << j << " " << z << endl;
			cout << "beta" << j <<  " " << z/quad_form << endl;
			cout << gsl_vector_get(beta, j) << endl;
			cout << quad_form << endl;
			cout << dot_prod << endl << endl;*/

			switch(penalty_enum){
				case LASSO:
					beta_new = soft_threashold(z, hyper_param1) / quad_form;
					break;

				/*case MCP:
					if(fabs(z) <= hyper_param1*hyper_param2*quad_form/n_indivs){
						beta_new = soft_threashold(z, hyper_param1) / (quad_form/n_indivs - 1/hyper_param2);
					}else{
						beta_new = z / (quad_form/n_indivs);
					}
					break;*/

				/*case SCAD:
					if(fabs(z) <= hyper_param1*(quad_form/n_indivs+1)){
						beta_new = soft_threashold(z, hyper_param1) / (quad_form/n_indivs);
					}else if( fabs(z) > hyper_param1*hyper_param2){
						beta_new = z / (quad_form/n_indivs);
					}else{
						beta_new = soft_threashold(z, hyper_param1*hyper_param2/(hyper_param2-1)) / (quad_form/n_indivs - 1/(hyper_param2-1));
					}
					break;

				case ELASTIC_NET:
					beta_new = soft_threashold(z, hyper_param1) / ((quad_form/n_indivs) + hyper_param2);
					break;*/

				case BRIDGE:
				case VBAY:
				case NONE:
					perror("No penalty / unsupported penalty");
					break;
			}// End switch

			// set very small values to exactly zero
			if( fabs(beta_new) < ACTIVE_MODEL_EFFECTIVELY_ZERO ){
				beta_new = 0;
				if( gsl_vector_get(beta, j) != 0 ) info->n_beta_active--;
			}

			// If beta should be updated
			if(fabs(beta_new - gsl_vector_get(beta, j)) > DELTA_BETA_TOLERANCE){

				// if previous value was zero
				if(gsl_vector_get(beta, j) == 0) info->n_beta_active++;

				// v = v - X[,j] * (beta_new - beta[j])
				//gsl_blas_daxpy( (beta_new - gsl_vector_get(beta, j)), X->at(j), v);
				row_view = gsl_matrix_row( (gsl_matrix*)(t_X), j);
				gsl_blas_daxpy( -(beta_new - gsl_vector_get(beta, j)), &row_view.vector, v);

				// beta[j] = beta_new
				gsl_vector_set(beta, j, beta_new);

				/*cout << "v" << endl;
				cout << gsl_vector_get(v, 0) << endl;
				cout << gsl_vector_get(v, 1) << endl;
				cout << gsl_vector_get(v, 2) << endl << endl;*/

				gsl_vector_memcpy( v_inv_s_delta, v );
				gsl_vector_mul(v_inv_s_delta, inv_s_delta);
			}

			// If beta is zero and not updated, compare the value to the max value seen in this pass through the features
			if( gsl_vector_get(beta, j) == 0){
				info->mid_unselected_beta = max( info->mid_unselected_beta, fabs(z));
				if( info->max_unselected_beta < info->mid_unselected_beta ){
					swap( info->max_unselected_beta, info->mid_unselected_beta );
				}
			}
		} // End update of beta[j]

		if( info->n_iter >= maxit ) break;

		// If lambda = Inf, then just testing for starting lambda value
		// Normal termination doesn't work because Inf messes up evaluation of log_likelihood
		if( isinf(hyper_param1) ) break;
	}

	info->nIterArray.push_back( info->n_iter );
}


