/*
 * gsl_lapack.cpp
 *
 *  Created on: Jan 13, 2011
 *      Author: gh258
 */

#include "gsl_lapack.h"

#include <iostream>
#include <string.h>
#include <vector>

#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cdf.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>

#include <limits>

#include "gsl_additions.h"

using namespace std;

int gsl_lapack_svd(const gsl_matrix *X, gsl_matrix *& t_U, gsl_vector *& singular_values, gsl_matrix *& V){

	// The matrix X must be WIDE in C, so that it is TALL in Fortran
	if( X->size1 > X->size2){
		#ifdef PRINT_TO_CONSOLE
		cout << "Trying to do SVD of a " << X->size1 << "x" << X->size2 << " matrix which is WIDE in Fortran" << endl;
		cout << "Need to pass the transpose of this matrix" << endl;
		exit(1);
		#endif
	}

	// Assume X->size2 >= X->size1

	char jobu = 'S';
	char jobvt = 'S';
	int m = X->size2; // n_rows
	int n = X->size1; // n_cols
	double *a = X->data;
	int lda = m;
	double *s = (double *) malloc( sizeof(double) * n );
	double *u = (double *) malloc( sizeof(double) * m*n );
	int ldu = m;
	double *vt = (double *) malloc( sizeof(double) * n*n );
	int ldvt = n;
	//int lwork = max(3*min(m,n)+max(m,n),5*min(m,n)-4);
	int info;

	//cout << "n: " << n << endl;
	//cout << "m: " << m << endl;

	double work_query[1];
	int lwork = -1;

	dgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work_query, &lwork, &info);

	lwork = work_query[0];
	double *work = (double *) malloc( sizeof(double) * lwork);

	dgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info);

	free( work );

	//cout << "info: " << info << endl;

	////////////
	// Save V //
	////////////

	V = gsl_matrix_attach_array(vt, n, n);

	//////////////////////////
	// Save singular values //
	//////////////////////////

	singular_values = gsl_vector_attach_array( s, n);

	//////////////
	// Save t_U //
	//////////////

	t_U = gsl_matrix_attach_array(u, n, m);

	return info;
}

int gsl_lapack_eigen(const gsl_matrix *C, gsl_matrix *& t_U, gsl_vector *& eigen_values){

	if( C->size1 != C->size2){
		#ifdef PRINT_TO_CONSOLE
		cout << "Cannot do eigen decomposition of non-symmetric matrix: " << C->size1 << "x" << C->size2 << endl;
		exit(1);
		#endif
	}

	t_U = gsl_matrix_alloc( C->size1, C->size2 );
	gsl_matrix_memcpy( t_U, C );

	char jobz = 'V';
	char uplo = 'U';
	int n = C->size1;
	int lda = C->size1;
	double *w = (double *) malloc(sizeof(double) * n);
	double work_query[1];
	int lwork = -1;
	int info;

	// query workspace size
	dsyev_(&jobz, &uplo, &n, t_U->data, &lda, w, work_query, &lwork, &info);

	//cout << "work[0]: " << work_query[0] << endl;

	// allocate work to be correct size
	double *work = (double *) malloc(sizeof(double)*work_query[0]);
	lwork = work_query[0];

	// compute eigen decomposition
	dsyev_(&jobz, &uplo, &n, t_U->data, &lda, w, work, &lwork, &info);

	//cout << "info: " << info << endl;

	// attach singular values to external variable
	eigen_values = gsl_vector_attach_array( w, n);

	free( work );

	if( info != 0 ){
		return GSL_ERANGE;
	}
	return( GSL_SUCCESS );
}


int gsl_lapack_eigen_lowrank(const gsl_matrix *C, gsl_matrix *& t_U, gsl_vector *& eigen_values, const int rank){

	if( C->size1 != C->size2){
		#ifdef PRINT_TO_CONSOLE
		cout << "Cannot do eigen decomposition of non-symmetric matrix: " << C->size1 << "x" << C->size2 << endl;
		exit(1);
		#endif
	}

	t_U = gsl_matrix_alloc( C->size1, C->size2 );
	gsl_matrix_memcpy( t_U, C );

	char jobz = 'V';
	char uplo = 'U';
	int n = C->size1;
	int lda = C->size1;
	double *w = (double *) malloc(sizeof(double) * n);
	double work_query[1];
	int lwork = -1;
	int info;

	// query workspace size
	dsyev_(&jobz, &uplo, &n, t_U->data, &lda, w, work_query, &lwork, &info);

	//cout << "work[0]: " << work_query[0] << endl;

	// allocate work to be correct size
	double *work = (double *) malloc(sizeof(double)*work_query[0]);
	lwork = work_query[0];

	// compute eigen decomposition
	dsyev_(&jobz, &uplo, &n, t_U->data, &lda, w, work, &lwork, &info);

	//cout << "info: " << info << endl;

	// attach singular values to external variable
	eigen_values = gsl_vector_attach_array( w, n);

	free( work );

	if( info != 0 ){
		return GSL_ERANGE;
	}
	return( GSL_SUCCESS );
}


gsl_vector* gsl_lapack_fit_least_squares(const gsl_vector *Y, const gsl_matrix *X, gsl_vector *& beta, int * rank, double *sigma_sq_e, double *log_L, bool report_p_values){

	// transpose X
	gsl_matrix *t_X = gsl_matrix_alloc( X->size2, X->size1);
	gsl_matrix_transpose_memcpy(t_X, X);

	// copy Y
	double *b = (double *) malloc(sizeof(double)*Y->size);
	memcpy(b, Y->data, Y->size * sizeof(double));

	// copy Y again
	gsl_vector *Y_residuals = gsl_vector_alloc( Y->size );
	gsl_vector_memcpy(Y_residuals, Y);

	int m = t_X->size2;
	int n = t_X->size1;
	int nrhs = 1;
	double *a = t_X->data;
	int lda = m;
	int ldb = Y->size;
	//int *jpvt = (int *) malloc(sizeof(int) * n);
	double *jpvt = (double *) malloc(sizeof(double) * min(n,m));
	//double rcond = 0.01; // good for dgelsy
	double rcond = 1e-10; // good for dgelss
	double work_query[1];
	int lwork = -1;
	int info = 0;

	// query workspace size
	dgelss_( &m, &n, &nrhs, a, &lda, b, &ldb, jpvt, &rcond, rank, work_query, &lwork, &info);

	//cout << "work[0]: " << work_query[0] << endl;

	// allocate work to be correct size
	double *work = (double *) malloc(sizeof(double)*work_query[0]);
	lwork = work_query[0];

	// Evaluate least squares system
	dgelss_( &m, &n, &nrhs, a, &lda, b, &ldb, jpvt, &rcond, rank, work, &lwork, &info);

	//cout << "info: " << info << endl;

	//cout << "b[0]: " << b[0] << endl;

	//cout << "rcond: " << rcond << endl;
	//cout << "rank: " << *rank << endl;

	//int incx, INCY;

	//////////////////////////////////////////////////////////////////
	// Get regression coefficients: the first X->size2 entries in b //
	//////////////////////////////////////////////////////////////////

	// re-allocate beta if it is not the correct size
	if( beta->size != X->size2){
		gsl_vector_free( beta );
		beta = gsl_vector_alloc( X->size2 );
	}

	memcpy( beta->data, b, sizeof(double) * beta->size );

	//gsl_vector_print( beta_coefficients );

	/* Y_residuals is initialized to Y
	 * Use dgemv to evaluate: Y_residuals <- -X\beta + Y_residuals
	 */
	gsl_blas_dgemv (CblasNoTrans, -1.0, X, beta, 1.0, Y_residuals);

	//gsl_vector_print(Y_residuals);

	////////////////////////////////////////////////////////////////////////
	// Calculate \sigma_e^2 and the log-likelihood based on the residuals //
	////////////////////////////////////////////////////////////////////////

	int n_indivs = Y->size;

	// get residual variance
	*sigma_sq_e = pow(gsl_blas_dnrm2(Y_residuals), 2) / (double) n_indivs;

	// calculate log likelihood from Gaussian model
	*log_L = -n_indivs / (double) 2 * ( log(2*M_PI) + log(*sigma_sq_e) + 1);

	gsl_matrix_free( t_X );
	free( b );
	gsl_vector_free(Y_residuals);
	free( jpvt );
	free( work );

	// If p_values requested, and X is full rank
	if( report_p_values ){

		// Allocate p-values and set values to NaN
		gsl_vector *p_values = gsl_vector_alloc( X->size2 );
		gsl_vector_set_all(p_values, numeric_limits<double>::quiet_NaN());

		if( (unsigned int) *rank == X->size2 ){

			gsl_matrix *C = gsl_matrix_alloc( X->size2, X->size2 );

			// C = X^TX
			gsl_blas_dgemm(CblasTrans,CblasNoTrans, 1.0, X, X, 0.0, C);

			//////////////
			// Invert C //
			//////////////

			int info = 0;
			char uplo = 'U';
			int n = C->size1;

			// Compute Cholesky decomposition of C
			dpotrf_(&uplo, &n, C->data, &n, &info);

			// C is full rank
			if( info == 0 ){

				// Invert C based on its Cholesky decomposition
				dpotri_(&uplo, &n, C->data, &n, &info);

				// If there is an error with the Cholesky decomposition
				if( info == 0 ){

					double sigma_sq_hat = (*sigma_sq_e) *n_indivs;
					double sd, p;

					int df = n_indivs - beta->size;

					////////////////////////
					// Calculate p-values //
					////////////////////////

					for(unsigned int i=0; i<C->size1; i++){

						sd = sqrt( gsl_matrix_get(C, i, i) * sigma_sq_hat / df );

						if(df != 0){
							// T-test
							p = gsl_cdf_tdist_Q( fabs(gsl_vector_get(beta, i)/ sd ), df) * 2 ;
						}else{
							// df can be 0 so use Gaussian rather than t-distribution
							p = gsl_cdf_ugaussian_Q( fabs(gsl_vector_get(beta, i)/ sd )) * 2 ;
						}

						gsl_vector_set(p_values, i, p);
					}
				} // End If inversion had error
			} // End If Cholesky had error

			gsl_matrix_free( C );

		} // End If X is full rank

		return p_values;
	}

	return NULL;
}


int gsl_lapack_eigenValues(gsl_matrix *C, gsl_vector *eigen_values){

	char JOBVL = 'V';
	char JOBVR = 'V';
	int N = C->size1;
	double *A = C->data;
	int LDA = N;
	int LDVL = N; 
	int LDVR = N;
	int info, lwork;

    double wkopt;
    double* work;
    /* Local arrays */
    double WI[N], vl[LDVL*N], vr[LDVR*N];

    double *WR = eigen_values->data;

    /* Query and allocate the optimal workspace */
    lwork = -1;
    dgeev_( &JOBVL, &JOBVR, &N, A, &LDA, WR, WI, vl, &LDVL, vr, &LDVR, &wkopt, &lwork, &info );

    lwork = (int) wkopt;

    work = (double*) malloc( lwork*sizeof(double) );

    /* Solve eigenproblem */
    dgeev_( &JOBVL, &JOBVR, &N, A, &LDA, WR, WI, vl, &LDVL, vr, &LDVR, work, &lwork, &info );
    
    free( work);

    return info;
}


void gsl_lapack_weighted_least_squares(const gsl_vector *Y, const gsl_matrix *X, const gsl_vector *weights, gsl_vector *& beta ){

	// copy Y
	double *b = (double *) malloc(sizeof(double)*Y->size);
	memcpy(b, Y->data, Y->size * sizeof(double));

	gsl_vector *weights_sqrt = gsl_vector_alloc( weights->size );

	for(unsigned int i=0; i<weights_sqrt->size; i++){
		gsl_vector_set(weights_sqrt, i, sqrt(gsl_vector_get(weights, i)) );
	}

	// Y = Y*weight
	// b = b * weight
	for(unsigned int i=0; i<Y->size; i++){
		b[i] = b[i]*gsl_vector_get(weights_sqrt, i);
	}

	// apply weights to X
	gsl_matrix *X_weighted = NULL;
	gsl_matrix_diagonal_multiply( X, weights_sqrt, X_weighted);

	// transpose X
	gsl_matrix *t_X = gsl_matrix_alloc( X_weighted->size2, X_weighted->size1);
	gsl_matrix_transpose_memcpy(t_X, X_weighted);

	int m = t_X->size2;
	int n = t_X->size1;
	int nrhs = 1;
	double *a = t_X->data;
	int lda = m;
	int ldb = Y->size;
	//int *jpvt = (int *) malloc(sizeof(int) * n);
	double *jpvt = (double *) malloc(sizeof(double) * min(n,m));
	//double rcond = 0.01; // good for dgelsy
	double rcond = 1e-10; // good for dgelss
	int rank;
	double work_query[1];
	int lwork = -1;
	int info = 0;

	// dgelss_
	// dgels_  14s
	// dgelsy_ 15s

	// query workspace size
	dgelss_( &m, &n, &nrhs, a, &lda, b, &ldb, jpvt, &rcond, &rank, work_query, &lwork, &info);

	// allocate work to be correct size
	double *work = (double *) malloc(sizeof(double)*work_query[0]);
	lwork = work_query[0];

	// Evaluate least squares system
	dgelss_( &m, &n, &nrhs, a, &lda, b, &ldb, jpvt, &rcond, &rank, work, &lwork, &info);


	/*dgels_( &m, &n, &nrhs, a, &lda, b, &ldb, work_query, &lwork, &info);

	// allocate work to be correct size
	double *work = (double *) malloc(sizeof(double)*work_query[0]);
	lwork = work_query[0];

	// Evaluate least squares system
	dgels_( &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &info);*/

	/*int *jpvt1 = (int *) malloc(sizeof(double) * n);
	dgelsy_( &m, &n, &nrhs, a, &lda, b, &ldb, jpvt1, &rcond, &rank, work_query, &lwork, &info);

	// allocate work to be correct size
	double *work = (double *) malloc(sizeof(double)*work_query[0]);
	lwork = work_query[0];

	// Evaluate least squares system
	dgelsy_( &m, &n, &nrhs, a, &lda, b, &ldb, jpvt1, &rcond, &rank, work, &lwork, &info);*/

	//////////////////////////////////////////////////////////////////
	// Get regression coefficients: the first X->size2 entries in b //
	//////////////////////////////////////////////////////////////////

	// re-allocate beta if it is not the correct size
	if( beta->size != X->size2){
		gsl_vector_free( beta );
		beta = gsl_vector_alloc( X->size2 );
	}

	memcpy( beta->data, b, sizeof(double) * beta->size );

	gsl_matrix_free( t_X );
	gsl_matrix_free( X_weighted );
	gsl_vector_free( weights_sqrt );
	free( b );
	free( jpvt );
	free( work );

}

gsl_matrix * gsl_lapack_get_full_rank_matrix( const gsl_matrix *M, const double rankPrecision){

	gsl_matrix *t_U, *V;
	gsl_vector *singular_values;
	int info;

	// SVD take a wide matrix, so transpose M
	gsl_matrix *t_M = gsl_matrix_alloc( M->size2, M->size1);
	gsl_matrix_transpose_memcpy( t_M, M);

	// perform SVD
	info = gsl_lapack_svd( t_M, t_U, singular_values, V);

	if ( info != 0 ){
		#ifdef PRINT_TO_CONSOLE
		cerr << "Lapack Error in gsl_lapack_get_full_rank_matrix: info = " << info << endl;
		exit(1);
		#endif
	}

	// get largest singular value
	double sigmaMax = gsl_vector_get( singular_values, 0);

	// set cutoffIndex to the index of the last singular value that is > precision * sigmaMax
	unsigned int cutoffIndex = 0;
	for(unsigned int i=0; i<singular_values->size; i++){
		if( gsl_vector_get( singular_values, i) < 1e-8 * sigmaMax){
			break;
		}
		cutoffIndex++;
		//cout << "s" << i << "   " << gsl_vector_get( singular_values, i) << endl;
	}

	//cout << "cutoffIndex: " << cutoffIndex << endl;

	// copy relevant rows of t_U to columns of M_fullRank

	gsl_matrix *M_fullRank = gsl_matrix_alloc(M->size1, cutoffIndex);
	gsl_vector *temp = gsl_vector_alloc( M->size1 );

	for(unsigned int i=0; i<cutoffIndex; i++){

		gsl_matrix_get_row( temp, t_U, i);

		gsl_matrix_set_col( M_fullRank, i, temp);
	}

	// clean up
	gsl_matrix_free( t_U );
	gsl_matrix_free( V );
	gsl_matrix_free( t_M );
	gsl_vector_free( singular_values );
	gsl_vector_free( temp );

	// save
	//gsl_matrix_save(M, "M.txt");
	//gsl_matrix_save(t_U, "t_U.txt");
	//gsl_matrix_save(V, "V.txt");
	//gsl_vector_save(singular_values, "sv.txt");

	return( M_fullRank);
}

int gsl_lapack_chol_invert(gsl_matrix *M){

	int info = 0;
	char uplo = 'U';
	int n = M->size1;

	// Cholesky decomposition
	dpotrf_(&uplo, &n, M->data, &n, &info);

	// If A does not span full space
	// i.e. A does not have a valid Cholesky decomposition
	if( info != 0 ){
		//cerr << "Cholesky decomposition of singular matrix failed" << endl;
		//assert( info == 0);

		// return error
		return GSL_ERANGE;
	}

	// inversion from Cholesky
	dpotri_(&uplo, &n, M->data, &n, &info);

	if( info != 0 ){
		return GSL_ERANGE;
	}
	return GSL_SUCCESS;
}

int gsl_lapack_chol_invert_logdet(gsl_matrix *M, double &log_det){

	int info = 0;
	char uplo = 'U';
	int n = M->size1;

	// Cholesky decomposition
	dpotrf_(&uplo, &n, M->data, &n, &info);

	// If A does not span full space
	// i.e. A does not have a valid Cholesky decomposition
	if( info != 0 ){
		//cerr << "Cholesky decomposition of singular matrix failed" << endl;
		//assert( info == 0);
		return GSL_ERANGE;
	}

	// Log determinant
	log_det = 0;

	// ln_det = log( prod( diag( M ))) = sum( log( diag( M )))
	for(unsigned int i=0; i<M->size1; i++){
		log_det += log(gsl_matrix_get(M, i, i));
	}

	// multiply by 2 because A Cholesky decomp was used
	log_det *= 2;

	// inversion from Cholesky
	dpotri_(&uplo, &n, M->data, &n, &info);

	if( info != 0 ){
		return GSL_ERANGE;
	}
	return GSL_SUCCESS;
}

double gsl_lapack_chol_logdet(gsl_matrix *M){

	int info = 0;
	char uplo = 'U';
	int n = M->size1;

	// Cholesky decomposition
	dpotrf_(&uplo, &n, M->data, &n, &info);

	// If A does not span full space
	// i.e. A does not have a valid Cholesky decomposition
	if( info != 0 ){
		#ifdef PRINT_TO_CONSOLE
		cerr << "Cholesky decomposition of singular matrix failed" << endl;
		assert( info == 0);
		#endif
	}

	// Log determinant
	double log_det = 0;

	// ln_det = log( prod( diag( M ))) = sum( log( diag( M )))
	for(unsigned int i=0; i<M->size1; i++){
		log_det += log(gsl_matrix_get(M, i, i));
	}

	// multiply by 2 because A Cholesky decomp was used
	log_det *= 2;

	return( log_det );
}

bool gsl_lapack_is_full_rank(const gsl_matrix* X, const double conditionNumber){

	gsl_matrix *C = gsl_matrix_alloc( X->size2, X->size2);
	gsl_matrix *t_U;
	gsl_vector *eigenValues;

	// C = crossprod(X)
	gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, X, X, 0.0, C);

	gsl_lapack_eigen(C, t_U, eigenValues);

	double Cnum = gsl_vector_get(eigenValues, 0) / gsl_vector_get(eigenValues, eigenValues->size-1);

	gsl_matrix_free( C );
	gsl_matrix_free( t_U );
	gsl_vector_free( eigenValues );

	return( fabs(Cnum) > conditionNumber );
}

// Confirm behvaior using LU in library(Matrix)

int gsl_lapack_lu_invert(gsl_matrix *M){

	int n = M->size1;
	int *IPIV = (int *) malloc((n+1)*sizeof(int));
	int LWORK = n*n;
	double *WORK = (double*) malloc(LWORK*sizeof(double));
	int info;

	// LU decomposition
	dgetrf_(&n, &n, M->data, &n, IPIV, &info);

	if( info != 0 )return GSL_ERANGE;

	// Invert	
	dgetri_(&n, M->data,&n, IPIV, WORK, &LWORK, &info);

	if( info != 0 ) return GSL_ERANGE;
	
	return GSL_SUCCESS;
}

int gsl_lapack_lu_invert_logdet(gsl_matrix *M, double *log_det){

	int n = M->size1;
	int *IPIV = (int *) malloc((n+1)*sizeof(int));
	int LWORK = n*n;
	double *WORK = (double*) malloc(LWORK*sizeof(double));
	int info;

	// LU decomposition
	dgetrf_(&n, &n, M->data, &n, IPIV, &info);

	if( info != 0 ){
		free(IPIV);
		free(WORK);
		return GSL_ERANGE;
	}

	// Log Det
	*log_det = 0;

	for(int i=0; i<n; i++){
		// Check if first entry == 1, etc
		if( IPIV[i] != i+1)
			*log_det -= log( gsl_matrix_get(M,i,i) );
		else			
			*log_det += log( gsl_matrix_get(M,i,i) );
	}

	// Invert	
	dgetri_(&n, M->data, &n, IPIV, WORK, &LWORK, &info);

	if( info != 0 ){
		free(IPIV);
		free(WORK);
		return GSL_ERANGE;
	}

	free(IPIV);
	free(WORK);	

	return GSL_SUCCESS;
}

int gsl_lapack_lu_logdet(gsl_matrix *M, double *log_det){

	int n = M->size1;
	int *IPIV = (int *) malloc((n+1)*sizeof(int));
	int LWORK = n*n;
	double *WORK = (double*) malloc(LWORK*sizeof(double));
	int info;

	// LU decomposition
	dgetrf_(&n, &n, M->data, &n, IPIV, &info);

	if( info != 0 ){
		free(IPIV);
		free(WORK);
		return GSL_ERANGE;
	}

	// Log Det
	*log_det = 0;

	for(int i=0; i<n; i++){		
		// Check if first entry == 1, etc
		if( IPIV[i] != i+1)
			*log_det -= log( gsl_matrix_get(M,i,i) );
		else			
			*log_det += log( gsl_matrix_get(M,i,i) );
	}
	
	free(IPIV);
	free(WORK);

	return GSL_SUCCESS;
}
