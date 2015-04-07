/*
 * gsl_lapack.h
 *
 *  Created on: Jan 12, 2011
 *      Author: gh258
 */

#ifndef GSL_LAPACK_H_
#define GSL_LAPACK_H_

#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_vector_double.h>

/*
#ifdef __GSL_CBLAS_H__
#define __MKL_CBLAS_H__
#endif*/

//#ifndef INTEL
extern "C" {
	int dgeev_(char *jobvl, char *jobvr, int *n, double *a, int *lda, double *wr, double *wi, double *vl, int *ldvl, double *vr, int *ldvr, double *work, int *lwork, int *info);

	//int dgels_(char *trans, int *m, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, double *work, int *lwork, int *info);

	int dgels_(int *m, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, double *work, int *lwork, int *info);

	// invert matrix with LU decomposition
	int dsytri_(char *uplo, int *n, double *a, int *lda, int *ipiv, double *work, int *info);

	// Cholesky decomposition
	int dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);

	// Inversion based on results of Cholesky decomposition
	int dpotri_(char *uplo, int *n, double *a, int *lda, int *info);

	// Solve least squares system based on results of Cholesky decomposition
	int dpotrs_(char *uplo, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, int *info);

	// SVD
	int dgesvd_(char *jobu, char *jobvt, int *m, int *n, double *a, int *lda, double *s, double *u, int *ldu,
			double *vt, int *ldvt, double *work, int *lwork, int *info);

	/**
	 * Computes the minimum norm least squares solution to an over-
	 * or under-determined system of linear equations A X=B, using a
     * complete orthogonal factorization of A.
	 */
	int dgelsy_( int *m, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, int *jpvt, double *rcond, int *rank, double *work, int *lwork, int *info);

	/**
	 * Computes the minimum norm least squares solution to an over-
     * or under-determined system of linear equations A X=B,  using
     * the singular value decomposition of A.
	*/
	int dgelss_( int *m, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, double *S, double *rcond, int *rank, double *work, int *lwork, int *info);

	/**
	 * Computes all eigenvalues and, optionally, eigenvectors of a real symmetric matrix A.
	 */
	int dsyev_(char *jobz, char *uplo, int *n, double *a, int *lda, double *w, double *work, int *lwork, int *info);

	/**
	 * Computes eigen-values (and eigen-vectors) for non-symmetic square matrix
	*/
	int dgeev_(char *JOBVL, char *JOBVR, int *N, double *A, int *LDA, double *WR, double *WI, double *VL, int *LDVL, double *VR, int *LDVR, double *WORK, int *LWORK, int *INFO );

	// LU decomoposition of a general matrix
    void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);

    // generate inverse of a matrix given its LU decomposition
    void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);

 }
//#endif

/*! \brief Computes the singular value decomposition of t(X), and returns t_U, s, and V using the LAPACK function dgesvd_
 * \return info from dgesvd_, which is zero on success
 */
int gsl_lapack_svd(const gsl_matrix *X, gsl_matrix *& t_U, gsl_vector *& singular_values, gsl_matrix *& V);

/*! \brief Computes the eigen-decomposition of symmetric matrix C, and returns eigenvectors U, and eigenvalues eigen_values, using the LAPACK function dsyev_
 * \return info from dsyev_, which is zero on success
 */
int gsl_lapack_eigen(const gsl_matrix *C, gsl_matrix *& t_U, gsl_vector *& eigen_values);

int gsl_lapack_eigen_lowrank(const gsl_matrix *C, gsl_matrix *& t_U, gsl_vector *& eigen_values, const int rank);

/*!\brief Computes eigen values for non-symmetric matrices, and C is modified
*/
int gsl_lapack_eigenValues(gsl_matrix *C, gsl_vector *eigen_values);

/*! \brief Evaluate least squares solution to Y = X\beta, by calculating \beta where X is over- or under-determined
 *  \param Y gsl_vector of responses for each individual
 *  \param X design matrix
 *  \param beta on exit, set to regression coefficients
 *  \param rank on exit, set to rank of X
 *  \param sigma_sq_e on exit, set to the residual variance of the regression
 *  \param log_L on exit, set to log-likelihood of model
 *  \param report_p_values if true, and if X is determined to be full rank, evaluate solve(X^TX) and return p-values for each variable based on a t-test
 *  \return if report_p_values is true, and system is full rank, return a gsl_vector of p-values from a t-test, otherwise return a NULL pointer
 */
gsl_vector* gsl_lapack_fit_least_squares(const gsl_vector *Y, const gsl_matrix *X, gsl_vector *& beta, int * rank, double *sigma_sq_e, double *log_L, bool report_p_values);

/*! \brief Evaluate weighted least squares system
 *  \param Y gsl_vector of responses for each individual
 *  \param X design matrix
 *  \param weights weights, one for each individuals
 *  \param beta on exit, set to regression coefficients
 */
void gsl_lapack_weighted_least_squares(const gsl_vector *Y, const gsl_matrix *X, const gsl_vector *weights, gsl_vector *& beta );

/*! \brief Given a matrix M, possible singular, return a full rank matrix that spans the same space.  Determine computational singularity based on s[i] > rankPrecision *max(s)
 *
 * \param M input matrix
 * \param rankPrecision scalar multiple of maximum singular value below which, singular values are considered effectively zero
 */
gsl_matrix * gsl_lapack_get_full_rank_matrix( const gsl_matrix *M, const double rankPrecision = 1e-8);

/*! \brief Invert M using a Cholesky decomposition
 * @param M square matrix.  Values in M are destroyed and replaced with inverse
 */
int gsl_lapack_chol_invert(gsl_matrix *M);

/*! \brief Invert M using a Cholesky decomposition and calculate the log determinant
 * @param M square matrix.  Values in M are destroyed and replaced with inverse
 */
int gsl_lapack_chol_invert_logdet(gsl_matrix *M, double &log_det);

/*! \brief Use Cholesky decomposition to calculate the log determinant
 * @param M square matrix.  Values in M are destroyed and replaced with inverse
 */
double gsl_lapack_chol_logdet(gsl_matrix *M, double &log_det);

bool gsl_lapack_is_full_rank(const gsl_matrix* X, const double conditionNumber );

/*! \brief Invert M using a LU decomposition
 * @param M square matrix.  Values in M are destroyed and replaced with inverse
 */
int gsl_lapack_lu_invert(gsl_matrix *M);

/*! \brief Invert M using a LU decomposition and calculate the log determinant
 * @param M square matrix.  Values in M are destroyed and replaced with inverse
 */
int gsl_lapack_lu_invert_logdet(gsl_matrix *M, double *log_det);

/*! \brief Use LU decomposition to calculate the log determinant
 * @param M square matrix.  Values in M are destroyed and replaced with inverse
 */
int gsl_lapack_lu_logdet(gsl_matrix *M, double *log_det);

/*! \brief Invert 2x2 matrix manually from the standard formula
 * @param M 2x2 matrix. 
 * @param result the inverse of M
 */
inline int gsl_lapack_inverse_2by2(const gsl_matrix *M, gsl_matrix * result, double *det);




/*
void test_lapack(){

 	{
 		int info;
 		double *w, *vect1, *vect2, *wr, *wi;
 		int c_m1, i;
 		char yes;
 		int dim = 4;
 		int worksize_int;
 		double worksize_double;

 		// double matrice[16] ={43,216,254,249,216,198,193,211,254,193,177,171,249,211,171,169};


 		double matrice[16] ={10, 2, 4, 2, 2, 10, 3, 3, 4, 3, 10, 1, 2, 3, 1, 10};

 		gsl_matrix *H = gsl_matrix_alloc(4, 4);

 		H->data = matrice;

 		cout << "H:" << endl;
 		gsl_matrix_print(H);
 		cout << endl << endl;

 		/////////
 		// GSL //
 		/////////

 		gsl_linalg_cholesky_decomp(H);
 		cout << "chol(H):" << endl;
 		gsl_matrix_print(H);
 		cout << endl << endl;

 		gsl_linalg_cholesky_invert(H);
 		cout << "solve(H):" << endl;
 		gsl_matrix_print(H);
 		cout << endl << endl;

 		cout << "____________________________________" << endl;


 		double g[16] ={10, 2, 4, 2, 2, 10, 3, 3, 4, 3, 10, 1, 2, 3, 1, 10};

 		H->data = g;

 		/////////////
 		// Lapack  //
 		/////////////

 		//int dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);

 		int info_chol = 0;
 		char uplo = 'U';
 		int n = H->size1;

 		dpotrf_(&uplo, &n, H->data, &n, &info_chol);

 		cout << "chol(H):" << endl;
		gsl_matrix_print(H);
		cout << endl << endl;

		dpotri_(&uplo, &n, H->data, &n, &info_chol);

		gsl_matrix_triangular_to_full(H, uplo);

		cout << "solve(H):" << endl;
		gsl_matrix_print(H);
		cout << endl << endl;


		clock_t start,end;
		start = clock();
 		for(int i=0; i<1e7; i++){

 			double temp[16] ={10, 2, 4, 2, 2, 10, 3, 3, 4, 3, 10, 1, 2, 3, 1, 10};

 			H->data = temp;

 	  		//gsl_linalg_cholesky_decomp(H);
 	  		//gsl_linalg_cholesky_invert(H);

 	  		dpotrf_(&uplo, &n, H->data, &n, &info_chol);
 			dpotri_(&uplo, &n, H->data, &n, &info_chol);

 			gsl_matrix_triangular_to_full(H, uplo);

 		}
 		end = clock();
		double dif = (double) (end - start) / CLOCKS_PER_SEC;
		printf ( "%.2f sec\n", dif );

 		exit(1);

 		yes = 'V';
 		c_m1 = -1;

 		vect1 = (double *) malloc(dim * dim * sizeof(double));
 		vect2 = (double *) malloc(dim * dim * sizeof(double));
 		wr = (double *) malloc(dim * sizeof(double));
 		wi = (double *) malloc(dim * sizeof(double));


 		dgeev_(&yes, &yes, &dim, matrice, &dim, wr, wi, vect1, &dim, vect2, &dim, &worksize_double, &c_m1, &info);

 		worksize_int = (int)worksize_double;
 		w = (double *) malloc(worksize_int * sizeof(double));
 		printf("Work size : %d\n",worksize_int);

 		// here is the bug:
 		dgeev_(&yes, &yes, &dim, matrice, &dim, wr, wi, vect1, &dim, vect2, &dim, w, &worksize_int, &info);

 		for (i=0;i<16;i++)
 		{
 		printf("Matrice : %f\n",wr[i]);
 		}

 		free(vect1); free(vect2); free(w);

 	}

 	printf("try lapack\n");




 	int i, j;
 	int N, NRHS, LDA, LDB;
 	double *B, *AA;
 	static int NDIM = 4;
 	int IPIV[4];
 	int INFO;

 	AA = (double*) malloc(NDIM*NDIM*sizeof(double));
 	B = (double*) malloc (NDIM*sizeof(double));

 	N=NDIM; NRHS=1; LDA=NDIM; LDB=NDIM;

 	AA[0] = 1.0;
 	AA[4] = -1.0;
 	AA[8] = 2.0;
 	AA[12] = -1.0;
 	AA[1] = 2.0;
 	AA[5] = -2.0;
 	AA[9] = 3.0;
 	AA[13] = -3.0;
 	AA[2] = 1.0;
 	AA[6] = 1.0;
 	AA[10] = 1.0;
 	AA[14] = 0.0;
 	AA[3] = 1.0;
 	AA[7] = -1.0;
 	AA[11] = 4.0;
 	AA[15] = 3.0;

 	for (i=0;i<N;i++){
 	   for (j=0;j<N;j++) {
 		 printf("   %f  \n",AA[i+N*j]);
 	   }
 	}

    B[0] = -8.0;
    B[1] = -20.0;
    B[2] = -2.0;
    B[3] = 4.0;


   dgesv_(&N, &NRHS, AA, &LDA, &IPIV, B, &LDB, &INFO);

 	printf("info %d \n",INFO);

 	   for (i=0;i<N;i++)
 	      printf("   %f \n",B[i]);
 	   //



 }*/


#endif /* GSL_LAPACK_H_ */
