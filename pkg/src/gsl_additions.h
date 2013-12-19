#ifndef GSL_ADDITIONS_H
#define GSL_ADDITIONS_H

#include <vector>
#include <string>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_permutation.h>

const double EFFECTIVELY_ZERO = 1e-10;

typedef enum {LINEAR, LOGISTIC, LINEAR_MIXED_MODEL} regressionType;

using namespace std;

void gsl_vector_subtr_restrict(const gsl_vector * __restrict__ A, const gsl_vector * __restrict__ B, gsl_vector * __restrict__ C);

// print gsl_matrix to STDOUT
void gsl_matrix_print(const gsl_matrix *mat);

void gsl_matrix_print(const gsl_matrix_short *mat);

void gsl_matrix_print(const gsl_matrix_char *mat);

// print gsl_matrix to file
void gsl_matrix_save(const gsl_matrix *mat, const string &path);

// print the transpose of a gsl_matrix to file
void gsl_matrix_save_transpose(const gsl_matrix *mat, const string &path);

// Read in gsl_matrix from file
gsl_matrix * gsl_matrix_read( const string &file );

// Read in gsl_vector from file
gsl_vector * gsl_vector_read( const string &file );

// print gsl_matrix_short to file
void gsl_matrix_short_save(const gsl_matrix_short *mat, const string &path);

// add 1 to mat[row, col]
void gsl_matrix_increment_element(gsl_matrix *mat, int row, int col);

// result = cbind(left, right)
void gsl_matrix_cbind(const gsl_matrix * left, const gsl_matrix * right, gsl_matrix * result);

// result = cbind(left, right)
void gsl_vector_cbind(const gsl_vector * left, const gsl_vector * right, gsl_matrix * result);

// gsl_matrix multiplication
// C = A %*% B
// calls cblas_dgemm
void gsl_matrix_product(const gsl_matrix * A, const gsl_matrix * B, gsl_matrix * C );

// gsl_matrix multiplication for symmetric matrix A
// C = A %*% B
// calls cblas_dsymm
void gsl_matrix_product_sym(const gsl_matrix * A, const gsl_matrix * B, gsl_matrix * C );

//
void gsl_matrix_get_omission(const gsl_matrix *mat, const int & index, gsl_matrix *result);
void gsl_matrix_get_omission_from_vector(const gsl_matrix *mat, const int &index, gsl_matrix *result);
void gsl_matrix_get_omission_first2(const gsl_matrix *mat, gsl_matrix *result);

// invert gsl_matrix mat using LU decomposition
void gsl_matrix_LU_solve(const gsl_matrix *mat, gsl_matrix *mat_solve);

// invert gsl_matrix mat using Cholesky decomposition
// Very fast for positve definite matrices
void gsl_matrix_chol_solve(const gsl_matrix *mat, gsl_matrix *mat_solve);

// matrix addition
// C = A + B
void gsl_matrix_add(const gsl_matrix * A, const gsl_matrix * B, gsl_matrix * C);

// vector addition
// C = A + B
void gsl_vector_add(const gsl_vector * A, const gsl_vector * B, gsl_vector * C);

// matrix subtraction
// C = A - B
void  gsl_matrix_subtr(const gsl_matrix * A, const gsl_matrix * B, gsl_matrix * C);

// calculates .025, .5 and .975 quantiles of a vector of doubles
void post_process_MCMC_chain(vector<double> chain_in, double &median, double &quantile_025, double &quantile_975);

// mean of entries in matrix
double gsl_matrix_mean(const gsl_matrix *mat);

// mean of entries in vector
double gsl_vector_mean(const gsl_vector *V);

// scalar product
// result = scalar * mat
void gsl_matrix_scalar_product(const gsl_matrix * mat, const double & scalar, gsl_matrix *result);

// R-style crossprod
// result = crossprod(mat)
void gsl_matrix_crossprod(const gsl_matrix *mat, gsl_matrix *result);

// R-style tcrossprod
// result = tcrossprod(mat)
void gsl_matrix_tcrossprod(const gsl_matrix *mat, gsl_matrix *result);


// place a column of ones on the left of gsl_matrix
// Used for including a intercept term in regression
void gsl_matrix_augment_with_intercept(const gsl_matrix *mat, gsl_matrix *result);

// trace
double gsl_matrix_trace(const gsl_matrix *mat);

// quadratic form
// evaluates t(a) %*% B %*% a when a is a column vector
double gsl_matrix_quadratic_form(const gsl_matrix *a, const gsl_matrix *B);


/*! \brief Return quadratic form t(y) %*% Q %*% y
 * @param Q matrix
 * @param y vector
 */
double gsl_matrix_quadratic_form( const gsl_matrix *Q, const gsl_vector *y);


/*! \brief Return quadratic form t(y) %*% Q %*% y
 * @param Q matrix with is lower symmetric
 * @param y vector
 */
double gsl_matrix_quadratic_form_sym( const gsl_matrix *Q, const gsl_vector *y);

/*! \brief Evaluate quadratic form S = t(X) %*% Q %*% X
 * @param Q matrix with is lower symmetric
 * @param X matrix
 */
void gsl_matrix_quadratic_form_sym( const gsl_matrix *Q, const gsl_matrix *X, gsl_matrix *S);
void gsl_matrix_quadratic_form_sym( const gsl_matrix *Q, const gsl_vector *X, gsl_matrix *S);

// use an LU decomposition to calculate inverse and determinant of matrix
void gsl_matrix_LU_solve_lndet(const gsl_matrix *mat, gsl_matrix * mat_solve, double &det);

void gsl_matrix_pseudo_inverse_lndet(const gsl_matrix *M, gsl_matrix * result, double &ln_det);


// Pseudo_inverse of K where K = Q %*% t(Q), where only Q is needed
// Calculating svd(Q) and creating pseudo_inverse(K) is much faster than u eigen(K)
// If make_Q_full_rank == true, return a full rank Q using the previously computed SVD

// Justification
// If Q = UDV^T then QQ^T = (UDV^T)(VDU^T)
// 		since V is an orthogonal matirx V^TV = I
//		so QQ^T = (UD)(DU^T)
// Squaring and inverting D,
// 		pseudo_inverse(QQ^T) = U(D^-2)U^T
gsl_matrix * gsl_matrix_pseudo_inverse_factor(gsl_matrix *& Q, gsl_matrix *result, bool make_Q_full_rank = false);


// use a Cholesky decomposition to calculate inverset and determinant of matrix
// facter for positve definite matrices
void gsl_matrix_chol_solve_lndet(const gsl_matrix *mat, gsl_matrix * mat_solve, double &ln_det);


//void gsl_matrix_add_scalar_to_diags(const gsl_matrix *mat, const double & scalar, gsl_matrix *result);

// dot product between A and B
double gsl_vector_dot_prod(const gsl_vector *A, const gsl_vector *B);

// print gsl_vector
void gsl_vector_print(const gsl_vector *vec);

// print gsl_vector_short
void gsl_vector_short_print(const gsl_vector_short *vec);

// omit a column from a matrix
// result = Mat[,-col_omit]
void gsl_matrix_omit_col(const gsl_matrix *Mat, const int &col_omit, gsl_matrix *result);

// eigen decomposition
void gsl_eigen_decomp(const gsl_matrix *A_in, gsl_vector *eigen_values, gsl_matrix *eigen_vectors, bool sort_eigen_vals);

// copies a subset of the rows of A in to sub_A
// vector col_indeces identifies which rows
void gsl_matrix_sub_row(const gsl_matrix *A, const vector<int> &col_indeces, gsl_matrix *sub_A);

// copies a subset of the cols of A in to sub_A
// vector col_indeces identifies which cols
void gsl_matrix_sub_col(const gsl_matrix *A, const vector<int> &col_indeces, gsl_matrix *sub_A);

// extracts the first 0:(max_col-1) columns
void gsl_matrix_sub_col(const gsl_matrix *A, const int max_col, gsl_matrix *sub_A);

// copies a subset of the rows and cols of A in to sub_A
void gsl_matrix_sub_row_col(const gsl_matrix *C, const vector<int> &col_indeces, gsl_matrix *sub_C);

// copies a subset of the entries of v to v_sub
void gsl_vector_subset(const gsl_vector *v, const vector<int> &indeces, gsl_vector *v_sub);

// trace of a matrix product
// trace(A %*% B)
// this function only evaluates the diagonals of the product
// so the trace can be evaluated in O(n^2) instead of O(n^3)
double gsl_matrix_product_trace(const gsl_matrix *A, const gsl_matrix *B);

// return diag(A * B)
// Diag only uses the diagonals, so don't actually compute A*B
// so the diag can be evaluated in O(n^2) instead of O(n^3)
gsl_vector *gsl_matrix_product_diag(const gsl_matrix *A, const gsl_matrix *B);

// Divide each column in M by the corresponding entry in v
// element-wise division so that first column of M is divided by first element in v
void gsl_matrix_vect_div_col(const gsl_matrix *M, const gsl_vector *v, gsl_matrix *result);


// Divide each row in M by the corresponding entry in v
// element-wise division so that first column of M is divided by first element in v
void gsl_matrix_vect_div_row(const gsl_matrix *M, const gsl_vector *v, gsl_matrix *result);


// Multiply each column in M by the corresponding entry in v
void gsl_matrix_vect_mult(const gsl_matrix *M, const gsl_vector *v, gsl_matrix *result);

// element-wise reciprocal
void gsl_matrix_recip(const gsl_matrix *M, gsl_matrix *result);

// sum of all elements in the vector
double gsl_vector_sum_elements(const gsl_vector * vect);

// set X_signif equal to a subset of columns of X defined by signif_markers
void gsl_matrix_cbind_from_list(const gsl_matrix *X, const vector<int> &marker_list, gsl_matrix *X_subset);

// add scalar to the diagonals of M
void gsl_matrix_add_diag_scalar(gsl_matrix *M, const double scalar);

// add scalar to the diagonals of M if M is square
void gsl_matrix_add_diag_scalar(const gsl_matrix *M, const double scalar, gsl_matrix *result);

// add vector to the diagonals of M if M is square
void gsl_matrix_add_diag(gsl_matrix *M, const gsl_vector *v);

// set diagonals of result to scalar, where result is already diagonal
void gsl_matrix_identity_scalar(const double scalar, gsl_matrix *M);

// set diag(M) = v
// In other words, set M[i,i] = v[i]
void gsl_matrix_set_diag(gsl_matrix *M, const gsl_vector *v);

// set diag(M) = v
void gsl_matrix_set_diag(gsl_matrix *M, const double value);

gsl_vector *gsl_matrix_get_diag(gsl_matrix *M);

enum {PARALLEL, ORTHOGONAL};


// result = scalar * v
void gsl_vector_scalar_product(const gsl_vector* v, const double & scalar, gsl_vector *result);

// C = A - B
void gsl_vector_subtr(const gsl_vector * A, const gsl_vector * B, gsl_vector * C);

void gsl_matrix_pseudo_inverse(const gsl_matrix * M, gsl_matrix *result, const double min_e_value = EFFECTIVELY_ZERO);

void gsl_matrix_pseudo_inverse_log_det(const gsl_matrix * M, gsl_matrix *solve_M, double &log_det);


// Input:
// a, b: gsl_vectors of the same dimension
// direction: which component should be returned (PARALLEL or ORTHOGONAL)

// Output:
// Get the component of a that is orthogonal to b.
void gsl_matrix_project_onto(const gsl_vector *a, const gsl_vector *b, gsl_vector *result, int direction=ORTHOGONAL);

// Input:
//	A: a matrix of any dimension
//	X: a 1 col matrix, with as many rows as A
//	result: the location where the resulting matrix is strored.  Must be the same dimensions as A

// Output:
//	Returns the components of each column of A that is orthogonal to X
void gsl_matrix_orthogonal_components(const gsl_matrix *A, const gsl_vector *X, gsl_matrix *result);

// mean center the entries in M
void gsl_matrix_mean_center(const gsl_matrix *M, gsl_matrix *result);

// mean center the entries in v
void gsl_vector_mean_center(const gsl_vector *v, gsl_vector *result);


// Input: M as symmetric positive definite matrix

// Output:
// 		returns L such that L %*% t(L) = M
void gsl_linalg_cholesky_upper(const gsl_matrix *M, gsl_matrix *L);

// return column i of gsl_matrix M
// same as gsl_matrix_get_col except it returns a gsl_matrix instead of gsl_vector
void gsl_matrix_get_column(const gsl_matrix *M, const int i, gsl_matrix *result);

// Use SVD to return linearly independent columns of M
// return a gsl_matrix with n_cols = rank of M
// M = U %*% diag(lambda) %*% t(V)
gsl_matrix * gsl_matrix_SVD_independent_columns(const gsl_matrix *M);

void gsl_vector_elementwise_product(const gsl_vector *A, const gsl_vector *B, gsl_vector *C);

// Input:
//	Q is a tall matrix with a decaying spectrum of singular values

// Output:
//	return a matrix that is a low rank approximation of Q and captures x% of the variance determined by percent_variance_captured
gsl_matrix * gsl_low_rank_approximation(const gsl_matrix *Q, double percent_variance_captured, int min_rank);

// adds a scalar to each entry in V
// V and results can point to the same gsl_vector
void gsl_vector_add_scalar(const gsl_vector *V, const double &scalar, gsl_vector *result);

// return log of probabilities of entries in x
// return sum(dexp(x, rate, log=T))
double gsl_ran_exponential_pdf_ln(const gsl_vector *x, const double &rate);

// return log of probabilities of entries in x
// return sum(dexp(x, rate, log=T))
double gsl_ran_scaled_inverse_chisquare_pdf_ln(const gsl_vector *x, const double &shape, const double &scale);

// Calculate the log probability of x under a Int{Exp(x|phi)Gamma(lambda, gamma)}d\phi
double gsl_exponential_gamma_ln(const gsl_vector *x, const double &lambda, const double &gamma);

void gsl_permutation_print(const gsl_permutation *perm);

void gsl_permutation_save(const gsl_permutation *perm, const string &path);

gsl_matrix_short *convert_to_gsl_matrix_short(const gsl_matrix *X);

/*
 * Reads a single column or single row file and returns a gsl_vector with the entries
 */
gsl_vector *gsl_vector_read_unknown_size(const string &file);

void gsl_vector_save(const gsl_vector *v, const string &path);

// Save entries in names and values in gsl_vector
void gsl_vector_save(const vector<string> &names, const gsl_vector *v, const string &path);

/*
 * Lapack function like dpotrf_ and dpotri_ return trianglular matrices
 * This function converts trianglular to symmetric
 * uplo indicates where the valid values are
 */
void gsl_matrix_triangular_to_full(gsl_matrix *M, const char uplo);

/*! \brief Returns a gsl_vector which has the elementwise product of and b
 *
 */
gsl_vector *gsl_vector_elementwise_product(const gsl_vector *a, const gsl_vector *b);

/*! \brief Save X in TPED format, making up chrom, name, genetic pos, physical pos
 * This is analogous to the out put from using --transpose --recode12 in plink
 */
void gsl_matrix_short_save_TPED_recode12(const gsl_matrix_short *X, const string &path);

/*! \brief Save X in TPED format, making up chrom, name, genetic pos, physical pos
 * The output is designed to be read by plink so genotypes are coded 0 -> A A; 1 -> A C; 2 -> CC
 * With pairs separated by a tab
 *
 */
void gsl_matrix_short_save_TPED(const gsl_matrix_short *X, const string &path);

/*! \brief Convert an array of doubles into a gsl_matrix of dimension size1 x size2 without copying or using extra memory
 *
 */
gsl_matrix * gsl_matrix_attach_array(double *array, const int size1, const int size2);

/*! \brief Convert an array of doubles into a gsl_vector  wit out copying or using extra memory
 *
 */
gsl_vector * gsl_vector_attach_array(double *array, const int size);

/**
 * \brief Like seq in R, return n evenly spaced values between start and stop
 */
gsl_vector *gsl_vector_seq( double start, double stop, int n);

/**
 * \brief Evaluate W %*% X where W is diagonal. If W_X == NULL, the matrix is allocated inside the function
 *
 * \return W_X = W %*% X, return by reference
 */
void gsl_matrix_diagonal_multiply( const gsl_matrix *X, const gsl_vector *w, gsl_matrix *&W_X, const bool byRow = true);

/**
 * \brief Multipy X = X %*% diag(w) in place
 */
void gsl_matrix_diagonal_multiply( gsl_matrix *X, const gsl_vector *w);

/**
 * \brief Evaluate t(X) %*% W %*% X where W is diagonal. M and W_X are allocated in the function if a NULL value is passed
 *
 * \return M return by reference t(X) %*% W %*% X and W_X = W %*% X
 */

void gsl_matrix_diagonal_quadratic_form( const gsl_matrix *X, const gsl_vector *w, gsl_matrix *&M , gsl_matrix * W_X = NULL);

/**
 * \brief return a subset of the entries in v that are indicated by sample_index
 */
gsl_vector *gsl_vector_subsample( const gsl_vector *v, const vector<int> sample_index);

/**
 * \brief Reverse the columns of M
 */
void gsl_matrix_reverse( gsl_matrix *M);

/**
 * \brief b = sqrt(a)
 */
void gsl_vector_sqrt( const gsl_vector *a, gsl_vector *b);

double gsl_ran_chisq_pdf_log( const double x, const double df );

double gsl_ran_inverse_chisq_pdf_log( const double x, const double df );

void inverse_logit( gsl_vector * __restrict__ mu, const gsl_vector * __restrict__ eta);

// y += alpha * x
int gsl_matrix_daxpy( double alpha, const gsl_matrix * x, gsl_matrix * y);

/**
* \brief Replace missing missing values with the mean.  If there is a missing entry, return true
*/
bool gsl_vector_set_missing_mean(gsl_vector *v);

#endif
