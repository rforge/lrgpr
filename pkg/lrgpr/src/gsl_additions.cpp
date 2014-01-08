#include "gsl_additions.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <sys/stat.h>
#include <assert.h>
#include <math.h>
#include <sys/timeb.h>
#include <fstream>

#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_sf.h>

#include "misc_functions.h"

using namespace std;

void gsl_vector_subtr_restrict(const gsl_vector * __restrict__ A, const gsl_vector * __restrict__ B, gsl_vector * __restrict__ C){
	for(unsigned int r=0; r<A->size; r++){
		gsl_vector_set(C, r, gsl_vector_get(A, r) - gsl_vector_get(B, r));
	}
}

// print gsl_matrix to STDOUT
void gsl_matrix_print(const gsl_matrix *mat){

	#ifdef PRINT_TO_CONSOLE

	unsigned int row, col;
	for(row=0; row< mat->size1; row++){
		for(col=0; col<mat->size2; col++){
			//cout << gsl_matrix_get(mat, row, col) << " ";
			printf("%12.10f ", gsl_matrix_get(mat, row, col));
		}
		cout << endl;
	}
	cout << endl;
	#endif
}

// print gsl_matrix to STDOUT
void gsl_matrix_print(const gsl_matrix_short *mat){

	#ifdef PRINT_TO_CONSOLE

	unsigned int row, col;
	for(row=0; row< mat->size1; row++){
		for(col=0; col<mat->size2; col++){
			//cout << gsl_matrix_get(mat, row, col) << " ";
			printf("%i ", gsl_matrix_short_get(mat, row, col));
		}
		cout << endl;
	}
	#endif
}

// print gsl_matrix to STDOUT
void gsl_matrix_print(const gsl_matrix_char *mat){

	#ifdef PRINT_TO_CONSOLE

	unsigned int row, col;
	for(row=0; row< mat->size1; row++){
		for(col=0; col<mat->size2; col++){
			//cout << gsl_matrix_get(mat, row, col) << " ";
			printf("%i ", gsl_matrix_char_get(mat, row, col));
		}
		cout << endl;
	}
	#endif
}

// print gsl_matrix to file
void gsl_matrix_save(const gsl_matrix *mat, const string &path){

	ofstream file(path.c_str());

	unsigned int row, col;
	for(row=0; row< mat->size1; row++){
		for(col=0; col<mat->size2; col++){
			//cout << gsl_matrix_get(mat, row, col) << " ";
			file << gsl_matrix_get(mat, row, col) << " ";
		}
		file << endl;
	}

	file.close();
}


// print the transpose of a gsl_matrix to file
void gsl_matrix_save_transpose(const gsl_matrix *mat, const string &path){

	ofstream file(path.c_str());

	unsigned int row, col;

	for(col=0; col<mat->size2; col++){
		for(row=0; row< mat->size1; row++){
			file << gsl_matrix_get(mat, row, col) << " ";
		}
		file << endl;
	}

	file.close();

}

// Read in gsl_matrix from file
gsl_matrix * gsl_matrix_read( const string &file ){

	if( ! fileExists( file ) ){

		#ifdef PRINT_TO_CONSOLE
		cout << "Does not exist: "<< file <<endl;
		exit(1);
		#endif
	}

	unsigned int nrow, ncol;

	get_matrix_dimensions( file, &nrow, &ncol );

	gsl_matrix *M = gsl_matrix_alloc( nrow, ncol );

	FILE *f = fopen( file.c_str(), "r");
	gsl_matrix_fscanf( f, M );
	fclose( f );

	return M;
}

// Read in gsl_vector from file
gsl_vector * gsl_vector_read( const string &file ){

	if( ! fileExists( file ) ){
	#ifdef PRINT_TO_CONSOLE
		cout << "Does not exist: "<< file <<endl;
		exit(1);
	#endif
	}

	unsigned int nrow, ncol;

	get_matrix_dimensions( file, &nrow, &ncol );

	if( min( nrow, ncol ) != 1 ){
		#ifdef PRINT_TO_CONSOLE
		cout << "Error in file: " << file << endl;
		cout << "Expected a vector in text format, where either the number of rows or columns is one" << endl;
		exit(1);
		#endif
	}

	gsl_vector *v = gsl_vector_alloc( max( nrow, ncol ) );

	FILE *f = fopen( file.c_str(), "r");
	gsl_vector_fscanf( f, v );
	fclose( f );

	return v;
}


// print gsl_matrix_short to file
void gsl_matrix_short_save(const gsl_matrix_short *mat, const string &path){

	ofstream file(path.c_str());

	unsigned int row, col;
	for(row=0; row< mat->size1; row++){
		for(col=0; col<mat->size2; col++){
			//cout << gsl_matrix_get(mat, row, col) << " ";
			file << gsl_matrix_short_get(mat, row, col) << " ";
		}
		file << endl;
	}

	file.close();
}


// increment mat[row, col]
void gsl_matrix_increment_element(gsl_matrix *mat, int row, int col){
	gsl_matrix_set(mat, row, col, gsl_matrix_get(mat, row, col) + 1);
}

// concatenate left and right matrices to get result
// result = cbind(left, right)
void gsl_matrix_cbind(const gsl_matrix * left, const gsl_matrix * right, gsl_matrix * result){
	// if left and right don't have the same number of rows
	if(left->size1 != right->size1){
		#ifdef PRINT_TO_CONSOLE
		cout<<"Cannot cbind two matrices different nrows\n";
		#endif
	}

	unsigned int nrow = left->size1;
	unsigned int ncol = left->size2 + right->size2;

	unsigned int r, c;
	for(r=0; r<nrow; r++){
		// left matrix
		for(c=0; c<left->size2; c++){
			gsl_matrix_set(result, r, c, gsl_matrix_get(left, r, c));
		}

		// right matrix
		for(c=left->size2; c<ncol; c++){
			gsl_matrix_set(result, r, c, gsl_matrix_get(right, r, c-left->size2));
		}
	}
}

// concatenate left and right matrices to get result
// result = cbind(left, right)
void gsl_vector_cbind(const gsl_vector * left, const gsl_vector * right, gsl_matrix * result){
	// if left and right don't have the same number of rows
	if(left->size != right->size){
		#ifdef PRINT_TO_CONSOLE
		cout<<"Cannot cbind two matrices different nrows\n";
		#endif
	}

	unsigned int nrow = left->size;
	unsigned int r;

	for(r=0; r<nrow; r++){
		// left matrix
		gsl_matrix_set(result, r, 0, gsl_vector_get(left, r));

		// right matrix
		gsl_matrix_set(result, r, 1, gsl_vector_get(right, r));
	}
}



void gsl_matrix_product(const gsl_matrix * A, const gsl_matrix * B, gsl_matrix * C ){

	// C = A*B
  	/*cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, matrixC->size1,matrixC->size2,
	 	matrixA->size2, 1.0, matrixA->data,matrixA->size2, matrixB->data,
	 	matrixB->size2,0.0, matrixC->data, matrixC->size2);
	 */
	 gsl_blas_dgemm(CblasNoTrans,CblasNoTrans, 1.0, A, B, 0.0, C);
}

// matrix A is symmetric
void gsl_matrix_product_sym(const gsl_matrix * A, const gsl_matrix * B, gsl_matrix * C ){

	// C = A*B
  	gsl_blas_dsymm(CblasLeft,CblasUpper, 1.0, A, B, 0.0, C);
}



// return mat[i, -i]
void gsl_matrix_get_omission(const gsl_matrix *mat, const int & index, gsl_matrix *result){


	for(int c=0; c<index; c++){
		gsl_matrix_set(result, 0, c, gsl_matrix_get(mat, index, c));
	}

	for(unsigned int c=index+1; c<mat->size1; c++){
		gsl_matrix_set(result, 0, c-1, gsl_matrix_get(mat, index, c));
	}

}

// Input: column matrix (i.e. vector)
// return mat[1:i, -i]
// do not include index
void gsl_matrix_get_omission_from_vector(const gsl_matrix *mat, const int &index, gsl_matrix *result){

	for(int c=0; c<index; c++){
		//cout<<c<<" ";
		gsl_matrix_set(result, c, 0, gsl_matrix_get(mat, c, 0));

	}

	for(unsigned int c=index+1; c<mat->size1; c++){
		//cout<<c<<" ";
		gsl_matrix_set(result, c-1, 0, gsl_matrix_get(mat, c, 0));

	}
}

// result = mat[2:n]
void gsl_matrix_get_omission_first2(const gsl_matrix *mat, gsl_matrix *result){
	for(unsigned int c=2; c<mat->size1; c++){
		gsl_matrix_set(result, c-2, 0, gsl_matrix_get(mat, c, 0));
	}


}



// mat_solve = solve(mat)
// invert matrix using LU decomposition
void gsl_matrix_LU_solve(const gsl_matrix *mat, gsl_matrix *mat_solve){
 	/*timeb t_start_solve, t_current_solve;
 	int t_diff_solve;

	 ftime(&t_start_solve);*/

	gsl_matrix *mat_copy = gsl_matrix_alloc(mat->size1, mat->size2);
	gsl_matrix_memcpy(mat_copy, mat);

	gsl_permutation *p = gsl_permutation_calloc(mat_copy->size2);
	int s=0;
	gsl_linalg_LU_decomp(mat_copy, p, &s);
	gsl_linalg_LU_invert(mat_copy, p, mat_solve);

	gsl_permutation_free(p);
	gsl_matrix_free(mat_copy);

	/*int solve_time = 0;
	 ftime(&t_current_solve);
	 t_diff_solve = (int) (1000.0 * (t_current_solve.time - t_start_solve.time)  + (t_current_solve.millitm - t_start_solve.millitm));
	 solve_time += t_diff_solve;
	 cout<< t_diff_solve<<endl;*/

}

// invert positive definite matrix using Cholesky decomposition
void gsl_matrix_chol_solve(const gsl_matrix *mat, gsl_matrix *mat_solve){

	// copy matrix because solving is destructive
	/*gsl_matrix *mat_copy = gsl_matrix_alloc(mat->size1, mat->size2);
	gsl_matrix_memcpy(mat_copy, mat);

	gsl_linalg_cholesky_decomp(mat_copy);

	gsl_linalg_cholesky_invert(mat_copy);

	gsl_matrix_memcpy(mat_solve, mat_copy);

	gsl_matrix_free(mat_copy);*/


	// copy matrix because solving is destructive
	gsl_matrix_memcpy(mat_solve, mat);

	gsl_linalg_cholesky_decomp(mat_solve);

	gsl_linalg_cholesky_invert(mat_solve);
}



// C = A + B
void gsl_matrix_add(const gsl_matrix * A, const gsl_matrix * B, gsl_matrix * C){
	unsigned int r, c;
	for(r=0; r<A->size1; r++){
		for(c=0; c<A->size2; c++){
			gsl_matrix_set(C, r, c, gsl_matrix_get(A, r, c) + gsl_matrix_get(B, r, c));
		}
	}
}

// vector addition
// C = A + B
void gsl_vector_add(const gsl_vector * A, const gsl_vector * B, gsl_vector * C){
	unsigned int r;
	for(r=0; r<A->size; r++){
		gsl_vector_set(C, r, gsl_vector_get(A, r) + gsl_vector_get(B, r));
	}
}

// C = A - B
void gsl_matrix_subtr(const gsl_matrix * A, const gsl_matrix * B, gsl_matrix * C){
	unsigned int r, c;
	for(r=0; r<A->size1; r++){
		for(c=0; c<A->size2; c++){
			gsl_matrix_set(C, r, c, gsl_matrix_get(A, r, c) - gsl_matrix_get(B, r, c));
		}
	}
}

// copies chain by value, return quantiles, .025, .5, .975
void post_process_MCMC_chain(vector<double> chain_in, double &median, double &quantile_025, double &quantile_975){
	vector<double>::iterator iter = chain_in.begin();
	vector<double> chain(iter, chain_in.end());

	sort(chain.begin(), chain.end());

	float mid_index = chain.size() / (float) 2;
	// if index is an integer
	if(mid_index == (int) mid_index){
		median = chain[(int) mid_index];
	}else{
		int index1 = (int) ceil(mid_index);
		int index2 = (int) floor(mid_index);

		median = .5 * (chain[index1] +  chain[index2]);
	}

	float q1 = chain.size() * .025;

	if(q1 == (int) q1){
		quantile_025 = chain[(int) q1];
	}else{
		int index1 = (int) ceil(q1);
		int index2 = (int) floor(q1);

		quantile_025 = .5 * (chain[index1] +  chain[index2]);
	}

	float q3 = chain.size() * .975;

	if(q3 == (int) q3){
		quantile_975 = chain[(int) q3];
	}else{
		int index1 = (int) ceil(q3);
		int index2 = (int) floor(q3);

		quantile_975 = .5 * (chain[index1] +  chain[index2]);
	}

}

double gsl_matrix_mean(const gsl_matrix *mat){
	double sum=0;
	unsigned int r, c;
	for(r=0; r<mat->size1; r++){
		for(c=0; c<mat->size2; c++){
			sum += gsl_matrix_get(mat, r, c);
		}
	}
	return(sum/float(mat->size1 * mat->size2));


}

double gsl_vector_mean(const gsl_vector *V){
	double sum=0;
	unsigned int i;
	for(i=0; i<V->size; i++){
		sum += gsl_vector_get(V, i);
	}
	return(sum/float(V->size));
}

// result = scalar*mat
void gsl_matrix_scalar_product(const gsl_matrix * mat, const double & scalar, gsl_matrix *result){
	unsigned int r, c;
	for(r=0; r<mat->size1; r++){
		for(c=0; c<mat->size2; c++){
			gsl_matrix_set(result, r, c, scalar*gsl_matrix_get(mat, r, c));
		}
	}
}

// result = t(mat)%*%mat
void gsl_matrix_crossprod(const gsl_matrix *mat, gsl_matrix *result){
	/*gsl_matrix *t_mat = gsl_matrix_alloc(mat->size2, mat->size1);
	gsl_matrix_transpose_memcpy(t_mat, mat);

	gsl_matrix_product(t_mat, mat, result);

	gsl_matrix_free(t_mat);*/

	 gsl_blas_dgemm(CblasTrans,CblasNoTrans, 1.0, mat, mat, 0.0, result);
}

// result = mat%*%t(mat)
void gsl_matrix_tcrossprod(const gsl_matrix *mat, gsl_matrix *result){
	/*gsl_matrix *t_mat = gsl_matrix_alloc(mat->size2, mat->size1);
	gsl_matrix_transpose_memcpy(t_mat, mat);

	gsl_matrix_product(mat, t_mat, result);

	gsl_matrix_free(t_mat);*/


	 gsl_blas_dgemm(CblasNoTrans,CblasTrans, 1.0, mat, mat, 0.0, result);
}

// result = cbind(rep(1, n), mat)
void gsl_matrix_augment_with_intercept(const gsl_matrix *mat, gsl_matrix *result){
	gsl_matrix *col_of_ones = gsl_matrix_alloc(mat->size1, 1);
	gsl_matrix_set_all(col_of_ones, 1);

	gsl_matrix_cbind(col_of_ones, mat, result);

	gsl_matrix_free(col_of_ones);
}

double gsl_matrix_trace(const gsl_matrix *mat){

	double sum = 0;
	for(unsigned int i=0; i<mat->size1;i++){
		sum += gsl_matrix_get(mat, i, i);
	}
	return(sum);
}

// return t(a) %*% B %*% a
double gsl_matrix_quadratic_form(const gsl_matrix *a, const gsl_matrix *B){
	gsl_matrix *t_a = gsl_matrix_alloc(a->size2, a->size1);
	gsl_matrix_transpose_memcpy(t_a, a);

	gsl_matrix *t_a_B = gsl_matrix_alloc(t_a->size1, t_a->size2);
	gsl_matrix_product(t_a, B, t_a_B);

	gsl_matrix *t_a_B_a = gsl_matrix_alloc(1, 1);
	gsl_matrix_product(t_a_B, a, t_a_B_a);

	double result = gsl_matrix_get(t_a_B_a, 0, 0);

	gsl_matrix_free(t_a);
	gsl_matrix_free(t_a_B);
	gsl_matrix_free(t_a_B_a);

	return(result);
}

double gsl_matrix_quadratic_form( const gsl_matrix *Q, const gsl_vector *y){

	gsl_vector *tmp = gsl_vector_alloc( y->size );

	// tmp = Q %*% y
	gsl_blas_dgemv(CblasNoTrans, 1.0, Q, y, 0.0, tmp);

	double result;
	gsl_blas_ddot( y, tmp, &result );

	return result;
}

double gsl_matrix_quadratic_form_sym( const gsl_matrix *Q, const gsl_vector *y){

	gsl_vector *tmp = gsl_vector_alloc( y->size );

	// tmp = Q %*% y
	//gsl_blas_dgemv(CblasNoTrans, 1.0, Q, y, 0.0, tmp);
	gsl_blas_dsymv (CblasLower, 1.0, Q, y, 0.0, tmp);

	double result;
	gsl_blas_ddot( y, tmp, &result );

	gsl_vector_free( tmp );

	return result;
}

void gsl_matrix_quadratic_form_sym( const gsl_matrix *Q, const gsl_matrix *X, gsl_matrix *S){


	// t(X) %*%  Q %*% X
	gsl_matrix *tmp = gsl_matrix_alloc( Q->size1, X->size2 );

	// tmp = Q %*% X
	gsl_blas_dsymm(CblasLeft, CblasLower, 1.0, Q, X, 0.0, tmp);

	// S = t(X) %*% tmp
	gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, X, tmp, 0.0, S);

	gsl_matrix_free( tmp );
}

// returns by reference inverse(mat) and ln(det(mat))
// efficient because only do LU decomposition once
void gsl_matrix_LU_solve_lndet(const gsl_matrix *mat, gsl_matrix * mat_solve, double &det){

	gsl_matrix *mat_LU = gsl_matrix_alloc(mat->size1, mat->size2);
	gsl_matrix_memcpy(mat_LU, mat);

	gsl_permutation *p = gsl_permutation_calloc(mat_LU->size2);
	int s=0;
	gsl_linalg_LU_decomp(mat_LU, p, &s);
	gsl_linalg_LU_invert(mat_LU, p, mat_solve);

	det = gsl_linalg_LU_lndet(mat_LU);

	gsl_permutation_free(p);
	gsl_matrix_free(mat_LU);
}

// when matrix is positive definate, a Cholesky decomposition is faster than LU decomp
void gsl_matrix_chol_solve_lndet(const gsl_matrix *mat, gsl_matrix * mat_solve, double &ln_det){

	gsl_matrix *mat_chol = gsl_matrix_alloc(mat->size1, mat->size2);
	gsl_matrix_memcpy(mat_chol, mat);

	gsl_linalg_cholesky_decomp(mat_chol);

	ln_det = 0;

	// ln_det = log( prod( diag( M ))) = sum( log( diag( M )))
	for(unsigned int i=0; i<mat_chol->size1; i++){
		ln_det += log(gsl_matrix_get(mat_chol, i, i));
	}

	// multiply by 2 because A Cholesky decomp was used
	ln_det *= 2;

	gsl_linalg_cholesky_invert(mat_chol);
	gsl_matrix_memcpy(mat_solve, mat_chol);

	gsl_matrix_free(mat_chol);
}

// when matrix is positive definite, a Cholesky decomposition is faster than LU decomp
void gsl_matrix_pseudo_inverse_lndet(const gsl_matrix *M, gsl_matrix * result, double &ln_det){

	gsl_vector *lambda = gsl_vector_alloc(M->size2);
	gsl_vector *work = gsl_vector_alloc(M->size2);
	gsl_matrix *U = gsl_matrix_alloc(M->size1, M->size2);
	gsl_matrix_memcpy(U, M);
	gsl_matrix *V = gsl_matrix_alloc(M->size2, M->size2);


	if(M->size1 == M->size2){
		gsl_eigen_decomp(M, lambda, U, true);

		gsl_matrix_LU_solve(U, V);

	}else{
		// consider gsl_linalg_SV_decomp_mod for nrow >> ncol
		gsl_linalg_SV_decomp(U, V, lambda, work);

	}

	ln_det = 1;

	// inverse nonzero values and set others to zero
	for(unsigned int i=0; i<lambda->size; i++){
		// if eigenvalue is effectively zero
		if(gsl_vector_get(lambda, i) < EFFECTIVELY_ZERO && gsl_vector_get(lambda, i) > -EFFECTIVELY_ZERO){
			gsl_vector_set(lambda, i, 0);
		}else{
			gsl_vector_set(lambda, i, 1/gsl_vector_get(lambda, i));

			ln_det *= log(gsl_vector_get(lambda, i)); //should this be a +=? and ln_det = 0 at the beginning?
		}
	}

	gsl_matrix *U_lambda = gsl_matrix_alloc(M->size1, M->size2);
	gsl_matrix_vect_mult(U, lambda, U_lambda);
	gsl_matrix_product(U_lambda, V, result);



	gsl_matrix_free(U);
	gsl_matrix_free(V);
	gsl_matrix_free(U_lambda);
	gsl_vector_free(lambda);
	gsl_vector_free(work);
}

/*void gsl_matrix_add_scalar_to_diags(const gsl_matrix *mat, const double & scalar, gsl_matrix *result){
	for(int j=0; j<mat->size1; j++){
		gsl_matrix_set(result, j, j, gsl_matrix_get(mat, j, j)+scalar);
	}
}*/

double gsl_vector_dot_prod(const gsl_vector *A, const gsl_vector *B){
	#ifdef PRINT_TO_CONSOLE
	assert(B->size == A->size);
	#endif

	gsl_vector *A_local = gsl_vector_alloc(A->size);
	gsl_vector_memcpy(A_local, A);

	gsl_vector_mul(A_local, B);

	double sum = 0;
	for(unsigned int i=0; i<A->size; i++){
		sum += gsl_vector_get(A_local, i);
	}

	gsl_vector_free(A_local);
	return(sum);
}

// print gsl_matrix to STDOUT
void gsl_vector_print(const gsl_vector *vec){

	#ifdef PRINT_TO_CONSOLE
	for(unsigned int row=0; row< vec->size; row++){
			cout << gsl_vector_get(vec, row) << endl;
	}
	cout << endl;
	#endif
}


// print gsl_vector_short
void gsl_vector_short_print(const gsl_vector_short *vec){

	#ifdef PRINT_TO_CONSOLE
	for(unsigned int row=0; row< vec->size; row++){
			cout << gsl_vector_short_get(vec, row) << " ";
	}
	cout << endl;
	#endif
}

// returns Mat[ ,-col]
void gsl_matrix_omit_col(const gsl_matrix *Mat, const int &col_omit, gsl_matrix *result){
	unsigned int row, col;
	for(col=0; col<(unsigned int) col_omit; col++){
		for(row=0; row<Mat->size1; row++){
			gsl_matrix_set(result, row, col, gsl_matrix_get(Mat, row, col));
		}
	}

	for(col=col_omit+1; col<Mat->size2; col++){
		for(row=0; row<Mat->size1; row++){
			gsl_matrix_set(result, row, col-1, gsl_matrix_get(Mat, row, col));
		}
	}
}


/*
 gsl_vector *eigen_values = gsl_vector_alloc(distance->size1);
       gsl_matrix *eigen_vectors = gsl_matrix_alloc(distance->size1, distance->size2);

        gsl_matrix_scalar_product
	gsl_eigen_decomp(distance, eigen_values, eigen_vectors);

	gsl_vector_print(eigen_values);
*/
void gsl_eigen_decomp(const gsl_matrix *A_in, gsl_vector *eigen_values, gsl_matrix *eigen_vectors, bool sort_eigen_vals){
	if(A_in->size1 != A_in->size2){
		throw("Eigen decomp of non-square matrix\n");
	}

	// make a new copy of the matrix because gsl_eigen_symmv destroyes the covariance matrix
	gsl_matrix *A = gsl_matrix_alloc(A_in->size1, A_in->size2);
	gsl_matrix_memcpy(A, A_in);

	// allocate workspace
	gsl_eigen_symmv_workspace *eigen_workspace = gsl_eigen_symmv_alloc(A->size1);

   // eigen decomp
   gsl_eigen_symmv(A, eigen_values, eigen_vectors, eigen_workspace);

   // free eigen_workspace
   gsl_eigen_symmv_free(eigen_workspace);

   // free A
   gsl_matrix_free(A);

   if( sort_eigen_vals){
	   // sort eigen vectors by eigen values
	   gsl_eigen_symmv_sort(eigen_values, eigen_vectors, GSL_EIGEN_SORT_VAL_DESC);
   }

}

// copies the a subset of the columns of A to sub_A
// col_indeces indicates which columns should be copied
void gsl_matrix_sub_row(const gsl_matrix *A, const vector<int> &col_indeces, gsl_matrix *sub_A){
	if(A->size2 != sub_A->size2){
		throw("gsl_matrix_sub_col: matrices of incompatible sizes\n");
	}

	// initialize column vector
	gsl_vector * row_vect = gsl_vector_alloc(A->size2);

	// for each desired index
	for(unsigned int i=0; i<col_indeces.size(); i++){
		// get col from A
		gsl_matrix_get_row(row_vect, A, col_indeces[i]);

		// set col in sub_A
		gsl_matrix_set_row(sub_A, i, row_vect);
	}
	gsl_vector_free(row_vect);
}

// copies the a subset of the columns of A to sub_A
// col_indeces indicates which columns should be copied
void gsl_matrix_sub_col(const gsl_matrix *A, const vector<int> &col_indeces, gsl_matrix *sub_A){
	if(A->size1 != sub_A->size1){
		perror("gsl_matrix_sub_col: matrices of incompatible sizes\n");
	}

	// initialize column vector
	gsl_vector * col_vect = gsl_vector_alloc(A->size1);

	// for each desired index
	for(unsigned int i=0; i<col_indeces.size(); i++){
		// get col from A
		gsl_matrix_get_col(col_vect, A, col_indeces[i]);

		// set col in sub_A
		gsl_matrix_set_col(sub_A, i, col_vect);
	}
	gsl_vector_free(col_vect);
}

void gsl_matrix_sub_col(const gsl_matrix *A, const int max_col, gsl_matrix *sub_A){

	// initialize column vector
	gsl_vector * col_vect = gsl_vector_alloc(A->size1);

	// for each desired index
	for(unsigned int i=0; i<max_col; i++){
		// get col from A
		gsl_matrix_get_col(col_vect, A, i);

		// set col in sub_A
		gsl_matrix_set_col(sub_A, i, col_vect);
	}
	gsl_vector_free(col_vect);
}



// copies the a subset of the columns of C to sub_C
// col_indeces indicates which columns and rows should be copied
void gsl_matrix_sub_row_col(const gsl_matrix *C, const vector<int> &col_indeces, gsl_matrix *sub_C){

	// initialize
	gsl_matrix *intermediate_C = gsl_matrix_alloc(col_indeces.size(), C->size2);

	// get sub rows
	gsl_matrix_sub_row(C, col_indeces, intermediate_C);

	//cout << "C: " << C->size1 << " x " << C->size2 << endl;
	//cout << "intermediate_C: " << intermediate_C->size1 << " x " << intermediate_C->size2 << endl;
	//cout << "sub_C: " << sub_C->size1 << " x " << sub_C->size2 << endl;

	// get sub cols
	gsl_matrix_sub_col(intermediate_C, col_indeces, sub_C);

	gsl_matrix_free(intermediate_C);

}

void gsl_vector_subset(const gsl_vector *v, const vector<int> &indeces, gsl_vector *v_sub){

	assert( v_sub->size == indeces.size() );

	for(unsigned int i=0; i<v_sub->size; i++){
		gsl_vector_set( v_sub, i, gsl_vector_get( v, indeces[i] ) );
	}
}

// return trace(A * B)
// Trace only uses the diagonals, so dont actually compute A*B
double gsl_matrix_product_trace(const gsl_matrix *A, const gsl_matrix *B){

	assert( A->size2 == B->size1);

	// Simple way using loop directly
	//////////////////////////////////

	// (AB)[i,j] = = sum_k A[i,k]*B[k,j]
	/*double trace = 0;

	for(unsigned int i=0; i<A->size1; i++){
		for(unsigned int k=0; k<A->size2; k++){
			trace += gsl_matrix_get(A, i, k) * gsl_matrix_get(B, k, i);
		}
	}*/

	// Use cblas to evaluate each diagonal element
	//////////////////////////////////////////////

	double trace = 0;
	for(unsigned int k=0; k<A->size1; k++){
		// cblas_ddot (const int N, const double * x, const int incx, const double * y, const int incy)
		trace += cblas_ddot( A->size2, A->data + k*A->tda, 1, B->data + k, B->tda );
	}

	return(trace);
}

// return diag(A * B)
// Diag only uses the diagonals, so don't actually compute A*B
gsl_vector *gsl_matrix_product_diag(const gsl_matrix *A, const gsl_matrix *B){

	assert( A->size2 == B->size1);

	// Simple way using loop directly
	//////////////////////////////////

	// (AB)[i,j] = = sum_k A[i,k]*B[k,j]

	/*gsl_vector *v = gsl_vector_alloc( A->size1 );

	double sum;

	for(unsigned int i=0; i<A->size1; i++){
		sum = 0;
		for(unsigned int k=0; k<A->size2; k++){
			sum += gsl_matrix_get(A, i, k) * gsl_matrix_get(B, k, i);
		}
		gsl_vector_set( v, i, sum);
	}*/


	// Use cblas to evaluate each diagonal element
	//////////////////////////////////////////////

	gsl_vector *v2 = gsl_vector_alloc( A->size1 );

	for(unsigned int k=0; k<A->size1; k++){
		// cblas_ddot (const int N, const double * x, const int incx, const double * y, const int incy)
		gsl_vector_set( v2, k, cblas_ddot( A->size2, A->data + k*A->tda, 1, B->data + k, B->tda ) );
	}

	/*for(unsigned int k=0; k<A->size1; k++){
		cout << gsl_vector_get( v2, k ) << " " << gsl_vector_get( v, k ) << endl;
	}*/

	return(v2);
}


// Divide each column in M by the corresponding entry in v
// element-wise division so that first column of M is divided by first element in v
void gsl_matrix_vect_div_col(const gsl_matrix *M, const gsl_vector *v, gsl_matrix *result){

	double numerator;
	for(unsigned int c=0; c<M->size2; c++){
		numerator = gsl_vector_get(v, c);
		for(unsigned int r=0; r<M->size1; r++){
			gsl_matrix_set(result, r, c, gsl_matrix_get(M, r, c) / numerator);
		}
	}

}

// Divide each row in M by the corresponding entry in v
// element-wise division so that first column of M is divided by first element in v
void gsl_matrix_vect_div_row(const gsl_matrix *M, const gsl_vector *v, gsl_matrix *result){

	double numerator;

	for(unsigned int r=0; r<M->size1; r++){
		numerator = gsl_vector_get(v, r);
		for(unsigned int c=0; c<M->size2; c++){
			gsl_matrix_set(result, r, c, gsl_matrix_get(M, r, c) / numerator);
		}
	}

}

// Multiply each column in M by the corresponding entry in v
void gsl_matrix_vect_mult(const gsl_matrix *M, const gsl_vector *v, gsl_matrix *result){

	double scalar;
	for(unsigned int c=0; c<M->size2; c++){
		scalar = gsl_vector_get(v, c);
		for(unsigned int r=0; r<M->size1; r++){
			gsl_matrix_set(result, r, c, scalar*gsl_matrix_get(M, r, c));
		}
	}

}



// element-wise reciprocal
void gsl_matrix_recip(const gsl_matrix *M, gsl_matrix *result){
	for(unsigned int i=0; i<M->size1; i++){
		for(unsigned int j=0; j<M->size2; j++){
			gsl_matrix_set(result, i, j, 1/gsl_matrix_get(M, i, j));
		}
	}
}


// sum of all elements in the vector
double gsl_vector_sum_elements(const gsl_vector * vect){
	double sum = 0;
	for(unsigned int i=0; i<vect->size; i++){
		sum += gsl_vector_get(vect, i);
	}
	return(sum);
}

// sum of all elements in the vector which are finite and not NaN
double gsl_vector_sum_elements_finite(const gsl_vector * vect){
	double sum = 0;

	double v;

	for(unsigned int i=0; i<vect->size; i++){

		v = gsl_vector_get(vect, i);

		if( ! isnan(v) && isfinite(v) )
			sum += v;
	}
	return(sum);
}


// set X_signif equal to a subset of columns of X defined by signif_markers
void gsl_matrix_cbind_from_list(const gsl_matrix *X, const vector<int> &marker_list, gsl_matrix *X_subset){

	gsl_vector *column_vect = gsl_vector_alloc(X->size1);

	for(unsigned int i=0; i<marker_list.size(); i++){
		gsl_matrix_get_col(column_vect, X, marker_list[i]);

		gsl_matrix_set_col(X_subset, i, column_vect);
	}

	gsl_vector_free(column_vect);
}

void gsl_matrix_add_diag_scalar(gsl_matrix *M, const double scalar){
	
	for(unsigned int i=0; i<M->size1; i++){
		gsl_matrix_set(M, i, i, gsl_matrix_get(M, i, i) + scalar);
	}	
}
// add scalar to the diagonals of M if M is square
void gsl_matrix_add_diag_scalar(const gsl_matrix *M, const double scalar, gsl_matrix *result){

	gsl_matrix_memcpy(result, M);

	for(unsigned int i=0; i<M->size1; i++){
		gsl_matrix_set(result, i, i, gsl_matrix_get(result, i, i) + scalar);
	}
}

void gsl_matrix_add_diag(gsl_matrix *M, const gsl_vector *v){

	for(unsigned int i=0; i<M->size1; i++){
		gsl_matrix_set(M, i, i, gsl_matrix_get(M, i, i) + gsl_vector_get(v, i));
	}
}

void gsl_matrix_set_diag(gsl_matrix *M, const gsl_vector *v){

	for(unsigned int i=0; i<M->size1; i++){
		gsl_matrix_set(M, i, i, gsl_vector_get(v, i));
	}
}

void gsl_matrix_set_diag(gsl_matrix *M, const double value){

	for(unsigned int i=0; i<M->size1; i++){
		gsl_matrix_set(M, i, i, value);
	}
}

// set diagonals of result to scalar, where result is already diagonal
void gsl_matrix_identity_scalar(const double scalar, gsl_matrix *M){
	for(unsigned int i=0; i<M->size1; i++){
		gsl_matrix_set(M, i, i, scalar);
	}
}

gsl_vector *gsl_matrix_get_diag(gsl_matrix *M){

	if(M->size1 != M->size2){
		#ifdef PRINT_TO_CONSOLE
		cout << "Cannot extract diag of non square matrix" << endl;
		exit(1);
		#endif
	}

	gsl_vector *v = gsl_vector_alloc(M->size1);

	for(unsigned int i=0; i<M->size1; i++){
		gsl_vector_set(v, i, gsl_matrix_get(M, i, i));
	}

	return v;
}

// result = scalar * v
void gsl_vector_scalar_product(const gsl_vector* v, const double & scalar, gsl_vector *result){
	for(unsigned int r=0; r<v->size; r++){
			gsl_vector_set(result, r, scalar*gsl_vector_get(v, r));
	}
}

// C = A - B
void gsl_vector_subtr(const gsl_vector * A, const gsl_vector * B, gsl_vector * C){
	for(unsigned int r=0; r<A->size; r++){
		gsl_vector_set(C, r, gsl_vector_get(A, r) - gsl_vector_get(B, r));
	}
}


// Pseudo inverse by eigen decomposition for square matrices and SVD or all others
void gsl_matrix_pseudo_inverse(const gsl_matrix * M, gsl_matrix *result, const double min_e_value){

	gsl_vector *lambda = gsl_vector_alloc(M->size2);
	gsl_vector *work = gsl_vector_alloc(M->size2);
	gsl_matrix *U = gsl_matrix_alloc(M->size1, M->size2);
	gsl_matrix_memcpy(U, M);
	gsl_matrix *V = gsl_matrix_alloc(M->size2, M->size2);


	if(M->size1 == M->size2){
		gsl_eigen_decomp(M, lambda, U, true);

		//gsl_matrix_LU_solve(U, V);
		gsl_matrix_transpose_memcpy(V, U);

	}else{
		// consider gsl_linalg_SV_decomp_mod for nrow >> ncol
		gsl_linalg_SV_decomp(U, V, lambda, work);

	}

	// invert nonzero values and set others to zero
	for(unsigned int i=0; i<lambda->size; i++){
		// if eigenvalue is effectively zero
		if(fabs(gsl_vector_get(lambda, i)) < min_e_value){
			gsl_vector_set(lambda, i, 0);
		}else{
			gsl_vector_set(lambda, i, 1/gsl_vector_get(lambda, i));
		}
	}

	gsl_matrix *U_lambda = gsl_matrix_alloc(M->size1, M->size2);
	gsl_matrix_vect_mult(U, lambda, U_lambda);
	gsl_matrix_product(U_lambda, V, result);

	gsl_matrix_free(U);
	gsl_matrix_free(V);
	gsl_matrix_free(U_lambda);
	gsl_vector_free(lambda);
	gsl_vector_free(work);
}


// Pseudo inverse by eigen decomposition for square matrices and SVD or all others
void gsl_matrix_pseudo_inverse_log_det(const gsl_matrix * M, gsl_matrix *result, double &log_det){

	gsl_vector *lambda = gsl_vector_alloc(M->size2);
	gsl_vector *work = gsl_vector_alloc(M->size2);
	gsl_matrix *U = gsl_matrix_alloc(M->size1, M->size2);
	gsl_matrix_memcpy(U, M);
	gsl_matrix *V = gsl_matrix_alloc(M->size2, M->size2);


	if(M->size1 == M->size2){
		gsl_eigen_decomp(M, lambda, U, true);

		gsl_matrix_LU_solve(U, V);

	}else{
		// consider gsl_linalg_SV_decomp_mod for nrow >> ncol
		gsl_linalg_SV_decomp(U, V, lambda, work);

	}

	log_det = 0;

	// inverse nonzero values and set others to zero
	for(unsigned int i=0; i<lambda->size; i++){
		// if eigenvalue is effectively zero
		if(gsl_vector_get(lambda, i) < EFFECTIVELY_ZERO && gsl_vector_get(lambda, i) > -EFFECTIVELY_ZERO){
			gsl_vector_set(lambda, i, 0);
		}else{
			gsl_vector_set(lambda, i, 1/gsl_vector_get(lambda, i));

			log_det += log(gsl_vector_get(lambda, i));
		}
	}

	gsl_matrix *U_lambda = gsl_matrix_alloc(M->size1, M->size2);
	gsl_matrix_vect_mult(U, lambda, U_lambda);
	gsl_matrix_product(U_lambda, V, result);

	gsl_matrix_free(U);
	gsl_matrix_free(V);
	gsl_matrix_free(U_lambda);
	gsl_vector_free(lambda);
	gsl_vector_free(work);
}



// Pseudo_inverse of K where K = Q %*% t(Q), where only Q is needed
// Calculating svd(Q) and creating pseudo_inverse(K) is much faster than u eigen(K)
// If make_Q_full_rank == true, return a full rank Q using the previously computed SVD

// Justification
// If Q = UDV^T then QQ^T = (UDV^T)(VDU^T)
// 		since V is an orthogonal matirx V^TV = I
//		so QQ^T = (UD)(DU^T)
// Squaring and inverting D,
// 		pseudo_inverse(QQ^T) = U(D^-2)U^T


gsl_matrix * gsl_matrix_pseudo_inverse_factor(gsl_matrix *& Q, gsl_matrix *result, bool make_Q_full_rank){

	gsl_vector *lambda = gsl_vector_alloc(Q->size2);
	gsl_vector *lambda_inverse = gsl_vector_alloc(Q->size2);
	gsl_vector *work = gsl_vector_alloc(Q->size2);
	gsl_matrix *U = gsl_matrix_alloc(Q->size1, Q->size2);
	gsl_matrix *t_U = gsl_matrix_alloc(Q->size2, Q->size1);
	gsl_matrix *V = gsl_matrix_alloc(Q->size2, Q->size2);
	gsl_matrix_memcpy(U, Q);

	// consider gsl_linalg_SV_decomp_mod for nrow >> ncol
	gsl_linalg_SV_decomp(U, V, lambda, work);

	// count the number of nonzero singular values in lambda
	unsigned int n_nonzero_singular_values = lambda->size;
	for(unsigned int i=0; i<lambda->size; i++){
		if(gsl_vector_get(lambda, i) < EFFECTIVELY_ZERO ){
			n_nonzero_singular_values = i;
			break;
		}
	}

	#ifdef PRINT_TO_CONSOLE
	cout << "n_nonzero_singular_values: " << n_nonzero_singular_values << endl;
	#endif

	for(unsigned int i=0; i<n_nonzero_singular_values; i++){
		gsl_vector_set(lambda_inverse, i, 1/pow(gsl_vector_get(lambda, i),2) );
		//cout << i << " "  << gsl_vector_get(lambda, i) << endl;
	}
	//exit(1);

	gsl_matrix *U_lambda_inverse = gsl_matrix_alloc(Q->size1, Q->size2);
	gsl_matrix_vect_mult(U, lambda_inverse, U_lambda_inverse);
	gsl_matrix_transpose_memcpy(t_U, U);

	// U(D^-2)U^T
	gsl_matrix_product(U_lambda_inverse, t_U, result);

	///////////////////////////////////////////
	// made Q full rank if it is not already //
	///////////////////////////////////////////

	gsl_matrix *Q_full_rank;

	if(make_Q_full_rank && n_nonzero_singular_values < Q->size2){

		// if Q is not full rank, get its basis from Q
		if(n_nonzero_singular_values < lambda->size){
			Q_full_rank = gsl_matrix_alloc(Q->size1, n_nonzero_singular_values);

			// copy columns from 0 to n_nonzero_singular_values
			for(unsigned int i=0; i<Q->size1; i++){
				for(unsigned int j=0; j<n_nonzero_singular_values; j++){
						gsl_matrix_set(Q_full_rank, i, j, gsl_matrix_get(U, i, j) * gsl_vector_get(lambda, j) );
				}
			}
		}
	}else{
		Q_full_rank = gsl_matrix_alloc(Q->size1, Q->size2);
		gsl_matrix_memcpy(Q_full_rank, Q);
	}

	gsl_matrix_free(U);
	gsl_matrix_free(t_U);
	gsl_matrix_free(V);
	gsl_matrix_free(U_lambda_inverse);
	gsl_vector_free(lambda);
	gsl_vector_free(lambda_inverse);
	gsl_vector_free(work);

	return(Q_full_rank);
}



// Input:
// a, b: gsl_vectors of the same dimension
// direction: which component should be returned (PARALLEL or ORTHOGONAL)
// 	defaults to ORTHOGONAL

// Output:
// Get the component of a that is orthogonal to b.
void gsl_matrix_project_onto(const gsl_vector *a, const gsl_vector *b, gsl_vector *result, int direction){

	/* get component of a that is orthogonal to b
	result = b * sum(a*b)/sum(b*b)
	result = a-result
	sum(b*result)
	 */

	double a_dot_b, b_dot_b;

	gsl_blas_ddot(a, b, &a_dot_b);
	gsl_blas_ddot(b, b, &b_dot_b);

	if(direction == ORTHOGONAL){
		gsl_vector_memcpy( result, a );
		gsl_blas_daxpy( -a_dot_b/b_dot_b, b, result);
	}else{
		gsl_vector_scalar_product(b, a_dot_b/b_dot_b, result);
	}
}




// Input:
//	A: a matrix of any dimension
//	X: a 1 col matrix, with as many rows as A
//	result: the location where the resulting matrix is stored.  Must be the same dimensions as A

// Output:
//	Returns the components of each column of A that is orthogonal to X
void gsl_matrix_orthogonal_components(const gsl_matrix *A, const gsl_vector *X, gsl_matrix *result){

	gsl_vector *A_col = gsl_vector_alloc(A->size1);

	gsl_vector *A_proj = gsl_vector_alloc(A->size1);

	// for each col in A
	for(unsigned int i=0; i<A->size2; i++){
		gsl_matrix_get_col(A_col, A, i);

		gsl_matrix_project_onto(A_col, X, A_proj, ORTHOGONAL);

		gsl_matrix_set_col(result, i, A_proj);
	}

	gsl_vector_free(A_col);
	gsl_vector_free(A_proj);
}



void gsl_matrix_mean_center(const gsl_matrix *M, gsl_matrix *result){

	double mu = gsl_matrix_mean(M);

	for(unsigned int i=0; i<M->size1; i++){
		for(unsigned int j=0; j<M->size2; j++){
			gsl_matrix_set(result, i, j, gsl_matrix_get(M, i, j) - mu);
		}
	}
}

void gsl_vector_mean_center(const gsl_vector *v, gsl_vector *result){

	double mu = gsl_vector_mean(v);

	for(unsigned int i=0; i<v->size; i++){
		gsl_vector_set(result, i, gsl_vector_get(v, i) - mu);
	}
}


// Input: M as symmetric positive definite matrix

// Output:
// 		returns L such that L %*% t(L) = M
void gsl_linalg_cholesky_upper(const gsl_matrix *M, gsl_matrix *L){

	// copy L <- M
	gsl_matrix_memcpy(L, M);

	// get the Cholesky decomposition of M
	gsl_linalg_cholesky_decomp(L);

	// now L contains both L and t(L) and it is not actually lower diagonal
	//so we want to set upper triangle to zeros so desired L is left

	for(unsigned int i=0; i<L->size1; i++){
		for(unsigned int j=0; j<i; j++){
			gsl_matrix_set(L, i, j, 0);
		}
	}
}

// return column i of gsl_matrix M
// same as gsl_matrix_get_col except it returns a gsl_matrix instead of gsl_vector
void gsl_matrix_get_column(const gsl_matrix *M, const int col, gsl_matrix *result){

	// for each row
	for(unsigned int row=0; row<M->size1; row++){
		gsl_matrix_set(result, row, 0, gsl_matrix_get(M, row, col));

	}
}

// Use SVD to return linearly independent columns of M
// return a gsl_matrix with n_cols = rank of M
// M = U %*% diag(lambda) %*% t(V)
gsl_matrix * gsl_matrix_SVD_independent_columns(const gsl_matrix *M){

	gsl_vector *lambda = gsl_vector_alloc(M->size2);
	gsl_vector *work = gsl_vector_alloc(M->size2);
	gsl_matrix *U = gsl_matrix_alloc(M->size1, M->size2);
	gsl_matrix_memcpy(U, M);
	gsl_matrix *V = gsl_matrix_alloc(M->size2, M->size2);

	// consider gsl_linalg_SV_decomp_mod for nrow >> ncol
	gsl_linalg_SV_decomp(U, V, lambda, work);

	int rank = 0;
	// inverse nonzero values and set others to zero
	for(unsigned int i=0; i<lambda->size; i++){
		// if eigenvalue > 0, add 1 to the rank of M
		if(fabs(gsl_vector_get(lambda, i)) > EFFECTIVELY_ZERO){
			rank++;
		}
	}

	gsl_vector_free(lambda);
	gsl_vector_free(work);
	gsl_matrix_free(V);

	gsl_matrix *U_reduced = gsl_matrix_alloc(M->size1, rank);

	for(unsigned int i=0; i<M->size1; i++){
		for(int j=0; j<rank; j++){

			gsl_matrix_set(U_reduced, i, j, gsl_matrix_get(U, i, j));
		}
	}
	gsl_matrix_free(U);

	return(U_reduced);
}

void gsl_vector_elementwise_product(const gsl_vector *A, const gsl_vector *B, gsl_vector *C){

	for(unsigned int i=0; i<A->size; i++){
		gsl_vector_set(C, i, gsl_vector_get(A, i) * gsl_vector_get(B, i));
	}
}

// Input:
//	Q is a tall matrix with a decaying spectrum of singular values

// Output:
//	return a matrix that is a low rank approximation of Q and captures x% of the variance determined by percent_variance_captured
gsl_matrix * gsl_low_rank_approximation(const gsl_matrix *Q, double percent_variance_captured, int min_rank){

	gsl_vector *lambda = gsl_vector_alloc(Q->size1);
	gsl_vector *work = gsl_vector_alloc(Q->size1);
	gsl_matrix *U = gsl_matrix_alloc(Q->size2, Q->size1);
	gsl_matrix *V = gsl_matrix_alloc(Q->size1, Q->size1);
	gsl_matrix_transpose_memcpy(U, Q);


	// consider gsl_linalg_SV_decomp_mod for nrow >> ncol
	gsl_linalg_SV_decomp(U, V, lambda, work);

	//gsl_matrix *work_more = gsl_matrix_alloc(Q->size1, Q->size1);
	//gsl_linalg_SV_decomp_mod(U, work_more, V, lambda, work);

	double total_variance = 0;
	unsigned int desired_rank = lambda->size;
	gsl_vector *cumulative_loadings = gsl_vector_calloc(Q->size1+1);

	// total variance = sum of sing_values^2
	for(unsigned int i=0; i<lambda->size; i++){
		total_variance += pow(gsl_vector_get(lambda, i),2);
	}

	// cumulative_loadings is the sum of variance from index 0 to i
	for(unsigned int i=0; i<lambda->size; i++){
		gsl_vector_set(cumulative_loadings, i+1, pow(gsl_vector_get(lambda, i),2)/total_variance + gsl_vector_get(cumulative_loadings, i)   );

		if(gsl_vector_get(cumulative_loadings, i+1) >= percent_variance_captured || gsl_vector_get(lambda, i) <= EFFECTIVELY_ZERO){
			desired_rank = i;
			break;
		}
	}

	desired_rank = max( (unsigned int) min_rank, desired_rank);

	//gsl_vector_print(cumulative_loadings);
	gsl_matrix *result = gsl_matrix_alloc(Q->size1, desired_rank);

	// low rank matrix has desired_rank corresponding to percent_variance_captured

	// for each column
	for(unsigned int i=0; i<desired_rank; i++){
		// for each row
		for(unsigned int j=0; j<Q->size1; j++){
			gsl_matrix_set(result, j, i, gsl_matrix_get(V, j, i) * gsl_vector_get(lambda, i));
		}
	}

	gsl_matrix_free(U);
	gsl_matrix_free(V);
	gsl_vector_free(lambda);
	gsl_vector_free(work);
	//gsl_matrix_free(work_more);

	return(result);
}

void gsl_vector_add_scalar(const gsl_vector *V, const double &scalar, gsl_vector *result){

	for(unsigned int r=0; r<V->size; r++){
		gsl_vector_set(result, r, gsl_vector_get(V, r) + scalar);
	}
}

double gsl_ran_exponential_pdf_ln(const gsl_vector *x, const double &rate){
	double value = 0;

	for(unsigned int i=0; i<x->size; i++){
		value += gsl_vector_get(x, i);
	}

	value *= -rate;

	value += log(rate);

	return(value);
}


double gsl_ran_scaled_inverse_chisquare_pdf_ln(const gsl_vector *x, const double &v, const double &s_sq){

	double value = -gsl_sf_lngamma(v/2) + v/2 *log(v*s_sq/2);

	for(unsigned int i=0; i<x->size; i++){
		value += -(1+v/2) *log(gsl_vector_get(x,i)) - v*s_sq/(2*gsl_vector_get(x,i));
	}
	return(value);
}

double gsl_exponential_gamma_ln(const gsl_vector *x, const double &lambda, const double &gamma){
	double value = 0;

	for(unsigned int i=0; i<x->size; i++){
		value += log(1 + gsl_vector_get(x,i) / pow(gamma,2));
	}

	value *= - (lambda + 1);

	value += log(lambda) - 2*log(gamma);

	return(value);
}

void gsl_permutation_print(const gsl_permutation *perm){

	#ifdef PRINT_TO_CONSOLE
	for(unsigned int i=0; i<perm->size; i++){
		cout << gsl_permutation_get(perm, i) << " ";
	}
	#endif
}

void gsl_permutation_save(const gsl_permutation *perm, const string &path){

	ofstream file( path.c_str() );

	for(unsigned int i=0; i<perm->size; i++){
		file << gsl_permutation_get(perm, i) << " ";
	}
	file.close();
}


gsl_matrix_short *convert_to_gsl_matrix_short(const gsl_matrix *X){

	gsl_matrix_short *X_short = gsl_matrix_short_alloc(X->size1, X->size2);

	for(unsigned int i=0; i<X->size1; i++){
		for(unsigned int j=0; j<X->size2; j++){
			gsl_matrix_short_set(X_short, i, j, (short) gsl_matrix_get(X, i, j));
		}
	}

	return X_short;
}

gsl_vector *gsl_vector_read_unknown_size(const string &file){

	int entry_count = 0;
	string entry;

	/////////////////////////////////
	// Count the number of entries //
	/////////////////////////////////
	ifstream file_stream(file.c_str());

	while(file_stream >> entry){
		entry_count++;
	}

	file_stream.close();

	/////////////////////////
	// Read and store data //
	/////////////////////////

	gsl_vector *v = gsl_vector_alloc(entry_count);

	file_stream.open(file.c_str());

	entry_count = 0;

	while(file_stream >> entry){
		gsl_vector_set(v, entry_count, atof(entry.c_str()));
		entry_count++;
	}

	file_stream.close();

	return v;
}


void gsl_vector_save(const gsl_vector *v, const string &path){

	ofstream file( path.c_str() );

	for(unsigned int i=0; i<v->size; i++){
		file << gsl_vector_get(v, i) << endl;
	}
	file.close();
}


void gsl_vector_save(const vector<string> &names, const gsl_vector *v, const string &path){

	if( names.size() != v->size ){
		#ifdef PRINT_TO_CONSOLE
		cerr << "Cannot save gsl_vector to file, names vector is not the same size..." << endl;
		cerr << "v->size: " << v->size << endl;
		cerr << "names.size(): " << names.size() << endl;
		exit(1);
		#endif
	}

	ofstream file( path.c_str() );

	for(unsigned int i=0; i<v->size; i++){
		file << names[i] << " " << gsl_vector_get(v, i) << endl;
	}
	file.close();
}

void gsl_matrix_triangular_to_full(gsl_matrix *M, const char uplo){

	if( uplo == 'U'){
		for(unsigned int i=0; i<M->size1; i++){
			for(unsigned int j=i+1; j<M->size2; j++){
				gsl_matrix_set(M, i, j, gsl_matrix_get(M, j, i) );
			}
		}
	}else{
		for(unsigned int i=0; i<M->size2; i++){
			for(unsigned int j=i+1; j<M->size1; j++){
				gsl_matrix_set(M, j, i, gsl_matrix_get(M, i, j) );
			}
		}
	}
}


gsl_vector *gsl_vector_elementwise_product(const gsl_vector *a, const gsl_vector *b){

	if( a->size != b->size ){
		#ifdef PRINT_TO_CONSOLE
		cout << "gsl_vector_elementwise_product: vectors have different sizes" << endl;
		exit(1);
		#endif
	}

	gsl_vector *result = gsl_vector_alloc( a->size );

	for(unsigned int i=0; i<a->size; i++){
		gsl_vector_set(result, i, gsl_vector_get(a, i) * gsl_vector_get(b, i) );
	}

	return result;
}



void gsl_matrix_short_save_TPED_recode12(const gsl_matrix_short *X, const string &path){

	unsigned int n_indivs = X->size1;
	unsigned int n_markers = X->size2;

	ofstream file( path.c_str() );

	// for each marker
	for(unsigned int j=0; j<n_markers; j++){

		// print chrom, name, genetic pos, physical pos
		file << 1 << " " << j << " " << 0 << " " << 0 << " ";

		for(unsigned int i=0; i<n_indivs; i++){
			file << gsl_matrix_short_get( X, i, j ) << " ";
		}
		file << endl;
	}

	file.close();
}

void gsl_matrix_short_save_TPED(const gsl_matrix_short *X, const string &path){

	unsigned int n_indivs = X->size1;
	unsigned int n_markers = X->size2;

	ofstream file( path.c_str() );

	// for each marker
	for(unsigned int j=0; j<n_markers; j++){

		// print chrom, name, genetic pos, physical pos
		file << 1 << " " << j << " " << 0 << " " << 0 << " ";

		for(unsigned int i=0; i<n_indivs; i++){

			switch( gsl_matrix_short_get( X, i, j ) ){
				case 0:
					file << "A A\t";
					break;
				case 1:
					file << "A C\t";
					break;
				case 2:
					file << "C C\t";
					break;
				case '?':
					#ifdef PRINT_TO_CONSOLE
					cout << "gsl_matrix_short_save_TPED: Invalid genotype value for individual " << i << ": " << gsl_matrix_short_get( X, i, j ) << endl;
					exit(1);
					#endif
					break;
			}

		}
		file << endl;
	}

	file.close();
}


gsl_matrix * gsl_matrix_attach_array(double *array, const int size1, const int size2){

	gsl_block *block = (gsl_block *) malloc(sizeof(gsl_block));
	block->data = array;
	block->size = size1*size2;

	gsl_matrix *M = (gsl_matrix *) malloc(sizeof(gsl_matrix));
	M->size1 = size1;
	M->size2 = size2;
	M->tda = size2;
	M->data = array;
	M->owner = 0;
	M->block = block;

	return M;
}


gsl_vector * gsl_vector_attach_array(double *array, const int size){

	gsl_block *block = (gsl_block *) malloc(sizeof(gsl_block));
	block->data = array;
	block->size = size;

	gsl_vector *V = (gsl_vector *) malloc(sizeof(gsl_vector));
	V->size = size;
	V->stride = 1;
	V->data = array;
	V->owner = 0;
	V->block = block;

	return V;
}

gsl_vector *gsl_vector_seq( double start, double stop, int n){

	gsl_vector *grid = gsl_vector_alloc( n );

	double value = start;

	for( int i=0; i<n-1; i++){

		gsl_vector_set( grid, i, value );

		value += (stop - start) / (double) (n-1);
	}
	gsl_vector_set( grid, n-1, stop );

	return( grid );
}

void gsl_matrix_diagonal_multiply( const gsl_matrix *X, const gsl_vector *w, gsl_matrix *&W_X, const bool byRow ){

	/*if( w->size != X->size1 ){
		cout << "gsl_matrix_diagonal_multiply: w->size = " << w->size << "and X->size1 = " << X->size1 << "must be equal" << endl;
		exit(1);
	}*/
	/* Evaluate W_X = W %*% X
	 * Use cblas directly to scale X[k,] by w[k]
	 * This is MUCH faster that evaluating W %*% X, and saves time in calloc'ing a huge W matrix
	 *
	 * Access row i and col j of a gsl_matrix directly: m->data[i * m->tda + j]
	 * So access beginning of row k by: m->data + k*m->tda
	 * 		since accessing matrix elements is done by pointer addition
	 */

	// malloc a new matrix if return pointer is NULL
	if( W_X == NULL ){
		W_X = gsl_matrix_alloc( X->size1, X->size2 );
	}

	gsl_matrix_memcpy(W_X, X);

	if( byRow ){
		// Multiply X[i,] *w[i]
		for(unsigned int k=0; k<w->size; k++){
			cblas_dscal( X->size2, gsl_vector_get(w, k), W_X->data + k*W_X->tda, 1 );
		}
	}else{
		// Multiply X[,i] *w[i]
		for(unsigned int k=0; k<w->size; k++){
			cblas_dscal( X->size1, gsl_vector_get(w, k), W_X->data + k, X->size2 );
		}
	}
}

/*void gsl_matrix_diagonal_multiply( gsl_matrix *X, const gsl_vector *w){

	for(unsigned int k=0; k<w->size; k++){
		cblas_dscal( X->size2, gsl_vector_get(w, k), X->data + k*X->tda, 1 );
	}
}*/


void gsl_matrix_diagonal_quadratic_form( const gsl_matrix *X, const gsl_vector *w, gsl_matrix *&M , gsl_matrix * W_X){

	bool freeBeforeReturn = false;
	if( W_X == NULL ) freeBeforeReturn = true;

	/*if( w->size != X->size1 ){
		cout << "gsl_matrix_diagonal_quadratic_form: w->size = " << w->size << "and X->size1 = " << X->size1 << "must be equal" << endl;
		exit(1);
	}*/

	// W_X = W %*% X
	gsl_matrix_diagonal_multiply( X, w, W_X );

	// malloc a new matrix if return pointer is NULL
	if( M == NULL ){
		M = gsl_matrix_alloc( X->size2, X->size2 );
	}

	// M = t(X) %*% W %*% X
	gsl_blas_dgemm(CblasTrans,CblasNoTrans, 1.0, X, W_X, 0.0, M);

	if( freeBeforeReturn ){
		gsl_matrix_free( W_X );
	}
}

gsl_vector *gsl_vector_subsample( const gsl_vector *v, const vector<int> sample_index){

	gsl_vector *v_sub = gsl_vector_alloc( sample_index.size() );

	for(unsigned int i=0; i<sample_index.size(); i++){
		gsl_vector_set( v_sub, i, gsl_vector_get( v, sample_index[i] ) );
	}

	return v_sub;
}

void gsl_matrix_reverse( gsl_matrix *M){

	for(unsigned int j=0; j<M->size2/2; j++){
		gsl_matrix_swap_columns( M, j, M->size2-j-1);
	}
}

void gsl_vector_sqrt( const gsl_vector *a, gsl_vector *b){

	for(unsigned int i=0; i<a->size; i++){
		gsl_vector_set(b, i, sqrt(gsl_vector_get(a, i)) );
	}
}

double gsl_ran_chisq_pdf_log( const double x, const double df ){

	//dchisq(sig_g, nu_a, log=TRUE)
	//-nu_a/2*log(2) - lgamma(nu_a/2) + (nu_a/2-1)*log(sig_g) - 1/2*sig_g

	return -df/2*log(2) - gsl_sf_lngamma(df/2) + (df/2-1)*log(x) - 0.5*x;
}

double gsl_ran_inverse_chisq_pdf_log( const double x, const double df ){

	//-nu_a/2*log(2) - lgamma(nu_a/2) - (nu_a/2+1)*log(sig_g) - 1/(2*sig_g)

	return -df/2*log(2) - gsl_sf_lngamma(df/2) - (df/2+1)*log(x) - 1/(2*x);
}

void inverse_logit( gsl_vector * __restrict__ mu, const gsl_vector * __restrict__ eta){

	for(unsigned int i=0; i< eta->size; i++){
		gsl_vector_set(mu, i, 1/(1 + exp( -gsl_vector_get(eta, i))));
	}
}

int gsl_matrix_daxpy( double alpha, const gsl_matrix * x, gsl_matrix * y){

	if( x->size1 != y->size1 || x->size2 != y->size2)
		GSL_ERROR("incompatible dimensions", GSL_EDOM);

	cblas_daxpy( x->size1 * x->size2, alpha, x->data, 1, y->data, 1);
}


bool gsl_vector_set_missing_mean(gsl_vector *v){

	double sum = 0;
	vector<int> index;

	// sum all finite non-nan values, saving indeces of infinte values
	for(unsigned int i=0; i<v->size; i++){
		if( isfinite(gsl_vector_get(v, i)) && ! isnan(gsl_vector_get(v, i)) )
			sum += gsl_vector_get(v, i);
		else
			index.push_back(i);
	}

	// divide by number of finite values
	double mean = sum / ((double) v->size - index.size() );

	for(unsigned int i=0; i<index.size(); i++){
		 gsl_vector_set(v, index[i], mean);
	}

	return( index.size() != 0) ;
}

long gsl_vector_count_missing(gsl_vector *v){

	double count = 0;

	for(unsigned int i=0; i<v->size; i++){

		if( isnan(gsl_vector_get(v, i)) || ! isfinite(gsl_vector_get(v, i)) ) count++;
	}

	return count;
}

gsl_vector *gsl_vector_get_nonmissing(gsl_vector *v){

	long n_missing = gsl_vector_count_missing(v);

 	// Return NULL if the vector is of size 0
	if( n_missing == v->size ) return NULL;

	gsl_vector *v_clean = gsl_vector_alloc(v->size - n_missing);

	long k = 0;

	for(unsigned int i=0; i<v->size; i++){

		if( ! isnan(gsl_vector_get(v, i)) && isfinite(gsl_vector_get(v, i)) ){	

			gsl_vector_set(v_clean, k++, gsl_vector_get(v, i));
		}		
	}

	return v_clean;
}

void gsl_matrix_set_missing_mean_col( gsl_matrix *X ){

	// set missing to mean
	gsl_vector_view col_view;

	for( unsigned int j=0; j<X->size2; j++){
		col_view = gsl_matrix_column( (gsl_matrix*)(X), j );
		gsl_vector_set_missing_mean( &col_view.vector );
	}		
}


void gsl_matrix_center_scale( gsl_matrix *X ){
	
	gsl_vector_view v;

	double scaled_norm;

	int n_indivs = X->size1;
	int n_features = X->size2;

	// for each feature
	for(int j=0; j<n_features; j++){

		v = gsl_matrix_column(X, j);

		// mean center
		///////////////
		
		// set v = v - mean(v)
		gsl_vector_add_constant(&v.vector, -1.0 * gsl_vector_sum_elements(&v.vector) / n_indivs );

		// scale so sample variance is 1
		////////////////////////////////

		scaled_norm = sqrt(n_indivs-1) / gsl_blas_dnrm2(&v.vector);

		gsl_blas_dscal( scaled_norm, &v.vector);
	}
}
