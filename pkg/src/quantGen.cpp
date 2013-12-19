/*
 * quantGen.h
 *
 *  Created on: Feb 12, 2013
 *      Author: gh258
 */

#include "quantGen.h"

#include <math.h>

#include <gsl/gsl_blas.h>

void standardize_columns(gsl_matrix *X, const bool center, const bool standardize){

	unsigned int n_indivs = X->size1;
	unsigned int n_markers = X->size2;

	int OMP_CHUNK_SIZE = 100;

	#pragma omp parallel
	{
		double col_sum, scaled_norm;
		gsl_vector_view column_view;

		#pragma omp for schedule(static, OMP_CHUNK_SIZE)
		for(unsigned j=0; j<n_markers; j++){

			column_view = gsl_matrix_column( (gsl_matrix*)(X), j);

			if( center ){
				// mean center
				///////////////

				col_sum = 0;
				for(unsigned int i=0; i<X->size1; i++){
					col_sum += gsl_vector_get(&column_view.vector, i);
				}

				// set X_col= X_col- col_mean
				gsl_vector_add_constant( &column_view.vector, -col_sum/(double) n_indivs);
			}

			if(standardize){
				gsl_blas_dscal( sqrt(n_indivs) / gsl_blas_dnrm2(&column_view.vector) , &column_view.vector);
			}
		}
	}
}


void standardize_rows(gsl_matrix *X, const bool center, const bool standardize){

	unsigned int n_indivs = X->size2;
	unsigned int n_markers = X->size1;

	int OMP_CHUNK_SIZE = 100;

	#pragma omp parallel
	{
		double col_sum, scaled_norm;
		gsl_vector_view row_view;

		#pragma omp for schedule(static, OMP_CHUNK_SIZE)
		for(unsigned j=0; j<n_markers; j++){

			row_view = gsl_matrix_row( (gsl_matrix*)(X), j);

			if( center ){
				// mean center
				///////////////

				col_sum = 0;
				for(unsigned int i=0; i<n_indivs; i++){
					col_sum += gsl_vector_get(&row_view.vector, i);
				}

				// set X_col= X_col- col_mean
				gsl_vector_add_constant( &row_view.vector, -col_sum/(double) n_indivs);
			}

			if(standardize){
				gsl_blas_dscal( sqrt(n_indivs) / gsl_blas_dnrm2(&row_view.vector) , &row_view.vector);
			}
		}
	}
}

gsl_vector *get_column_scales(gsl_matrix *X){

	unsigned int n_markers = X->size2;

	gsl_vector *scales = gsl_vector_alloc( n_markers );

	int OMP_CHUNK_SIZE = 100;

	#pragma omp parallel
	{
		gsl_vector_view column_view;

		#pragma omp for schedule(static, OMP_CHUNK_SIZE)
		for(unsigned j=0; j<n_markers; j++){

			column_view = gsl_matrix_column( (gsl_matrix*)(X), j);
			gsl_vector_set(scales, j, pow(gsl_blas_dnrm2(&column_view.vector),2) );
		}
	}
	return scales;
}


gsl_vector *get_row_scales(gsl_matrix *X){

	unsigned int n_markers = X->size1;

	gsl_vector *scales = gsl_vector_alloc( n_markers );

	int OMP_CHUNK_SIZE = 100;

	#pragma omp parallel
	{
		gsl_vector_view row_view;

		#pragma omp for schedule(static, OMP_CHUNK_SIZE)
		for(unsigned j=0; j<n_markers; j++){

			row_view = gsl_matrix_row( (gsl_matrix*)(X), j);
			gsl_vector_set(scales, j, pow(gsl_blas_dnrm2(&row_view.vector),2) );
		}
	}
	return scales;
}


gsl_vector *get_column_centers(gsl_matrix *X){

	unsigned int n_indivs = X->size1;
	unsigned int n_markers = X->size2;

	gsl_vector *centers = gsl_vector_alloc(n_markers );

	int OMP_CHUNK_SIZE = 100;

	#pragma omp parallel
	{
		gsl_vector_view column_view;
		double sum;

		#pragma omp for schedule(static, OMP_CHUNK_SIZE)
		for(unsigned j=0; j<n_markers; j++){

			column_view = gsl_matrix_column( (gsl_matrix*)(X), j);

			sum = 0;
			for(unsigned int i=0; i<n_indivs; i++){
				sum += gsl_vector_get(&column_view.vector, i);
			}
			gsl_vector_set(centers, j, sum/(double) n_indivs);
		}
	}
	return centers;
}


gsl_vector *get_row_centers(gsl_matrix *X){

	unsigned int n_indivs = X->size2;
	unsigned int n_markers = X->size1;

	gsl_vector *centers = gsl_vector_alloc(n_markers );

	int OMP_CHUNK_SIZE = 100;

	#pragma omp parallel
	{
		gsl_vector_view row_view;
		double sum;

		#pragma omp for schedule(static, OMP_CHUNK_SIZE)
		for(unsigned j=0; j<n_markers; j++){

			row_view = gsl_matrix_row( (gsl_matrix*)(X), j);

			sum = 0;
			for(unsigned int i=0; i<n_indivs; i++){
				sum += gsl_vector_get(&row_view.vector, i);
			}
			gsl_vector_set(centers, j, sum/(double) n_indivs);
		}
	}
	return centers;
}
