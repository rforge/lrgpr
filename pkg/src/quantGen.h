/*
 * quantGen.h
 *
 *  Created on: Feb 12, 2013
 *      Author: gh258
 */

#ifndef QUANTGEN_H_
#define QUANTGEN_H_

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

void standardize_columns(gsl_matrix *X, const bool center = true, const bool standardize = true);

void standardize_rows(gsl_matrix *X, const bool center = true, const bool standardize = true);

gsl_vector *get_column_scales(gsl_matrix *X);
gsl_vector *get_row_scales(gsl_matrix *X);


gsl_vector *get_column_centers(gsl_matrix *X);
gsl_vector *get_row_centers(gsl_matrix *X);

#endif /* QUANTGEN_H_ */
