/**
 * gsl_additions_restrict
 *
 * @file
 * @author  Gabriel Hoffman
 * @date February 24, 2010
 *
 * @section DESCRIPTION
 *
 * Re-implementation of some simple gsl_vector functions from gsl_additions.h in order to take into account auto-vectorization using __restrict__
 * Use of __restrict__ tells the compiler that a pointer to an address is unique in that no other pointer is an alias to that address.
 * The relative performance of these methods vs their unoptimized counterparts depends heavily on the compiler flags and the processor
 */

#ifndef GSL_ADDITIONS_RESTRICT_H
#define GSL_ADDITIONS_RESTRICT_H

#include <gsl/gsl_vector.h>

// C = A - B
//void gsl_vector_subtr_restrict(const gsl_vector * __restrict__ A, const gsl_vector * __restrict__ B, gsl_vector * __restrict__ C);

#endif // GSL_ADDITIONS_RESTRICT_H
