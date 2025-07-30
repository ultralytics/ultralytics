/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#ifndef ISL_INT_H
#define ISL_INT_H
#define ISL_DEPRECATED_INT_H

#include <isl/hash.h>
#include <isl/printer.h>
#include <string.h>
#include <isl_config.h>

#ifdef USE_GMP_FOR_MP
#include <isl_int_gmp.h>
#endif

#ifdef USE_IMATH_FOR_MP
#ifdef USE_SMALL_INT_OPT
#include <isl_int_sioimath.h>
#else /* USE_SMALL_INT_OPT */
#include <isl_int_imath.h>
#endif /* USE_SMALL_INT_OPT */
#endif /* USE_IMATH_FOR_MP */

#define isl_int_is_zero(i)	(isl_int_sgn(i) == 0)
#define isl_int_is_one(i)	(isl_int_cmp_si(i,1) == 0)
#define isl_int_is_negone(i)	(isl_int_cmp_si(i,-1) == 0)
#define isl_int_is_pos(i)	(isl_int_sgn(i) > 0)
#define isl_int_is_neg(i)	(isl_int_sgn(i) < 0)
#define isl_int_is_nonpos(i)	(isl_int_sgn(i) <= 0)
#define isl_int_is_nonneg(i)	(isl_int_sgn(i) >= 0)

#ifndef USE_SMALL_INT_OPT
#define isl_int_print(out,i,width)					\
	do {								\
		char *s;						\
		s = isl_int_get_str(i);					\
		fprintf(out, "%*s", width, s);				\
		isl_int_free_str(s);                                        \
	} while (0)
#endif /* USE_SMALL_INT_OPT */

__isl_give isl_printer *isl_printer_print_isl_int(__isl_take isl_printer *p,
	isl_int i);

#endif /* ISL_INT_H */
