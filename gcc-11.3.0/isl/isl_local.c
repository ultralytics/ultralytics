/*
 * Copyright 2014      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl_mat_private.h>
#include <isl_seq.h>

/* Given a matrix "div" representing local variables,
 * does the variable at position "pos" have an explicit representation?
 */
isl_bool isl_local_div_is_known(__isl_keep isl_mat *div, int pos)
{
	if (!div)
		return isl_bool_error;
	if (pos < 0 || pos >= div->n_row)
		isl_die(isl_mat_get_ctx(div), isl_error_invalid,
			"position out of bounds", return isl_bool_error);
	return !isl_int_is_zero(div->row[pos][0]);
}

/* Compare two matrices representing local variables, defined over
 * the same space.
 *
 * Return -1 if "div1" is "smaller" than "div2", 1 if "div1" is "greater"
 * than "div2" and 0 if they are equal.
 *
 * The order is fairly arbitrary.  We do "prefer" divs that only involve
 * earlier dimensions in the sense that we consider matrices where
 * the first differing div involves earlier dimensions to be smaller.
 */
int isl_local_cmp(__isl_keep isl_mat *div1, __isl_keep isl_mat *div2)
{
	int i;
	int cmp;
	int known1, known2;
	int last1, last2;
	int n_col;

	if (div1 == div2)
		return 0;
	if (!div1)
		return -1;
	if (!div2)
		return 1;

	if (div1->n_row != div2->n_row)
		return div1->n_row - div2->n_row;

	n_col = isl_mat_cols(div1);
	for (i = 0; i < div1->n_row; ++i) {
		known1 = isl_local_div_is_known(div1, i);
		known2 = isl_local_div_is_known(div2, i);
		if (!known1 && !known2)
			continue;
		if (!known1)
			return 1;
		if (!known2)
			return -1;
		last1 = isl_seq_last_non_zero(div1->row[i] + 1, n_col - 1);
		last2 = isl_seq_last_non_zero(div2->row[i] + 1, n_col - 1);
		if (last1 != last2)
			return last1 - last2;
		cmp = isl_seq_cmp(div1->row[i], div2->row[i], n_col);
		if (cmp != 0)
			return cmp;
	}

	return 0;
}
