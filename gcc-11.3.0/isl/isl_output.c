/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 * Copyright 2010      INRIA Saclay
 * Copyright 2012-2013 Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 * and INRIA Saclay - Ile-de-France, Parc Club Orsay Universite,
 * ZAC des vignes, 4 rue Jacques Monod, 91893 Orsay, France 
 * and Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 */

#include <stdlib.h>
#include <string.h>
#include <isl_ctx_private.h>
#include <isl_map_private.h>
#include <isl/set.h>
#include <isl_seq.h>
#include <isl_polynomial_private.h>
#include <isl_printer_private.h>
#include <isl_space_private.h>
#include <isl_mat_private.h>
#include <isl_vec_private.h>
#include <isl/union_map.h>
#include <isl/constraint.h>
#include <isl_local_space_private.h>
#include <isl_aff_private.h>
#include <isl_val_private.h>
#include <isl/ast_build.h>
#include <isl_sort.h>
#include <isl_output_private.h>

#include <bset_to_bmap.c>
#include <set_to_map.c>

static const char *s_to[2] = { " -> ", " \\to " };
static const char *s_and[2] = { " and ", " \\wedge " };
static const char *s_or[2] = { " or ", " \\vee " };
static const char *s_le[2] = { "<=", "\\le" };
static const char *s_ge[2] = { ">=", "\\ge" };
static const char *s_open_set[2] = { "{ ", "\\{\\, " };
static const char *s_close_set[2] = { " }", " \\,\\}" };
static const char *s_open_list[2] = { "[", "(" };
static const char *s_close_list[2] = { "]", ")" };
static const char *s_such_that[2] = { " : ", " \\mid " };
static const char *s_open_exists[2] = { "exists (", "\\exists \\, " };
static const char *s_close_exists[2] = { ")", "" };
static const char *s_div_prefix[2] = { "e", "\\alpha_" };
static const char *s_param_prefix[2] = { "p", "p_" };
static const char *s_input_prefix[2] = { "i", "i_" };
static const char *s_output_prefix[2] = { "o", "o_" };

static __isl_give isl_printer *print_constraint_polylib(
	struct isl_basic_map *bmap, int ineq, int n, __isl_take isl_printer *p)
{
	int i;
	unsigned n_in = isl_basic_map_dim(bmap, isl_dim_in);
	unsigned n_out = isl_basic_map_dim(bmap, isl_dim_out);
	unsigned nparam = isl_basic_map_dim(bmap, isl_dim_param);
	isl_int *c = ineq ? bmap->ineq[n] : bmap->eq[n];

	p = isl_printer_start_line(p);
	p = isl_printer_print_int(p, ineq);
	for (i = 0; i < n_out; ++i) {
		p = isl_printer_print_str(p, " ");
		p = isl_printer_print_isl_int(p, c[1+nparam+n_in+i]);
	}
	for (i = 0; i < n_in; ++i) {
		p = isl_printer_print_str(p, " ");
		p = isl_printer_print_isl_int(p, c[1+nparam+i]);
	}
	for (i = 0; i < bmap->n_div; ++i) {
		p = isl_printer_print_str(p, " ");
		p = isl_printer_print_isl_int(p, c[1+nparam+n_in+n_out+i]);
	}
	for (i = 0; i < nparam; ++i) {
		p = isl_printer_print_str(p, " ");
		p = isl_printer_print_isl_int(p, c[1+i]);
	}
	p = isl_printer_print_str(p, " ");
	p = isl_printer_print_isl_int(p, c[0]);
	p = isl_printer_end_line(p);
	return p;
}

static __isl_give isl_printer *print_constraints_polylib(
	struct isl_basic_map *bmap, __isl_take isl_printer *p)
{
	int i;

	p = isl_printer_set_isl_int_width(p, 5);

	for (i = 0; i < bmap->n_eq; ++i)
		p = print_constraint_polylib(bmap, 0, i, p);
	for (i = 0; i < bmap->n_ineq; ++i)
		p = print_constraint_polylib(bmap, 1, i, p);

	return p;
}

static __isl_give isl_printer *bset_print_constraints_polylib(
	struct isl_basic_set *bset, __isl_take isl_printer *p)
{
	return print_constraints_polylib(bset_to_bmap(bset), p);
}

static __isl_give isl_printer *isl_basic_map_print_polylib(
	__isl_keep isl_basic_map *bmap, __isl_take isl_printer *p, int ext)
{
	unsigned total = isl_basic_map_total_dim(bmap);
	p = isl_printer_start_line(p);
	p = isl_printer_print_int(p, bmap->n_eq + bmap->n_ineq);
	p = isl_printer_print_str(p, " ");
	p = isl_printer_print_int(p, 1 + total + 1);
	if (ext) {
		p = isl_printer_print_str(p, " ");
		p = isl_printer_print_int(p,
				    isl_basic_map_dim(bmap, isl_dim_out));
		p = isl_printer_print_str(p, " ");
		p = isl_printer_print_int(p,
				    isl_basic_map_dim(bmap, isl_dim_in));
		p = isl_printer_print_str(p, " ");
		p = isl_printer_print_int(p,
				    isl_basic_map_dim(bmap, isl_dim_div));
		p = isl_printer_print_str(p, " ");
		p = isl_printer_print_int(p,
				    isl_basic_map_dim(bmap, isl_dim_param));
	}
	p = isl_printer_end_line(p);
	return print_constraints_polylib(bmap, p);
}

static __isl_give isl_printer *isl_basic_set_print_polylib(
	__isl_keep isl_basic_set *bset, __isl_take isl_printer *p, int ext)
{
	return isl_basic_map_print_polylib(bset_to_bmap(bset), p, ext);
}

static __isl_give isl_printer *isl_map_print_polylib(__isl_keep isl_map *map,
	__isl_take isl_printer *p, int ext)
{
	int i;

	p = isl_printer_start_line(p);
	p = isl_printer_print_int(p, map->n);
	p = isl_printer_end_line(p);
	for (i = 0; i < map->n; ++i) {
		p = isl_printer_start_line(p);
		p = isl_printer_end_line(p);
		p = isl_basic_map_print_polylib(map->p[i], p, ext);
	}
	return p;
}

static __isl_give isl_printer *isl_set_print_polylib(__isl_keep isl_set *set,
	__isl_take isl_printer *p, int ext)
{
	return isl_map_print_polylib(set_to_map(set), p, ext);
}

static int count_same_name(__isl_keep isl_space *dim,
	enum isl_dim_type type, unsigned pos, const char *name)
{
	enum isl_dim_type t;
	unsigned p, s;
	int count = 0;

	for (t = isl_dim_param; t <= type && t <= isl_dim_out; ++t) {
		s = t == type ? pos : isl_space_dim(dim, t);
		for (p = 0; p < s; ++p) {
			const char *n = isl_space_get_dim_name(dim, t, p);
			if (n && !strcmp(n, name))
				count++;
		}
	}
	return count;
}

/* Print the name of the variable of type "type" and position "pos"
 * in "space" to "p".
 */
static __isl_give isl_printer *print_name(__isl_keep isl_space *space,
	__isl_take isl_printer *p, enum isl_dim_type type, unsigned pos,
	int latex)
{
	const char *name;
	char buffer[20];
	int primes;

	name = type == isl_dim_div ? NULL
				   : isl_space_get_dim_name(space, type, pos);

	if (!name) {
		const char *prefix;
		if (type == isl_dim_param)
			prefix = s_param_prefix[latex];
		else if (type == isl_dim_div)
			prefix = s_div_prefix[latex];
		else if (isl_space_is_set(space) || type == isl_dim_in)
			prefix = s_input_prefix[latex];
		else
			prefix = s_output_prefix[latex];
		snprintf(buffer, sizeof(buffer), "%s%d", prefix, pos);
		name = buffer;
	}
	primes = count_same_name(space, name == buffer ? isl_dim_div : type,
				 pos, name);
	p = isl_printer_print_str(p, name);
	while (primes-- > 0)
		p = isl_printer_print_str(p, "'");
	return p;
}

static enum isl_dim_type pos2type(__isl_keep isl_space *dim, unsigned *pos)
{
	enum isl_dim_type type;
	unsigned n_in = isl_space_dim(dim, isl_dim_in);
	unsigned n_out = isl_space_dim(dim, isl_dim_out);
	unsigned nparam = isl_space_dim(dim, isl_dim_param);

	if (*pos < 1 + nparam) {
		type = isl_dim_param;
		*pos -= 1;
	} else if (*pos < 1 + nparam + n_in) {
		type = isl_dim_in;
		*pos -= 1 + nparam;
	} else if (*pos < 1 + nparam + n_in + n_out) {
		type = isl_dim_out;
		*pos -= 1 + nparam + n_in;
	} else {
		type = isl_dim_div;
		*pos -= 1 + nparam + n_in + n_out;
	}

	return type;
}

/* Can the div expression of the integer division at position "row" of "div"
 * be printed?
 * In particular, are the div expressions available and does the selected
 * variable have a known explicit representation?
 * Furthermore, the Omega format does not allow any div expressions
 * to be printed.
 */
static isl_bool can_print_div_expr(__isl_keep isl_printer *p,
	__isl_keep isl_mat *div, int pos)
{
	if (p->output_format == ISL_FORMAT_OMEGA)
		return isl_bool_false;
	if (!div)
		return isl_bool_false;
	return !isl_int_is_zero(div->row[pos][0]);
}

static __isl_give isl_printer *print_div(__isl_keep isl_space *dim,
	__isl_keep isl_mat *div, int pos, __isl_take isl_printer *p);

static __isl_give isl_printer *print_term(__isl_keep isl_space *space,
	__isl_keep isl_mat *div,
	isl_int c, unsigned pos, __isl_take isl_printer *p, int latex)
{
	enum isl_dim_type type;
	int print_div_def;

	if (pos == 0)
		return isl_printer_print_isl_int(p, c);

	type = pos2type(space, &pos);
	print_div_def = type == isl_dim_div && can_print_div_expr(p, div, pos);

	if (isl_int_is_one(c))
		;
	else if (isl_int_is_negone(c))
		p = isl_printer_print_str(p, "-");
	else {
		p = isl_printer_print_isl_int(p, c);
		if (p->output_format == ISL_FORMAT_C || print_div_def)
			p = isl_printer_print_str(p, "*");
	}
	if (print_div_def)
		p = print_div(space, div, pos, p);
	else
		p = print_name(space, p, type, pos, latex);
	return p;
}

static __isl_give isl_printer *print_affine_of_len(__isl_keep isl_space *dim,
	__isl_keep isl_mat *div,
	__isl_take isl_printer *p, isl_int *c, int len)
{
	int i;
	int first;

	for (i = 0, first = 1; i < len; ++i) {
		int flip = 0;
		if (isl_int_is_zero(c[i]))
			continue;
		if (!first) {
			if (isl_int_is_neg(c[i])) {
				flip = 1;
				isl_int_neg(c[i], c[i]);
				p = isl_printer_print_str(p, " - ");
			} else 
				p = isl_printer_print_str(p, " + ");
		}
		first = 0;
		p = print_term(dim, div, c[i], i, p, 0);
		if (flip)
			isl_int_neg(c[i], c[i]);
	}
	if (first)
		p = isl_printer_print_str(p, "0");
	return p;
}

/* Print an affine expression "c" corresponding to a constraint in "bmap"
 * to "p", with the variable names taken from "space" and
 * the integer division definitions taken from "div".
 */
static __isl_give isl_printer *print_affine(__isl_keep isl_basic_map *bmap,
	__isl_keep isl_space *space, __isl_keep isl_mat *div,
	__isl_take isl_printer *p, isl_int *c)
{
	unsigned len = 1 + isl_basic_map_total_dim(bmap);
	return print_affine_of_len(space, div, p, c, len);
}

/* offset is the offset of local_dim inside data->type of data->space.
 */
static __isl_give isl_printer *print_nested_var_list(__isl_take isl_printer *p,
	__isl_keep isl_space *local_dim, enum isl_dim_type local_type,
	struct isl_print_space_data *data, int offset)
{
	int i;

	if (data->space != local_dim && local_type == isl_dim_out)
		offset += local_dim->n_in;

	for (i = 0; i < isl_space_dim(local_dim, local_type); ++i) {
		if (i)
			p = isl_printer_print_str(p, ", ");
		if (data->print_dim)
			p = data->print_dim(p, data, offset + i);
		else
			p = print_name(data->space, p, data->type, offset + i,
					data->latex);
	}
	return p;
}

static __isl_give isl_printer *print_var_list(__isl_take isl_printer *p,
	__isl_keep isl_space *space, enum isl_dim_type type)
{
	struct isl_print_space_data data = { .space = space, .type = type };

	return print_nested_var_list(p, space, type, &data, 0);
}

static __isl_give isl_printer *print_nested_map_dim(__isl_take isl_printer *p,
	__isl_keep isl_space *local_dim,
	struct isl_print_space_data *data, int offset);

static __isl_give isl_printer *print_nested_tuple(__isl_take isl_printer *p,
	__isl_keep isl_space *local_dim, enum isl_dim_type local_type,
	struct isl_print_space_data *data, int offset)
{
	const char *name = NULL;
	unsigned n = isl_space_dim(local_dim, local_type);
	if ((local_type == isl_dim_in || local_type == isl_dim_out)) {
		name = isl_space_get_tuple_name(local_dim, local_type);
		if (name) {
			if (data->latex)
				p = isl_printer_print_str(p, "\\mathrm{");
			p = isl_printer_print_str(p, name);
			if (data->latex)
				p = isl_printer_print_str(p, "}");
		}
	}
	if (!data->latex || n != 1 || name)
		p = isl_printer_print_str(p, s_open_list[data->latex]);
	if ((local_type == isl_dim_in || local_type == isl_dim_out) &&
	    local_dim->nested[local_type - isl_dim_in]) {
		if (data->space != local_dim && local_type == isl_dim_out)
			offset += local_dim->n_in;
		p = print_nested_map_dim(p,
				local_dim->nested[local_type - isl_dim_in],
				data, offset);
	} else
		p = print_nested_var_list(p, local_dim, local_type, data,
					  offset);
	if (!data->latex || n != 1 || name)
		p = isl_printer_print_str(p, s_close_list[data->latex]);
	return p;
}

static __isl_give isl_printer *print_tuple(__isl_keep isl_space *dim,
	__isl_take isl_printer *p, enum isl_dim_type type,
	struct isl_print_space_data *data)
{
	data->space = dim;
	data->type = type;
	return print_nested_tuple(p, dim, type, data, 0);
}

static __isl_give isl_printer *print_nested_map_dim(__isl_take isl_printer *p,
	__isl_keep isl_space *local_dim,
	struct isl_print_space_data *data, int offset)
{
	p = print_nested_tuple(p, local_dim, isl_dim_in, data, offset);
	p = isl_printer_print_str(p, s_to[data->latex]);
	p = print_nested_tuple(p, local_dim, isl_dim_out, data, offset);

	return p;
}

__isl_give isl_printer *isl_print_space(__isl_keep isl_space *space,
	__isl_take isl_printer *p, int rational,
	struct isl_print_space_data *data)
{
	if (rational && !data->latex)
		p = isl_printer_print_str(p, "rat: ");
	if (isl_space_is_params(space))
		;
	else if (isl_space_is_set(space))
		p = print_tuple(space, p, isl_dim_set, data);
	else {
		p = print_tuple(space, p, isl_dim_in, data);
		p = isl_printer_print_str(p, s_to[data->latex]);
		p = print_tuple(space, p, isl_dim_out, data);
	}

	return p;
}

static __isl_give isl_printer *print_omega_parameters(__isl_keep isl_space *dim,
	__isl_take isl_printer *p)
{
	if (isl_space_dim(dim, isl_dim_param) == 0)
		return p;

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "symbolic ");
	p = print_var_list(p, dim, isl_dim_param);
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);
	return p;
}

/* Does the inequality constraint following "i" in "bmap"
 * have an opposite value for the same last coefficient?
 * "last" is the position of the last coefficient of inequality "i".
 * If the next constraint is a div constraint, then it is ignored
 * since div constraints are not printed.
 */
static int next_is_opposite(__isl_keep isl_basic_map *bmap, int i, int last)
{
	unsigned total = isl_basic_map_total_dim(bmap);
	unsigned o_div = isl_basic_map_offset(bmap, isl_dim_div);

	if (i + 1 >= bmap->n_ineq)
		return 0;
	if (isl_seq_last_non_zero(bmap->ineq[i + 1], 1 + total) != last)
		return 0;
	if (last >= o_div &&
	    isl_basic_map_is_div_constraint(bmap, bmap->ineq[i + 1],
					    last - o_div))
		return 0;
	return isl_int_abs_eq(bmap->ineq[i][last], bmap->ineq[i + 1][last]) &&
		!isl_int_eq(bmap->ineq[i][last], bmap->ineq[i + 1][last]);
}

/* Return a string representation of the operator used when
 * printing a constraint where the LHS is greater than or equal to the LHS
 * (sign > 0) or smaller than or equal to the LHS (sign < 0).
 * If "strict" is set, then return the strict version of the comparison
 * operator.
 */
static const char *constraint_op(int sign, int strict, int latex)
{
	if (strict)
		return sign < 0 ? "<" : ">";
	if (sign < 0)
		return s_le[latex];
	else
		return s_ge[latex];
}

/* Print one side of a constraint "c" from "bmap" to "p", with
 * the variable names taken from "space" and the integer division definitions
 * taken from "div".
 * "last" is the position of the last non-zero coefficient.
 * Let c' be the result of zeroing out this coefficient, then
 * the partial constraint
 *
 *	c' op
 *
 * is printed.
 * "first_constraint" is set if this is the first constraint
 * in the conjunction.
 */
static __isl_give isl_printer *print_half_constraint(struct isl_basic_map *bmap,
	__isl_keep isl_space *space, __isl_keep isl_mat *div,
	__isl_take isl_printer *p, isl_int *c, int last, const char *op,
	int first_constraint, int latex)
{
	if (!first_constraint)
		p = isl_printer_print_str(p, s_and[latex]);

	isl_int_set_si(c[last], 0);
	p = print_affine(bmap, space, div, p, c);

	p = isl_printer_print_str(p, " ");
	p = isl_printer_print_str(p, op);
	p = isl_printer_print_str(p, " ");

	return p;
}

/* Print a constraint "c" from "bmap" to "p", with the variable names
 * taken from "space" and the integer division definitions taken from "div".
 * "last" is the position of the last non-zero coefficient, which is
 * moreover assumed to be negative.
 * Let c' be the result of zeroing out this coefficient, then
 * the constraint is printed in the form
 *
 *	-c[last] op c'
 *
 * "first_constraint" is set if this is the first constraint
 * in the conjunction.
 */
static __isl_give isl_printer *print_constraint(struct isl_basic_map *bmap,
	__isl_keep isl_space *space, __isl_keep isl_mat *div,
	__isl_take isl_printer *p,
	isl_int *c, int last, const char *op, int first_constraint, int latex)
{
	if (!first_constraint)
		p = isl_printer_print_str(p, s_and[latex]);

	isl_int_abs(c[last], c[last]);

	p = print_term(space, div, c[last], last, p, latex);

	p = isl_printer_print_str(p, " ");
	p = isl_printer_print_str(p, op);
	p = isl_printer_print_str(p, " ");

	isl_int_set_si(c[last], 0);
	p = print_affine(bmap, space, div, p, c);

	return p;
}

/* Print the constraints of "bmap" to "p".
 * The names of the variables are taken from "space" and
 * the integer division definitions are taken from "div".
 * Div constraints are only printed in "dump" mode.
 * The constraints are sorted prior to printing (except in "dump" mode).
 *
 * If x is the last variable with a non-zero coefficient,
 * then a lower bound
 *
 *	f - a x >= 0
 *
 * is printed as
 *
 *	a x <= f
 *
 * while an upper bound
 *
 *	f + a x >= 0
 *
 * is printed as
 *
 *	a x >= -f
 *
 * If the next constraint has an opposite sign for the same last coefficient,
 * then it is printed as
 *
 *	f >= a x
 *
 * or
 *
 *	-f <= a x
 *
 * instead.  In fact, the "a x" part is not printed explicitly, but
 * reused from the next constraint, which is therefore treated as
 * a first constraint in the conjunction.
 *
 * If the constant term of "f" is -1, then "f" is replaced by "f + 1" and
 * the comparison operator is replaced by the strict variant.
 * Essentially, ">= 1" is replaced by "> 0".
 */
static __isl_give isl_printer *print_constraints(__isl_keep isl_basic_map *bmap,
	__isl_keep isl_space *space, __isl_keep isl_mat *div,
	__isl_take isl_printer *p, int latex)
{
	int i;
	isl_vec *c = NULL;
	int rational = ISL_F_ISSET(bmap, ISL_BASIC_MAP_RATIONAL);
	unsigned total = isl_basic_map_total_dim(bmap);
	unsigned o_div = isl_basic_map_offset(bmap, isl_dim_div);
	int first = 1;
	int dump;

	if (!p)
		return NULL;
	bmap = isl_basic_map_copy(bmap);
	dump = p->dump;
	if (!dump)
		bmap = isl_basic_map_sort_constraints(bmap);
	if (!bmap)
		goto error;

	c = isl_vec_alloc(bmap->ctx, 1 + total);
	if (!c)
		goto error;

	for (i = bmap->n_eq - 1; i >= 0; --i) {
		int l = isl_seq_last_non_zero(bmap->eq[i], 1 + total);
		if (l < 0) {
			if (i != bmap->n_eq - 1)
				p = isl_printer_print_str(p, s_and[latex]);
			p = isl_printer_print_str(p, "0 = 0");
			continue;
		}
		if (isl_int_is_neg(bmap->eq[i][l]))
			isl_seq_cpy(c->el, bmap->eq[i], 1 + total);
		else
			isl_seq_neg(c->el, bmap->eq[i], 1 + total);
		p = print_constraint(bmap, space, div, p, c->el, l,
				    "=", first, latex);
		first = 0;
	}
	for (i = 0; i < bmap->n_ineq; ++i) {
		int l = isl_seq_last_non_zero(bmap->ineq[i], 1 + total);
		int strict;
		int s;
		const char *op;
		if (l < 0)
			continue;
		if (!dump && l >= o_div &&
		    can_print_div_expr(p, div, l - o_div) &&
		    isl_basic_map_is_div_constraint(bmap, bmap->ineq[i],
						    l - o_div))
			continue;
		s = isl_int_sgn(bmap->ineq[i][l]);
		strict = !rational && isl_int_is_negone(bmap->ineq[i][0]);
		if (s < 0)
			isl_seq_cpy(c->el, bmap->ineq[i], 1 + total);
		else
			isl_seq_neg(c->el, bmap->ineq[i], 1 + total);
		if (strict)
			isl_int_set_si(c->el[0], 0);
		if (!dump && next_is_opposite(bmap, i, l)) {
			op = constraint_op(-s, strict, latex);
			p = print_half_constraint(bmap, space, div, p, c->el, l,
						op, first, latex);
			first = 1;
		} else {
			op = constraint_op(s, strict, latex);
			p = print_constraint(bmap, space, div, p, c->el, l,
						op, first, latex);
			first = 0;
		}
	}

	isl_basic_map_free(bmap);
	isl_vec_free(c);

	return p;
error:
	isl_basic_map_free(bmap);
	isl_vec_free(c);
	isl_printer_free(p);
	return NULL;
}

static __isl_give isl_printer *print_div(__isl_keep isl_space *dim,
	__isl_keep isl_mat *div, int pos, __isl_take isl_printer *p)
{
	int c;

	if (!p || !div)
		return isl_printer_free(p);

	c = p->output_format == ISL_FORMAT_C;
	p = isl_printer_print_str(p, c ? "floord(" : "floor((");
	p = print_affine_of_len(dim, div, p,
				div->row[pos] + 1, div->n_col - 1);
	p = isl_printer_print_str(p, c ? ", " : ")/");
	p = isl_printer_print_isl_int(p, div->row[pos][0]);
	p = isl_printer_print_str(p, ")");
	return p;
}

/* Print a comma separated list of div names, except those that have
 * a definition that can be printed.
 * If "print_defined_divs" is set, then those div names are printed
 * as well, along with their definitions.
 */
static __isl_give isl_printer *print_div_list(__isl_take isl_printer *p,
	__isl_keep isl_space *space, __isl_keep isl_mat *div, int latex,
	int print_defined_divs)
{
	int i;
	int first = 1;
	unsigned n_div;

	if (!p || !space || !div)
		return isl_printer_free(p);

	n_div = isl_mat_rows(div);

	for (i = 0; i < n_div; ++i) {
		if (!print_defined_divs && can_print_div_expr(p, div, i))
			continue;
		if (!first)
			p = isl_printer_print_str(p, ", ");
		p = print_name(space, p, isl_dim_div, i, latex);
		first = 0;
		if (!can_print_div_expr(p, div, i))
			continue;
		p = isl_printer_print_str(p, " = ");
		p = print_div(space, div, i, p);
	}

	return p;
}

/* Does printing "bmap" require an "exists" clause?
 * That is, are there any local variables without an explicit representation?
 */
static isl_bool need_exists(__isl_keep isl_printer *p,
	__isl_keep isl_basic_map *bmap, __isl_keep isl_mat *div)
{
	int i;

	if (!p || !bmap)
		return isl_bool_error;
	if (bmap->n_div == 0)
		return isl_bool_false;
	for (i = 0; i < bmap->n_div; ++i)
		if (!can_print_div_expr(p, div, i))
			return isl_bool_true;
	return isl_bool_false;
}

/* Print the constraints of "bmap" to "p".
 * The names of the variables are taken from "space".
 * "latex" is set if the constraints should be printed in LaTeX format.
 * Do not print inline explicit div representations in "dump" mode.
 */
static __isl_give isl_printer *print_disjunct(__isl_keep isl_basic_map *bmap,
	__isl_keep isl_space *space, __isl_take isl_printer *p, int latex)
{
	int dump;
	isl_mat *div;
	isl_bool exists;

	if (!p)
		return NULL;
	dump = p->dump;
	div = isl_basic_map_get_divs(bmap);
	if (dump)
		exists = bmap->n_div > 0;
	else
		exists = need_exists(p, bmap, div);
	if (exists >= 0 && exists) {
		p = isl_printer_print_str(p, s_open_exists[latex]);
		p = print_div_list(p, space, div, latex, dump);
		p = isl_printer_print_str(p, ": ");
	}

	if (dump)
		div = isl_mat_free(div);
	p = print_constraints(bmap, space, div, p, latex);
	isl_mat_free(div);

	if (exists >= 0 && exists)
		p = isl_printer_print_str(p, s_close_exists[latex]);
	return p;
}

/* Print a colon followed by the constraints of "bmap"
 * to "p", provided there are any constraints.
 * The names of the variables are taken from "space".
 * "latex" is set if the constraints should be printed in LaTeX format.
 */
static __isl_give isl_printer *print_optional_disjunct(
	__isl_keep isl_basic_map *bmap, __isl_keep isl_space *space,
	__isl_take isl_printer *p, int latex)
{
	if (isl_basic_map_plain_is_universe(bmap))
		return p;

	p = isl_printer_print_str(p, ": ");
	p = print_disjunct(bmap, space, p, latex);

	return p;
}

static __isl_give isl_printer *basic_map_print_omega(
	__isl_keep isl_basic_map *bmap, __isl_take isl_printer *p)
{
	p = isl_printer_print_str(p, "{ [");
	p = print_var_list(p, bmap->dim, isl_dim_in);
	p = isl_printer_print_str(p, "] -> [");
	p = print_var_list(p, bmap->dim, isl_dim_out);
	p = isl_printer_print_str(p, "] ");
	p = print_optional_disjunct(bmap, bmap->dim, p, 0);
	p = isl_printer_print_str(p, " }");
	return p;
}

static __isl_give isl_printer *basic_set_print_omega(
	__isl_keep isl_basic_set *bset, __isl_take isl_printer *p)
{
	p = isl_printer_print_str(p, "{ [");
	p = print_var_list(p, bset->dim, isl_dim_set);
	p = isl_printer_print_str(p, "] ");
	p = print_optional_disjunct(bset, bset->dim, p, 0);
	p = isl_printer_print_str(p, " }");
	return p;
}

static __isl_give isl_printer *isl_map_print_omega(__isl_keep isl_map *map,
	__isl_take isl_printer *p)
{
	int i;

	for (i = 0; i < map->n; ++i) {
		if (i)
			p = isl_printer_print_str(p, " union ");
		p = basic_map_print_omega(map->p[i], p);
	}
	return p;
}

static __isl_give isl_printer *isl_set_print_omega(__isl_keep isl_set *set,
	__isl_take isl_printer *p)
{
	int i;

	for (i = 0; i < set->n; ++i) {
		if (i)
			p = isl_printer_print_str(p, " union ");
		p = basic_set_print_omega(set->p[i], p);
	}
	return p;
}

static __isl_give isl_printer *isl_basic_map_print_isl(
	__isl_keep isl_basic_map *bmap, __isl_take isl_printer *p,
	int latex)
{
	struct isl_print_space_data data = { .latex = latex };
	int rational = ISL_F_ISSET(bmap, ISL_BASIC_MAP_RATIONAL);

	if (isl_basic_map_dim(bmap, isl_dim_param) > 0) {
		p = print_tuple(bmap->dim, p, isl_dim_param, &data);
		p = isl_printer_print_str(p, " -> ");
	}
	p = isl_printer_print_str(p, "{ ");
	p = isl_print_space(bmap->dim, p, rational, &data);
	p = isl_printer_print_str(p, " : ");
	p = print_disjunct(bmap, bmap->dim, p, latex);
	p = isl_printer_print_str(p, " }");
	return p;
}

/* Print the disjuncts of a map (or set) "map" to "p".
 * The names of the variables are taken from "space".
 * "latex" is set if the constraints should be printed in LaTeX format.
 */
static __isl_give isl_printer *print_disjuncts_core(__isl_keep isl_map *map,
	__isl_keep isl_space *space, __isl_take isl_printer *p, int latex)
{
	int i;

	if (map->n == 0)
		p = isl_printer_print_str(p, "1 = 0");
	for (i = 0; i < map->n; ++i) {
		if (i)
			p = isl_printer_print_str(p, s_or[latex]);
		if (map->n > 1 && map->p[i]->n_eq + map->p[i]->n_ineq > 1)
			p = isl_printer_print_str(p, "(");
		p = print_disjunct(map->p[i], space, p, latex);
		if (map->n > 1 && map->p[i]->n_eq + map->p[i]->n_ineq > 1)
			p = isl_printer_print_str(p, ")");
	}
	return p;
}

/* Print the disjuncts of a map (or set) "map" to "p".
 * The names of the variables are taken from "space".
 * "hull" describes constraints shared by all disjuncts of "map".
 * "latex" is set if the constraints should be printed in LaTeX format.
 *
 * Print the disjuncts as a conjunction of "hull" and
 * the result of removing the constraints of "hull" from "map".
 * If this result turns out to be the universe, then simply print "hull".
 */
static __isl_give isl_printer *print_disjuncts_in_hull(__isl_keep isl_map *map,
	__isl_keep isl_space *space, __isl_take isl_basic_map *hull,
	__isl_take isl_printer *p, int latex)
{
	isl_bool is_universe;

	p = print_disjunct(hull, space, p, latex);
	map = isl_map_plain_gist_basic_map(isl_map_copy(map), hull);
	is_universe = isl_map_plain_is_universe(map);
	if (is_universe < 0)
		goto error;
	if (!is_universe) {
		p = isl_printer_print_str(p, s_and[latex]);
		p = isl_printer_print_str(p, "(");
		p = print_disjuncts_core(map, space, p, latex);
		p = isl_printer_print_str(p, ")");
	}
	isl_map_free(map);

	return p;
error:
	isl_map_free(map);
	isl_printer_free(p);
	return NULL;
}

/* Print the disjuncts of a map (or set) "map" to "p".
 * The names of the variables are taken from "space".
 * "latex" is set if the constraints should be printed in LaTeX format.
 *
 * If there are at least two disjuncts and "dump" mode is not turned out,
 * check for any shared constraints among all disjuncts.
 * If there are any, then print them separately in print_disjuncts_in_hull.
 */
static __isl_give isl_printer *print_disjuncts(__isl_keep isl_map *map,
	__isl_keep isl_space *space, __isl_take isl_printer *p, int latex)
{
	if (isl_map_plain_is_universe(map))
		return p;

	p = isl_printer_print_str(p, s_such_that[latex]);
	if (!p)
		return NULL;

	if (!p->dump && map->n >= 2) {
		isl_basic_map *hull;
		isl_bool is_universe;

		hull = isl_map_plain_unshifted_simple_hull(isl_map_copy(map));
		is_universe = isl_basic_map_plain_is_universe(hull);
		if (is_universe < 0)
			p = isl_printer_free(p);
		else if (!is_universe)
			return print_disjuncts_in_hull(map, space, hull,
							p, latex);
		isl_basic_map_free(hull);
	}

	return print_disjuncts_core(map, space, p, latex);
}

/* Print the disjuncts of a map (or set).
 * The names of the variables are taken from "space".
 * "latex" is set if the constraints should be printed in LaTeX format.
 *
 * If the map turns out to be a universal parameter domain, then
 * we need to print the colon.  Otherwise, the output looks identical
 * to the empty set.
 */
static __isl_give isl_printer *print_disjuncts_map(__isl_keep isl_map *map,
	__isl_keep isl_space *space, __isl_take isl_printer *p, int latex)
{
	if (isl_map_plain_is_universe(map) && isl_space_is_params(map->dim))
		return isl_printer_print_str(p, s_such_that[latex]);
	else
		return print_disjuncts(map, space, p, latex);
}

struct isl_aff_split {
	isl_basic_map *aff;
	isl_map *map;
};

static void free_split(__isl_take struct isl_aff_split *split, int n)
{
	int i;

	if (!split)
		return;

	for (i = 0; i < n; ++i) {
		isl_basic_map_free(split[i].aff);
		isl_map_free(split[i].map);
	}

	free(split);
}

static __isl_give isl_basic_map *get_aff(__isl_take isl_basic_map *bmap)
{
	int i, j;
	unsigned nparam, n_in, n_out, total;

	bmap = isl_basic_map_cow(bmap);
	if (!bmap)
		return NULL;
	if (isl_basic_map_free_inequality(bmap, bmap->n_ineq) < 0)
		goto error;

	nparam = isl_basic_map_dim(bmap, isl_dim_param);
	n_in = isl_basic_map_dim(bmap, isl_dim_in);
	n_out = isl_basic_map_dim(bmap, isl_dim_out);
	total = isl_basic_map_dim(bmap, isl_dim_all);
	for (i = bmap->n_eq - 1; i >= 0; --i) {
		j = isl_seq_last_non_zero(bmap->eq[i] + 1, total);
		if (j >= nparam && j < nparam + n_in + n_out &&
		    (isl_int_is_one(bmap->eq[i][1 + j]) ||
		     isl_int_is_negone(bmap->eq[i][1 + j])))
			continue;
		if (isl_basic_map_drop_equality(bmap, i) < 0)
			goto error;
	}

	bmap = isl_basic_map_finalize(bmap);

	return bmap;
error:
	isl_basic_map_free(bmap);
	return NULL;
}

static int aff_split_cmp(const void *p1, const void *p2, void *user)
{
	const struct isl_aff_split *s1, *s2;
	s1 = (const struct isl_aff_split *) p1;
	s2 = (const struct isl_aff_split *) p2;

	return isl_basic_map_plain_cmp(s1->aff, s2->aff);
}

static __isl_give isl_basic_map *drop_aff(__isl_take isl_basic_map *bmap,
	__isl_keep isl_basic_map *aff)
{
	int i, j;
	unsigned total;

	if (!bmap || !aff)
		goto error;

	total = isl_space_dim(bmap->dim, isl_dim_all);

	for (i = bmap->n_eq - 1; i >= 0; --i) {
		if (isl_seq_first_non_zero(bmap->eq[i] + 1 + total,
					    bmap->n_div) != -1)
			continue;
		for (j = 0; j < aff->n_eq; ++j) {
			if (!isl_seq_eq(bmap->eq[i], aff->eq[j], 1 + total) &&
			    !isl_seq_is_neg(bmap->eq[i], aff->eq[j], 1 + total))
				continue;
			if (isl_basic_map_drop_equality(bmap, i) < 0)
				goto error;
			break;
		}
	}

	return bmap;
error:
	isl_basic_map_free(bmap);
	return NULL;
}

static __isl_give struct isl_aff_split *split_aff(__isl_keep isl_map *map)
{
	int i, n;
	struct isl_aff_split *split;
	isl_ctx *ctx;

	ctx = isl_map_get_ctx(map);
	split = isl_calloc_array(ctx, struct isl_aff_split, map->n);
	if (!split)
		return NULL;

	for (i = 0; i < map->n; ++i) {
		isl_basic_map *bmap;
		split[i].aff = get_aff(isl_basic_map_copy(map->p[i]));
		bmap = isl_basic_map_copy(map->p[i]);
		bmap = isl_basic_map_cow(bmap);
		bmap = drop_aff(bmap, split[i].aff);
		split[i].map = isl_map_from_basic_map(bmap);
		if (!split[i].aff || !split[i].map)
			goto error;
	}

	if (isl_sort(split, map->n, sizeof(struct isl_aff_split),
			&aff_split_cmp, NULL) < 0)
		goto error;

	n = map->n;
	for (i = n - 1; i >= 1; --i) {
		if (!isl_basic_map_plain_is_equal(split[i - 1].aff,
						 split[i].aff))
			continue;
		isl_basic_map_free(split[i].aff);
		split[i - 1].map = isl_map_union(split[i - 1].map,
						 split[i].map);
		if (i != n - 1)
			split[i] = split[n - 1];
		split[n - 1].aff = NULL;
		split[n - 1].map = NULL;
		--n;
	}

	return split;
error:
	free_split(split, map->n);
	return NULL;
}

static int defining_equality(__isl_keep isl_basic_map *eq,
	__isl_keep isl_space *dim, enum isl_dim_type type, int pos)
{
	int i;
	unsigned total;

	if (!eq)
		return -1;

	pos += isl_space_offset(dim, type);
	total = isl_basic_map_total_dim(eq);

	for (i = 0; i < eq->n_eq; ++i) {
		if (isl_seq_last_non_zero(eq->eq[i] + 1, total) != pos)
			continue;
		if (isl_int_is_one(eq->eq[i][1 + pos]))
			isl_seq_neg(eq->eq[i], eq->eq[i], 1 + total);
		return i;
	}

	return -1;
}

/* Print dimension "pos" of data->space to "p".
 *
 * data->user is assumed to be an isl_basic_map keeping track of equalities.
 *
 * If the current dimension is defined by these equalities, then print
 * the corresponding expression, assigned to the name of the dimension
 * if there is any.  Otherwise, print the name of the dimension.
 */
static __isl_give isl_printer *print_dim_eq(__isl_take isl_printer *p,
	struct isl_print_space_data *data, unsigned pos)
{
	isl_basic_map *eq = data->user;
	int j;

	j = defining_equality(eq, data->space, data->type, pos);
	if (j >= 0) {
		if (isl_space_has_dim_name(data->space, data->type, pos)) {
			p = print_name(data->space, p, data->type, pos,
					data->latex);
			p = isl_printer_print_str(p, " = ");
		}
		pos += 1 + isl_space_offset(data->space, data->type);
		p = print_affine_of_len(data->space, NULL, p, eq->eq[j], pos);
	} else {
		p = print_name(data->space, p, data->type, pos, data->latex);
	}

	return p;
}

static __isl_give isl_printer *print_split_map(__isl_take isl_printer *p,
	struct isl_aff_split *split, int n, __isl_keep isl_space *space)
{
	struct isl_print_space_data data = { 0 };
	int i;
	int rational;

	data.print_dim = &print_dim_eq;
	for (i = 0; i < n; ++i) {
		if (!split[i].map)
			break;
		rational = split[i].map->n > 0 &&
		    ISL_F_ISSET(split[i].map->p[0], ISL_BASIC_MAP_RATIONAL);
		if (i)
			p = isl_printer_print_str(p, "; ");
		data.user = split[i].aff;
		p = isl_print_space(space, p, rational, &data);
		p = print_disjuncts_map(split[i].map, space, p, 0);
	}

	return p;
}

static __isl_give isl_printer *isl_map_print_isl_body(__isl_keep isl_map *map,
	__isl_take isl_printer *p)
{
	struct isl_print_space_data data = { 0 };
	struct isl_aff_split *split = NULL;
	int rational;

	if (!p || !map)
		return isl_printer_free(p);
	if (!p->dump && map->n > 0)
		split = split_aff(map);
	if (split) {
		p = print_split_map(p, split, map->n, map->dim);
	} else {
		rational = map->n > 0 &&
		    ISL_F_ISSET(map->p[0], ISL_BASIC_MAP_RATIONAL);
		p = isl_print_space(map->dim, p, rational, &data);
		p = print_disjuncts_map(map, map->dim, p, 0);
	}
	free_split(split, map->n);
	return p;
}

static __isl_give isl_printer *isl_map_print_isl(__isl_keep isl_map *map,
	__isl_take isl_printer *p)
{
	struct isl_print_space_data data = { 0 };

	if (isl_map_dim(map, isl_dim_param) > 0) {
		p = print_tuple(map->dim, p, isl_dim_param, &data);
		p = isl_printer_print_str(p, s_to[0]);
	}
	p = isl_printer_print_str(p, s_open_set[0]);
	p = isl_map_print_isl_body(map, p);
	p = isl_printer_print_str(p, s_close_set[0]);
	return p;
}

static __isl_give isl_printer *print_latex_map(__isl_keep isl_map *map,
	__isl_take isl_printer *p, __isl_keep isl_basic_map *aff)
{
	struct isl_print_space_data data = { 0 };

	data.latex = 1;
	if (isl_map_dim(map, isl_dim_param) > 0) {
		p = print_tuple(map->dim, p, isl_dim_param, &data);
		p = isl_printer_print_str(p, s_to[1]);
	}
	p = isl_printer_print_str(p, s_open_set[1]);
	data.print_dim = &print_dim_eq;
	data.user = aff;
	p = isl_print_space(map->dim, p, 0, &data);
	p = print_disjuncts_map(map, map->dim, p, 1);
	p = isl_printer_print_str(p, s_close_set[1]);

	return p;
}

static __isl_give isl_printer *isl_map_print_latex(__isl_keep isl_map *map,
	__isl_take isl_printer *p)
{
	int i;
	struct isl_aff_split *split = NULL;

	if (map->n > 0)
		split = split_aff(map);

	if (!split)
		return print_latex_map(map, p, NULL);

	for (i = 0; i < map->n; ++i) {
		if (!split[i].map)
			break;
		if (i)
			p = isl_printer_print_str(p, " \\cup ");
		p = print_latex_map(split[i].map, p, split[i].aff);
	}

	free_split(split, map->n);
	return p;
}

__isl_give isl_printer *isl_printer_print_basic_map(__isl_take isl_printer *p,
	__isl_keep isl_basic_map *bmap)
{
	if (!p || !bmap)
		goto error;
	if (p->output_format == ISL_FORMAT_ISL)
		return isl_basic_map_print_isl(bmap, p, 0);
	else if (p->output_format == ISL_FORMAT_OMEGA)
		return basic_map_print_omega(bmap, p);
	isl_assert(bmap->ctx, 0, goto error);
error:
	isl_printer_free(p);
	return NULL;
}

__isl_give isl_printer *isl_printer_print_basic_set(__isl_take isl_printer *p,
	__isl_keep isl_basic_set *bset)
{
	if (!p || !bset)
		goto error;

	if (p->output_format == ISL_FORMAT_ISL)
		return isl_basic_map_print_isl(bset, p, 0);
	else if (p->output_format == ISL_FORMAT_POLYLIB)
		return isl_basic_set_print_polylib(bset, p, 0);
	else if (p->output_format == ISL_FORMAT_EXT_POLYLIB)
		return isl_basic_set_print_polylib(bset, p, 1);
	else if (p->output_format == ISL_FORMAT_POLYLIB_CONSTRAINTS)
		return bset_print_constraints_polylib(bset, p);
	else if (p->output_format == ISL_FORMAT_OMEGA)
		return basic_set_print_omega(bset, p);
	isl_assert(p->ctx, 0, goto error);
error:
	isl_printer_free(p);
	return NULL;
}

__isl_give isl_printer *isl_printer_print_set(__isl_take isl_printer *p,
	__isl_keep isl_set *set)
{
	if (!p || !set)
		goto error;
	if (p->output_format == ISL_FORMAT_ISL)
		return isl_map_print_isl(set_to_map(set), p);
	else if (p->output_format == ISL_FORMAT_POLYLIB)
		return isl_set_print_polylib(set, p, 0);
	else if (p->output_format == ISL_FORMAT_EXT_POLYLIB)
		return isl_set_print_polylib(set, p, 1);
	else if (p->output_format == ISL_FORMAT_OMEGA)
		return isl_set_print_omega(set, p);
	else if (p->output_format == ISL_FORMAT_LATEX)
		return isl_map_print_latex(set_to_map(set), p);
	isl_assert(set->ctx, 0, goto error);
error:
	isl_printer_free(p);
	return NULL;
}

__isl_give isl_printer *isl_printer_print_map(__isl_take isl_printer *p,
	__isl_keep isl_map *map)
{
	if (!p || !map)
		goto error;

	if (p->output_format == ISL_FORMAT_ISL)
		return isl_map_print_isl(map, p);
	else if (p->output_format == ISL_FORMAT_POLYLIB)
		return isl_map_print_polylib(map, p, 0);
	else if (p->output_format == ISL_FORMAT_EXT_POLYLIB)
		return isl_map_print_polylib(map, p, 1);
	else if (p->output_format == ISL_FORMAT_OMEGA)
		return isl_map_print_omega(map, p);
	else if (p->output_format == ISL_FORMAT_LATEX)
		return isl_map_print_latex(map, p);
	isl_assert(map->ctx, 0, goto error);
error:
	isl_printer_free(p);
	return NULL;
}

struct isl_union_print_data {
	isl_printer *p;
	int first;
};

static isl_stat print_map_body(__isl_take isl_map *map, void *user)
{
	struct isl_union_print_data *data;
	data = (struct isl_union_print_data *)user;

	if (!data->first)
		data->p = isl_printer_print_str(data->p, "; ");
	data->first = 0;

	data->p = isl_map_print_isl_body(map, data->p);
	isl_map_free(map);

	return isl_stat_ok;
}

static __isl_give isl_printer *isl_union_map_print_isl(
	__isl_keep isl_union_map *umap, __isl_take isl_printer *p)
{
	struct isl_union_print_data data;
	struct isl_print_space_data space_data = { 0 };
	isl_space *dim;

	dim = isl_union_map_get_space(umap);
	if (isl_space_dim(dim, isl_dim_param) > 0) {
		p = print_tuple(dim, p, isl_dim_param, &space_data);
		p = isl_printer_print_str(p, s_to[0]);
	}
	isl_space_free(dim);
	p = isl_printer_print_str(p, s_open_set[0]);
	data.p = p;
	data.first = 1;
	isl_union_map_foreach_map(umap, &print_map_body, &data);
	p = data.p;
	p = isl_printer_print_str(p, s_close_set[0]);
	return p;
}

static isl_stat print_latex_map_body(__isl_take isl_map *map, void *user)
{
	struct isl_union_print_data *data;
	data = (struct isl_union_print_data *)user;

	if (!data->first)
		data->p = isl_printer_print_str(data->p, " \\cup ");
	data->first = 0;

	data->p = isl_map_print_latex(map, data->p);
	isl_map_free(map);

	return isl_stat_ok;
}

static __isl_give isl_printer *isl_union_map_print_latex(
	__isl_keep isl_union_map *umap, __isl_take isl_printer *p)
{
	struct isl_union_print_data data = { p, 1 };
	isl_union_map_foreach_map(umap, &print_latex_map_body, &data);
	p = data.p;
	return p;
}

__isl_give isl_printer *isl_printer_print_union_map(__isl_take isl_printer *p,
	__isl_keep isl_union_map *umap)
{
	if (!p || !umap)
		goto error;

	if (p->output_format == ISL_FORMAT_ISL)
		return isl_union_map_print_isl(umap, p);
	if (p->output_format == ISL_FORMAT_LATEX)
		return isl_union_map_print_latex(umap, p);

	isl_die(p->ctx, isl_error_invalid,
		"invalid output format for isl_union_map", goto error);
error:
	isl_printer_free(p);
	return NULL;
}

__isl_give isl_printer *isl_printer_print_union_set(__isl_take isl_printer *p,
	__isl_keep isl_union_set *uset)
{
	if (!p || !uset)
		goto error;

	if (p->output_format == ISL_FORMAT_ISL)
		return isl_union_map_print_isl((isl_union_map *)uset, p);
	if (p->output_format == ISL_FORMAT_LATEX)
		return isl_union_map_print_latex((isl_union_map *)uset, p);

	isl_die(p->ctx, isl_error_invalid,
		"invalid output format for isl_union_set", goto error);
error:
	isl_printer_free(p);
	return NULL;
}

static int upoly_rec_n_non_zero(__isl_keep struct isl_upoly_rec *rec)
{
	int i;
	int n;

	for (i = 0, n = 0; i < rec->n; ++i)
		if (!isl_upoly_is_zero(rec->p[i]))
			++n;

	return n;
}

static __isl_give isl_printer *upoly_print_cst(__isl_keep struct isl_upoly *up,
	__isl_take isl_printer *p, int first)
{
	struct isl_upoly_cst *cst;
	int neg;

	cst = isl_upoly_as_cst(up);
	if (!cst)
		goto error;
	neg = !first && isl_int_is_neg(cst->n);
	if (!first)
		p = isl_printer_print_str(p, neg ? " - " :  " + ");
	if (neg)
		isl_int_neg(cst->n, cst->n);
	if (isl_int_is_zero(cst->d)) {
		int sgn = isl_int_sgn(cst->n);
		p = isl_printer_print_str(p, sgn < 0 ? "-infty" :
					    sgn == 0 ? "NaN" : "infty");
	} else
		p = isl_printer_print_isl_int(p, cst->n);
	if (neg)
		isl_int_neg(cst->n, cst->n);
	if (!isl_int_is_zero(cst->d) && !isl_int_is_one(cst->d)) {
		p = isl_printer_print_str(p, "/");
		p = isl_printer_print_isl_int(p, cst->d);
	}
	return p;
error:
	isl_printer_free(p);
	return NULL;
}

static __isl_give isl_printer *print_base(__isl_take isl_printer *p,
	__isl_keep isl_space *dim, __isl_keep isl_mat *div, int var)
{
	unsigned total;

	total = isl_space_dim(dim, isl_dim_all);
	if (var < total)
		p = print_term(dim, NULL, dim->ctx->one, 1 + var, p, 0);
	else
		p = print_div(dim, div, var - total, p);
	return p;
}

static __isl_give isl_printer *print_pow(__isl_take isl_printer *p,
	__isl_keep isl_space *dim, __isl_keep isl_mat *div, int var, int exp)
{
	p = print_base(p, dim, div, var);
	if (exp == 1)
		return p;
	if (p->output_format == ISL_FORMAT_C) {
		int i;
		for (i = 1; i < exp; ++i) {
			p = isl_printer_print_str(p, "*");
			p = print_base(p, dim, div, var);
		}
	} else {
		p = isl_printer_print_str(p, "^");
		p = isl_printer_print_int(p, exp);
	}
	return p;
}

static __isl_give isl_printer *upoly_print(__isl_keep struct isl_upoly *up,
	__isl_keep isl_space *dim, __isl_keep isl_mat *div,
	__isl_take isl_printer *p, int outer)
{
	int i, n, first, print_parens;
	struct isl_upoly_rec *rec;

	if (!p || !up || !dim || !div)
		goto error;

	if (isl_upoly_is_cst(up))
		return upoly_print_cst(up, p, 1);

	rec = isl_upoly_as_rec(up);
	if (!rec)
		goto error;
	n = upoly_rec_n_non_zero(rec);
	print_parens = n > 1 ||
		    (outer && rec->up.var >= isl_space_dim(dim, isl_dim_all));
	if (print_parens)
		p = isl_printer_print_str(p, "(");
	for (i = 0, first = 1; i < rec->n; ++i) {
		if (isl_upoly_is_zero(rec->p[i]))
			continue;
		if (isl_upoly_is_negone(rec->p[i])) {
			if (!i)
				p = isl_printer_print_str(p, "-1");
			else if (first)
				p = isl_printer_print_str(p, "-");
			else
				p = isl_printer_print_str(p, " - ");
		} else if (isl_upoly_is_cst(rec->p[i]) &&
				!isl_upoly_is_one(rec->p[i]))
			p = upoly_print_cst(rec->p[i], p, first);
		else {
			if (!first)
				p = isl_printer_print_str(p, " + ");
			if (i == 0 || !isl_upoly_is_one(rec->p[i]))
				p = upoly_print(rec->p[i], dim, div, p, 0);
		}
		first = 0;
		if (i == 0)
			continue;
		if (!isl_upoly_is_one(rec->p[i]) &&
		    !isl_upoly_is_negone(rec->p[i]))
			p = isl_printer_print_str(p, " * ");
		p = print_pow(p, dim, div, rec->up.var, i);
	}
	if (print_parens)
		p = isl_printer_print_str(p, ")");
	return p;
error:
	isl_printer_free(p);
	return NULL;
}

static __isl_give isl_printer *print_qpolynomial(__isl_take isl_printer *p,
	__isl_keep isl_qpolynomial *qp)
{
	if (!p || !qp)
		goto error;
	p = upoly_print(qp->upoly, qp->dim, qp->div, p, 1);
	return p;
error:
	isl_printer_free(p);
	return NULL;
}

static __isl_give isl_printer *print_qpolynomial_isl(__isl_take isl_printer *p,
	__isl_keep isl_qpolynomial *qp)
{
	struct isl_print_space_data data = { 0 };

	if (!p || !qp)
		goto error;

	if (isl_space_dim(qp->dim, isl_dim_param) > 0) {
		p = print_tuple(qp->dim, p, isl_dim_param, &data);
		p = isl_printer_print_str(p, " -> ");
	}
	p = isl_printer_print_str(p, "{ ");
	if (!isl_space_is_params(qp->dim)) {
		p = isl_print_space(qp->dim, p, 0, &data);
		p = isl_printer_print_str(p, " -> ");
	}
	p = print_qpolynomial(p, qp);
	p = isl_printer_print_str(p, " }");
	return p;
error:
	isl_printer_free(p);
	return NULL;
}

static __isl_give isl_printer *print_qpolynomial_c(__isl_take isl_printer *p,
	__isl_keep isl_space *dim, __isl_keep isl_qpolynomial *qp)
{
	isl_int den;

	isl_int_init(den);
	isl_qpolynomial_get_den(qp, &den);
	if (!isl_int_is_one(den)) {
		isl_qpolynomial *f;
		p = isl_printer_print_str(p, "(");
		qp = isl_qpolynomial_copy(qp);
		f = isl_qpolynomial_rat_cst_on_domain(isl_space_copy(qp->dim),
						den, qp->dim->ctx->one);
		qp = isl_qpolynomial_mul(qp, f);
	}
	if (qp)
		p = upoly_print(qp->upoly, dim, qp->div, p, 0);
	else
		p = isl_printer_free(p);
	if (!isl_int_is_one(den)) {
		p = isl_printer_print_str(p, ")/");
		p = isl_printer_print_isl_int(p, den);
		isl_qpolynomial_free(qp);
	}
	isl_int_clear(den);
	return p;
}

__isl_give isl_printer *isl_printer_print_qpolynomial(
	__isl_take isl_printer *p, __isl_keep isl_qpolynomial *qp)
{
	if (!p || !qp)
		goto error;

	if (p->output_format == ISL_FORMAT_ISL)
		return print_qpolynomial_isl(p, qp);
	else if (p->output_format == ISL_FORMAT_C)
		return print_qpolynomial_c(p, qp->dim, qp);
	else
		isl_die(qp->dim->ctx, isl_error_unsupported,
			"output format not supported for isl_qpolynomials",
			goto error);
error:
	isl_printer_free(p);
	return NULL;
}

void isl_qpolynomial_print(__isl_keep isl_qpolynomial *qp, FILE *out,
	unsigned output_format)
{
	isl_printer *p;

	if  (!qp)
		return;

	isl_assert(qp->dim->ctx, output_format == ISL_FORMAT_ISL, return);
	p = isl_printer_to_file(qp->dim->ctx, out);
	p = isl_printer_print_qpolynomial(p, qp);
	isl_printer_free(p);
}

static __isl_give isl_printer *qpolynomial_fold_print(
	__isl_keep isl_qpolynomial_fold *fold, __isl_take isl_printer *p)
{
	int i;

	if (fold->type == isl_fold_min)
		p = isl_printer_print_str(p, "min");
	else if (fold->type == isl_fold_max)
		p = isl_printer_print_str(p, "max");
	p = isl_printer_print_str(p, "(");
	for (i = 0; i < fold->n; ++i) {
		if (i)
			p = isl_printer_print_str(p, ", ");
		p = print_qpolynomial(p, fold->qp[i]);
	}
	p = isl_printer_print_str(p, ")");
	return p;
}

void isl_qpolynomial_fold_print(__isl_keep isl_qpolynomial_fold *fold,
	FILE *out, unsigned output_format)
{
	isl_printer *p;

	if (!fold)
		return;

	isl_assert(fold->dim->ctx, output_format == ISL_FORMAT_ISL, return);

	p = isl_printer_to_file(fold->dim->ctx, out);
	p = isl_printer_print_qpolynomial_fold(p, fold);

	isl_printer_free(p);
}

static __isl_give isl_printer *isl_pwqp_print_isl_body(
	__isl_take isl_printer *p, __isl_keep isl_pw_qpolynomial *pwqp)
{
	struct isl_print_space_data data = { 0 };
	int i = 0;

	for (i = 0; i < pwqp->n; ++i) {
		isl_space *space;

		if (i)
			p = isl_printer_print_str(p, "; ");
		space = isl_qpolynomial_get_domain_space(pwqp->p[i].qp);
		if (!isl_space_is_params(space)) {
			p = isl_print_space(space, p, 0, &data);
			p = isl_printer_print_str(p, " -> ");
		}
		p = print_qpolynomial(p, pwqp->p[i].qp);
		p = print_disjuncts(set_to_map(pwqp->p[i].set), space, p, 0);
		isl_space_free(space);
	}

	return p;
}

static __isl_give isl_printer *print_pw_qpolynomial_isl(
	__isl_take isl_printer *p, __isl_keep isl_pw_qpolynomial *pwqp)
{
	struct isl_print_space_data data = { 0 };

	if (!p || !pwqp)
		goto error;

	if (isl_space_dim(pwqp->dim, isl_dim_param) > 0) {
		p = print_tuple(pwqp->dim, p, isl_dim_param, &data);
		p = isl_printer_print_str(p, " -> ");
	}
	p = isl_printer_print_str(p, "{ ");
	if (pwqp->n == 0) {
		if (!isl_space_is_set(pwqp->dim)) {
			p = print_tuple(pwqp->dim, p, isl_dim_in, &data);
			p = isl_printer_print_str(p, " -> ");
		}
		p = isl_printer_print_str(p, "0");
	}
	p = isl_pwqp_print_isl_body(p, pwqp);
	p = isl_printer_print_str(p, " }");
	return p;
error:
	isl_printer_free(p);
	return NULL;
}

void isl_pw_qpolynomial_print(__isl_keep isl_pw_qpolynomial *pwqp, FILE *out,
	unsigned output_format)
{
	isl_printer *p;

	if (!pwqp)
		return;

	p = isl_printer_to_file(pwqp->dim->ctx, out);
	p = isl_printer_set_output_format(p, output_format);
	p = isl_printer_print_pw_qpolynomial(p, pwqp);

	isl_printer_free(p);
}

static __isl_give isl_printer *isl_pwf_print_isl_body(
	__isl_take isl_printer *p, __isl_keep isl_pw_qpolynomial_fold *pwf)
{
	struct isl_print_space_data data = { 0 };
	int i = 0;

	for (i = 0; i < pwf->n; ++i) {
		isl_space *space;

		if (i)
			p = isl_printer_print_str(p, "; ");
		space = isl_qpolynomial_fold_get_domain_space(pwf->p[i].fold);
		if (!isl_space_is_params(space)) {
			p = isl_print_space(space, p, 0, &data);
			p = isl_printer_print_str(p, " -> ");
		}
		p = qpolynomial_fold_print(pwf->p[i].fold, p);
		p = print_disjuncts(set_to_map(pwf->p[i].set), space, p, 0);
		isl_space_free(space);
	}

	return p;
}

static __isl_give isl_printer *print_pw_qpolynomial_fold_isl(
	__isl_take isl_printer *p, __isl_keep isl_pw_qpolynomial_fold *pwf)
{
	struct isl_print_space_data data = { 0 };

	if (isl_space_dim(pwf->dim, isl_dim_param) > 0) {
		p = print_tuple(pwf->dim, p, isl_dim_param, &data);
		p = isl_printer_print_str(p, " -> ");
	}
	p = isl_printer_print_str(p, "{ ");
	if (pwf->n == 0) {
		if (!isl_space_is_set(pwf->dim)) {
			p = print_tuple(pwf->dim, p, isl_dim_in, &data);
			p = isl_printer_print_str(p, " -> ");
		}
		p = isl_printer_print_str(p, "0");
	}
	p = isl_pwf_print_isl_body(p, pwf);
	p = isl_printer_print_str(p, " }");
	return p;
}

static __isl_give isl_printer *print_affine_c(__isl_take isl_printer *p,
	__isl_keep isl_space *dim, __isl_keep isl_basic_set *bset, isl_int *c);

static __isl_give isl_printer *print_name_c(__isl_take isl_printer *p,
	__isl_keep isl_space *dim,
	__isl_keep isl_basic_set *bset, enum isl_dim_type type, unsigned pos)
{
	if (type == isl_dim_div) {
		p = isl_printer_print_str(p, "floord(");
		p = print_affine_c(p, dim, bset, bset->div[pos] + 1);
		p = isl_printer_print_str(p, ", ");
		p = isl_printer_print_isl_int(p, bset->div[pos][0]);
		p = isl_printer_print_str(p, ")");
	} else {
		const char *name;

		name = isl_space_get_dim_name(dim, type, pos);
		if (!name)
			name = "UNNAMED";
		p = isl_printer_print_str(p, name);
	}
	return p;
}

static __isl_give isl_printer *print_term_c(__isl_take isl_printer *p,
	__isl_keep isl_space *dim,
	__isl_keep isl_basic_set *bset, isl_int c, unsigned pos)
{
	enum isl_dim_type type;

	if (pos == 0)
		return isl_printer_print_isl_int(p, c);

	if (isl_int_is_one(c))
		;
	else if (isl_int_is_negone(c))
		p = isl_printer_print_str(p, "-");
	else {
		p = isl_printer_print_isl_int(p, c);
		p = isl_printer_print_str(p, "*");
	}
	type = pos2type(dim, &pos);
	p = print_name_c(p, dim, bset, type, pos);
	return p;
}

static __isl_give isl_printer *print_partial_affine_c(__isl_take isl_printer *p,
	__isl_keep isl_space *dim,
	__isl_keep isl_basic_set *bset, isl_int *c, unsigned len)
{
	int i;
	int first;

	for (i = 0, first = 1; i < len; ++i) {
		int flip = 0;
		if (isl_int_is_zero(c[i]))
			continue;
		if (!first) {
			if (isl_int_is_neg(c[i])) {
				flip = 1;
				isl_int_neg(c[i], c[i]);
				p = isl_printer_print_str(p, " - ");
			} else 
				p = isl_printer_print_str(p, " + ");
		}
		first = 0;
		p = print_term_c(p, dim, bset, c[i], i);
		if (flip)
			isl_int_neg(c[i], c[i]);
	}
	if (first)
		p = isl_printer_print_str(p, "0");
	return p;
}

static __isl_give isl_printer *print_affine_c(__isl_take isl_printer *p,
	__isl_keep isl_space *dim, __isl_keep isl_basic_set *bset, isl_int *c)
{
	unsigned len = 1 + isl_basic_set_total_dim(bset);
	return print_partial_affine_c(p, dim, bset, c, len);
}

/* We skip the constraint if it is implied by the div expression.
 *
 * *first indicates whether this is the first constraint in the conjunction and
 * is updated if the constraint is actually printed.
 */
static __isl_give isl_printer *print_constraint_c(__isl_take isl_printer *p,
	__isl_keep isl_space *dim,
	__isl_keep isl_basic_set *bset, isl_int *c, const char *op, int *first)
{
	unsigned o_div;
	unsigned n_div;
	int div;

	o_div = isl_basic_set_offset(bset, isl_dim_div);
	n_div = isl_basic_set_dim(bset, isl_dim_div);
	div = isl_seq_last_non_zero(c + o_div, n_div);
	if (div >= 0 && isl_basic_set_is_div_constraint(bset, c, div))
		return p;

	if (!*first)
		p = isl_printer_print_str(p, " && ");

	p = print_affine_c(p, dim, bset, c);
	p = isl_printer_print_str(p, " ");
	p = isl_printer_print_str(p, op);
	p = isl_printer_print_str(p, " 0");

	*first = 0;

	return p;
}

static __isl_give isl_printer *print_basic_set_c(__isl_take isl_printer *p,
	__isl_keep isl_space *dim, __isl_keep isl_basic_set *bset)
{
	int i, j;
	int first = 1;
	unsigned n_div = isl_basic_set_dim(bset, isl_dim_div);
	unsigned total = isl_basic_set_total_dim(bset) - n_div;

	for (i = 0; i < bset->n_eq; ++i) {
		j = isl_seq_last_non_zero(bset->eq[i] + 1 + total, n_div);
		if (j < 0)
			p = print_constraint_c(p, dim, bset,
						bset->eq[i], "==", &first);
		else {
			if (i)
				p = isl_printer_print_str(p, " && ");
			p = isl_printer_print_str(p, "(");
			p = print_partial_affine_c(p, dim, bset, bset->eq[i],
						   1 + total + j);
			p = isl_printer_print_str(p, ") % ");
			p = isl_printer_print_isl_int(p,
						bset->eq[i][1 + total + j]);
			p = isl_printer_print_str(p, " == 0");
			first = 0;
		}
	}
	for (i = 0; i < bset->n_ineq; ++i)
		p = print_constraint_c(p, dim, bset, bset->ineq[i], ">=",
					&first);
	return p;
}

static __isl_give isl_printer *print_set_c(__isl_take isl_printer *p,
	__isl_keep isl_space *dim, __isl_keep isl_set *set)
{
	int i;

	if (!set)
		return isl_printer_free(p);

	if (set->n == 0)
		p = isl_printer_print_str(p, "0");

	for (i = 0; i < set->n; ++i) {
		if (i)
			p = isl_printer_print_str(p, " || ");
		if (set->n > 1)
			p = isl_printer_print_str(p, "(");
		p = print_basic_set_c(p, dim, set->p[i]);
		if (set->n > 1)
			p = isl_printer_print_str(p, ")");
	}
	return p;
}

static __isl_give isl_printer *print_pw_qpolynomial_c(
	__isl_take isl_printer *p, __isl_keep isl_pw_qpolynomial *pwqp)
{
	int i;

	if (pwqp->n == 1 && isl_set_plain_is_universe(pwqp->p[0].set))
		return print_qpolynomial_c(p, pwqp->dim, pwqp->p[0].qp);

	for (i = 0; i < pwqp->n; ++i) {
		p = isl_printer_print_str(p, "(");
		p = print_set_c(p, pwqp->dim, pwqp->p[i].set);
		p = isl_printer_print_str(p, ") ? (");
		p = print_qpolynomial_c(p, pwqp->dim, pwqp->p[i].qp);
		p = isl_printer_print_str(p, ") : ");
	}

	p = isl_printer_print_str(p, "0");
	return p;
}

__isl_give isl_printer *isl_printer_print_pw_qpolynomial(
	__isl_take isl_printer *p, __isl_keep isl_pw_qpolynomial *pwqp)
{
	if (!p || !pwqp)
		goto error;

	if (p->output_format == ISL_FORMAT_ISL)
		return print_pw_qpolynomial_isl(p, pwqp);
	else if (p->output_format == ISL_FORMAT_C)
		return print_pw_qpolynomial_c(p, pwqp);
	isl_assert(p->ctx, 0, goto error);
error:
	isl_printer_free(p);
	return NULL;
}

static isl_stat print_pwqp_body(__isl_take isl_pw_qpolynomial *pwqp, void *user)
{
	struct isl_union_print_data *data;
	data = (struct isl_union_print_data *)user;

	if (!data->first)
		data->p = isl_printer_print_str(data->p, "; ");
	data->first = 0;

	data->p = isl_pwqp_print_isl_body(data->p, pwqp);
	isl_pw_qpolynomial_free(pwqp);

	return isl_stat_ok;
}

static __isl_give isl_printer *print_union_pw_qpolynomial_isl(
	__isl_take isl_printer *p, __isl_keep isl_union_pw_qpolynomial *upwqp)
{
	struct isl_union_print_data data;
	struct isl_print_space_data space_data = { 0 };
	isl_space *dim;

	dim = isl_union_pw_qpolynomial_get_space(upwqp);
	if (isl_space_dim(dim, isl_dim_param) > 0) {
		p = print_tuple(dim, p, isl_dim_param, &space_data);
		p = isl_printer_print_str(p, " -> ");
	}
	isl_space_free(dim);
	p = isl_printer_print_str(p, "{ ");
	data.p = p;
	data.first = 1;
	isl_union_pw_qpolynomial_foreach_pw_qpolynomial(upwqp, &print_pwqp_body,
							&data);
	p = data.p;
	p = isl_printer_print_str(p, " }");
	return p;
}

__isl_give isl_printer *isl_printer_print_union_pw_qpolynomial(
	__isl_take isl_printer *p, __isl_keep isl_union_pw_qpolynomial *upwqp)
{
	if (!p || !upwqp)
		goto error;

	if (p->output_format == ISL_FORMAT_ISL)
		return print_union_pw_qpolynomial_isl(p, upwqp);
	isl_die(p->ctx, isl_error_invalid,
		"invalid output format for isl_union_pw_qpolynomial",
		goto error);
error:
	isl_printer_free(p);
	return NULL;
}

static __isl_give isl_printer *print_qpolynomial_fold_c(
	__isl_take isl_printer *p, __isl_keep isl_space *dim,
	__isl_keep isl_qpolynomial_fold *fold)
{
	int i;

	for (i = 0; i < fold->n - 1; ++i)
		if (fold->type == isl_fold_min)
			p = isl_printer_print_str(p, "min(");
		else if (fold->type == isl_fold_max)
			p = isl_printer_print_str(p, "max(");

	for (i = 0; i < fold->n; ++i) {
		if (i)
			p = isl_printer_print_str(p, ", ");
		p = print_qpolynomial_c(p, dim, fold->qp[i]);
		if (i)
			p = isl_printer_print_str(p, ")");
	}
	return p;
}

__isl_give isl_printer *isl_printer_print_qpolynomial_fold(
	__isl_take isl_printer *p, __isl_keep isl_qpolynomial_fold *fold)
{
	if  (!p || !fold)
		goto error;
	if (p->output_format == ISL_FORMAT_ISL)
		return qpolynomial_fold_print(fold, p);
	else if (p->output_format == ISL_FORMAT_C)
		return print_qpolynomial_fold_c(p, fold->dim, fold);
	isl_die(p->ctx, isl_error_unsupported, "unsupported output format",
		goto error);
error:
	isl_printer_free(p);
	return NULL;
}

static __isl_give isl_printer *print_pw_qpolynomial_fold_c(
	__isl_take isl_printer *p, __isl_keep isl_pw_qpolynomial_fold *pwf)
{
	int i;

	if (pwf->n == 1 && isl_set_plain_is_universe(pwf->p[0].set))
		return print_qpolynomial_fold_c(p, pwf->dim, pwf->p[0].fold);

	for (i = 0; i < pwf->n; ++i) {
		p = isl_printer_print_str(p, "(");
		p = print_set_c(p, pwf->dim, pwf->p[i].set);
		p = isl_printer_print_str(p, ") ? (");
		p = print_qpolynomial_fold_c(p, pwf->dim, pwf->p[i].fold);
		p = isl_printer_print_str(p, ") : ");
	}

	p = isl_printer_print_str(p, "0");
	return p;
}

__isl_give isl_printer *isl_printer_print_pw_qpolynomial_fold(
	__isl_take isl_printer *p, __isl_keep isl_pw_qpolynomial_fold *pwf)
{
	if (!p || !pwf)
		goto error;

	if (p->output_format == ISL_FORMAT_ISL)
		return print_pw_qpolynomial_fold_isl(p, pwf);
	else if (p->output_format == ISL_FORMAT_C)
		return print_pw_qpolynomial_fold_c(p, pwf);
	isl_assert(p->ctx, 0, goto error);
error:
	isl_printer_free(p);
	return NULL;
}

void isl_pw_qpolynomial_fold_print(__isl_keep isl_pw_qpolynomial_fold *pwf,
	FILE *out, unsigned output_format)
{
	isl_printer *p;

	if (!pwf)
		return;

	p = isl_printer_to_file(pwf->dim->ctx, out);
	p = isl_printer_set_output_format(p, output_format);
	p = isl_printer_print_pw_qpolynomial_fold(p, pwf);

	isl_printer_free(p);
}

static isl_stat print_pwf_body(__isl_take isl_pw_qpolynomial_fold *pwf,
	void *user)
{
	struct isl_union_print_data *data;
	data = (struct isl_union_print_data *)user;

	if (!data->first)
		data->p = isl_printer_print_str(data->p, "; ");
	data->first = 0;

	data->p = isl_pwf_print_isl_body(data->p, pwf);
	isl_pw_qpolynomial_fold_free(pwf);

	return isl_stat_ok;
}

static __isl_give isl_printer *print_union_pw_qpolynomial_fold_isl(
	__isl_take isl_printer *p,
	__isl_keep isl_union_pw_qpolynomial_fold *upwf)
{
	struct isl_union_print_data data;
	struct isl_print_space_data space_data = { 0 };
	isl_space *dim;

	dim = isl_union_pw_qpolynomial_fold_get_space(upwf);
	if (isl_space_dim(dim, isl_dim_param) > 0) {
		p = print_tuple(dim, p, isl_dim_param, &space_data);
		p = isl_printer_print_str(p, " -> ");
	}
	isl_space_free(dim);
	p = isl_printer_print_str(p, "{ ");
	data.p = p;
	data.first = 1;
	isl_union_pw_qpolynomial_fold_foreach_pw_qpolynomial_fold(upwf,
							&print_pwf_body, &data);
	p = data.p;
	p = isl_printer_print_str(p, " }");
	return p;
}

__isl_give isl_printer *isl_printer_print_union_pw_qpolynomial_fold(
	__isl_take isl_printer *p,
	__isl_keep isl_union_pw_qpolynomial_fold *upwf)
{
	if (!p || !upwf)
		goto error;

	if (p->output_format == ISL_FORMAT_ISL)
		return print_union_pw_qpolynomial_fold_isl(p, upwf);
	isl_die(p->ctx, isl_error_invalid,
		"invalid output format for isl_union_pw_qpolynomial_fold",
		goto error);
error:
	isl_printer_free(p);
	return NULL;
}

__isl_give isl_printer *isl_printer_print_constraint(__isl_take isl_printer *p,
	__isl_keep isl_constraint *c)
{
	isl_basic_map *bmap;

	if (!p || !c)
		goto error;

	bmap = isl_basic_map_from_constraint(isl_constraint_copy(c));
	p = isl_printer_print_basic_map(p, bmap);
	isl_basic_map_free(bmap);
	return p;
error:
	isl_printer_free(p);
	return NULL;
}

static __isl_give isl_printer *isl_printer_print_space_isl(
	__isl_take isl_printer *p, __isl_keep isl_space *space)
{
	struct isl_print_space_data data = { 0 };

	if (!space)
		goto error;

	if (isl_space_dim(space, isl_dim_param) > 0) {
		p = print_tuple(space, p, isl_dim_param, &data);
		p = isl_printer_print_str(p, " -> ");
	}

	p = isl_printer_print_str(p, "{ ");
	if (isl_space_is_params(space))
		p = isl_printer_print_str(p, s_such_that[0]);
	else
		p = isl_print_space(space, p, 0, &data);
	p = isl_printer_print_str(p, " }");

	return p;
error:
	isl_printer_free(p);
	return NULL;
}

__isl_give isl_printer *isl_printer_print_space(__isl_take isl_printer *p,
	__isl_keep isl_space *space)
{
	if (!p || !space)
		return isl_printer_free(p);
	if (p->output_format == ISL_FORMAT_ISL)
		return isl_printer_print_space_isl(p, space);
	else if (p->output_format == ISL_FORMAT_OMEGA)
		return print_omega_parameters(space, p);

	isl_die(isl_space_get_ctx(space), isl_error_unsupported,
		"output format not supported for space",
		return isl_printer_free(p));
}

__isl_give isl_printer *isl_printer_print_local_space(__isl_take isl_printer *p,
	__isl_keep isl_local_space *ls)
{
	struct isl_print_space_data data = { 0 };
	unsigned n_div;

	if (!ls)
		goto error;

	if (isl_local_space_dim(ls, isl_dim_param) > 0) {
		p = print_tuple(ls->dim, p, isl_dim_param, &data);
		p = isl_printer_print_str(p, " -> ");
	}
	p = isl_printer_print_str(p, "{ ");
	p = isl_print_space(ls->dim, p, 0, &data);
	n_div = isl_local_space_dim(ls, isl_dim_div);
	if (n_div > 0) {
		p = isl_printer_print_str(p, " : ");
		p = isl_printer_print_str(p, s_open_exists[0]);
		p = print_div_list(p, ls->dim, ls->div, 0, 1);
		p = isl_printer_print_str(p, s_close_exists[0]);
	} else if (isl_space_is_params(ls->dim))
		p = isl_printer_print_str(p, s_such_that[0]);
	p = isl_printer_print_str(p, " }");
	return p;
error:
	isl_printer_free(p);
	return NULL;
}

static __isl_give isl_printer *print_aff_body(__isl_take isl_printer *p,
	__isl_keep isl_aff *aff)
{
	unsigned total;

	if (isl_aff_is_nan(aff))
		return isl_printer_print_str(p, "NaN");

	total = isl_local_space_dim(aff->ls, isl_dim_all);
	p = isl_printer_print_str(p, "(");
	p = print_affine_of_len(aff->ls->dim, aff->ls->div, p,
				aff->v->el + 1, 1 + total);
	if (isl_int_is_one(aff->v->el[0]))
		p = isl_printer_print_str(p, ")");
	else {
		p = isl_printer_print_str(p, ")/");
		p = isl_printer_print_isl_int(p, aff->v->el[0]);
	}

	return p;
}

static __isl_give isl_printer *print_aff(__isl_take isl_printer *p,
	__isl_keep isl_aff *aff)
{
	struct isl_print_space_data data = { 0 };

	if (isl_space_is_params(aff->ls->dim))
		;
	else {
		p = print_tuple(aff->ls->dim, p, isl_dim_set, &data);
		p = isl_printer_print_str(p, " -> ");
	}
	p = isl_printer_print_str(p, "[");
	p = print_aff_body(p, aff);
	p = isl_printer_print_str(p, "]");

	return p;
}

static __isl_give isl_printer *print_aff_isl(__isl_take isl_printer *p,
	__isl_keep isl_aff *aff)
{
	struct isl_print_space_data data = { 0 };

	if (!aff)
		goto error;

	if (isl_local_space_dim(aff->ls, isl_dim_param) > 0) {
		p = print_tuple(aff->ls->dim, p, isl_dim_param, &data);
		p = isl_printer_print_str(p, " -> ");
	}
	p = isl_printer_print_str(p, "{ ");
	p = print_aff(p, aff);
	p = isl_printer_print_str(p, " }");
	return p;
error:
	isl_printer_free(p);
	return NULL;
}

/* Print the body of an isl_pw_aff, i.e., a semicolon delimited
 * sequence of affine expressions, each followed by constraints.
 */
static __isl_give isl_printer *print_pw_aff_body(
	__isl_take isl_printer *p, __isl_keep isl_pw_aff *pa)
{
	int i;

	if (!pa)
		return isl_printer_free(p);

	for (i = 0; i < pa->n; ++i) {
		isl_space *space;

		if (i)
			p = isl_printer_print_str(p, "; ");
		p = print_aff(p, pa->p[i].aff);
		space = isl_aff_get_domain_space(pa->p[i].aff);
		p = print_disjuncts(set_to_map(pa->p[i].set), space, p, 0);
		isl_space_free(space);
	}

	return p;
}

static __isl_give isl_printer *print_pw_aff_isl(__isl_take isl_printer *p,
	__isl_keep isl_pw_aff *pwaff)
{
	struct isl_print_space_data data = { 0 };

	if (!pwaff)
		goto error;

	if (isl_space_dim(pwaff->dim, isl_dim_param) > 0) {
		p = print_tuple(pwaff->dim, p, isl_dim_param, &data);
		p = isl_printer_print_str(p, " -> ");
	}
	p = isl_printer_print_str(p, "{ ");
	p = print_pw_aff_body(p, pwaff);
	p = isl_printer_print_str(p, " }");
	return p;
error:
	isl_printer_free(p);
	return NULL;
}

static __isl_give isl_printer *print_ls_affine_c(__isl_take isl_printer *p,
	__isl_keep isl_local_space *ls, isl_int *c);

static __isl_give isl_printer *print_ls_name_c(__isl_take isl_printer *p,
	__isl_keep isl_local_space *ls, enum isl_dim_type type, unsigned pos)
{
	if (type == isl_dim_div) {
		p = isl_printer_print_str(p, "floord(");
		p = print_ls_affine_c(p, ls, ls->div->row[pos] + 1);
		p = isl_printer_print_str(p, ", ");
		p = isl_printer_print_isl_int(p, ls->div->row[pos][0]);
		p = isl_printer_print_str(p, ")");
	} else {
		const char *name;

		name = isl_space_get_dim_name(ls->dim, type, pos);
		if (!name)
			name = "UNNAMED";
		p = isl_printer_print_str(p, name);
	}
	return p;
}

static __isl_give isl_printer *print_ls_term_c(__isl_take isl_printer *p,
	__isl_keep isl_local_space *ls, isl_int c, unsigned pos)
{
	enum isl_dim_type type;

	if (pos == 0)
		return isl_printer_print_isl_int(p, c);

	if (isl_int_is_one(c))
		;
	else if (isl_int_is_negone(c))
		p = isl_printer_print_str(p, "-");
	else {
		p = isl_printer_print_isl_int(p, c);
		p = isl_printer_print_str(p, "*");
	}
	type = pos2type(ls->dim, &pos);
	p = print_ls_name_c(p, ls, type, pos);
	return p;
}

static __isl_give isl_printer *print_ls_partial_affine_c(
	__isl_take isl_printer *p, __isl_keep isl_local_space *ls,
	isl_int *c, unsigned len)
{
	int i;
	int first;

	for (i = 0, first = 1; i < len; ++i) {
		int flip = 0;
		if (isl_int_is_zero(c[i]))
			continue;
		if (!first) {
			if (isl_int_is_neg(c[i])) {
				flip = 1;
				isl_int_neg(c[i], c[i]);
				p = isl_printer_print_str(p, " - ");
			} else 
				p = isl_printer_print_str(p, " + ");
		}
		first = 0;
		p = print_ls_term_c(p, ls, c[i], i);
		if (flip)
			isl_int_neg(c[i], c[i]);
	}
	if (first)
		p = isl_printer_print_str(p, "0");
	return p;
}

static __isl_give isl_printer *print_ls_affine_c(__isl_take isl_printer *p,
	__isl_keep isl_local_space *ls, isl_int *c)
{
	unsigned len = 1 + isl_local_space_dim(ls, isl_dim_all);
	return print_ls_partial_affine_c(p, ls, c, len);
}

static __isl_give isl_printer *print_aff_c(__isl_take isl_printer *p,
	__isl_keep isl_aff *aff)
{
	unsigned total;

	total = isl_local_space_dim(aff->ls, isl_dim_all);
	if (!isl_int_is_one(aff->v->el[0]))
		p = isl_printer_print_str(p, "(");
	p = print_ls_partial_affine_c(p, aff->ls, aff->v->el + 1, 1 + total);
	if (!isl_int_is_one(aff->v->el[0])) {
		p = isl_printer_print_str(p, ")/");
		p = isl_printer_print_isl_int(p, aff->v->el[0]);
	}
	return p;
}

/* In the C format, we cannot express that "pwaff" may be undefined
 * on parts of the domain space.  We therefore assume that the expression
 * will only be evaluated on its definition domain and compute the gist
 * of each cell with respect to this domain.
 */
static __isl_give isl_printer *print_pw_aff_c(__isl_take isl_printer *p,
	__isl_keep isl_pw_aff *pwaff)
{
	isl_set *domain;
	isl_ast_build *build;
	isl_ast_expr *expr;

	if (pwaff->n < 1)
		isl_die(p->ctx, isl_error_unsupported,
			"cannot print empty isl_pw_aff in C format",
			return isl_printer_free(p));

	domain = isl_pw_aff_domain(isl_pw_aff_copy(pwaff));
	build = isl_ast_build_from_context(domain);
	expr = isl_ast_build_expr_from_pw_aff(build, isl_pw_aff_copy(pwaff));
	p = isl_printer_print_ast_expr(p, expr);
	isl_ast_expr_free(expr);
	isl_ast_build_free(build);

	return p;
}

__isl_give isl_printer *isl_printer_print_aff(__isl_take isl_printer *p,
	__isl_keep isl_aff *aff)
{
	if (!p || !aff)
		goto error;

	if (p->output_format == ISL_FORMAT_ISL)
		return print_aff_isl(p, aff);
	else if (p->output_format == ISL_FORMAT_C)
		return print_aff_c(p, aff);
	isl_die(p->ctx, isl_error_unsupported, "unsupported output format",
		goto error);
error:
	isl_printer_free(p);
	return NULL;
}

__isl_give isl_printer *isl_printer_print_pw_aff(__isl_take isl_printer *p,
	__isl_keep isl_pw_aff *pwaff)
{
	if (!p || !pwaff)
		goto error;

	if (p->output_format == ISL_FORMAT_ISL)
		return print_pw_aff_isl(p, pwaff);
	else if (p->output_format == ISL_FORMAT_C)
		return print_pw_aff_c(p, pwaff);
	isl_die(p->ctx, isl_error_unsupported, "unsupported output format",
		goto error);
error:
	isl_printer_free(p);
	return NULL;
}

/* Print "pa" in a sequence of isl_pw_affs delimited by semicolons.
 * Each isl_pw_aff itself is also printed as semicolon delimited
 * sequence of pieces.
 * If data->first = 1, then this is the first in the sequence.
 * Update data->first to tell the next element that it is not the first.
 */
static isl_stat print_pw_aff_body_wrap(__isl_take isl_pw_aff *pa,
	void *user)
{
	struct isl_union_print_data *data;
	data = (struct isl_union_print_data *) user;

	if (!data->first)
		data->p = isl_printer_print_str(data->p, "; ");
	data->first = 0;

	data->p = print_pw_aff_body(data->p, pa);
	isl_pw_aff_free(pa);

	return data->p ? isl_stat_ok : isl_stat_error;
}

/* Print the body of an isl_union_pw_aff, i.e., a semicolon delimited
 * sequence of affine expressions, each followed by constraints,
 * with the sequence enclosed in braces.
 */
static __isl_give isl_printer *print_union_pw_aff_body(
	__isl_take isl_printer *p, __isl_keep isl_union_pw_aff *upa)
{
	struct isl_union_print_data data = { p, 1 };

	p = isl_printer_print_str(p, s_open_set[0]);
	data.p = p;
	if (isl_union_pw_aff_foreach_pw_aff(upa,
					    &print_pw_aff_body_wrap, &data) < 0)
		data.p = isl_printer_free(p);
	p = data.p;
	p = isl_printer_print_str(p, s_close_set[0]);

	return p;
}

/* Print the isl_union_pw_aff "upa" to "p" in isl format.
 *
 * The individual isl_pw_affs are delimited by a semicolon.
 */
static __isl_give isl_printer *print_union_pw_aff_isl(
	__isl_take isl_printer *p, __isl_keep isl_union_pw_aff *upa)
{
	struct isl_print_space_data data = { 0 };
	isl_space *space;

	space = isl_union_pw_aff_get_space(upa);
	if (isl_space_dim(space, isl_dim_param) > 0) {
		p = print_tuple(space, p, isl_dim_param, &data);
		p = isl_printer_print_str(p, s_to[0]);
	}
	isl_space_free(space);
	p = print_union_pw_aff_body(p, upa);
	return p;
}

/* Print the isl_union_pw_aff "upa" to "p".
 *
 * We currently only support an isl format.
 */
__isl_give isl_printer *isl_printer_print_union_pw_aff(
	__isl_take isl_printer *p, __isl_keep isl_union_pw_aff *upa)
{
	if (!p || !upa)
		return isl_printer_free(p);

	if (p->output_format == ISL_FORMAT_ISL)
		return print_union_pw_aff_isl(p, upa);
	isl_die(isl_printer_get_ctx(p), isl_error_unsupported,
		"unsupported output format", return isl_printer_free(p));
}

/* Print dimension "pos" of data->space to "p".
 *
 * data->user is assumed to be an isl_multi_aff.
 *
 * If the current dimension is an output dimension, then print
 * the corresponding expression.  Otherwise, print the name of the dimension.
 */
static __isl_give isl_printer *print_dim_ma(__isl_take isl_printer *p,
	struct isl_print_space_data *data, unsigned pos)
{
	isl_multi_aff *ma = data->user;

	if (data->type == isl_dim_out)
		p = print_aff_body(p, ma->p[pos]);
	else
		p = print_name(data->space, p, data->type, pos, data->latex);

	return p;
}

static __isl_give isl_printer *print_multi_aff(__isl_take isl_printer *p,
	__isl_keep isl_multi_aff *maff)
{
	struct isl_print_space_data data = { 0 };

	data.print_dim = &print_dim_ma;
	data.user = maff;
	return isl_print_space(maff->space, p, 0, &data);
}

static __isl_give isl_printer *print_multi_aff_isl(__isl_take isl_printer *p,
	__isl_keep isl_multi_aff *maff)
{
	struct isl_print_space_data data = { 0 };

	if (!maff)
		goto error;

	if (isl_space_dim(maff->space, isl_dim_param) > 0) {
		p = print_tuple(maff->space, p, isl_dim_param, &data);
		p = isl_printer_print_str(p, " -> ");
	}
	p = isl_printer_print_str(p, "{ ");
	p = print_multi_aff(p, maff);
	p = isl_printer_print_str(p, " }");
	return p;
error:
	isl_printer_free(p);
	return NULL;
}

__isl_give isl_printer *isl_printer_print_multi_aff(__isl_take isl_printer *p,
	__isl_keep isl_multi_aff *maff)
{
	if (!p || !maff)
		goto error;

	if (p->output_format == ISL_FORMAT_ISL)
		return print_multi_aff_isl(p, maff);
	isl_die(p->ctx, isl_error_unsupported, "unsupported output format",
		goto error);
error:
	isl_printer_free(p);
	return NULL;
}

static __isl_give isl_printer *print_pw_multi_aff_body(
	__isl_take isl_printer *p, __isl_keep isl_pw_multi_aff *pma)
{
	int i;

	if (!pma)
		goto error;

	for (i = 0; i < pma->n; ++i) {
		isl_space *space;

		if (i)
			p = isl_printer_print_str(p, "; ");
		p = print_multi_aff(p, pma->p[i].maff);
		space = isl_multi_aff_get_domain_space(pma->p[i].maff);
		p = print_disjuncts(set_to_map(pma->p[i].set), space, p, 0);
		isl_space_free(space);
	}
	return p;
error:
	isl_printer_free(p);
	return NULL;
}

static __isl_give isl_printer *print_pw_multi_aff_isl(__isl_take isl_printer *p,
	__isl_keep isl_pw_multi_aff *pma)
{
	struct isl_print_space_data data = { 0 };

	if (!pma)
		goto error;

	if (isl_space_dim(pma->dim, isl_dim_param) > 0) {
		p = print_tuple(pma->dim, p, isl_dim_param, &data);
		p = isl_printer_print_str(p, " -> ");
	}
	p = isl_printer_print_str(p, "{ ");
	p = print_pw_multi_aff_body(p, pma);
	p = isl_printer_print_str(p, " }");
	return p;
error:
	isl_printer_free(p);
	return NULL;
}

static __isl_give isl_printer *print_unnamed_pw_multi_aff_c(
	__isl_take isl_printer *p, __isl_keep isl_pw_multi_aff *pma)
{
	int i;

	for (i = 0; i < pma->n - 1; ++i) {
		p = isl_printer_print_str(p, "(");
		p = print_set_c(p, pma->dim, pma->p[i].set);
		p = isl_printer_print_str(p, ") ? (");
		p = print_aff_c(p, pma->p[i].maff->p[0]);
		p = isl_printer_print_str(p, ") : ");
	}

	return print_aff_c(p, pma->p[pma->n - 1].maff->p[0]);
}

static __isl_give isl_printer *print_pw_multi_aff_c(__isl_take isl_printer *p,
	__isl_keep isl_pw_multi_aff *pma)
{
	int n;
	const char *name;

	if (!pma)
		goto error;
	if (pma->n < 1)
		isl_die(p->ctx, isl_error_unsupported,
			"cannot print empty isl_pw_multi_aff in C format",
			goto error);
	name = isl_pw_multi_aff_get_tuple_name(pma, isl_dim_out);
	if (!name && isl_pw_multi_aff_dim(pma, isl_dim_out) == 1)
		return print_unnamed_pw_multi_aff_c(p, pma);
	if (!name)
		isl_die(p->ctx, isl_error_unsupported,
			"cannot print unnamed isl_pw_multi_aff in C format",
			goto error);

	p = isl_printer_print_str(p, name);
	n = isl_pw_multi_aff_dim(pma, isl_dim_out);
	if (n != 0)
		isl_die(p->ctx, isl_error_unsupported,
			"not supported yet", goto error);

	return p;
error:
	isl_printer_free(p);
	return NULL;
}

__isl_give isl_printer *isl_printer_print_pw_multi_aff(
	__isl_take isl_printer *p, __isl_keep isl_pw_multi_aff *pma)
{
	if (!p || !pma)
		goto error;

	if (p->output_format == ISL_FORMAT_ISL)
		return print_pw_multi_aff_isl(p, pma);
	if (p->output_format == ISL_FORMAT_C)
		return print_pw_multi_aff_c(p, pma);
	isl_die(p->ctx, isl_error_unsupported, "unsupported output format",
		goto error);
error:
	isl_printer_free(p);
	return NULL;
}

static isl_stat print_pw_multi_aff_body_wrap(__isl_take isl_pw_multi_aff *pma,
	void *user)
{
	struct isl_union_print_data *data;
	data = (struct isl_union_print_data *) user;

	if (!data->first)
		data->p = isl_printer_print_str(data->p, "; ");
	data->first = 0;

	data->p = print_pw_multi_aff_body(data->p, pma);
	isl_pw_multi_aff_free(pma);

	return isl_stat_ok;
}

static __isl_give isl_printer *print_union_pw_multi_aff_isl(
	__isl_take isl_printer *p, __isl_keep isl_union_pw_multi_aff *upma)
{
	struct isl_union_print_data data;
	struct isl_print_space_data space_data = { 0 };
	isl_space *space;

	space = isl_union_pw_multi_aff_get_space(upma);
	if (isl_space_dim(space, isl_dim_param) > 0) {
		p = print_tuple(space, p, isl_dim_param, &space_data);
		p = isl_printer_print_str(p, s_to[0]);
	}
	isl_space_free(space);
	p = isl_printer_print_str(p, s_open_set[0]);
	data.p = p;
	data.first = 1;
	isl_union_pw_multi_aff_foreach_pw_multi_aff(upma,
					&print_pw_multi_aff_body_wrap, &data);
	p = data.p;
	p = isl_printer_print_str(p, s_close_set[0]);
	return p;
}

__isl_give isl_printer *isl_printer_print_union_pw_multi_aff(
	__isl_take isl_printer *p, __isl_keep isl_union_pw_multi_aff *upma)
{
	if (!p || !upma)
		goto error;

	if (p->output_format == ISL_FORMAT_ISL)
		return print_union_pw_multi_aff_isl(p, upma);
	isl_die(p->ctx, isl_error_unsupported, "unsupported output format",
		goto error);
error:
	isl_printer_free(p);
	return NULL;
}

/* Print dimension "pos" of data->space to "p".
 *
 * data->user is assumed to be an isl_multi_pw_aff.
 *
 * If the current dimension is an output dimension, then print
 * the corresponding piecewise affine expression.
 * Otherwise, print the name of the dimension.
 */
static __isl_give isl_printer *print_dim_mpa(__isl_take isl_printer *p,
	struct isl_print_space_data *data, unsigned pos)
{
	int i;
	int need_parens;
	isl_multi_pw_aff *mpa = data->user;
	isl_pw_aff *pa;

	if (data->type != isl_dim_out)
		return print_name(data->space, p, data->type, pos, data->latex);

	pa = mpa->p[pos];
	if (pa->n == 0)
		return isl_printer_print_str(p, "(0 : 1 = 0)");

	need_parens = pa->n != 1 || !isl_set_plain_is_universe(pa->p[0].set);
	if (need_parens)
		p = isl_printer_print_str(p, "(");
	for (i = 0; i < pa->n; ++i) {
		isl_space *space;

		if (i)
			p = isl_printer_print_str(p, "; ");
		p = print_aff_body(p, pa->p[i].aff);
		space = isl_aff_get_domain_space(pa->p[i].aff);
		p = print_disjuncts(pa->p[i].set, space, p, 0);
		isl_space_free(space);
	}
	if (need_parens)
		p = isl_printer_print_str(p, ")");

	return p;
}

/* Print "mpa" to "p" in isl format.
 */
static __isl_give isl_printer *print_multi_pw_aff_isl(__isl_take isl_printer *p,
	__isl_keep isl_multi_pw_aff *mpa)
{
	struct isl_print_space_data data = { 0 };

	if (!mpa)
		return isl_printer_free(p);

	if (isl_space_dim(mpa->space, isl_dim_param) > 0) {
		p = print_tuple(mpa->space, p, isl_dim_param, &data);
		p = isl_printer_print_str(p, " -> ");
	}
	p = isl_printer_print_str(p, "{ ");
	data.print_dim = &print_dim_mpa;
	data.user = mpa;
	p = isl_print_space(mpa->space, p, 0, &data);
	p = isl_printer_print_str(p, " }");
	return p;
}

__isl_give isl_printer *isl_printer_print_multi_pw_aff(
	__isl_take isl_printer *p, __isl_keep isl_multi_pw_aff *mpa)
{
	if (!p || !mpa)
		return isl_printer_free(p);

	if (p->output_format == ISL_FORMAT_ISL)
		return print_multi_pw_aff_isl(p, mpa);
	isl_die(p->ctx, isl_error_unsupported, "unsupported output format",
		return isl_printer_free(p));
}

/* Print dimension "pos" of data->space to "p".
 *
 * data->user is assumed to be an isl_multi_val.
 *
 * If the current dimension is an output dimension, then print
 * the corresponding value.  Otherwise, print the name of the dimension.
 */
static __isl_give isl_printer *print_dim_mv(__isl_take isl_printer *p,
	struct isl_print_space_data *data, unsigned pos)
{
	isl_multi_val *mv = data->user;

	if (data->type == isl_dim_out)
		return isl_printer_print_val(p, mv->p[pos]);
	else
		return print_name(data->space, p, data->type, pos, data->latex);
}

/* Print the isl_multi_val "mv" to "p" in isl format.
 */
static __isl_give isl_printer *print_multi_val_isl(__isl_take isl_printer *p,
	__isl_keep isl_multi_val *mv)
{
	struct isl_print_space_data data = { 0 };

	if (!mv)
		return isl_printer_free(p);

	if (isl_space_dim(mv->space, isl_dim_param) > 0) {
		p = print_tuple(mv->space, p, isl_dim_param, &data);
		p = isl_printer_print_str(p, " -> ");
	}
	p = isl_printer_print_str(p, "{ ");
	data.print_dim = &print_dim_mv;
	data.user = mv;
	p = isl_print_space(mv->space, p, 0, &data);
	p = isl_printer_print_str(p, " }");
	return p;
}

/* Print the isl_multi_val "mv" to "p".
 *
 * Currently only supported in isl format.
 */
__isl_give isl_printer *isl_printer_print_multi_val(
	__isl_take isl_printer *p, __isl_keep isl_multi_val *mv)
{
	if (!p || !mv)
		return isl_printer_free(p);

	if (p->output_format == ISL_FORMAT_ISL)
		return print_multi_val_isl(p, mv);
	isl_die(p->ctx, isl_error_unsupported, "unsupported output format",
		return isl_printer_free(p));
}

/* Print dimension "pos" of data->space to "p".
 *
 * data->user is assumed to be an isl_multi_union_pw_aff.
 *
 * The current dimension is necessarily a set dimension, so
 * we print the corresponding isl_union_pw_aff, including
 * the braces.
 */
static __isl_give isl_printer *print_union_pw_aff_dim(__isl_take isl_printer *p,
	struct isl_print_space_data *data, unsigned pos)
{
	isl_multi_union_pw_aff *mupa = data->user;
	isl_union_pw_aff *upa;

	upa = isl_multi_union_pw_aff_get_union_pw_aff(mupa, pos);
	p = print_union_pw_aff_body(p, upa);
	isl_union_pw_aff_free(upa);

	return p;
}

/* Print the isl_multi_union_pw_aff "mupa" to "p" in isl format.
 */
static __isl_give isl_printer *print_multi_union_pw_aff_isl(
	__isl_take isl_printer *p, __isl_keep isl_multi_union_pw_aff *mupa)
{
	struct isl_print_space_data data = { 0 };
	isl_space *space;

	space = isl_multi_union_pw_aff_get_space(mupa);
	if (isl_space_dim(space, isl_dim_param) > 0) {
		struct isl_print_space_data space_data = { 0 };
		p = print_tuple(space, p, isl_dim_param, &space_data);
		p = isl_printer_print_str(p, s_to[0]);
	}

	data.print_dim = &print_union_pw_aff_dim;
	data.user = mupa;

	p = isl_print_space(space, p, 0, &data);
	isl_space_free(space);

	return p;
}

/* Print the isl_multi_union_pw_aff "mupa" to "p" in isl format.
 *
 * We currently only support an isl format.
 */
__isl_give isl_printer *isl_printer_print_multi_union_pw_aff(
	__isl_take isl_printer *p, __isl_keep isl_multi_union_pw_aff *mupa)
{
	if (!p || !mupa)
		return isl_printer_free(p);

	if (p->output_format == ISL_FORMAT_ISL)
		return print_multi_union_pw_aff_isl(p, mupa);
	isl_die(isl_printer_get_ctx(p), isl_error_unsupported,
		"unsupported output format", return isl_printer_free(p));
}
