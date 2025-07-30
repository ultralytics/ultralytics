/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#ifndef ISL_TAB_H
#define ISL_TAB_H

#include <isl/lp.h>
#include <isl/map.h>
#include <isl/mat.h>
#include <isl/set.h>
#include <isl_config.h>

struct isl_tab_var {
	int index;
	unsigned is_row : 1;
	unsigned is_nonneg : 1;
	unsigned is_zero : 1;
	unsigned is_redundant : 1;
	unsigned marked : 1;
	unsigned frozen : 1;
	unsigned negated : 1;
};

enum isl_tab_undo_type {
	isl_tab_undo_bottom,
	isl_tab_undo_rational,
	isl_tab_undo_empty,
	isl_tab_undo_nonneg,
	isl_tab_undo_redundant,
	isl_tab_undo_freeze,
	isl_tab_undo_zero,
	isl_tab_undo_allocate,
	isl_tab_undo_relax,
	isl_tab_undo_unrestrict,
	isl_tab_undo_bmap_ineq,
	isl_tab_undo_bmap_eq,
	isl_tab_undo_bmap_div,
	isl_tab_undo_saved_basis,
	isl_tab_undo_drop_sample,
	isl_tab_undo_saved_samples,
	isl_tab_undo_callback,
};

struct isl_tab_callback {
	int (*run)(struct isl_tab_callback *cb);
};

union isl_tab_undo_val {
	int		var_index;
	int		*col_var;
	int		n;
	struct isl_tab_callback	*callback;
};

struct isl_tab_undo {
	enum isl_tab_undo_type	type;
	union isl_tab_undo_val	u;
	struct isl_tab_undo	*next;
};

/* The tableau maintains equality relations.
 * Each column and each row is associated to a variable or a constraint.
 * The "value" of an inequality constraint is the value of the corresponding
 * slack variable.
 * The "row_var" and "col_var" arrays map column and row indices
 * to indices in the "var" and "con" arrays.  The elements of these
 * arrays maintain extra information about the variables and the constraints.
 * Each row expresses the corresponding row variable as an affine expression
 * of the column variables.
 * The first two columns in the matrix contain the common denominator of
 * the row and the numerator of the constant term.
 * If "M" is set, then the third column represents the "big parameter".
 * The third (M = 0) or fourth (M = 1) column
 * in the matrix is called column 0 with respect to the col_var array.
 * The sample value of the tableau is the value that assigns zero
 * to all the column variables and the constant term of each affine
 * expression to the corresponding row variable.
 * The operations on the tableau maintain the property that the sample
 * value satisfies the non-negativity constraints (usually on the slack
 * variables).
 *
 * The big parameter represents an arbitrarily big (and divisible)
 * positive number.  If present, then the sign of a row is determined
 * lexicographically, with the sign of the big parameter coefficient
 * considered first.  The big parameter is only used while
 * solving PILP problems.
 *
 * The first n_dead column variables have their values fixed to zero.
 * The corresponding tab_vars are flagged "is_zero".
 * Some of the rows that have have zero coefficients in all but
 * the dead columns are also flagged "is_zero".
 *
 * The first n_redundant rows correspond to inequality constraints
 * that are always satisfied for any value satisfying the non-redundant
 * rows.  The corresponding tab_vars are flagged "is_redundant".
 * A row variable that is flagged "is_zero" is also flagged "is_redundant"
 * since the constraint has been reduced to 0 = 0 and is therefore always
 * satisfied.
 *
 * There are "n_var" variables in total.  The first "n_param" of these
 * are called parameters and the last "n_div" of these are called divs.
 * The basic tableau operations makes no distinction between different
 * kinds of variables.  These special variables are only used while
 * solving PILP problems.
 *
 * Dead columns and redundant rows are detected on the fly.
 * However, the basic operations do not ensure that all dead columns
 * or all redundant rows are detected.
 * isl_tab_detect_implicit_equalities and isl_tab_detect_redundant can be used
 * to perform an exhaustive search for dead columns and redundant rows.
 *
 * The samples matrix contains "n_sample" integer points that have at some
 * point been elements satisfying the tableau.  The first "n_outside"
 * of them no longer satisfy the tableau.  They are kept because they
 * can be reinstated during rollback when the constraint that cut them
 * out is removed.  These samples are only maintained for the context
 * tableau while solving PILP problems.
 *
 * If "preserve" is set, then we want to keep all constraints in the
 * tableau, even if they turn out to be redundant.
 */
enum isl_tab_row_sign {
	isl_tab_row_unknown = 0,
	isl_tab_row_pos,
	isl_tab_row_neg,
	isl_tab_row_any,
};
struct isl_tab {
	struct isl_mat *mat;

	unsigned n_row;
	unsigned n_col;
	unsigned n_dead;
	unsigned n_redundant;

	unsigned n_var;
	unsigned n_param;
	unsigned n_div;
	unsigned max_var;
	unsigned n_con;
	unsigned n_eq;
	unsigned max_con;
	struct isl_tab_var *var;
	struct isl_tab_var *con;
	int *row_var;	/* v >= 0 -> var v;	v < 0 -> con ~v */
	int *col_var;	/* v >= 0 -> var v;	v < 0 -> con ~v */
	enum isl_tab_row_sign *row_sign;

	struct isl_tab_undo bottom;
	struct isl_tab_undo *top;

	struct isl_vec *dual;
	struct isl_basic_map *bmap;

	unsigned n_sample;
	unsigned n_outside;
	int *sample_index;
	struct isl_mat *samples;

	int n_zero;
	int n_unbounded;
	struct isl_mat *basis;

	int (*conflict)(int con, void *user);
	void *conflict_user;

	unsigned strict_redundant : 1;
	unsigned need_undo : 1;
	unsigned preserve : 1;
	unsigned rational : 1;
	unsigned empty : 1;
	unsigned in_undo : 1;
	unsigned M : 1;
	unsigned cone : 1;
};

struct isl_tab *isl_tab_alloc(struct isl_ctx *ctx,
	unsigned n_row, unsigned n_var, unsigned M);
void isl_tab_free(struct isl_tab *tab);

isl_ctx *isl_tab_get_ctx(struct isl_tab *tab);

__isl_give struct isl_tab *isl_tab_from_basic_map(
	__isl_keep isl_basic_map *bmap, int track);
__isl_give struct isl_tab *isl_tab_from_basic_set(
	__isl_keep isl_basic_set *bset, int track);
struct isl_tab *isl_tab_from_recession_cone(struct isl_basic_set *bset,
	int parametric);
int isl_tab_cone_is_bounded(struct isl_tab *tab);
struct isl_basic_map *isl_basic_map_update_from_tab(struct isl_basic_map *bmap,
	struct isl_tab *tab);
struct isl_basic_set *isl_basic_set_update_from_tab(struct isl_basic_set *bset,
	struct isl_tab *tab);
int isl_tab_detect_implicit_equalities(struct isl_tab *tab) WARN_UNUSED;
__isl_give isl_basic_map *isl_tab_make_equalities_explicit(struct isl_tab *tab,
	__isl_take isl_basic_map *bmap);
int isl_tab_detect_redundant(struct isl_tab *tab) WARN_UNUSED;
isl_stat isl_tab_restore_redundant(struct isl_tab *tab);
#define ISL_TAB_SAVE_DUAL	(1 << 0)
enum isl_lp_result isl_tab_min(struct isl_tab *tab,
	isl_int *f, isl_int denom, isl_int *opt, isl_int *opt_denom,
	unsigned flags) WARN_UNUSED;

int isl_tab_add_ineq(struct isl_tab *tab, isl_int *ineq) WARN_UNUSED;
int isl_tab_add_eq(struct isl_tab *tab, isl_int *eq) WARN_UNUSED;
int isl_tab_add_valid_eq(struct isl_tab *tab, isl_int *eq) WARN_UNUSED;

int isl_tab_freeze_constraint(struct isl_tab *tab, int con) WARN_UNUSED;

int isl_tab_track_bmap(struct isl_tab *tab, __isl_take isl_basic_map *bmap) WARN_UNUSED;
int isl_tab_track_bset(struct isl_tab *tab, __isl_take isl_basic_set *bset) WARN_UNUSED;
__isl_keep isl_basic_set *isl_tab_peek_bset(struct isl_tab *tab);

int isl_tab_is_equality(struct isl_tab *tab, int con);
int isl_tab_is_redundant(struct isl_tab *tab, int con);

int isl_tab_sample_is_integer(struct isl_tab *tab);
struct isl_vec *isl_tab_get_sample_value(struct isl_tab *tab);

enum isl_ineq_type {
	isl_ineq_error = -1,
	isl_ineq_redundant,
	isl_ineq_separate,
	isl_ineq_cut,
	isl_ineq_adj_eq,
	isl_ineq_adj_ineq,
};

enum isl_ineq_type isl_tab_ineq_type(struct isl_tab *tab, isl_int *ineq);

struct isl_tab_undo *isl_tab_snap(struct isl_tab *tab);
int isl_tab_rollback(struct isl_tab *tab, struct isl_tab_undo *snap) WARN_UNUSED;
isl_bool isl_tab_need_undo(struct isl_tab *tab);
void isl_tab_clear_undo(struct isl_tab *tab);

int isl_tab_relax(struct isl_tab *tab, int con) WARN_UNUSED;
int isl_tab_select_facet(struct isl_tab *tab, int con) WARN_UNUSED;
int isl_tab_unrestrict(struct isl_tab *tab, int con) WARN_UNUSED;

void isl_tab_dump(__isl_keep struct isl_tab *tab);

/* Compute maximum instead of minimum. */
#define ISL_OPT_MAX		(1 << 0)
/* Compute full instead of partial optimum; also, domain argument is NULL. */
#define ISL_OPT_FULL		(1 << 1)
/* Result should be free of (unknown) quantified variables. */
#define ISL_OPT_QE		(1 << 2)
__isl_give isl_map *isl_tab_basic_map_partial_lexopt(
	__isl_take isl_basic_map *bmap, __isl_take isl_basic_set *dom,
	__isl_give isl_set **empty, unsigned flags);
__isl_give isl_pw_multi_aff *isl_tab_basic_map_partial_lexopt_pw_multi_aff(
	__isl_take isl_basic_map *bmap, __isl_take isl_basic_set *dom,
	__isl_give isl_set **empty, unsigned flags);

/* An isl_region represents a sequence of consecutive variables.
 * pos is the location (starting at 0) of the first variable in the sequence.
 */
struct isl_region {
	int pos;
	int len;
};

__isl_give isl_vec *isl_tab_basic_set_non_trivial_lexmin(
	__isl_take isl_basic_set *bset, int n_op, int n_region,
	struct isl_region *region,
	int (*conflict)(int con, void *user), void *user);
__isl_give isl_vec *isl_tab_basic_set_non_neg_lexmin(
	__isl_take isl_basic_set *bset);

struct isl_tab_lexmin;
typedef struct isl_tab_lexmin isl_tab_lexmin;

__isl_give isl_tab_lexmin *isl_tab_lexmin_from_basic_set(
	__isl_take isl_basic_set *bset);
int isl_tab_lexmin_dim(__isl_keep isl_tab_lexmin *tl);
__isl_give isl_tab_lexmin *isl_tab_lexmin_add_eq(__isl_take isl_tab_lexmin *tl,
	isl_int *eq);
__isl_give isl_vec *isl_tab_lexmin_get_solution(__isl_keep isl_tab_lexmin *tl);
__isl_null isl_tab_lexmin *isl_tab_lexmin_free(__isl_take isl_tab_lexmin *tl);

/* private */

struct isl_tab_var *isl_tab_var_from_row(struct isl_tab *tab, int i);
int isl_tab_mark_redundant(struct isl_tab *tab, int row) WARN_UNUSED;
int isl_tab_mark_rational(struct isl_tab *tab) WARN_UNUSED;
int isl_tab_mark_empty(struct isl_tab *tab) WARN_UNUSED;
struct isl_tab *isl_tab_dup(struct isl_tab *tab);
struct isl_tab *isl_tab_product(struct isl_tab *tab1, struct isl_tab *tab2);
int isl_tab_extend_cons(struct isl_tab *tab, unsigned n_new) WARN_UNUSED;
int isl_tab_allocate_con(struct isl_tab *tab) WARN_UNUSED;
int isl_tab_extend_vars(struct isl_tab *tab, unsigned n_new) WARN_UNUSED;
int isl_tab_allocate_var(struct isl_tab *tab) WARN_UNUSED;
int isl_tab_insert_var(struct isl_tab *tab, int pos) WARN_UNUSED;
int isl_tab_pivot(struct isl_tab *tab, int row, int col) WARN_UNUSED;
int isl_tab_add_row(struct isl_tab *tab, isl_int *line) WARN_UNUSED;
int isl_tab_row_is_redundant(struct isl_tab *tab, int row);
int isl_tab_min_at_most_neg_one(struct isl_tab *tab, struct isl_tab_var *var);
int isl_tab_sign_of_max(struct isl_tab *tab, int con);
int isl_tab_kill_col(struct isl_tab *tab, int col) WARN_UNUSED;

int isl_tab_push(struct isl_tab *tab, enum isl_tab_undo_type type) WARN_UNUSED;
int isl_tab_push_var(struct isl_tab *tab,
	enum isl_tab_undo_type type, struct isl_tab_var *var) WARN_UNUSED;
int isl_tab_push_basis(struct isl_tab *tab) WARN_UNUSED;

struct isl_tab *isl_tab_init_samples(struct isl_tab *tab) WARN_UNUSED;
int isl_tab_add_sample(struct isl_tab *tab,
	__isl_take isl_vec *sample) WARN_UNUSED;
struct isl_tab *isl_tab_drop_sample(struct isl_tab *tab, int s);
int isl_tab_save_samples(struct isl_tab *tab) WARN_UNUSED;

struct isl_tab *isl_tab_detect_equalities(struct isl_tab *tab,
	struct isl_tab *tab_cone) WARN_UNUSED;

int isl_tab_push_callback(struct isl_tab *tab,
	struct isl_tab_callback *callback) WARN_UNUSED;

int isl_tab_insert_div(struct isl_tab *tab, int pos, __isl_keep isl_vec *div,
	int (*add_ineq)(void *user, isl_int *), void *user);
int isl_tab_add_div(struct isl_tab *tab, __isl_keep isl_vec *div);

int isl_tab_shift_var(struct isl_tab *tab, int pos, isl_int shift) WARN_UNUSED;

#endif
