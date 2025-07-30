#ifndef ISL_LOCAL_SPACE_PRIVATE_H
#define ISL_LOCAL_SPACE_PRIVATE_H

#include <isl/mat.h>
#include <isl/set.h>
#include <isl/local_space.h>

struct isl_local_space {
	int ref;

	isl_space *dim;
	isl_mat *div;
};

uint32_t isl_local_space_get_hash(__isl_keep isl_local_space *ls);

__isl_give isl_local_space *isl_local_space_alloc(__isl_take isl_space *dim,
	unsigned n_div);
__isl_give isl_local_space *isl_local_space_alloc_div(__isl_take isl_space *dim,
	__isl_take isl_mat *div);

__isl_give isl_local_space *isl_local_space_swap_div(
	__isl_take isl_local_space *ls, int a, int b);
__isl_give isl_local_space *isl_local_space_add_div(
	__isl_take isl_local_space *ls, __isl_take isl_vec *div);

int isl_mat_cmp_div(__isl_keep isl_mat *div, int i, int j);
__isl_give isl_mat *isl_merge_divs(__isl_keep isl_mat *div1,
	__isl_keep isl_mat *div2, int *exp1, int *exp2);

unsigned isl_local_space_offset(__isl_keep isl_local_space *ls,
	enum isl_dim_type type);

__isl_give isl_local_space *isl_local_space_replace_divs(
	__isl_take isl_local_space *ls, __isl_take isl_mat *div);
isl_bool isl_local_space_div_is_known(__isl_keep isl_local_space *ls, int div);
isl_bool isl_local_space_divs_known(__isl_keep isl_local_space *ls);

__isl_give isl_local_space *isl_local_space_substitute_equalities(
	__isl_take isl_local_space *ls, __isl_take isl_basic_set *eq);

int isl_local_space_is_named_or_nested(__isl_keep isl_local_space *ls,
	enum isl_dim_type type);

isl_bool isl_local_space_has_equal_space(__isl_keep isl_local_space *ls1,
	__isl_keep isl_local_space *ls2);

__isl_give isl_local_space *isl_local_space_reset_space(
	__isl_take isl_local_space *ls, __isl_take isl_space *dim);
__isl_give isl_local_space *isl_local_space_realign(
	__isl_take isl_local_space *ls, __isl_take isl_reordering *r);

int isl_local_space_is_div_constraint(__isl_keep isl_local_space *ls,
	isl_int *constraint, unsigned div);

int *isl_local_space_get_active(__isl_keep isl_local_space *ls, isl_int *l);

__isl_give isl_local_space *isl_local_space_substitute_seq(
	__isl_take isl_local_space *ls,
	enum isl_dim_type type, unsigned pos, isl_int *subs, int subs_len,
	int first, int n);
__isl_give isl_local_space *isl_local_space_substitute(
	__isl_take isl_local_space *ls,
	enum isl_dim_type type, unsigned pos, __isl_keep isl_aff *subs);

__isl_give isl_local_space *isl_local_space_lift(
	__isl_take isl_local_space *ls);

__isl_give isl_local_space *isl_local_space_preimage_multi_aff(
	__isl_take isl_local_space *ls, __isl_take isl_multi_aff *ma);

__isl_give isl_local_space *isl_local_space_move_dims(
	__isl_take isl_local_space *ls,
	enum isl_dim_type dst_type, unsigned dst_pos,
	enum isl_dim_type src_type, unsigned src_pos, unsigned n);

int isl_local_space_cmp(__isl_keep isl_local_space *ls1,
	__isl_keep isl_local_space *ls2);

#endif
