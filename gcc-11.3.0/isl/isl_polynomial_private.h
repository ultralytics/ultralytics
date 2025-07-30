#include <stdio.h>
#include <isl_int.h>
#include <isl/map.h>
#include <isl/mat.h>
#include <isl_morph.h>
#include <isl/polynomial.h>
#include <isl_reordering.h>

struct isl_upoly {
	int ref;
	struct isl_ctx *ctx;

	int var;
};

struct isl_upoly_cst {
	struct isl_upoly up;
	isl_int n;
	isl_int d;
};

struct isl_upoly_rec {
	struct isl_upoly up;
	int n;

	size_t size;
	struct isl_upoly *p[];
};

/* dim represents the domain space.
 */
struct isl_qpolynomial {
	int ref;

	isl_space *dim;
	struct isl_mat *div;
	struct isl_upoly *upoly;
};

struct isl_term {
	int ref;

	isl_int n;
	isl_int d;

	isl_space *dim;
	struct isl_mat *div;

	int pow[1];
};

struct isl_pw_qpolynomial_piece {
	struct isl_set *set;
	struct isl_qpolynomial *qp;
};

struct isl_pw_qpolynomial {
	int ref;

	isl_space *dim;

	int n;

	size_t size;
	struct isl_pw_qpolynomial_piece p[1];
};

/* dim represents the domain space.
 */
struct isl_qpolynomial_fold {
	int ref;

	enum isl_fold type;
	isl_space *dim;

	int n;

	size_t size;
	struct isl_qpolynomial *qp[1];
};

struct isl_pw_qpolynomial_fold_piece {
	struct isl_set *set;
	struct isl_qpolynomial_fold *fold;
};

struct isl_pw_qpolynomial_fold {
	int ref;

	enum isl_fold type;
	isl_space *dim;

	int n;

	size_t size;
	struct isl_pw_qpolynomial_fold_piece p[1];
};

void isl_term_get_num(__isl_keep isl_term *term, isl_int *n);

__isl_give struct isl_upoly *isl_upoly_zero(struct isl_ctx *ctx);
__isl_give struct isl_upoly *isl_upoly_copy(__isl_keep struct isl_upoly *up);
__isl_give struct isl_upoly *isl_upoly_cow(__isl_take struct isl_upoly *up);
__isl_give struct isl_upoly *isl_upoly_dup(__isl_keep struct isl_upoly *up);
void isl_upoly_free(__isl_take struct isl_upoly *up);
__isl_give struct isl_upoly *isl_upoly_mul(__isl_take struct isl_upoly *up1,
	__isl_take struct isl_upoly *up2);

int isl_upoly_is_cst(__isl_keep struct isl_upoly *up);
int isl_upoly_is_zero(__isl_keep struct isl_upoly *up);
int isl_upoly_is_one(__isl_keep struct isl_upoly *up);
int isl_upoly_is_negone(__isl_keep struct isl_upoly *up);
__isl_keep struct isl_upoly_cst *isl_upoly_as_cst(__isl_keep struct isl_upoly *up);
__isl_keep struct isl_upoly_rec *isl_upoly_as_rec(__isl_keep struct isl_upoly *up);

__isl_give struct isl_upoly *isl_upoly_sum(__isl_take struct isl_upoly *up1,
	__isl_take struct isl_upoly *up2);
__isl_give struct isl_upoly *isl_upoly_mul_isl_int(
	__isl_take struct isl_upoly *up, isl_int v);

__isl_give isl_qpolynomial *isl_qpolynomial_alloc(__isl_take isl_space *dim,
	unsigned n_div, __isl_take struct isl_upoly *up);
__isl_give isl_qpolynomial *isl_qpolynomial_cow(__isl_take isl_qpolynomial *qp);
__isl_give isl_qpolynomial *isl_qpolynomial_dup(__isl_keep isl_qpolynomial *qp);

__isl_give isl_qpolynomial *isl_qpolynomial_cst_on_domain(__isl_take isl_space *dim,
	isl_int v);
__isl_give isl_qpolynomial *isl_qpolynomial_rat_cst_on_domain(
	__isl_take isl_space *space, const isl_int n, const isl_int d);
__isl_give isl_qpolynomial *isl_qpolynomial_var_pow_on_domain(__isl_take isl_space *dim,
	int pos, int power);
isl_bool isl_qpolynomial_is_one(__isl_keep isl_qpolynomial *qp);
int isl_qpolynomial_is_affine(__isl_keep isl_qpolynomial *qp);
int isl_qpolynomial_is_cst(__isl_keep isl_qpolynomial *qp,
	isl_int *n, isl_int *d);

unsigned isl_qpolynomial_domain_offset(__isl_keep isl_qpolynomial *qp,
	enum isl_dim_type type);

__isl_give isl_qpolynomial *isl_qpolynomial_add_on_domain(
	__isl_keep isl_set *dom,
	__isl_take isl_qpolynomial *qp1,
	__isl_take isl_qpolynomial *qp2);

int isl_qpolynomial_plain_cmp(__isl_keep isl_qpolynomial *qp1,
	__isl_keep isl_qpolynomial *qp2);

int isl_qpolynomial_degree(__isl_keep isl_qpolynomial *poly);
__isl_give isl_qpolynomial *isl_qpolynomial_coeff(
	__isl_keep isl_qpolynomial *poly,
	enum isl_dim_type type, unsigned pos, int deg);

__isl_give isl_vec *isl_qpolynomial_extract_affine(
	__isl_keep isl_qpolynomial *qp);
__isl_give isl_qpolynomial *isl_qpolynomial_from_affine(__isl_take isl_space *dim,
	isl_int *f, isl_int denom);

__isl_give isl_pw_qpolynomial *isl_pw_qpolynomial_cow(
	__isl_take isl_pw_qpolynomial *pwqp);

__isl_give isl_pw_qpolynomial *isl_pw_qpolynomial_add_piece(
	__isl_take isl_pw_qpolynomial *pwqp,
	__isl_take isl_set *set, __isl_take isl_qpolynomial *qp);
int isl_pw_qpolynomial_is_one(__isl_keep isl_pw_qpolynomial *pwqp);

__isl_give isl_pw_qpolynomial *isl_pw_qpolynomial_project_out(
	__isl_take isl_pw_qpolynomial *pwqp,
	enum isl_dim_type type, unsigned first, unsigned n);

__isl_give isl_val *isl_qpolynomial_opt_on_domain(
	__isl_take isl_qpolynomial *qp, __isl_take isl_set *set, int max);

enum isl_fold isl_fold_type_negate(enum isl_fold type);

__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_cow(
	__isl_take isl_qpolynomial_fold *fold);
__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_dup(
	__isl_keep isl_qpolynomial_fold *fold);

__isl_give isl_pw_qpolynomial_fold *isl_pw_qpolynomial_fold_cow(
	__isl_take isl_pw_qpolynomial_fold *pwf);

__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_add_on_domain(
	__isl_keep isl_set *set,
	__isl_take isl_qpolynomial_fold *fold1,
	__isl_take isl_qpolynomial_fold *fold2);
__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_fold_on_domain(
	__isl_keep isl_set *set,
	__isl_take isl_qpolynomial_fold *fold1,
	__isl_take isl_qpolynomial_fold *fold2);

int isl_qpolynomial_fold_plain_cmp(__isl_keep isl_qpolynomial_fold *fold1,
	__isl_keep isl_qpolynomial_fold *fold2);

__isl_give isl_val *isl_qpolynomial_fold_opt_on_domain(
	__isl_take isl_qpolynomial_fold *fold, __isl_take isl_set *set, int max);

int isl_pw_qpolynomial_fold_covers(__isl_keep isl_pw_qpolynomial_fold *pwf1,
	__isl_keep isl_pw_qpolynomial_fold *pwf2);

__isl_give isl_qpolynomial *isl_qpolynomial_morph_domain(
	__isl_take isl_qpolynomial *qp, __isl_take isl_morph *morph);
__isl_give isl_pw_qpolynomial *isl_pw_qpolynomial_morph_domain(
	__isl_take isl_pw_qpolynomial *pwqp, __isl_take isl_morph *morph);
__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_morph_domain(
	__isl_take isl_qpolynomial_fold *fold, __isl_take isl_morph *morph);
__isl_give isl_pw_qpolynomial_fold *isl_pw_qpolynomial_fold_morph_domain(
	__isl_take isl_pw_qpolynomial_fold *pwf, __isl_take isl_morph *morph);

__isl_give isl_qpolynomial *isl_qpolynomial_lift(__isl_take isl_qpolynomial *qp,
	__isl_take isl_space *dim);
__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_lift(
	__isl_take isl_qpolynomial_fold *fold, __isl_take isl_space *dim);

__isl_give isl_qpolynomial *isl_qpolynomial_substitute_equalities(
	__isl_take isl_qpolynomial *qp, __isl_take isl_basic_set *eq);
__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_substitute_equalities(
	__isl_take isl_qpolynomial_fold *fold, __isl_take isl_basic_set *eq);
__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_gist(
	__isl_take isl_qpolynomial_fold *fold, __isl_take isl_set *context);

__isl_give isl_qpolynomial *isl_qpolynomial_realign_domain(
	__isl_take isl_qpolynomial *qp, __isl_take isl_reordering *r);
__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_realign_domain(
	__isl_take isl_qpolynomial_fold *fold, __isl_take isl_reordering *r);
__isl_give isl_pw_qpolynomial *isl_pw_qpolynomial_realign_domain(
	__isl_take isl_pw_qpolynomial *pwqp, __isl_take isl_reordering *r);
__isl_give isl_pw_qpolynomial_fold *isl_pw_qpolynomial_fold_realign_domain(
	__isl_take isl_pw_qpolynomial_fold *pwf, __isl_take isl_reordering *r);

__isl_give isl_pw_qpolynomial *isl_pw_qpolynomial_reset_space(
	__isl_take isl_pw_qpolynomial *pwqp, __isl_take isl_space *space);
__isl_give isl_qpolynomial *isl_qpolynomial_reset_domain_space(
	__isl_take isl_qpolynomial *qp, __isl_take isl_space *dim);
__isl_give isl_qpolynomial *isl_qpolynomial_reset_space_and_domain(
	__isl_take isl_qpolynomial *qp, __isl_take isl_space *space,
	__isl_take isl_space *domain);
__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_reset_domain_space(
	__isl_take isl_qpolynomial_fold *fold, __isl_take isl_space *dim);
__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_reset_space_and_domain(
	__isl_take isl_qpolynomial_fold *fold, __isl_take isl_space *space,
	__isl_take isl_space *domain);
__isl_give isl_pw_qpolynomial_fold *isl_pw_qpolynomial_fold_reset_domain_space(
	__isl_take isl_pw_qpolynomial_fold *pwf, __isl_take isl_space *dim);

void isl_qpolynomial_get_den(__isl_keep isl_qpolynomial *qp, isl_int *d);
__isl_give isl_qpolynomial *isl_qpolynomial_add_isl_int(
	__isl_take isl_qpolynomial *qp, isl_int v);
__isl_give isl_qpolynomial *isl_qpolynomial_mul_isl_int(
	__isl_take isl_qpolynomial *qp, isl_int v);
__isl_give isl_pw_qpolynomial *isl_pw_qpolynomial_mul_isl_int(
	__isl_take isl_pw_qpolynomial *pwqp, isl_int v);

__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_mul_isl_int(
	__isl_take isl_qpolynomial_fold *fold, isl_int v);
__isl_give isl_pw_qpolynomial_fold *isl_pw_qpolynomial_fold_mul_isl_int(
	__isl_take isl_pw_qpolynomial_fold *pwf, isl_int v);
__isl_give isl_union_pw_qpolynomial *isl_union_pw_qpolynomial_mul_isl_int(
	__isl_take isl_union_pw_qpolynomial *upwqp, isl_int v);
__isl_give isl_union_pw_qpolynomial_fold *
isl_union_pw_qpolynomial_fold_mul_isl_int(
	__isl_take isl_union_pw_qpolynomial_fold *upwf, isl_int v);
