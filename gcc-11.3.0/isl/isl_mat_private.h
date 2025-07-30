#include <isl/mat.h>
#include <isl_blk.h>

struct isl_mat {
	int ref;

	struct isl_ctx *ctx;

#define ISL_MAT_BORROWED		(1 << 0)
	unsigned flags;

	unsigned n_row;
	unsigned n_col;

	isl_int **row;

	/* actual size of the rows in memory; n_col <= max_col */
	unsigned max_col;

	struct isl_blk block;
};

uint32_t isl_mat_get_hash(__isl_keep isl_mat *mat);

__isl_give isl_mat *isl_mat_zero(isl_ctx *ctx, unsigned n_row, unsigned n_col);
__isl_give isl_mat *isl_mat_sub_alloc(__isl_keep isl_mat *mat,
	unsigned first_row, unsigned n_row, unsigned first_col, unsigned n_col);
__isl_give isl_mat *isl_mat_sub_alloc6(isl_ctx *ctx, isl_int **row,
	unsigned first_row, unsigned n_row, unsigned first_col, unsigned n_col);
void isl_mat_sub_copy(struct isl_ctx *ctx, isl_int **dst, isl_int **src,
	unsigned n_row, unsigned dst_col, unsigned src_col, unsigned n_col);
void isl_mat_sub_neg(struct isl_ctx *ctx, isl_int **dst, isl_int **src,
	unsigned n_row, unsigned dst_col, unsigned src_col, unsigned n_col);
__isl_give isl_mat *isl_mat_diag(isl_ctx *ctx, unsigned n_row, isl_int d);

__isl_give isl_mat *isl_mat_scale(__isl_take isl_mat *mat, isl_int m);
__isl_give isl_mat *isl_mat_scale_down_row(__isl_take isl_mat *mat, int row,
	isl_int m);

__isl_give isl_vec *isl_mat_get_row(__isl_keep isl_mat *mat, unsigned row);

int isl_mat_is_scaled_identity(__isl_keep isl_mat *mat);

isl_stat isl_mat_row_gcd(__isl_keep isl_mat *mat, int row, isl_int *gcd);

void isl_mat_col_mul(struct isl_mat *mat, int dst_col, isl_int f, int src_col);
void isl_mat_col_submul(struct isl_mat *mat,
			int dst_col, isl_int f, int src_col);
__isl_give isl_mat *isl_mat_col_addmul(__isl_take isl_mat *mat, int dst_col,
	isl_int f, int src_col);
__isl_give isl_mat *isl_mat_col_neg(__isl_take isl_mat *mat, int col);

int isl_mat_get_element(__isl_keep isl_mat *mat, int row, int col, isl_int *v);
__isl_give isl_mat *isl_mat_set_element(__isl_take isl_mat *mat,
	int row, int col, isl_int v);
