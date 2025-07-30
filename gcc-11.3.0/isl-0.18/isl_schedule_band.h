#ifndef ISL_SCHEDULE_BAND_H
#define ISL_SCHEDULE_BAND_H

#include <isl/aff.h>
#include <isl/ast_type.h>
#include <isl/union_map.h>

/* Information about a band within a schedule.
 *
 * n is the number of scheduling dimensions within the band.
 * coincident is an array of length n, indicating whether a scheduling dimension
 *	satisfies the coincidence constraints in the sense that
 *	the corresponding dependence distances are zero.
 * permutable is set if the band is permutable.
 * mupa is the partial schedule corresponding to this band.  The dimension
 *	of mupa is equal to n.
 * loop_type contains the loop AST generation types for the members
 * in the band.  It may be NULL, if all members are
 * of type isl_ast_loop_default.
 * isolate_loop_type contains the loop AST generation types for the members
 * in the band for the isolated part.  It may be NULL, if all members are
 * of type isl_ast_loop_default.
 * ast_build_options are the remaining AST build options associated
 * to the band.
 * anchored is set if the node depends on its position in the schedule tree.
 *	In particular, it is set if the AST build options include
 *	an isolate option.
 */
struct isl_schedule_band {
	int ref;

	int n;
	int *coincident;
	int permutable;

	isl_multi_union_pw_aff *mupa;

	int anchored;
	isl_union_set *ast_build_options;
	enum isl_ast_loop_type *loop_type;
	enum isl_ast_loop_type *isolate_loop_type;
};
typedef struct isl_schedule_band isl_schedule_band;

__isl_give isl_schedule_band *isl_schedule_band_from_multi_union_pw_aff(
	__isl_take isl_multi_union_pw_aff *mupa);
__isl_give isl_schedule_band *isl_schedule_band_copy(
	__isl_keep isl_schedule_band *band);
__isl_null isl_schedule_band *isl_schedule_band_free(
	__isl_take isl_schedule_band *band);

isl_ctx *isl_schedule_band_get_ctx(__isl_keep isl_schedule_band *band);

isl_bool isl_schedule_band_plain_is_equal(__isl_keep isl_schedule_band *band1,
	__isl_keep isl_schedule_band *band2);

int isl_schedule_band_is_anchored(__isl_keep isl_schedule_band *band);

__isl_give isl_space *isl_schedule_band_get_space(
	__isl_keep isl_schedule_band *band);
__isl_give isl_schedule_band *isl_schedule_band_intersect_domain(
	__isl_take isl_schedule_band *band, __isl_take isl_union_set *domain);
__isl_give isl_multi_union_pw_aff *isl_schedule_band_get_partial_schedule(
	__isl_keep isl_schedule_band *band);
__isl_give isl_schedule_band *isl_schedule_band_set_partial_schedule(
	__isl_take isl_schedule_band *band,
	__isl_take isl_multi_union_pw_aff *schedule);
enum isl_ast_loop_type isl_schedule_band_member_get_ast_loop_type(
	__isl_keep isl_schedule_band *band, int pos);
__isl_give isl_schedule_band *isl_schedule_band_member_set_ast_loop_type(
	__isl_take isl_schedule_band *band, int pos,
	enum isl_ast_loop_type type);
enum isl_ast_loop_type isl_schedule_band_member_get_isolate_ast_loop_type(
	__isl_keep isl_schedule_band *band, int pos);
__isl_give isl_schedule_band *
isl_schedule_band_member_set_isolate_ast_loop_type(
	__isl_take isl_schedule_band *band, int pos,
	enum isl_ast_loop_type type);
__isl_give isl_union_set *isl_schedule_band_get_ast_build_options(
	__isl_keep isl_schedule_band *band);
__isl_give isl_schedule_band *isl_schedule_band_set_ast_build_options(
	__isl_take isl_schedule_band *band, __isl_take isl_union_set *options);
__isl_give isl_set *isl_schedule_band_get_ast_isolate_option(
	__isl_keep isl_schedule_band *band, int depth);
__isl_give isl_schedule_band *isl_schedule_band_replace_ast_build_option(
	__isl_take isl_schedule_band *band, __isl_take isl_set *drop,
	__isl_take isl_set *add);

int isl_schedule_band_n_member(__isl_keep isl_schedule_band *band);
isl_bool isl_schedule_band_member_get_coincident(
	__isl_keep isl_schedule_band *band, int pos);
__isl_give isl_schedule_band *isl_schedule_band_member_set_coincident(
	__isl_take isl_schedule_band *band, int pos, int coincident);
isl_bool isl_schedule_band_get_permutable(__isl_keep isl_schedule_band *band);
__isl_give isl_schedule_band *isl_schedule_band_set_permutable(
	__isl_take isl_schedule_band *band, int permutable);

__isl_give isl_schedule_band *isl_schedule_band_scale(
	__isl_take isl_schedule_band *band, __isl_take isl_multi_val *mv);
__isl_give isl_schedule_band *isl_schedule_band_scale_down(
	__isl_take isl_schedule_band *band, __isl_take isl_multi_val *mv);
__isl_give isl_schedule_band *isl_schedule_band_mod(
	__isl_take isl_schedule_band *band, __isl_take isl_multi_val *mv);
__isl_give isl_schedule_band *isl_schedule_band_tile(
	__isl_take isl_schedule_band *band, __isl_take isl_multi_val *sizes);
__isl_give isl_schedule_band *isl_schedule_band_point(
	__isl_take isl_schedule_band *band, __isl_keep isl_schedule_band *tile,
	__isl_take isl_multi_val *sizes);
__isl_give isl_schedule_band *isl_schedule_band_shift(
	__isl_take isl_schedule_band *band,
	__isl_take isl_multi_union_pw_aff *shift);
__isl_give isl_schedule_band *isl_schedule_band_drop(
	__isl_take isl_schedule_band *band, int pos, int n);
__isl_give isl_schedule_band *isl_schedule_band_gist(
	__isl_take isl_schedule_band *band, __isl_take isl_union_set *context);

__isl_give isl_schedule_band *isl_schedule_band_reset_user(
	__isl_take isl_schedule_band *band);
__isl_give isl_schedule_band *isl_schedule_band_align_params(
	__isl_take isl_schedule_band *band, __isl_take isl_space *space);
__isl_give isl_schedule_band *isl_schedule_band_pullback_union_pw_multi_aff(
	__isl_take isl_schedule_band *band,
	__isl_take isl_union_pw_multi_aff *upma);

#endif
