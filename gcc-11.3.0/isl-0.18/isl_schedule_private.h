#ifndef ISL_SCHEDLUE_PRIVATE_H
#define ISL_SCHEDLUE_PRIVATE_H

#include <isl/aff.h>
#include <isl/schedule.h>
#include <isl_schedule_tree.h>

/* A complete schedule tree.
 *
 * band_forest points to a band forest representation of the schedule
 * and may be NULL if the forest hasn't been created yet.
 *
 * "root" is the root of the schedule tree and may be NULL if we
 * have created a band forest corresponding to the schedule.
 *
 * "leaf" may be used to represent a leaf of the schedule.
 * It should not appear as a child to any other isl_schedule_tree objects,
 * but an isl_schedule_node may have "leaf" as its tree if it refers to
 * a leaf of this schedule tree.
 */
struct isl_schedule {
	int ref;

	isl_band_list *band_forest;
	isl_schedule_tree *root;

	struct isl_schedule_tree *leaf;
};

__isl_give isl_schedule *isl_schedule_from_schedule_tree(isl_ctx *ctx,
	__isl_take isl_schedule_tree *tree);
__isl_give isl_schedule *isl_schedule_set_root(
	__isl_take isl_schedule *schedule, __isl_take isl_schedule_tree *tree);
__isl_give isl_space *isl_schedule_get_space(
	__isl_keep isl_schedule *schedule);
__isl_give isl_union_set *isl_schedule_get_domain(
	__isl_keep isl_schedule *schedule);
__isl_keep isl_schedule_tree *isl_schedule_peek_leaf(
	__isl_keep isl_schedule *schedule);

#endif
