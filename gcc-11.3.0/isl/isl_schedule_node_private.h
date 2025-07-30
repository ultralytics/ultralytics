#ifndef ISL_SCHEDLUE_NODE_PRIVATE_H
#define ISL_SCHEDLUE_NODE_PRIVATE_H

#include <isl/schedule_node.h>
#include <isl_schedule_band.h>
#include <isl_schedule_tree.h>

/* An isl_schedule_node points to a particular location in a schedule tree.
 *
 * "schedule" is the schedule that the node is pointing to.
 * "ancestors" is a list of the n ancestors of the node
 * that is being pointed to.
 * The first ancestor is the root of "schedule", while the last ancestor
 * is the parent of the specified location.
 * "child_pos" is an array of child positions of the same length as "ancestors",
 * where ancestor i (i > 0) appears in child_pos[i - 1] of ancestor i - 1 and
 * "tree" appears in child_pos[n - 1] of ancestor n - 1.
 * "tree" is the subtree at the specified location.
 *
 * Note that the same isl_schedule_tree object may appear several times
 * in a schedule tree and therefore does not uniquely identify a position
 * in the schedule tree.
 */
struct isl_schedule_node {
	int ref;

	isl_schedule *schedule;
	isl_schedule_tree_list *ancestors;
	int *child_pos;
	isl_schedule_tree *tree;
};

__isl_give isl_schedule_node *isl_schedule_node_alloc(
	__isl_take isl_schedule *schedule, __isl_take isl_schedule_tree *tree,
	__isl_take isl_schedule_tree_list *ancestors, int *child_pos);
__isl_give isl_schedule_node *isl_schedule_node_graft_tree(
	__isl_take isl_schedule_node *pos, __isl_take isl_schedule_tree *tree);

__isl_give isl_schedule_tree *isl_schedule_node_get_tree(
	__isl_keep isl_schedule_node *node);

__isl_give isl_schedule_node *isl_schedule_node_pullback_union_pw_multi_aff(
	__isl_take isl_schedule_node *node,
	__isl_take isl_union_pw_multi_aff *upma);

__isl_give isl_schedule_node *isl_schedule_node_expand(
	__isl_take isl_schedule_node *node,
	__isl_take isl_union_pw_multi_aff *contraction,
	__isl_take isl_union_set *domain,
	__isl_take isl_schedule_tree *tree);

__isl_give isl_schedule_node *isl_schedule_node_gist(
	__isl_take isl_schedule_node *node, __isl_take isl_union_set *context);

__isl_give isl_schedule_node *isl_schedule_node_domain_intersect_domain(
	__isl_take isl_schedule_node *node, __isl_take isl_union_set *domain);
__isl_give isl_schedule_node *isl_schedule_node_domain_gist_params(
	__isl_take isl_schedule_node *node, __isl_take isl_set *context);

__isl_give isl_schedule_node *isl_schedule_node_insert_expansion(
	__isl_take isl_schedule_node *node,
	__isl_take isl_union_pw_multi_aff *contraction,
	__isl_take isl_union_map *expansion);
__isl_give isl_schedule_node *isl_schedule_node_insert_extension(
	__isl_take isl_schedule_node *node,
	__isl_take isl_union_map *extension);

#endif
