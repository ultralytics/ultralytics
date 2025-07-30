#ifndef ISL_SCHEDLUE_TREE_H
#define ISL_SCHEDLUE_TREE_H

#include <isl_schedule_band.h>
#include <isl/schedule.h>
#include <isl/set.h>
#include <isl/union_set.h>

struct isl_schedule_tree;
typedef struct isl_schedule_tree isl_schedule_tree;

ISL_DECLARE_LIST(schedule_tree)

/* A schedule (sub)tree.
 *
 * The leaves of a tree are not explicitly represented inside
 * the isl_schedule_tree, except when the tree consists of only a leaf.
 *
 * The "band" field is valid when type is isl_schedule_node_band.
 * The "context" field is valid when type is isl_schedule_node_context
 * and represents constraints on the flat product of the outer band nodes,
 * possibly introducing additional parameters.
 * The "domain" field is valid when type is isl_schedule_node_domain
 * and introduces the statement instances scheduled by the tree.
 *
 * The "contraction" and "expansion" fields are valid when type
 * is isl_schedule_node_expansion.
 * "expansion" expands the reaching domain elements to one or more
 * domain elements for the subtree.
 * "contraction" maps these elements back to the corresponding
 * reaching domain element.  It does not involve any domain constraints.
 *
 * The "extension" field is valid when the is isl_schedule_node_extension
 * maps outer schedule dimensions (the flat product of the outer band nodes)
 * to additional iteration domains.
 *
 * The "filter" field is valid when type is isl_schedule_node_filter
 * and represents the statement instances selected by the node.
 *
 * The "guard" field is valid when type is isl_schedule_node_guard
 * and represents constraints on the flat product of the outer band nodes
 * that need to be enforced by the outer nodes in the generated AST.
 *
 * The "mark" field is valid when type is isl_schedule_node_mark and
 * identifies the mark.
 *
 * The "children" field is valid for all types except
 * isl_schedule_node_leaf.  This field is NULL if there are
 * no children (except for the implicit leaves).
 *
 * anchored is set if the node or any of its descendants depends
 * on its position in the schedule tree.
 */
struct isl_schedule_tree {
	int ref;
	isl_ctx *ctx;
	int anchored;
	enum isl_schedule_node_type type;
	union {
		isl_schedule_band *band;
		isl_set *context;
		isl_union_set *domain;
		struct {
			isl_union_pw_multi_aff *contraction;
			isl_union_map *expansion;
		};
		isl_union_map *extension;
		isl_union_set *filter;
		isl_set *guard;
		isl_id *mark;
	};
	isl_schedule_tree_list *children;
};

isl_ctx *isl_schedule_tree_get_ctx(__isl_keep isl_schedule_tree *tree);
enum isl_schedule_node_type isl_schedule_tree_get_type(
	__isl_keep isl_schedule_tree *tree);

__isl_give isl_schedule_tree *isl_schedule_tree_leaf(isl_ctx *ctx);
int isl_schedule_tree_is_leaf(__isl_keep isl_schedule_tree *tree);

isl_bool isl_schedule_tree_plain_is_equal(__isl_keep isl_schedule_tree *tree1,
	__isl_keep isl_schedule_tree *tree2);

__isl_give isl_schedule_tree *isl_schedule_tree_copy(
	__isl_keep isl_schedule_tree *tree);
__isl_null isl_schedule_tree *isl_schedule_tree_free(
	__isl_take isl_schedule_tree *tree);

__isl_give isl_schedule_tree *isl_schedule_tree_from_band(
	__isl_take isl_schedule_band *band);
__isl_give isl_schedule_tree *isl_schedule_tree_from_context(
	__isl_take isl_set *context);
__isl_give isl_schedule_tree *isl_schedule_tree_from_domain(
	__isl_take isl_union_set *domain);
__isl_give isl_schedule_tree *isl_schedule_tree_from_expansion(
	__isl_take isl_union_pw_multi_aff *contraction,
	__isl_take isl_union_map *expansion);
__isl_give isl_schedule_tree *isl_schedule_tree_from_extension(
	__isl_take isl_union_map *extension);
__isl_give isl_schedule_tree *isl_schedule_tree_from_filter(
	__isl_take isl_union_set *filter);
__isl_give isl_schedule_tree *isl_schedule_tree_from_guard(
	__isl_take isl_set *guard);
__isl_give isl_schedule_tree *isl_schedule_tree_from_children(
	enum isl_schedule_node_type type,
	__isl_take isl_schedule_tree_list *list);
__isl_give isl_schedule_tree *isl_schedule_tree_from_pair(
	enum isl_schedule_node_type type, __isl_take isl_schedule_tree *tree1,
	__isl_take isl_schedule_tree *tree2);
__isl_give isl_schedule_tree *isl_schedule_tree_sequence_pair(
	__isl_take isl_schedule_tree *tree1,
	__isl_take isl_schedule_tree *tree2);
__isl_give isl_schedule_tree *isl_schedule_tree_set_pair(
	__isl_take isl_schedule_tree *tree1,
	__isl_take isl_schedule_tree *tree2);

isl_bool isl_schedule_tree_is_subtree_anchored(
	__isl_keep isl_schedule_tree *tree);

__isl_give isl_space *isl_schedule_tree_band_get_space(
	__isl_keep isl_schedule_tree *tree);
__isl_give isl_schedule_tree *isl_schedule_tree_band_intersect_domain(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *domain);
__isl_give isl_multi_union_pw_aff *isl_schedule_tree_band_get_partial_schedule(
	__isl_keep isl_schedule_tree *tree);
__isl_give isl_schedule_tree *isl_schedule_tree_band_set_partial_schedule(
	__isl_take isl_schedule_tree *tree,
	__isl_take isl_multi_union_pw_aff *schedule);
enum isl_ast_loop_type isl_schedule_tree_band_member_get_ast_loop_type(
	__isl_keep isl_schedule_tree *tree, int pos);
__isl_give isl_schedule_tree *isl_schedule_tree_band_member_set_ast_loop_type(
	__isl_take isl_schedule_tree *tree, int pos,
	enum isl_ast_loop_type type);
enum isl_ast_loop_type isl_schedule_tree_band_member_get_isolate_ast_loop_type(
	__isl_keep isl_schedule_tree *tree, int pos);
__isl_give isl_schedule_tree *
isl_schedule_tree_band_member_set_isolate_ast_loop_type(
	__isl_take isl_schedule_tree *tree, int pos,
	enum isl_ast_loop_type type);
__isl_give isl_union_set *isl_schedule_tree_band_get_ast_build_options(
	__isl_keep isl_schedule_tree *tree);
__isl_give isl_schedule_tree *isl_schedule_tree_band_set_ast_build_options(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *options);
__isl_give isl_set *isl_schedule_tree_band_get_ast_isolate_option(
	__isl_keep isl_schedule_tree *tree, int depth);
__isl_give isl_set *isl_schedule_tree_context_get_context(
	__isl_keep isl_schedule_tree *tree);
__isl_give isl_union_set *isl_schedule_tree_domain_get_domain(
	__isl_keep isl_schedule_tree *tree);
__isl_give isl_schedule_tree *isl_schedule_tree_domain_set_domain(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *domain);
__isl_give isl_union_pw_multi_aff *isl_schedule_tree_expansion_get_contraction(
	__isl_keep isl_schedule_tree *tree);
__isl_give isl_union_map *isl_schedule_tree_expansion_get_expansion(
	__isl_keep isl_schedule_tree *tree);
__isl_give isl_schedule_tree *
isl_schedule_tree_expansion_set_contraction_and_expansion(
	__isl_take isl_schedule_tree *tree,
	__isl_take isl_union_pw_multi_aff *contraction,
	__isl_take isl_union_map *expansion);
__isl_give isl_union_map *isl_schedule_tree_extension_get_extension(
	__isl_keep isl_schedule_tree *tree);
__isl_give isl_schedule_tree *isl_schedule_tree_extension_set_extension(
	__isl_take isl_schedule_tree *tree,
	__isl_take isl_union_map *extension);
__isl_give isl_union_set *isl_schedule_tree_filter_get_filter(
	__isl_keep isl_schedule_tree *tree);
__isl_give isl_schedule_tree *isl_schedule_tree_filter_set_filter(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *filter);
__isl_give isl_set *isl_schedule_tree_guard_get_guard(
	__isl_keep isl_schedule_tree *tree);
__isl_give isl_id *isl_schedule_tree_mark_get_id(
	__isl_keep isl_schedule_tree *tree);

__isl_give isl_schedule_tree *isl_schedule_tree_first_schedule_descendant(
	__isl_take isl_schedule_tree *tree, __isl_keep isl_schedule_tree *leaf);
__isl_give isl_union_map *isl_schedule_tree_get_subtree_schedule_union_map(
	__isl_keep isl_schedule_tree *tree);

unsigned isl_schedule_tree_band_n_member(__isl_keep isl_schedule_tree *tree);

isl_bool isl_schedule_tree_band_member_get_coincident(
	__isl_keep isl_schedule_tree *tree, int pos);
__isl_give isl_schedule_tree *isl_schedule_tree_band_member_set_coincident(
	__isl_take isl_schedule_tree *tree, int pos, int coincident);
isl_bool isl_schedule_tree_band_get_permutable(
	__isl_keep isl_schedule_tree *tree);
__isl_give isl_schedule_tree *isl_schedule_tree_band_set_permutable(
	__isl_take isl_schedule_tree *tree, int permutable);

int isl_schedule_tree_has_children(__isl_keep isl_schedule_tree *tree);
int isl_schedule_tree_n_children(__isl_keep isl_schedule_tree *tree);
__isl_give isl_schedule_tree *isl_schedule_tree_get_child(
	__isl_keep isl_schedule_tree *tree, int pos);

__isl_give isl_schedule_tree *isl_schedule_tree_insert_band(
	__isl_take isl_schedule_tree *tree, __isl_take isl_schedule_band *band);
__isl_give isl_schedule_tree *isl_schedule_tree_insert_context(
	__isl_take isl_schedule_tree *tree, __isl_take isl_set *context);
__isl_give isl_schedule_tree *isl_schedule_tree_insert_domain(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *domain);
__isl_give isl_schedule_tree *isl_schedule_tree_insert_expansion(
	__isl_take isl_schedule_tree *tree,
	__isl_take isl_union_pw_multi_aff *contraction,
	__isl_take isl_union_map *expansion);
__isl_give isl_schedule_tree *isl_schedule_tree_insert_extension(
	__isl_take isl_schedule_tree *tree,
	__isl_take isl_union_map *extension);
__isl_give isl_schedule_tree *isl_schedule_tree_insert_filter(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *filter);
__isl_give isl_schedule_tree *isl_schedule_tree_children_insert_filter(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *filter);
__isl_give isl_schedule_tree *isl_schedule_tree_insert_guard(
	__isl_take isl_schedule_tree *tree, __isl_take isl_set *guard);
__isl_give isl_schedule_tree *isl_schedule_tree_insert_mark(
	__isl_take isl_schedule_tree *tree, __isl_take isl_id *mark);

__isl_give isl_schedule_tree *isl_schedule_tree_append_to_leaves(
	__isl_take isl_schedule_tree *tree1,
	__isl_take isl_schedule_tree *tree2);

__isl_give isl_schedule_tree *isl_schedule_tree_band_scale(
	__isl_take isl_schedule_tree *tree, __isl_take isl_multi_val *mv);
__isl_give isl_schedule_tree *isl_schedule_tree_band_scale_down(
	__isl_take isl_schedule_tree *tree, __isl_take isl_multi_val *mv);
__isl_give isl_schedule_tree *isl_schedule_tree_band_mod(
	__isl_take isl_schedule_tree *tree, __isl_take isl_multi_val *mv);
__isl_give isl_schedule_tree *isl_schedule_tree_band_tile(
	__isl_take isl_schedule_tree *tree, __isl_take isl_multi_val *sizes);
__isl_give isl_schedule_tree *isl_schedule_tree_band_shift(
	__isl_take isl_schedule_tree *tree,
	__isl_take isl_multi_union_pw_aff *shift);
__isl_give isl_schedule_tree *isl_schedule_tree_band_split(
	__isl_take isl_schedule_tree *tree, int pos, int depth);
__isl_give isl_schedule_tree *isl_schedule_tree_band_gist(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *context);

__isl_give isl_schedule_tree *isl_schedule_tree_child(
	__isl_take isl_schedule_tree *tree, int pos);
__isl_give isl_schedule_tree *isl_schedule_tree_reset_children(
	__isl_take isl_schedule_tree *tree);
__isl_give isl_schedule_tree *isl_schedule_tree_drop_child(
	__isl_take isl_schedule_tree *tree, int pos);
__isl_give isl_schedule_tree *isl_schedule_tree_replace_child(
	__isl_take isl_schedule_tree *tree, int pos,
	__isl_take isl_schedule_tree *new_child);
__isl_give isl_schedule_tree *isl_schedule_tree_sequence_splice(
	__isl_take isl_schedule_tree *tree, int pos,
	__isl_take isl_schedule_tree *child);

__isl_give isl_schedule_tree *isl_schedule_tree_reset_user(
	__isl_take isl_schedule_tree *tree);
__isl_give isl_schedule_tree *isl_schedule_tree_align_params(
	__isl_take isl_schedule_tree *tree, __isl_take isl_space *space);
__isl_give isl_schedule_tree *isl_schedule_tree_pullback_union_pw_multi_aff(
	__isl_take isl_schedule_tree *tree,
	__isl_take isl_union_pw_multi_aff *upma);

__isl_give isl_printer *isl_printer_print_schedule_tree(
	__isl_take isl_printer *p, __isl_keep isl_schedule_tree *tree);
__isl_give isl_printer *isl_printer_print_schedule_tree_mark(
	__isl_take isl_printer *p, __isl_keep isl_schedule_tree *tree,
	int n_ancestor, int *child_pos);

#endif
