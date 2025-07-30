#ifndef ISL_AST_GRAFT_PRIVATE_H
#define ISL_AST_GRAFT_PRIVATE_H

#include <isl/ast.h>
#include <isl/set.h>
#include <isl/list.h>
#include <isl/printer.h>

struct isl_ast_graft;
typedef struct isl_ast_graft isl_ast_graft;

/* Representation of part of an AST ("node") with some additional polyhedral
 * information about the tree.
 *
 * "guard" contains conditions that should still be enforced by
 * some ancestor of the current tree.  In particular, the already
 * generated tree assumes that these conditions hold, but may not
 * have enforced them itself.
 * The guard should not contain any unknown divs as it will be used
 * to generate an if condition.
 *
 * "enforced" expresses constraints that are already enforced by the for
 * nodes in the current tree and that therefore do not need to be enforced
 * by any ancestor.
 * The constraints only involve outer loop iterators.
 */
struct isl_ast_graft {
	int ref;

	isl_ast_node *node;

	isl_set *guard;
	isl_basic_set *enforced;
};

ISL_DECLARE_LIST(ast_graft)

#undef EL
#define EL isl_ast_graft

#include <isl_list_templ.h>

isl_ctx *isl_ast_graft_get_ctx(__isl_keep isl_ast_graft *graft);

__isl_give isl_ast_graft *isl_ast_graft_alloc(
	__isl_take isl_ast_node *node, __isl_keep isl_ast_build *build);
__isl_give isl_ast_graft *isl_ast_graft_alloc_from_children(
	__isl_take isl_ast_graft_list *list, __isl_take isl_set *guard,
	__isl_take isl_basic_set *enforced, __isl_keep isl_ast_build *build,
	__isl_keep isl_ast_build *sub_build);
__isl_give isl_ast_graft_list *isl_ast_graft_list_fuse(
	__isl_take isl_ast_graft_list *children,
	__isl_keep isl_ast_build *build);
__isl_give isl_ast_graft *isl_ast_graft_alloc_domain(
	__isl_take isl_map *schedule, __isl_keep isl_ast_build *build);
void *isl_ast_graft_free(__isl_take isl_ast_graft *graft);
__isl_give isl_ast_graft_list *isl_ast_graft_list_sort_guard(
	__isl_take isl_ast_graft_list *list);

__isl_give isl_ast_graft_list *isl_ast_graft_list_merge(
	__isl_take isl_ast_graft_list *list1,
	__isl_take isl_ast_graft_list *list2,
	__isl_keep isl_ast_build *build);

__isl_give isl_ast_node *isl_ast_graft_get_node(
	__isl_keep isl_ast_graft *graft);
__isl_give isl_basic_set *isl_ast_graft_get_enforced(
	__isl_keep isl_ast_graft *graft);
__isl_give isl_set *isl_ast_graft_get_guard(__isl_keep isl_ast_graft *graft);

__isl_give isl_ast_graft *isl_ast_graft_insert_for(
	__isl_take isl_ast_graft *graft, __isl_take isl_ast_node *node);
__isl_give isl_ast_graft *isl_ast_graft_add_guard(
	__isl_take isl_ast_graft *graft,
	__isl_take isl_set *guard, __isl_keep isl_ast_build *build);
__isl_give isl_ast_graft *isl_ast_graft_enforce(
	__isl_take isl_ast_graft *graft, __isl_take isl_basic_set *enforced);

__isl_give isl_ast_graft *isl_ast_graft_insert_mark(
	__isl_take isl_ast_graft *graft, __isl_take isl_id *mark);

__isl_give isl_ast_graft_list *isl_ast_graft_list_unembed(
	__isl_take isl_ast_graft_list *list, int product);
__isl_give isl_ast_graft_list *isl_ast_graft_list_preimage_multi_aff(
	__isl_take isl_ast_graft_list *list, __isl_take isl_multi_aff *ma);
__isl_give isl_ast_graft_list *isl_ast_graft_list_insert_pending_guard_nodes(
	__isl_take isl_ast_graft_list *list, __isl_keep isl_ast_build *build);

__isl_give isl_ast_node *isl_ast_node_from_graft_list(
	__isl_take isl_ast_graft_list *list, __isl_keep isl_ast_build *build);

__isl_give isl_basic_set *isl_ast_graft_list_extract_shared_enforced(
	__isl_keep isl_ast_graft_list *list, __isl_keep isl_ast_build *build);
__isl_give isl_set *isl_ast_graft_list_extract_hoistable_guard(
	__isl_keep isl_ast_graft_list *list, __isl_keep isl_ast_build *build);
__isl_give isl_ast_graft_list *isl_ast_graft_list_gist_guards(
	__isl_take isl_ast_graft_list *list, __isl_take isl_set *context);

__isl_give isl_printer *isl_printer_print_ast_graft(__isl_take isl_printer *p,
	__isl_keep isl_ast_graft *graft);

#endif
