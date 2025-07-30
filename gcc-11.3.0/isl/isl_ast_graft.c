/*
 * Copyright 2012      Ecole Normale Superieure
 * Copyright 2014      INRIA Rocquencourt
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 * and Inria Paris - Rocquencourt, Domaine de Voluceau - Rocquencourt,
 * B.P. 105 - 78153 Le Chesnay, France
 */

#include <isl_ast_private.h>
#include <isl_ast_build_expr.h>
#include <isl_ast_build_private.h>
#include <isl_ast_graft_private.h>

static __isl_give isl_ast_graft *isl_ast_graft_copy(
	__isl_keep isl_ast_graft *graft);

#undef BASE
#define BASE ast_graft

#include <isl_list_templ.c>

#undef BASE
#define BASE ast_graft
#include <print_templ.c>

isl_ctx *isl_ast_graft_get_ctx(__isl_keep isl_ast_graft *graft)
{
	if (!graft)
		return NULL;
	return isl_basic_set_get_ctx(graft->enforced);
}

__isl_give isl_ast_node *isl_ast_graft_get_node(
	__isl_keep isl_ast_graft *graft)
{
	return graft ? isl_ast_node_copy(graft->node) : NULL;
}

/* Create a graft for "node" with no guards and no enforced conditions.
 */
__isl_give isl_ast_graft *isl_ast_graft_alloc(
	__isl_take isl_ast_node *node, __isl_keep isl_ast_build *build)
{
	isl_ctx *ctx;
	isl_space *space;
	isl_ast_graft *graft;

	if (!node)
		return NULL;

	ctx = isl_ast_node_get_ctx(node);
	graft = isl_calloc_type(ctx, isl_ast_graft);
	if (!graft)
		goto error;

	space = isl_ast_build_get_space(build, 1);

	graft->ref = 1;
	graft->node = node;
	graft->guard = isl_set_universe(isl_space_copy(space));
	graft->enforced = isl_basic_set_universe(space);

	if (!graft->guard || !graft->enforced)
		return isl_ast_graft_free(graft);

	return graft;
error:
	isl_ast_node_free(node);
	return NULL;
}

/* Create a graft with no guards and no enforced conditions
 * encapsulating a call to the domain element specified by "executed".
 * "executed" is assumed to be single-valued.
 */
__isl_give isl_ast_graft *isl_ast_graft_alloc_domain(
	__isl_take isl_map *executed, __isl_keep isl_ast_build *build)
{
	isl_ast_node *node;

	node = isl_ast_build_call_from_executed(build, executed);

	return isl_ast_graft_alloc(node, build);
}

static __isl_give isl_ast_graft *isl_ast_graft_copy(
	__isl_keep isl_ast_graft *graft)
{
	if (!graft)
		return NULL;

	graft->ref++;
	return graft;
}

/* Do all the grafts in "list" have the same guard and is this guard
 * independent of the current depth?
 */
static int equal_independent_guards(__isl_keep isl_ast_graft_list *list,
	__isl_keep isl_ast_build *build)
{
	int i, n;
	int depth;
	isl_ast_graft *graft_0;
	int equal = 1;
	int skip;

	graft_0 = isl_ast_graft_list_get_ast_graft(list, 0);
	if (!graft_0)
		return -1;

	depth = isl_ast_build_get_depth(build);
	if (isl_set_dim(graft_0->guard, isl_dim_set) <= depth)
		skip = 0;
	else
		skip = isl_set_involves_dims(graft_0->guard,
						isl_dim_set, depth, 1);
	if (skip < 0 || skip) {
		isl_ast_graft_free(graft_0);
		return skip < 0 ? -1 : 0;
	}

	n = isl_ast_graft_list_n_ast_graft(list);
	for (i = 1; i < n; ++i) {
		isl_ast_graft *graft;
		graft = isl_ast_graft_list_get_ast_graft(list, i);
		if (!graft)
			equal = -1;
		else
			equal = isl_set_is_equal(graft_0->guard, graft->guard);
		isl_ast_graft_free(graft);
		if (equal < 0 || !equal)
			break;
	}

	isl_ast_graft_free(graft_0);

	return equal;
}

/* Hoist "guard" out of the current level (given by "build").
 *
 * In particular, eliminate the dimension corresponding to the current depth.
 */
static __isl_give isl_set *hoist_guard(__isl_take isl_set *guard,
	__isl_keep isl_ast_build *build)
{
	int depth;

	depth = isl_ast_build_get_depth(build);
	if (depth < isl_set_dim(guard, isl_dim_set)) {
		guard = isl_set_remove_divs_involving_dims(guard,
						isl_dim_set, depth, 1);
		guard = isl_set_eliminate(guard, isl_dim_set, depth, 1);
		guard = isl_set_compute_divs(guard);
	}

	return guard;
}

/* Extract a common guard from the grafts in "list" that can be hoisted
 * out of the current level.  If no such guard can be found, then return
 * a universal set.
 *
 * If all the grafts in the list have the same guard and if this guard
 * is independent of the current level, then it can be hoisted out.
 * If there is only one graft in the list and if its guard
 * depends on the current level, then we eliminate this level and
 * return the result.
 *
 * Otherwise, we return the unshifted simple hull of the guards.
 * In order to be able to hoist as many constraints as possible,
 * but at the same time avoid hoisting constraints that did not
 * appear in the guards in the first place, we intersect the guards
 * with all the information that is available (i.e., the domain
 * from the build and the enforced constraints of the graft) and
 * compute the unshifted hull of the result using only constraints
 * from the original guards.
 * In particular, intersecting the guards with other known information
 * allows us to hoist guards that are only explicit is some of
 * the grafts and implicit in the others.
 *
 * The special case for equal guards is needed in case those guards
 * are non-convex.  Taking the simple hull would remove information
 * and would not allow for these guards to be hoisted completely.
 */
__isl_give isl_set *isl_ast_graft_list_extract_hoistable_guard(
	__isl_keep isl_ast_graft_list *list, __isl_keep isl_ast_build *build)
{
	int i, n;
	int equal;
	isl_ctx *ctx;
	isl_set *guard;
	isl_set_list *set_list;
	isl_basic_set *hull;

	if (!list || !build)
		return NULL;

	n = isl_ast_graft_list_n_ast_graft(list);
	if (n == 0)
		return isl_set_universe(isl_ast_build_get_space(build, 1));

	equal = equal_independent_guards(list, build);
	if (equal < 0)
		return NULL;

	if (equal || n == 1) {
		isl_ast_graft *graft_0;

		graft_0 = isl_ast_graft_list_get_ast_graft(list, 0);
		if (!graft_0)
			return NULL;
		guard = isl_set_copy(graft_0->guard);
		if (!equal)
			guard = hoist_guard(guard, build);
		isl_ast_graft_free(graft_0);
		return guard;
	}

	ctx = isl_ast_build_get_ctx(build);
	set_list = isl_set_list_alloc(ctx, n);
	guard = isl_set_empty(isl_ast_build_get_space(build, 1));
	for (i = 0; i < n; ++i) {
		isl_ast_graft *graft;
		isl_basic_set *enforced;
		isl_set *guard_i;

		graft = isl_ast_graft_list_get_ast_graft(list, i);
		enforced = isl_ast_graft_get_enforced(graft);
		guard_i = isl_set_copy(graft->guard);
		isl_ast_graft_free(graft);
		set_list = isl_set_list_add(set_list, isl_set_copy(guard_i));
		guard_i = isl_set_intersect(guard_i,
					    isl_set_from_basic_set(enforced));
		guard_i = isl_set_intersect(guard_i,
					    isl_ast_build_get_domain(build));
		guard = isl_set_union(guard, guard_i);
	}
	hull = isl_set_unshifted_simple_hull_from_set_list(guard, set_list);
	guard = isl_set_from_basic_set(hull);
	return hoist_guard(guard, build);
}

/* Internal data structure used inside insert_if.
 *
 * list is the list of guarded nodes created by each call to insert_if.
 * node is the original node that is guarded by insert_if.
 * build is the build in which the AST is constructed.
 */
struct isl_insert_if_data {
	isl_ast_node_list *list;
	isl_ast_node *node;
	isl_ast_build *build;
};

static isl_stat insert_if(__isl_take isl_basic_set *bset, void *user);

/* Insert an if node around "node" testing the condition encoded
 * in guard "guard".
 *
 * If the user does not want any disjunctions in the if conditions
 * and if "guard" does involve a disjunction, then we make the different
 * disjuncts disjoint and insert an if node corresponding to each disjunct
 * around a copy of "node".  The result is then a block node containing
 * this sequence of guarded copies of "node".
 */
static __isl_give isl_ast_node *ast_node_insert_if(
	__isl_take isl_ast_node *node, __isl_take isl_set *guard,
	__isl_keep isl_ast_build *build)
{
	struct isl_insert_if_data data;
	isl_ctx *ctx;

	ctx = isl_ast_build_get_ctx(build);
	if (isl_options_get_ast_build_allow_or(ctx) ||
	    isl_set_n_basic_set(guard) <= 1) {
		isl_ast_node *if_node;
		isl_ast_expr *expr;

		expr = isl_ast_build_expr_from_set_internal(build, guard);

		if_node = isl_ast_node_alloc_if(expr);
		return isl_ast_node_if_set_then(if_node, node);
	}

	guard = isl_set_make_disjoint(guard);

	data.list = isl_ast_node_list_alloc(ctx, 0);
	data.node = node;
	data.build = build;
	if (isl_set_foreach_basic_set(guard, &insert_if, &data) < 0)
		data.list = isl_ast_node_list_free(data.list);

	isl_set_free(guard);
	isl_ast_node_free(data.node);
	return isl_ast_node_alloc_block(data.list);
}

/* Insert an if node around a copy of "data->node" testing the condition
 * encoded in guard "bset" and add the result to data->list.
 */
static isl_stat insert_if(__isl_take isl_basic_set *bset, void *user)
{
	struct isl_insert_if_data *data = user;
	isl_ast_node *node;
	isl_set *set;

	set = isl_set_from_basic_set(bset);
	node = isl_ast_node_copy(data->node);
	node = ast_node_insert_if(node, set, data->build);
	data->list = isl_ast_node_list_add(data->list, node);

	return isl_stat_ok;
}

/* Insert an if node around graft->node testing the condition encoded
 * in guard "guard", assuming guard involves any conditions.
 */
static __isl_give isl_ast_graft *insert_if_node(
	__isl_take isl_ast_graft *graft, __isl_take isl_set *guard,
	__isl_keep isl_ast_build *build)
{
	int univ;

	if (!graft)
		goto error;

	univ = isl_set_plain_is_universe(guard);
	if (univ < 0)
		goto error;
	if (univ) {
		isl_set_free(guard);
		return graft;
	}

	build = isl_ast_build_copy(build);
	graft->node = ast_node_insert_if(graft->node, guard, build);
	isl_ast_build_free(build);

	if (!graft->node)
		return isl_ast_graft_free(graft);

	return graft;
error:
	isl_set_free(guard);
	return isl_ast_graft_free(graft);
}

/* Insert an if node around graft->node testing the condition encoded
 * in graft->guard, assuming graft->guard involves any conditions.
 */
static __isl_give isl_ast_graft *insert_pending_guard_node(
	__isl_take isl_ast_graft *graft, __isl_keep isl_ast_build *build)
{
	if (!graft)
		return NULL;

	return insert_if_node(graft, isl_set_copy(graft->guard), build);
}

/* Replace graft->enforced by "enforced".
 */
__isl_give isl_ast_graft *isl_ast_graft_set_enforced(
	__isl_take isl_ast_graft *graft, __isl_take isl_basic_set *enforced)
{
	if (!graft || !enforced)
		goto error;

	isl_basic_set_free(graft->enforced);
	graft->enforced = enforced;

	return graft;
error:
	isl_basic_set_free(enforced);
	return isl_ast_graft_free(graft);
}

/* Update "enforced" such that it only involves constraints that are
 * also enforced by "graft".
 */
static __isl_give isl_basic_set *update_enforced(
	__isl_take isl_basic_set *enforced, __isl_keep isl_ast_graft *graft,
	int depth)
{
	isl_basic_set *enforced_g;

	enforced_g = isl_ast_graft_get_enforced(graft);
	if (depth < isl_basic_set_dim(enforced_g, isl_dim_set))
		enforced_g = isl_basic_set_eliminate(enforced_g,
							isl_dim_set, depth, 1);
	enforced_g = isl_basic_set_remove_unknown_divs(enforced_g);
	enforced_g = isl_basic_set_align_params(enforced_g,
				isl_basic_set_get_space(enforced));
	enforced = isl_basic_set_align_params(enforced,
				isl_basic_set_get_space(enforced_g));
	enforced = isl_set_simple_hull(isl_basic_set_union(enforced,
						enforced_g));

	return enforced;
}

/* Extend the node at *body with node.
 *
 * If body points to the else branch, then *body may still be NULL.
 * If so, we simply attach node to this else branch.
 * Otherwise, we attach a list containing the statements already
 * attached at *body followed by node.
 */
static void extend_body(__isl_keep isl_ast_node **body,
	__isl_take isl_ast_node *node)
{
	isl_ast_node_list *list;

	if  (!*body) {
		*body = node;
		return;
	}

	if ((*body)->type == isl_ast_node_block) {
		list = isl_ast_node_block_get_children(*body);
		isl_ast_node_free(*body);
	} else
		list = isl_ast_node_list_from_ast_node(*body);
	list = isl_ast_node_list_add(list, node);
	*body = isl_ast_node_alloc_block(list);
}

/* Merge "graft" into the last graft of "list".
 * body points to the then or else branch of an if node in that last graft.
 *
 * We attach graft->node to this branch and update the enforced
 * set of the last graft of "list" to take into account the enforced
 * set of "graft".
 */
static __isl_give isl_ast_graft_list *graft_extend_body(
	__isl_take isl_ast_graft_list *list,
	__isl_keep isl_ast_node **body, __isl_take isl_ast_graft *graft,
	__isl_keep isl_ast_build *build)
{
	int n;
	int depth;
	isl_ast_graft *last;
	isl_space *space;
	isl_basic_set *enforced;

	if (!list || !graft)
		goto error;
	extend_body(body, isl_ast_node_copy(graft->node));
	if (!*body)
		goto error;

	n = isl_ast_graft_list_n_ast_graft(list);
	last = isl_ast_graft_list_get_ast_graft(list, n - 1);

	depth = isl_ast_build_get_depth(build);
	space = isl_ast_build_get_space(build, 1);
	enforced = isl_basic_set_empty(space);
	enforced = update_enforced(enforced, last, depth);
	enforced = update_enforced(enforced, graft, depth);
	last = isl_ast_graft_set_enforced(last, enforced);

	list = isl_ast_graft_list_set_ast_graft(list, n - 1, last);
	isl_ast_graft_free(graft);
	return list;
error:
	isl_ast_graft_free(graft);
	return isl_ast_graft_list_free(list);
}

/* Merge "graft" into the last graft of "list", attaching graft->node
 * to the then branch of "last_if".
 */
static __isl_give isl_ast_graft_list *extend_then(
	__isl_take isl_ast_graft_list *list,
	__isl_keep isl_ast_node *last_if, __isl_take isl_ast_graft *graft,
	__isl_keep isl_ast_build *build)
{
	return graft_extend_body(list, &last_if->u.i.then, graft, build);
}

/* Merge "graft" into the last graft of "list", attaching graft->node
 * to the else branch of "last_if".
 */
static __isl_give isl_ast_graft_list *extend_else(
	__isl_take isl_ast_graft_list *list,
	__isl_keep isl_ast_node *last_if, __isl_take isl_ast_graft *graft,
	__isl_keep isl_ast_build *build)
{
	return graft_extend_body(list, &last_if->u.i.else_node, graft, build);
}

/* This data structure keeps track of an if node.
 *
 * "node" is the actual if-node
 * "guard" is the original, non-simplified guard of the node
 * "complement" is the complement of "guard" in the context of outer if nodes
 */
struct isl_if_node {
	isl_ast_node *node;
	isl_set *guard;
	isl_set *complement;
};

/* Given a list of "n" if nodes, clear those starting at "first"
 * and return "first" (i.e., the updated size of the array).
 */
static int clear_if_nodes(struct isl_if_node *if_node, int first, int n)
{
	int i;

	for (i = first; i < n; ++i) {
		isl_set_free(if_node[i].guard);
		isl_set_free(if_node[i].complement);
	}

	return first;
}

/* For each graft in "list",
 * insert an if node around graft->node testing the condition encoded
 * in graft->guard, assuming graft->guard involves any conditions.
 *
 * We keep track of a list of generated if nodes that can be extended
 * without changing the order of the elements in "list".
 * If the guard of a graft is a subset of either the guard or its complement
 * of one of those if nodes, then the node
 * of the new graft is inserted into the then or else branch of the last graft
 * and the current graft is discarded.
 * The guard of the node is then simplified based on the conditions
 * enforced at that then or else branch.
 * Otherwise, the current graft is appended to the list.
 *
 * We only construct else branches if allowed by the user.
 */
static __isl_give isl_ast_graft_list *insert_pending_guard_nodes(
	__isl_take isl_ast_graft_list *list,
	__isl_keep isl_ast_build *build)
{
	int i, j, n, n_if;
	int allow_else;
	isl_ctx *ctx;
	isl_ast_graft_list *res;
	struct isl_if_node *if_node = NULL;

	if (!build || !list)
		return isl_ast_graft_list_free(list);

	ctx = isl_ast_build_get_ctx(build);
	n = isl_ast_graft_list_n_ast_graft(list);

	allow_else = isl_options_get_ast_build_allow_else(ctx);

	n_if = 0;
	if (n > 1) {
		if_node = isl_alloc_array(ctx, struct isl_if_node, n - 1);
		if (!if_node)
			return isl_ast_graft_list_free(list);
	}

	res = isl_ast_graft_list_alloc(ctx, n);

	for (i = 0; i < n; ++i) {
		isl_set *guard;
		isl_ast_graft *graft;
		int subset, found_then, found_else;
		isl_ast_node *node;

		graft = isl_ast_graft_list_get_ast_graft(list, i);
		if (!graft)
			break;
		subset = 0;
		found_then = found_else = -1;
		if (n_if > 0) {
			isl_set *test;
			test = isl_set_copy(graft->guard);
			test = isl_set_intersect(test,
						isl_set_copy(build->domain));
			for (j = n_if - 1; j >= 0; --j) {
				subset = isl_set_is_subset(test,
							if_node[j].guard);
				if (subset < 0 || subset) {
					found_then = j;
					break;
				}
				if (!allow_else)
					continue;
				subset = isl_set_is_subset(test,
							if_node[j].complement);
				if (subset < 0 || subset) {
					found_else = j;
					break;
				}
			}
			n_if = clear_if_nodes(if_node, j + 1, n_if);
			isl_set_free(test);
		}
		if (subset < 0) {
			graft = isl_ast_graft_free(graft);
			break;
		}

		guard = isl_set_copy(graft->guard);
		if (found_then >= 0)
			graft->guard = isl_set_gist(graft->guard,
				isl_set_copy(if_node[found_then].guard));
		else if (found_else >= 0)
			graft->guard = isl_set_gist(graft->guard,
				isl_set_copy(if_node[found_else].complement));

		node = graft->node;
		if (!graft->guard)
			graft = isl_ast_graft_free(graft);
		graft = insert_pending_guard_node(graft, build);
		if (graft && graft->node != node && i != n - 1) {
			isl_set *set;
			if_node[n_if].node = graft->node;
			if_node[n_if].guard = guard;
			if (found_then >= 0)
				set = if_node[found_then].guard;
			else if (found_else >= 0)
				set = if_node[found_else].complement;
			else
				set = build->domain;
			set = isl_set_copy(set);
			set = isl_set_subtract(set, isl_set_copy(guard));
			if_node[n_if].complement = set;
			n_if++;
		} else
			isl_set_free(guard);
		if (!graft)
			break;

		if (found_then >= 0)
			res = extend_then(res, if_node[found_then].node,
						graft, build);
		else if (found_else >= 0)
			res = extend_else(res, if_node[found_else].node,
						graft, build);
		else
			res = isl_ast_graft_list_add(res, graft);
	}
	if (i < n)
		res = isl_ast_graft_list_free(res);

	isl_ast_graft_list_free(list);
	clear_if_nodes(if_node, 0, n_if);
	free(if_node);
	return res;
}

/* For each graft in "list",
 * insert an if node around graft->node testing the condition encoded
 * in graft->guard, assuming graft->guard involves any conditions.
 * Subsequently remove the guards from the grafts.
 */
__isl_give isl_ast_graft_list *isl_ast_graft_list_insert_pending_guard_nodes(
	__isl_take isl_ast_graft_list *list, __isl_keep isl_ast_build *build)
{
	int i, n;
	isl_set *universe;

	list = insert_pending_guard_nodes(list, build);
	if (!list)
		return NULL;

	universe = isl_set_universe(isl_ast_build_get_space(build, 1));
	n = isl_ast_graft_list_n_ast_graft(list);
	for (i = 0; i < n; ++i) {
		isl_ast_graft *graft;

		graft = isl_ast_graft_list_get_ast_graft(list, i);
		if (!graft)
			break;
		isl_set_free(graft->guard);
		graft->guard = isl_set_copy(universe);
		if (!graft->guard)
			graft = isl_ast_graft_free(graft);
		list = isl_ast_graft_list_set_ast_graft(list, i, graft);
	}
	isl_set_free(universe);
	if (i < n)
		return isl_ast_graft_list_free(list);

	return list;
}

/* Collect the nodes contained in the grafts in "list" in a node list.
 */
static __isl_give isl_ast_node_list *extract_node_list(
	__isl_keep isl_ast_graft_list *list)
{
	int i, n;
	isl_ctx *ctx;
	isl_ast_node_list *node_list;

	if (!list)
		return NULL;
	ctx = isl_ast_graft_list_get_ctx(list);
	n = isl_ast_graft_list_n_ast_graft(list);
	node_list = isl_ast_node_list_alloc(ctx, n);
	for (i = 0; i < n; ++i) {
		isl_ast_node *node;
		isl_ast_graft *graft;

		graft = isl_ast_graft_list_get_ast_graft(list, i);
		node = isl_ast_graft_get_node(graft);
		node_list = isl_ast_node_list_add(node_list, node);
		isl_ast_graft_free(graft);
	}

	return node_list;
}

/* Look for shared enforced constraints by all the elements in "list"
 * on outer loops (with respect to the current depth) and return the result.
 *
 * If there are no elements in "list", then return the empty set.
 */
__isl_give isl_basic_set *isl_ast_graft_list_extract_shared_enforced(
	__isl_keep isl_ast_graft_list *list,
	__isl_keep isl_ast_build *build)
{
	int i, n;
	int depth;
	isl_space *space;
	isl_basic_set *enforced;

	if (!list)
		return NULL;

	space = isl_ast_build_get_space(build, 1);
	enforced = isl_basic_set_empty(space);

	depth = isl_ast_build_get_depth(build);
	n = isl_ast_graft_list_n_ast_graft(list);
	for (i = 0; i < n; ++i) {
		isl_ast_graft *graft;

		graft = isl_ast_graft_list_get_ast_graft(list, i);
		enforced = update_enforced(enforced, graft, depth);
		isl_ast_graft_free(graft);
	}

	return enforced;
}

/* Record "guard" in "graft" so that it will be enforced somewhere
 * up the tree.  If the graft already has a guard, then it may be partially
 * redundant in combination with the new guard and in the context
 * the generated constraints of "build".  In fact, the new guard
 * may in itself have some redundant constraints.
 * We therefore (re)compute the gist of the intersection
 * and coalesce the result.
 */
static __isl_give isl_ast_graft *store_guard(__isl_take isl_ast_graft *graft,
	__isl_take isl_set *guard, __isl_keep isl_ast_build *build)
{
	int is_universe;

	if (!graft)
		goto error;

	is_universe = isl_set_plain_is_universe(guard);
	if (is_universe < 0)
		goto error;
	if (is_universe) {
		isl_set_free(guard);
		return graft;
	}

	graft->guard = isl_set_intersect(graft->guard, guard);
	graft->guard = isl_set_gist(graft->guard,
				    isl_ast_build_get_generated(build));
	graft->guard = isl_set_coalesce(graft->guard);
	if (!graft->guard)
		return isl_ast_graft_free(graft);

	return graft;
error:
	isl_set_free(guard);
	return isl_ast_graft_free(graft);
}

/* For each graft in "list", replace its guard with the gist with
 * respect to "context".
 */
static __isl_give isl_ast_graft_list *gist_guards(
	__isl_take isl_ast_graft_list *list, __isl_keep isl_set *context)
{
	int i, n;

	if (!list)
		return NULL;

	n = isl_ast_graft_list_n_ast_graft(list);
	for (i = 0; i < n; ++i) {
		isl_ast_graft *graft;

		graft = isl_ast_graft_list_get_ast_graft(list, i);
		if (!graft)
			break;
		graft->guard = isl_set_gist(graft->guard,
						isl_set_copy(context));
		if (!graft->guard)
			graft = isl_ast_graft_free(graft);
		list = isl_ast_graft_list_set_ast_graft(list, i, graft);
	}
	if (i < n)
		return isl_ast_graft_list_free(list);

	return list;
}

/* For each graft in "list", replace its guard with the gist with
 * respect to "context".
 */
__isl_give isl_ast_graft_list *isl_ast_graft_list_gist_guards(
	__isl_take isl_ast_graft_list *list, __isl_take isl_set *context)
{
	list = gist_guards(list, context);
	isl_set_free(context);

	return list;
}

/* Allocate a graft in "build" based on the list of grafts in "sub_build".
 * "guard" and "enforced" are the guard and enforced constraints
 * of the allocated graft.  The guard is used to simplify the guards
 * of the elements in "list".
 *
 * The node is initialized to either a block containing the nodes of "children"
 * or, if there is only a single child, the node of that child.
 * If the current level requires a for node, it should be inserted by
 * a subsequent call to isl_ast_graft_insert_for.
 */
__isl_give isl_ast_graft *isl_ast_graft_alloc_from_children(
	__isl_take isl_ast_graft_list *list, __isl_take isl_set *guard,
	__isl_take isl_basic_set *enforced, __isl_keep isl_ast_build *build,
	__isl_keep isl_ast_build *sub_build)
{
	isl_ast_build *guard_build;
	isl_ast_node *node;
	isl_ast_node_list *node_list;
	isl_ast_graft *graft;

	guard_build = isl_ast_build_copy(sub_build);
	guard_build = isl_ast_build_replace_pending_by_guard(guard_build,
						isl_set_copy(guard));
	list = gist_guards(list, guard);
	list = insert_pending_guard_nodes(list, guard_build);
	isl_ast_build_free(guard_build);

	node_list = extract_node_list(list);
	node = isl_ast_node_from_ast_node_list(node_list);
	isl_ast_graft_list_free(list);

	graft = isl_ast_graft_alloc(node, build);
	graft = store_guard(graft, guard, build);
	graft = isl_ast_graft_enforce(graft, enforced);

	return graft;
}

/* Combine the grafts in the list into a single graft.
 *
 * The guard is initialized to the shared guard of the list elements (if any),
 * provided it does not depend on the current dimension.
 * The guards in the elements are then simplified with respect to the
 * hoisted guard and materialized as if nodes around the contained AST nodes
 * in the context of "sub_build".
 *
 * The enforced set is initialized to the simple hull of the enforced sets
 * of the elements, provided the ast_build_exploit_nested_bounds option is set
 * or the new graft will be used at the same level.
 *
 * The node is initialized to either a block containing the nodes of "list"
 * or, if there is only a single element, the node of that element.
 */
static __isl_give isl_ast_graft *ast_graft_list_fuse(
	__isl_take isl_ast_graft_list *list, __isl_keep isl_ast_build *build)
{
	isl_ast_graft *graft;
	isl_basic_set *enforced;
	isl_set *guard;

	if (!list)
		return NULL;

	enforced = isl_ast_graft_list_extract_shared_enforced(list, build);
	guard = isl_ast_graft_list_extract_hoistable_guard(list, build);
	graft = isl_ast_graft_alloc_from_children(list, guard, enforced,
						    build, build);

	return graft;
}

/* Combine the grafts in the list into a single graft.
 * Return a list containing this single graft.
 * If the original list is empty, then return an empty list.
 */
__isl_give isl_ast_graft_list *isl_ast_graft_list_fuse(
	__isl_take isl_ast_graft_list *list,
	__isl_keep isl_ast_build *build)
{
	isl_ast_graft *graft;

	if (!list)
		return NULL;
	if (isl_ast_graft_list_n_ast_graft(list) <= 1)
		return list;
	graft = ast_graft_list_fuse(list, build);
	return isl_ast_graft_list_from_ast_graft(graft);
}

/* Combine the two grafts into a single graft.
 * Return a list containing this single graft.
 */
static __isl_give isl_ast_graft *isl_ast_graft_fuse(
	__isl_take isl_ast_graft *graft1, __isl_take isl_ast_graft *graft2,
	__isl_keep isl_ast_build *build)
{
	isl_ctx *ctx;
	isl_ast_graft_list *list;

	ctx = isl_ast_build_get_ctx(build);

	list = isl_ast_graft_list_alloc(ctx, 2);
	list = isl_ast_graft_list_add(list, graft1);
	list = isl_ast_graft_list_add(list, graft2);

	return ast_graft_list_fuse(list, build);
}

/* Insert a for node enclosing the current graft->node.
 */
__isl_give isl_ast_graft *isl_ast_graft_insert_for(
	__isl_take isl_ast_graft *graft, __isl_take isl_ast_node *node)
{
	if (!graft)
		goto error;

	graft->node = isl_ast_node_for_set_body(node, graft->node);
	if (!graft->node)
		return isl_ast_graft_free(graft);

	return graft;
error:
	isl_ast_node_free(node);
	isl_ast_graft_free(graft);
	return NULL;
}

/* Insert a mark governing the current graft->node.
 */
__isl_give isl_ast_graft *isl_ast_graft_insert_mark(
	__isl_take isl_ast_graft *graft, __isl_take isl_id *mark)
{
	if (!graft)
		goto error;

	graft->node = isl_ast_node_alloc_mark(mark, graft->node);
	if (!graft->node)
		return isl_ast_graft_free(graft);

	return graft;
error:
	isl_id_free(mark);
	isl_ast_graft_free(graft);
	return NULL;
}

/* Represent the graft list as an AST node.
 * This operation drops the information about guards in the grafts, so
 * if there are any pending guards, then they are materialized as if nodes.
 */
__isl_give isl_ast_node *isl_ast_node_from_graft_list(
	__isl_take isl_ast_graft_list *list,
	__isl_keep isl_ast_build *build)
{
	isl_ast_node_list *node_list;

	list = insert_pending_guard_nodes(list, build);
	node_list = extract_node_list(list);
	isl_ast_graft_list_free(list);

	return isl_ast_node_from_ast_node_list(node_list);
}

void *isl_ast_graft_free(__isl_take isl_ast_graft *graft)
{
	if (!graft)
		return NULL;

	if (--graft->ref > 0)
		return NULL;

	isl_ast_node_free(graft->node);
	isl_set_free(graft->guard);
	isl_basic_set_free(graft->enforced);
	free(graft);

	return NULL;
}

/* Record that the grafted tree enforces
 * "enforced" by intersecting graft->enforced with "enforced".
 */
__isl_give isl_ast_graft *isl_ast_graft_enforce(
	__isl_take isl_ast_graft *graft, __isl_take isl_basic_set *enforced)
{
	if (!graft || !enforced)
		goto error;

	enforced = isl_basic_set_align_params(enforced,
				isl_basic_set_get_space(graft->enforced));
	graft->enforced = isl_basic_set_align_params(graft->enforced,
				isl_basic_set_get_space(enforced));
	graft->enforced = isl_basic_set_intersect(graft->enforced, enforced);
	if (!graft->enforced)
		return isl_ast_graft_free(graft);

	return graft;
error:
	isl_basic_set_free(enforced);
	return isl_ast_graft_free(graft);
}

__isl_give isl_basic_set *isl_ast_graft_get_enforced(
	__isl_keep isl_ast_graft *graft)
{
	return graft ? isl_basic_set_copy(graft->enforced) : NULL;
}

__isl_give isl_set *isl_ast_graft_get_guard(__isl_keep isl_ast_graft *graft)
{
	return graft ? isl_set_copy(graft->guard) : NULL;
}

/* Record that "guard" needs to be inserted in "graft".
 */
__isl_give isl_ast_graft *isl_ast_graft_add_guard(
	__isl_take isl_ast_graft *graft,
	__isl_take isl_set *guard, __isl_keep isl_ast_build *build)
{
	return store_guard(graft, guard, build);
}

/* Reformulate the "graft", which was generated in the context
 * of an inner code generation, in terms of the outer code generation
 * AST build.
 *
 * If "product" is set, then the domain of the inner code generation build is
 *
 *	[O -> S]
 *
 * with O the domain of the outer code generation build.
 * We essentially need to project out S.
 *
 * If "product" is not set, then we need to project the domains onto
 * their parameter spaces.
 */
__isl_give isl_ast_graft *isl_ast_graft_unembed(__isl_take isl_ast_graft *graft,
	int product)
{
	isl_basic_set *enforced;

	if (!graft)
		return NULL;

	if (product) {
		enforced = graft->enforced;
		enforced = isl_basic_map_domain(isl_basic_set_unwrap(enforced));
		graft->enforced = enforced;
		graft->guard = isl_map_domain(isl_set_unwrap(graft->guard));
	} else {
		graft->enforced = isl_basic_set_params(graft->enforced);
		graft->guard = isl_set_params(graft->guard);
	}
	graft->guard = isl_set_compute_divs(graft->guard);

	if (!graft->enforced || !graft->guard)
		return isl_ast_graft_free(graft);

	return graft;
}

/* Reformulate the grafts in "list", which were generated in the context
 * of an inner code generation, in terms of the outer code generation
 * AST build.
 */
__isl_give isl_ast_graft_list *isl_ast_graft_list_unembed(
	__isl_take isl_ast_graft_list *list, int product)
{
	int i, n;

	n = isl_ast_graft_list_n_ast_graft(list);
	for (i = 0; i < n; ++i) {
		isl_ast_graft *graft;

		graft = isl_ast_graft_list_get_ast_graft(list, i);
		graft = isl_ast_graft_unembed(graft, product);
		list = isl_ast_graft_list_set_ast_graft(list, i, graft);
	}

	return list;
}

/* Compute the preimage of "graft" under the function represented by "ma".
 * In other words, plug in "ma" in "enforced" and "guard" fields of "graft".
 */
__isl_give isl_ast_graft *isl_ast_graft_preimage_multi_aff(
	__isl_take isl_ast_graft *graft, __isl_take isl_multi_aff *ma)
{
	isl_basic_set *enforced;

	if (!graft)
		return NULL;

	enforced = graft->enforced;
	graft->enforced = isl_basic_set_preimage_multi_aff(enforced,
						isl_multi_aff_copy(ma));
	graft->guard = isl_set_preimage_multi_aff(graft->guard, ma);

	if (!graft->enforced || !graft->guard)
		return isl_ast_graft_free(graft);

	return graft;
}

/* Compute the preimage of all the grafts in "list" under
 * the function represented by "ma".
 */
__isl_give isl_ast_graft_list *isl_ast_graft_list_preimage_multi_aff(
	__isl_take isl_ast_graft_list *list, __isl_take isl_multi_aff *ma)
{
	int i, n;

	n = isl_ast_graft_list_n_ast_graft(list);
	for (i = 0; i < n; ++i) {
		isl_ast_graft *graft;

		graft = isl_ast_graft_list_get_ast_graft(list, i);
		graft = isl_ast_graft_preimage_multi_aff(graft,
						    isl_multi_aff_copy(ma));
		list = isl_ast_graft_list_set_ast_graft(list, i, graft);
	}

	isl_multi_aff_free(ma);
	return list;
}

/* Compare two grafts based on their guards.
 */
static int cmp_graft(__isl_keep isl_ast_graft *a, __isl_keep isl_ast_graft *b,
	void *user)
{
	return isl_set_plain_cmp(a->guard, b->guard);
}

/* Order the elements in "list" based on their guards.
 */
__isl_give isl_ast_graft_list *isl_ast_graft_list_sort_guard(
	__isl_take isl_ast_graft_list *list)
{
	return isl_ast_graft_list_sort(list, &cmp_graft, NULL);
}

/* Merge the given two lists into a single list of grafts,
 * merging grafts with the same guard into a single graft.
 *
 * "list2" has been sorted using isl_ast_graft_list_sort.
 * "list1" may be the result of a previous call to isl_ast_graft_list_merge
 * and may therefore not be completely sorted.
 *
 * The elements in "list2" need to be executed after those in "list1",
 * but if the guard of a graft in "list2" is disjoint from the guards
 * of some final elements in "list1", then it can be moved up to before
 * those final elements.
 *
 * In particular, we look at each element g of "list2" in turn
 * and move it up beyond elements of "list1" that would be sorted
 * after g as long as each of these elements has a guard that is disjoint
 * from that of g.
 *
 * We do not allow the second or any later element of "list2" to be moved
 * before a previous elements of "list2" even if the reason that
 * that element didn't move up further was that its guard was not disjoint
 * from that of the previous element in "list1".
 */
__isl_give isl_ast_graft_list *isl_ast_graft_list_merge(
	__isl_take isl_ast_graft_list *list1,
	__isl_take isl_ast_graft_list *list2,
	__isl_keep isl_ast_build *build)
{
	int i, j, first;

	if (!list1 || !list2 || !build)
		goto error;
	if (list2->n == 0) {
		isl_ast_graft_list_free(list2);
		return list1;
	}
	if (list1->n == 0) {
		isl_ast_graft_list_free(list1);
		return list2;
	}

	first = 0;
	for (i = 0; i < list2->n; ++i) {
		isl_ast_graft *graft;
		graft = isl_ast_graft_list_get_ast_graft(list2, i);
		if (!graft)
			break;

		for (j = list1->n; j >= 0; --j) {
			int cmp, disjoint;
			isl_ast_graft *graft_j;

			if (j == first)
				cmp = -1;
			else
				cmp = isl_set_plain_cmp(list1->p[j - 1]->guard,
							graft->guard);
			if (cmp > 0) {
				disjoint = isl_set_is_disjoint(graft->guard,
							list1->p[j - 1]->guard);
				if (disjoint < 0) {
					list1 = isl_ast_graft_list_free(list1);
					break;
				}
				if (!disjoint)
					cmp = -1;
			}
			if (cmp > 0)
				continue;
			if (cmp < 0) {
				list1 = isl_ast_graft_list_insert(list1, j,
								graft);
				break;
			}

			--j;

			graft_j = isl_ast_graft_list_get_ast_graft(list1, j);
			graft_j = isl_ast_graft_fuse(graft_j, graft, build);
			list1 = isl_ast_graft_list_set_ast_graft(list1, j,
								graft_j);
			break;
		}

		if (j < 0)
			isl_die(isl_ast_build_get_ctx(build),
				isl_error_internal,
				"element failed to get inserted", break);

		first = j + 1;
		if (!list1)
			break;
	}
	if (i < list2->n)
		list1 = isl_ast_graft_list_free(list1);
	isl_ast_graft_list_free(list2);

	return list1;
error:
	isl_ast_graft_list_free(list1);
	isl_ast_graft_list_free(list2);
	return NULL;
}

__isl_give isl_printer *isl_printer_print_ast_graft(__isl_take isl_printer *p,
	__isl_keep isl_ast_graft *graft)
{
	if (!p)
		return NULL;
	if (!graft)
		return isl_printer_free(p);

	p = isl_printer_print_str(p, "(");
	p = isl_printer_print_str(p, "guard: ");
	p = isl_printer_print_set(p, graft->guard);
	p = isl_printer_print_str(p, ", ");
	p = isl_printer_print_str(p, "enforced: ");
	p = isl_printer_print_basic_set(p, graft->enforced);
	p = isl_printer_print_str(p, ", ");
	p = isl_printer_print_str(p, "node: ");
	p = isl_printer_print_ast_node(p, graft->node);
	p = isl_printer_print_str(p, ")");

	return p;
}
