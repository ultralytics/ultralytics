/*
 * Copyright 2011      INRIA Saclay
 * Copyright 2012-2014 Ecole Normale Superieure
 * Copyright 2016      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl/ctx.h>
#include <isl_aff_private.h>
#include <isl/map.h>
#include <isl/set.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl_sort.h>
#include <isl_schedule_private.h>
#include <isl_schedule_tree.h>
#include <isl_schedule_node_private.h>
#include <isl_band_private.h>

/* Return a schedule encapsulating the given schedule tree.
 *
 * We currently only allow schedule trees with a domain or extension as root.
 *
 * The leaf field is initialized as a leaf node so that it can be
 * used to represent leaves in the constructed schedule.
 * The reference count is set to -1 since the isl_schedule_tree
 * should never be freed.  It is up to the (internal) users of
 * these leaves to ensure that they are only used while the schedule
 * is still alive.
 */
__isl_give isl_schedule *isl_schedule_from_schedule_tree(isl_ctx *ctx,
	__isl_take isl_schedule_tree *tree)
{
	enum isl_schedule_node_type type;
	isl_schedule *schedule;

	if (!tree)
		return NULL;
	type = isl_schedule_tree_get_type(tree);
	if (type != isl_schedule_node_domain &&
	    type != isl_schedule_node_extension)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_unsupported,
			"root of schedule tree should be a domain or extension",
			goto error);

	schedule = isl_calloc_type(ctx, isl_schedule);
	if (!schedule)
		goto error;

	schedule->ref = 1;
	schedule->root = tree;
	schedule->leaf = isl_schedule_tree_leaf(ctx);

	if (!schedule->leaf)
		return isl_schedule_free(schedule);
	return schedule;
error:
	isl_schedule_tree_free(tree);
	return NULL;
}

/* Return a pointer to a schedule with as single node
 * a domain node with the given domain.
 */
__isl_give isl_schedule *isl_schedule_from_domain(
	__isl_take isl_union_set *domain)
{
	isl_ctx *ctx;
	isl_schedule_tree *tree;

	ctx = isl_union_set_get_ctx(domain);
	tree = isl_schedule_tree_from_domain(domain);
	return isl_schedule_from_schedule_tree(ctx, tree);
}

/* Return a pointer to a schedule with as single node
 * a domain node with an empty domain.
 */
__isl_give isl_schedule *isl_schedule_empty(__isl_take isl_space *space)
{
	return isl_schedule_from_domain(isl_union_set_empty(space));
}

/* Return a new reference to "sched".
 */
__isl_give isl_schedule *isl_schedule_copy(__isl_keep isl_schedule *sched)
{
	if (!sched)
		return NULL;

	sched->ref++;
	return sched;
}

/* Return an isl_schedule that is equal to "schedule" and that has only
 * a single reference.
 *
 * We only need and support this function when the schedule is represented
 * as a schedule tree.
 */
__isl_give isl_schedule *isl_schedule_cow(__isl_take isl_schedule *schedule)
{
	isl_ctx *ctx;
	isl_schedule_tree *tree;

	if (!schedule)
		return NULL;
	if (schedule->ref == 1)
		return schedule;

	ctx = isl_schedule_get_ctx(schedule);
	if (!schedule->root)
		isl_die(ctx, isl_error_internal,
			"only for schedule tree based schedules",
			return isl_schedule_free(schedule));
	schedule->ref--;
	tree = isl_schedule_tree_copy(schedule->root);
	return isl_schedule_from_schedule_tree(ctx, tree);
}

__isl_null isl_schedule *isl_schedule_free(__isl_take isl_schedule *sched)
{
	if (!sched)
		return NULL;

	if (--sched->ref > 0)
		return NULL;

	isl_band_list_free(sched->band_forest);
	isl_schedule_tree_free(sched->root);
	isl_schedule_tree_free(sched->leaf);
	free(sched);
	return NULL;
}

/* Replace the root of "schedule" by "tree".
 */
__isl_give isl_schedule *isl_schedule_set_root(
	__isl_take isl_schedule *schedule, __isl_take isl_schedule_tree *tree)
{
	if (!schedule || !tree)
		goto error;
	if (schedule->root == tree) {
		isl_schedule_tree_free(tree);
		return schedule;
	}

	schedule = isl_schedule_cow(schedule);
	if (!schedule)
		goto error;
	isl_schedule_tree_free(schedule->root);
	schedule->root = tree;

	return schedule;
error:
	isl_schedule_free(schedule);
	isl_schedule_tree_free(tree);
	return NULL;
}

isl_ctx *isl_schedule_get_ctx(__isl_keep isl_schedule *schedule)
{
	return schedule ? isl_schedule_tree_get_ctx(schedule->leaf) : NULL;
}

/* Return a pointer to the leaf of "schedule".
 */
__isl_keep isl_schedule_tree *isl_schedule_peek_leaf(
	__isl_keep isl_schedule *schedule)
{
	return schedule ? schedule->leaf : NULL;
}

/* Are "schedule1" and "schedule2" obviously equal to each other?
 */
isl_bool isl_schedule_plain_is_equal(__isl_keep isl_schedule *schedule1,
	__isl_keep isl_schedule *schedule2)
{
	if (!schedule1 || !schedule2)
		return isl_bool_error;
	if (schedule1 == schedule2)
		return isl_bool_true;
	return isl_schedule_tree_plain_is_equal(schedule1->root,
						schedule2->root);
}

/* Return the (parameter) space of the schedule, i.e., the space
 * of the root domain.
 */
__isl_give isl_space *isl_schedule_get_space(
	__isl_keep isl_schedule *schedule)
{
	enum isl_schedule_node_type type;
	isl_space *space;
	isl_union_set *domain;

	if (!schedule)
		return NULL;
	if (!schedule->root)
		isl_die(isl_schedule_get_ctx(schedule), isl_error_invalid,
			"schedule tree representation not available",
			return NULL);
	type = isl_schedule_tree_get_type(schedule->root);
	if (type != isl_schedule_node_domain)
		isl_die(isl_schedule_get_ctx(schedule), isl_error_internal,
			"root node not a domain node", return NULL);

	domain = isl_schedule_tree_domain_get_domain(schedule->root);
	space = isl_union_set_get_space(domain);
	isl_union_set_free(domain);

	return space;
}

/* Return a pointer to the root of "schedule".
 */
__isl_give isl_schedule_node *isl_schedule_get_root(
	__isl_keep isl_schedule *schedule)
{
	isl_ctx *ctx;
	isl_schedule_tree *tree;
	isl_schedule_tree_list *ancestors;

	if (!schedule)
		return NULL;

	if (!schedule->root)
		isl_die(isl_schedule_get_ctx(schedule), isl_error_invalid,
			"schedule tree representation not available",
			return NULL);

	ctx = isl_schedule_get_ctx(schedule);
	tree = isl_schedule_tree_copy(schedule->root);
	schedule = isl_schedule_copy(schedule);
	ancestors = isl_schedule_tree_list_alloc(ctx, 0);
	return isl_schedule_node_alloc(schedule, tree, ancestors, NULL);
}

/* Set max_out to the maximal number of output dimensions over
 * all maps.
 */
static isl_stat update_max_out(__isl_take isl_map *map, void *user)
{
	int *max_out = user;
	int n_out = isl_map_dim(map, isl_dim_out);

	if (n_out > *max_out)
		*max_out = n_out;

	isl_map_free(map);
	return isl_stat_ok;
}

/* Internal data structure for map_pad_range.
 *
 * "max_out" is the maximal schedule dimension.
 * "res" collects the results.
 */
struct isl_pad_schedule_map_data {
	int max_out;
	isl_union_map *res;
};

/* Pad the range of the given map with zeros to data->max_out and
 * then add the result to data->res.
 */
static isl_stat map_pad_range(__isl_take isl_map *map, void *user)
{
	struct isl_pad_schedule_map_data *data = user;
	int i;
	int n_out = isl_map_dim(map, isl_dim_out);

	map = isl_map_add_dims(map, isl_dim_out, data->max_out - n_out);
	for (i = n_out; i < data->max_out; ++i)
		map = isl_map_fix_si(map, isl_dim_out, i, 0);

	data->res = isl_union_map_add_map(data->res, map);
	if (!data->res)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Pad the ranges of the maps in the union map with zeros such they all have
 * the same dimension.
 */
static __isl_give isl_union_map *pad_schedule_map(
	__isl_take isl_union_map *umap)
{
	struct isl_pad_schedule_map_data data;

	if (!umap)
		return NULL;
	if (isl_union_map_n_map(umap) <= 1)
		return umap;

	data.max_out = 0;
	if (isl_union_map_foreach_map(umap, &update_max_out, &data.max_out) < 0)
		return isl_union_map_free(umap);

	data.res = isl_union_map_empty(isl_union_map_get_space(umap));
	if (isl_union_map_foreach_map(umap, &map_pad_range, &data) < 0)
		data.res = isl_union_map_free(data.res);

	isl_union_map_free(umap);
	return data.res;
}

/* Return the domain of the root domain node of "schedule".
 */
__isl_give isl_union_set *isl_schedule_get_domain(
	__isl_keep isl_schedule *schedule)
{
	if (!schedule)
		return NULL;
	if (!schedule->root)
		isl_die(isl_schedule_get_ctx(schedule), isl_error_invalid,
			"schedule tree representation not available",
			return NULL);
	return isl_schedule_tree_domain_get_domain(schedule->root);
}

/* Traverse all nodes of "sched" in depth first preorder.
 *
 * If "fn" returns -1 on any of the nodes, then the traversal is aborted.
 * If "fn" returns 0 on any of the nodes, then the subtree rooted
 * at that node is skipped.
 *
 * Return 0 on success and -1 on failure.
 */
isl_stat isl_schedule_foreach_schedule_node_top_down(
	__isl_keep isl_schedule *sched,
	isl_bool (*fn)(__isl_keep isl_schedule_node *node, void *user),
	void *user)
{
	isl_schedule_node *node;
	isl_stat r;

	if (!sched)
		return isl_stat_error;

	node = isl_schedule_get_root(sched);
	r = isl_schedule_node_foreach_descendant_top_down(node, fn, user);
	isl_schedule_node_free(node);

	return r;
}

/* Traverse the node of "sched" in depth first postorder,
 * allowing the user to modify the visited node.
 * The traversal continues from the node returned by the callback function.
 * It is the responsibility of the user to ensure that this does not
 * lead to an infinite loop.  It is safest to always return a pointer
 * to the same position (same ancestors and child positions) as the input node.
 */
__isl_give isl_schedule *isl_schedule_map_schedule_node_bottom_up(
	__isl_take isl_schedule *schedule,
	__isl_give isl_schedule_node *(*fn)(
		__isl_take isl_schedule_node *node, void *user), void *user)
{
	isl_schedule_node *node;

	node = isl_schedule_get_root(schedule);
	isl_schedule_free(schedule);

	node = isl_schedule_node_map_descendant_bottom_up(node, fn, user);
	schedule = isl_schedule_node_get_schedule(node);
	isl_schedule_node_free(node);

	return schedule;
}

/* Wrapper around isl_schedule_node_reset_user for use as
 * an isl_schedule_map_schedule_node_bottom_up callback.
 */
static __isl_give isl_schedule_node *reset_user(
	__isl_take isl_schedule_node *node, void *user)
{
	return isl_schedule_node_reset_user(node);
}

/* Reset the user pointer on all identifiers of parameters and tuples
 * in the schedule "schedule".
 */
__isl_give isl_schedule *isl_schedule_reset_user(
	__isl_take isl_schedule *schedule)
{
	return isl_schedule_map_schedule_node_bottom_up(schedule, &reset_user,
							NULL);
}

/* Wrapper around isl_schedule_node_align_params for use as
 * an isl_schedule_map_schedule_node_bottom_up callback.
 */
static __isl_give isl_schedule_node *align_params(
	__isl_take isl_schedule_node *node, void *user)
{
	isl_space *space = user;

	return isl_schedule_node_align_params(node, isl_space_copy(space));
}

/* Align the parameters of all nodes in schedule "schedule"
 * to those of "space".
 */
__isl_give isl_schedule *isl_schedule_align_params(
	__isl_take isl_schedule *schedule, __isl_take isl_space *space)
{
	schedule = isl_schedule_map_schedule_node_bottom_up(schedule,
						    &align_params, space);
	isl_space_free(space);
	return schedule;
}

/* Wrapper around isl_schedule_node_pullback_union_pw_multi_aff for use as
 * an isl_schedule_map_schedule_node_bottom_up callback.
 */
static __isl_give isl_schedule_node *pullback_upma(
	__isl_take isl_schedule_node *node, void *user)
{
	isl_union_pw_multi_aff *upma = user;

	return isl_schedule_node_pullback_union_pw_multi_aff(node,
					isl_union_pw_multi_aff_copy(upma));
}

/* Compute the pullback of "schedule" by the function represented by "upma".
 * In other words, plug in "upma" in the iteration domains of "schedule".
 *
 * The schedule tree is not allowed to contain any expansion nodes.
 */
__isl_give isl_schedule *isl_schedule_pullback_union_pw_multi_aff(
	__isl_take isl_schedule *schedule,
	__isl_take isl_union_pw_multi_aff *upma)
{
	schedule = isl_schedule_map_schedule_node_bottom_up(schedule,
						&pullback_upma, upma);
	isl_union_pw_multi_aff_free(upma);
	return schedule;
}

/* Expand the schedule "schedule" by extending all leaves
 * with an expansion node with as subtree the tree of "expansion".
 * The expansion of the expansion node is determined by "contraction"
 * and the domain of "expansion".  That is, the domain of "expansion"
 * is contracted according to "contraction".
 *
 * Call isl_schedule_node_expand after extracting the required
 * information from "expansion".
 */
__isl_give isl_schedule *isl_schedule_expand(__isl_take isl_schedule *schedule,
	__isl_take isl_union_pw_multi_aff *contraction,
	__isl_take isl_schedule *expansion)
{
	isl_union_set *domain;
	isl_schedule_node *node;
	isl_schedule_tree *tree;

	domain = isl_schedule_get_domain(expansion);

	node = isl_schedule_get_root(expansion);
	node = isl_schedule_node_child(node, 0);
	tree = isl_schedule_node_get_tree(node);
	isl_schedule_node_free(node);
	isl_schedule_free(expansion);

	node = isl_schedule_get_root(schedule);
	isl_schedule_free(schedule);
	node = isl_schedule_node_expand(node, contraction, domain, tree);
	schedule = isl_schedule_node_get_schedule(node);
	isl_schedule_node_free(node);

	return schedule;
}

/* Intersect the domain of the schedule "schedule" with "domain".
 * The root of "schedule" is required to be a domain node.
 */
__isl_give isl_schedule *isl_schedule_intersect_domain(
	__isl_take isl_schedule *schedule, __isl_take isl_union_set *domain)
{
	enum isl_schedule_node_type root_type;
	isl_schedule_node *node;

	if (!schedule || !domain)
		goto error;

	root_type = isl_schedule_tree_get_type(schedule->root);
	if (root_type != isl_schedule_node_domain)
		isl_die(isl_schedule_get_ctx(schedule), isl_error_invalid,
			"root node must be a domain node", goto error);

	node = isl_schedule_get_root(schedule);
	isl_schedule_free(schedule);
	node = isl_schedule_node_domain_intersect_domain(node, domain);
	schedule = isl_schedule_node_get_schedule(node);
	isl_schedule_node_free(node);

	return schedule;
error:
	isl_schedule_free(schedule);
	isl_union_set_free(domain);
	return NULL;
}

/* Replace the domain of the schedule "schedule" with the gist
 * of the original domain with respect to the parameter domain "context".
 */
__isl_give isl_schedule *isl_schedule_gist_domain_params(
	__isl_take isl_schedule *schedule, __isl_take isl_set *context)
{
	enum isl_schedule_node_type root_type;
	isl_schedule_node *node;

	if (!schedule || !context)
		goto error;

	root_type = isl_schedule_tree_get_type(schedule->root);
	if (root_type != isl_schedule_node_domain)
		isl_die(isl_schedule_get_ctx(schedule), isl_error_invalid,
			"root node must be a domain node", goto error);

	node = isl_schedule_get_root(schedule);
	isl_schedule_free(schedule);
	node = isl_schedule_node_domain_gist_params(node, context);
	schedule = isl_schedule_node_get_schedule(node);
	isl_schedule_node_free(node);

	return schedule;
error:
	isl_schedule_free(schedule);
	isl_set_free(context);
	return NULL;
}

/* Return an isl_union_map representation of the schedule.
 * If we still have access to the schedule tree, then we return
 * an isl_union_map corresponding to the subtree schedule of the child
 * of the root domain node.  That is, we do not intersect the domain
 * of the returned isl_union_map with the domain constraints.
 * Otherwise, we must have removed it because we created a band forest.
 * If so, we extract the isl_union_map from the forest.
 * This reconstructed schedule map
 * then needs to be padded with zeros to unify the schedule space
 * since the result of isl_band_list_get_suffix_schedule may not have
 * a unified schedule space.
 */
__isl_give isl_union_map *isl_schedule_get_map(__isl_keep isl_schedule *sched)
{
	enum isl_schedule_node_type type;
	isl_schedule_node *node;
	isl_union_map *umap;

	if (!sched)
		return NULL;

	if (sched->root) {
		type = isl_schedule_tree_get_type(sched->root);
		if (type != isl_schedule_node_domain)
			isl_die(isl_schedule_get_ctx(sched), isl_error_internal,
				"root node not a domain node", return NULL);

		node = isl_schedule_get_root(sched);
		node = isl_schedule_node_child(node, 0);
		umap = isl_schedule_node_get_subtree_schedule_union_map(node);
		isl_schedule_node_free(node);

		return umap;
	}

	umap = isl_band_list_get_suffix_schedule(sched->band_forest);
	return pad_schedule_map(umap);
}

static __isl_give isl_band_list *construct_band_list(
	__isl_take isl_schedule_node *node, __isl_take isl_union_set *domain,
	__isl_keep isl_band *parent);

/* Construct an isl_band structure from the given schedule tree node,
 * which may be either a band node or a leaf node.
 * In the latter case, construct a zero-dimensional band.
 * "domain" is the universe set of the domain elements that reach "node".
 * "parent" is the parent isl_band of the isl_band constructed
 * by this function.
 *
 * In case of a band node, we copy the properties (except tilability,
 * which is implicit in an isl_band) to the isl_band.
 * We assume that the band node is not zero-dimensional.
 * If the child of the band node is not a leaf node,
 * then we extract the children of the isl_band from this child.
 */
static __isl_give isl_band *construct_band(__isl_take isl_schedule_node *node,
	__isl_take isl_union_set *domain, __isl_keep isl_band *parent)
{
	int i;
	isl_ctx *ctx;
	isl_band *band = NULL;
	isl_multi_union_pw_aff *mupa;

	if (!node || !domain)
		goto error;

	ctx = isl_schedule_node_get_ctx(node);
	band = isl_band_alloc(ctx);
	if (!band)
		goto error;

	band->schedule = node->schedule;
	band->parent = parent;

	if (isl_schedule_node_get_type(node) == isl_schedule_node_leaf) {
		band->n = 0;
		band->pma = isl_union_pw_multi_aff_from_domain(domain);
		isl_schedule_node_free(node);
		return band;
	}

	band->n = isl_schedule_node_band_n_member(node);
	if (band->n == 0)
		isl_die(ctx, isl_error_unsupported,
			"zero-dimensional band nodes not supported",
			goto error);
	band->coincident = isl_alloc_array(ctx, int, band->n);
	if (band->n && !band->coincident)
		goto error;
	for (i = 0; i < band->n; ++i)
		band->coincident[i] =
			isl_schedule_node_band_member_get_coincident(node, i);
	mupa = isl_schedule_node_band_get_partial_schedule(node);
	band->pma = isl_union_pw_multi_aff_from_multi_union_pw_aff(mupa);
	if (!band->pma)
		goto error;

	node = isl_schedule_node_child(node, 0);
	if (isl_schedule_node_get_type(node) == isl_schedule_node_leaf) {
		isl_schedule_node_free(node);
		isl_union_set_free(domain);
		return band;
	}

	band->children = construct_band_list(node, domain, band);
	if (!band->children)
		return isl_band_free(band);

	return band;
error:
	isl_union_set_free(domain);
	isl_schedule_node_free(node);
	isl_band_free(band);
	return NULL;
}

/* Construct a list of isl_band structures from the children of "node".
 * "node" itself is a sequence or set node, so that each of the child nodes
 * is a filter node and the list returned by node_construct_band_list
 * consists of a single element.
 * "domain" is the universe set of the domain elements that reach "node".
 * "parent" is the parent isl_band of the isl_band structures constructed
 * by this function.
 */
static __isl_give isl_band_list *construct_band_list_from_children(
	__isl_take isl_schedule_node *node, __isl_take isl_union_set *domain,
	__isl_keep isl_band *parent)
{
	int i, n;
	isl_ctx *ctx;
	isl_band_list *list;

	n = isl_schedule_node_n_children(node);

	ctx = isl_schedule_node_get_ctx(node);
	list = isl_band_list_alloc(ctx, 0);
	for (i = 0; i < n; ++i) {
		isl_schedule_node *child;
		isl_band_list *list_i;

		child = isl_schedule_node_get_child(node, i);
		list_i = construct_band_list(child, isl_union_set_copy(domain),
						parent);
		list = isl_band_list_concat(list, list_i);
	}

	isl_union_set_free(domain);
	isl_schedule_node_free(node);

	return list;
}

/* Construct an isl_band structure from the given sequence node
 * (or set node that is treated as a sequence node).
 * A single-dimensional band is created with as schedule for each of
 * filters of the children, the corresponding child position.
 * "domain" is the universe set of the domain elements that reach "node".
 * "parent" is the parent isl_band of the isl_band constructed
 * by this function.
 */
static __isl_give isl_band_list *construct_band_list_sequence(
	__isl_take isl_schedule_node *node, __isl_take isl_union_set *domain,
	__isl_keep isl_band *parent)
{
	int i, n;
	isl_ctx *ctx;
	isl_band *band = NULL;
	isl_space *space;
	isl_union_pw_multi_aff *upma;

	if (!node || !domain)
		goto error;

	ctx = isl_schedule_node_get_ctx(node);
	band = isl_band_alloc(ctx);
	if (!band)
		goto error;

	band->schedule = node->schedule;
	band->parent = parent;
	band->n = 1;
	band->coincident = isl_calloc_array(ctx, int, band->n);
	if (!band->coincident)
		goto error;

	n = isl_schedule_node_n_children(node);
	space = isl_union_set_get_space(domain);
	upma = isl_union_pw_multi_aff_empty(isl_space_copy(space));

	space = isl_space_set_from_params(space);
	space = isl_space_add_dims(space, isl_dim_set, 1);

	for (i = 0; i < n; ++i) {
		isl_schedule_node *child;
		isl_union_set *filter;
		isl_val *v;
		isl_val_list *vl;
		isl_multi_val *mv;
		isl_union_pw_multi_aff *upma_i;

		child = isl_schedule_node_get_child(node, i);
		filter = isl_schedule_node_filter_get_filter(child);
		isl_schedule_node_free(child);
		filter = isl_union_set_intersect(filter,
						isl_union_set_copy(domain));
		v = isl_val_int_from_si(ctx, i);
		vl = isl_val_list_from_val(v);
		mv = isl_multi_val_from_val_list(isl_space_copy(space), vl);
		upma_i = isl_union_pw_multi_aff_multi_val_on_domain(filter, mv);
		upma = isl_union_pw_multi_aff_union_add(upma, upma_i);
	}

	isl_space_free(space);

	band->pma = upma;
	if (!band->pma)
		goto error;

	band->children = construct_band_list_from_children(node, domain, band);
	if (!band->children)
		band = isl_band_free(band);
	return isl_band_list_from_band(band);
error:
	isl_union_set_free(domain);
	isl_schedule_node_free(node);
	isl_band_free(band);
	return NULL;
}

/* Construct a list of isl_band structures from "node" depending
 * on the type of "node".
 * "domain" is the universe set of the domain elements that reach "node".
 * "parent" is the parent isl_band of the isl_band structures constructed
 * by this function.
 *
 * If schedule_separate_components is set then set nodes are treated
 * as sequence nodes.  Otherwise, we directly extract an (implicitly
 * parallel) list of isl_band structures.
 *
 * If "node" is a filter, then "domain" is updated by the filter.
 */
static __isl_give isl_band_list *construct_band_list(
	__isl_take isl_schedule_node *node, __isl_take isl_union_set *domain,
	__isl_keep isl_band *parent)
{
	enum isl_schedule_node_type type;
	isl_ctx *ctx;
	isl_band *band;
	isl_band_list *list;
	isl_union_set *filter;

	if (!node || !domain)
		goto error;

	type = isl_schedule_node_get_type(node);
	switch (type) {
	case isl_schedule_node_error:
		goto error;
	case isl_schedule_node_context:
		isl_die(isl_schedule_node_get_ctx(node), isl_error_unsupported,
			"context nodes not supported", goto error);
	case isl_schedule_node_domain:
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"internal domain nodes not allowed", goto error);
	case isl_schedule_node_expansion:
		isl_die(isl_schedule_node_get_ctx(node), isl_error_unsupported,
			"expansion nodes not supported", goto error);
	case isl_schedule_node_extension:
		isl_die(isl_schedule_node_get_ctx(node), isl_error_unsupported,
			"extension nodes not supported", goto error);
	case isl_schedule_node_filter:
		filter = isl_schedule_node_filter_get_filter(node);
		domain = isl_union_set_intersect(domain, filter);
		node = isl_schedule_node_child(node, 0);
		return construct_band_list(node, domain, parent);
	case isl_schedule_node_guard:
		isl_die(isl_schedule_node_get_ctx(node), isl_error_unsupported,
			"guard nodes not supported", goto error);
	case isl_schedule_node_mark:
		isl_die(isl_schedule_node_get_ctx(node), isl_error_unsupported,
			"mark nodes not supported", goto error);
	case isl_schedule_node_set:
		ctx = isl_schedule_node_get_ctx(node);
		if (isl_options_get_schedule_separate_components(ctx))
			return construct_band_list_sequence(node, domain,
							    parent);
		else
			return construct_band_list_from_children(node, domain,
							    parent);
	case isl_schedule_node_sequence:
		return construct_band_list_sequence(node, domain, parent);
	case isl_schedule_node_leaf:
	case isl_schedule_node_band:
		band = construct_band(node, domain, parent);
		list = isl_band_list_from_band(band);
		break;
	}

	return list;
error:
	isl_union_set_free(domain);
	isl_schedule_node_free(node);
	return NULL;
}

/* Return the roots of a band forest representation of the schedule.
 * The band forest is constructed from the schedule tree,
 * but once such a band forest is
 * constructed, we forget about the original schedule tree since
 * the user may modify the schedule through the band forest.
 */
__isl_give isl_band_list *isl_schedule_get_band_forest(
	__isl_keep isl_schedule *schedule)
{
	isl_schedule_node *node;
	isl_union_set *domain;

	if (!schedule)
		return NULL;
	if (schedule->root) {
		node = isl_schedule_get_root(schedule);
		domain = isl_schedule_node_domain_get_domain(node);
		domain = isl_union_set_universe(domain);
		node = isl_schedule_node_child(node, 0);

		schedule->band_forest = construct_band_list(node, domain, NULL);
		schedule->root = isl_schedule_tree_free(schedule->root);
	}
	return isl_band_list_dup(schedule->band_forest);
}

/* Call "fn" on each band in the schedule in depth-first post-order.
 */
int isl_schedule_foreach_band(__isl_keep isl_schedule *sched,
	int (*fn)(__isl_keep isl_band *band, void *user), void *user)
{
	int r;
	isl_band_list *forest;

	if (!sched)
		return -1;

	forest = isl_schedule_get_band_forest(sched);
	r = isl_band_list_foreach_band(forest, fn, user);
	isl_band_list_free(forest);

	return r;
}

static __isl_give isl_printer *print_band_list(__isl_take isl_printer *p,
	__isl_keep isl_band_list *list);

static __isl_give isl_printer *print_band(__isl_take isl_printer *p,
	__isl_keep isl_band *band)
{
	isl_band_list *children;

	p = isl_printer_start_line(p);
	p = isl_printer_print_union_pw_multi_aff(p, band->pma);
	p = isl_printer_end_line(p);

	if (!isl_band_has_children(band))
		return p;

	children = isl_band_get_children(band);

	p = isl_printer_indent(p, 4);
	p = print_band_list(p, children);
	p = isl_printer_indent(p, -4);

	isl_band_list_free(children);

	return p;
}

static __isl_give isl_printer *print_band_list(__isl_take isl_printer *p,
	__isl_keep isl_band_list *list)
{
	int i, n;

	n = isl_band_list_n_band(list);
	for (i = 0; i < n; ++i) {
		isl_band *band;
		band = isl_band_list_get_band(list, i);
		p = print_band(p, band);
		isl_band_free(band);
	}

	return p;
}

/* Insert a band node with partial schedule "partial" between the domain
 * root node of "schedule" and its single child.
 * Return a pointer to the updated schedule.
 *
 * If any of the nodes in the tree depend on the set of outer band nodes
 * then we refuse to insert the band node.
 */
__isl_give isl_schedule *isl_schedule_insert_partial_schedule(
	__isl_take isl_schedule *schedule,
	__isl_take isl_multi_union_pw_aff *partial)
{
	isl_schedule_node *node;
	int anchored;

	node = isl_schedule_get_root(schedule);
	isl_schedule_free(schedule);
	if (!node)
		goto error;
	if (isl_schedule_node_get_type(node) != isl_schedule_node_domain)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_internal,
			"root node not a domain node", goto error);

	node = isl_schedule_node_child(node, 0);
	anchored = isl_schedule_node_is_subtree_anchored(node);
	if (anchored < 0)
		goto error;
	if (anchored)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"cannot insert band node in anchored subtree",
			goto error);
	node = isl_schedule_node_insert_partial_schedule(node, partial);

	schedule = isl_schedule_node_get_schedule(node);
	isl_schedule_node_free(node);

	return schedule;
error:
	isl_schedule_node_free(node);
	isl_multi_union_pw_aff_free(partial);
	return NULL;
}

/* Insert a context node with constraints "context" between the domain
 * root node of "schedule" and its single child.
 * Return a pointer to the updated schedule.
 */
__isl_give isl_schedule *isl_schedule_insert_context(
	__isl_take isl_schedule *schedule, __isl_take isl_set *context)
{
	isl_schedule_node *node;

	node = isl_schedule_get_root(schedule);
	isl_schedule_free(schedule);
	node = isl_schedule_node_child(node, 0);
	node = isl_schedule_node_insert_context(node, context);
	schedule = isl_schedule_node_get_schedule(node);
	isl_schedule_node_free(node);

	return schedule;
}

/* Insert a guard node with constraints "guard" between the domain
 * root node of "schedule" and its single child.
 * Return a pointer to the updated schedule.
 */
__isl_give isl_schedule *isl_schedule_insert_guard(
	__isl_take isl_schedule *schedule, __isl_take isl_set *guard)
{
	isl_schedule_node *node;

	node = isl_schedule_get_root(schedule);
	isl_schedule_free(schedule);
	node = isl_schedule_node_child(node, 0);
	node = isl_schedule_node_insert_guard(node, guard);
	schedule = isl_schedule_node_get_schedule(node);
	isl_schedule_node_free(node);

	return schedule;
}

/* Return a tree with as top-level node a filter corresponding to "filter" and
 * as child, the (single) child of "tree".
 * However, if this single child is of type "type", then the filter is inserted
 * in the children of this single child instead.
 */
static __isl_give isl_schedule_tree *insert_filter_in_child_of_type(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *filter,
	enum isl_schedule_node_type type)
{
	if (!isl_schedule_tree_has_children(tree)) {
		isl_schedule_tree_free(tree);
		return isl_schedule_tree_from_filter(filter);
	} else {
		tree = isl_schedule_tree_child(tree, 0);
	}

	if (isl_schedule_tree_get_type(tree) == type)
		tree = isl_schedule_tree_children_insert_filter(tree, filter);
	else
		tree = isl_schedule_tree_insert_filter(tree, filter);

	return tree;
}

/* Construct a schedule that combines the schedules "schedule1" and "schedule2"
 * with a top-level node (underneath the domain node) of type "type",
 * either isl_schedule_node_sequence or isl_schedule_node_set.
 * The domains of the two schedules are assumed to be disjoint.
 *
 * The new schedule has as domain the union of the domains of the two
 * schedules.  The child of the domain node is a node of type "type"
 * with two filters corresponding to the domains of the input schedules.
 * If one (or both) of the top-level nodes of the two schedules is itself
 * of type "type", then the filter is pushed into the children of that
 * node and the sequence of set is flattened.
 */
__isl_give isl_schedule *isl_schedule_pair(enum isl_schedule_node_type type,
	__isl_take isl_schedule *schedule1, __isl_take isl_schedule *schedule2)
{
	int disjoint;
	isl_ctx *ctx;
	enum isl_schedule_node_type root_type;
	isl_schedule_tree *tree1, *tree2;
	isl_union_set *filter1, *filter2, *domain;

	if (!schedule1 || !schedule2)
		goto error;

	root_type = isl_schedule_tree_get_type(schedule1->root);
	if (root_type != isl_schedule_node_domain)
		isl_die(isl_schedule_get_ctx(schedule1), isl_error_internal,
			"root node not a domain node", goto error);
	root_type = isl_schedule_tree_get_type(schedule2->root);
	if (root_type != isl_schedule_node_domain)
		isl_die(isl_schedule_get_ctx(schedule1), isl_error_internal,
			"root node not a domain node", goto error);

	ctx = isl_schedule_get_ctx(schedule1);
	tree1 = isl_schedule_tree_copy(schedule1->root);
	filter1 = isl_schedule_tree_domain_get_domain(tree1);
	tree2 = isl_schedule_tree_copy(schedule2->root);
	filter2 = isl_schedule_tree_domain_get_domain(tree2);

	isl_schedule_free(schedule1);
	isl_schedule_free(schedule2);

	disjoint = isl_union_set_is_disjoint(filter1, filter2);
	if (disjoint < 0)
		filter1 = isl_union_set_free(filter1);
	if (!disjoint)
		isl_die(ctx, isl_error_invalid,
			"schedule domains not disjoint",
			filter1 = isl_union_set_free(filter1));

	domain = isl_union_set_union(isl_union_set_copy(filter1),
				    isl_union_set_copy(filter2));
	filter1 = isl_union_set_gist(filter1, isl_union_set_copy(domain));
	filter2 = isl_union_set_gist(filter2, isl_union_set_copy(domain));

	tree1 = insert_filter_in_child_of_type(tree1, filter1, type);
	tree2 = insert_filter_in_child_of_type(tree2, filter2, type);

	tree1 = isl_schedule_tree_from_pair(type, tree1, tree2);
	tree1 = isl_schedule_tree_insert_domain(tree1, domain);

	return isl_schedule_from_schedule_tree(ctx, tree1);
error:
	isl_schedule_free(schedule1);
	isl_schedule_free(schedule2);
	return NULL;
}

/* Construct a schedule that combines the schedules "schedule1" and "schedule2"
 * through a sequence node.
 * The domains of the input schedules are assumed to be disjoint.
 */
__isl_give isl_schedule *isl_schedule_sequence(
	__isl_take isl_schedule *schedule1, __isl_take isl_schedule *schedule2)
{
	return isl_schedule_pair(isl_schedule_node_sequence,
				schedule1, schedule2);
}

/* Construct a schedule that combines the schedules "schedule1" and "schedule2"
 * through a set node.
 * The domains of the input schedules are assumed to be disjoint.
 */
__isl_give isl_schedule *isl_schedule_set(
	__isl_take isl_schedule *schedule1, __isl_take isl_schedule *schedule2)
{
	return isl_schedule_pair(isl_schedule_node_set, schedule1, schedule2);
}

/* Print "schedule" to "p".
 *
 * If "schedule" was created from a schedule tree, then we print
 * the schedule tree representation.  Otherwise, we print
 * the band forest representation.
 */
__isl_give isl_printer *isl_printer_print_schedule(__isl_take isl_printer *p,
	__isl_keep isl_schedule *schedule)
{
	isl_band_list *forest;

	if (!schedule)
		return isl_printer_free(p);

	if (schedule->root)
		return isl_printer_print_schedule_tree(p, schedule->root);

	forest = isl_schedule_get_band_forest(schedule);

	p = print_band_list(p, forest);

	isl_band_list_free(forest);

	return p;
}

#undef BASE
#define BASE schedule
#include <print_templ_yaml.c>
