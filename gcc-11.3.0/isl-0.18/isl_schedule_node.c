/*
 * Copyright 2013-2014 Ecole Normale Superieure
 * Copyright 2014      INRIA Rocquencourt
 * Copyright 2016      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 * and Inria Paris - Rocquencourt, Domaine de Voluceau - Rocquencourt,
 * B.P. 105 - 78153 Le Chesnay, France
 */

#include <isl/set.h>
#include <isl_schedule_band.h>
#include <isl_schedule_private.h>
#include <isl_schedule_node_private.h>

/* Create a new schedule node in the given schedule, point at the given
 * tree with given ancestors and child positions.
 * "child_pos" may be NULL if there are no ancestors.
 */
__isl_give isl_schedule_node *isl_schedule_node_alloc(
	__isl_take isl_schedule *schedule, __isl_take isl_schedule_tree *tree,
	__isl_take isl_schedule_tree_list *ancestors, int *child_pos)
{
	isl_ctx *ctx;
	isl_schedule_node *node;
	int i, n;

	if (!schedule || !tree || !ancestors)
		goto error;
	n = isl_schedule_tree_list_n_schedule_tree(ancestors);
	if (n > 0 && !child_pos)
		goto error;
	ctx = isl_schedule_get_ctx(schedule);
	node = isl_calloc_type(ctx, isl_schedule_node);
	if (!node)
		goto error;
	node->ref = 1;
	node->schedule = schedule;
	node->tree = tree;
	node->ancestors = ancestors;
	node->child_pos = isl_alloc_array(ctx, int, n);
	if (n && !node->child_pos)
		return isl_schedule_node_free(node);
	for (i = 0; i < n; ++i)
		node->child_pos[i] = child_pos[i];

	return node;
error:
	isl_schedule_free(schedule);
	isl_schedule_tree_free(tree);
	isl_schedule_tree_list_free(ancestors);
	return NULL;
}

/* Return a pointer to the root of a schedule tree with as single
 * node a domain node with the given domain.
 */
__isl_give isl_schedule_node *isl_schedule_node_from_domain(
	__isl_take isl_union_set *domain)
{
	isl_schedule *schedule;
	isl_schedule_node *node;

	schedule = isl_schedule_from_domain(domain);
	node = isl_schedule_get_root(schedule);
	isl_schedule_free(schedule);

	return node;
}

/* Return a pointer to the root of a schedule tree with as single
 * node a extension node with the given extension.
 */
__isl_give isl_schedule_node *isl_schedule_node_from_extension(
	__isl_take isl_union_map *extension)
{
	isl_ctx *ctx;
	isl_schedule *schedule;
	isl_schedule_tree *tree;
	isl_schedule_node *node;

	if (!extension)
		return NULL;

	ctx = isl_union_map_get_ctx(extension);
	tree = isl_schedule_tree_from_extension(extension);
	schedule = isl_schedule_from_schedule_tree(ctx, tree);
	node = isl_schedule_get_root(schedule);
	isl_schedule_free(schedule);

	return node;
}

/* Return the isl_ctx to which "node" belongs.
 */
isl_ctx *isl_schedule_node_get_ctx(__isl_keep isl_schedule_node *node)
{
	return node ? isl_schedule_get_ctx(node->schedule) : NULL;
}

/* Return a pointer to the leaf of the schedule into which "node" points.
 */
__isl_keep isl_schedule_tree *isl_schedule_node_peek_leaf(
	__isl_keep isl_schedule_node *node)
{
	return node ? isl_schedule_peek_leaf(node->schedule) : NULL;
}

/* Return a copy of the leaf of the schedule into which "node" points.
 */
__isl_give isl_schedule_tree *isl_schedule_node_get_leaf(
	__isl_keep isl_schedule_node *node)
{
	return isl_schedule_tree_copy(isl_schedule_node_peek_leaf(node));
}

/* Return the type of the node or isl_schedule_node_error on error.
 */
enum isl_schedule_node_type isl_schedule_node_get_type(
	__isl_keep isl_schedule_node *node)
{
	return node ? isl_schedule_tree_get_type(node->tree)
		    : isl_schedule_node_error;
}

/* Return the type of the parent of "node" or isl_schedule_node_error on error.
 */
enum isl_schedule_node_type isl_schedule_node_get_parent_type(
	__isl_keep isl_schedule_node *node)
{
	int pos;
	int has_parent;
	isl_schedule_tree *parent;
	enum isl_schedule_node_type type;

	if (!node)
		return isl_schedule_node_error;
	has_parent = isl_schedule_node_has_parent(node);
	if (has_parent < 0)
		return isl_schedule_node_error;
	if (!has_parent)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"node has no parent", return isl_schedule_node_error);

	pos = isl_schedule_tree_list_n_schedule_tree(node->ancestors) - 1;
	parent = isl_schedule_tree_list_get_schedule_tree(node->ancestors, pos);
	type = isl_schedule_tree_get_type(parent);
	isl_schedule_tree_free(parent);

	return type;
}

/* Return a copy of the subtree that this node points to.
 */
__isl_give isl_schedule_tree *isl_schedule_node_get_tree(
	__isl_keep isl_schedule_node *node)
{
	if (!node)
		return NULL;

	return isl_schedule_tree_copy(node->tree);
}

/* Return a copy of the schedule into which "node" points.
 */
__isl_give isl_schedule *isl_schedule_node_get_schedule(
	__isl_keep isl_schedule_node *node)
{
	if (!node)
		return NULL;
	return isl_schedule_copy(node->schedule);
}

/* Return a fresh copy of "node".
 */
__isl_take isl_schedule_node *isl_schedule_node_dup(
	__isl_keep isl_schedule_node *node)
{
	if (!node)
		return NULL;

	return isl_schedule_node_alloc(isl_schedule_copy(node->schedule),
				isl_schedule_tree_copy(node->tree),
				isl_schedule_tree_list_copy(node->ancestors),
				node->child_pos);
}

/* Return an isl_schedule_node that is equal to "node" and that has only
 * a single reference.
 */
__isl_give isl_schedule_node *isl_schedule_node_cow(
	__isl_take isl_schedule_node *node)
{
	if (!node)
		return NULL;

	if (node->ref == 1)
		return node;
	node->ref--;
	return isl_schedule_node_dup(node);
}

/* Return a new reference to "node".
 */
__isl_give isl_schedule_node *isl_schedule_node_copy(
	__isl_keep isl_schedule_node *node)
{
	if (!node)
		return NULL;

	node->ref++;
	return node;
}

/* Free "node" and return NULL.
 */
__isl_null isl_schedule_node *isl_schedule_node_free(
	__isl_take isl_schedule_node *node)
{
	if (!node)
		return NULL;
	if (--node->ref > 0)
		return NULL;

	isl_schedule_tree_list_free(node->ancestors);
	free(node->child_pos);
	isl_schedule_tree_free(node->tree);
	isl_schedule_free(node->schedule);
	free(node);

	return NULL;
}

/* Do "node1" and "node2" point to the same position in the same
 * schedule?
 */
isl_bool isl_schedule_node_is_equal(__isl_keep isl_schedule_node *node1,
	__isl_keep isl_schedule_node *node2)
{
	int i, n1, n2;

	if (!node1 || !node2)
		return isl_bool_error;
	if (node1 == node2)
		return isl_bool_true;
	if (node1->schedule != node2->schedule)
		return isl_bool_false;

	n1 = isl_schedule_node_get_tree_depth(node1);
	n2 = isl_schedule_node_get_tree_depth(node2);
	if (n1 != n2)
		return isl_bool_false;
	for (i = 0; i < n1; ++i)
		if (node1->child_pos[i] != node2->child_pos[i])
			return isl_bool_false;

	return isl_bool_true;
}

/* Return the number of outer schedule dimensions of "node"
 * in its schedule tree.
 *
 * Return -1 on error.
 */
int isl_schedule_node_get_schedule_depth(__isl_keep isl_schedule_node *node)
{
	int i, n;
	int depth = 0;

	if (!node)
		return -1;

	n = isl_schedule_tree_list_n_schedule_tree(node->ancestors);
	for (i = n - 1; i >= 0; --i) {
		isl_schedule_tree *tree;

		tree = isl_schedule_tree_list_get_schedule_tree(
						    node->ancestors, i);
		if (!tree)
			return -1;
		if (tree->type == isl_schedule_node_band)
			depth += isl_schedule_tree_band_n_member(tree);
		isl_schedule_tree_free(tree);
	}

	return depth;
}

/* Internal data structure for
 * isl_schedule_node_get_prefix_schedule_union_pw_multi_aff
 *
 * "initialized" is set if the filter field has been initialized.
 * If "universe_domain" is not set, then the collected filter is intersected
 * with the the domain of the root domain node.
 * "universe_filter" is set if we are only collecting the universes of filters
 * "collect_prefix" is set if we are collecting prefixes.
 * "filter" collects all outer filters and is NULL until "initialized" is set.
 * "prefix" collects all outer band partial schedules (if "collect_prefix"
 * is set).  If it is used, then it is initialized by the caller
 * of collect_filter_prefix to a zero-dimensional function.
 */
struct isl_schedule_node_get_filter_prefix_data {
	int initialized;
	int universe_domain;
	int universe_filter;
	int collect_prefix;
	isl_union_set *filter;
	isl_multi_union_pw_aff *prefix;
};

static int collect_filter_prefix(__isl_keep isl_schedule_tree_list *list,
	int n, struct isl_schedule_node_get_filter_prefix_data *data);

/* Update the filter and prefix information in "data" based on the first "n"
 * elements in "list" and the expansion tree root "tree".
 *
 * We first collect the information from the elements in "list",
 * initializing the filter based on the domain of the expansion.
 * Then we map the results to the expanded space and combined them
 * with the results already in "data".
 */
static int collect_filter_prefix_expansion(__isl_take isl_schedule_tree *tree,
	__isl_keep isl_schedule_tree_list *list, int n,
	struct isl_schedule_node_get_filter_prefix_data *data)
{
	struct isl_schedule_node_get_filter_prefix_data contracted;
	isl_union_pw_multi_aff *c;
	isl_union_map *exp, *universe;
	isl_union_set *filter;

	c = isl_schedule_tree_expansion_get_contraction(tree);
	exp = isl_schedule_tree_expansion_get_expansion(tree);

	contracted.initialized = 1;
	contracted.universe_domain = data->universe_domain;
	contracted.universe_filter = data->universe_filter;
	contracted.collect_prefix = data->collect_prefix;
	universe = isl_union_map_universe(isl_union_map_copy(exp));
	filter = isl_union_map_domain(universe);
	if (data->collect_prefix) {
		isl_space *space = isl_union_set_get_space(filter);
		space = isl_space_set_from_params(space);
		contracted.prefix = isl_multi_union_pw_aff_zero(space);
	}
	contracted.filter = filter;

	if (collect_filter_prefix(list, n, &contracted) < 0)
		contracted.filter = isl_union_set_free(contracted.filter);
	if (data->collect_prefix) {
		isl_multi_union_pw_aff *prefix;

		prefix = contracted.prefix;
		prefix =
		    isl_multi_union_pw_aff_pullback_union_pw_multi_aff(prefix,
						isl_union_pw_multi_aff_copy(c));
		data->prefix = isl_multi_union_pw_aff_flat_range_product(
						prefix, data->prefix);
	}
	filter = contracted.filter;
	if (data->universe_domain)
		filter = isl_union_set_preimage_union_pw_multi_aff(filter,
						isl_union_pw_multi_aff_copy(c));
	else
		filter = isl_union_set_apply(filter, isl_union_map_copy(exp));
	if (!data->initialized)
		data->filter = filter;
	else
		data->filter = isl_union_set_intersect(filter, data->filter);
	data->initialized = 1;

	isl_union_pw_multi_aff_free(c);
	isl_union_map_free(exp);
	isl_schedule_tree_free(tree);

	return 0;
}

/* Update the filter information in "data" based on the first "n"
 * elements in "list" and the extension tree root "tree", in case
 * data->universe_domain is set and data->collect_prefix is not.
 *
 * We collect the universe domain of the elements in "list" and
 * add it to the universe range of the extension (intersected
 * with the already collected filter, if any).
 */
static int collect_universe_domain_extension(__isl_take isl_schedule_tree *tree,
	__isl_keep isl_schedule_tree_list *list, int n,
	struct isl_schedule_node_get_filter_prefix_data *data)
{
	struct isl_schedule_node_get_filter_prefix_data data_outer;
	isl_union_map *extension;
	isl_union_set *filter;

	data_outer.initialized = 0;
	data_outer.universe_domain = 1;
	data_outer.universe_filter = data->universe_filter;
	data_outer.collect_prefix = 0;
	data_outer.filter = NULL;
	data_outer.prefix = NULL;

	if (collect_filter_prefix(list, n, &data_outer) < 0)
		data_outer.filter = isl_union_set_free(data_outer.filter);

	extension = isl_schedule_tree_extension_get_extension(tree);
	extension = isl_union_map_universe(extension);
	filter = isl_union_map_range(extension);
	if (data_outer.initialized)
		filter = isl_union_set_union(filter, data_outer.filter);
	if (data->initialized)
		filter = isl_union_set_intersect(filter, data->filter);

	data->filter = filter;

	isl_schedule_tree_free(tree);

	return 0;
}

/* Update "data" based on the tree node "tree" in case "data" has
 * not been initialized yet.
 *
 * Return 0 on success and -1 on error.
 *
 * If "tree" is a filter, then we set data->filter to this filter
 * (or its universe).
 * If "tree" is a domain, then this means we have reached the root
 * of the schedule tree without being able to extract any information.
 * We therefore initialize data->filter to the universe of the domain,
 * or the domain itself if data->universe_domain is not set.
 * If "tree" is a band with at least one member, then we set data->filter
 * to the universe of the schedule domain and replace the zero-dimensional
 * data->prefix by the band schedule (if data->collect_prefix is set).
 */
static int collect_filter_prefix_init(__isl_keep isl_schedule_tree *tree,
	struct isl_schedule_node_get_filter_prefix_data *data)
{
	enum isl_schedule_node_type type;
	isl_multi_union_pw_aff *mupa;
	isl_union_set *filter;

	type = isl_schedule_tree_get_type(tree);
	switch (type) {
	case isl_schedule_node_error:
		return -1;
	case isl_schedule_node_expansion:
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_internal,
			"should be handled by caller", return -1);
	case isl_schedule_node_extension:
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"cannot handle extension nodes", return -1);
	case isl_schedule_node_context:
	case isl_schedule_node_leaf:
	case isl_schedule_node_guard:
	case isl_schedule_node_mark:
	case isl_schedule_node_sequence:
	case isl_schedule_node_set:
		return 0;
	case isl_schedule_node_domain:
		filter = isl_schedule_tree_domain_get_domain(tree);
		if (data->universe_domain)
			filter = isl_union_set_universe(filter);
		data->filter = filter;
		break;
	case isl_schedule_node_band:
		if (isl_schedule_tree_band_n_member(tree) == 0)
			return 0;
		mupa = isl_schedule_tree_band_get_partial_schedule(tree);
		if (data->collect_prefix) {
			isl_multi_union_pw_aff_free(data->prefix);
			mupa = isl_multi_union_pw_aff_reset_tuple_id(mupa,
								isl_dim_set);
			data->prefix = isl_multi_union_pw_aff_copy(mupa);
		}
		filter = isl_multi_union_pw_aff_domain(mupa);
		filter = isl_union_set_universe(filter);
		data->filter = filter;
		break;
	case isl_schedule_node_filter:
		filter = isl_schedule_tree_filter_get_filter(tree);
		if (data->universe_filter)
			filter = isl_union_set_universe(filter);
		data->filter = filter;
		break;
	}

	if ((data->collect_prefix && !data->prefix) || !data->filter)
		return -1;

	data->initialized = 1;

	return 0;
}

/* Update "data" based on the tree node "tree" in case "data" has
 * already been initialized.
 *
 * Return 0 on success and -1 on error.
 *
 * If "tree" is a domain and data->universe_domain is not set, then
 * intersect data->filter with the domain.
 * If "tree" is a filter, then we intersect data->filter with this filter
 * (or its universe).
 * If "tree" is a band with at least one member and data->collect_prefix
 * is set, then we extend data->prefix with the band schedule.
 * If "tree" is an extension, then we make sure that we are not collecting
 * information on any extended domain elements.
 */
static int collect_filter_prefix_update(__isl_keep isl_schedule_tree *tree,
	struct isl_schedule_node_get_filter_prefix_data *data)
{
	enum isl_schedule_node_type type;
	isl_multi_union_pw_aff *mupa;
	isl_union_set *filter;
	isl_union_map *extension;
	int empty;

	type = isl_schedule_tree_get_type(tree);
	switch (type) {
	case isl_schedule_node_error:
		return -1;
	case isl_schedule_node_expansion:
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_internal,
			"should be handled by caller", return -1);
	case isl_schedule_node_extension:
		extension = isl_schedule_tree_extension_get_extension(tree);
		extension = isl_union_map_intersect_range(extension,
					isl_union_set_copy(data->filter));
		empty = isl_union_map_is_empty(extension);
		isl_union_map_free(extension);
		if (empty < 0)
			return -1;
		if (empty)
			break;
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"cannot handle extension nodes", return -1);
	case isl_schedule_node_context:
	case isl_schedule_node_leaf:
	case isl_schedule_node_guard:
	case isl_schedule_node_mark:
	case isl_schedule_node_sequence:
	case isl_schedule_node_set:
		break;
	case isl_schedule_node_domain:
		if (data->universe_domain)
			break;
		filter = isl_schedule_tree_domain_get_domain(tree);
		data->filter = isl_union_set_intersect(data->filter, filter);
		break;
	case isl_schedule_node_band:
		if (isl_schedule_tree_band_n_member(tree) == 0)
			break;
		if (!data->collect_prefix)
			break;
		mupa = isl_schedule_tree_band_get_partial_schedule(tree);
		data->prefix = isl_multi_union_pw_aff_flat_range_product(mupa,
								data->prefix);
		if (!data->prefix)
			return -1;
		break;
	case isl_schedule_node_filter:
		filter = isl_schedule_tree_filter_get_filter(tree);
		if (data->universe_filter)
			filter = isl_union_set_universe(filter);
		data->filter = isl_union_set_intersect(data->filter, filter);
		if (!data->filter)
			return -1;
		break;
	}

	return 0;
}

/* Collect filter and/or prefix information from the first "n"
 * elements in "list" (which represent the ancestors of a node).
 * Store the results in "data".
 *
 * Extension nodes are only supported if they do not affect the outcome,
 * i.e., if we are collecting information on non-extended domain elements,
 * or if we are collecting the universe domain (without prefix).
 *
 * Return 0 on success and -1 on error.
 *
 * We traverse the list from innermost ancestor (last element)
 * to outermost ancestor (first element), calling collect_filter_prefix_init
 * on each node as long as we have not been able to extract any information
 * yet and collect_filter_prefix_update afterwards.
 * If we come across an expansion node, then we interrupt the traversal
 * and call collect_filter_prefix_expansion to restart the traversal
 * over the remaining ancestors and to combine the results with those
 * that have already been collected.
 * If we come across an extension node and we are only computing
 * the universe domain, then we interrupt the traversal and call
 * collect_universe_domain_extension to restart the traversal
 * over the remaining ancestors and to combine the results with those
 * that have already been collected.
 * On successful return, data->initialized will be set since the outermost
 * ancestor is a domain node, which always results in an initialization.
 */
static int collect_filter_prefix(__isl_keep isl_schedule_tree_list *list,
	int n, struct isl_schedule_node_get_filter_prefix_data *data)
{
	int i;

	if (!list)
		return -1;

	for (i = n - 1; i >= 0; --i) {
		isl_schedule_tree *tree;
		enum isl_schedule_node_type type;
		int r;

		tree = isl_schedule_tree_list_get_schedule_tree(list, i);
		if (!tree)
			return -1;
		type = isl_schedule_tree_get_type(tree);
		if (type == isl_schedule_node_expansion)
			return collect_filter_prefix_expansion(tree, list, i,
								data);
		if (type == isl_schedule_node_extension &&
		    data->universe_domain && !data->collect_prefix)
			return collect_universe_domain_extension(tree, list, i,
								data);
		if (!data->initialized)
			r = collect_filter_prefix_init(tree, data);
		else
			r = collect_filter_prefix_update(tree, data);
		isl_schedule_tree_free(tree);
		if (r < 0)
			return -1;
	}

	return 0;
}

/* Return the concatenation of the partial schedules of all outer band
 * nodes of "node" interesected with all outer filters
 * as an isl_multi_union_pw_aff.
 * None of the ancestors of "node" may be an extension node, unless
 * there is also a filter ancestor that filters out all the extended
 * domain elements.
 *
 * If "node" is pointing at the root of the schedule tree, then
 * there are no domain elements reaching the current node, so
 * we return an empty result.
 *
 * We collect all the filters and partial schedules in collect_filter_prefix
 * and intersect the domain of the combined schedule with the combined filter.
 */
__isl_give isl_multi_union_pw_aff *
isl_schedule_node_get_prefix_schedule_multi_union_pw_aff(
	__isl_keep isl_schedule_node *node)
{
	int n;
	isl_space *space;
	struct isl_schedule_node_get_filter_prefix_data data;

	if (!node)
		return NULL;

	space = isl_schedule_get_space(node->schedule);
	space = isl_space_set_from_params(space);
	if (node->tree == node->schedule->root)
		return isl_multi_union_pw_aff_zero(space);

	data.initialized = 0;
	data.universe_domain = 1;
	data.universe_filter = 0;
	data.collect_prefix = 1;
	data.filter = NULL;
	data.prefix = isl_multi_union_pw_aff_zero(space);

	n = isl_schedule_tree_list_n_schedule_tree(node->ancestors);
	if (collect_filter_prefix(node->ancestors, n, &data) < 0)
		data.prefix = isl_multi_union_pw_aff_free(data.prefix);

	data.prefix = isl_multi_union_pw_aff_intersect_domain(data.prefix,
								data.filter);

	return data.prefix;
}

/* Return the concatenation of the partial schedules of all outer band
 * nodes of "node" interesected with all outer filters
 * as an isl_union_pw_multi_aff.
 * None of the ancestors of "node" may be an extension node, unless
 * there is also a filter ancestor that filters out all the extended
 * domain elements.
 *
 * If "node" is pointing at the root of the schedule tree, then
 * there are no domain elements reaching the current node, so
 * we return an empty result.
 *
 * We collect all the filters and partial schedules in collect_filter_prefix.
 * The partial schedules are collected as an isl_multi_union_pw_aff.
 * If this isl_multi_union_pw_aff is zero-dimensional, then it does not
 * contain any domain information, so we construct the isl_union_pw_multi_aff
 * result as a zero-dimensional function on the collected filter.
 * Otherwise, we convert the isl_multi_union_pw_aff to
 * an isl_multi_union_pw_aff and intersect the domain with the filter.
 */
__isl_give isl_union_pw_multi_aff *
isl_schedule_node_get_prefix_schedule_union_pw_multi_aff(
	__isl_keep isl_schedule_node *node)
{
	int n;
	isl_space *space;
	isl_union_pw_multi_aff *prefix;
	struct isl_schedule_node_get_filter_prefix_data data;

	if (!node)
		return NULL;

	space = isl_schedule_get_space(node->schedule);
	if (node->tree == node->schedule->root)
		return isl_union_pw_multi_aff_empty(space);

	space = isl_space_set_from_params(space);
	data.initialized = 0;
	data.universe_domain = 1;
	data.universe_filter = 0;
	data.collect_prefix = 1;
	data.filter = NULL;
	data.prefix = isl_multi_union_pw_aff_zero(space);

	n = isl_schedule_tree_list_n_schedule_tree(node->ancestors);
	if (collect_filter_prefix(node->ancestors, n, &data) < 0)
		data.prefix = isl_multi_union_pw_aff_free(data.prefix);

	if (data.prefix &&
	    isl_multi_union_pw_aff_dim(data.prefix, isl_dim_set) == 0) {
		isl_multi_union_pw_aff_free(data.prefix);
		prefix = isl_union_pw_multi_aff_from_domain(data.filter);
	} else {
		prefix =
		    isl_union_pw_multi_aff_from_multi_union_pw_aff(data.prefix);
		prefix = isl_union_pw_multi_aff_intersect_domain(prefix,
								data.filter);
	}

	return prefix;
}

/* Return the concatenation of the partial schedules of all outer band
 * nodes of "node" interesected with all outer filters
 * as an isl_union_map.
 */
__isl_give isl_union_map *isl_schedule_node_get_prefix_schedule_union_map(
	__isl_keep isl_schedule_node *node)
{
	isl_union_pw_multi_aff *upma;

	upma = isl_schedule_node_get_prefix_schedule_union_pw_multi_aff(node);
	return isl_union_map_from_union_pw_multi_aff(upma);
}

/* Return the concatenation of the partial schedules of all outer band
 * nodes of "node" intersected with all outer domain constraints.
 * None of the ancestors of "node" may be an extension node, unless
 * there is also a filter ancestor that filters out all the extended
 * domain elements.
 *
 * Essentially, this function intersects the domain of the output
 * of isl_schedule_node_get_prefix_schedule_union_map with the output
 * of isl_schedule_node_get_domain, except that it only traverses
 * the ancestors of "node" once.
 */
__isl_give isl_union_map *isl_schedule_node_get_prefix_schedule_relation(
	__isl_keep isl_schedule_node *node)
{
	int n;
	isl_space *space;
	isl_union_map *prefix;
	struct isl_schedule_node_get_filter_prefix_data data;

	if (!node)
		return NULL;

	space = isl_schedule_get_space(node->schedule);
	if (node->tree == node->schedule->root)
		return isl_union_map_empty(space);

	space = isl_space_set_from_params(space);
	data.initialized = 0;
	data.universe_domain = 0;
	data.universe_filter = 0;
	data.collect_prefix = 1;
	data.filter = NULL;
	data.prefix = isl_multi_union_pw_aff_zero(space);

	n = isl_schedule_tree_list_n_schedule_tree(node->ancestors);
	if (collect_filter_prefix(node->ancestors, n, &data) < 0)
		data.prefix = isl_multi_union_pw_aff_free(data.prefix);

	if (data.prefix &&
	    isl_multi_union_pw_aff_dim(data.prefix, isl_dim_set) == 0) {
		isl_multi_union_pw_aff_free(data.prefix);
		prefix = isl_union_map_from_domain(data.filter);
	} else {
		prefix = isl_union_map_from_multi_union_pw_aff(data.prefix);
		prefix = isl_union_map_intersect_domain(prefix, data.filter);
	}

	return prefix;
}

/* Return the domain elements that reach "node".
 *
 * If "node" is pointing at the root of the schedule tree, then
 * there are no domain elements reaching the current node, so
 * we return an empty result.
 * None of the ancestors of "node" may be an extension node, unless
 * there is also a filter ancestor that filters out all the extended
 * domain elements.
 *
 * Otherwise, we collect all filters reaching the node,
 * intersected with the root domain in collect_filter_prefix.
 */
__isl_give isl_union_set *isl_schedule_node_get_domain(
	__isl_keep isl_schedule_node *node)
{
	int n;
	struct isl_schedule_node_get_filter_prefix_data data;

	if (!node)
		return NULL;

	if (node->tree == node->schedule->root) {
		isl_space *space;

		space = isl_schedule_get_space(node->schedule);
		return isl_union_set_empty(space);
	}

	data.initialized = 0;
	data.universe_domain = 0;
	data.universe_filter = 0;
	data.collect_prefix = 0;
	data.filter = NULL;
	data.prefix = NULL;

	n = isl_schedule_tree_list_n_schedule_tree(node->ancestors);
	if (collect_filter_prefix(node->ancestors, n, &data) < 0)
		data.filter = isl_union_set_free(data.filter);

	return data.filter;
}

/* Return the union of universe sets of the domain elements that reach "node".
 *
 * If "node" is pointing at the root of the schedule tree, then
 * there are no domain elements reaching the current node, so
 * we return an empty result.
 *
 * Otherwise, we collect the universes of all filters reaching the node
 * in collect_filter_prefix.
 */
__isl_give isl_union_set *isl_schedule_node_get_universe_domain(
	__isl_keep isl_schedule_node *node)
{
	int n;
	struct isl_schedule_node_get_filter_prefix_data data;

	if (!node)
		return NULL;

	if (node->tree == node->schedule->root) {
		isl_space *space;

		space = isl_schedule_get_space(node->schedule);
		return isl_union_set_empty(space);
	}

	data.initialized = 0;
	data.universe_domain = 1;
	data.universe_filter = 1;
	data.collect_prefix = 0;
	data.filter = NULL;
	data.prefix = NULL;

	n = isl_schedule_tree_list_n_schedule_tree(node->ancestors);
	if (collect_filter_prefix(node->ancestors, n, &data) < 0)
		data.filter = isl_union_set_free(data.filter);

	return data.filter;
}

/* Return the subtree schedule of "node".
 *
 * Since isl_schedule_tree_get_subtree_schedule_union_map does not handle
 * trees that do not contain any schedule information, we first
 * move down to the first relevant descendant and handle leaves ourselves.
 *
 * If the subtree rooted at "node" contains any expansion nodes, then
 * the returned subtree schedule is formulated in terms of the expanded
 * domains.
 * The subtree is not allowed to contain any extension nodes.
 */
__isl_give isl_union_map *isl_schedule_node_get_subtree_schedule_union_map(
	__isl_keep isl_schedule_node *node)
{
	isl_schedule_tree *tree, *leaf;
	isl_union_map *umap;

	tree = isl_schedule_node_get_tree(node);
	leaf = isl_schedule_node_peek_leaf(node);
	tree = isl_schedule_tree_first_schedule_descendant(tree, leaf);
	if (!tree)
		return NULL;
	if (tree == leaf) {
		isl_union_set *domain;
		domain = isl_schedule_node_get_universe_domain(node);
		isl_schedule_tree_free(tree);
		return isl_union_map_from_domain(domain);
	}

	umap = isl_schedule_tree_get_subtree_schedule_union_map(tree);
	isl_schedule_tree_free(tree);
	return umap;
}

/* Return the number of ancestors of "node" in its schedule tree.
 */
int isl_schedule_node_get_tree_depth(__isl_keep isl_schedule_node *node)
{
	if (!node)
		return -1;
	return isl_schedule_tree_list_n_schedule_tree(node->ancestors);
}

/* Does "node" have a parent?
 *
 * That is, does it point to any node of the schedule other than the root?
 */
isl_bool isl_schedule_node_has_parent(__isl_keep isl_schedule_node *node)
{
	if (!node)
		return isl_bool_error;
	if (!node->ancestors)
		return isl_bool_error;

	return isl_schedule_tree_list_n_schedule_tree(node->ancestors) != 0;
}

/* Return the position of "node" among the children of its parent.
 */
int isl_schedule_node_get_child_position(__isl_keep isl_schedule_node *node)
{
	int n;
	int has_parent;

	if (!node)
		return -1;
	has_parent = isl_schedule_node_has_parent(node);
	if (has_parent < 0)
		return -1;
	if (!has_parent)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"node has no parent", return -1);

	n = isl_schedule_tree_list_n_schedule_tree(node->ancestors);
	return node->child_pos[n - 1];
}

/* Does the parent (if any) of "node" have any children with a smaller child
 * position than this one?
 */
isl_bool isl_schedule_node_has_previous_sibling(
	__isl_keep isl_schedule_node *node)
{
	int n;
	isl_bool has_parent;

	if (!node)
		return isl_bool_error;
	has_parent = isl_schedule_node_has_parent(node);
	if (has_parent < 0 || !has_parent)
		return has_parent;

	n = isl_schedule_tree_list_n_schedule_tree(node->ancestors);

	return node->child_pos[n - 1] > 0;
}

/* Does the parent (if any) of "node" have any children with a greater child
 * position than this one?
 */
isl_bool isl_schedule_node_has_next_sibling(__isl_keep isl_schedule_node *node)
{
	int n, n_child;
	isl_bool has_parent;
	isl_schedule_tree *tree;

	if (!node)
		return isl_bool_error;
	has_parent = isl_schedule_node_has_parent(node);
	if (has_parent < 0 || !has_parent)
		return has_parent;

	n = isl_schedule_tree_list_n_schedule_tree(node->ancestors);
	tree = isl_schedule_tree_list_get_schedule_tree(node->ancestors, n - 1);
	if (!tree)
		return isl_bool_error;
	n_child = isl_schedule_tree_list_n_schedule_tree(tree->children);
	isl_schedule_tree_free(tree);

	return node->child_pos[n - 1] + 1 < n_child;
}

/* Does "node" have any children?
 *
 * Any node other than the leaf nodes is considered to have at least
 * one child, even if the corresponding isl_schedule_tree does not
 * have any children.
 */
isl_bool isl_schedule_node_has_children(__isl_keep isl_schedule_node *node)
{
	if (!node)
		return isl_bool_error;
	return !isl_schedule_tree_is_leaf(node->tree);
}

/* Return the number of children of "node"?
 *
 * Any node other than the leaf nodes is considered to have at least
 * one child, even if the corresponding isl_schedule_tree does not
 * have any children.  That is, the number of children of "node" is
 * only zero if its tree is the explicit empty tree.  Otherwise,
 * if the isl_schedule_tree has any children, then it is equal
 * to the number of children of "node".  If it has zero children,
 * then "node" still has a leaf node as child.
 */
int isl_schedule_node_n_children(__isl_keep isl_schedule_node *node)
{
	int n;

	if (!node)
		return -1;

	if (isl_schedule_tree_is_leaf(node->tree))
		return 0;

	n = isl_schedule_tree_n_children(node->tree);
	if (n == 0)
		return 1;

	return n;
}

/* Move the "node" pointer to the ancestor of the given generation
 * of the node it currently points to, where generation 0 is the node
 * itself and generation 1 is its parent.
 */
__isl_give isl_schedule_node *isl_schedule_node_ancestor(
	__isl_take isl_schedule_node *node, int generation)
{
	int n;
	isl_schedule_tree *tree;

	if (!node)
		return NULL;
	if (generation == 0)
		return node;
	n = isl_schedule_node_get_tree_depth(node);
	if (n < 0)
		return isl_schedule_node_free(node);
	if (generation < 0 || generation > n)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"generation out of bounds",
			return isl_schedule_node_free(node));
	node = isl_schedule_node_cow(node);
	if (!node)
		return NULL;

	tree = isl_schedule_tree_list_get_schedule_tree(node->ancestors,
							n - generation);
	isl_schedule_tree_free(node->tree);
	node->tree = tree;
	node->ancestors = isl_schedule_tree_list_drop(node->ancestors,
						    n - generation, generation);
	if (!node->ancestors || !node->tree)
		return isl_schedule_node_free(node);

	return node;
}

/* Move the "node" pointer to the parent of the node it currently points to.
 */
__isl_give isl_schedule_node *isl_schedule_node_parent(
	__isl_take isl_schedule_node *node)
{
	if (!node)
		return NULL;
	if (!isl_schedule_node_has_parent(node))
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"node has no parent",
			return isl_schedule_node_free(node));
	return isl_schedule_node_ancestor(node, 1);
}

/* Move the "node" pointer to the root of its schedule tree.
 */
__isl_give isl_schedule_node *isl_schedule_node_root(
	__isl_take isl_schedule_node *node)
{
	int n;

	if (!node)
		return NULL;
	n = isl_schedule_node_get_tree_depth(node);
	if (n < 0)
		return isl_schedule_node_free(node);
	return isl_schedule_node_ancestor(node, n);
}

/* Move the "node" pointer to the child at position "pos" of the node
 * it currently points to.
 */
__isl_give isl_schedule_node *isl_schedule_node_child(
	__isl_take isl_schedule_node *node, int pos)
{
	int n;
	isl_ctx *ctx;
	isl_schedule_tree *tree;
	int *child_pos;

	node = isl_schedule_node_cow(node);
	if (!node)
		return NULL;
	if (!isl_schedule_node_has_children(node))
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"node has no children",
			return isl_schedule_node_free(node));

	ctx = isl_schedule_node_get_ctx(node);
	n = isl_schedule_tree_list_n_schedule_tree(node->ancestors);
	child_pos = isl_realloc_array(ctx, node->child_pos, int, n + 1);
	if (!child_pos)
		return isl_schedule_node_free(node);
	node->child_pos = child_pos;
	node->child_pos[n] = pos;

	node->ancestors = isl_schedule_tree_list_add(node->ancestors,
				isl_schedule_tree_copy(node->tree));
	tree = node->tree;
	if (isl_schedule_tree_has_children(tree))
		tree = isl_schedule_tree_get_child(tree, pos);
	else
		tree = isl_schedule_node_get_leaf(node);
	isl_schedule_tree_free(node->tree);
	node->tree = tree;

	if (!node->tree || !node->ancestors)
		return isl_schedule_node_free(node);

	return node;
}

/* Move the "node" pointer to the first child of the node
 * it currently points to.
 */
__isl_give isl_schedule_node *isl_schedule_node_first_child(
	__isl_take isl_schedule_node *node)
{
	return isl_schedule_node_child(node, 0);
}

/* Move the "node" pointer to the child of this node's parent in
 * the previous child position.
 */
__isl_give isl_schedule_node *isl_schedule_node_previous_sibling(
	__isl_take isl_schedule_node *node)
{
	int n;
	isl_schedule_tree *parent, *tree;

	node = isl_schedule_node_cow(node);
	if (!node)
		return NULL;
	if (!isl_schedule_node_has_previous_sibling(node))
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"node has no previous sibling",
			return isl_schedule_node_free(node));

	n = isl_schedule_tree_list_n_schedule_tree(node->ancestors);
	parent = isl_schedule_tree_list_get_schedule_tree(node->ancestors,
									n - 1);
	if (!parent)
		return isl_schedule_node_free(node);
	node->child_pos[n - 1]--;
	tree = isl_schedule_tree_list_get_schedule_tree(parent->children,
							node->child_pos[n - 1]);
	isl_schedule_tree_free(parent);
	if (!tree)
		return isl_schedule_node_free(node);
	isl_schedule_tree_free(node->tree);
	node->tree = tree;

	return node;
}

/* Move the "node" pointer to the child of this node's parent in
 * the next child position.
 */
__isl_give isl_schedule_node *isl_schedule_node_next_sibling(
	__isl_take isl_schedule_node *node)
{
	int n;
	isl_schedule_tree *parent, *tree;

	node = isl_schedule_node_cow(node);
	if (!node)
		return NULL;
	if (!isl_schedule_node_has_next_sibling(node))
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"node has no next sibling",
			return isl_schedule_node_free(node));

	n = isl_schedule_tree_list_n_schedule_tree(node->ancestors);
	parent = isl_schedule_tree_list_get_schedule_tree(node->ancestors,
									n - 1);
	if (!parent)
		return isl_schedule_node_free(node);
	node->child_pos[n - 1]++;
	tree = isl_schedule_tree_list_get_schedule_tree(parent->children,
							node->child_pos[n - 1]);
	isl_schedule_tree_free(parent);
	if (!tree)
		return isl_schedule_node_free(node);
	isl_schedule_tree_free(node->tree);
	node->tree = tree;

	return node;
}

/* Return a copy to the child at position "pos" of "node".
 */
__isl_give isl_schedule_node *isl_schedule_node_get_child(
	__isl_keep isl_schedule_node *node, int pos)
{
	return isl_schedule_node_child(isl_schedule_node_copy(node), pos);
}

/* Traverse the descendant of "node" in depth-first order, including
 * "node" itself.  Call "enter" whenever a node is entered and "leave"
 * whenever a node is left.  The callback "enter" is responsible
 * for moving to the deepest initial subtree of its argument that
 * should be traversed.
 */
static __isl_give isl_schedule_node *traverse(
	__isl_take isl_schedule_node *node,
	__isl_give isl_schedule_node *(*enter)(
		__isl_take isl_schedule_node *node, void *user),
	__isl_give isl_schedule_node *(*leave)(
		__isl_take isl_schedule_node *node, void *user),
	void *user)
{
	int depth;

	if (!node)
		return NULL;

	depth = isl_schedule_node_get_tree_depth(node);
	do {
		node = enter(node, user);
		node = leave(node, user);
		while (node && isl_schedule_node_get_tree_depth(node) > depth &&
				!isl_schedule_node_has_next_sibling(node)) {
			node = isl_schedule_node_parent(node);
			node = leave(node, user);
		}
		if (node && isl_schedule_node_get_tree_depth(node) > depth)
			node = isl_schedule_node_next_sibling(node);
	} while (node && isl_schedule_node_get_tree_depth(node) > depth);

	return node;
}

/* Internal data structure for isl_schedule_node_foreach_descendant_top_down.
 *
 * "fn" is the user-specified callback function.
 * "user" is the user-specified argument for the callback.
 */
struct isl_schedule_node_preorder_data {
	isl_bool (*fn)(__isl_keep isl_schedule_node *node, void *user);
	void *user;
};

/* Callback for "traverse" to enter a node and to move
 * to the deepest initial subtree that should be traversed
 * for use in a preorder visit.
 *
 * If the user callback returns a negative value, then we abort
 * the traversal.  If this callback returns zero, then we skip
 * the subtree rooted at the current node.  Otherwise, we move
 * down to the first child and repeat the process until a leaf
 * is reached.
 */
static __isl_give isl_schedule_node *preorder_enter(
	__isl_take isl_schedule_node *node, void *user)
{
	struct isl_schedule_node_preorder_data *data = user;

	if (!node)
		return NULL;

	do {
		isl_bool r;

		r = data->fn(node, data->user);
		if (r < 0)
			return isl_schedule_node_free(node);
		if (r == isl_bool_false)
			return node;
	} while (isl_schedule_node_has_children(node) &&
		(node = isl_schedule_node_first_child(node)) != NULL);

	return node;
}

/* Callback for "traverse" to leave a node
 * for use in a preorder visit.
 * Since we already visited the node when we entered it,
 * we do not need to do anything here.
 */
static __isl_give isl_schedule_node *preorder_leave(
	__isl_take isl_schedule_node *node, void *user)
{
	return node;
}

/* Traverse the descendants of "node" (including the node itself)
 * in depth first preorder.
 *
 * If "fn" returns -1 on any of the nodes, then the traversal is aborted.
 * If "fn" returns 0 on any of the nodes, then the subtree rooted
 * at that node is skipped.
 *
 * Return 0 on success and -1 on failure.
 */
isl_stat isl_schedule_node_foreach_descendant_top_down(
	__isl_keep isl_schedule_node *node,
	isl_bool (*fn)(__isl_keep isl_schedule_node *node, void *user),
	void *user)
{
	struct isl_schedule_node_preorder_data data = { fn, user };

	node = isl_schedule_node_copy(node);
	node = traverse(node, &preorder_enter, &preorder_leave, &data);
	isl_schedule_node_free(node);

	return node ? isl_stat_ok : isl_stat_error;
}

/* Internal data structure for isl_schedule_node_map_descendant_bottom_up.
 *
 * "fn" is the user-specified callback function.
 * "user" is the user-specified argument for the callback.
 */
struct isl_schedule_node_postorder_data {
	__isl_give isl_schedule_node *(*fn)(__isl_take isl_schedule_node *node,
		void *user);
	void *user;
};

/* Callback for "traverse" to enter a node and to move
 * to the deepest initial subtree that should be traversed
 * for use in a postorder visit.
 *
 * Since we are performing a postorder visit, we only need
 * to move to the deepest initial leaf here.
 */
static __isl_give isl_schedule_node *postorder_enter(
	__isl_take isl_schedule_node *node, void *user)
{
	while (node && isl_schedule_node_has_children(node))
		node = isl_schedule_node_first_child(node);

	return node;
}

/* Callback for "traverse" to leave a node
 * for use in a postorder visit.
 *
 * Since we are performing a postorder visit, we need
 * to call the user callback here.
 */
static __isl_give isl_schedule_node *postorder_leave(
	__isl_take isl_schedule_node *node, void *user)
{
	struct isl_schedule_node_postorder_data *data = user;

	return data->fn(node, data->user);
}

/* Traverse the descendants of "node" (including the node itself)
 * in depth first postorder, allowing the user to modify the visited node.
 * The traversal continues from the node returned by the callback function.
 * It is the responsibility of the user to ensure that this does not
 * lead to an infinite loop.  It is safest to always return a pointer
 * to the same position (same ancestors and child positions) as the input node.
 */
__isl_give isl_schedule_node *isl_schedule_node_map_descendant_bottom_up(
	__isl_take isl_schedule_node *node,
	__isl_give isl_schedule_node *(*fn)(__isl_take isl_schedule_node *node,
		void *user), void *user)
{
	struct isl_schedule_node_postorder_data data = { fn, user };

	return traverse(node, &postorder_enter, &postorder_leave, &data);
}

/* Traverse the ancestors of "node" from the root down to and including
 * the parent of "node", calling "fn" on each of them.
 *
 * If "fn" returns -1 on any of the nodes, then the traversal is aborted.
 *
 * Return 0 on success and -1 on failure.
 */
isl_stat isl_schedule_node_foreach_ancestor_top_down(
	__isl_keep isl_schedule_node *node,
	isl_stat (*fn)(__isl_keep isl_schedule_node *node, void *user),
	void *user)
{
	int i, n;

	if (!node)
		return isl_stat_error;

	n = isl_schedule_node_get_tree_depth(node);
	for (i = 0; i < n; ++i) {
		isl_schedule_node *ancestor;
		isl_stat r;

		ancestor = isl_schedule_node_copy(node);
		ancestor = isl_schedule_node_ancestor(ancestor, n - i);
		r = fn(ancestor, user);
		isl_schedule_node_free(ancestor);
		if (r < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Is any node in the subtree rooted at "node" anchored?
 * That is, do any of these nodes reference the outer band nodes?
 */
isl_bool isl_schedule_node_is_subtree_anchored(
	__isl_keep isl_schedule_node *node)
{
	if (!node)
		return isl_bool_error;
	return isl_schedule_tree_is_subtree_anchored(node->tree);
}

/* Return the number of members in the given band node.
 */
unsigned isl_schedule_node_band_n_member(__isl_keep isl_schedule_node *node)
{
	return node ? isl_schedule_tree_band_n_member(node->tree) : 0;
}

/* Is the band member at position "pos" of the band node "node"
 * marked coincident?
 */
isl_bool isl_schedule_node_band_member_get_coincident(
	__isl_keep isl_schedule_node *node, int pos)
{
	if (!node)
		return isl_bool_error;
	return isl_schedule_tree_band_member_get_coincident(node->tree, pos);
}

/* Mark the band member at position "pos" the band node "node"
 * as being coincident or not according to "coincident".
 */
__isl_give isl_schedule_node *isl_schedule_node_band_member_set_coincident(
	__isl_take isl_schedule_node *node, int pos, int coincident)
{
	int c;
	isl_schedule_tree *tree;

	if (!node)
		return NULL;
	c = isl_schedule_node_band_member_get_coincident(node, pos);
	if (c == coincident)
		return node;

	tree = isl_schedule_tree_copy(node->tree);
	tree = isl_schedule_tree_band_member_set_coincident(tree, pos,
							    coincident);
	node = isl_schedule_node_graft_tree(node, tree);

	return node;
}

/* Is the band node "node" marked permutable?
 */
isl_bool isl_schedule_node_band_get_permutable(
	__isl_keep isl_schedule_node *node)
{
	if (!node)
		return isl_bool_error;

	return isl_schedule_tree_band_get_permutable(node->tree);
}

/* Mark the band node "node" permutable or not according to "permutable"?
 */
__isl_give isl_schedule_node *isl_schedule_node_band_set_permutable(
	__isl_take isl_schedule_node *node, int permutable)
{
	isl_schedule_tree *tree;

	if (!node)
		return NULL;
	if (isl_schedule_node_band_get_permutable(node) == permutable)
		return node;

	tree = isl_schedule_tree_copy(node->tree);
	tree = isl_schedule_tree_band_set_permutable(tree, permutable);
	node = isl_schedule_node_graft_tree(node, tree);

	return node;
}

/* Return the schedule space of the band node.
 */
__isl_give isl_space *isl_schedule_node_band_get_space(
	__isl_keep isl_schedule_node *node)
{
	if (!node)
		return NULL;

	return isl_schedule_tree_band_get_space(node->tree);
}

/* Return the schedule of the band node in isolation.
 */
__isl_give isl_multi_union_pw_aff *isl_schedule_node_band_get_partial_schedule(
	__isl_keep isl_schedule_node *node)
{
	if (!node)
		return NULL;

	return isl_schedule_tree_band_get_partial_schedule(node->tree);
}

/* Return the schedule of the band node in isolation in the form of
 * an isl_union_map.
 *
 * If the band does not have any members, then we construct a universe map
 * with the universe of the domain elements reaching the node as domain.
 * Otherwise, we extract an isl_multi_union_pw_aff representation and
 * convert that to an isl_union_map.
 */
__isl_give isl_union_map *isl_schedule_node_band_get_partial_schedule_union_map(
	__isl_keep isl_schedule_node *node)
{
	isl_multi_union_pw_aff *mupa;

	if (!node)
		return NULL;

	if (isl_schedule_node_get_type(node) != isl_schedule_node_band)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"not a band node", return NULL);
	if (isl_schedule_node_band_n_member(node) == 0) {
		isl_union_set *domain;

		domain = isl_schedule_node_get_universe_domain(node);
		return isl_union_map_from_domain(domain);
	}

	mupa = isl_schedule_node_band_get_partial_schedule(node);
	return isl_union_map_from_multi_union_pw_aff(mupa);
}

/* Return the loop AST generation type for the band member of band node "node"
 * at position "pos".
 */
enum isl_ast_loop_type isl_schedule_node_band_member_get_ast_loop_type(
	__isl_keep isl_schedule_node *node, int pos)
{
	if (!node)
		return isl_ast_loop_error;

	return isl_schedule_tree_band_member_get_ast_loop_type(node->tree, pos);
}

/* Set the loop AST generation type for the band member of band node "node"
 * at position "pos" to "type".
 */
__isl_give isl_schedule_node *isl_schedule_node_band_member_set_ast_loop_type(
	__isl_take isl_schedule_node *node, int pos,
	enum isl_ast_loop_type type)
{
	isl_schedule_tree *tree;

	if (!node)
		return NULL;

	tree = isl_schedule_tree_copy(node->tree);
	tree = isl_schedule_tree_band_member_set_ast_loop_type(tree, pos, type);
	return isl_schedule_node_graft_tree(node, tree);
}

/* Return the loop AST generation type for the band member of band node "node"
 * at position "pos" for the isolated part.
 */
enum isl_ast_loop_type isl_schedule_node_band_member_get_isolate_ast_loop_type(
	__isl_keep isl_schedule_node *node, int pos)
{
	if (!node)
		return isl_ast_loop_error;

	return isl_schedule_tree_band_member_get_isolate_ast_loop_type(
							    node->tree, pos);
}

/* Set the loop AST generation type for the band member of band node "node"
 * at position "pos" for the isolated part to "type".
 */
__isl_give isl_schedule_node *
isl_schedule_node_band_member_set_isolate_ast_loop_type(
	__isl_take isl_schedule_node *node, int pos,
	enum isl_ast_loop_type type)
{
	isl_schedule_tree *tree;

	if (!node)
		return NULL;

	tree = isl_schedule_tree_copy(node->tree);
	tree = isl_schedule_tree_band_member_set_isolate_ast_loop_type(tree,
								    pos, type);
	return isl_schedule_node_graft_tree(node, tree);
}

/* Return the AST build options associated to band node "node".
 */
__isl_give isl_union_set *isl_schedule_node_band_get_ast_build_options(
	__isl_keep isl_schedule_node *node)
{
	if (!node)
		return NULL;

	return isl_schedule_tree_band_get_ast_build_options(node->tree);
}

/* Replace the AST build options associated to band node "node" by "options".
 */
__isl_give isl_schedule_node *isl_schedule_node_band_set_ast_build_options(
	__isl_take isl_schedule_node *node, __isl_take isl_union_set *options)
{
	isl_schedule_tree *tree;

	if (!node || !options)
		goto error;

	tree = isl_schedule_tree_copy(node->tree);
	tree = isl_schedule_tree_band_set_ast_build_options(tree, options);
	return isl_schedule_node_graft_tree(node, tree);
error:
	isl_schedule_node_free(node);
	isl_union_set_free(options);
	return NULL;
}

/* Return the "isolate" option associated to band node "node".
 */
__isl_give isl_set *isl_schedule_node_band_get_ast_isolate_option(
	__isl_keep isl_schedule_node *node)
{
	int depth;

	if (!node)
		return NULL;

	depth = isl_schedule_node_get_schedule_depth(node);
	return isl_schedule_tree_band_get_ast_isolate_option(node->tree, depth);
}

/* Make sure that that spaces of "node" and "mv" are the same.
 * Return -1 on error, reporting the error to the user.
 */
static int check_space_multi_val(__isl_keep isl_schedule_node *node,
	__isl_keep isl_multi_val *mv)
{
	isl_space *node_space, *mv_space;
	int equal;

	node_space = isl_schedule_node_band_get_space(node);
	mv_space = isl_multi_val_get_space(mv);
	equal = isl_space_tuple_is_equal(node_space, isl_dim_set,
					mv_space, isl_dim_set);
	isl_space_free(mv_space);
	isl_space_free(node_space);
	if (equal < 0)
		return -1;
	if (!equal)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"spaces don't match", return -1);

	return 0;
}

/* Multiply the partial schedule of the band node "node"
 * with the factors in "mv".
 */
__isl_give isl_schedule_node *isl_schedule_node_band_scale(
	__isl_take isl_schedule_node *node, __isl_take isl_multi_val *mv)
{
	isl_schedule_tree *tree;
	int anchored;

	if (!node || !mv)
		goto error;
	if (check_space_multi_val(node, mv) < 0)
		goto error;
	anchored = isl_schedule_node_is_subtree_anchored(node);
	if (anchored < 0)
		goto error;
	if (anchored)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"cannot scale band node with anchored subtree",
			goto error);

	tree = isl_schedule_node_get_tree(node);
	tree = isl_schedule_tree_band_scale(tree, mv);
	return isl_schedule_node_graft_tree(node, tree);
error:
	isl_multi_val_free(mv);
	isl_schedule_node_free(node);
	return NULL;
}

/* Divide the partial schedule of the band node "node"
 * by the factors in "mv".
 */
__isl_give isl_schedule_node *isl_schedule_node_band_scale_down(
	__isl_take isl_schedule_node *node, __isl_take isl_multi_val *mv)
{
	isl_schedule_tree *tree;
	int anchored;

	if (!node || !mv)
		goto error;
	if (check_space_multi_val(node, mv) < 0)
		goto error;
	anchored = isl_schedule_node_is_subtree_anchored(node);
	if (anchored < 0)
		goto error;
	if (anchored)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"cannot scale down band node with anchored subtree",
			goto error);

	tree = isl_schedule_node_get_tree(node);
	tree = isl_schedule_tree_band_scale_down(tree, mv);
	return isl_schedule_node_graft_tree(node, tree);
error:
	isl_multi_val_free(mv);
	isl_schedule_node_free(node);
	return NULL;
}

/* Reduce the partial schedule of the band node "node"
 * modulo the factors in "mv".
 */
__isl_give isl_schedule_node *isl_schedule_node_band_mod(
	__isl_take isl_schedule_node *node, __isl_take isl_multi_val *mv)
{
	isl_schedule_tree *tree;
	isl_bool anchored;

	if (!node || !mv)
		goto error;
	if (check_space_multi_val(node, mv) < 0)
		goto error;
	anchored = isl_schedule_node_is_subtree_anchored(node);
	if (anchored < 0)
		goto error;
	if (anchored)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"cannot perform mod on band node with anchored subtree",
			goto error);

	tree = isl_schedule_node_get_tree(node);
	tree = isl_schedule_tree_band_mod(tree, mv);
	return isl_schedule_node_graft_tree(node, tree);
error:
	isl_multi_val_free(mv);
	isl_schedule_node_free(node);
	return NULL;
}

/* Make sure that that spaces of "node" and "mupa" are the same.
 * Return isl_stat_error on error, reporting the error to the user.
 */
static isl_stat check_space_multi_union_pw_aff(
	__isl_keep isl_schedule_node *node,
	__isl_keep isl_multi_union_pw_aff *mupa)
{
	isl_space *node_space, *mupa_space;
	isl_bool equal;

	node_space = isl_schedule_node_band_get_space(node);
	mupa_space = isl_multi_union_pw_aff_get_space(mupa);
	equal = isl_space_tuple_is_equal(node_space, isl_dim_set,
					mupa_space, isl_dim_set);
	isl_space_free(mupa_space);
	isl_space_free(node_space);
	if (equal < 0)
		return isl_stat_error;
	if (!equal)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"spaces don't match", return isl_stat_error);

	return isl_stat_ok;
}

/* Shift the partial schedule of the band node "node" by "shift".
 */
__isl_give isl_schedule_node *isl_schedule_node_band_shift(
	__isl_take isl_schedule_node *node,
	__isl_take isl_multi_union_pw_aff *shift)
{
	isl_schedule_tree *tree;
	int anchored;

	if (!node || !shift)
		goto error;
	if (check_space_multi_union_pw_aff(node, shift) < 0)
		goto error;
	anchored = isl_schedule_node_is_subtree_anchored(node);
	if (anchored < 0)
		goto error;
	if (anchored)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"cannot shift band node with anchored subtree",
			goto error);

	tree = isl_schedule_node_get_tree(node);
	tree = isl_schedule_tree_band_shift(tree, shift);
	return isl_schedule_node_graft_tree(node, tree);
error:
	isl_multi_union_pw_aff_free(shift);
	isl_schedule_node_free(node);
	return NULL;
}

/* Tile "node" with tile sizes "sizes".
 *
 * The current node is replaced by two nested nodes corresponding
 * to the tile dimensions and the point dimensions.
 *
 * Return a pointer to the outer (tile) node.
 *
 * If any of the descendants of "node" depend on the set of outer band nodes,
 * then we refuse to tile the node.
 *
 * If the scale tile loops option is set, then the tile loops
 * are scaled by the tile sizes.  If the shift point loops option is set,
 * then the point loops are shifted to start at zero.
 * In particular, these options affect the tile and point loop schedules
 * as follows
 *
 *	scale	shift	original	tile		point
 *
 *	0	0	i		floor(i/s)	i
 *	1	0	i		s * floor(i/s)	i
 *	0	1	i		floor(i/s)	i - s * floor(i/s)
 *	1	1	i		s * floor(i/s)	i - s * floor(i/s)
 */
__isl_give isl_schedule_node *isl_schedule_node_band_tile(
	__isl_take isl_schedule_node *node, __isl_take isl_multi_val *sizes)
{
	isl_schedule_tree *tree;
	int anchored;

	if (!node || !sizes)
		goto error;
	anchored = isl_schedule_node_is_subtree_anchored(node);
	if (anchored < 0)
		goto error;
	if (anchored)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"cannot tile band node with anchored subtree",
			goto error);

	if (check_space_multi_val(node, sizes) < 0)
		goto error;

	tree = isl_schedule_node_get_tree(node);
	tree = isl_schedule_tree_band_tile(tree, sizes);
	return isl_schedule_node_graft_tree(node, tree);
error:
	isl_multi_val_free(sizes);
	isl_schedule_node_free(node);
	return NULL;
}

/* Move the band node "node" down to all the leaves in the subtree
 * rooted at "node".
 * Return a pointer to the node in the resulting tree that is in the same
 * position as the node pointed to by "node" in the original tree.
 *
 * If the node only has a leaf child, then nothing needs to be done.
 * Otherwise, the child of the node is removed and the result is
 * appended to all the leaves in the subtree rooted at the original child.
 * Since the node is moved to the leaves, it needs to be expanded
 * according to the expansion, if any, defined by that subtree.
 * In the end, the original node is replaced by the result of
 * attaching copies of the expanded node to the leaves.
 *
 * If any of the nodes in the subtree rooted at "node" depend on
 * the set of outer band nodes then we refuse to sink the band node.
 */
__isl_give isl_schedule_node *isl_schedule_node_band_sink(
	__isl_take isl_schedule_node *node)
{
	enum isl_schedule_node_type type;
	isl_schedule_tree *tree, *child;
	isl_union_pw_multi_aff *contraction;
	int anchored;

	if (!node)
		return NULL;

	type = isl_schedule_node_get_type(node);
	if (type != isl_schedule_node_band)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"not a band node", isl_schedule_node_free(node));
	anchored = isl_schedule_node_is_subtree_anchored(node);
	if (anchored < 0)
		return isl_schedule_node_free(node);
	if (anchored)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"cannot sink band node in anchored subtree",
			isl_schedule_node_free(node));
	if (isl_schedule_tree_n_children(node->tree) == 0)
		return node;

	contraction = isl_schedule_node_get_subtree_contraction(node);

	tree = isl_schedule_node_get_tree(node);
	child = isl_schedule_tree_get_child(tree, 0);
	tree = isl_schedule_tree_reset_children(tree);
	tree = isl_schedule_tree_pullback_union_pw_multi_aff(tree, contraction);
	tree = isl_schedule_tree_append_to_leaves(child, tree);

	return isl_schedule_node_graft_tree(node, tree);
}

/* Split "node" into two nested band nodes, one with the first "pos"
 * dimensions and one with the remaining dimensions.
 * The schedules of the two band nodes live in anonymous spaces.
 * The loop AST generation type options and the isolate option
 * are split over the the two band nodes.
 */
__isl_give isl_schedule_node *isl_schedule_node_band_split(
	__isl_take isl_schedule_node *node, int pos)
{
	int depth;
	isl_schedule_tree *tree;

	depth = isl_schedule_node_get_schedule_depth(node);
	tree = isl_schedule_node_get_tree(node);
	tree = isl_schedule_tree_band_split(tree, pos, depth);
	return isl_schedule_node_graft_tree(node, tree);
}

/* Return the context of the context node "node".
 */
__isl_give isl_set *isl_schedule_node_context_get_context(
	__isl_keep isl_schedule_node *node)
{
	if (!node)
		return NULL;

	return isl_schedule_tree_context_get_context(node->tree);
}

/* Return the domain of the domain node "node".
 */
__isl_give isl_union_set *isl_schedule_node_domain_get_domain(
	__isl_keep isl_schedule_node *node)
{
	if (!node)
		return NULL;

	return isl_schedule_tree_domain_get_domain(node->tree);
}

/* Return the expansion map of expansion node "node".
 */
__isl_give isl_union_map *isl_schedule_node_expansion_get_expansion(
	__isl_keep isl_schedule_node *node)
{
	if (!node)
		return NULL;

	return isl_schedule_tree_expansion_get_expansion(node->tree);
}

/* Return the contraction of expansion node "node".
 */
__isl_give isl_union_pw_multi_aff *isl_schedule_node_expansion_get_contraction(
	__isl_keep isl_schedule_node *node)
{
	if (!node)
		return NULL;

	return isl_schedule_tree_expansion_get_contraction(node->tree);
}

/* Replace the contraction and the expansion of the expansion node "node"
 * by "contraction" and "expansion".
 */
__isl_give isl_schedule_node *
isl_schedule_node_expansion_set_contraction_and_expansion(
	__isl_take isl_schedule_node *node,
	__isl_take isl_union_pw_multi_aff *contraction,
	__isl_take isl_union_map *expansion)
{
	isl_schedule_tree *tree;

	if (!node || !contraction || !expansion)
		goto error;

	tree = isl_schedule_tree_copy(node->tree);
	tree = isl_schedule_tree_expansion_set_contraction_and_expansion(tree,
							contraction, expansion);
	return isl_schedule_node_graft_tree(node, tree);
error:
	isl_schedule_node_free(node);
	isl_union_pw_multi_aff_free(contraction);
	isl_union_map_free(expansion);
	return NULL;
}

/* Return the extension of the extension node "node".
 */
__isl_give isl_union_map *isl_schedule_node_extension_get_extension(
	__isl_keep isl_schedule_node *node)
{
	if (!node)
		return NULL;

	return isl_schedule_tree_extension_get_extension(node->tree);
}

/* Replace the extension of extension node "node" by "extension".
 */
__isl_give isl_schedule_node *isl_schedule_node_extension_set_extension(
	__isl_take isl_schedule_node *node, __isl_take isl_union_map *extension)
{
	isl_schedule_tree *tree;

	if (!node || !extension)
		goto error;

	tree = isl_schedule_tree_copy(node->tree);
	tree = isl_schedule_tree_extension_set_extension(tree, extension);
	return isl_schedule_node_graft_tree(node, tree);
error:
	isl_schedule_node_free(node);
	isl_union_map_free(extension);
	return NULL;
}

/* Return the filter of the filter node "node".
 */
__isl_give isl_union_set *isl_schedule_node_filter_get_filter(
	__isl_keep isl_schedule_node *node)
{
	if (!node)
		return NULL;

	return isl_schedule_tree_filter_get_filter(node->tree);
}

/* Replace the filter of filter node "node" by "filter".
 */
__isl_give isl_schedule_node *isl_schedule_node_filter_set_filter(
	__isl_take isl_schedule_node *node, __isl_take isl_union_set *filter)
{
	isl_schedule_tree *tree;

	if (!node || !filter)
		goto error;

	tree = isl_schedule_tree_copy(node->tree);
	tree = isl_schedule_tree_filter_set_filter(tree, filter);
	return isl_schedule_node_graft_tree(node, tree);
error:
	isl_schedule_node_free(node);
	isl_union_set_free(filter);
	return NULL;
}

/* Intersect the filter of filter node "node" with "filter".
 *
 * If the filter of the node is already a subset of "filter",
 * then leave the node unchanged.
 */
__isl_give isl_schedule_node *isl_schedule_node_filter_intersect_filter(
	__isl_take isl_schedule_node *node, __isl_take isl_union_set *filter)
{
	isl_union_set *node_filter = NULL;
	isl_bool subset;

	if (!node || !filter)
		goto error;

	node_filter = isl_schedule_node_filter_get_filter(node);
	subset = isl_union_set_is_subset(node_filter, filter);
	if (subset < 0)
		goto error;
	if (subset) {
		isl_union_set_free(node_filter);
		isl_union_set_free(filter);
		return node;
	}
	node_filter = isl_union_set_intersect(node_filter, filter);
	node = isl_schedule_node_filter_set_filter(node, node_filter);
	return node;
error:
	isl_schedule_node_free(node);
	isl_union_set_free(node_filter);
	isl_union_set_free(filter);
	return NULL;
}

/* Return the guard of the guard node "node".
 */
__isl_give isl_set *isl_schedule_node_guard_get_guard(
	__isl_keep isl_schedule_node *node)
{
	if (!node)
		return NULL;

	return isl_schedule_tree_guard_get_guard(node->tree);
}

/* Return the mark identifier of the mark node "node".
 */
__isl_give isl_id *isl_schedule_node_mark_get_id(
	__isl_keep isl_schedule_node *node)
{
	if (!node)
		return NULL;

	return isl_schedule_tree_mark_get_id(node->tree);
}

/* Replace the child at position "pos" of the sequence node "node"
 * by the children of sequence root node of "tree".
 */
__isl_give isl_schedule_node *isl_schedule_node_sequence_splice(
	__isl_take isl_schedule_node *node, int pos,
	__isl_take isl_schedule_tree *tree)
{
	isl_schedule_tree *node_tree;

	if (!node || !tree)
		goto error;
	if (isl_schedule_node_get_type(node) != isl_schedule_node_sequence)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"not a sequence node", goto error);
	if (isl_schedule_tree_get_type(tree) != isl_schedule_node_sequence)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"not a sequence node", goto error);
	node_tree = isl_schedule_node_get_tree(node);
	node_tree = isl_schedule_tree_sequence_splice(node_tree, pos, tree);
	node = isl_schedule_node_graft_tree(node, node_tree);

	return node;
error:
	isl_schedule_node_free(node);
	isl_schedule_tree_free(tree);
	return NULL;
}

/* Given a sequence node "node", with a child at position "pos" that
 * is also a sequence node, attach the children of that node directly
 * as children of "node" at that position, replacing the original child.
 *
 * The filters of these children are intersected with the filter
 * of the child at position "pos".
 */
__isl_give isl_schedule_node *isl_schedule_node_sequence_splice_child(
	__isl_take isl_schedule_node *node, int pos)
{
	int i, n;
	isl_union_set *filter;
	isl_schedule_node *child;
	isl_schedule_tree *tree;

	if (!node)
		return NULL;
	if (isl_schedule_node_get_type(node) != isl_schedule_node_sequence)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"not a sequence node", isl_schedule_node_free(node));
	node = isl_schedule_node_child(node, pos);
	node = isl_schedule_node_child(node, 0);
	if (isl_schedule_node_get_type(node) != isl_schedule_node_sequence)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"not a sequence node", isl_schedule_node_free(node));
	child = isl_schedule_node_copy(node);
	node = isl_schedule_node_parent(node);
	filter = isl_schedule_node_filter_get_filter(node);
	n = isl_schedule_node_n_children(child);
	for (i = 0; i < n; ++i) {
		child = isl_schedule_node_child(child, i);
		child = isl_schedule_node_filter_intersect_filter(child,
						isl_union_set_copy(filter));
		child = isl_schedule_node_parent(child);
	}
	isl_union_set_free(filter);
	tree = isl_schedule_node_get_tree(child);
	isl_schedule_node_free(child);
	node = isl_schedule_node_parent(node);
	node = isl_schedule_node_sequence_splice(node, pos, tree);

	return node;
}

/* Update the ancestors of "node" to point to the tree that "node"
 * now points to.
 * That is, replace the child in the original parent that corresponds
 * to the current tree position by node->tree and continue updating
 * the ancestors in the same way until the root is reached.
 *
 * If "fn" is not NULL, then it is called on each ancestor as we move up
 * the tree so that it can modify the ancestor before it is added
 * to the list of ancestors of the modified node.
 * The additional "pos" argument records the position
 * of the "tree" argument in the original schedule tree.
 *
 * If "node" originally points to a leaf of the schedule tree, then make sure
 * that in the end it points to a leaf in the updated schedule tree.
 */
static __isl_give isl_schedule_node *update_ancestors(
	__isl_take isl_schedule_node *node,
	__isl_give isl_schedule_tree *(*fn)(__isl_take isl_schedule_tree *tree,
		__isl_keep isl_schedule_node *pos, void *user), void *user)
{
	int i, n;
	int is_leaf;
	isl_ctx *ctx;
	isl_schedule_tree *tree;
	isl_schedule_node *pos = NULL;

	if (fn)
		pos = isl_schedule_node_copy(node);

	node = isl_schedule_node_cow(node);
	if (!node)
		return isl_schedule_node_free(pos);

	ctx = isl_schedule_node_get_ctx(node);
	n = isl_schedule_tree_list_n_schedule_tree(node->ancestors);
	tree = isl_schedule_tree_copy(node->tree);

	for (i = n - 1; i >= 0; --i) {
		isl_schedule_tree *parent;

		parent = isl_schedule_tree_list_get_schedule_tree(
						    node->ancestors, i);
		parent = isl_schedule_tree_replace_child(parent,
						    node->child_pos[i], tree);
		if (fn) {
			pos = isl_schedule_node_parent(pos);
			parent = fn(parent, pos, user);
		}
		node->ancestors = isl_schedule_tree_list_set_schedule_tree(
			    node->ancestors, i, isl_schedule_tree_copy(parent));

		tree = parent;
	}

	if (fn)
		isl_schedule_node_free(pos);

	is_leaf = isl_schedule_tree_is_leaf(node->tree);
	node->schedule = isl_schedule_set_root(node->schedule, tree);
	if (is_leaf) {
		isl_schedule_tree_free(node->tree);
		node->tree = isl_schedule_node_get_leaf(node);
	}

	if (!node->schedule || !node->ancestors)
		return isl_schedule_node_free(node);

	return node;
}

/* Replace the subtree that "pos" points to by "tree", updating
 * the ancestors to maintain a consistent state.
 */
__isl_give isl_schedule_node *isl_schedule_node_graft_tree(
	__isl_take isl_schedule_node *pos, __isl_take isl_schedule_tree *tree)
{
	if (!tree || !pos)
		goto error;
	if (pos->tree == tree) {
		isl_schedule_tree_free(tree);
		return pos;
	}

	pos = isl_schedule_node_cow(pos);
	if (!pos)
		goto error;

	isl_schedule_tree_free(pos->tree);
	pos->tree = tree;

	return update_ancestors(pos, NULL, NULL);
error:
	isl_schedule_node_free(pos);
	isl_schedule_tree_free(tree);
	return NULL;
}

/* Make sure we can insert a node between "node" and its parent.
 * Return -1 on error, reporting the reason why we cannot insert a node.
 */
static int check_insert(__isl_keep isl_schedule_node *node)
{
	int has_parent;
	enum isl_schedule_node_type type;

	has_parent = isl_schedule_node_has_parent(node);
	if (has_parent < 0)
		return -1;
	if (!has_parent)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"cannot insert node outside of root", return -1);

	type = isl_schedule_node_get_parent_type(node);
	if (type == isl_schedule_node_error)
		return -1;
	if (type == isl_schedule_node_set || type == isl_schedule_node_sequence)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"cannot insert node between set or sequence node "
			"and its filter children", return -1);

	return 0;
}

/* Insert a band node with partial schedule "mupa" between "node" and
 * its parent.
 * Return a pointer to the new band node.
 *
 * If any of the nodes in the subtree rooted at "node" depend on
 * the set of outer band nodes then we refuse to insert the band node.
 */
__isl_give isl_schedule_node *isl_schedule_node_insert_partial_schedule(
	__isl_take isl_schedule_node *node,
	__isl_take isl_multi_union_pw_aff *mupa)
{
	int anchored;
	isl_schedule_band *band;
	isl_schedule_tree *tree;

	if (check_insert(node) < 0)
		node = isl_schedule_node_free(node);
	anchored = isl_schedule_node_is_subtree_anchored(node);
	if (anchored < 0)
		goto error;
	if (anchored)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"cannot insert band node in anchored subtree",
			goto error);

	tree = isl_schedule_node_get_tree(node);
	band = isl_schedule_band_from_multi_union_pw_aff(mupa);
	tree = isl_schedule_tree_insert_band(tree, band);
	node = isl_schedule_node_graft_tree(node, tree);

	return node;
error:
	isl_schedule_node_free(node);
	isl_multi_union_pw_aff_free(mupa);
	return NULL;
}

/* Insert a context node with context "context" between "node" and its parent.
 * Return a pointer to the new context node.
 */
__isl_give isl_schedule_node *isl_schedule_node_insert_context(
	__isl_take isl_schedule_node *node, __isl_take isl_set *context)
{
	isl_schedule_tree *tree;

	if (check_insert(node) < 0)
		node = isl_schedule_node_free(node);

	tree = isl_schedule_node_get_tree(node);
	tree = isl_schedule_tree_insert_context(tree, context);
	node = isl_schedule_node_graft_tree(node, tree);

	return node;
}

/* Insert an expansion node with the given "contraction" and "expansion"
 * between "node" and its parent.
 * Return a pointer to the new expansion node.
 *
 * Typically the domain and range spaces of the expansion are different.
 * This means that only one of them can refer to the current domain space
 * in a consistent tree.  It is up to the caller to ensure that the tree
 * returns to a consistent state.
 */
__isl_give isl_schedule_node *isl_schedule_node_insert_expansion(
	__isl_take isl_schedule_node *node,
	__isl_take isl_union_pw_multi_aff *contraction,
	__isl_take isl_union_map *expansion)
{
	isl_schedule_tree *tree;

	if (check_insert(node) < 0)
		node = isl_schedule_node_free(node);

	tree = isl_schedule_node_get_tree(node);
	tree = isl_schedule_tree_insert_expansion(tree, contraction, expansion);
	node = isl_schedule_node_graft_tree(node, tree);

	return node;
}

/* Insert an extension node with extension "extension" between "node" and
 * its parent.
 * Return a pointer to the new extension node.
 */
__isl_give isl_schedule_node *isl_schedule_node_insert_extension(
	__isl_take isl_schedule_node *node,
	__isl_take isl_union_map *extension)
{
	isl_schedule_tree *tree;

	tree = isl_schedule_node_get_tree(node);
	tree = isl_schedule_tree_insert_extension(tree, extension);
	node = isl_schedule_node_graft_tree(node, tree);

	return node;
}

/* Insert a filter node with filter "filter" between "node" and its parent.
 * Return a pointer to the new filter node.
 */
__isl_give isl_schedule_node *isl_schedule_node_insert_filter(
	__isl_take isl_schedule_node *node, __isl_take isl_union_set *filter)
{
	isl_schedule_tree *tree;

	if (check_insert(node) < 0)
		node = isl_schedule_node_free(node);

	tree = isl_schedule_node_get_tree(node);
	tree = isl_schedule_tree_insert_filter(tree, filter);
	node = isl_schedule_node_graft_tree(node, tree);

	return node;
}

/* Insert a guard node with guard "guard" between "node" and its parent.
 * Return a pointer to the new guard node.
 */
__isl_give isl_schedule_node *isl_schedule_node_insert_guard(
	__isl_take isl_schedule_node *node, __isl_take isl_set *guard)
{
	isl_schedule_tree *tree;

	if (check_insert(node) < 0)
		node = isl_schedule_node_free(node);

	tree = isl_schedule_node_get_tree(node);
	tree = isl_schedule_tree_insert_guard(tree, guard);
	node = isl_schedule_node_graft_tree(node, tree);

	return node;
}

/* Insert a mark node with mark identifier "mark" between "node" and
 * its parent.
 * Return a pointer to the new mark node.
 */
__isl_give isl_schedule_node *isl_schedule_node_insert_mark(
	__isl_take isl_schedule_node *node, __isl_take isl_id *mark)
{
	isl_schedule_tree *tree;

	if (check_insert(node) < 0)
		node = isl_schedule_node_free(node);

	tree = isl_schedule_node_get_tree(node);
	tree = isl_schedule_tree_insert_mark(tree, mark);
	node = isl_schedule_node_graft_tree(node, tree);

	return node;
}

/* Attach the current subtree of "node" to a sequence of filter tree nodes
 * with filters described by "filters", attach this sequence
 * of filter tree nodes as children to a new tree of type "type" and
 * replace the original subtree of "node" by this new tree.
 * Each copy of the original subtree is simplified with respect
 * to the corresponding filter.
 */
static __isl_give isl_schedule_node *isl_schedule_node_insert_children(
	__isl_take isl_schedule_node *node,
	enum isl_schedule_node_type type,
	__isl_take isl_union_set_list *filters)
{
	int i, n;
	isl_ctx *ctx;
	isl_schedule_tree *tree;
	isl_schedule_tree_list *list;

	if (check_insert(node) < 0)
		node = isl_schedule_node_free(node);

	if (!node || !filters)
		goto error;

	ctx = isl_schedule_node_get_ctx(node);
	n = isl_union_set_list_n_union_set(filters);
	list = isl_schedule_tree_list_alloc(ctx, n);
	for (i = 0; i < n; ++i) {
		isl_schedule_node *node_i;
		isl_schedule_tree *tree;
		isl_union_set *filter;

		filter = isl_union_set_list_get_union_set(filters, i);
		node_i = isl_schedule_node_copy(node);
		node_i = isl_schedule_node_gist(node_i,
						isl_union_set_copy(filter));
		tree = isl_schedule_node_get_tree(node_i);
		isl_schedule_node_free(node_i);
		tree = isl_schedule_tree_insert_filter(tree, filter);
		list = isl_schedule_tree_list_add(list, tree);
	}
	tree = isl_schedule_tree_from_children(type, list);
	node = isl_schedule_node_graft_tree(node, tree);

	isl_union_set_list_free(filters);
	return node;
error:
	isl_union_set_list_free(filters);
	isl_schedule_node_free(node);
	return NULL;
}

/* Insert a sequence node with child filters "filters" between "node" and
 * its parent.  That is, the tree that "node" points to is attached
 * to each of the child nodes of the filter nodes.
 * Return a pointer to the new sequence node.
 */
__isl_give isl_schedule_node *isl_schedule_node_insert_sequence(
	__isl_take isl_schedule_node *node,
	__isl_take isl_union_set_list *filters)
{
	return isl_schedule_node_insert_children(node,
					isl_schedule_node_sequence, filters);
}

/* Insert a set node with child filters "filters" between "node" and
 * its parent.  That is, the tree that "node" points to is attached
 * to each of the child nodes of the filter nodes.
 * Return a pointer to the new set node.
 */
__isl_give isl_schedule_node *isl_schedule_node_insert_set(
	__isl_take isl_schedule_node *node,
	__isl_take isl_union_set_list *filters)
{
	return isl_schedule_node_insert_children(node,
					isl_schedule_node_set, filters);
}

/* Remove "node" from its schedule tree and return a pointer
 * to the leaf at the same position in the updated schedule tree.
 *
 * It is not allowed to remove the root of a schedule tree or
 * a child of a set or sequence node.
 */
__isl_give isl_schedule_node *isl_schedule_node_cut(
	__isl_take isl_schedule_node *node)
{
	isl_schedule_tree *leaf;
	enum isl_schedule_node_type parent_type;

	if (!node)
		return NULL;
	if (!isl_schedule_node_has_parent(node))
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"cannot cut root", return isl_schedule_node_free(node));

	parent_type = isl_schedule_node_get_parent_type(node);
	if (parent_type == isl_schedule_node_set ||
	    parent_type == isl_schedule_node_sequence)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"cannot cut child of set or sequence",
			return isl_schedule_node_free(node));

	leaf = isl_schedule_node_get_leaf(node);
	return isl_schedule_node_graft_tree(node, leaf);
}

/* Remove a single node from the schedule tree, attaching the child
 * of "node" directly to its parent.
 * Return a pointer to this former child or to the leaf the position
 * of the original node if there was no child.
 * It is not allowed to remove the root of a schedule tree,
 * a set or sequence node, a child of a set or sequence node or
 * a band node with an anchored subtree.
 */
__isl_give isl_schedule_node *isl_schedule_node_delete(
	__isl_take isl_schedule_node *node)
{
	int n;
	isl_schedule_tree *tree;
	enum isl_schedule_node_type type;

	if (!node)
		return NULL;

	if (isl_schedule_node_get_tree_depth(node) == 0)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"cannot delete root node",
			return isl_schedule_node_free(node));
	n = isl_schedule_node_n_children(node);
	if (n != 1)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"can only delete node with a single child",
			return isl_schedule_node_free(node));
	type = isl_schedule_node_get_parent_type(node);
	if (type == isl_schedule_node_sequence || type == isl_schedule_node_set)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"cannot delete child of set or sequence",
			return isl_schedule_node_free(node));
	if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
		int anchored;

		anchored = isl_schedule_node_is_subtree_anchored(node);
		if (anchored < 0)
			return isl_schedule_node_free(node);
		if (anchored)
			isl_die(isl_schedule_node_get_ctx(node),
				isl_error_invalid,
				"cannot delete band node with anchored subtree",
				return isl_schedule_node_free(node));
	}

	tree = isl_schedule_node_get_tree(node);
	if (!tree || isl_schedule_tree_has_children(tree)) {
		tree = isl_schedule_tree_child(tree, 0);
	} else {
		isl_schedule_tree_free(tree);
		tree = isl_schedule_node_get_leaf(node);
	}
	node = isl_schedule_node_graft_tree(node, tree);

	return node;
}

/* Internal data structure for the group_ancestor callback.
 *
 * If "finished" is set, then we no longer need to modify
 * any further ancestors.
 *
 * "contraction" and "expansion" represent the expansion
 * that reflects the grouping.
 *
 * "domain" contains the domain elements that reach the position
 * where the grouping is performed.  That is, it is the range
 * of the resulting expansion.
 * "domain_universe" is the universe of "domain".
 * "group" is the set of group elements, i.e., the domain
 * of the resulting expansion.
 * "group_universe" is the universe of "group".
 *
 * "sched" is the schedule for the group elements, in pratice
 * an identity mapping on "group_universe".
 * "dim" is the dimension of "sched".
 */
struct isl_schedule_group_data {
	int finished;

	isl_union_map *expansion;
	isl_union_pw_multi_aff *contraction;

	isl_union_set *domain;
	isl_union_set *domain_universe;
	isl_union_set *group;
	isl_union_set *group_universe;

	int dim;
	isl_multi_aff *sched;
};

/* Is domain covered by data->domain within data->domain_universe?
 */
static int locally_covered_by_domain(__isl_keep isl_union_set *domain,
	struct isl_schedule_group_data *data)
{
	int is_subset;
	isl_union_set *test;

	test = isl_union_set_copy(domain);
	test = isl_union_set_intersect(test,
			    isl_union_set_copy(data->domain_universe));
	is_subset = isl_union_set_is_subset(test, data->domain);
	isl_union_set_free(test);

	return is_subset;
}

/* Update the band tree root "tree" to refer to the group instances
 * in data->group rather than the original domain elements in data->domain.
 * "pos" is the position in the original schedule tree where the modified
 * "tree" will be attached.
 *
 * Add the part of the identity schedule on the group instances data->sched
 * that corresponds to this band node to the band schedule.
 * If the domain elements that reach the node and that are part
 * of data->domain_universe are all elements of data->domain (and therefore
 * replaced by the group instances) then this data->domain_universe
 * is removed from the domain of the band schedule.
 */
static __isl_give isl_schedule_tree *group_band(
	__isl_take isl_schedule_tree *tree, __isl_keep isl_schedule_node *pos,
	struct isl_schedule_group_data *data)
{
	isl_union_set *domain;
	isl_multi_aff *ma;
	isl_multi_union_pw_aff *mupa, *partial;
	int is_covered;
	int depth, n, has_id;

	domain = isl_schedule_node_get_domain(pos);
	is_covered = locally_covered_by_domain(domain, data);
	if (is_covered >= 0 && is_covered) {
		domain = isl_union_set_universe(domain);
		domain = isl_union_set_subtract(domain,
			    isl_union_set_copy(data->domain_universe));
		tree = isl_schedule_tree_band_intersect_domain(tree, domain);
	} else
		isl_union_set_free(domain);
	if (is_covered < 0)
		return isl_schedule_tree_free(tree);
	depth = isl_schedule_node_get_schedule_depth(pos);
	n = isl_schedule_tree_band_n_member(tree);
	ma = isl_multi_aff_copy(data->sched);
	ma = isl_multi_aff_drop_dims(ma, isl_dim_out, 0, depth);
	ma = isl_multi_aff_drop_dims(ma, isl_dim_out, n, data->dim - depth - n);
	mupa = isl_multi_union_pw_aff_from_multi_aff(ma);
	partial = isl_schedule_tree_band_get_partial_schedule(tree);
	has_id = isl_multi_union_pw_aff_has_tuple_id(partial, isl_dim_set);
	if (has_id < 0) {
		partial = isl_multi_union_pw_aff_free(partial);
	} else if (has_id) {
		isl_id *id;
		id = isl_multi_union_pw_aff_get_tuple_id(partial, isl_dim_set);
		mupa = isl_multi_union_pw_aff_set_tuple_id(mupa,
							    isl_dim_set, id);
	}
	partial = isl_multi_union_pw_aff_union_add(partial, mupa);
	tree = isl_schedule_tree_band_set_partial_schedule(tree, partial);

	return tree;
}

/* Drop the parameters in "uset" that are not also in "space".
 * "n" is the number of parameters in "space".
 */
static __isl_give isl_union_set *union_set_drop_extra_params(
	__isl_take isl_union_set *uset, __isl_keep isl_space *space, int n)
{
	int n2;

	uset = isl_union_set_align_params(uset, isl_space_copy(space));
	n2 = isl_union_set_dim(uset, isl_dim_param);
	uset = isl_union_set_project_out(uset, isl_dim_param, n, n2 - n);

	return uset;
}

/* Update the context tree root "tree" to refer to the group instances
 * in data->group rather than the original domain elements in data->domain.
 * "pos" is the position in the original schedule tree where the modified
 * "tree" will be attached.
 *
 * We do not actually need to update "tree" since a context node only
 * refers to the schedule space.  However, we may need to update "data"
 * to not refer to any parameters introduced by the context node.
 */
static __isl_give isl_schedule_tree *group_context(
	__isl_take isl_schedule_tree *tree, __isl_keep isl_schedule_node *pos,
	struct isl_schedule_group_data *data)
{
	isl_space *space;
	isl_union_set *domain;
	int n1, n2;
	int involves;

	if (isl_schedule_node_get_tree_depth(pos) == 1)
		return tree;

	domain = isl_schedule_node_get_universe_domain(pos);
	space = isl_union_set_get_space(domain);
	isl_union_set_free(domain);

	n1 = isl_space_dim(space, isl_dim_param);
	data->expansion = isl_union_map_align_params(data->expansion, space);
	n2 = isl_union_map_dim(data->expansion, isl_dim_param);

	if (!data->expansion)
		return isl_schedule_tree_free(tree);
	if (n1 == n2)
		return tree;

	involves = isl_union_map_involves_dims(data->expansion,
				isl_dim_param, n1, n2 - n1);
	if (involves < 0)
		return isl_schedule_tree_free(tree);
	if (involves)
		isl_die(isl_schedule_node_get_ctx(pos), isl_error_invalid,
			"grouping cannot only refer to global parameters",
			return isl_schedule_tree_free(tree));

	data->expansion = isl_union_map_project_out(data->expansion,
				isl_dim_param, n1, n2 - n1);
	space = isl_union_map_get_space(data->expansion);

	data->contraction = isl_union_pw_multi_aff_align_params(
				data->contraction, isl_space_copy(space));
	n2 = isl_union_pw_multi_aff_dim(data->contraction, isl_dim_param);
	data->contraction = isl_union_pw_multi_aff_drop_dims(data->contraction,
				isl_dim_param, n1, n2 - n1);

	data->domain = union_set_drop_extra_params(data->domain, space, n1);
	data->domain_universe =
		union_set_drop_extra_params(data->domain_universe, space, n1);
	data->group = union_set_drop_extra_params(data->group, space, n1);
	data->group_universe =
		union_set_drop_extra_params(data->group_universe, space, n1);

	data->sched = isl_multi_aff_align_params(data->sched,
				isl_space_copy(space));
	n2 = isl_multi_aff_dim(data->sched, isl_dim_param);
	data->sched = isl_multi_aff_drop_dims(data->sched,
				isl_dim_param, n1, n2 - n1);

	isl_space_free(space);

	return tree;
}

/* Update the domain tree root "tree" to refer to the group instances
 * in data->group rather than the original domain elements in data->domain.
 * "pos" is the position in the original schedule tree where the modified
 * "tree" will be attached.
 *
 * We first double-check that all grouped domain elements are actually
 * part of the root domain and then replace those elements by the group
 * instances.
 */
static __isl_give isl_schedule_tree *group_domain(
	__isl_take isl_schedule_tree *tree, __isl_keep isl_schedule_node *pos,
	struct isl_schedule_group_data *data)
{
	isl_union_set *domain;
	int is_subset;

	domain = isl_schedule_tree_domain_get_domain(tree);
	is_subset = isl_union_set_is_subset(data->domain, domain);
	isl_union_set_free(domain);
	if (is_subset < 0)
		return isl_schedule_tree_free(tree);
	if (!is_subset)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_internal,
			"grouped domain should be part of outer domain",
			return isl_schedule_tree_free(tree));
	domain = isl_schedule_tree_domain_get_domain(tree);
	domain = isl_union_set_subtract(domain,
				isl_union_set_copy(data->domain));
	domain = isl_union_set_union(domain, isl_union_set_copy(data->group));
	tree = isl_schedule_tree_domain_set_domain(tree, domain);

	return tree;
}

/* Update the expansion tree root "tree" to refer to the group instances
 * in data->group rather than the original domain elements in data->domain.
 * "pos" is the position in the original schedule tree where the modified
 * "tree" will be attached.
 *
 * Let G_1 -> D_1 be the expansion of "tree" and G_2 -> D_2 the newly
 * introduced expansion in a descendant of "tree".
 * We first double-check that D_2 is a subset of D_1.
 * Then we remove D_2 from the range of G_1 -> D_1 and add the mapping
 * G_1 -> D_1 . D_2 -> G_2.
 * Simmilarly, we restrict the domain of the contraction to the universe
 * of the range of the updated expansion and add G_2 -> D_2 . D_1 -> G_1,
 * attempting to remove the domain constraints of this additional part.
 */
static __isl_give isl_schedule_tree *group_expansion(
	__isl_take isl_schedule_tree *tree, __isl_keep isl_schedule_node *pos,
	struct isl_schedule_group_data *data)
{
	isl_union_set *domain;
	isl_union_map *expansion, *umap;
	isl_union_pw_multi_aff *contraction, *upma;
	int is_subset;

	expansion = isl_schedule_tree_expansion_get_expansion(tree);
	domain = isl_union_map_range(expansion);
	is_subset = isl_union_set_is_subset(data->domain, domain);
	isl_union_set_free(domain);
	if (is_subset < 0)
		return isl_schedule_tree_free(tree);
	if (!is_subset)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_internal,
			"grouped domain should be part "
			"of outer expansion domain",
			return isl_schedule_tree_free(tree));
	expansion = isl_schedule_tree_expansion_get_expansion(tree);
	umap = isl_union_map_from_union_pw_multi_aff(
			isl_union_pw_multi_aff_copy(data->contraction));
	umap = isl_union_map_apply_range(expansion, umap);
	expansion = isl_schedule_tree_expansion_get_expansion(tree);
	expansion = isl_union_map_subtract_range(expansion,
				isl_union_set_copy(data->domain));
	expansion = isl_union_map_union(expansion, umap);
	umap = isl_union_map_universe(isl_union_map_copy(expansion));
	domain = isl_union_map_range(umap);
	contraction = isl_schedule_tree_expansion_get_contraction(tree);
	umap = isl_union_map_from_union_pw_multi_aff(contraction);
	umap = isl_union_map_apply_range(isl_union_map_copy(data->expansion),
					umap);
	upma = isl_union_pw_multi_aff_from_union_map(umap);
	contraction = isl_schedule_tree_expansion_get_contraction(tree);
	contraction = isl_union_pw_multi_aff_intersect_domain(contraction,
								domain);
	domain = isl_union_pw_multi_aff_domain(
				isl_union_pw_multi_aff_copy(upma));
	upma = isl_union_pw_multi_aff_gist(upma, domain);
	contraction = isl_union_pw_multi_aff_union_add(contraction, upma);
	tree = isl_schedule_tree_expansion_set_contraction_and_expansion(tree,
							contraction, expansion);

	return tree;
}

/* Update the tree root "tree" to refer to the group instances
 * in data->group rather than the original domain elements in data->domain.
 * "pos" is the position in the original schedule tree where the modified
 * "tree" will be attached.
 *
 * If we have come across a domain or expansion node before (data->finished
 * is set), then we no longer need perform any modifications.
 *
 * If "tree" is a filter, then we add data->group_universe to the filter.
 * We also remove data->domain_universe from the filter if all the domain
 * elements in this universe that reach the filter node are part of
 * the elements that are being grouped by data->expansion.
 * If "tree" is a band, domain or expansion, then it is handled
 * in a separate function.
 */
static __isl_give isl_schedule_tree *group_ancestor(
	__isl_take isl_schedule_tree *tree, __isl_keep isl_schedule_node *pos,
	void *user)
{
	struct isl_schedule_group_data *data = user;
	isl_union_set *domain;
	int is_covered;

	if (!tree || !pos)
		return isl_schedule_tree_free(tree);

	if (data->finished)
		return tree;

	switch (isl_schedule_tree_get_type(tree)) {
	case isl_schedule_node_error:
		return isl_schedule_tree_free(tree);
	case isl_schedule_node_extension:
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_unsupported,
			"grouping not allowed in extended tree",
			return isl_schedule_tree_free(tree));
	case isl_schedule_node_band:
		tree = group_band(tree, pos, data);
		break;
	case isl_schedule_node_context:
		tree = group_context(tree, pos, data);
		break;
	case isl_schedule_node_domain:
		tree = group_domain(tree, pos, data);
		data->finished = 1;
		break;
	case isl_schedule_node_filter:
		domain = isl_schedule_node_get_domain(pos);
		is_covered = locally_covered_by_domain(domain, data);
		isl_union_set_free(domain);
		if (is_covered < 0)
			return isl_schedule_tree_free(tree);
		domain = isl_schedule_tree_filter_get_filter(tree);
		if (is_covered)
			domain = isl_union_set_subtract(domain,
				    isl_union_set_copy(data->domain_universe));
		domain = isl_union_set_union(domain,
				    isl_union_set_copy(data->group_universe));
		tree = isl_schedule_tree_filter_set_filter(tree, domain);
		break;
	case isl_schedule_node_expansion:
		tree = group_expansion(tree, pos, data);
		data->finished = 1;
		break;
	case isl_schedule_node_leaf:
	case isl_schedule_node_guard:
	case isl_schedule_node_mark:
	case isl_schedule_node_sequence:
	case isl_schedule_node_set:
		break;
	}

	return tree;
}

/* Group the domain elements that reach "node" into instances
 * of a single statement with identifier "group_id".
 * In particular, group the domain elements according to their
 * prefix schedule.
 *
 * That is, introduce an expansion node with as contraction
 * the prefix schedule (with the target space replaced by "group_id")
 * and as expansion the inverse of this contraction (with its range
 * intersected with the domain elements that reach "node").
 * The outer nodes are then modified to refer to the group instances
 * instead of the original domain elements.
 *
 * No instance of "group_id" is allowed to reach "node" prior
 * to the grouping.
 * No ancestor of "node" is allowed to be an extension node.
 *
 * Return a pointer to original node in tree, i.e., the child
 * of the newly introduced expansion node.
 */
__isl_give isl_schedule_node *isl_schedule_node_group(
	__isl_take isl_schedule_node *node, __isl_take isl_id *group_id)
{
	struct isl_schedule_group_data data = { 0 };
	isl_space *space;
	isl_union_set *domain;
	isl_union_pw_multi_aff *contraction;
	isl_union_map *expansion;
	int disjoint;

	if (!node || !group_id)
		goto error;
	if (check_insert(node) < 0)
		goto error;

	domain = isl_schedule_node_get_domain(node);
	data.domain = isl_union_set_copy(domain);
	data.domain_universe = isl_union_set_copy(domain);
	data.domain_universe = isl_union_set_universe(data.domain_universe);

	data.dim = isl_schedule_node_get_schedule_depth(node);
	if (data.dim == 0) {
		isl_ctx *ctx;
		isl_set *set;
		isl_union_set *group;
		isl_union_map *univ;

		ctx = isl_schedule_node_get_ctx(node);
		space = isl_space_set_alloc(ctx, 0, 0);
		space = isl_space_set_tuple_id(space, isl_dim_set, group_id);
		set = isl_set_universe(isl_space_copy(space));
		group = isl_union_set_from_set(set);
		expansion = isl_union_map_from_domain_and_range(domain, group);
		univ = isl_union_map_universe(isl_union_map_copy(expansion));
		contraction = isl_union_pw_multi_aff_from_union_map(univ);
		expansion = isl_union_map_reverse(expansion);
	} else {
		isl_multi_union_pw_aff *prefix;
		isl_union_set *univ;

		prefix =
		isl_schedule_node_get_prefix_schedule_multi_union_pw_aff(node);
		prefix = isl_multi_union_pw_aff_set_tuple_id(prefix,
							isl_dim_set, group_id);
		space = isl_multi_union_pw_aff_get_space(prefix);
		contraction = isl_union_pw_multi_aff_from_multi_union_pw_aff(
							prefix);
		univ = isl_union_set_universe(isl_union_set_copy(domain));
		contraction =
		    isl_union_pw_multi_aff_intersect_domain(contraction, univ);
		expansion = isl_union_map_from_union_pw_multi_aff(
				    isl_union_pw_multi_aff_copy(contraction));
		expansion = isl_union_map_reverse(expansion);
		expansion = isl_union_map_intersect_range(expansion, domain);
	}
	space = isl_space_map_from_set(space);
	data.sched = isl_multi_aff_identity(space);
	data.group = isl_union_map_domain(isl_union_map_copy(expansion));
	data.group = isl_union_set_coalesce(data.group);
	data.group_universe = isl_union_set_copy(data.group);
	data.group_universe = isl_union_set_universe(data.group_universe);
	data.expansion = isl_union_map_copy(expansion);
	data.contraction = isl_union_pw_multi_aff_copy(contraction);
	node = isl_schedule_node_insert_expansion(node, contraction, expansion);

	disjoint = isl_union_set_is_disjoint(data.domain_universe,
					    data.group_universe);

	node = update_ancestors(node, &group_ancestor, &data);

	isl_union_set_free(data.domain);
	isl_union_set_free(data.domain_universe);
	isl_union_set_free(data.group);
	isl_union_set_free(data.group_universe);
	isl_multi_aff_free(data.sched);
	isl_union_map_free(data.expansion);
	isl_union_pw_multi_aff_free(data.contraction);

	node = isl_schedule_node_child(node, 0);

	if (!node || disjoint < 0)
		return isl_schedule_node_free(node);
	if (!disjoint)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"group instances already reach node",
			isl_schedule_node_free(node));

	return node;
error:
	isl_schedule_node_free(node);
	isl_id_free(group_id);
	return NULL;
}

/* Compute the gist of the given band node with respect to "context".
 */
__isl_give isl_schedule_node *isl_schedule_node_band_gist(
	__isl_take isl_schedule_node *node, __isl_take isl_union_set *context)
{
	isl_schedule_tree *tree;

	tree = isl_schedule_node_get_tree(node);
	tree = isl_schedule_tree_band_gist(tree, context);
	return isl_schedule_node_graft_tree(node, tree);
}

/* Internal data structure for isl_schedule_node_gist.
 * "n_expansion" is the number of outer expansion nodes
 * with respect to the current position
 * "filters" contains an element for each outer filter, expansion or
 * extension node with respect to the current position, each representing
 * the intersection of the previous element and the filter on the filter node
 * or the expansion/extension of the previous element.
 * The first element in the original context passed to isl_schedule_node_gist.
 */
struct isl_node_gist_data {
	int n_expansion;
	isl_union_set_list *filters;
};

/* Enter the expansion node "node" during a isl_schedule_node_gist traversal.
 *
 * In particular, add an extra element to data->filters containing
 * the expansion of the previous element and replace the expansion
 * and contraction on "node" by the gist with respect to these filters.
 * Also keep track of the fact that we have entered another expansion.
 */
static __isl_give isl_schedule_node *gist_enter_expansion(
	__isl_take isl_schedule_node *node, struct isl_node_gist_data *data)
{
	int n;
	isl_union_set *inner;
	isl_union_map *expansion;
	isl_union_pw_multi_aff *contraction;

	data->n_expansion++;

	n = isl_union_set_list_n_union_set(data->filters);
	inner = isl_union_set_list_get_union_set(data->filters, n - 1);
	expansion = isl_schedule_node_expansion_get_expansion(node);
	inner = isl_union_set_apply(inner, expansion);

	contraction = isl_schedule_node_expansion_get_contraction(node);
	contraction = isl_union_pw_multi_aff_gist(contraction,
						isl_union_set_copy(inner));

	data->filters = isl_union_set_list_add(data->filters, inner);

	inner = isl_union_set_list_get_union_set(data->filters, n - 1);
	expansion = isl_schedule_node_expansion_get_expansion(node);
	expansion = isl_union_map_gist_domain(expansion, inner);
	node = isl_schedule_node_expansion_set_contraction_and_expansion(node,
						contraction, expansion);

	return node;
}

/* Leave the expansion node "node" during a isl_schedule_node_gist traversal.
 *
 * In particular, remove the element in data->filters that was added by
 * gist_enter_expansion and decrement the number of outer expansions.
 *
 * The expansion has already been simplified in gist_enter_expansion.
 * If this simplification results in an identity expansion, then
 * it is removed here.
 */
static __isl_give isl_schedule_node *gist_leave_expansion(
	__isl_take isl_schedule_node *node, struct isl_node_gist_data *data)
{
	int n;
	isl_bool identity;
	isl_union_map *expansion;

	expansion = isl_schedule_node_expansion_get_expansion(node);
	identity = isl_union_map_is_identity(expansion);
	isl_union_map_free(expansion);

	if (identity < 0)
		node = isl_schedule_node_free(node);
	else if (identity)
		node = isl_schedule_node_delete(node);

	n = isl_union_set_list_n_union_set(data->filters);
	data->filters = isl_union_set_list_drop(data->filters, n - 1, 1);

	data->n_expansion--;

	return node;
}

/* Enter the extension node "node" during a isl_schedule_node_gist traversal.
 *
 * In particular, add an extra element to data->filters containing
 * the union of the previous element with the additional domain elements
 * introduced by the extension.
 */
static __isl_give isl_schedule_node *gist_enter_extension(
	__isl_take isl_schedule_node *node, struct isl_node_gist_data *data)
{
	int n;
	isl_union_set *inner, *extra;
	isl_union_map *extension;

	n = isl_union_set_list_n_union_set(data->filters);
	inner = isl_union_set_list_get_union_set(data->filters, n - 1);
	extension = isl_schedule_node_extension_get_extension(node);
	extra = isl_union_map_range(extension);
	inner = isl_union_set_union(inner, extra);

	data->filters = isl_union_set_list_add(data->filters, inner);

	return node;
}

/* Can we finish gisting at this node?
 * That is, is the filter on the current filter node a subset of
 * the original context passed to isl_schedule_node_gist?
 * If we have gone through any expansions, then we cannot perform
 * this test since the current domain elements are incomparable
 * to the domain elements in the original context.
 */
static int gist_done(__isl_keep isl_schedule_node *node,
	struct isl_node_gist_data *data)
{
	isl_union_set *filter, *outer;
	int subset;

	if (data->n_expansion != 0)
		return 0;

	filter = isl_schedule_node_filter_get_filter(node);
	outer = isl_union_set_list_get_union_set(data->filters, 0);
	subset = isl_union_set_is_subset(filter, outer);
	isl_union_set_free(outer);
	isl_union_set_free(filter);

	return subset;
}

/* Callback for "traverse" to enter a node and to move
 * to the deepest initial subtree that should be traversed
 * by isl_schedule_node_gist.
 *
 * The "filters" list is extended by one element each time
 * we come across a filter node by the result of intersecting
 * the last element in the list with the filter on the filter node.
 *
 * If the filter on the current filter node is a subset of
 * the original context passed to isl_schedule_node_gist,
 * then there is no need to go into its subtree since it cannot
 * be further simplified by the context.  The "filters" list is
 * still extended for consistency, but the actual value of the
 * added element is immaterial since it will not be used.
 *
 * Otherwise, the filter on the current filter node is replaced by
 * the gist of the original filter with respect to the intersection
 * of the original context with the intermediate filters.
 *
 * If the new element in the "filters" list is empty, then no elements
 * can reach the descendants of the current filter node.  The subtree
 * underneath the filter node is therefore removed.
 *
 * Each expansion node we come across is handled by
 * gist_enter_expansion.
 *
 * Each extension node we come across is handled by
 * gist_enter_extension.
 */
static __isl_give isl_schedule_node *gist_enter(
	__isl_take isl_schedule_node *node, void *user)
{
	struct isl_node_gist_data *data = user;

	do {
		isl_union_set *filter, *inner;
		int done, empty;
		int n;

		switch (isl_schedule_node_get_type(node)) {
		case isl_schedule_node_error:
			return isl_schedule_node_free(node);
		case isl_schedule_node_expansion:
			node = gist_enter_expansion(node, data);
			continue;
		case isl_schedule_node_extension:
			node = gist_enter_extension(node, data);
			continue;
		case isl_schedule_node_band:
		case isl_schedule_node_context:
		case isl_schedule_node_domain:
		case isl_schedule_node_guard:
		case isl_schedule_node_leaf:
		case isl_schedule_node_mark:
		case isl_schedule_node_sequence:
		case isl_schedule_node_set:
			continue;
		case isl_schedule_node_filter:
			break;
		}
		done = gist_done(node, data);
		filter = isl_schedule_node_filter_get_filter(node);
		if (done < 0 || done) {
			data->filters = isl_union_set_list_add(data->filters,
								filter);
			if (done < 0)
				return isl_schedule_node_free(node);
			return node;
		}
		n = isl_union_set_list_n_union_set(data->filters);
		inner = isl_union_set_list_get_union_set(data->filters, n - 1);
		filter = isl_union_set_gist(filter, isl_union_set_copy(inner));
		node = isl_schedule_node_filter_set_filter(node,
						isl_union_set_copy(filter));
		filter = isl_union_set_intersect(filter, inner);
		empty = isl_union_set_is_empty(filter);
		data->filters = isl_union_set_list_add(data->filters, filter);
		if (empty < 0)
			return isl_schedule_node_free(node);
		if (!empty)
			continue;
		node = isl_schedule_node_child(node, 0);
		node = isl_schedule_node_cut(node);
		node = isl_schedule_node_parent(node);
		return node;
	} while (isl_schedule_node_has_children(node) &&
		(node = isl_schedule_node_first_child(node)) != NULL);

	return node;
}

/* Callback for "traverse" to leave a node for isl_schedule_node_gist.
 *
 * In particular, if the current node is a filter node, then we remove
 * the element on the "filters" list that was added when we entered
 * the node.  There is no need to compute any gist here, since we
 * already did that when we entered the node.
 *
 * Expansion nodes are handled by gist_leave_expansion.
 *
 * If the current node is an extension, then remove the element
 * in data->filters that was added by gist_enter_extension.
 *
 * If the current node is a band node, then we compute the gist of
 * the band node with respect to the intersection of the original context
 * and the intermediate filters.
 *
 * If the current node is a sequence or set node, then some of
 * the filter children may have become empty and so they are removed.
 * If only one child is left, then the set or sequence node along with
 * the single remaining child filter is removed.  The filter can be
 * removed because the filters on a sequence or set node are supposed
 * to partition the incoming domain instances.
 * In principle, it should then be impossible for there to be zero
 * remaining children, but should this happen, we replace the entire
 * subtree with an empty filter.
 */
static __isl_give isl_schedule_node *gist_leave(
	__isl_take isl_schedule_node *node, void *user)
{
	struct isl_node_gist_data *data = user;
	isl_schedule_tree *tree;
	int i, n;
	isl_union_set *filter;

	switch (isl_schedule_node_get_type(node)) {
	case isl_schedule_node_error:
		return isl_schedule_node_free(node);
	case isl_schedule_node_expansion:
		node = gist_leave_expansion(node, data);
		break;
	case isl_schedule_node_extension:
	case isl_schedule_node_filter:
		n = isl_union_set_list_n_union_set(data->filters);
		data->filters = isl_union_set_list_drop(data->filters,
							n - 1, 1);
		break;
	case isl_schedule_node_band:
		n = isl_union_set_list_n_union_set(data->filters);
		filter = isl_union_set_list_get_union_set(data->filters, n - 1);
		node = isl_schedule_node_band_gist(node, filter);
		break;
	case isl_schedule_node_set:
	case isl_schedule_node_sequence:
		tree = isl_schedule_node_get_tree(node);
		n = isl_schedule_tree_n_children(tree);
		for (i = n - 1; i >= 0; --i) {
			isl_schedule_tree *child;
			isl_union_set *filter;
			int empty;

			child = isl_schedule_tree_get_child(tree, i);
			filter = isl_schedule_tree_filter_get_filter(child);
			empty = isl_union_set_is_empty(filter);
			isl_union_set_free(filter);
			isl_schedule_tree_free(child);
			if (empty < 0)
				tree = isl_schedule_tree_free(tree);
			else if (empty)
				tree = isl_schedule_tree_drop_child(tree, i);
		}
		n = isl_schedule_tree_n_children(tree);
		node = isl_schedule_node_graft_tree(node, tree);
		if (n == 1) {
			node = isl_schedule_node_delete(node);
			node = isl_schedule_node_delete(node);
		} else if (n == 0) {
			isl_space *space;

			filter =
			    isl_union_set_list_get_union_set(data->filters, 0);
			space = isl_union_set_get_space(filter);
			isl_union_set_free(filter);
			filter = isl_union_set_empty(space);
			node = isl_schedule_node_cut(node);
			node = isl_schedule_node_insert_filter(node, filter);
		}
		break;
	case isl_schedule_node_context:
	case isl_schedule_node_domain:
	case isl_schedule_node_guard:
	case isl_schedule_node_leaf:
	case isl_schedule_node_mark:
		break;
	}

	return node;
}

/* Compute the gist of the subtree at "node" with respect to
 * the reaching domain elements in "context".
 * In particular, compute the gist of all band and filter nodes
 * in the subtree with respect to "context".  Children of set or sequence
 * nodes that end up with an empty filter are removed completely.
 *
 * We keep track of the intersection of "context" with all outer filters
 * of the current node within the subtree in the final element of "filters".
 * Initially, this list contains the single element "context" and it is
 * extended or shortened each time we enter or leave a filter node.
 */
__isl_give isl_schedule_node *isl_schedule_node_gist(
	__isl_take isl_schedule_node *node, __isl_take isl_union_set *context)
{
	struct isl_node_gist_data data;

	data.n_expansion = 0;
	data.filters = isl_union_set_list_from_union_set(context);
	node = traverse(node, &gist_enter, &gist_leave, &data);
	isl_union_set_list_free(data.filters);
	return node;
}

/* Intersect the domain of domain node "node" with "domain".
 *
 * If the domain of "node" is already a subset of "domain",
 * then nothing needs to be changed.
 *
 * Otherwise, we replace the domain of the domain node by the intersection
 * and simplify the subtree rooted at "node" with respect to this intersection.
 */
__isl_give isl_schedule_node *isl_schedule_node_domain_intersect_domain(
	__isl_take isl_schedule_node *node, __isl_take isl_union_set *domain)
{
	isl_schedule_tree *tree;
	isl_union_set *uset;
	int is_subset;

	if (!node || !domain)
		goto error;

	uset = isl_schedule_tree_domain_get_domain(node->tree);
	is_subset = isl_union_set_is_subset(uset, domain);
	isl_union_set_free(uset);
	if (is_subset < 0)
		goto error;
	if (is_subset) {
		isl_union_set_free(domain);
		return node;
	}

	tree = isl_schedule_tree_copy(node->tree);
	uset = isl_schedule_tree_domain_get_domain(tree);
	uset = isl_union_set_intersect(uset, domain);
	tree = isl_schedule_tree_domain_set_domain(tree,
						    isl_union_set_copy(uset));
	node = isl_schedule_node_graft_tree(node, tree);

	node = isl_schedule_node_child(node, 0);
	node = isl_schedule_node_gist(node, uset);
	node = isl_schedule_node_parent(node);

	return node;
error:
	isl_schedule_node_free(node);
	isl_union_set_free(domain);
	return NULL;
}

/* Replace the domain of domain node "node" with the gist
 * of the original domain with respect to the parameter domain "context".
 */
__isl_give isl_schedule_node *isl_schedule_node_domain_gist_params(
	__isl_take isl_schedule_node *node, __isl_take isl_set *context)
{
	isl_union_set *domain;
	isl_schedule_tree *tree;

	if (!node || !context)
		goto error;

	tree = isl_schedule_tree_copy(node->tree);
	domain = isl_schedule_tree_domain_get_domain(node->tree);
	domain = isl_union_set_gist_params(domain, context);
	tree = isl_schedule_tree_domain_set_domain(tree, domain);
	node = isl_schedule_node_graft_tree(node, tree);

	return node;
error:
	isl_schedule_node_free(node);
	isl_set_free(context);
	return NULL;
}

/* Internal data structure for isl_schedule_node_get_subtree_expansion.
 * "expansions" contains a list of accumulated expansions
 * for each outer expansion, set or sequence node.  The first element
 * in the list is an identity mapping on the reaching domain elements.
 * "res" collects the results.
 */
struct isl_subtree_expansion_data {
	isl_union_map_list *expansions;
	isl_union_map *res;
};

/* Callback for "traverse" to enter a node and to move
 * to the deepest initial subtree that should be traversed
 * by isl_schedule_node_get_subtree_expansion.
 *
 * Whenever we come across an expansion node, the last element
 * of data->expansions is combined with the expansion
 * on the expansion node.
 *
 * Whenever we come across a filter node that is the child
 * of a set or sequence node, data->expansions is extended
 * with a new element that restricts the previous element
 * to the elements selected by the filter.
 * The previous element can then be reused while backtracking.
 */
static __isl_give isl_schedule_node *subtree_expansion_enter(
	__isl_take isl_schedule_node *node, void *user)
{
	struct isl_subtree_expansion_data *data = user;

	do {
		enum isl_schedule_node_type type;
		isl_union_set *filter;
		isl_union_map *inner, *expansion;
		int n;

		switch (isl_schedule_node_get_type(node)) {
		case isl_schedule_node_error:
			return isl_schedule_node_free(node);
		case isl_schedule_node_filter:
			type = isl_schedule_node_get_parent_type(node);
			if (type != isl_schedule_node_set &&
			    type != isl_schedule_node_sequence)
				break;
			filter = isl_schedule_node_filter_get_filter(node);
			n = isl_union_map_list_n_union_map(data->expansions);
			inner =
			    isl_union_map_list_get_union_map(data->expansions,
								n - 1);
			inner = isl_union_map_intersect_range(inner, filter);
			data->expansions =
			    isl_union_map_list_add(data->expansions, inner);
			break;
		case isl_schedule_node_expansion:
			n = isl_union_map_list_n_union_map(data->expansions);
			expansion =
				isl_schedule_node_expansion_get_expansion(node);
			inner =
			    isl_union_map_list_get_union_map(data->expansions,
								n - 1);
			inner = isl_union_map_apply_range(inner, expansion);
			data->expansions =
			    isl_union_map_list_set_union_map(data->expansions,
								n - 1, inner);
			break;
		case isl_schedule_node_band:
		case isl_schedule_node_context:
		case isl_schedule_node_domain:
		case isl_schedule_node_extension:
		case isl_schedule_node_guard:
		case isl_schedule_node_leaf:
		case isl_schedule_node_mark:
		case isl_schedule_node_sequence:
		case isl_schedule_node_set:
			break;
		}
	} while (isl_schedule_node_has_children(node) &&
		(node = isl_schedule_node_first_child(node)) != NULL);

	return node;
}

/* Callback for "traverse" to leave a node for
 * isl_schedule_node_get_subtree_expansion.
 *
 * If we come across a filter node that is the child
 * of a set or sequence node, then we remove the element
 * of data->expansions that was added in subtree_expansion_enter.
 *
 * If we reach a leaf node, then the accumulated expansion is
 * added to data->res.
 */
static __isl_give isl_schedule_node *subtree_expansion_leave(
	__isl_take isl_schedule_node *node, void *user)
{
	struct isl_subtree_expansion_data *data = user;
	int n;
	isl_union_map *inner;
	enum isl_schedule_node_type type;

	switch (isl_schedule_node_get_type(node)) {
	case isl_schedule_node_error:
		return isl_schedule_node_free(node);
	case isl_schedule_node_filter:
		type = isl_schedule_node_get_parent_type(node);
		if (type != isl_schedule_node_set &&
		    type != isl_schedule_node_sequence)
			break;
		n = isl_union_map_list_n_union_map(data->expansions);
		data->expansions = isl_union_map_list_drop(data->expansions,
							n - 1, 1);
		break;
	case isl_schedule_node_leaf:
		n = isl_union_map_list_n_union_map(data->expansions);
		inner = isl_union_map_list_get_union_map(data->expansions,
							n - 1);
		data->res = isl_union_map_union(data->res, inner);
		break;
	case isl_schedule_node_band:
	case isl_schedule_node_context:
	case isl_schedule_node_domain:
	case isl_schedule_node_expansion:
	case isl_schedule_node_extension:
	case isl_schedule_node_guard:
	case isl_schedule_node_mark:
	case isl_schedule_node_sequence:
	case isl_schedule_node_set:
		break;
	}

	return node;
}

/* Return a mapping from the domain elements that reach "node"
 * to the corresponding domain elements in the leaves of the subtree
 * rooted at "node" obtained by composing the intermediate expansions.
 *
 * We start out with an identity mapping between the domain elements
 * that reach "node" and compose it with all the expansions
 * on a path from "node" to a leaf while traversing the subtree.
 * Within the children of an a sequence or set node, the
 * accumulated expansion is restricted to the elements selected
 * by the filter child.
 */
__isl_give isl_union_map *isl_schedule_node_get_subtree_expansion(
	__isl_keep isl_schedule_node *node)
{
	struct isl_subtree_expansion_data data;
	isl_space *space;
	isl_union_set *domain;
	isl_union_map *expansion;

	if (!node)
		return NULL;

	domain = isl_schedule_node_get_universe_domain(node);
	space = isl_union_set_get_space(domain);
	expansion = isl_union_set_identity(domain);
	data.res = isl_union_map_empty(space);
	data.expansions = isl_union_map_list_from_union_map(expansion);

	node = isl_schedule_node_copy(node);
	node = traverse(node, &subtree_expansion_enter,
			&subtree_expansion_leave, &data);
	if (!node)
		data.res = isl_union_map_free(data.res);
	isl_schedule_node_free(node);

	isl_union_map_list_free(data.expansions);

	return data.res;
}

/* Internal data structure for isl_schedule_node_get_subtree_contraction.
 * "contractions" contains a list of accumulated contractions
 * for each outer expansion, set or sequence node.  The first element
 * in the list is an identity mapping on the reaching domain elements.
 * "res" collects the results.
 */
struct isl_subtree_contraction_data {
	isl_union_pw_multi_aff_list *contractions;
	isl_union_pw_multi_aff *res;
};

/* Callback for "traverse" to enter a node and to move
 * to the deepest initial subtree that should be traversed
 * by isl_schedule_node_get_subtree_contraction.
 *
 * Whenever we come across an expansion node, the last element
 * of data->contractions is combined with the contraction
 * on the expansion node.
 *
 * Whenever we come across a filter node that is the child
 * of a set or sequence node, data->contractions is extended
 * with a new element that restricts the previous element
 * to the elements selected by the filter.
 * The previous element can then be reused while backtracking.
 */
static __isl_give isl_schedule_node *subtree_contraction_enter(
	__isl_take isl_schedule_node *node, void *user)
{
	struct isl_subtree_contraction_data *data = user;

	do {
		enum isl_schedule_node_type type;
		isl_union_set *filter;
		isl_union_pw_multi_aff *inner, *contraction;
		int n;

		switch (isl_schedule_node_get_type(node)) {
		case isl_schedule_node_error:
			return isl_schedule_node_free(node);
		case isl_schedule_node_filter:
			type = isl_schedule_node_get_parent_type(node);
			if (type != isl_schedule_node_set &&
			    type != isl_schedule_node_sequence)
				break;
			filter = isl_schedule_node_filter_get_filter(node);
			n = isl_union_pw_multi_aff_list_n_union_pw_multi_aff(
						data->contractions);
			inner =
			    isl_union_pw_multi_aff_list_get_union_pw_multi_aff(
						data->contractions, n - 1);
			inner = isl_union_pw_multi_aff_intersect_domain(inner,
								filter);
			data->contractions =
			    isl_union_pw_multi_aff_list_add(data->contractions,
								inner);
			break;
		case isl_schedule_node_expansion:
			n = isl_union_pw_multi_aff_list_n_union_pw_multi_aff(
						data->contractions);
			contraction =
			    isl_schedule_node_expansion_get_contraction(node);
			inner =
			    isl_union_pw_multi_aff_list_get_union_pw_multi_aff(
						data->contractions, n - 1);
			inner =
			    isl_union_pw_multi_aff_pullback_union_pw_multi_aff(
						inner, contraction);
			data->contractions =
			    isl_union_pw_multi_aff_list_set_union_pw_multi_aff(
					data->contractions, n - 1, inner);
			break;
		case isl_schedule_node_band:
		case isl_schedule_node_context:
		case isl_schedule_node_domain:
		case isl_schedule_node_extension:
		case isl_schedule_node_guard:
		case isl_schedule_node_leaf:
		case isl_schedule_node_mark:
		case isl_schedule_node_sequence:
		case isl_schedule_node_set:
			break;
		}
	} while (isl_schedule_node_has_children(node) &&
		(node = isl_schedule_node_first_child(node)) != NULL);

	return node;
}

/* Callback for "traverse" to leave a node for
 * isl_schedule_node_get_subtree_contraction.
 *
 * If we come across a filter node that is the child
 * of a set or sequence node, then we remove the element
 * of data->contractions that was added in subtree_contraction_enter.
 *
 * If we reach a leaf node, then the accumulated contraction is
 * added to data->res.
 */
static __isl_give isl_schedule_node *subtree_contraction_leave(
	__isl_take isl_schedule_node *node, void *user)
{
	struct isl_subtree_contraction_data *data = user;
	int n;
	isl_union_pw_multi_aff *inner;
	enum isl_schedule_node_type type;

	switch (isl_schedule_node_get_type(node)) {
	case isl_schedule_node_error:
		return isl_schedule_node_free(node);
	case isl_schedule_node_filter:
		type = isl_schedule_node_get_parent_type(node);
		if (type != isl_schedule_node_set &&
		    type != isl_schedule_node_sequence)
			break;
		n = isl_union_pw_multi_aff_list_n_union_pw_multi_aff(
						data->contractions);
		data->contractions =
			isl_union_pw_multi_aff_list_drop(data->contractions,
							n - 1, 1);
		break;
	case isl_schedule_node_leaf:
		n = isl_union_pw_multi_aff_list_n_union_pw_multi_aff(
						data->contractions);
		inner = isl_union_pw_multi_aff_list_get_union_pw_multi_aff(
						data->contractions, n - 1);
		data->res = isl_union_pw_multi_aff_union_add(data->res, inner);
		break;
	case isl_schedule_node_band:
	case isl_schedule_node_context:
	case isl_schedule_node_domain:
	case isl_schedule_node_expansion:
	case isl_schedule_node_extension:
	case isl_schedule_node_guard:
	case isl_schedule_node_mark:
	case isl_schedule_node_sequence:
	case isl_schedule_node_set:
		break;
	}

	return node;
}

/* Return a mapping from the domain elements in the leaves of the subtree
 * rooted at "node" to the corresponding domain elements that reach "node"
 * obtained by composing the intermediate contractions.
 *
 * We start out with an identity mapping between the domain elements
 * that reach "node" and compose it with all the contractions
 * on a path from "node" to a leaf while traversing the subtree.
 * Within the children of an a sequence or set node, the
 * accumulated contraction is restricted to the elements selected
 * by the filter child.
 */
__isl_give isl_union_pw_multi_aff *isl_schedule_node_get_subtree_contraction(
	__isl_keep isl_schedule_node *node)
{
	struct isl_subtree_contraction_data data;
	isl_space *space;
	isl_union_set *domain;
	isl_union_pw_multi_aff *contraction;

	if (!node)
		return NULL;

	domain = isl_schedule_node_get_universe_domain(node);
	space = isl_union_set_get_space(domain);
	contraction = isl_union_set_identity_union_pw_multi_aff(domain);
	data.res = isl_union_pw_multi_aff_empty(space);
	data.contractions =
	    isl_union_pw_multi_aff_list_from_union_pw_multi_aff(contraction);

	node = isl_schedule_node_copy(node);
	node = traverse(node, &subtree_contraction_enter,
			&subtree_contraction_leave, &data);
	if (!node)
		data.res = isl_union_pw_multi_aff_free(data.res);
	isl_schedule_node_free(node);

	isl_union_pw_multi_aff_list_free(data.contractions);

	return data.res;
}

/* Do the nearest "n" ancestors of "node" have the types given in "types"
 * (starting at the parent of "node")?
 */
static int has_ancestors(__isl_keep isl_schedule_node *node,
	int n, enum isl_schedule_node_type *types)
{
	int i, n_ancestor;

	if (!node)
		return -1;

	n_ancestor = isl_schedule_tree_list_n_schedule_tree(node->ancestors);
	if (n_ancestor < n)
		return 0;

	for (i = 0; i < n; ++i) {
		isl_schedule_tree *tree;
		int correct_type;

		tree = isl_schedule_tree_list_get_schedule_tree(node->ancestors,
							    n_ancestor - 1 - i);
		if (!tree)
			return -1;
		correct_type = isl_schedule_tree_get_type(tree) == types[i];
		isl_schedule_tree_free(tree);
		if (!correct_type)
			return 0;
	}

	return 1;
}

/* Given a node "node" that appears in an extension (i.e., it is the child
 * of a filter in a sequence inside an extension node), are the spaces
 * of the extension specified by "extension" disjoint from those
 * of both the original extension and the domain elements that reach
 * that original extension?
 */
static int is_disjoint_extension(__isl_keep isl_schedule_node *node,
	__isl_keep isl_union_map *extension)
{
	isl_union_map *old;
	isl_union_set *domain;
	int empty;

	node = isl_schedule_node_copy(node);
	node = isl_schedule_node_parent(node);
	node = isl_schedule_node_parent(node);
	node = isl_schedule_node_parent(node);
	old = isl_schedule_node_extension_get_extension(node);
	domain = isl_schedule_node_get_universe_domain(node);
	isl_schedule_node_free(node);
	old = isl_union_map_universe(old);
	domain = isl_union_set_union(domain, isl_union_map_range(old));
	extension = isl_union_map_copy(extension);
	extension = isl_union_map_intersect_range(extension, domain);
	empty = isl_union_map_is_empty(extension);
	isl_union_map_free(extension);

	return empty;
}

/* Given a node "node" that is governed by an extension node, extend
 * that extension node with "extension".
 *
 * In particular, "node" is the child of a filter in a sequence that
 * is in turn a child of an extension node.  Extend that extension node
 * with "extension".
 *
 * Return a pointer to the parent of the original node (i.e., a filter).
 */
static __isl_give isl_schedule_node *extend_extension(
	__isl_take isl_schedule_node *node, __isl_take isl_union_map *extension)
{
	int pos;
	int disjoint;
	isl_union_map *node_extension;

	node = isl_schedule_node_parent(node);
	pos = isl_schedule_node_get_child_position(node);
	node = isl_schedule_node_parent(node);
	node = isl_schedule_node_parent(node);
	node_extension = isl_schedule_node_extension_get_extension(node);
	disjoint = isl_union_map_is_disjoint(extension, node_extension);
	extension = isl_union_map_union(extension, node_extension);
	node = isl_schedule_node_extension_set_extension(node, extension);
	node = isl_schedule_node_child(node, 0);
	node = isl_schedule_node_child(node, pos);

	if (disjoint < 0)
		return isl_schedule_node_free(node);
	if (!node)
		return NULL;
	if (!disjoint)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"extension domain should be disjoint from earlier "
			"extensions", return isl_schedule_node_free(node));

	return node;
}

/* Return the universe of "uset" if this universe is disjoint from "ref".
 * Otherwise, return "uset".
 *
 * Also check if "uset" itself is disjoint from "ref", reporting
 * an error if it is not.
 */
static __isl_give isl_union_set *replace_by_universe_if_disjoint(
	__isl_take isl_union_set *uset, __isl_keep isl_union_set *ref)
{
	int disjoint;
	isl_union_set *universe;

	disjoint = isl_union_set_is_disjoint(uset, ref);
	if (disjoint < 0)
		return isl_union_set_free(uset);
	if (!disjoint)
		isl_die(isl_union_set_get_ctx(uset), isl_error_invalid,
			"extension domain should be disjoint from "
			"current domain", return isl_union_set_free(uset));

	universe = isl_union_set_universe(isl_union_set_copy(uset));
	disjoint = isl_union_set_is_disjoint(universe, ref);
	if (disjoint >= 0 && disjoint) {
		isl_union_set_free(uset);
		return universe;
	}
	isl_union_set_free(universe);

	if (disjoint < 0)
		return isl_union_set_free(uset);
	return uset;
}

/* Insert an extension node on top of "node" with extension "extension".
 * In addition, insert a filter that separates node from the extension
 * between the extension node and "node".
 * Return a pointer to the inserted filter node.
 *
 * If "node" already appears in an extension (i.e., if it is the child
 * of a filter in a sequence inside an extension node), then extend that
 * extension with "extension" instead.
 * In this case, a pointer to the original filter node is returned.
 * Note that if some of the elements in the new extension live in the
 * same space as those of the original extension or the domain elements
 * reaching the original extension, then we insert a new extension anyway.
 * Otherwise, we would have to adjust the filters in the sequence child
 * of the extension to ensure that the elements in the new extension
 * are filtered out.
 */
static __isl_give isl_schedule_node *insert_extension(
	__isl_take isl_schedule_node *node, __isl_take isl_union_map *extension)
{
	enum isl_schedule_node_type ancestors[] =
		{ isl_schedule_node_filter, isl_schedule_node_sequence,
		  isl_schedule_node_extension };
	isl_union_set *domain;
	isl_union_set *filter;
	int in_ext;

	in_ext = has_ancestors(node, 3, ancestors);
	if (in_ext < 0)
		goto error;
	if (in_ext) {
		int disjoint;

		disjoint = is_disjoint_extension(node, extension);
		if (disjoint < 0)
			goto error;
		if (disjoint)
			return extend_extension(node, extension);
	}

	filter = isl_schedule_node_get_domain(node);
	domain = isl_union_map_range(isl_union_map_copy(extension));
	filter = replace_by_universe_if_disjoint(filter, domain);
	isl_union_set_free(domain);

	node = isl_schedule_node_insert_filter(node, filter);
	node = isl_schedule_node_insert_extension(node, extension);
	node = isl_schedule_node_child(node, 0);
	return node;
error:
	isl_schedule_node_free(node);
	isl_union_map_free(extension);
	return NULL;
}

/* Replace the subtree that "node" points to by "tree" (which has
 * a sequence root with two children), except if the parent of "node"
 * is a sequence as well, in which case "tree" is spliced at the position
 * of "node" in its parent.
 * Return a pointer to the child of the "tree_pos" (filter) child of "tree"
 * in the updated schedule tree.
 */
static __isl_give isl_schedule_node *graft_or_splice(
	__isl_take isl_schedule_node *node, __isl_take isl_schedule_tree *tree,
	int tree_pos)
{
	int pos;

	if (isl_schedule_node_get_parent_type(node) ==
	    isl_schedule_node_sequence) {
		pos = isl_schedule_node_get_child_position(node);
		node = isl_schedule_node_parent(node);
		node = isl_schedule_node_sequence_splice(node, pos, tree);
	} else {
		pos = 0;
		node = isl_schedule_node_graft_tree(node, tree);
	}
	node = isl_schedule_node_child(node, pos + tree_pos);
	node = isl_schedule_node_child(node, 0);

	return node;
}

/* Insert a node "graft" into the schedule tree of "node" such that it
 * is executed before (if "before" is set) or after (if "before" is not set)
 * the node that "node" points to.
 * The root of "graft" is an extension node.
 * Return a pointer to the node that "node" pointed to.
 *
 * We first insert an extension node on top of "node" (or extend
 * the extension node if there already is one), with a filter on "node"
 * separating it from the extension.
 * We then insert a filter in the graft to separate it from the original
 * domain elements and combine the original and new tree in a sequence.
 * If we have extended an extension node, then the children of this
 * sequence are spliced in the sequence of the extended extension
 * at the position where "node" appears in the original extension.
 * Otherwise, the sequence pair is attached to the new extension node.
 */
static __isl_give isl_schedule_node *graft_extension(
	__isl_take isl_schedule_node *node, __isl_take isl_schedule_node *graft,
	int before)
{
	isl_union_map *extension;
	isl_union_set *graft_domain;
	isl_union_set *node_domain;
	isl_schedule_tree *tree, *tree_graft;

	extension = isl_schedule_node_extension_get_extension(graft);
	graft_domain = isl_union_map_range(isl_union_map_copy(extension));
	node_domain = isl_schedule_node_get_universe_domain(node);
	node = insert_extension(node, extension);

	graft_domain = replace_by_universe_if_disjoint(graft_domain,
							node_domain);
	isl_union_set_free(node_domain);

	tree = isl_schedule_node_get_tree(node);
	if (!isl_schedule_node_has_children(graft)) {
		tree_graft = isl_schedule_tree_from_filter(graft_domain);
	} else {
		graft = isl_schedule_node_child(graft, 0);
		tree_graft = isl_schedule_node_get_tree(graft);
		tree_graft = isl_schedule_tree_insert_filter(tree_graft,
								graft_domain);
	}
	if (before)
		tree = isl_schedule_tree_sequence_pair(tree_graft, tree);
	else
		tree = isl_schedule_tree_sequence_pair(tree, tree_graft);
	node = graft_or_splice(node, tree, before);

	isl_schedule_node_free(graft);

	return node;
}

/* Replace the root domain node of "node" by an extension node suitable
 * for insertion at "pos".
 * That is, create an extension node that maps the outer band nodes
 * at "pos" to the domain of the root node of "node" and attach
 * the child of this root node to the extension node.
 */
static __isl_give isl_schedule_node *extension_from_domain(
	__isl_take isl_schedule_node *node, __isl_keep isl_schedule_node *pos)
{
	isl_union_set *universe;
	isl_union_set *domain;
	isl_union_map *ext;
	int depth;
	int anchored;
	isl_space *space;
	isl_schedule_node *res;
	isl_schedule_tree *tree;

	anchored = isl_schedule_node_is_subtree_anchored(node);
	if (anchored < 0)
		return isl_schedule_node_free(node);
	if (anchored)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_unsupported,
			"cannot graft anchored tree with domain root",
			return isl_schedule_node_free(node));

	depth = isl_schedule_node_get_schedule_depth(pos);
	domain = isl_schedule_node_domain_get_domain(node);
	space = isl_union_set_get_space(domain);
	space = isl_space_set_from_params(space);
	space = isl_space_add_dims(space, isl_dim_set, depth);
	universe = isl_union_set_from_set(isl_set_universe(space));
	ext = isl_union_map_from_domain_and_range(universe, domain);
	res = isl_schedule_node_from_extension(ext);
	node = isl_schedule_node_child(node, 0);
	if (!node)
		return isl_schedule_node_free(res);
	if (!isl_schedule_tree_is_leaf(node->tree)) {
		tree = isl_schedule_node_get_tree(node);
		res = isl_schedule_node_child(res, 0);
		res = isl_schedule_node_graft_tree(res, tree);
		res = isl_schedule_node_parent(res);
	}
	isl_schedule_node_free(node);

	return res;
}

/* Insert a node "graft" into the schedule tree of "node" such that it
 * is executed before (if "before" is set) or after (if "before" is not set)
 * the node that "node" points to.
 * The root of "graft" may be either a domain or an extension node.
 * In the latter case, the domain of the extension needs to correspond
 * to the outer band nodes of "node".
 * The elements of the domain or the range of the extension may not
 * intersect with the domain elements that reach "node".
 * The schedule tree of "graft" may not be anchored.
 *
 * The schedule tree of "node" is modified to include an extension node
 * corresponding to the root node of "graft" as a child of the original
 * parent of "node".  The original node that "node" points to and the
 * child of the root node of "graft" are attached to this extension node
 * through a sequence, with appropriate filters and with the child
 * of "graft" appearing before or after the original "node".
 *
 * If "node" already appears inside a sequence that is the child of
 * an extension node and if the spaces of the new domain elements
 * do not overlap with those of the original domain elements,
 * then that extension node is extended with the new extension
 * rather than introducing a new segment of extension and sequence nodes.
 *
 * Return a pointer to the same node in the modified tree that
 * "node" pointed to in the original tree.
 */
static __isl_give isl_schedule_node *isl_schedule_node_graft_before_or_after(
	__isl_take isl_schedule_node *node, __isl_take isl_schedule_node *graft,
	int before)
{
	if (!node || !graft)
		goto error;
	if (check_insert(node) < 0)
		goto error;

	if (isl_schedule_node_get_type(graft) == isl_schedule_node_domain)
		graft = extension_from_domain(graft, node);

	if (isl_schedule_node_get_type(graft) != isl_schedule_node_extension)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"expecting domain or extension as root of graft",
			goto error);

	return graft_extension(node, graft, before);
error:
	isl_schedule_node_free(node);
	isl_schedule_node_free(graft);
	return NULL;
}

/* Insert a node "graft" into the schedule tree of "node" such that it
 * is executed before the node that "node" points to.
 * The root of "graft" may be either a domain or an extension node.
 * In the latter case, the domain of the extension needs to correspond
 * to the outer band nodes of "node".
 * The elements of the domain or the range of the extension may not
 * intersect with the domain elements that reach "node".
 * The schedule tree of "graft" may not be anchored.
 *
 * Return a pointer to the same node in the modified tree that
 * "node" pointed to in the original tree.
 */
__isl_give isl_schedule_node *isl_schedule_node_graft_before(
	__isl_take isl_schedule_node *node, __isl_take isl_schedule_node *graft)
{
	return isl_schedule_node_graft_before_or_after(node, graft, 1);
}

/* Insert a node "graft" into the schedule tree of "node" such that it
 * is executed after the node that "node" points to.
 * The root of "graft" may be either a domain or an extension node.
 * In the latter case, the domain of the extension needs to correspond
 * to the outer band nodes of "node".
 * The elements of the domain or the range of the extension may not
 * intersect with the domain elements that reach "node".
 * The schedule tree of "graft" may not be anchored.
 *
 * Return a pointer to the same node in the modified tree that
 * "node" pointed to in the original tree.
 */
__isl_give isl_schedule_node *isl_schedule_node_graft_after(
	__isl_take isl_schedule_node *node,
	__isl_take isl_schedule_node *graft)
{
	return isl_schedule_node_graft_before_or_after(node, graft, 0);
}

/* Split the domain elements that reach "node" into those that satisfy
 * "filter" and those that do not.  Arrange for the first subset to be
 * executed before or after the second subset, depending on the value
 * of "before".
 * Return a pointer to the tree corresponding to the second subset,
 * except when this subset is empty in which case the original pointer
 * is returned.
 * If both subsets are non-empty, then a sequence node is introduced
 * to impose the order.  If the grandparent of the original node was
 * itself a sequence, then the original child is replaced by two children
 * in this sequence instead.
 * The children in the sequence are copies of the original subtree,
 * simplified with respect to their filters.
 */
static __isl_give isl_schedule_node *isl_schedule_node_order_before_or_after(
	__isl_take isl_schedule_node *node, __isl_take isl_union_set *filter,
	int before)
{
	enum isl_schedule_node_type ancestors[] =
		{ isl_schedule_node_filter, isl_schedule_node_sequence };
	isl_union_set *node_domain, *node_filter = NULL, *parent_filter;
	isl_schedule_node *node2;
	isl_schedule_tree *tree1, *tree2;
	int empty1, empty2;
	int in_seq;

	if (!node || !filter)
		goto error;
	if (check_insert(node) < 0)
		goto error;

	in_seq = has_ancestors(node, 2, ancestors);
	if (in_seq < 0)
		goto error;
	node_domain = isl_schedule_node_get_domain(node);
	filter = isl_union_set_gist(filter, isl_union_set_copy(node_domain));
	node_filter = isl_union_set_copy(node_domain);
	node_filter = isl_union_set_subtract(node_filter,
						isl_union_set_copy(filter));
	node_filter = isl_union_set_gist(node_filter, node_domain);
	empty1 = isl_union_set_is_empty(filter);
	empty2 = isl_union_set_is_empty(node_filter);
	if (empty1 < 0 || empty2 < 0)
		goto error;
	if (empty1 || empty2) {
		isl_union_set_free(filter);
		isl_union_set_free(node_filter);
		return node;
	}

	if (in_seq) {
		node = isl_schedule_node_parent(node);
		parent_filter = isl_schedule_node_filter_get_filter(node);
		node_filter = isl_union_set_intersect(node_filter,
					    isl_union_set_copy(parent_filter));
		filter = isl_union_set_intersect(filter, parent_filter);
	}

	node2 = isl_schedule_node_copy(node);
	node = isl_schedule_node_gist(node, isl_union_set_copy(node_filter));
	node2 = isl_schedule_node_gist(node2, isl_union_set_copy(filter));
	tree1 = isl_schedule_node_get_tree(node);
	tree2 = isl_schedule_node_get_tree(node2);
	tree1 = isl_schedule_tree_insert_filter(tree1, node_filter);
	tree2 = isl_schedule_tree_insert_filter(tree2, filter);
	isl_schedule_node_free(node2);

	if (before) {
		tree1 = isl_schedule_tree_sequence_pair(tree2, tree1);
		node = graft_or_splice(node, tree1, 1);
	} else {
		tree1 = isl_schedule_tree_sequence_pair(tree1, tree2);
		node = graft_or_splice(node, tree1, 0);
	}

	return node;
error:
	isl_schedule_node_free(node);
	isl_union_set_free(filter);
	isl_union_set_free(node_filter);
	return NULL;
}

/* Split the domain elements that reach "node" into those that satisfy
 * "filter" and those that do not.  Arrange for the first subset to be
 * executed before the second subset.
 * Return a pointer to the tree corresponding to the second subset,
 * except when this subset is empty in which case the original pointer
 * is returned.
 */
__isl_give isl_schedule_node *isl_schedule_node_order_before(
	__isl_take isl_schedule_node *node, __isl_take isl_union_set *filter)
{
	return isl_schedule_node_order_before_or_after(node, filter, 1);
}

/* Split the domain elements that reach "node" into those that satisfy
 * "filter" and those that do not.  Arrange for the first subset to be
 * executed after the second subset.
 * Return a pointer to the tree corresponding to the second subset,
 * except when this subset is empty in which case the original pointer
 * is returned.
 */
__isl_give isl_schedule_node *isl_schedule_node_order_after(
	__isl_take isl_schedule_node *node, __isl_take isl_union_set *filter)
{
	return isl_schedule_node_order_before_or_after(node, filter, 0);
}

/* Reset the user pointer on all identifiers of parameters and tuples
 * in the schedule node "node".
 */
__isl_give isl_schedule_node *isl_schedule_node_reset_user(
	__isl_take isl_schedule_node *node)
{
	isl_schedule_tree *tree;

	tree = isl_schedule_node_get_tree(node);
	tree = isl_schedule_tree_reset_user(tree);
	node = isl_schedule_node_graft_tree(node, tree);

	return node;
}

/* Align the parameters of the schedule node "node" to those of "space".
 */
__isl_give isl_schedule_node *isl_schedule_node_align_params(
	__isl_take isl_schedule_node *node, __isl_take isl_space *space)
{
	isl_schedule_tree *tree;

	tree = isl_schedule_node_get_tree(node);
	tree = isl_schedule_tree_align_params(tree, space);
	node = isl_schedule_node_graft_tree(node, tree);

	return node;
}

/* Compute the pullback of schedule node "node"
 * by the function represented by "upma".
 * In other words, plug in "upma" in the iteration domains
 * of schedule node "node".
 * We currently do not handle expansion nodes.
 *
 * Note that this is only a helper function for
 * isl_schedule_pullback_union_pw_multi_aff.  In order to maintain consistency,
 * this function should not be called on a single node without also
 * calling it on all the other nodes.
 */
__isl_give isl_schedule_node *isl_schedule_node_pullback_union_pw_multi_aff(
	__isl_take isl_schedule_node *node,
	__isl_take isl_union_pw_multi_aff *upma)
{
	isl_schedule_tree *tree;

	tree = isl_schedule_node_get_tree(node);
	tree = isl_schedule_tree_pullback_union_pw_multi_aff(tree, upma);
	node = isl_schedule_node_graft_tree(node, tree);

	return node;
}

/* Internal data structure for isl_schedule_node_expand.
 * "tree" is the tree that needs to be plugged in in all the leaves.
 * "domain" is the set of domain elements in the original leaves
 * to which the tree applies.
 */
struct isl_schedule_expand_data {
	isl_schedule_tree *tree;
	isl_union_set *domain;
};

/* If "node" is a leaf, then plug in data->tree, simplifying it
 * within its new context.
 *
 * If there are any domain elements at the leaf where the tree
 * should not be plugged in (i.e., there are elements not in data->domain)
 * then first extend the tree to only apply to the elements in data->domain
 * by constructing a set node that selects data->tree for elements
 * in data->domain and a leaf for the other elements.
 */
static __isl_give isl_schedule_node *expand(__isl_take isl_schedule_node *node,
	void *user)
{
	struct isl_schedule_expand_data *data = user;
	isl_schedule_tree *tree, *leaf;
	isl_union_set *domain, *left;
	isl_bool empty;

	if (isl_schedule_node_get_type(node) != isl_schedule_node_leaf)
		return node;

	domain = isl_schedule_node_get_domain(node);
	tree = isl_schedule_tree_copy(data->tree);

	left = isl_union_set_copy(domain);
	left = isl_union_set_subtract(left, isl_union_set_copy(data->domain));
	empty = isl_union_set_is_empty(left);
	if (empty >= 0 && !empty) {
		leaf = isl_schedule_node_get_leaf(node);
		leaf = isl_schedule_tree_insert_filter(leaf, left);
		left = isl_union_set_copy(data->domain);
		tree = isl_schedule_tree_insert_filter(tree, left);
		tree = isl_schedule_tree_set_pair(tree, leaf);
	} else {
		if (empty < 0)
			node = isl_schedule_node_free(node);
		isl_union_set_free(left);
	}

	node = isl_schedule_node_graft_tree(node, tree);
	node = isl_schedule_node_gist(node, domain);

	return node;
}

/* Expand the tree rooted at "node" by extending all leaves
 * with an expansion node with as child "tree".
 * The expansion is determined by "contraction" and "domain".
 * That is, the elements of "domain" are contracted according
 * to "contraction".  The expansion relation is then the inverse
 * of "contraction" with its range intersected with "domain".
 *
 * Insert the appropriate expansion node on top of "tree" and
 * then plug in the result in all leaves of "node".
 */
__isl_give isl_schedule_node *isl_schedule_node_expand(
	__isl_take isl_schedule_node *node,
	__isl_take isl_union_pw_multi_aff *contraction,
	__isl_take isl_union_set *domain,
	__isl_take isl_schedule_tree *tree)
{
	struct isl_schedule_expand_data data;
	isl_union_map *expansion;
	isl_union_pw_multi_aff *copy;

	if (!node || !contraction || !tree)
		node = isl_schedule_node_free(node);

	copy = isl_union_pw_multi_aff_copy(contraction);
	expansion = isl_union_map_from_union_pw_multi_aff(copy);
	expansion = isl_union_map_reverse(expansion);
	expansion = isl_union_map_intersect_range(expansion, domain);
	data.domain = isl_union_map_domain(isl_union_map_copy(expansion));

	tree = isl_schedule_tree_insert_expansion(tree, contraction, expansion);
	data.tree = tree;

	node = isl_schedule_node_map_descendant_bottom_up(node, &expand, &data);
	isl_union_set_free(data.domain);
	isl_schedule_tree_free(data.tree);
	return node;
}

/* Return the position of the subtree containing "node" among the children
 * of "ancestor".  "node" is assumed to be a descendant of "ancestor".
 * In particular, both nodes should point to the same schedule tree.
 *
 * Return -1 on error.
 */
int isl_schedule_node_get_ancestor_child_position(
	__isl_keep isl_schedule_node *node,
	__isl_keep isl_schedule_node *ancestor)
{
	int n1, n2;
	isl_schedule_tree *tree;

	if (!node || !ancestor)
		return -1;

	if (node->schedule != ancestor->schedule)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"not a descendant", return -1);

	n1 = isl_schedule_node_get_tree_depth(ancestor);
	n2 = isl_schedule_node_get_tree_depth(node);

	if (n1 >= n2)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"not a descendant", return -1);
	tree = isl_schedule_tree_list_get_schedule_tree(node->ancestors, n1);
	isl_schedule_tree_free(tree);
	if (tree != ancestor->tree)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_invalid,
			"not a descendant", return -1);

	return node->child_pos[n1];
}

/* Given two nodes that point to the same schedule tree, return their
 * closest shared ancestor.
 *
 * Since the two nodes point to the same schedule, they share at least
 * one ancestor, the root of the schedule.  We move down from the root
 * to the first ancestor where the respective children have a different
 * child position.  This is the requested ancestor.
 * If there is no ancestor where the children have a different position,
 * then one node is an ancestor of the other and then this node is
 * the requested ancestor.
 */
__isl_give isl_schedule_node *isl_schedule_node_get_shared_ancestor(
	__isl_keep isl_schedule_node *node1,
	__isl_keep isl_schedule_node *node2)
{
	int i, n1, n2;

	if (!node1 || !node2)
		return NULL;
	if (node1->schedule != node2->schedule)
		isl_die(isl_schedule_node_get_ctx(node1), isl_error_invalid,
			"not part of same schedule", return NULL);
	n1 = isl_schedule_node_get_tree_depth(node1);
	n2 = isl_schedule_node_get_tree_depth(node2);
	if (n2 < n1)
		return isl_schedule_node_get_shared_ancestor(node2, node1);
	if (n1 == 0)
		return isl_schedule_node_copy(node1);
	if (isl_schedule_node_is_equal(node1, node2))
		return isl_schedule_node_copy(node1);

	for (i = 0; i < n1; ++i)
		if (node1->child_pos[i] != node2->child_pos[i])
			break;

	node1 = isl_schedule_node_copy(node1);
	return isl_schedule_node_ancestor(node1, n1 - i);
}

/* Print "node" to "p".
 */
__isl_give isl_printer *isl_printer_print_schedule_node(
	__isl_take isl_printer *p, __isl_keep isl_schedule_node *node)
{
	if (!node)
		return isl_printer_free(p);
	return isl_printer_print_schedule_tree_mark(p, node->schedule->root,
			isl_schedule_tree_list_n_schedule_tree(node->ancestors),
			node->child_pos);
}

void isl_schedule_node_dump(__isl_keep isl_schedule_node *node)
{
	isl_ctx *ctx;
	isl_printer *printer;

	if (!node)
		return;

	ctx = isl_schedule_node_get_ctx(node);
	printer = isl_printer_to_file(ctx, stderr);
	printer = isl_printer_set_yaml_style(printer, ISL_YAML_STYLE_BLOCK);
	printer = isl_printer_print_schedule_node(printer, node);

	isl_printer_free(printer);
}

/* Return a string representation of "node".
 * Print the schedule node in block format as it would otherwise
 * look identical to the entire schedule.
 */
__isl_give char *isl_schedule_node_to_str(__isl_keep isl_schedule_node *node)
{
	isl_printer *printer;
	char *s;

	if (!node)
		return NULL;

	printer = isl_printer_to_str(isl_schedule_node_get_ctx(node));
	printer = isl_printer_set_yaml_style(printer, ISL_YAML_STYLE_BLOCK);
	printer = isl_printer_print_schedule_node(printer, node);
	s = isl_printer_get_str(printer);
	isl_printer_free(printer);

	return s;
}
