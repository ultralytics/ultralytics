/*
 * Copyright 2013-2014 Ecole Normale Superieure
 * Copyright 2014      INRIA Rocquencourt
 * Copyright 2016      INRIA Paris
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 * and Inria Paris - Rocquencourt, Domaine de Voluceau - Rocquencourt,
 * B.P. 105 - 78153 Le Chesnay, France
 * and Centre de Recherche Inria de Paris, 2 rue Simone Iff - Voie DQ12,
 * CS 42112, 75589 Paris Cedex 12, France
 */

#include <isl/map.h>
#include <isl_schedule_band.h>
#include <isl_schedule_private.h>

#undef EL
#define EL isl_schedule_tree

#include <isl_list_templ.h>

#undef BASE
#define BASE schedule_tree

#include <isl_list_templ.c>

/* Is "tree" the leaf of a schedule tree?
 */
int isl_schedule_tree_is_leaf(__isl_keep isl_schedule_tree *tree)
{
	return isl_schedule_tree_get_type(tree) == isl_schedule_node_leaf;
}

/* Create a new schedule tree of type "type".
 * The caller is responsible for filling in the type specific fields and
 * the children.
 *
 * By default, the single node tree does not have any anchored nodes.
 * The caller is responsible for updating the anchored field if needed.
 */
static __isl_give isl_schedule_tree *isl_schedule_tree_alloc(isl_ctx *ctx,
	enum isl_schedule_node_type type)
{
	isl_schedule_tree *tree;

	if (type == isl_schedule_node_error)
		return NULL;

	tree = isl_calloc_type(ctx, isl_schedule_tree);
	if (!tree)
		return NULL;

	tree->ref = 1;
	tree->ctx = ctx;
	isl_ctx_ref(ctx);
	tree->type = type;
	tree->anchored = 0;

	return tree;
}

/* Return a fresh copy of "tree".
 */
__isl_take isl_schedule_tree *isl_schedule_tree_dup(
	__isl_keep isl_schedule_tree *tree)
{
	isl_ctx *ctx;
	isl_schedule_tree *dup;

	if (!tree)
		return NULL;

	ctx = isl_schedule_tree_get_ctx(tree);
	dup = isl_schedule_tree_alloc(ctx, tree->type);
	if (!dup)
		return NULL;

	switch (tree->type) {
	case isl_schedule_node_error:
		isl_die(ctx, isl_error_internal,
			"allocation should have failed",
			isl_schedule_tree_free(dup));
	case isl_schedule_node_band:
		dup->band = isl_schedule_band_copy(tree->band);
		if (!dup->band)
			return isl_schedule_tree_free(dup);
		break;
	case isl_schedule_node_context:
		dup->context = isl_set_copy(tree->context);
		if (!dup->context)
			return isl_schedule_tree_free(dup);
		break;
	case isl_schedule_node_domain:
		dup->domain = isl_union_set_copy(tree->domain);
		if (!dup->domain)
			return isl_schedule_tree_free(dup);
		break;
	case isl_schedule_node_expansion:
		dup->contraction =
			isl_union_pw_multi_aff_copy(tree->contraction);
		dup->expansion = isl_union_map_copy(tree->expansion);
		if (!dup->contraction || !dup->expansion)
			return isl_schedule_tree_free(dup);
		break;
	case isl_schedule_node_extension:
		dup->extension = isl_union_map_copy(tree->extension);
		if (!dup->extension)
			return isl_schedule_tree_free(dup);
		break;
	case isl_schedule_node_filter:
		dup->filter = isl_union_set_copy(tree->filter);
		if (!dup->filter)
			return isl_schedule_tree_free(dup);
		break;
	case isl_schedule_node_guard:
		dup->guard = isl_set_copy(tree->guard);
		if (!dup->guard)
			return isl_schedule_tree_free(dup);
		break;
	case isl_schedule_node_mark:
		dup->mark = isl_id_copy(tree->mark);
		if (!dup->mark)
			return isl_schedule_tree_free(dup);
		break;
	case isl_schedule_node_leaf:
	case isl_schedule_node_sequence:
	case isl_schedule_node_set:
		break;
	}

	if (tree->children) {
		dup->children = isl_schedule_tree_list_copy(tree->children);
		if (!dup->children)
			return isl_schedule_tree_free(dup);
	}
	dup->anchored = tree->anchored;

	return dup;
}

/* Return an isl_schedule_tree that is equal to "tree" and that has only
 * a single reference.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_cow(
	__isl_take isl_schedule_tree *tree)
{
	if (!tree)
		return NULL;

	if (tree->ref == 1)
		return tree;
	tree->ref--;
	return isl_schedule_tree_dup(tree);
}

/* Return a new reference to "tree".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_copy(
	__isl_keep isl_schedule_tree *tree)
{
	if (!tree)
		return NULL;

	tree->ref++;
	return tree;
}

/* Free "tree" and return NULL.
 */
__isl_null isl_schedule_tree *isl_schedule_tree_free(
	__isl_take isl_schedule_tree *tree)
{
	if (!tree)
		return NULL;
	if (--tree->ref > 0)
		return NULL;

	switch (tree->type) {
	case isl_schedule_node_band:
		isl_schedule_band_free(tree->band);
		break;
	case isl_schedule_node_context:
		isl_set_free(tree->context);
		break;
	case isl_schedule_node_domain:
		isl_union_set_free(tree->domain);
		break;
	case isl_schedule_node_expansion:
		isl_union_pw_multi_aff_free(tree->contraction);
		isl_union_map_free(tree->expansion);
		break;
	case isl_schedule_node_extension:
		isl_union_map_free(tree->extension);
		break;
	case isl_schedule_node_filter:
		isl_union_set_free(tree->filter);
		break;
	case isl_schedule_node_guard:
		isl_set_free(tree->guard);
		break;
	case isl_schedule_node_mark:
		isl_id_free(tree->mark);
		break;
	case isl_schedule_node_sequence:
	case isl_schedule_node_set:
	case isl_schedule_node_error:
	case isl_schedule_node_leaf:
		break;
	}
	isl_schedule_tree_list_free(tree->children);
	isl_ctx_deref(tree->ctx);
	free(tree);

	return NULL;
}

/* Create and return a new leaf schedule tree.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_leaf(isl_ctx *ctx)
{
	return isl_schedule_tree_alloc(ctx, isl_schedule_node_leaf);
}

/* Create a new band schedule tree referring to "band"
 * with no children.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_from_band(
	__isl_take isl_schedule_band *band)
{
	isl_ctx *ctx;
	isl_schedule_tree *tree;

	if (!band)
		return NULL;

	ctx = isl_schedule_band_get_ctx(band);
	tree = isl_schedule_tree_alloc(ctx, isl_schedule_node_band);
	if (!tree)
		goto error;

	tree->band = band;
	tree->anchored = isl_schedule_band_is_anchored(band);

	return tree;
error:
	isl_schedule_band_free(band);
	return NULL;
}

/* Create a new context schedule tree with the given context and no children.
 * Since the context references the outer schedule dimension,
 * the tree is anchored.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_from_context(
	__isl_take isl_set *context)
{
	isl_ctx *ctx;
	isl_schedule_tree *tree;

	if (!context)
		return NULL;

	ctx = isl_set_get_ctx(context);
	tree = isl_schedule_tree_alloc(ctx, isl_schedule_node_context);
	if (!tree)
		goto error;

	tree->context = context;
	tree->anchored = 1;

	return tree;
error:
	isl_set_free(context);
	return NULL;
}

/* Create a new domain schedule tree with the given domain and no children.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_from_domain(
	__isl_take isl_union_set *domain)
{
	isl_ctx *ctx;
	isl_schedule_tree *tree;

	if (!domain)
		return NULL;

	ctx = isl_union_set_get_ctx(domain);
	tree = isl_schedule_tree_alloc(ctx, isl_schedule_node_domain);
	if (!tree)
		goto error;

	tree->domain = domain;

	return tree;
error:
	isl_union_set_free(domain);
	return NULL;
}

/* Create a new expansion schedule tree with the given contraction and
 * expansion and no children.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_from_expansion(
	__isl_take isl_union_pw_multi_aff *contraction,
	__isl_take isl_union_map *expansion)
{
	isl_ctx *ctx;
	isl_schedule_tree *tree;

	if (!contraction || !expansion)
		goto error;

	ctx = isl_union_map_get_ctx(expansion);
	tree = isl_schedule_tree_alloc(ctx, isl_schedule_node_expansion);
	if (!tree)
		goto error;

	tree->contraction = contraction;
	tree->expansion = expansion;

	return tree;
error:
	isl_union_pw_multi_aff_free(contraction);
	isl_union_map_free(expansion);
	return NULL;
}

/* Create a new extension schedule tree with the given extension and
 * no children.
 * Since the domain of the extension refers to the outer schedule dimension,
 * the tree is anchored.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_from_extension(
	__isl_take isl_union_map *extension)
{
	isl_ctx *ctx;
	isl_schedule_tree *tree;

	if (!extension)
		return NULL;

	ctx = isl_union_map_get_ctx(extension);
	tree = isl_schedule_tree_alloc(ctx, isl_schedule_node_extension);
	if (!tree)
		goto error;

	tree->extension = extension;
	tree->anchored = 1;

	return tree;
error:
	isl_union_map_free(extension);
	return NULL;
}

/* Create a new filter schedule tree with the given filter and no children.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_from_filter(
	__isl_take isl_union_set *filter)
{
	isl_ctx *ctx;
	isl_schedule_tree *tree;

	if (!filter)
		return NULL;

	ctx = isl_union_set_get_ctx(filter);
	tree = isl_schedule_tree_alloc(ctx, isl_schedule_node_filter);
	if (!tree)
		goto error;

	tree->filter = filter;

	return tree;
error:
	isl_union_set_free(filter);
	return NULL;
}

/* Create a new guard schedule tree with the given guard and no children.
 * Since the guard references the outer schedule dimension,
 * the tree is anchored.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_from_guard(
	__isl_take isl_set *guard)
{
	isl_ctx *ctx;
	isl_schedule_tree *tree;

	if (!guard)
		return NULL;

	ctx = isl_set_get_ctx(guard);
	tree = isl_schedule_tree_alloc(ctx, isl_schedule_node_guard);
	if (!tree)
		goto error;

	tree->guard = guard;
	tree->anchored = 1;

	return tree;
error:
	isl_set_free(guard);
	return NULL;
}

/* Create a new mark schedule tree with the given mark identifier and
 * no children.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_from_mark(
	__isl_take isl_id *mark)
{
	isl_ctx *ctx;
	isl_schedule_tree *tree;

	if (!mark)
		return NULL;

	ctx = isl_id_get_ctx(mark);
	tree = isl_schedule_tree_alloc(ctx, isl_schedule_node_mark);
	if (!tree)
		goto error;

	tree->mark = mark;

	return tree;
error:
	isl_id_free(mark);
	return NULL;
}

/* Does "tree" have any node that depends on its position
 * in the complete schedule tree?
 */
isl_bool isl_schedule_tree_is_subtree_anchored(
	__isl_keep isl_schedule_tree *tree)
{
	return tree ? tree->anchored : isl_bool_error;
}

/* Does the root node of "tree" depend on its position in the complete
 * schedule tree?
 * Band nodes may be anchored depending on the associated AST build options.
 * Context, extension and guard nodes are always anchored.
 */
int isl_schedule_tree_is_anchored(__isl_keep isl_schedule_tree *tree)
{
	if (!tree)
		return -1;

	switch (isl_schedule_tree_get_type(tree)) {
	case isl_schedule_node_error:
		return -1;
	case isl_schedule_node_band:
		return isl_schedule_band_is_anchored(tree->band);
	case isl_schedule_node_context:
	case isl_schedule_node_extension:
	case isl_schedule_node_guard:
		return 1;
	case isl_schedule_node_domain:
	case isl_schedule_node_expansion:
	case isl_schedule_node_filter:
	case isl_schedule_node_leaf:
	case isl_schedule_node_mark:
	case isl_schedule_node_sequence:
	case isl_schedule_node_set:
		return 0;
	}

	isl_die(isl_schedule_tree_get_ctx(tree), isl_error_internal,
		"unhandled case", return -1);
}

/* Update the anchored field of "tree" based on whether the root node
 * itself in anchored and the anchored fields of the children.
 *
 * This function should be called whenever the children of a tree node
 * are changed or the anchoredness of the tree root itself changes.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_update_anchored(
	__isl_take isl_schedule_tree *tree)
{
	int i, n;
	int anchored;

	if (!tree)
		return NULL;

	anchored = isl_schedule_tree_is_anchored(tree);
	if (anchored < 0)
		return isl_schedule_tree_free(tree);

	n = isl_schedule_tree_list_n_schedule_tree(tree->children);
	for (i = 0; !anchored && i < n; ++i) {
		isl_schedule_tree *child;

		child = isl_schedule_tree_get_child(tree, i);
		if (!child)
			return isl_schedule_tree_free(tree);
		anchored = child->anchored;
		isl_schedule_tree_free(child);
	}

	if (anchored == tree->anchored)
		return tree;
	tree = isl_schedule_tree_cow(tree);
	if (!tree)
		return NULL;
	tree->anchored = anchored;
	return tree;
}

/* Create a new tree of the given type (isl_schedule_node_sequence or
 * isl_schedule_node_set) with the given children.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_from_children(
	enum isl_schedule_node_type type,
	__isl_take isl_schedule_tree_list *list)
{
	isl_ctx *ctx;
	isl_schedule_tree *tree;

	if (!list)
		return NULL;

	ctx = isl_schedule_tree_list_get_ctx(list);
	tree = isl_schedule_tree_alloc(ctx, type);
	if (!tree)
		goto error;

	tree->children = list;
	tree = isl_schedule_tree_update_anchored(tree);

	return tree;
error:
	isl_schedule_tree_list_free(list);
	return NULL;
}

/* Construct a tree with a root node of type "type" and as children
 * "tree1" and "tree2".
 * If the root of one (or both) of the input trees is itself of type "type",
 * then the tree is replaced by its children.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_from_pair(
	enum isl_schedule_node_type type, __isl_take isl_schedule_tree *tree1,
	__isl_take isl_schedule_tree *tree2)
{
	isl_ctx *ctx;
	isl_schedule_tree_list *list;

	if (!tree1 || !tree2)
		goto error;

	ctx = isl_schedule_tree_get_ctx(tree1);
	if (isl_schedule_tree_get_type(tree1) == type) {
		list = isl_schedule_tree_list_copy(tree1->children);
		isl_schedule_tree_free(tree1);
	} else {
		list = isl_schedule_tree_list_alloc(ctx, 2);
		list = isl_schedule_tree_list_add(list, tree1);
	}
	if (isl_schedule_tree_get_type(tree2) == type) {
		isl_schedule_tree_list *children;

		children = isl_schedule_tree_list_copy(tree2->children);
		list = isl_schedule_tree_list_concat(list, children);
		isl_schedule_tree_free(tree2);
	} else {
		list = isl_schedule_tree_list_add(list, tree2);
	}

	return isl_schedule_tree_from_children(type, list);
error:
	isl_schedule_tree_free(tree1);
	isl_schedule_tree_free(tree2);
	return NULL;
}

/* Construct a tree with a sequence root node and as children
 * "tree1" and "tree2".
 * If the root of one (or both) of the input trees is itself a sequence,
 * then the tree is replaced by its children.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_sequence_pair(
	__isl_take isl_schedule_tree *tree1,
	__isl_take isl_schedule_tree *tree2)
{
	return isl_schedule_tree_from_pair(isl_schedule_node_sequence,
						tree1, tree2);
}

/* Construct a tree with a set root node and as children
 * "tree1" and "tree2".
 * If the root of one (or both) of the input trees is itself a set,
 * then the tree is replaced by its children.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_set_pair(
	__isl_take isl_schedule_tree *tree1,
	__isl_take isl_schedule_tree *tree2)
{
	return isl_schedule_tree_from_pair(isl_schedule_node_set, tree1, tree2);
}

/* Return the isl_ctx to which "tree" belongs.
 */
isl_ctx *isl_schedule_tree_get_ctx(__isl_keep isl_schedule_tree *tree)
{
	return tree ? tree->ctx : NULL;
}

/* Return the type of the root of the tree or isl_schedule_node_error
 * on error.
 */
enum isl_schedule_node_type isl_schedule_tree_get_type(
	__isl_keep isl_schedule_tree *tree)
{
	return tree ? tree->type : isl_schedule_node_error;
}

/* Are "tree1" and "tree2" obviously equal to each other?
 */
isl_bool isl_schedule_tree_plain_is_equal(__isl_keep isl_schedule_tree *tree1,
	__isl_keep isl_schedule_tree *tree2)
{
	isl_bool equal;
	int i, n;

	if (!tree1 || !tree2)
		return isl_bool_error;
	if (tree1 == tree2)
		return isl_bool_true;
	if (tree1->type != tree2->type)
		return isl_bool_false;

	switch (tree1->type) {
	case isl_schedule_node_band:
		equal = isl_schedule_band_plain_is_equal(tree1->band,
							tree2->band);
		break;
	case isl_schedule_node_context:
		equal = isl_set_is_equal(tree1->context, tree2->context);
		break;
	case isl_schedule_node_domain:
		equal = isl_union_set_is_equal(tree1->domain, tree2->domain);
		break;
	case isl_schedule_node_expansion:
		equal = isl_union_map_is_equal(tree1->expansion,
						tree2->expansion);
		if (equal >= 0 && equal)
			equal = isl_union_pw_multi_aff_plain_is_equal(
				    tree1->contraction, tree2->contraction);
		break;
	case isl_schedule_node_extension:
		equal = isl_union_map_is_equal(tree1->extension,
						tree2->extension);
		break;
	case isl_schedule_node_filter:
		equal = isl_union_set_is_equal(tree1->filter, tree2->filter);
		break;
	case isl_schedule_node_guard:
		equal = isl_set_is_equal(tree1->guard, tree2->guard);
		break;
	case isl_schedule_node_mark:
		equal = tree1->mark == tree2->mark;
		break;
	case isl_schedule_node_leaf:
	case isl_schedule_node_sequence:
	case isl_schedule_node_set:
		equal = isl_bool_true;
		break;
	case isl_schedule_node_error:
		equal = isl_bool_error;
		break;
	}

	if (equal < 0 || !equal)
		return equal;

	n = isl_schedule_tree_n_children(tree1);
	if (n != isl_schedule_tree_n_children(tree2))
		return isl_bool_false;
	for (i = 0; i < n; ++i) {
		isl_schedule_tree *child1, *child2;

		child1 = isl_schedule_tree_get_child(tree1, i);
		child2 = isl_schedule_tree_get_child(tree2, i);
		equal = isl_schedule_tree_plain_is_equal(child1, child2);
		isl_schedule_tree_free(child1);
		isl_schedule_tree_free(child2);

		if (equal < 0 || !equal)
			return equal;
	}

	return isl_bool_true;
}

/* Does "tree" have any children, other than an implicit leaf.
 */
int isl_schedule_tree_has_children(__isl_keep isl_schedule_tree *tree)
{
	if (!tree)
		return -1;

	return tree->children != NULL;
}

/* Return the number of children of "tree", excluding implicit leaves.
 */
int isl_schedule_tree_n_children(__isl_keep isl_schedule_tree *tree)
{
	if (!tree)
		return -1;

	return isl_schedule_tree_list_n_schedule_tree(tree->children);
}

/* Return a copy of the (explicit) child at position "pos" of "tree".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_get_child(
	__isl_keep isl_schedule_tree *tree, int pos)
{
	if (!tree)
		return NULL;
	if (!tree->children)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_internal,
			"schedule tree has no explicit children", return NULL);
	return isl_schedule_tree_list_get_schedule_tree(tree->children, pos);
}

/* Return a copy of the (explicit) child at position "pos" of "tree" and
 * free "tree".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_child(
	__isl_take isl_schedule_tree *tree, int pos)
{
	isl_schedule_tree *child;

	child = isl_schedule_tree_get_child(tree, pos);
	isl_schedule_tree_free(tree);
	return child;
}

/* Remove all (explicit) children from "tree".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_reset_children(
	__isl_take isl_schedule_tree *tree)
{
	tree = isl_schedule_tree_cow(tree);
	if (!tree)
		return NULL;
	tree->children = isl_schedule_tree_list_free(tree->children);
	return tree;
}

/* Remove the child at position "pos" from the children of "tree".
 * If there was only one child to begin with, then remove all children.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_drop_child(
	__isl_take isl_schedule_tree *tree, int pos)
{
	int n;

	tree = isl_schedule_tree_cow(tree);
	if (!tree)
		return NULL;

	if (!isl_schedule_tree_has_children(tree))
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"tree does not have any explicit children",
			return isl_schedule_tree_free(tree));
	n = isl_schedule_tree_list_n_schedule_tree(tree->children);
	if (pos < 0 || pos >= n)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"position out of bounds",
			return isl_schedule_tree_free(tree));
	if (n == 1)
		return isl_schedule_tree_reset_children(tree);

	tree->children = isl_schedule_tree_list_drop(tree->children, pos, 1);
	if (!tree->children)
		return isl_schedule_tree_free(tree);

	return tree;
}

/* Replace the child at position "pos" of "tree" by "child".
 *
 * If the new child is a leaf, then it is not explicitly
 * recorded in the list of children.  Instead, the list of children
 * (which is assumed to have only one element) is removed.
 * Note that the children of set and sequence nodes are always
 * filters, so they cannot be replaced by empty trees.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_replace_child(
	__isl_take isl_schedule_tree *tree, int pos,
	__isl_take isl_schedule_tree *child)
{
	tree = isl_schedule_tree_cow(tree);
	if (!tree || !child)
		goto error;

	if (isl_schedule_tree_is_leaf(child)) {
		isl_schedule_tree_free(child);
		if (!tree->children && pos == 0)
			return tree;
		if (isl_schedule_tree_n_children(tree) != 1)
			isl_die(isl_schedule_tree_get_ctx(tree),
				isl_error_internal,
				"can only replace single child by leaf",
				goto error);
		return isl_schedule_tree_reset_children(tree);
	}

	if (!tree->children && pos == 0)
		tree->children =
			isl_schedule_tree_list_from_schedule_tree(child);
	else
		tree->children = isl_schedule_tree_list_set_schedule_tree(
				tree->children, pos, child);

	if (!tree->children)
		return isl_schedule_tree_free(tree);
	tree = isl_schedule_tree_update_anchored(tree);

	return tree;
error:
	isl_schedule_tree_free(tree);
	isl_schedule_tree_free(child);
	return NULL;
}

/* Replace the (explicit) children of "tree" by "children"?
 */
__isl_give isl_schedule_tree *isl_schedule_tree_set_children(
	__isl_take isl_schedule_tree *tree,
	__isl_take isl_schedule_tree_list *children)
{
	tree = isl_schedule_tree_cow(tree);
	if (!tree || !children)
		goto error;
	isl_schedule_tree_list_free(tree->children);
	tree->children = children;
	return tree;
error:
	isl_schedule_tree_free(tree);
	isl_schedule_tree_list_free(children);
	return NULL;
}

/* Create a new band schedule tree referring to "band"
 * with "tree" as single child.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_insert_band(
	__isl_take isl_schedule_tree *tree, __isl_take isl_schedule_band *band)
{
	isl_schedule_tree *res;

	res = isl_schedule_tree_from_band(band);
	return isl_schedule_tree_replace_child(res, 0, tree);
}

/* Create a new context schedule tree with the given context and
 * with "tree" as single child.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_insert_context(
	__isl_take isl_schedule_tree *tree, __isl_take isl_set *context)
{
	isl_schedule_tree *res;

	res = isl_schedule_tree_from_context(context);
	return isl_schedule_tree_replace_child(res, 0, tree);
}

/* Create a new domain schedule tree with the given domain and
 * with "tree" as single child.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_insert_domain(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *domain)
{
	isl_schedule_tree *res;

	res = isl_schedule_tree_from_domain(domain);
	return isl_schedule_tree_replace_child(res, 0, tree);
}

/* Create a new expansion schedule tree with the given contraction and
 * expansion and with "tree" as single child.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_insert_expansion(
	__isl_take isl_schedule_tree *tree,
	__isl_take isl_union_pw_multi_aff *contraction,
	__isl_take isl_union_map *expansion)
{
	isl_schedule_tree *res;

	res = isl_schedule_tree_from_expansion(contraction, expansion);
	return isl_schedule_tree_replace_child(res, 0, tree);
}

/* Create a new extension schedule tree with the given extension and
 * with "tree" as single child.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_insert_extension(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_map *extension)
{
	isl_schedule_tree *res;

	res = isl_schedule_tree_from_extension(extension);
	return isl_schedule_tree_replace_child(res, 0, tree);
}

/* Create a new filter schedule tree with the given filter and single child.
 *
 * If the root of "tree" is itself a filter node, then the two
 * filter nodes are merged into one node.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_insert_filter(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *filter)
{
	isl_schedule_tree *res;

	if (isl_schedule_tree_get_type(tree) == isl_schedule_node_filter) {
		isl_union_set *tree_filter;

		tree_filter = isl_schedule_tree_filter_get_filter(tree);
		tree_filter = isl_union_set_intersect(tree_filter, filter);
		tree = isl_schedule_tree_filter_set_filter(tree, tree_filter);
		return tree;
	}

	res = isl_schedule_tree_from_filter(filter);
	return isl_schedule_tree_replace_child(res, 0, tree);
}

/* Insert a filter node with filter set "filter"
 * in each of the children of "tree".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_children_insert_filter(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *filter)
{
	int i, n;

	if (!tree || !filter)
		goto error;

	n = isl_schedule_tree_n_children(tree);
	for (i = 0; i < n; ++i) {
		isl_schedule_tree *child;

		child = isl_schedule_tree_get_child(tree, i);
		child = isl_schedule_tree_insert_filter(child,
						    isl_union_set_copy(filter));
		tree = isl_schedule_tree_replace_child(tree, i, child);
	}

	isl_union_set_free(filter);
	return tree;
error:
	isl_union_set_free(filter);
	isl_schedule_tree_free(tree);
	return NULL;
}

/* Create a new guard schedule tree with the given guard and
 * with "tree" as single child.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_insert_guard(
	__isl_take isl_schedule_tree *tree, __isl_take isl_set *guard)
{
	isl_schedule_tree *res;

	res = isl_schedule_tree_from_guard(guard);
	return isl_schedule_tree_replace_child(res, 0, tree);
}

/* Create a new mark schedule tree with the given mark identifier and
 * single child.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_insert_mark(
	__isl_take isl_schedule_tree *tree, __isl_take isl_id *mark)
{
	isl_schedule_tree *res;

	res = isl_schedule_tree_from_mark(mark);
	return isl_schedule_tree_replace_child(res, 0, tree);
}

/* Return the number of members in the band tree root.
 */
unsigned isl_schedule_tree_band_n_member(__isl_keep isl_schedule_tree *tree)
{
	if (!tree)
		return 0;

	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", return 0);

	return isl_schedule_band_n_member(tree->band);
}

/* Is the band member at position "pos" of the band tree root
 * marked coincident?
 */
isl_bool isl_schedule_tree_band_member_get_coincident(
	__isl_keep isl_schedule_tree *tree, int pos)
{
	if (!tree)
		return isl_bool_error;

	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", return isl_bool_error);

	return isl_schedule_band_member_get_coincident(tree->band, pos);
}

/* Mark the given band member as being coincident or not
 * according to "coincident".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_band_member_set_coincident(
	__isl_take isl_schedule_tree *tree, int pos, int coincident)
{
	if (!tree)
		return NULL;
	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", return isl_schedule_tree_free(tree));
	if (isl_schedule_tree_band_member_get_coincident(tree, pos) ==
								    coincident)
		return tree;
	tree = isl_schedule_tree_cow(tree);
	if (!tree)
		return NULL;

	tree->band = isl_schedule_band_member_set_coincident(tree->band, pos,
							coincident);
	if (!tree->band)
		return isl_schedule_tree_free(tree);
	return tree;
}

/* Is the band tree root marked permutable?
 */
isl_bool isl_schedule_tree_band_get_permutable(
	__isl_keep isl_schedule_tree *tree)
{
	if (!tree)
		return isl_bool_error;

	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", return isl_bool_error);

	return isl_schedule_band_get_permutable(tree->band);
}

/* Mark the band tree root permutable or not according to "permutable"?
 */
__isl_give isl_schedule_tree *isl_schedule_tree_band_set_permutable(
	__isl_take isl_schedule_tree *tree, int permutable)
{
	if (!tree)
		return NULL;
	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", return isl_schedule_tree_free(tree));
	if (isl_schedule_tree_band_get_permutable(tree) == permutable)
		return tree;
	tree = isl_schedule_tree_cow(tree);
	if (!tree)
		return NULL;

	tree->band = isl_schedule_band_set_permutable(tree->band, permutable);
	if (!tree->band)
		return isl_schedule_tree_free(tree);
	return tree;
}

/* Return the schedule space of the band tree root.
 */
__isl_give isl_space *isl_schedule_tree_band_get_space(
	__isl_keep isl_schedule_tree *tree)
{
	if (!tree)
		return NULL;

	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", return NULL);

	return isl_schedule_band_get_space(tree->band);
}

/* Intersect the domain of the band schedule of the band tree root
 * with "domain".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_band_intersect_domain(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *domain)
{
	if (!tree || !domain)
		goto error;

	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", goto error);

	tree->band = isl_schedule_band_intersect_domain(tree->band, domain);
	if (!tree->band)
		return isl_schedule_tree_free(tree);

	return tree;
error:
	isl_schedule_tree_free(tree);
	isl_union_set_free(domain);
	return NULL;
}

/* Return the schedule of the band tree root in isolation.
 */
__isl_give isl_multi_union_pw_aff *isl_schedule_tree_band_get_partial_schedule(
	__isl_keep isl_schedule_tree *tree)
{
	if (!tree)
		return NULL;

	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", return NULL);

	return isl_schedule_band_get_partial_schedule(tree->band);
}

/* Replace the schedule of the band tree root by "schedule".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_band_set_partial_schedule(
	__isl_take isl_schedule_tree *tree,
	__isl_take isl_multi_union_pw_aff *schedule)
{
	tree = isl_schedule_tree_cow(tree);
	if (!tree || !schedule)
		goto error;

	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", return NULL);
	tree->band = isl_schedule_band_set_partial_schedule(tree->band,
								schedule);

	return tree;
error:
	isl_schedule_tree_free(tree);
	isl_multi_union_pw_aff_free(schedule);
	return NULL;
}

/* Return the loop AST generation type for the band member
 * of the band tree root at position "pos".
 */
enum isl_ast_loop_type isl_schedule_tree_band_member_get_ast_loop_type(
	__isl_keep isl_schedule_tree *tree, int pos)
{
	if (!tree)
		return isl_ast_loop_error;

	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", return isl_ast_loop_error);

	return isl_schedule_band_member_get_ast_loop_type(tree->band, pos);
}

/* Set the loop AST generation type for the band member of the band tree root
 * at position "pos" to "type".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_band_member_set_ast_loop_type(
	__isl_take isl_schedule_tree *tree, int pos,
	enum isl_ast_loop_type type)
{
	tree = isl_schedule_tree_cow(tree);
	if (!tree)
		return NULL;

	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", return isl_schedule_tree_free(tree));

	tree->band = isl_schedule_band_member_set_ast_loop_type(tree->band,
								pos, type);
	if (!tree->band)
		return isl_schedule_tree_free(tree);

	return tree;
}

/* Return the loop AST generation type for the band member
 * of the band tree root at position "pos" for the isolated part.
 */
enum isl_ast_loop_type isl_schedule_tree_band_member_get_isolate_ast_loop_type(
	__isl_keep isl_schedule_tree *tree, int pos)
{
	if (!tree)
		return isl_ast_loop_error;

	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", return isl_ast_loop_error);

	return isl_schedule_band_member_get_isolate_ast_loop_type(tree->band,
									pos);
}

/* Set the loop AST generation type for the band member of the band tree root
 * at position "pos" for the isolated part to "type".
 */
__isl_give isl_schedule_tree *
isl_schedule_tree_band_member_set_isolate_ast_loop_type(
	__isl_take isl_schedule_tree *tree, int pos,
	enum isl_ast_loop_type type)
{
	tree = isl_schedule_tree_cow(tree);
	if (!tree)
		return NULL;

	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", return isl_schedule_tree_free(tree));

	tree->band = isl_schedule_band_member_set_isolate_ast_loop_type(
							tree->band, pos, type);
	if (!tree->band)
		return isl_schedule_tree_free(tree);

	return tree;
}

/* Return the AST build options associated to the band tree root.
 */
__isl_give isl_union_set *isl_schedule_tree_band_get_ast_build_options(
	__isl_keep isl_schedule_tree *tree)
{
	if (!tree)
		return NULL;

	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", return NULL);

	return isl_schedule_band_get_ast_build_options(tree->band);
}

/* Replace the AST build options associated to band tree root by "options".
 * Updated the anchored field if the anchoredness of the root node itself
 * changes.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_band_set_ast_build_options(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *options)
{
	int was_anchored;

	tree = isl_schedule_tree_cow(tree);
	if (!tree || !options)
		goto error;

	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", goto error);

	was_anchored = isl_schedule_tree_is_anchored(tree);
	tree->band = isl_schedule_band_set_ast_build_options(tree->band,
								options);
	if (!tree->band)
		return isl_schedule_tree_free(tree);
	if (isl_schedule_tree_is_anchored(tree) != was_anchored)
		tree = isl_schedule_tree_update_anchored(tree);

	return tree;
error:
	isl_schedule_tree_free(tree);
	isl_union_set_free(options);
	return NULL;
}

/* Return the "isolate" option associated to the band tree root of "tree",
 * which is assumed to appear at schedule depth "depth".
 */
__isl_give isl_set *isl_schedule_tree_band_get_ast_isolate_option(
	__isl_keep isl_schedule_tree *tree, int depth)
{
	if (!tree)
		return NULL;

	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", return NULL);

	return isl_schedule_band_get_ast_isolate_option(tree->band, depth);
}

/* Return the context of the context tree root.
 */
__isl_give isl_set *isl_schedule_tree_context_get_context(
	__isl_keep isl_schedule_tree *tree)
{
	if (!tree)
		return NULL;

	if (tree->type != isl_schedule_node_context)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a context node", return NULL);

	return isl_set_copy(tree->context);
}

/* Return the domain of the domain tree root.
 */
__isl_give isl_union_set *isl_schedule_tree_domain_get_domain(
	__isl_keep isl_schedule_tree *tree)
{
	if (!tree)
		return NULL;

	if (tree->type != isl_schedule_node_domain)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a domain node", return NULL);

	return isl_union_set_copy(tree->domain);
}

/* Replace the domain of domain tree root "tree" by "domain".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_domain_set_domain(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *domain)
{
	tree = isl_schedule_tree_cow(tree);
	if (!tree || !domain)
		goto error;

	if (tree->type != isl_schedule_node_domain)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a domain node", goto error);

	isl_union_set_free(tree->domain);
	tree->domain = domain;

	return tree;
error:
	isl_schedule_tree_free(tree);
	isl_union_set_free(domain);
	return NULL;
}

/* Return the contraction of the expansion tree root.
 */
__isl_give isl_union_pw_multi_aff *isl_schedule_tree_expansion_get_contraction(
	__isl_keep isl_schedule_tree *tree)
{
	if (!tree)
		return NULL;

	if (tree->type != isl_schedule_node_expansion)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not an expansion node", return NULL);

	return isl_union_pw_multi_aff_copy(tree->contraction);
}

/* Return the expansion of the expansion tree root.
 */
__isl_give isl_union_map *isl_schedule_tree_expansion_get_expansion(
	__isl_keep isl_schedule_tree *tree)
{
	if (!tree)
		return NULL;

	if (tree->type != isl_schedule_node_expansion)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not an expansion node", return NULL);

	return isl_union_map_copy(tree->expansion);
}

/* Replace the contraction and the expansion of the expansion tree root "tree"
 * by "contraction" and "expansion".
 */
__isl_give isl_schedule_tree *
isl_schedule_tree_expansion_set_contraction_and_expansion(
	__isl_take isl_schedule_tree *tree,
	__isl_take isl_union_pw_multi_aff *contraction,
	__isl_take isl_union_map *expansion)
{
	tree = isl_schedule_tree_cow(tree);
	if (!tree || !contraction || !expansion)
		goto error;

	if (tree->type != isl_schedule_node_expansion)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not an expansion node", return NULL);

	isl_union_pw_multi_aff_free(tree->contraction);
	tree->contraction = contraction;
	isl_union_map_free(tree->expansion);
	tree->expansion = expansion;

	return tree;
error:
	isl_schedule_tree_free(tree);
	isl_union_pw_multi_aff_free(contraction);
	isl_union_map_free(expansion);
	return NULL;
}

/* Return the extension of the extension tree root.
 */
__isl_give isl_union_map *isl_schedule_tree_extension_get_extension(
	__isl_take isl_schedule_tree *tree)
{
	if (!tree)
		return NULL;

	if (tree->type != isl_schedule_node_extension)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not an extension node", return NULL);

	return isl_union_map_copy(tree->extension);
}

/* Replace the extension of extension tree root "tree" by "extension".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_extension_set_extension(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_map *extension)
{
	tree = isl_schedule_tree_cow(tree);
	if (!tree || !extension)
		goto error;

	if (tree->type != isl_schedule_node_extension)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not an extension node", return NULL);
	isl_union_map_free(tree->extension);
	tree->extension = extension;

	return tree;
error:
	isl_schedule_tree_free(tree);
	isl_union_map_free(extension);
	return NULL;
}

/* Return the filter of the filter tree root.
 */
__isl_give isl_union_set *isl_schedule_tree_filter_get_filter(
	__isl_keep isl_schedule_tree *tree)
{
	if (!tree)
		return NULL;

	if (tree->type != isl_schedule_node_filter)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a filter node", return NULL);

	return isl_union_set_copy(tree->filter);
}

/* Replace the filter of the filter tree root by "filter".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_filter_set_filter(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *filter)
{
	tree = isl_schedule_tree_cow(tree);
	if (!tree || !filter)
		goto error;

	if (tree->type != isl_schedule_node_filter)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a filter node", return NULL);

	isl_union_set_free(tree->filter);
	tree->filter = filter;

	return tree;
error:
	isl_schedule_tree_free(tree);
	isl_union_set_free(filter);
	return NULL;
}

/* Return the guard of the guard tree root.
 */
__isl_give isl_set *isl_schedule_tree_guard_get_guard(
	__isl_take isl_schedule_tree *tree)
{
	if (!tree)
		return NULL;

	if (tree->type != isl_schedule_node_guard)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a guard node", return NULL);

	return isl_set_copy(tree->guard);
}

/* Return the mark identifier of the mark tree root "tree".
 */
__isl_give isl_id *isl_schedule_tree_mark_get_id(
	__isl_keep isl_schedule_tree *tree)
{
	if (!tree)
		return NULL;

	if (tree->type != isl_schedule_node_mark)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a mark node", return NULL);

	return isl_id_copy(tree->mark);
}

/* Set dim to the range dimension of "map" and abort the search.
 */
static isl_stat set_range_dim(__isl_take isl_map *map, void *user)
{
	int *dim = user;

	*dim = isl_map_dim(map, isl_dim_out);
	isl_map_free(map);

	return isl_stat_error;
}

/* Return the dimension of the range of "umap".
 * "umap" is assumed not to be empty and
 * all maps inside "umap" are assumed to have the same range.
 *
 * We extract the range dimension from the first map in "umap".
 */
static int range_dim(__isl_keep isl_union_map *umap)
{
	int dim = -1;

	if (!umap)
		return -1;
	if (isl_union_map_n_map(umap) == 0)
		isl_die(isl_union_map_get_ctx(umap), isl_error_internal,
			"unexpected empty input", return -1);

	isl_union_map_foreach_map(umap, &set_range_dim, &dim);

	return dim;
}

/* Append an "extra" number of zeros to the range of "umap" and
 * return the result.
 */
static __isl_give isl_union_map *append_range(__isl_take isl_union_map *umap,
	int extra)
{
	isl_union_set *dom;
	isl_space *space;
	isl_multi_val *mv;
	isl_union_pw_multi_aff *suffix;
	isl_union_map *universe;
	isl_union_map *suffix_umap;

	universe = isl_union_map_universe(isl_union_map_copy(umap));
	dom = isl_union_map_domain(universe);
	space = isl_union_set_get_space(dom);
	space = isl_space_set_from_params(space);
	space = isl_space_add_dims(space, isl_dim_set, extra);
	mv = isl_multi_val_zero(space);

	suffix = isl_union_pw_multi_aff_multi_val_on_domain(dom, mv);
	suffix_umap = isl_union_map_from_union_pw_multi_aff(suffix);
	umap = isl_union_map_flat_range_product(umap, suffix_umap);

	return umap;
}

/* Should we skip the root of "tree" while looking for the first
 * descendant with schedule information?
 * That is, is it impossible to derive any information about
 * the iteration domain from this node?
 *
 * We do not want to skip leaf or error nodes because there is
 * no point in looking any deeper from these nodes.
 * We can only extract partial iteration domain information
 * from an extension node, but extension nodes are not supported
 * by the caller and it will error out on them.
 */
static int domain_less(__isl_keep isl_schedule_tree *tree)
{
	enum isl_schedule_node_type type;

	type = isl_schedule_tree_get_type(tree);
	switch (type) {
	case isl_schedule_node_band:
		return isl_schedule_tree_band_n_member(tree) == 0;
	case isl_schedule_node_context:
	case isl_schedule_node_guard:
	case isl_schedule_node_mark:
		return 1;
	case isl_schedule_node_leaf:
	case isl_schedule_node_error:
	case isl_schedule_node_domain:
	case isl_schedule_node_expansion:
	case isl_schedule_node_extension:
	case isl_schedule_node_filter:
	case isl_schedule_node_set:
	case isl_schedule_node_sequence:
		return 0;
	}

	isl_die(isl_schedule_tree_get_ctx(tree), isl_error_internal,
		"unhandled case", return 0);
}

/* Move down to the first descendant of "tree" that contains any schedule
 * information or return "leaf" if there is no such descendant.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_first_schedule_descendant(
	__isl_take isl_schedule_tree *tree, __isl_keep isl_schedule_tree *leaf)
{
	while (domain_less(tree)) {
		if (!isl_schedule_tree_has_children(tree)) {
			isl_schedule_tree_free(tree);
			return isl_schedule_tree_copy(leaf);
		}
		tree = isl_schedule_tree_child(tree, 0);
	}

	return tree;
}

static __isl_give isl_union_map *subtree_schedule_extend(
	__isl_keep isl_schedule_tree *tree, __isl_take isl_union_map *outer);

/* Extend the schedule map "outer" with the subtree schedule
 * of the (single) child of "tree", if any.
 *
 * If "tree" does not have any descendants (apart from those that
 * do not carry any schedule information), then we simply return "outer".
 * Otherwise, we extend the schedule map "outer" with the subtree schedule
 * of the single child.
 */
static __isl_give isl_union_map *subtree_schedule_extend_child(
	__isl_keep isl_schedule_tree *tree, __isl_take isl_union_map *outer)
{
	isl_schedule_tree *child;
	isl_union_map *res;

	if (!tree)
		return isl_union_map_free(outer);
	if (!isl_schedule_tree_has_children(tree))
		return outer;
	child = isl_schedule_tree_get_child(tree, 0);
	if (!child)
		return isl_union_map_free(outer);
	res = subtree_schedule_extend(child, outer);
	isl_schedule_tree_free(child);
	return res;
}

/* Extract the parameter space from one of the children of "tree",
 * which are assumed to be filters.
 */
static __isl_give isl_space *extract_space_from_filter_child(
	__isl_keep isl_schedule_tree *tree)
{
	isl_space *space;
	isl_union_set *dom;
	isl_schedule_tree *child;

	child = isl_schedule_tree_list_get_schedule_tree(tree->children, 0);
	dom = isl_schedule_tree_filter_get_filter(child);
	space = isl_union_set_get_space(dom);
	isl_union_set_free(dom);
	isl_schedule_tree_free(child);

	return space;
}

/* Extend the schedule map "outer" with the subtree schedule
 * of a set or sequence node.
 *
 * The schedule for the set or sequence node itself is composed of
 * pieces of the form
 *
 *	filter -> []
 *
 * or
 *
 *	filter -> [index]
 *
 * The first form is used if there is only a single child or
 * if the current node is a set node and the schedule_separate_components
 * option is not set.
 *
 * Each of the pieces above is extended with the subtree schedule of
 * the child of the corresponding filter, if any, padded with zeros
 * to ensure that all pieces have the same range dimension.
 */
static __isl_give isl_union_map *subtree_schedule_extend_from_children(
	__isl_keep isl_schedule_tree *tree, __isl_take isl_union_map *outer)
{
	int i, n;
	int dim;
	int separate;
	isl_ctx *ctx;
	isl_val *v = NULL;
	isl_multi_val *mv;
	isl_space *space;
	isl_union_map *umap;

	if (!tree)
		return NULL;

	ctx = isl_schedule_tree_get_ctx(tree);
	if (!tree->children)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_internal,
			"missing children", return NULL);
	n = isl_schedule_tree_list_n_schedule_tree(tree->children);
	if (n == 0)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_internal,
			"missing children", return NULL);

	separate = n > 1 && (tree->type == isl_schedule_node_sequence ||
			    isl_options_get_schedule_separate_components(ctx));

	space = extract_space_from_filter_child(tree);

	umap = isl_union_map_empty(isl_space_copy(space));
	space = isl_space_set_from_params(space);
	if (separate) {
		space = isl_space_add_dims(space, isl_dim_set, 1);
		v = isl_val_zero(ctx);
	}
	mv = isl_multi_val_zero(space);

	dim = isl_multi_val_dim(mv, isl_dim_set);
	for (i = 0; i < n; ++i) {
		isl_union_pw_multi_aff *upma;
		isl_union_map *umap_i;
		isl_union_set *dom;
		isl_schedule_tree *child;
		int dim_i;
		int empty;

		child = isl_schedule_tree_list_get_schedule_tree(
							tree->children, i);
		dom = isl_schedule_tree_filter_get_filter(child);

		if (separate) {
			mv = isl_multi_val_set_val(mv, 0, isl_val_copy(v));
			v = isl_val_add_ui(v, 1);
		}
		upma = isl_union_pw_multi_aff_multi_val_on_domain(dom,
							isl_multi_val_copy(mv));
		umap_i = isl_union_map_from_union_pw_multi_aff(upma);
		umap_i = isl_union_map_flat_range_product(
					    isl_union_map_copy(outer), umap_i);
		umap_i = subtree_schedule_extend_child(child, umap_i);
		isl_schedule_tree_free(child);

		empty = isl_union_map_is_empty(umap_i);
		if (empty < 0)
			umap_i = isl_union_map_free(umap_i);
		else if (empty) {
			isl_union_map_free(umap_i);
			continue;
		}

		dim_i = range_dim(umap_i);
		if (dim_i < 0) {
			umap = isl_union_map_free(umap);
		} else if (dim < dim_i) {
			umap = append_range(umap, dim_i - dim);
			dim = dim_i;
		} else if (dim_i < dim) {
			umap_i = append_range(umap_i, dim - dim_i);
		}
		umap = isl_union_map_union(umap, umap_i);
	}

	isl_val_free(v);
	isl_multi_val_free(mv);
	isl_union_map_free(outer);

	return umap;
}

/* Extend the schedule map "outer" with the subtree schedule of "tree".
 *
 * If the root of the tree is a set or a sequence, then we extend
 * the schedule map in subtree_schedule_extend_from_children.
 * Otherwise, we extend the schedule map with the partial schedule
 * corresponding to the root of the tree and then continue with
 * the single child of this root.
 * In the special case of an expansion, the schedule map is "extended"
 * by applying the expansion to the domain of the schedule map.
 */
static __isl_give isl_union_map *subtree_schedule_extend(
	__isl_keep isl_schedule_tree *tree, __isl_take isl_union_map *outer)
{
	isl_multi_union_pw_aff *mupa;
	isl_union_map *umap;
	isl_union_set *domain;

	if (!tree)
		return NULL;

	switch (tree->type) {
	case isl_schedule_node_error:
		return isl_union_map_free(outer);
	case isl_schedule_node_extension:
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"cannot construct subtree schedule of tree "
			"with extension nodes",
			return isl_union_map_free(outer));
	case isl_schedule_node_context:
	case isl_schedule_node_guard:
	case isl_schedule_node_mark:
		return subtree_schedule_extend_child(tree, outer);
	case isl_schedule_node_band:
		if (isl_schedule_tree_band_n_member(tree) == 0)
			return subtree_schedule_extend_child(tree, outer);
		mupa = isl_schedule_band_get_partial_schedule(tree->band);
		umap = isl_union_map_from_multi_union_pw_aff(mupa);
		outer = isl_union_map_flat_range_product(outer, umap);
		umap = subtree_schedule_extend_child(tree, outer);
		break;
	case isl_schedule_node_domain:
		domain = isl_schedule_tree_domain_get_domain(tree);
		umap = isl_union_map_from_domain(domain);
		outer = isl_union_map_flat_range_product(outer, umap);
		umap = subtree_schedule_extend_child(tree, outer);
		break;
	case isl_schedule_node_expansion:
		umap = isl_schedule_tree_expansion_get_expansion(tree);
		outer = isl_union_map_apply_domain(outer, umap);
		umap = subtree_schedule_extend_child(tree, outer);
		break;
	case isl_schedule_node_filter:
		domain = isl_schedule_tree_filter_get_filter(tree);
		umap = isl_union_map_from_domain(domain);
		outer = isl_union_map_flat_range_product(outer, umap);
		umap = subtree_schedule_extend_child(tree, outer);
		break;
	case isl_schedule_node_leaf:
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_internal,
			"leaf node should be handled by caller", return NULL);
	case isl_schedule_node_set:
	case isl_schedule_node_sequence:
		umap = subtree_schedule_extend_from_children(tree, outer);
		break;
	}

	return umap;
}

static __isl_give isl_union_set *initial_domain(
	__isl_keep isl_schedule_tree *tree);

/* Extract a universe domain from the children of the tree root "tree",
 * which is a set or sequence, meaning that its children are filters.
 * In particular, return the union of the universes of the filters.
 */
static __isl_give isl_union_set *initial_domain_from_children(
	__isl_keep isl_schedule_tree *tree)
{
	int i, n;
	isl_space *space;
	isl_union_set *domain;

	if (!tree->children)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_internal,
			"missing children", return NULL);
	n = isl_schedule_tree_list_n_schedule_tree(tree->children);
	if (n == 0)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_internal,
			"missing children", return NULL);

	space = extract_space_from_filter_child(tree);
	domain = isl_union_set_empty(space);

	for (i = 0; i < n; ++i) {
		isl_schedule_tree *child;
		isl_union_set *domain_i;

		child = isl_schedule_tree_get_child(tree, i);
		domain_i = initial_domain(child);
		domain = isl_union_set_union(domain, domain_i);
		isl_schedule_tree_free(child);
	}

	return domain;
}

/* Extract a universe domain from the tree root "tree".
 * The caller is responsible for making sure that this node
 * would not be skipped by isl_schedule_tree_first_schedule_descendant
 * and that it is not a leaf node.
 */
static __isl_give isl_union_set *initial_domain(
	__isl_keep isl_schedule_tree *tree)
{
	isl_multi_union_pw_aff *mupa;
	isl_union_set *domain;
	isl_union_map *exp;

	if (!tree)
		return NULL;

	switch (tree->type) {
	case isl_schedule_node_error:
		return NULL;
	case isl_schedule_node_context:
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_internal,
			"context node should be handled by caller",
			return NULL);
	case isl_schedule_node_guard:
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_internal,
			"guard node should be handled by caller",
			return NULL);
	case isl_schedule_node_mark:
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_internal,
			"mark node should be handled by caller",
			return NULL);
	case isl_schedule_node_extension:
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"cannot construct subtree schedule of tree "
			"with extension nodes", return NULL);
	case isl_schedule_node_band:
		if (isl_schedule_tree_band_n_member(tree) == 0)
			isl_die(isl_schedule_tree_get_ctx(tree),
				isl_error_internal,
				"0D band should be handled by caller",
				return NULL);
		mupa = isl_schedule_band_get_partial_schedule(tree->band);
		domain = isl_multi_union_pw_aff_domain(mupa);
		domain = isl_union_set_universe(domain);
		break;
	case isl_schedule_node_domain:
		domain = isl_schedule_tree_domain_get_domain(tree);
		domain = isl_union_set_universe(domain);
		break;
	case isl_schedule_node_expansion:
		exp = isl_schedule_tree_expansion_get_expansion(tree);
		exp = isl_union_map_universe(exp);
		domain = isl_union_map_domain(exp);
		break;
	case isl_schedule_node_filter:
		domain = isl_schedule_tree_filter_get_filter(tree);
		domain = isl_union_set_universe(domain);
		break;
	case isl_schedule_node_leaf:
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_internal,
			"leaf node should be handled by caller", return NULL);
	case isl_schedule_node_set:
	case isl_schedule_node_sequence:
		domain = initial_domain_from_children(tree);
		break;
	}

	return domain;
}

/* Return the subtree schedule of a node that contains some schedule
 * information, i.e., a node that would not be skipped by
 * isl_schedule_tree_first_schedule_descendant and that is not a leaf.
 *
 * If the tree contains any expansions, then the returned subtree
 * schedule is formulated in terms of the expanded domains.
 * The tree is not allowed to contain any extension nodes.
 *
 * We start with an initial zero-dimensional subtree schedule based
 * on the domain information in the root node and then extend it
 * based on the schedule information in the root node and its descendants.
 */
__isl_give isl_union_map *isl_schedule_tree_get_subtree_schedule_union_map(
	__isl_keep isl_schedule_tree *tree)
{
	isl_union_set *domain;
	isl_union_map *umap;

	domain = initial_domain(tree);
	umap = isl_union_map_from_domain(domain);
	return subtree_schedule_extend(tree, umap);
}

/* Multiply the partial schedule of the band root node of "tree"
 * with the factors in "mv".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_band_scale(
	__isl_take isl_schedule_tree *tree, __isl_take isl_multi_val *mv)
{
	if (!tree || !mv)
		goto error;
	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", goto error);

	tree = isl_schedule_tree_cow(tree);
	if (!tree)
		goto error;

	tree->band = isl_schedule_band_scale(tree->band, mv);
	if (!tree->band)
		return isl_schedule_tree_free(tree);

	return tree;
error:
	isl_schedule_tree_free(tree);
	isl_multi_val_free(mv);
	return NULL;
}

/* Divide the partial schedule of the band root node of "tree"
 * by the factors in "mv".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_band_scale_down(
	__isl_take isl_schedule_tree *tree, __isl_take isl_multi_val *mv)
{
	if (!tree || !mv)
		goto error;
	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", goto error);

	tree = isl_schedule_tree_cow(tree);
	if (!tree)
		goto error;

	tree->band = isl_schedule_band_scale_down(tree->band, mv);
	if (!tree->band)
		return isl_schedule_tree_free(tree);

	return tree;
error:
	isl_schedule_tree_free(tree);
	isl_multi_val_free(mv);
	return NULL;
}

/* Reduce the partial schedule of the band root node of "tree"
 * modulo the factors in "mv".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_band_mod(
	__isl_take isl_schedule_tree *tree, __isl_take isl_multi_val *mv)
{
	if (!tree || !mv)
		goto error;
	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", goto error);

	tree = isl_schedule_tree_cow(tree);
	if (!tree)
		goto error;

	tree->band = isl_schedule_band_mod(tree->band, mv);
	if (!tree->band)
		return isl_schedule_tree_free(tree);

	return tree;
error:
	isl_schedule_tree_free(tree);
	isl_multi_val_free(mv);
	return NULL;
}

/* Shift the partial schedule of the band root node of "tree" by "shift".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_band_shift(
	__isl_take isl_schedule_tree *tree,
	__isl_take isl_multi_union_pw_aff *shift)
{
	if (!tree || !shift)
		goto error;
	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", goto error);

	tree = isl_schedule_tree_cow(tree);
	if (!tree)
		goto error;

	tree->band = isl_schedule_band_shift(tree->band, shift);
	if (!tree->band)
		return isl_schedule_tree_free(tree);

	return tree;
error:
	isl_schedule_tree_free(tree);
	isl_multi_union_pw_aff_free(shift);
	return NULL;
}

/* Given two trees with sequence roots, replace the child at position
 * "pos" of "tree" with the children of "child".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_sequence_splice(
	__isl_take isl_schedule_tree *tree, int pos,
	__isl_take isl_schedule_tree *child)
{
	int n;
	isl_schedule_tree_list *list1, *list2;

	tree = isl_schedule_tree_cow(tree);
	if (!tree || !child)
		goto error;
	if (isl_schedule_tree_get_type(tree) != isl_schedule_node_sequence)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a sequence node", goto error);
	n = isl_schedule_tree_n_children(tree);
	if (pos < 0 || pos >= n)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"position out of bounds", goto error);
	if (isl_schedule_tree_get_type(child) != isl_schedule_node_sequence)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a sequence node", goto error);

	list1 = isl_schedule_tree_list_copy(tree->children);
	list1 = isl_schedule_tree_list_drop(list1, pos, n - pos);
	list2 = isl_schedule_tree_list_copy(tree->children);
	list2 = isl_schedule_tree_list_drop(list2, 0, pos + 1);
	list1 = isl_schedule_tree_list_concat(list1,
				isl_schedule_tree_list_copy(child->children));
	list1 = isl_schedule_tree_list_concat(list1, list2);

	isl_schedule_tree_free(tree);
	isl_schedule_tree_free(child);
	return isl_schedule_tree_from_children(isl_schedule_node_sequence,
						list1);
error:
	isl_schedule_tree_free(tree);
	isl_schedule_tree_free(child);
	return NULL;
}

/* Tile the band root node of "tree" with tile sizes "sizes".
 *
 * We duplicate the band node, change the schedule of one of them
 * to the tile schedule and the other to the point schedule and then
 * attach the point band as a child to the tile band.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_band_tile(
	__isl_take isl_schedule_tree *tree, __isl_take isl_multi_val *sizes)
{
	isl_schedule_tree *child = NULL;

	if (!tree || !sizes)
		goto error;
	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", goto error);

	child = isl_schedule_tree_copy(tree);
	tree = isl_schedule_tree_cow(tree);
	child = isl_schedule_tree_cow(child);
	if (!tree || !child)
		goto error;

	tree->band = isl_schedule_band_tile(tree->band,
					    isl_multi_val_copy(sizes));
	if (!tree->band)
		goto error;
	child->band = isl_schedule_band_point(child->band, tree->band, sizes);
	if (!child->band)
		child = isl_schedule_tree_free(child);

	tree = isl_schedule_tree_replace_child(tree, 0, child);

	return tree;
error:
	isl_schedule_tree_free(child);
	isl_schedule_tree_free(tree);
	isl_multi_val_free(sizes);
	return NULL;
}

/* Given an isolate AST generation option "isolate" for a band of size pos + n,
 * return the corresponding option for a band covering the first "pos"
 * members.
 *
 * The input isolate option is of the form
 *
 *	isolate[[flattened outer bands] -> [pos; n]]
 *
 * The output isolate option is of the form
 *
 *	isolate[[flattened outer bands] -> [pos]]
 */
static __isl_give isl_set *isolate_initial(__isl_keep isl_set *isolate,
	int pos, int n)
{
	isl_id *id;
	isl_map *map;

	isolate = isl_set_copy(isolate);
	id = isl_set_get_tuple_id(isolate);
	map = isl_set_unwrap(isolate);
	map = isl_map_project_out(map, isl_dim_out, pos, n);
	isolate = isl_map_wrap(map);
	isolate = isl_set_set_tuple_id(isolate, id);

	return isolate;
}

/* Given an isolate AST generation option "isolate" for a band of size pos + n,
 * return the corresponding option for a band covering the final "n"
 * members within a band covering the first "pos" members.
 *
 * The input isolate option is of the form
 *
 *	isolate[[flattened outer bands] -> [pos; n]]
 *
 * The output isolate option is of the form
 *
 *	isolate[[flattened outer bands; pos] -> [n]]
 *
 *
 * The range is first split into
 *
 *	isolate[[flattened outer bands] -> [[pos] -> [n]]]
 *
 * and then the first pos members are moved to the domain
 *
 *	isolate[[[flattened outer bands] -> [pos]] -> [n]]
 *
 * after which the domain is flattened to obtain the desired output.
 */
static __isl_give isl_set *isolate_final(__isl_keep isl_set *isolate,
	int pos, int n)
{
	isl_id *id;
	isl_space *space;
	isl_multi_aff *ma1, *ma2;
	isl_map *map;

	isolate = isl_set_copy(isolate);
	id = isl_set_get_tuple_id(isolate);
	map = isl_set_unwrap(isolate);
	space = isl_space_range(isl_map_get_space(map));
	ma1 = isl_multi_aff_project_out_map(isl_space_copy(space),
						   isl_dim_set, pos, n);
	ma2 = isl_multi_aff_project_out_map(space, isl_dim_set, 0, pos);
	ma1 = isl_multi_aff_range_product(ma1, ma2);
	map = isl_map_apply_range(map, isl_map_from_multi_aff(ma1));
	map = isl_map_uncurry(map);
	map = isl_map_flatten_domain(map);
	isolate = isl_map_wrap(map);
	isolate = isl_set_set_tuple_id(isolate, id);

	return isolate;
}

/* Split the band root node of "tree" into two nested band nodes,
 * one with the first "pos" dimensions and
 * one with the remaining dimensions.
 * The tree is itself positioned at schedule depth "depth".
 *
 * The loop AST generation type options and the isolate option
 * are split over the the two band nodes.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_band_split(
	__isl_take isl_schedule_tree *tree, int pos, int depth)
{
	int n;
	isl_set *isolate, *tree_isolate, *child_isolate;
	isl_schedule_tree *child;

	if (!tree)
		return NULL;
	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", return isl_schedule_tree_free(tree));

	n = isl_schedule_tree_band_n_member(tree);
	if (pos < 0 || pos > n)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"position out of bounds",
			return isl_schedule_tree_free(tree));

	child = isl_schedule_tree_copy(tree);
	tree = isl_schedule_tree_cow(tree);
	child = isl_schedule_tree_cow(child);
	if (!tree || !child)
		goto error;

	isolate = isl_schedule_tree_band_get_ast_isolate_option(tree, depth);
	tree_isolate = isolate_initial(isolate, pos, n - pos);
	child_isolate = isolate_final(isolate, pos, n - pos);
	child->band = isl_schedule_band_drop(child->band, 0, pos);
	child->band = isl_schedule_band_replace_ast_build_option(child->band,
					isl_set_copy(isolate), child_isolate);
	tree->band = isl_schedule_band_drop(tree->band, pos, n - pos);
	tree->band = isl_schedule_band_replace_ast_build_option(tree->band,
					isl_set_copy(isolate), tree_isolate);
	isl_set_free(isolate);
	if (!child->band || !tree->band)
		goto error;

	tree = isl_schedule_tree_replace_child(tree, 0, child);

	return tree;
error:
	isl_schedule_tree_free(child);
	isl_schedule_tree_free(tree);
	return NULL;
}

/* Attach "tree2" at each of the leaves of "tree1".
 *
 * If "tree1" does not have any explicit children, then make "tree2"
 * its single child.  Otherwise, attach "tree2" to the leaves of
 * each of the children of "tree1".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_append_to_leaves(
	__isl_take isl_schedule_tree *tree1,
	__isl_take isl_schedule_tree *tree2)
{
	int i, n;

	if (!tree1 || !tree2)
		goto error;
	n = isl_schedule_tree_n_children(tree1);
	if (n == 0) {
		isl_schedule_tree_list *list;
		list = isl_schedule_tree_list_from_schedule_tree(tree2);
		tree1 = isl_schedule_tree_set_children(tree1, list);
		return tree1;
	}
	for (i = 0; i < n; ++i) {
		isl_schedule_tree *child;

		child = isl_schedule_tree_get_child(tree1, i);
		child = isl_schedule_tree_append_to_leaves(child,
					isl_schedule_tree_copy(tree2));
		tree1 = isl_schedule_tree_replace_child(tree1, i, child);
	}

	isl_schedule_tree_free(tree2);
	return tree1;
error:
	isl_schedule_tree_free(tree1);
	isl_schedule_tree_free(tree2);
	return NULL;
}

/* Reset the user pointer on all identifiers of parameters and tuples
 * in the root of "tree".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_reset_user(
	__isl_take isl_schedule_tree *tree)
{
	if (isl_schedule_tree_is_leaf(tree))
		return tree;

	tree = isl_schedule_tree_cow(tree);
	if (!tree)
		return NULL;

	switch (tree->type) {
	case isl_schedule_node_error:
		return isl_schedule_tree_free(tree);
	case isl_schedule_node_band:
		tree->band = isl_schedule_band_reset_user(tree->band);
		if (!tree->band)
			return isl_schedule_tree_free(tree);
		break;
	case isl_schedule_node_context:
		tree->context = isl_set_reset_user(tree->context);
		if (!tree->context)
			return isl_schedule_tree_free(tree);
		break;
	case isl_schedule_node_domain:
		tree->domain = isl_union_set_reset_user(tree->domain);
		if (!tree->domain)
			return isl_schedule_tree_free(tree);
		break;
	case isl_schedule_node_expansion:
		tree->contraction =
			isl_union_pw_multi_aff_reset_user(tree->contraction);
		tree->expansion = isl_union_map_reset_user(tree->expansion);
		if (!tree->contraction || !tree->expansion)
			return isl_schedule_tree_free(tree);
		break;
	case isl_schedule_node_extension:
		tree->extension = isl_union_map_reset_user(tree->extension);
		if (!tree->extension)
			return isl_schedule_tree_free(tree);
		break;
	case isl_schedule_node_filter:
		tree->filter = isl_union_set_reset_user(tree->filter);
		if (!tree->filter)
			return isl_schedule_tree_free(tree);
		break;
	case isl_schedule_node_guard:
		tree->guard = isl_set_reset_user(tree->guard);
		if (!tree->guard)
			return isl_schedule_tree_free(tree);
		break;
	case isl_schedule_node_leaf:
	case isl_schedule_node_mark:
	case isl_schedule_node_sequence:
	case isl_schedule_node_set:
		break;
	}

	return tree;
}

/* Align the parameters of the root of "tree" to those of "space".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_align_params(
	__isl_take isl_schedule_tree *tree, __isl_take isl_space *space)
{
	if (!space)
		goto error;

	if (isl_schedule_tree_is_leaf(tree)) {
		isl_space_free(space);
		return tree;
	}

	tree = isl_schedule_tree_cow(tree);
	if (!tree)
		goto error;

	switch (tree->type) {
	case isl_schedule_node_error:
		goto error;
	case isl_schedule_node_band:
		tree->band = isl_schedule_band_align_params(tree->band, space);
		if (!tree->band)
			return isl_schedule_tree_free(tree);
		break;
	case isl_schedule_node_context:
		tree->context = isl_set_align_params(tree->context, space);
		if (!tree->context)
			return isl_schedule_tree_free(tree);
		break;
	case isl_schedule_node_domain:
		tree->domain = isl_union_set_align_params(tree->domain, space);
		if (!tree->domain)
			return isl_schedule_tree_free(tree);
		break;
	case isl_schedule_node_expansion:
		tree->contraction =
			isl_union_pw_multi_aff_align_params(tree->contraction,
							isl_space_copy(space));
		tree->expansion = isl_union_map_align_params(tree->expansion,
								space);
		if (!tree->contraction || !tree->expansion)
			return isl_schedule_tree_free(tree);
		break;
	case isl_schedule_node_extension:
		tree->extension = isl_union_map_align_params(tree->extension,
								space);
		if (!tree->extension)
			return isl_schedule_tree_free(tree);
		break;
	case isl_schedule_node_filter:
		tree->filter = isl_union_set_align_params(tree->filter, space);
		if (!tree->filter)
			return isl_schedule_tree_free(tree);
		break;
	case isl_schedule_node_guard:
		tree->guard = isl_set_align_params(tree->guard, space);
		if (!tree->guard)
			return isl_schedule_tree_free(tree);
		break;
	case isl_schedule_node_leaf:
	case isl_schedule_node_mark:
	case isl_schedule_node_sequence:
	case isl_schedule_node_set:
		isl_space_free(space);
		break;
	}

	return tree;
error:
	isl_space_free(space);
	isl_schedule_tree_free(tree);
	return NULL;
}

/* Does "tree" involve the iteration domain?
 * That is, does it need to be modified
 * by isl_schedule_tree_pullback_union_pw_multi_aff?
 */
static int involves_iteration_domain(__isl_keep isl_schedule_tree *tree)
{
	if (!tree)
		return -1;

	switch (tree->type) {
	case isl_schedule_node_error:
		return -1;
	case isl_schedule_node_band:
	case isl_schedule_node_domain:
	case isl_schedule_node_expansion:
	case isl_schedule_node_extension:
	case isl_schedule_node_filter:
		return 1;
	case isl_schedule_node_context:
	case isl_schedule_node_leaf:
	case isl_schedule_node_guard:
	case isl_schedule_node_mark:
	case isl_schedule_node_sequence:
	case isl_schedule_node_set:
		return 0;
	}

	isl_die(isl_schedule_tree_get_ctx(tree), isl_error_internal,
		"unhandled case", return -1);
}

/* Compute the pullback of the root node of "tree" by the function
 * represented by "upma".
 * In other words, plug in "upma" in the iteration domains of
 * the root node of "tree".
 * We currently do not handle expansion nodes.
 *
 * We first check if the root node involves any iteration domains.
 * If so, we handle the specific cases.
 */
__isl_give isl_schedule_tree *isl_schedule_tree_pullback_union_pw_multi_aff(
	__isl_take isl_schedule_tree *tree,
	__isl_take isl_union_pw_multi_aff *upma)
{
	int involves;

	if (!tree || !upma)
		goto error;

	involves = involves_iteration_domain(tree);
	if (involves < 0)
		goto error;
	if (!involves) {
		isl_union_pw_multi_aff_free(upma);
		return tree;
	}

	tree = isl_schedule_tree_cow(tree);
	if (!tree)
		goto error;

	if (tree->type == isl_schedule_node_band) {
		tree->band = isl_schedule_band_pullback_union_pw_multi_aff(
							    tree->band, upma);
		if (!tree->band)
			return isl_schedule_tree_free(tree);
	} else if (tree->type == isl_schedule_node_domain) {
		tree->domain =
			isl_union_set_preimage_union_pw_multi_aff(tree->domain,
									upma);
		if (!tree->domain)
			return isl_schedule_tree_free(tree);
	} else if (tree->type == isl_schedule_node_expansion) {
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_unsupported,
			"cannot pullback expansion node", goto error);
	} else if (tree->type == isl_schedule_node_extension) {
		tree->extension =
			isl_union_map_preimage_range_union_pw_multi_aff(
			    tree->extension, upma);
		if (!tree->extension)
			return isl_schedule_tree_free(tree);
	} else if (tree->type == isl_schedule_node_filter) {
		tree->filter =
			isl_union_set_preimage_union_pw_multi_aff(tree->filter,
									upma);
		if (!tree->filter)
			return isl_schedule_tree_free(tree);
	}

	return tree;
error:
	isl_union_pw_multi_aff_free(upma);
	isl_schedule_tree_free(tree);
	return NULL;
}

/* Compute the gist of the band tree root with respect to "context".
 */
__isl_give isl_schedule_tree *isl_schedule_tree_band_gist(
	__isl_take isl_schedule_tree *tree, __isl_take isl_union_set *context)
{
	if (!tree)
		return NULL;
	if (tree->type != isl_schedule_node_band)
		isl_die(isl_schedule_tree_get_ctx(tree), isl_error_invalid,
			"not a band node", goto error);
	tree = isl_schedule_tree_cow(tree);
	if (!tree)
		goto error;

	tree->band = isl_schedule_band_gist(tree->band, context);
	if (!tree->band)
		return isl_schedule_tree_free(tree);
	return tree;
error:
	isl_union_set_free(context);
	isl_schedule_tree_free(tree);
	return NULL;
}

/* Are any members in "band" marked coincident?
 */
static int any_coincident(__isl_keep isl_schedule_band *band)
{
	int i, n;

	n = isl_schedule_band_n_member(band);
	for (i = 0; i < n; ++i)
		if (isl_schedule_band_member_get_coincident(band, i))
			return 1;

	return 0;
}

/* Print the band node "band" to "p".
 *
 * The permutable and coincident properties are only printed if they
 * are different from the defaults.
 * The coincident property is always printed in YAML flow style.
 */
static __isl_give isl_printer *print_tree_band(__isl_take isl_printer *p,
	__isl_keep isl_schedule_band *band)
{
	isl_union_set *options;
	int empty;

	p = isl_printer_print_str(p, "schedule");
	p = isl_printer_yaml_next(p);
	p = isl_printer_print_str(p, "\"");
	p = isl_printer_print_multi_union_pw_aff(p, band->mupa);
	p = isl_printer_print_str(p, "\"");
	if (isl_schedule_band_get_permutable(band)) {
		p = isl_printer_yaml_next(p);
		p = isl_printer_print_str(p, "permutable");
		p = isl_printer_yaml_next(p);
		p = isl_printer_print_int(p, 1);
	}
	if (any_coincident(band)) {
		int i, n;
		int style;

		p = isl_printer_yaml_next(p);
		p = isl_printer_print_str(p, "coincident");
		p = isl_printer_yaml_next(p);
		style = isl_printer_get_yaml_style(p);
		p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_FLOW);
		p = isl_printer_yaml_start_sequence(p);
		n = isl_schedule_band_n_member(band);
		for (i = 0; i < n; ++i) {
			p = isl_printer_print_int(p,
			    isl_schedule_band_member_get_coincident(band, i));
			p = isl_printer_yaml_next(p);
		}
		p = isl_printer_yaml_end_sequence(p);
		p = isl_printer_set_yaml_style(p, style);
	}
	options = isl_schedule_band_get_ast_build_options(band);
	empty = isl_union_set_is_empty(options);
	if (empty < 0)
		p = isl_printer_free(p);
	if (!empty) {
		p = isl_printer_yaml_next(p);
		p = isl_printer_print_str(p, "options");
		p = isl_printer_yaml_next(p);
		p = isl_printer_print_str(p, "\"");
		p = isl_printer_print_union_set(p, options);
		p = isl_printer_print_str(p, "\"");
	}
	isl_union_set_free(options);

	return p;
}

/* Print "tree" to "p".
 *
 * If "n_ancestor" is non-negative, then "child_pos" contains the child
 * positions of a descendant of the current node that should be marked
 * (by the comment "YOU ARE HERE").  In particular, if "n_ancestor"
 * is zero, then the current node should be marked.
 * The marking is only printed in YAML block format.
 *
 * Implicit leaf nodes are not printed, except if they correspond
 * to the node that should be marked.
 */
__isl_give isl_printer *isl_printer_print_schedule_tree_mark(
	__isl_take isl_printer *p, __isl_keep isl_schedule_tree *tree,
	int n_ancestor, int *child_pos)
{
	int i, n;
	int sequence = 0;
	int block;

	block = isl_printer_get_yaml_style(p) == ISL_YAML_STYLE_BLOCK;

	p = isl_printer_yaml_start_mapping(p);
	if (n_ancestor == 0 && block) {
		p = isl_printer_print_str(p, "# YOU ARE HERE");
		p = isl_printer_end_line(p);
		p = isl_printer_start_line(p);
	}
	switch (tree->type) {
	case isl_schedule_node_error:
		p = isl_printer_print_str(p, "ERROR");
		break;
	case isl_schedule_node_leaf:
		p = isl_printer_print_str(p, "leaf");
		break;
	case isl_schedule_node_sequence:
		p = isl_printer_print_str(p, "sequence");
		sequence = 1;
		break;
	case isl_schedule_node_set:
		p = isl_printer_print_str(p, "set");
		sequence = 1;
		break;
	case isl_schedule_node_context:
		p = isl_printer_print_str(p, "context");
		p = isl_printer_yaml_next(p);
		p = isl_printer_print_str(p, "\"");
		p = isl_printer_print_set(p, tree->context);
		p = isl_printer_print_str(p, "\"");
		break;
	case isl_schedule_node_domain:
		p = isl_printer_print_str(p, "domain");
		p = isl_printer_yaml_next(p);
		p = isl_printer_print_str(p, "\"");
		p = isl_printer_print_union_set(p, tree->domain);
		p = isl_printer_print_str(p, "\"");
		break;
	case isl_schedule_node_expansion:
		p = isl_printer_print_str(p, "contraction");
		p = isl_printer_yaml_next(p);
		p = isl_printer_print_str(p, "\"");
		p = isl_printer_print_union_pw_multi_aff(p, tree->contraction);
		p = isl_printer_print_str(p, "\"");
		p = isl_printer_yaml_next(p);
		p = isl_printer_print_str(p, "expansion");
		p = isl_printer_yaml_next(p);
		p = isl_printer_print_str(p, "\"");
		p = isl_printer_print_union_map(p, tree->expansion);
		p = isl_printer_print_str(p, "\"");
		break;
	case isl_schedule_node_extension:
		p = isl_printer_print_str(p, "extension");
		p = isl_printer_yaml_next(p);
		p = isl_printer_print_str(p, "\"");
		p = isl_printer_print_union_map(p, tree->extension);
		p = isl_printer_print_str(p, "\"");
		break;
	case isl_schedule_node_filter:
		p = isl_printer_print_str(p, "filter");
		p = isl_printer_yaml_next(p);
		p = isl_printer_print_str(p, "\"");
		p = isl_printer_print_union_set(p, tree->filter);
		p = isl_printer_print_str(p, "\"");
		break;
	case isl_schedule_node_guard:
		p = isl_printer_print_str(p, "guard");
		p = isl_printer_yaml_next(p);
		p = isl_printer_print_str(p, "\"");
		p = isl_printer_print_set(p, tree->guard);
		p = isl_printer_print_str(p, "\"");
		break;
	case isl_schedule_node_mark:
		p = isl_printer_print_str(p, "mark");
		p = isl_printer_yaml_next(p);
		p = isl_printer_print_str(p, "\"");
		p = isl_printer_print_str(p, isl_id_get_name(tree->mark));
		p = isl_printer_print_str(p, "\"");
		break;
	case isl_schedule_node_band:
		p = print_tree_band(p, tree->band);
		break;
	}
	p = isl_printer_yaml_next(p);

	if (!tree->children) {
		if (n_ancestor > 0 && block) {
			isl_schedule_tree *leaf;

			p = isl_printer_print_str(p, "child");
			p = isl_printer_yaml_next(p);
			leaf = isl_schedule_tree_leaf(isl_printer_get_ctx(p));
			p = isl_printer_print_schedule_tree_mark(p,
					leaf, 0, NULL);
			isl_schedule_tree_free(leaf);
			p = isl_printer_yaml_next(p);
		}
		return isl_printer_yaml_end_mapping(p);
	}

	if (sequence) {
		p = isl_printer_yaml_start_sequence(p);
	} else {
		p = isl_printer_print_str(p, "child");
		p = isl_printer_yaml_next(p);
	}

	n = isl_schedule_tree_list_n_schedule_tree(tree->children);
	for (i = 0; i < n; ++i) {
		isl_schedule_tree *t;

		t = isl_schedule_tree_get_child(tree, i);
		if (n_ancestor > 0 && child_pos[0] == i)
			p = isl_printer_print_schedule_tree_mark(p, t,
						n_ancestor - 1, child_pos + 1);
		else
			p = isl_printer_print_schedule_tree_mark(p, t,
						-1, NULL);
		isl_schedule_tree_free(t);

		p = isl_printer_yaml_next(p);
	}

	if (sequence)
		p = isl_printer_yaml_end_sequence(p);
	p = isl_printer_yaml_end_mapping(p);

	return p;
}

/* Print "tree" to "p".
 */
__isl_give isl_printer *isl_printer_print_schedule_tree(
	__isl_take isl_printer *p, __isl_keep isl_schedule_tree *tree)
{
	return isl_printer_print_schedule_tree_mark(p, tree, -1, NULL);
}

void isl_schedule_tree_dump(__isl_keep isl_schedule_tree *tree)
{
	isl_ctx *ctx;
	isl_printer *printer;

	if (!tree)
		return;

	ctx = isl_schedule_tree_get_ctx(tree);
	printer = isl_printer_to_file(ctx, stderr);
	printer = isl_printer_set_yaml_style(printer, ISL_YAML_STYLE_BLOCK);
	printer = isl_printer_print_schedule_tree(printer, tree);

	isl_printer_free(printer);
}
