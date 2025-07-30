/*
 * Copyright 2012-2013 Ecole Normale Superieure
 * Copyright 2014      INRIA Rocquencourt
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 * and Inria Paris - Rocquencourt, Domaine de Voluceau - Rocquencourt,
 * B.P. 105 - 78153 Le Chesnay, France
 */

#include <isl/map.h>
#include <isl/aff.h>
#include <isl/constraint.h>
#include <isl/map.h>
#include <isl/union_set.h>
#include <isl/union_map.h>
#include <isl_ast_build_private.h>
#include <isl_ast_private.h>
#include <isl_config.h>

/* Construct a map that isolates the current dimension.
 *
 * Essentially, the current dimension of "set" is moved to the single output
 * dimension in the result, with the current dimension in the domain replaced
 * by an unconstrained variable.
 */
__isl_give isl_map *isl_ast_build_map_to_iterator(
	__isl_keep isl_ast_build *build, __isl_take isl_set *set)
{
	isl_map *map;

	map = isl_map_from_domain(set);
	map = isl_map_add_dims(map, isl_dim_out, 1);

	if (!build)
		return isl_map_free(map);

	map = isl_map_equate(map, isl_dim_in, build->depth, isl_dim_out, 0);
	map = isl_map_eliminate(map, isl_dim_in, build->depth, 1);

	return map;
}

/* Initialize the information derived during the AST generation to default
 * values for a schedule domain in "space".
 *
 * We also check that the remaining fields are not NULL so that
 * the calling functions don't have to perform this test.
 */
static __isl_give isl_ast_build *isl_ast_build_init_derived(
	__isl_take isl_ast_build *build, __isl_take isl_space *space)
{
	isl_ctx *ctx;
	isl_vec *strides;

	build = isl_ast_build_cow(build);
	if (!build || !build->domain)
		goto error;

	ctx = isl_ast_build_get_ctx(build);
	strides = isl_vec_alloc(ctx, isl_space_dim(space, isl_dim_set));
	strides = isl_vec_set_si(strides, 1);

	isl_vec_free(build->strides);
	build->strides = strides;

	space = isl_space_map_from_set(space);
	isl_multi_aff_free(build->offsets);
	build->offsets = isl_multi_aff_zero(isl_space_copy(space));
	isl_multi_aff_free(build->values);
	build->values = isl_multi_aff_identity(isl_space_copy(space));
	isl_multi_aff_free(build->internal2input);
	build->internal2input = isl_multi_aff_identity(space);

	if (!build->iterators || !build->domain || !build->generated ||
	    !build->pending || !build->values || !build->internal2input ||
	    !build->strides || !build->offsets || !build->options)
		return isl_ast_build_free(build);

	return build;
error:
	isl_space_free(space);
	return isl_ast_build_free(build);
}

/* Return an isl_id called "c%d", with "%d" set to "i".
 * If an isl_id with such a name already appears among the parameters
 * in build->domain, then adjust the name to "c%d_%d".
 */
static __isl_give isl_id *generate_name(isl_ctx *ctx, int i,
	__isl_keep isl_ast_build *build)
{
	int j;
	char name[16];
	isl_set *dom = build->domain;

	snprintf(name, sizeof(name), "c%d", i);
	j = 0;
	while (isl_set_find_dim_by_name(dom, isl_dim_param, name) >= 0)
		snprintf(name, sizeof(name), "c%d_%d", i, j++);
	return isl_id_alloc(ctx, name, NULL);
}

/* Create an isl_ast_build with "set" as domain.
 *
 * The input set is usually a parameter domain, but we currently allow it to
 * be any kind of set.  We set the domain of the returned isl_ast_build
 * to "set" and initialize all the other fields to default values.
 */
__isl_give isl_ast_build *isl_ast_build_from_context(__isl_take isl_set *set)
{
	int i, n;
	isl_ctx *ctx;
	isl_space *space;
	isl_ast_build *build;

	set = isl_set_compute_divs(set);
	if (!set)
		return NULL;

	ctx = isl_set_get_ctx(set);

	build = isl_calloc_type(ctx, isl_ast_build);
	if (!build)
		goto error;

	build->ref = 1;
	build->domain = set;
	build->generated = isl_set_copy(build->domain);
	build->pending = isl_set_universe(isl_set_get_space(build->domain));
	build->options = isl_union_map_empty(isl_space_params_alloc(ctx, 0));
	n = isl_set_dim(set, isl_dim_set);
	build->depth = n;
	build->iterators = isl_id_list_alloc(ctx, n);
	for (i = 0; i < n; ++i) {
		isl_id *id;
		if (isl_set_has_dim_id(set, isl_dim_set, i))
			id = isl_set_get_dim_id(set, isl_dim_set, i);
		else
			id = generate_name(ctx, i, build);
		build->iterators = isl_id_list_add(build->iterators, id);
	}
	space = isl_set_get_space(set);
	if (isl_space_is_params(space))
		space = isl_space_set_from_params(space);

	return isl_ast_build_init_derived(build, space);
error:
	isl_set_free(set);
	return NULL;
}

/* Create an isl_ast_build with a universe (parametric) context.
 */
__isl_give isl_ast_build *isl_ast_build_alloc(isl_ctx *ctx)
{
	isl_space *space;
	isl_set *context;

	space = isl_space_params_alloc(ctx, 0);
	context = isl_set_universe(space);

	return isl_ast_build_from_context(context);
}

__isl_give isl_ast_build *isl_ast_build_copy(__isl_keep isl_ast_build *build)
{
	if (!build)
		return NULL;

	build->ref++;
	return build;
}

__isl_give isl_ast_build *isl_ast_build_dup(__isl_keep isl_ast_build *build)
{
	isl_ctx *ctx;
	isl_ast_build *dup;

	if (!build)
		return NULL;

	ctx = isl_ast_build_get_ctx(build);
	dup = isl_calloc_type(ctx, isl_ast_build);
	if (!dup)
		return NULL;

	dup->ref = 1;
	dup->outer_pos = build->outer_pos;
	dup->depth = build->depth;
	dup->iterators = isl_id_list_copy(build->iterators);
	dup->domain = isl_set_copy(build->domain);
	dup->generated = isl_set_copy(build->generated);
	dup->pending = isl_set_copy(build->pending);
	dup->values = isl_multi_aff_copy(build->values);
	dup->internal2input = isl_multi_aff_copy(build->internal2input);
	dup->value = isl_pw_aff_copy(build->value);
	dup->strides = isl_vec_copy(build->strides);
	dup->offsets = isl_multi_aff_copy(build->offsets);
	dup->executed = isl_union_map_copy(build->executed);
	dup->single_valued = build->single_valued;
	dup->options = isl_union_map_copy(build->options);
	dup->at_each_domain = build->at_each_domain;
	dup->at_each_domain_user = build->at_each_domain_user;
	dup->before_each_for = build->before_each_for;
	dup->before_each_for_user = build->before_each_for_user;
	dup->after_each_for = build->after_each_for;
	dup->after_each_for_user = build->after_each_for_user;
	dup->before_each_mark = build->before_each_mark;
	dup->before_each_mark_user = build->before_each_mark_user;
	dup->after_each_mark = build->after_each_mark;
	dup->after_each_mark_user = build->after_each_mark_user;
	dup->create_leaf = build->create_leaf;
	dup->create_leaf_user = build->create_leaf_user;
	dup->node = isl_schedule_node_copy(build->node);
	if (build->loop_type) {
		int i;

		dup->n = build->n;
		dup->loop_type = isl_alloc_array(ctx,
						enum isl_ast_loop_type, dup->n);
		if (dup->n && !dup->loop_type)
			return isl_ast_build_free(dup);
		for (i = 0; i < dup->n; ++i)
			dup->loop_type[i] = build->loop_type[i];
	}

	if (!dup->iterators || !dup->domain || !dup->generated ||
	    !dup->pending || !dup->values ||
	    !dup->strides || !dup->offsets || !dup->options ||
	    (build->internal2input && !dup->internal2input) ||
	    (build->executed && !dup->executed) ||
	    (build->value && !dup->value) ||
	    (build->node && !dup->node))
		return isl_ast_build_free(dup);

	return dup;
}

/* Align the parameters of "build" to those of "model", introducing
 * additional parameters if needed.
 */
__isl_give isl_ast_build *isl_ast_build_align_params(
	__isl_take isl_ast_build *build, __isl_take isl_space *model)
{
	build = isl_ast_build_cow(build);
	if (!build)
		goto error;

	build->domain = isl_set_align_params(build->domain,
						isl_space_copy(model));
	build->generated = isl_set_align_params(build->generated,
						isl_space_copy(model));
	build->pending = isl_set_align_params(build->pending,
						isl_space_copy(model));
	build->values = isl_multi_aff_align_params(build->values,
						isl_space_copy(model));
	build->offsets = isl_multi_aff_align_params(build->offsets,
						isl_space_copy(model));
	build->options = isl_union_map_align_params(build->options,
						isl_space_copy(model));
	if (build->internal2input) {
		build->internal2input =
			isl_multi_aff_align_params(build->internal2input,
						model);
		if (!build->internal2input)
			return isl_ast_build_free(build);
	} else {
		isl_space_free(model);
	}

	if (!build->domain || !build->values || !build->offsets ||
	    !build->options)
		return isl_ast_build_free(build);

	return build;
error:
	isl_space_free(model);
	return NULL;
}

__isl_give isl_ast_build *isl_ast_build_cow(__isl_take isl_ast_build *build)
{
	if (!build)
		return NULL;

	if (build->ref == 1)
		return build;
	build->ref--;
	return isl_ast_build_dup(build);
}

__isl_null isl_ast_build *isl_ast_build_free(
	__isl_take isl_ast_build *build)
{
	if (!build)
		return NULL;

	if (--build->ref > 0)
		return NULL;

	isl_id_list_free(build->iterators);
	isl_set_free(build->domain);
	isl_set_free(build->generated);
	isl_set_free(build->pending);
	isl_multi_aff_free(build->values);
	isl_multi_aff_free(build->internal2input);
	isl_pw_aff_free(build->value);
	isl_vec_free(build->strides);
	isl_multi_aff_free(build->offsets);
	isl_multi_aff_free(build->schedule_map);
	isl_union_map_free(build->executed);
	isl_union_map_free(build->options);
	isl_schedule_node_free(build->node);
	free(build->loop_type);
	isl_set_free(build->isolated);

	free(build);

	return NULL;
}

isl_ctx *isl_ast_build_get_ctx(__isl_keep isl_ast_build *build)
{
	return build ? isl_set_get_ctx(build->domain) : NULL;
}

/* Replace build->options by "options".
 */
__isl_give isl_ast_build *isl_ast_build_set_options(
	__isl_take isl_ast_build *build, __isl_take isl_union_map *options)
{
	build = isl_ast_build_cow(build);

	if (!build || !options)
		goto error;

	isl_union_map_free(build->options);
	build->options = options;

	return build;
error:
	isl_union_map_free(options);
	return isl_ast_build_free(build);
}

/* Set the iterators for the next code generation.
 *
 * If we still have some iterators left from the previous code generation
 * (if any) or if iterators have already been set by a previous
 * call to this function, then we remove them first.
 */
__isl_give isl_ast_build *isl_ast_build_set_iterators(
	__isl_take isl_ast_build *build, __isl_take isl_id_list *iterators)
{
	int dim, n_it;

	build = isl_ast_build_cow(build);
	if (!build)
		goto error;

	dim = isl_set_dim(build->domain, isl_dim_set);
	n_it = isl_id_list_n_id(build->iterators);
	if (n_it < dim)
		isl_die(isl_ast_build_get_ctx(build), isl_error_internal,
			"isl_ast_build in inconsistent state", goto error);
	if (n_it > dim)
		build->iterators = isl_id_list_drop(build->iterators,
							dim, n_it - dim);
	build->iterators = isl_id_list_concat(build->iterators, iterators);
	if (!build->iterators)
		return isl_ast_build_free(build);

	return build;
error:
	isl_id_list_free(iterators);
	return isl_ast_build_free(build);
}

/* Set the "at_each_domain" callback of "build" to "fn".
 */
__isl_give isl_ast_build *isl_ast_build_set_at_each_domain(
	__isl_take isl_ast_build *build,
	__isl_give isl_ast_node *(*fn)(__isl_take isl_ast_node *node,
		__isl_keep isl_ast_build *build, void *user), void *user)
{
	build = isl_ast_build_cow(build);

	if (!build)
		return NULL;

	build->at_each_domain = fn;
	build->at_each_domain_user = user;

	return build;
}

/* Set the "before_each_for" callback of "build" to "fn".
 */
__isl_give isl_ast_build *isl_ast_build_set_before_each_for(
	__isl_take isl_ast_build *build,
	__isl_give isl_id *(*fn)(__isl_keep isl_ast_build *build,
		void *user), void *user)
{
	build = isl_ast_build_cow(build);

	if (!build)
		return NULL;

	build->before_each_for = fn;
	build->before_each_for_user = user;

	return build;
}

/* Set the "after_each_for" callback of "build" to "fn".
 */
__isl_give isl_ast_build *isl_ast_build_set_after_each_for(
	__isl_take isl_ast_build *build,
	__isl_give isl_ast_node *(*fn)(__isl_take isl_ast_node *node,
		__isl_keep isl_ast_build *build, void *user), void *user)
{
	build = isl_ast_build_cow(build);

	if (!build)
		return NULL;

	build->after_each_for = fn;
	build->after_each_for_user = user;

	return build;
}

/* Set the "before_each_mark" callback of "build" to "fn".
 */
__isl_give isl_ast_build *isl_ast_build_set_before_each_mark(
	__isl_take isl_ast_build *build,
	isl_stat (*fn)(__isl_keep isl_id *mark, __isl_keep isl_ast_build *build,
		void *user), void *user)
{
	build = isl_ast_build_cow(build);

	if (!build)
		return NULL;

	build->before_each_mark = fn;
	build->before_each_mark_user = user;

	return build;
}

/* Set the "after_each_mark" callback of "build" to "fn".
 */
__isl_give isl_ast_build *isl_ast_build_set_after_each_mark(
	__isl_take isl_ast_build *build,
	__isl_give isl_ast_node *(*fn)(__isl_take isl_ast_node *node,
		__isl_keep isl_ast_build *build, void *user), void *user)
{
	build = isl_ast_build_cow(build);

	if (!build)
		return NULL;

	build->after_each_mark = fn;
	build->after_each_mark_user = user;

	return build;
}

/* Set the "create_leaf" callback of "build" to "fn".
 */
__isl_give isl_ast_build *isl_ast_build_set_create_leaf(
	__isl_take isl_ast_build *build,
	__isl_give isl_ast_node *(*fn)(__isl_take isl_ast_build *build,
		void *user), void *user)
{
	build = isl_ast_build_cow(build);

	if (!build)
		return NULL;

	build->create_leaf = fn;
	build->create_leaf_user = user;

	return build;
}

/* Clear all information that is specific to this code generation
 * and that is (probably) not meaningful to any nested code generation.
 */
__isl_give isl_ast_build *isl_ast_build_clear_local_info(
	__isl_take isl_ast_build *build)
{
	isl_space *space;

	build = isl_ast_build_cow(build);
	if (!build)
		return NULL;

	space = isl_union_map_get_space(build->options);
	isl_union_map_free(build->options);
	build->options = isl_union_map_empty(space);

	build->at_each_domain = NULL;
	build->at_each_domain_user = NULL;
	build->before_each_for = NULL;
	build->before_each_for_user = NULL;
	build->after_each_for = NULL;
	build->after_each_for_user = NULL;
	build->before_each_mark = NULL;
	build->before_each_mark_user = NULL;
	build->after_each_mark = NULL;
	build->after_each_mark_user = NULL;
	build->create_leaf = NULL;
	build->create_leaf_user = NULL;

	if (!build->options)
		return isl_ast_build_free(build);

	return build;
}

/* Have any loops been eliminated?
 * That is, do any of the original schedule dimensions have a fixed
 * value that has been substituted?
 */
static int any_eliminated(isl_ast_build *build)
{
	int i;

	for (i = 0; i < build->depth; ++i)
		if (isl_ast_build_has_affine_value(build, i))
			return 1;

	return 0;
}

/* Clear build->schedule_map.
 * This function should be called whenever anything that might affect
 * the result of isl_ast_build_get_schedule_map_multi_aff changes.
 * In particular, it should be called when the depth is changed or
 * when an iterator is determined to have a fixed value.
 */
static void isl_ast_build_reset_schedule_map(__isl_keep isl_ast_build *build)
{
	if (!build)
		return;
	isl_multi_aff_free(build->schedule_map);
	build->schedule_map = NULL;
}

/* Do we need a (non-trivial) schedule map?
 * That is, is the internal schedule space different from
 * the external schedule space?
 *
 * The internal and external schedule spaces are only the same
 * if code has been generated for the entire schedule and if none
 * of the loops have been eliminated.
 */
__isl_give int isl_ast_build_need_schedule_map(__isl_keep isl_ast_build *build)
{
	int dim;

	if (!build)
		return -1;

	dim = isl_set_dim(build->domain, isl_dim_set);
	return build->depth != dim || any_eliminated(build);
}

/* Return a mapping from the internal schedule space to the external
 * schedule space in the form of an isl_multi_aff.
 * The internal schedule space originally corresponds to that of the
 * input schedule.  This may change during the code generation if
 * if isl_ast_build_insert_dim is ever called.
 * The external schedule space corresponds to the
 * loops that have been generated.
 *
 * Currently, the only difference between the internal schedule domain
 * and the external schedule domain is that some dimensions are projected
 * out in the external schedule domain.  In particular, the dimensions
 * for which no code has been generated yet and the dimensions that correspond
 * to eliminated loops.
 *
 * We cache a copy of the schedule_map in build->schedule_map.
 * The cache is cleared through isl_ast_build_reset_schedule_map
 * whenever anything changes that might affect the result of this function.
 */
__isl_give isl_multi_aff *isl_ast_build_get_schedule_map_multi_aff(
	__isl_keep isl_ast_build *build)
{
	isl_space *space;
	isl_multi_aff *ma;

	if (!build)
		return NULL;
	if (build->schedule_map)
		return isl_multi_aff_copy(build->schedule_map);

	space = isl_ast_build_get_space(build, 1);
	space = isl_space_map_from_set(space);
	ma = isl_multi_aff_identity(space);
	if (isl_ast_build_need_schedule_map(build)) {
		int i;
		int dim = isl_set_dim(build->domain, isl_dim_set);
		ma = isl_multi_aff_drop_dims(ma, isl_dim_out,
					build->depth, dim - build->depth);
		for (i = build->depth - 1; i >= 0; --i)
			if (isl_ast_build_has_affine_value(build, i))
				ma = isl_multi_aff_drop_dims(ma,
							isl_dim_out, i, 1);
	}

	build->schedule_map = ma;
	return isl_multi_aff_copy(build->schedule_map);
}

/* Return a mapping from the internal schedule space to the external
 * schedule space in the form of an isl_map.
 */
__isl_give isl_map *isl_ast_build_get_schedule_map(
	__isl_keep isl_ast_build *build)
{
	isl_multi_aff *ma;

	ma = isl_ast_build_get_schedule_map_multi_aff(build);
	return isl_map_from_multi_aff(ma);
}

/* Return the position of the dimension in build->domain for which
 * an AST node is currently being generated.
 */
int isl_ast_build_get_depth(__isl_keep isl_ast_build *build)
{
	return build ? build->depth : -1;
}

/* Prepare for generating code for the next level.
 * In particular, increase the depth and reset any information
 * that is local to the current depth.
 */
__isl_give isl_ast_build *isl_ast_build_increase_depth(
	__isl_take isl_ast_build *build)
{
	build = isl_ast_build_cow(build);
	if (!build)
		return NULL;
	build->depth++;
	isl_ast_build_reset_schedule_map(build);
	build->value = isl_pw_aff_free(build->value);
	return build;
}

void isl_ast_build_dump(__isl_keep isl_ast_build *build)
{
	if (!build)
		return;

	fprintf(stderr, "domain: ");
	isl_set_dump(build->domain);
	fprintf(stderr, "generated: ");
	isl_set_dump(build->generated);
	fprintf(stderr, "pending: ");
	isl_set_dump(build->pending);
	fprintf(stderr, "iterators: ");
	isl_id_list_dump(build->iterators);
	fprintf(stderr, "values: ");
	isl_multi_aff_dump(build->values);
	if (build->value) {
		fprintf(stderr, "value: ");
		isl_pw_aff_dump(build->value);
	}
	fprintf(stderr, "strides: ");
	isl_vec_dump(build->strides);
	fprintf(stderr, "offsets: ");
	isl_multi_aff_dump(build->offsets);
	fprintf(stderr, "internal2input: ");
	isl_multi_aff_dump(build->internal2input);
}

/* Initialize "build" for AST construction in schedule space "space"
 * in the case that build->domain is a parameter set.
 *
 * build->iterators is assumed to have been updated already.
 */
static __isl_give isl_ast_build *isl_ast_build_init(
	__isl_take isl_ast_build *build, __isl_take isl_space *space)
{
	isl_set *set;

	build = isl_ast_build_cow(build);
	if (!build)
		goto error;

	set = isl_set_universe(isl_space_copy(space));
	build->domain = isl_set_intersect_params(isl_set_copy(set),
						    build->domain);
	build->pending = isl_set_intersect_params(isl_set_copy(set),
						    build->pending);
	build->generated = isl_set_intersect_params(set, build->generated);

	return isl_ast_build_init_derived(build, space);
error:
	isl_ast_build_free(build);
	isl_space_free(space);
	return NULL;
}

/* Assign "aff" to *user and return -1, effectively extracting
 * the first (and presumably only) affine expression in the isl_pw_aff
 * on which this function is used.
 */
static isl_stat extract_single_piece(__isl_take isl_set *set,
	__isl_take isl_aff *aff, void *user)
{
	isl_aff **p = user;

	*p = aff;
	isl_set_free(set);

	return isl_stat_error;
}

/* Intersect "set" with the stride constraint of "build", if any.
 */
static __isl_give isl_set *intersect_stride_constraint(__isl_take isl_set *set,
	__isl_keep isl_ast_build *build)
{
	isl_set *stride;

	if (!build)
		return isl_set_free(set);
	if (!isl_ast_build_has_stride(build, build->depth))
		return set;

	stride = isl_ast_build_get_stride_constraint(build);
	return isl_set_intersect(set, stride);
}

/* Check if the given bounds on the current dimension (together with
 * the stride constraint, if any) imply that
 * this current dimension attains only a single value (in terms of
 * parameters and outer dimensions).
 * If so, we record it in build->value.
 * If, moreover, this value can be represented as a single affine expression,
 * then we also update build->values, effectively marking the current
 * dimension as "eliminated".
 *
 * When computing the gist of the fixed value that can be represented
 * as a single affine expression, it is important to only take into
 * account the domain constraints in the original AST build and
 * not the domain of the affine expression itself.
 * Otherwise, a [i/3] is changed into a i/3 because we know that i
 * is a multiple of 3, but then we end up not expressing anywhere
 * in the context that i is a multiple of 3.
 */
static __isl_give isl_ast_build *update_values(
	__isl_take isl_ast_build *build, __isl_take isl_basic_set *bounds)
{
	int sv;
	isl_pw_multi_aff *pma;
	isl_aff *aff = NULL;
	isl_map *it_map;
	isl_set *set;

	set = isl_set_from_basic_set(bounds);
	set = isl_set_intersect(set, isl_set_copy(build->domain));
	set = intersect_stride_constraint(set, build);
	it_map = isl_ast_build_map_to_iterator(build, set);

	sv = isl_map_is_single_valued(it_map);
	if (sv < 0)
		build = isl_ast_build_free(build);
	if (!build || !sv) {
		isl_map_free(it_map);
		return build;
	}

	pma = isl_pw_multi_aff_from_map(it_map);
	build->value = isl_pw_multi_aff_get_pw_aff(pma, 0);
	build->value = isl_ast_build_compute_gist_pw_aff(build, build->value);
	build->value = isl_pw_aff_coalesce(build->value);
	isl_pw_multi_aff_free(pma);

	if (!build->value)
		return isl_ast_build_free(build);

	if (isl_pw_aff_n_piece(build->value) != 1)
		return build;

	isl_pw_aff_foreach_piece(build->value, &extract_single_piece, &aff);

	build->values = isl_multi_aff_set_aff(build->values, build->depth, aff);
	if (!build->values)
		return isl_ast_build_free(build);
	isl_ast_build_reset_schedule_map(build);
	return build;
}

/* Update the AST build based on the given loop bounds for
 * the current dimension and the stride information available in the build.
 *
 * We first make sure that the bounds do not refer to any iterators
 * that have already been eliminated.
 * Then, we check if the bounds imply that the current iterator
 * has a fixed value.
 * If they do and if this fixed value can be expressed as a single
 * affine expression, we eliminate the iterators from the bounds.
 * Note that we cannot simply plug in this single value using
 * isl_basic_set_preimage_multi_aff as the single value may only
 * be defined on a subset of the domain.  Plugging in the value
 * would restrict the build domain to this subset, while this
 * restriction may not be reflected in the generated code.
 * Finally, we intersect build->domain with the updated bounds.
 * We also add the stride constraint unless we have been able
 * to find a fixed value expressed as a single affine expression.
 *
 * Note that the check for a fixed value in update_values requires
 * us to intersect the bounds with the current build domain.
 * When we intersect build->domain with the updated bounds in
 * the final step, we make sure that these updated bounds have
 * not been intersected with the old build->domain.
 * Otherwise, we would indirectly intersect the build domain with itself,
 * which can lead to inefficiencies, in particular if the build domain
 * contains any unknown divs.
 *
 * The pending and generated sets are not updated by this function to
 * match the updated domain.
 * The caller still needs to call isl_ast_build_set_pending_generated.
 */
__isl_give isl_ast_build *isl_ast_build_set_loop_bounds(
	__isl_take isl_ast_build *build, __isl_take isl_basic_set *bounds)
{
	isl_set *set;

	build = isl_ast_build_cow(build);
	if (!build)
		goto error;

	build = update_values(build, isl_basic_set_copy(bounds));
	if (!build)
		goto error;
	set = isl_set_from_basic_set(bounds);
	if (isl_ast_build_has_affine_value(build, build->depth)) {
		set = isl_set_eliminate(set, isl_dim_set, build->depth, 1);
		set = isl_set_compute_divs(set);
		build->pending = isl_set_intersect(build->pending,
							isl_set_copy(set));
		build->domain = isl_set_intersect(build->domain, set);
	} else {
		build->domain = isl_set_intersect(build->domain, set);
		build = isl_ast_build_include_stride(build);
		if (!build)
			goto error;
	}

	if (!build->domain || !build->pending || !build->generated)
		return isl_ast_build_free(build);

	return build;
error:
	isl_ast_build_free(build);
	isl_basic_set_free(bounds);
	return NULL;
}

/* Update the pending and generated sets of "build" according to "bounds".
 * If the build has an affine value at the current depth,
 * then isl_ast_build_set_loop_bounds has already set the pending set.
 * Otherwise, do it here.
 */
__isl_give isl_ast_build *isl_ast_build_set_pending_generated(
	__isl_take isl_ast_build *build, __isl_take isl_basic_set *bounds)
{
	isl_basic_set *generated, *pending;

	if (!build)
		goto error;

	if (isl_ast_build_has_affine_value(build, build->depth)) {
		isl_basic_set_free(bounds);
		return build;
	}

	build = isl_ast_build_cow(build);
	if (!build)
		goto error;

	pending = isl_basic_set_copy(bounds);
	pending = isl_basic_set_drop_constraints_involving_dims(pending,
				isl_dim_set, build->depth, 1);
	build->pending = isl_set_intersect(build->pending,
				isl_set_from_basic_set(pending));
	generated = bounds;
	generated = isl_basic_set_drop_constraints_not_involving_dims(
			    generated, isl_dim_set, build->depth, 1);
	build->generated = isl_set_intersect(build->generated,
				isl_set_from_basic_set(generated));

	if (!build->pending || !build->generated)
		return isl_ast_build_free(build);

	return build;
error:
	isl_ast_build_free(build);
	isl_basic_set_free(bounds);
	return NULL;
}

/* Intersect build->domain with "set", where "set" is specified
 * in terms of the internal schedule domain.
 */
static __isl_give isl_ast_build *isl_ast_build_restrict_internal(
	__isl_take isl_ast_build *build, __isl_take isl_set *set)
{
	build = isl_ast_build_cow(build);
	if (!build)
		goto error;

	set = isl_set_compute_divs(set);
	build->domain = isl_set_intersect(build->domain, set);
	build->domain = isl_set_coalesce(build->domain);

	if (!build->domain)
		return isl_ast_build_free(build);

	return build;
error:
	isl_ast_build_free(build);
	isl_set_free(set);
	return NULL;
}

/* Intersect build->generated and build->domain with "set",
 * where "set" is specified in terms of the internal schedule domain.
 */
__isl_give isl_ast_build *isl_ast_build_restrict_generated(
	__isl_take isl_ast_build *build, __isl_take isl_set *set)
{
	set = isl_set_compute_divs(set);
	build = isl_ast_build_restrict_internal(build, isl_set_copy(set));
	build = isl_ast_build_cow(build);
	if (!build)
		goto error;

	build->generated = isl_set_intersect(build->generated, set);
	build->generated = isl_set_coalesce(build->generated);

	if (!build->generated)
		return isl_ast_build_free(build);

	return build;
error:
	isl_ast_build_free(build);
	isl_set_free(set);
	return NULL;
}

/* Replace the set of pending constraints by "guard", which is then
 * no longer considered as pending.
 * That is, add "guard" to the generated constraints and clear all pending
 * constraints, making the domain equal to the generated constraints.
 */
__isl_give isl_ast_build *isl_ast_build_replace_pending_by_guard(
	__isl_take isl_ast_build *build, __isl_take isl_set *guard)
{
	build = isl_ast_build_restrict_generated(build, guard);
	build = isl_ast_build_cow(build);
	if (!build)
		return NULL;

	isl_set_free(build->domain);
	build->domain = isl_set_copy(build->generated);
	isl_set_free(build->pending);
	build->pending = isl_set_universe(isl_set_get_space(build->domain));

	if (!build->pending)
		return isl_ast_build_free(build);

	return build;
}

/* Intersect build->domain with "set", where "set" is specified
 * in terms of the external schedule domain.
 */
__isl_give isl_ast_build *isl_ast_build_restrict(
	__isl_take isl_ast_build *build, __isl_take isl_set *set)
{
	if (isl_set_is_params(set))
		return isl_ast_build_restrict_generated(build, set);

	if (isl_ast_build_need_schedule_map(build)) {
		isl_multi_aff *ma;
		ma = isl_ast_build_get_schedule_map_multi_aff(build);
		set = isl_set_preimage_multi_aff(set, ma);
	}
	return isl_ast_build_restrict_generated(build, set);
}

/* Replace build->executed by "executed".
 */
__isl_give isl_ast_build *isl_ast_build_set_executed(
	__isl_take isl_ast_build *build, __isl_take isl_union_map *executed)
{
	build = isl_ast_build_cow(build);
	if (!build)
		goto error;

	isl_union_map_free(build->executed);
	build->executed = executed;

	return build;
error:
	isl_ast_build_free(build);
	isl_union_map_free(executed);
	return NULL;
}

/* Does "build" point to a band node?
 * That is, are we currently handling a band node inside a schedule tree?
 */
int isl_ast_build_has_schedule_node(__isl_keep isl_ast_build *build)
{
	if (!build)
		return -1;
	return build->node != NULL;
}

/* Return a copy of the band node that "build" refers to.
 */
__isl_give isl_schedule_node *isl_ast_build_get_schedule_node(
	__isl_keep isl_ast_build *build)
{
	if (!build)
		return NULL;
	return isl_schedule_node_copy(build->node);
}

/* Extract the loop AST generation types for the members of build->node
 * and store them in build->loop_type.
 */
static __isl_give isl_ast_build *extract_loop_types(
	__isl_take isl_ast_build *build)
{
	int i;
	isl_ctx *ctx;
	isl_schedule_node *node;

	if (!build)
		return NULL;
	ctx = isl_ast_build_get_ctx(build);
	if (!build->node)
		isl_die(ctx, isl_error_internal, "missing AST node",
			return isl_ast_build_free(build));

	free(build->loop_type);
	build->n = isl_schedule_node_band_n_member(build->node);
	build->loop_type = isl_alloc_array(ctx,
					    enum isl_ast_loop_type, build->n);
	if (build->n && !build->loop_type)
		return isl_ast_build_free(build);
	node = build->node;
	for (i = 0; i < build->n; ++i)
		build->loop_type[i] =
		    isl_schedule_node_band_member_get_ast_loop_type(node, i);

	return build;
}

/* Replace the band node that "build" refers to by "node" and
 * extract the corresponding loop AST generation types.
 */
__isl_give isl_ast_build *isl_ast_build_set_schedule_node(
	__isl_take isl_ast_build *build,
	__isl_take isl_schedule_node *node)
{
	build = isl_ast_build_cow(build);
	if (!build || !node)
		goto error;

	isl_schedule_node_free(build->node);
	build->node = node;

	build = extract_loop_types(build);

	return build;
error:
	isl_ast_build_free(build);
	isl_schedule_node_free(node);
	return NULL;
}

/* Remove any reference to a band node from "build".
 */
__isl_give isl_ast_build *isl_ast_build_reset_schedule_node(
	__isl_take isl_ast_build *build)
{
	build = isl_ast_build_cow(build);
	if (!build)
		return NULL;

	isl_schedule_node_free(build->node);
	build->node = NULL;

	return build;
}

/* Return a copy of the current schedule domain.
 */
__isl_give isl_set *isl_ast_build_get_domain(__isl_keep isl_ast_build *build)
{
	return build ? isl_set_copy(build->domain) : NULL;
}

/* Return a copy of the set of pending constraints.
 */
__isl_give isl_set *isl_ast_build_get_pending(
	__isl_keep isl_ast_build *build)
{
	return build ? isl_set_copy(build->pending) : NULL;
}

/* Return a copy of the set of generated constraints.
 */
__isl_give isl_set *isl_ast_build_get_generated(
	__isl_keep isl_ast_build *build)
{
	return build ? isl_set_copy(build->generated) : NULL;
}

/* Return a copy of the map from the internal schedule domain
 * to the original input schedule domain.
 */
__isl_give isl_multi_aff *isl_ast_build_get_internal2input(
	__isl_keep isl_ast_build *build)
{
	return build ? isl_multi_aff_copy(build->internal2input) : NULL;
}

/* Return the number of variables of the given type
 * in the (internal) schedule space.
 */
unsigned isl_ast_build_dim(__isl_keep isl_ast_build *build,
	enum isl_dim_type type)
{
	if (!build)
		return 0;
	return isl_set_dim(build->domain, type);
}

/* Return the (schedule) space of "build".
 *
 * If "internal" is set, then this space is the space of the internal
 * representation of the entire schedule, including those parts for
 * which no code has been generated yet.
 *
 * If "internal" is not set, then this space is the external representation
 * of the loops generated so far.
 */
__isl_give isl_space *isl_ast_build_get_space(__isl_keep isl_ast_build *build,
	int internal)
{
	int i;
	int dim;
	isl_space *space;

	if (!build)
		return NULL;

	space = isl_set_get_space(build->domain);
	if (internal)
		return space;

	if (!isl_ast_build_need_schedule_map(build))
		return space;

	dim = isl_set_dim(build->domain, isl_dim_set);
	space = isl_space_drop_dims(space, isl_dim_set,
				    build->depth, dim - build->depth);
	for (i = build->depth - 1; i >= 0; --i)
		if (isl_ast_build_has_affine_value(build, i))
			space = isl_space_drop_dims(space, isl_dim_set, i, 1);

	return space;
}

/* Return the external representation of the schedule space of "build",
 * i.e., a space with a dimension for each loop generated so far,
 * with the names of the dimensions set to the loop iterators.
 */
__isl_give isl_space *isl_ast_build_get_schedule_space(
	__isl_keep isl_ast_build *build)
{
	isl_space *space;
	int i, skip;

	if (!build)
		return NULL;

	space = isl_ast_build_get_space(build, 0);

	skip = 0;
	for (i = 0; i < build->depth; ++i) {
		isl_id *id;

		if (isl_ast_build_has_affine_value(build, i)) {
			skip++;
			continue;
		}

		id = isl_ast_build_get_iterator_id(build, i);
		space = isl_space_set_dim_id(space, isl_dim_set, i - skip, id);
	}

	return space;
}

/* Return the current schedule, as stored in build->executed, in terms
 * of the external schedule domain.
 */
__isl_give isl_union_map *isl_ast_build_get_schedule(
	__isl_keep isl_ast_build *build)
{
	isl_union_map *executed;
	isl_union_map *schedule;

	if (!build)
		return NULL;

	executed = isl_union_map_copy(build->executed);
	if (isl_ast_build_need_schedule_map(build)) {
		isl_map *proj = isl_ast_build_get_schedule_map(build);
		executed = isl_union_map_apply_domain(executed,
					isl_union_map_from_map(proj));
	}
	schedule = isl_union_map_reverse(executed);

	return schedule;
}

/* Return the iterator attached to the internal schedule dimension "pos".
 */
__isl_give isl_id *isl_ast_build_get_iterator_id(
	__isl_keep isl_ast_build *build, int pos)
{
	if (!build)
		return NULL;

	return isl_id_list_get_id(build->iterators, pos);
}

/* Set the stride and offset of the current dimension to the given
 * value and expression.
 *
 * If we had already found a stride before, then the two strides
 * are combined into a single stride.
 *
 * In particular, if the new stride information is of the form
 *
 *	i = f + s (...)
 *
 * and the old stride information is of the form
 *
 *	i = f2 + s2 (...)
 *
 * then we compute the extended gcd of s and s2
 *
 *	a s + b s2 = g,
 *
 * with g = gcd(s,s2), multiply the first equation with t1 = b s2/g
 * and the second with t2 = a s1/g.
 * This results in
 *
 *	i = (b s2 + a s1)/g i = t1 f + t2 f2 + (s s2)/g (...)
 *
 * so that t1 f + t2 f2 is the combined offset and (s s2)/g = lcm(s,s2)
 * is the combined stride.
 */
static __isl_give isl_ast_build *set_stride(__isl_take isl_ast_build *build,
	__isl_take isl_val *stride, __isl_take isl_aff *offset)
{
	int pos;

	build = isl_ast_build_cow(build);
	if (!build || !stride || !offset)
		goto error;

	pos = build->depth;

	if (isl_ast_build_has_stride(build, pos)) {
		isl_val *stride2, *a, *b, *g;
		isl_aff *offset2;

		stride2 = isl_vec_get_element_val(build->strides, pos);
		g = isl_val_gcdext(isl_val_copy(stride), isl_val_copy(stride2),
					&a, &b);
		a = isl_val_mul(a, isl_val_copy(stride));
		a = isl_val_div(a, isl_val_copy(g));
		stride2 = isl_val_div(stride2, g);
		b = isl_val_mul(b, isl_val_copy(stride2));
		stride = isl_val_mul(stride, stride2);

		offset2 = isl_multi_aff_get_aff(build->offsets, pos);
		offset2 = isl_aff_scale_val(offset2, a);
		offset = isl_aff_scale_val(offset, b);
		offset = isl_aff_add(offset, offset2);
	}

	build->strides = isl_vec_set_element_val(build->strides, pos, stride);
	build->offsets = isl_multi_aff_set_aff(build->offsets, pos, offset);
	if (!build->strides || !build->offsets)
		return isl_ast_build_free(build);

	return build;
error:
	isl_val_free(stride);
	isl_aff_free(offset);
	return isl_ast_build_free(build);
}

/* Return a set expressing the stride constraint at the current depth.
 *
 * In particular, if the current iterator (i) is known to attain values
 *
 *	f + s a
 *
 * where f is the offset and s is the stride, then the returned set
 * expresses the constraint
 *
 *	(f - i) mod s = 0
 */
__isl_give isl_set *isl_ast_build_get_stride_constraint(
	__isl_keep isl_ast_build *build)
{
	isl_aff *aff;
	isl_set *set;
	isl_val *stride;
	int pos;

	if (!build)
		return NULL;

	pos = build->depth;

	if (!isl_ast_build_has_stride(build, pos))
		return isl_set_universe(isl_ast_build_get_space(build, 1));

	stride = isl_ast_build_get_stride(build, pos);
	aff = isl_ast_build_get_offset(build, pos);
	aff = isl_aff_add_coefficient_si(aff, isl_dim_in, pos, -1);
	aff = isl_aff_mod_val(aff, stride);
	set = isl_set_from_basic_set(isl_aff_zero_basic_set(aff));

	return set;
}

/* Return the expansion implied by the stride and offset at the current
 * depth.
 *
 * That is, return the mapping
 *
 *	[i_0, ..., i_{d-1}, i_d, i_{d+1}, ...]
 *		-> [i_0, ..., i_{d-1}, s * i_d + offset(i),  i_{d+1}, ...]
 *
 * where s is the stride at the current depth d and offset(i) is
 * the corresponding offset.
 */
__isl_give isl_multi_aff *isl_ast_build_get_stride_expansion(
	__isl_keep isl_ast_build *build)
{
	isl_space *space;
	isl_multi_aff *ma;
	int pos;
	isl_aff *aff, *offset;
	isl_val *stride;

	if (!build)
		return NULL;

	pos = isl_ast_build_get_depth(build);
	space = isl_ast_build_get_space(build, 1);
	space = isl_space_map_from_set(space);
	ma = isl_multi_aff_identity(space);

	if (!isl_ast_build_has_stride(build, pos))
		return ma;

	offset = isl_ast_build_get_offset(build, pos);
	stride = isl_ast_build_get_stride(build, pos);
	aff = isl_multi_aff_get_aff(ma, pos);
	aff = isl_aff_scale_val(aff, stride);
	aff = isl_aff_add(aff, offset);
	ma = isl_multi_aff_set_aff(ma, pos, aff);

	return ma;
}

/* Add constraints corresponding to any previously detected
 * stride on the current dimension to build->domain.
 */
__isl_give isl_ast_build *isl_ast_build_include_stride(
	__isl_take isl_ast_build *build)
{
	isl_set *set;

	if (!build)
		return NULL;
	if (!isl_ast_build_has_stride(build, build->depth))
		return build;
	build = isl_ast_build_cow(build);
	if (!build)
		return NULL;

	set = isl_ast_build_get_stride_constraint(build);

	build->domain = isl_set_intersect(build->domain, isl_set_copy(set));
	build->generated = isl_set_intersect(build->generated, set);
	if (!build->domain || !build->generated)
		return isl_ast_build_free(build);

	return build;
}

/* Information used inside detect_stride.
 *
 * "build" may be updated by detect_stride to include stride information.
 * "pos" is equal to build->depth.
 */
struct isl_detect_stride_data {
	isl_ast_build *build;
	int pos;
};

/* Check if constraint "c" imposes any stride on dimension data->pos
 * and, if so, update the stride information in data->build.
 *
 * In order to impose a stride on the dimension, "c" needs to be an equality
 * and it needs to involve the dimension.  Note that "c" may also be
 * a div constraint and thus an inequality that we cannot use.
 *
 * Let c be of the form
 *
 *	h(p) + g * v * i + g * stride * f(alpha) = 0
 *
 * with h(p) an expression in terms of the parameters and outer dimensions
 * and f(alpha) an expression in terms of the existentially quantified
 * variables.  Note that the inner dimensions have been eliminated so
 * they do not appear in "c".
 *
 * If "stride" is not zero and not one, then it represents a non-trivial stride
 * on "i".  We compute a and b such that
 *
 *	a v + b stride = 1
 *
 * We have
 *
 *	g v i = -h(p) + g stride f(alpha)
 *
 *	a g v i = -a h(p) + g stride f(alpha)
 *
 *	a g v i + b g stride i = -a h(p) + g stride * (...)
 *
 *	g i = -a h(p) + g stride * (...)
 *
 *	i = -a h(p)/g + stride * (...)
 *
 * The expression "-a h(p)/g" can therefore be used as offset.
 */
static isl_stat detect_stride(__isl_take isl_constraint *c, void *user)
{
	struct isl_detect_stride_data *data = user;
	int i, n_div;
	isl_ctx *ctx;
	isl_val *v, *stride, *m;

	if (!isl_constraint_is_equality(c) ||
	    !isl_constraint_involves_dims(c, isl_dim_set, data->pos, 1)) {
		isl_constraint_free(c);
		return isl_stat_ok;
	}

	ctx = isl_constraint_get_ctx(c);
	stride = isl_val_zero(ctx);
	n_div = isl_constraint_dim(c, isl_dim_div);
	for (i = 0; i < n_div; ++i) {
		v = isl_constraint_get_coefficient_val(c, isl_dim_div, i);
		stride = isl_val_gcd(stride, v);
	}

	v = isl_constraint_get_coefficient_val(c, isl_dim_set, data->pos);
	m = isl_val_gcd(isl_val_copy(stride), isl_val_copy(v));
	stride = isl_val_div(stride, isl_val_copy(m));
	v = isl_val_div(v, isl_val_copy(m));

	if (!isl_val_is_zero(stride) && !isl_val_is_one(stride)) {
		isl_aff *aff;
		isl_val *gcd, *a, *b;

		gcd = isl_val_gcdext(v, isl_val_copy(stride), &a, &b);
		isl_val_free(gcd);
		isl_val_free(b);

		aff = isl_constraint_get_aff(c);
		for (i = 0; i < n_div; ++i)
			aff = isl_aff_set_coefficient_si(aff,
							 isl_dim_div, i, 0);
		aff = isl_aff_set_coefficient_si(aff, isl_dim_in, data->pos, 0);
		a = isl_val_neg(a);
		aff = isl_aff_scale_val(aff, a);
		aff = isl_aff_scale_down_val(aff, m);
		data->build = set_stride(data->build, stride, aff);
	} else {
		isl_val_free(stride);
		isl_val_free(m);
		isl_val_free(v);
	}

	isl_constraint_free(c);
	return isl_stat_ok;
}

/* Check if the constraints in "set" imply any stride on the current
 * dimension and, if so, record the stride information in "build"
 * and return the updated "build".
 *
 * We compute the affine hull and then check if any of the constraints
 * in the hull imposes any stride on the current dimension.
 *
 * We assume that inner dimensions have been eliminated from "set"
 * by the caller.  This is needed because the common stride
 * may be imposed by different inner dimensions on different parts of
 * the domain.
 */
__isl_give isl_ast_build *isl_ast_build_detect_strides(
	__isl_take isl_ast_build *build, __isl_take isl_set *set)
{
	isl_basic_set *hull;
	struct isl_detect_stride_data data;

	if (!build)
		goto error;

	data.build = build;
	data.pos = isl_ast_build_get_depth(build);
	hull = isl_set_affine_hull(set);

	if (isl_basic_set_foreach_constraint(hull, &detect_stride, &data) < 0)
		data.build = isl_ast_build_free(data.build);

	isl_basic_set_free(hull);
	return data.build;
error:
	isl_set_free(set);
	return NULL;
}

struct isl_ast_build_involves_data {
	int depth;
	int involves;
};

/* Check if "map" involves the input dimension data->depth.
 */
static isl_stat involves_depth(__isl_take isl_map *map, void *user)
{
	struct isl_ast_build_involves_data *data = user;

	data->involves = isl_map_involves_dims(map, isl_dim_in, data->depth, 1);
	isl_map_free(map);

	if (data->involves < 0 || data->involves)
		return isl_stat_error;
	return isl_stat_ok;
}

/* Do any options depend on the value of the dimension at the current depth?
 */
int isl_ast_build_options_involve_depth(__isl_keep isl_ast_build *build)
{
	struct isl_ast_build_involves_data data;

	if (!build)
		return -1;

	data.depth = build->depth;
	data.involves = 0;

	if (isl_union_map_foreach_map(build->options,
					&involves_depth, &data) < 0) {
		if (data.involves < 0 || !data.involves)
			return -1;
	}

	return data.involves;
}

/* Construct the map
 *
 *	{ [i] -> [i] : i < pos; [i] -> [i + 1] : i >= pos }
 *
 * with "space" the parameter space of the constructed map.
 */
static __isl_give isl_map *construct_insertion_map(__isl_take isl_space *space,
	int pos)
{
	isl_constraint *c;
	isl_basic_map *bmap1, *bmap2;

	space = isl_space_set_from_params(space);
	space = isl_space_add_dims(space, isl_dim_set, 1);
	space = isl_space_map_from_set(space);
	c = isl_constraint_alloc_equality(isl_local_space_from_space(space));
	c = isl_constraint_set_coefficient_si(c, isl_dim_in, 0, 1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_out, 0, -1);
	bmap1 = isl_basic_map_from_constraint(isl_constraint_copy(c));
	c = isl_constraint_set_constant_si(c, 1);
	bmap2 = isl_basic_map_from_constraint(c);

	bmap1 = isl_basic_map_upper_bound_si(bmap1, isl_dim_in, 0, pos - 1);
	bmap2 = isl_basic_map_lower_bound_si(bmap2, isl_dim_in, 0, pos);

	return isl_basic_map_union(bmap1, bmap2);
}

static const char *option_str[] = {
	[isl_ast_loop_atomic] = "atomic",
	[isl_ast_loop_unroll] = "unroll",
	[isl_ast_loop_separate] = "separate"
};

/* Update the "options" to reflect the insertion of a dimension
 * at position "pos" in the schedule domain space.
 * "space" is the original domain space before the insertion and
 * may be named and/or structured.
 *
 * The (relevant) input options all have "space" as domain, which
 * has to be mapped to the extended space.
 * The values of the ranges also refer to the schedule domain positions
 * and they therefore also need to be adjusted.  In particular, values
 * smaller than pos do not need to change, while values greater than or
 * equal to pos need to be incremented.
 * That is, we need to apply the following map.
 *
 *	{ atomic[i] -> atomic[i] : i < pos; [i] -> [i + 1] : i >= pos;
 *	  unroll[i] -> unroll[i] : i < pos; [i] -> [i + 1] : i >= pos;
 *	  separate[i] -> separate[i] : i < pos; [i] -> [i + 1] : i >= pos;
 *	  separation_class[[i] -> [c]]
 *		-> separation_class[[i] -> [c]] : i < pos;
 *	  separation_class[[i] -> [c]]
 *		-> separation_class[[i + 1] -> [c]] : i >= pos }
 */
static __isl_give isl_union_map *options_insert_dim(
	__isl_take isl_union_map *options, __isl_take isl_space *space, int pos)
{
	isl_map *map;
	isl_union_map *insertion;
	enum isl_ast_loop_type type;
	const char *name = "separation_class";

	space = isl_space_map_from_set(space);
	map = isl_map_identity(space);
	map = isl_map_insert_dims(map, isl_dim_out, pos, 1);
	options = isl_union_map_apply_domain(options,
						isl_union_map_from_map(map));

	if (!options)
		return NULL;

	map = construct_insertion_map(isl_union_map_get_space(options), pos);

	insertion = isl_union_map_empty(isl_union_map_get_space(options));

	for (type = isl_ast_loop_atomic;
	    type <= isl_ast_loop_separate; ++type) {
		isl_map *map_type = isl_map_copy(map);
		const char *name = option_str[type];
		map_type = isl_map_set_tuple_name(map_type, isl_dim_in, name);
		map_type = isl_map_set_tuple_name(map_type, isl_dim_out, name);
		insertion = isl_union_map_add_map(insertion, map_type);
	}

	map = isl_map_product(map, isl_map_identity(isl_map_get_space(map)));
	map = isl_map_set_tuple_name(map, isl_dim_in, name);
	map = isl_map_set_tuple_name(map, isl_dim_out, name);
	insertion = isl_union_map_add_map(insertion, map);

	options = isl_union_map_apply_range(options, insertion);

	return options;
}

/* If we are generating an AST from a schedule tree (build->node is set),
 * then update the loop AST generation types
 * to reflect the insertion of a dimension at (global) position "pos"
 * in the schedule domain space.
 * We do not need to adjust any isolate option since we would not be inserting
 * any dimensions if there were any isolate option.
 */
static __isl_give isl_ast_build *node_insert_dim(
	__isl_take isl_ast_build *build, int pos)
{
	int i;
	int local_pos;
	enum isl_ast_loop_type *loop_type;
	isl_ctx *ctx;

	build = isl_ast_build_cow(build);
	if (!build)
		return NULL;
	if (!build->node)
		return build;

	ctx = isl_ast_build_get_ctx(build);
	local_pos = pos - build->outer_pos;
	loop_type = isl_realloc_array(ctx, build->loop_type,
					enum isl_ast_loop_type, build->n + 1);
	if (!loop_type)
		return isl_ast_build_free(build);
	build->loop_type = loop_type;
	for (i = build->n - 1; i >= local_pos; --i)
		loop_type[i + 1] = loop_type[i];
	loop_type[local_pos] = isl_ast_loop_default;
	build->n++;

	return build;
}

/* Insert a single dimension in the schedule domain at position "pos".
 * The new dimension is given an isl_id with the empty string as name.
 *
 * The main difficulty is updating build->options to reflect the
 * extra dimension.  This is handled in options_insert_dim.
 *
 * Note that because of the dimension manipulations, the resulting
 * schedule domain space will always be unnamed and unstructured.
 * However, the original schedule domain space may be named and/or
 * structured, so we have to take this possibility into account
 * while performing the transformations.
 *
 * Since the inserted schedule dimension is used by the caller
 * to differentiate between different domain spaces, there is
 * no longer a uniform mapping from the internal schedule space
 * to the input schedule space.  The internal2input mapping is
 * therefore removed.
 */
__isl_give isl_ast_build *isl_ast_build_insert_dim(
	__isl_take isl_ast_build *build, int pos)
{
	isl_ctx *ctx;
	isl_space *space, *ma_space;
	isl_id *id;
	isl_multi_aff *ma;

	build = isl_ast_build_cow(build);
	if (!build)
		return NULL;

	ctx = isl_ast_build_get_ctx(build);
	id = isl_id_alloc(ctx, "", NULL);
	if (!build->node)
		space = isl_ast_build_get_space(build, 1);
	build->iterators = isl_id_list_insert(build->iterators, pos, id);
	build->domain = isl_set_insert_dims(build->domain,
						isl_dim_set, pos, 1);
	build->generated = isl_set_insert_dims(build->generated,
						isl_dim_set, pos, 1);
	build->pending = isl_set_insert_dims(build->pending,
						isl_dim_set, pos, 1);
	build->strides = isl_vec_insert_els(build->strides, pos, 1);
	build->strides = isl_vec_set_element_si(build->strides, pos, 1);
	ma_space = isl_space_params(isl_multi_aff_get_space(build->offsets));
	ma_space = isl_space_set_from_params(ma_space);
	ma_space = isl_space_add_dims(ma_space, isl_dim_set, 1);
	ma_space = isl_space_map_from_set(ma_space);
	ma = isl_multi_aff_zero(isl_space_copy(ma_space));
	build->offsets = isl_multi_aff_splice(build->offsets, pos, pos, ma);
	ma = isl_multi_aff_identity(ma_space);
	build->values = isl_multi_aff_splice(build->values, pos, pos, ma);
	if (!build->node)
		build->options = options_insert_dim(build->options, space, pos);
	build->internal2input = isl_multi_aff_free(build->internal2input);

	if (!build->iterators || !build->domain || !build->generated ||
	    !build->pending || !build->values ||
	    !build->strides || !build->offsets || !build->options)
		return isl_ast_build_free(build);

	build = node_insert_dim(build, pos);

	return build;
}

/* Scale down the current dimension by a factor of "m".
 * "umap" is an isl_union_map that implements the scaling down.
 * That is, it is of the form
 *
 *	{ [.... i ....] -> [.... i' ....] : i = m i' }
 *
 * This function is called right after the strides have been
 * detected, but before any constraints on the current dimension
 * have been included in build->domain.
 * We therefore only need to update stride, offset, the options and
 * the mapping from internal schedule space to the original schedule
 * space, if we are still keeping track of such a mapping.
 * The latter mapping is updated by plugging in
 * { [... i ...] -> [... m i ... ] }.
 */
__isl_give isl_ast_build *isl_ast_build_scale_down(
	__isl_take isl_ast_build *build, __isl_take isl_val *m,
	__isl_take isl_union_map *umap)
{
	isl_aff *aff;
	isl_val *v;
	int depth;

	build = isl_ast_build_cow(build);
	if (!build || !umap || !m)
		goto error;

	depth = build->depth;

	if (build->internal2input) {
		isl_space *space;
		isl_multi_aff *ma;
		isl_aff *aff;

		space = isl_multi_aff_get_space(build->internal2input);
		space = isl_space_map_from_set(isl_space_domain(space));
		ma = isl_multi_aff_identity(space);
		aff = isl_multi_aff_get_aff(ma, depth);
		aff = isl_aff_scale_val(aff, isl_val_copy(m));
		ma = isl_multi_aff_set_aff(ma, depth, aff);
		build->internal2input =
		    isl_multi_aff_pullback_multi_aff(build->internal2input, ma);
		if (!build->internal2input)
			goto error;
	}

	v = isl_vec_get_element_val(build->strides, depth);
	v = isl_val_div(v, isl_val_copy(m));
	build->strides = isl_vec_set_element_val(build->strides, depth, v);

	aff = isl_multi_aff_get_aff(build->offsets, depth);
	aff = isl_aff_scale_down_val(aff, m);
	build->offsets = isl_multi_aff_set_aff(build->offsets, depth, aff);
	build->options = isl_union_map_apply_domain(build->options, umap);
	if (!build->strides || !build->offsets || !build->options)
		return isl_ast_build_free(build);

	return build;
error:
	isl_val_free(m);
	isl_union_map_free(umap);
	return isl_ast_build_free(build);
}

/* Return a list of "n" isl_ids called "c%d", with "%d" starting at "first".
 * If an isl_id with such a name already appears among the parameters
 * in build->domain, then adjust the name to "c%d_%d".
 */
static __isl_give isl_id_list *generate_names(isl_ctx *ctx, int n, int first,
	__isl_keep isl_ast_build *build)
{
	int i;
	isl_id_list *names;

	names = isl_id_list_alloc(ctx, n);
	for (i = 0; i < n; ++i) {
		isl_id *id;

		id = generate_name(ctx, first + i, build);
		names = isl_id_list_add(names, id);
	}

	return names;
}

/* Embed "options" into the given isl_ast_build space.
 *
 * This function is called from within a nested call to
 * isl_ast_build_node_from_schedule_map.
 * "options" refers to the additional schedule,
 * while space refers to both the space of the outer isl_ast_build and
 * that of the additional schedule.
 * Specifically, space is of the form
 *
 *	[I -> S]
 *
 * while options lives in the space(s)
 *
 *	S -> *
 *
 * We compute
 *
 *	[I -> S] -> S
 *
 * and compose this with options, to obtain the new options
 * living in the space(s)
 *
 *	[I -> S] -> *
 */
static __isl_give isl_union_map *embed_options(
	__isl_take isl_union_map *options, __isl_take isl_space *space)
{
	isl_map *map;

	map = isl_map_universe(isl_space_unwrap(space));
	map = isl_map_range_map(map);

	options = isl_union_map_apply_range(
				isl_union_map_from_map(map), options);

	return options;
}

/* Update "build" for use in a (possibly nested) code generation.  That is,
 * extend "build" from an AST build on some domain O to an AST build
 * on domain [O -> S], with S corresponding to "space".
 * If the original domain is a parameter domain, then the new domain is
 * simply S.
 * "iterators" is a list of iterators for S, but the number of elements
 * may be smaller or greater than the number of set dimensions of S.
 * If "keep_iterators" is set, then any extra ids in build->iterators
 * are reused for S.  Otherwise, these extra ids are dropped.
 *
 * We first update build->outer_pos to the current depth.
 * This depth is zero in case this is the outermost code generation.
 *
 * We then add additional ids such that the number of iterators is at least
 * equal to the dimension of the new build domain.
 *
 * If the original domain is parametric, then we are constructing
 * an isl_ast_build for the outer code generation and we pass control
 * to isl_ast_build_init.
 *
 * Otherwise, we adjust the fields of "build" to include "space".
 */
__isl_give isl_ast_build *isl_ast_build_product(
	__isl_take isl_ast_build *build, __isl_take isl_space *space)
{
	isl_ctx *ctx;
	isl_vec *strides;
	isl_set *set;
	isl_multi_aff *embedding;
	int dim, n_it;

	build = isl_ast_build_cow(build);
	if (!build)
		goto error;

	build->outer_pos = build->depth;

	ctx = isl_ast_build_get_ctx(build);
	dim = isl_set_dim(build->domain, isl_dim_set);
	dim += isl_space_dim(space, isl_dim_set);
	n_it = isl_id_list_n_id(build->iterators);
	if (n_it < dim) {
		isl_id_list *l;
		l = generate_names(ctx, dim - n_it, n_it, build);
		build->iterators = isl_id_list_concat(build->iterators, l);
	}

	if (isl_set_is_params(build->domain))
		return isl_ast_build_init(build, space);

	set = isl_set_universe(isl_space_copy(space));
	build->domain = isl_set_product(build->domain, isl_set_copy(set));
	build->pending = isl_set_product(build->pending, isl_set_copy(set));
	build->generated = isl_set_product(build->generated, set);

	strides = isl_vec_alloc(ctx, isl_space_dim(space, isl_dim_set));
	strides = isl_vec_set_si(strides, 1);
	build->strides = isl_vec_concat(build->strides, strides);

	space = isl_space_map_from_set(space);
	build->offsets = isl_multi_aff_align_params(build->offsets,
						    isl_space_copy(space));
	build->offsets = isl_multi_aff_product(build->offsets,
				isl_multi_aff_zero(isl_space_copy(space)));
	build->values = isl_multi_aff_align_params(build->values,
						    isl_space_copy(space));
	embedding = isl_multi_aff_identity(space);
	build->values = isl_multi_aff_product(build->values,
					isl_multi_aff_copy(embedding));
	if (build->internal2input) {
		build->internal2input =
			isl_multi_aff_product(build->internal2input, embedding);
		build->internal2input =
			isl_multi_aff_flatten_range(build->internal2input);
		if (!build->internal2input)
			return isl_ast_build_free(build);
	} else {
		isl_multi_aff_free(embedding);
	}

	space = isl_ast_build_get_space(build, 1);
	build->options = embed_options(build->options, space);

	if (!build->iterators || !build->domain || !build->generated ||
	    !build->pending || !build->values ||
	    !build->strides || !build->offsets || !build->options)
		return isl_ast_build_free(build);

	return build;
error:
	isl_ast_build_free(build);
	isl_space_free(space);
	return NULL;
}

/* Does "aff" only attain non-negative values over build->domain?
 * That is, does it not attain any negative values?
 */
int isl_ast_build_aff_is_nonneg(__isl_keep isl_ast_build *build,
	__isl_keep isl_aff *aff)
{
	isl_set *test;
	int empty;

	if (!build)
		return -1;

	aff = isl_aff_copy(aff);
	test = isl_set_from_basic_set(isl_aff_neg_basic_set(aff));
	test = isl_set_intersect(test, isl_set_copy(build->domain));
	empty = isl_set_is_empty(test);
	isl_set_free(test);

	return empty;
}

/* Does the dimension at (internal) position "pos" have a non-trivial stride?
 */
isl_bool isl_ast_build_has_stride(__isl_keep isl_ast_build *build, int pos)
{
	isl_val *v;
	isl_bool has_stride;

	if (!build)
		return isl_bool_error;

	v = isl_vec_get_element_val(build->strides, pos);
	has_stride = isl_bool_not(isl_val_is_one(v));
	isl_val_free(v);

	return has_stride;
}

/* Given that the dimension at position "pos" takes on values
 *
 *	f + s a
 *
 * with a an integer, return s through *stride.
 */
__isl_give isl_val *isl_ast_build_get_stride(__isl_keep isl_ast_build *build,
	int pos)
{
	if (!build)
		return NULL;

	return isl_vec_get_element_val(build->strides, pos);
}

/* Given that the dimension at position "pos" takes on values
 *
 *	f + s a
 *
 * with a an integer, return f.
 */
__isl_give isl_aff *isl_ast_build_get_offset(
	__isl_keep isl_ast_build *build, int pos)
{
	if (!build)
		return NULL;

	return isl_multi_aff_get_aff(build->offsets, pos);
}

/* Is the dimension at position "pos" known to attain only a single
 * value that, moreover, can be described by a single affine expression
 * in terms of the outer dimensions and parameters?
 *
 * If not, then the corresponding affine expression in build->values
 * is set to be equal to the same input dimension.
 * Otherwise, it is set to the requested expression in terms of
 * outer dimensions and parameters.
 */
int isl_ast_build_has_affine_value(__isl_keep isl_ast_build *build,
	int pos)
{
	isl_aff *aff;
	int involves;

	if (!build)
		return -1;

	aff = isl_multi_aff_get_aff(build->values, pos);
	involves = isl_aff_involves_dims(aff, isl_dim_in, pos, 1);
	isl_aff_free(aff);

	if (involves < 0)
		return -1;

	return !involves;
}

/* Plug in the known values (fixed affine expressions in terms of
 * parameters and outer loop iterators) of all loop iterators
 * in the domain of "umap".
 *
 * We simply precompose "umap" with build->values.
 */
__isl_give isl_union_map *isl_ast_build_substitute_values_union_map_domain(
	__isl_keep isl_ast_build *build, __isl_take isl_union_map *umap)
{
	isl_multi_aff *values;

	if (!build)
		return isl_union_map_free(umap);

	values = isl_multi_aff_copy(build->values);
	umap = isl_union_map_preimage_domain_multi_aff(umap, values);

	return umap;
}

/* Is the current dimension known to attain only a single value?
 */
int isl_ast_build_has_value(__isl_keep isl_ast_build *build)
{
	if (!build)
		return -1;

	return build->value != NULL;
}

/* Simplify the basic set "bset" based on what we know about
 * the iterators of already generated loops.
 *
 * "bset" is assumed to live in the (internal) schedule domain.
 */
__isl_give isl_basic_set *isl_ast_build_compute_gist_basic_set(
	__isl_keep isl_ast_build *build, __isl_take isl_basic_set *bset)
{
	if (!build)
		goto error;

	bset = isl_basic_set_preimage_multi_aff(bset,
					isl_multi_aff_copy(build->values));
	bset = isl_basic_set_gist(bset,
			isl_set_simple_hull(isl_set_copy(build->domain)));

	return bset;
error:
	isl_basic_set_free(bset);
	return NULL;
}

/* Simplify the set "set" based on what we know about
 * the iterators of already generated loops.
 *
 * "set" is assumed to live in the (internal) schedule domain.
 */
__isl_give isl_set *isl_ast_build_compute_gist(
	__isl_keep isl_ast_build *build, __isl_take isl_set *set)
{
	if (!build)
		goto error;

	if (!isl_set_is_params(set))
		set = isl_set_preimage_multi_aff(set,
					isl_multi_aff_copy(build->values));
	set = isl_set_gist(set, isl_set_copy(build->domain));

	return set;
error:
	isl_set_free(set);
	return NULL;
}

/* Include information about what we know about the iterators of
 * already generated loops to "set".
 *
 * We currently only plug in the known affine values of outer loop
 * iterators.
 * In principle we could also introduce equalities or even other
 * constraints implied by the intersection of "set" and build->domain.
 */
__isl_give isl_set *isl_ast_build_specialize(__isl_keep isl_ast_build *build,
	__isl_take isl_set *set)
{
	if (!build)
		return isl_set_free(set);

	return isl_set_preimage_multi_aff(set,
					isl_multi_aff_copy(build->values));
}

/* Plug in the known affine values of outer loop iterators in "bset".
 */
__isl_give isl_basic_set *isl_ast_build_specialize_basic_set(
	__isl_keep isl_ast_build *build, __isl_take isl_basic_set *bset)
{
	if (!build)
		return isl_basic_set_free(bset);

	return isl_basic_set_preimage_multi_aff(bset,
					isl_multi_aff_copy(build->values));
}

/* Simplify the map "map" based on what we know about
 * the iterators of already generated loops.
 *
 * The domain of "map" is assumed to live in the (internal) schedule domain.
 */
__isl_give isl_map *isl_ast_build_compute_gist_map_domain(
	__isl_keep isl_ast_build *build, __isl_take isl_map *map)
{
	if (!build)
		goto error;

	map = isl_map_gist_domain(map, isl_set_copy(build->domain));

	return map;
error:
	isl_map_free(map);
	return NULL;
}

/* Simplify the affine expression "aff" based on what we know about
 * the iterators of already generated loops.
 *
 * The domain of "aff" is assumed to live in the (internal) schedule domain.
 */
__isl_give isl_aff *isl_ast_build_compute_gist_aff(
	__isl_keep isl_ast_build *build, __isl_take isl_aff *aff)
{
	if (!build)
		goto error;

	aff = isl_aff_gist(aff, isl_set_copy(build->domain));

	return aff;
error:
	isl_aff_free(aff);
	return NULL;
}

/* Simplify the piecewise affine expression "aff" based on what we know about
 * the iterators of already generated loops.
 *
 * The domain of "pa" is assumed to live in the (internal) schedule domain.
 */
__isl_give isl_pw_aff *isl_ast_build_compute_gist_pw_aff(
	__isl_keep isl_ast_build *build, __isl_take isl_pw_aff *pa)
{
	if (!build)
		goto error;

	if (!isl_set_is_params(build->domain))
		pa = isl_pw_aff_pullback_multi_aff(pa,
					isl_multi_aff_copy(build->values));
	pa = isl_pw_aff_gist(pa, isl_set_copy(build->domain));

	return pa;
error:
	isl_pw_aff_free(pa);
	return NULL;
}

/* Simplify the piecewise multi-affine expression "aff" based on what
 * we know about the iterators of already generated loops.
 *
 * The domain of "pma" is assumed to live in the (internal) schedule domain.
 */
__isl_give isl_pw_multi_aff *isl_ast_build_compute_gist_pw_multi_aff(
	__isl_keep isl_ast_build *build, __isl_take isl_pw_multi_aff *pma)
{
	if (!build)
		goto error;

	pma = isl_pw_multi_aff_pullback_multi_aff(pma,
					isl_multi_aff_copy(build->values));
	pma = isl_pw_multi_aff_gist(pma, isl_set_copy(build->domain));

	return pma;
error:
	isl_pw_multi_aff_free(pma);
	return NULL;
}

/* Extract the schedule domain of the given type from build->options
 * at the current depth.
 *
 * In particular, find the subset of build->options that is of
 * the following form
 *
 *	schedule_domain -> type[depth]
 *
 * and return the corresponding domain, after eliminating inner dimensions
 * and divs that depend on the current dimension.
 *
 * Note that the domain of build->options has been reformulated
 * in terms of the internal build space in embed_options,
 * but the position is still that within the current code generation.
 */
__isl_give isl_set *isl_ast_build_get_option_domain(
	__isl_keep isl_ast_build *build, enum isl_ast_loop_type type)
{
	const char *name;
	isl_space *space;
	isl_map *option;
	isl_set *domain;
	int local_pos;

	if (!build)
		return NULL;

	name = option_str[type];
	local_pos = build->depth - build->outer_pos;

	space = isl_ast_build_get_space(build, 1);
	space = isl_space_from_domain(space);
	space = isl_space_add_dims(space, isl_dim_out, 1);
	space = isl_space_set_tuple_name(space, isl_dim_out, name);

	option = isl_union_map_extract_map(build->options, space);
	option = isl_map_fix_si(option, isl_dim_out, 0, local_pos);

	domain = isl_map_domain(option);
	domain = isl_ast_build_eliminate(build, domain);

	return domain;
}

/* How does the user want the current schedule dimension to be generated?
 * These choices have been extracted from the schedule node
 * in extract_loop_types and stored in build->loop_type.
 * They have been updated to reflect any dimension insertion in
 * node_insert_dim.
 * Return isl_ast_domain_error on error.
 *
 * If "isolated" is set, then we get the loop AST generation type
 * directly from the band node since node_insert_dim cannot have been
 * called on a band with the isolate option.
 */
enum isl_ast_loop_type isl_ast_build_get_loop_type(
	__isl_keep isl_ast_build *build, int isolated)
{
	int local_pos;
	isl_ctx *ctx;

	if (!build)
		return isl_ast_loop_error;
	ctx = isl_ast_build_get_ctx(build);
	if (!build->node)
		isl_die(ctx, isl_error_internal,
			"only works for schedule tree based AST generation",
			return isl_ast_loop_error);

	local_pos = build->depth - build->outer_pos;
	if (!isolated)
		return build->loop_type[local_pos];
	return isl_schedule_node_band_member_get_isolate_ast_loop_type(
							build->node, local_pos);
}

/* Extract the isolated set from the isolate option, if any,
 * and store in the build.
 * If there is no isolate option, then the isolated set is
 * set to the empty set.
 *
 * The isolate option is of the form
 *
 *	isolate[[outer bands] -> current_band]
 *
 * We flatten this set and then map it back to the internal
 * schedule space.
 *
 * If we have already extracted the isolated set
 * or if internal2input is no longer set, then we do not
 * need to do anything.  In the latter case, we know
 * that the current band cannot have any isolate option.
 */
__isl_give isl_ast_build *isl_ast_build_extract_isolated(
	__isl_take isl_ast_build *build)
{
	isl_set *isolated;

	if (!build)
		return NULL;
	if (!build->internal2input)
		return build;
	if (build->isolated)
		return build;

	build = isl_ast_build_cow(build);
	if (!build)
		return NULL;

	isolated = isl_schedule_node_band_get_ast_isolate_option(build->node);
	isolated = isl_set_flatten(isolated);
	isolated = isl_set_preimage_multi_aff(isolated,
				    isl_multi_aff_copy(build->internal2input));

	build->isolated = isolated;
	if (!build->isolated)
		return isl_ast_build_free(build);

	return build;
}

/* Does "build" have a non-empty isolated set?
 *
 * The caller is assumed to have called isl_ast_build_extract_isolated first.
 */
int isl_ast_build_has_isolated(__isl_keep isl_ast_build *build)
{
	int empty;

	if (!build)
		return -1;
	if (!build->internal2input)
		return 0;
	if (!build->isolated)
		isl_die(isl_ast_build_get_ctx(build), isl_error_internal,
			"isolated set not extracted yet", return -1);

	empty = isl_set_plain_is_empty(build->isolated);
	return empty < 0 ? -1 : !empty;
}

/* Return a copy of the isolated set of "build".
 *
 * The caller is assume to have called isl_ast_build_has_isolated first,
 * with this function returning true.
 * In particular, this function should not be called if we are no
 * longer keeping track of internal2input (and there therefore could
 * not possibly be any isolated set).
 */
__isl_give isl_set *isl_ast_build_get_isolated(__isl_keep isl_ast_build *build)
{
	if (!build)
		return NULL;
	if (!build->internal2input)
		isl_die(isl_ast_build_get_ctx(build), isl_error_internal,
			"build cannot have isolated set", return NULL);

	return isl_set_copy(build->isolated);
}

/* Extract the separation class mapping at the current depth.
 *
 * In particular, find and return the subset of build->options that is of
 * the following form
 *
 *	schedule_domain -> separation_class[[depth] -> [class]]
 *
 * The caller is expected to eliminate inner dimensions from the domain.
 *
 * Note that the domain of build->options has been reformulated
 * in terms of the internal build space in embed_options,
 * but the position is still that within the current code generation.
 */
__isl_give isl_map *isl_ast_build_get_separation_class(
	__isl_keep isl_ast_build *build)
{
	isl_ctx *ctx;
	isl_space *space_sep, *space;
	isl_map *res;
	int local_pos;

	if (!build)
		return NULL;

	local_pos = build->depth - build->outer_pos;
	ctx = isl_ast_build_get_ctx(build);
	space_sep = isl_space_alloc(ctx, 0, 1, 1);
	space_sep = isl_space_wrap(space_sep);
	space_sep = isl_space_set_tuple_name(space_sep, isl_dim_set,
						"separation_class");
	space = isl_ast_build_get_space(build, 1);
	space_sep = isl_space_align_params(space_sep, isl_space_copy(space));
	space = isl_space_map_from_domain_and_range(space, space_sep);

	res = isl_union_map_extract_map(build->options, space);
	res = isl_map_fix_si(res, isl_dim_out, 0, local_pos);
	res = isl_map_coalesce(res);

	return res;
}

/* Eliminate dimensions inner to the current dimension.
 */
__isl_give isl_set *isl_ast_build_eliminate_inner(
	__isl_keep isl_ast_build *build, __isl_take isl_set *set)
{
	int dim;
	int depth;

	if (!build)
		return isl_set_free(set);

	dim = isl_set_dim(set, isl_dim_set);
	depth = build->depth;
	set = isl_set_detect_equalities(set);
	set = isl_set_eliminate(set, isl_dim_set, depth + 1, dim - (depth + 1));

	return set;
}

/* Eliminate unknown divs and divs that depend on the current dimension.
 *
 * Note that during the elimination of unknown divs, we may discover
 * an explicit representation of some other unknown divs, which may
 * depend on the current dimension.  We therefore need to eliminate
 * unknown divs first.
 */
__isl_give isl_set *isl_ast_build_eliminate_divs(
	__isl_keep isl_ast_build *build, __isl_take isl_set *set)
{
	int depth;

	if (!build)
		return isl_set_free(set);

	set = isl_set_remove_unknown_divs(set);
	depth = build->depth;
	set = isl_set_remove_divs_involving_dims(set, isl_dim_set, depth, 1);

	return set;
}

/* Eliminate dimensions inner to the current dimension as well as
 * unknown divs and divs that depend on the current dimension.
 * The result then consists only of constraints that are independent
 * of the current dimension and upper and lower bounds on the current
 * dimension.
 */
__isl_give isl_set *isl_ast_build_eliminate(
	__isl_keep isl_ast_build *build, __isl_take isl_set *domain)
{
	domain = isl_ast_build_eliminate_inner(build, domain);
	domain = isl_ast_build_eliminate_divs(build, domain);
	return domain;
}

/* Replace build->single_valued by "sv".
 */
__isl_give isl_ast_build *isl_ast_build_set_single_valued(
	__isl_take isl_ast_build *build, int sv)
{
	if (!build)
		return build;
	if (build->single_valued == sv)
		return build;
	build = isl_ast_build_cow(build);
	if (!build)
		return build;
	build->single_valued = sv;

	return build;
}
