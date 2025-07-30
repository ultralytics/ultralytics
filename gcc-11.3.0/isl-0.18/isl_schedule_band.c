/*
 * Copyright 2013-2014 Ecole Normale Superieure
 * Copyright 2014      INRIA Rocquencourt
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 * and Inria Paris - Rocquencourt, Domaine de Voluceau - Rocquencourt,
 * B.P. 105 - 78153 Le Chesnay, France
 */

#include <string.h>
#include <isl/map.h>
#include <isl/schedule_node.h>
#include <isl_schedule_band.h>
#include <isl_schedule_private.h>

isl_ctx *isl_schedule_band_get_ctx(__isl_keep isl_schedule_band *band)
{
	return band ? isl_multi_union_pw_aff_get_ctx(band->mupa) : NULL;
}

/* Return a new uninitialized isl_schedule_band.
 */
static __isl_give isl_schedule_band *isl_schedule_band_alloc(isl_ctx *ctx)
{
	isl_schedule_band *band;

	band = isl_calloc_type(ctx, isl_schedule_band);
	if (!band)
		return NULL;

	band->ref = 1;

	return band;
}

/* Return a new isl_schedule_band with partial schedule "mupa".
 * First replace "mupa" by its greatest integer part to ensure
 * that the schedule is always integral.
 * The band is not marked permutable, the dimensions are not
 * marked coincident and the AST build options are empty.
 * Since there are no build options, the node is not anchored.
 */
__isl_give isl_schedule_band *isl_schedule_band_from_multi_union_pw_aff(
	__isl_take isl_multi_union_pw_aff *mupa)
{
	isl_ctx *ctx;
	isl_schedule_band *band;
	isl_space *space;

	mupa = isl_multi_union_pw_aff_floor(mupa);
	if (!mupa)
		return NULL;
	ctx = isl_multi_union_pw_aff_get_ctx(mupa);
	band = isl_schedule_band_alloc(ctx);
	if (!band)
		goto error;

	band->n = isl_multi_union_pw_aff_dim(mupa, isl_dim_set);
	band->coincident = isl_calloc_array(ctx, int, band->n);
	band->mupa = mupa;
	space = isl_space_params_alloc(ctx, 0);
	band->ast_build_options = isl_union_set_empty(space);
	band->anchored = 0;

	if ((band->n && !band->coincident) || !band->ast_build_options)
		return isl_schedule_band_free(band);

	return band;
error:
	isl_multi_union_pw_aff_free(mupa);
	return NULL;
}

/* Create a duplicate of the given isl_schedule_band.
 */
__isl_give isl_schedule_band *isl_schedule_band_dup(
	__isl_keep isl_schedule_band *band)
{
	int i;
	isl_ctx *ctx;
	isl_schedule_band *dup;

	if (!band)
		return NULL;

	ctx = isl_schedule_band_get_ctx(band);
	dup = isl_schedule_band_alloc(ctx);
	if (!dup)
		return NULL;

	dup->n = band->n;
	dup->coincident = isl_alloc_array(ctx, int, band->n);
	if (band->n && !dup->coincident)
		return isl_schedule_band_free(dup);

	for (i = 0; i < band->n; ++i)
		dup->coincident[i] = band->coincident[i];
	dup->permutable = band->permutable;

	dup->mupa = isl_multi_union_pw_aff_copy(band->mupa);
	dup->ast_build_options = isl_union_set_copy(band->ast_build_options);
	if (!dup->mupa || !dup->ast_build_options)
		return isl_schedule_band_free(dup);

	if (band->loop_type) {
		dup->loop_type = isl_alloc_array(ctx,
					    enum isl_ast_loop_type, band->n);
		if (band->n && !dup->loop_type)
			return isl_schedule_band_free(dup);
		for (i = 0; i < band->n; ++i)
			dup->loop_type[i] = band->loop_type[i];
	}
	if (band->isolate_loop_type) {
		dup->isolate_loop_type = isl_alloc_array(ctx,
					    enum isl_ast_loop_type, band->n);
		if (band->n && !dup->isolate_loop_type)
			return isl_schedule_band_free(dup);
		for (i = 0; i < band->n; ++i)
			dup->isolate_loop_type[i] = band->isolate_loop_type[i];
	}

	return dup;
}

/* Return an isl_schedule_band that is equal to "band" and that has only
 * a single reference.
 */
__isl_give isl_schedule_band *isl_schedule_band_cow(
	__isl_take isl_schedule_band *band)
{
	if (!band)
		return NULL;

	if (band->ref == 1)
		return band;
	band->ref--;
	return isl_schedule_band_dup(band);
}

/* Return a new reference to "band".
 */
__isl_give isl_schedule_band *isl_schedule_band_copy(
	__isl_keep isl_schedule_band *band)
{
	if (!band)
		return NULL;

	band->ref++;
	return band;
}

/* Free a reference to "band" and return NULL.
 */
__isl_null isl_schedule_band *isl_schedule_band_free(
	__isl_take isl_schedule_band *band)
{
	if (!band)
		return NULL;

	if (--band->ref > 0)
		return NULL;

	isl_multi_union_pw_aff_free(band->mupa);
	isl_union_set_free(band->ast_build_options);
	free(band->loop_type);
	free(band->isolate_loop_type);
	free(band->coincident);
	free(band);

	return NULL;
}

/* Are "band1" and "band2" obviously equal?
 */
isl_bool isl_schedule_band_plain_is_equal(__isl_keep isl_schedule_band *band1,
	__isl_keep isl_schedule_band *band2)
{
	int i;
	isl_bool equal;

	if (!band1 || !band2)
		return isl_bool_error;
	if (band1 == band2)
		return isl_bool_true;

	if (band1->n != band2->n)
		return isl_bool_false;
	for (i = 0; i < band1->n; ++i)
		if (band1->coincident[i] != band2->coincident[i])
			return isl_bool_false;
	if (band1->permutable != band2->permutable)
		return isl_bool_false;

	equal = isl_multi_union_pw_aff_plain_is_equal(band1->mupa, band2->mupa);
	if (equal < 0 || !equal)
		return equal;

	if (!band1->loop_type != !band2->loop_type)
		return isl_bool_false;
	if (band1->loop_type)
		for (i = 0; i < band1->n; ++i)
			if (band1->loop_type[i] != band2->loop_type[i])
				return isl_bool_false;

	if (!band1->isolate_loop_type != !band2->isolate_loop_type)
		return isl_bool_false;
	if (band1->isolate_loop_type)
		for (i = 0; i < band1->n; ++i)
			if (band1->isolate_loop_type[i] !=
						band2->isolate_loop_type[i])
				return isl_bool_false;

	return isl_union_set_is_equal(band1->ast_build_options,
					band2->ast_build_options);
}

/* Return the number of scheduling dimensions in the band.
 */
int isl_schedule_band_n_member(__isl_keep isl_schedule_band *band)
{
	return band ? band->n : 0;
}

/* Is the given scheduling dimension coincident within the band and
 * with respect to the coincidence constraints?
 */
isl_bool isl_schedule_band_member_get_coincident(
	__isl_keep isl_schedule_band *band, int pos)
{
	if (!band)
		return isl_bool_error;

	if (pos < 0 || pos >= band->n)
		isl_die(isl_schedule_band_get_ctx(band), isl_error_invalid,
			"invalid member position", return isl_bool_error);

	return band->coincident[pos];
}

/* Mark the given scheduling dimension as being coincident or not
 * according to "coincident".
 */
__isl_give isl_schedule_band *isl_schedule_band_member_set_coincident(
	__isl_take isl_schedule_band *band, int pos, int coincident)
{
	if (!band)
		return NULL;
	if (isl_schedule_band_member_get_coincident(band, pos) == coincident)
		return band;
	band = isl_schedule_band_cow(band);
	if (!band)
		return NULL;

	if (pos < 0 || pos >= band->n)
		isl_die(isl_schedule_band_get_ctx(band), isl_error_invalid,
			"invalid member position",
			isl_schedule_band_free(band));

	band->coincident[pos] = coincident;

	return band;
}

/* Is the schedule band mark permutable?
 */
isl_bool isl_schedule_band_get_permutable(__isl_keep isl_schedule_band *band)
{
	if (!band)
		return isl_bool_error;
	return band->permutable;
}

/* Mark the schedule band permutable or not according to "permutable"?
 */
__isl_give isl_schedule_band *isl_schedule_band_set_permutable(
	__isl_take isl_schedule_band *band, int permutable)
{
	if (!band)
		return NULL;
	if (band->permutable == permutable)
		return band;
	band = isl_schedule_band_cow(band);
	if (!band)
		return NULL;

	band->permutable = permutable;

	return band;
}

/* Is the band node "node" anchored?  That is, does it reference
 * the outer band nodes?
 */
int isl_schedule_band_is_anchored(__isl_keep isl_schedule_band *band)
{
	return band ? band->anchored : -1;
}

/* Return the schedule space of the band.
 */
__isl_give isl_space *isl_schedule_band_get_space(
	__isl_keep isl_schedule_band *band)
{
	if (!band)
		return NULL;
	return isl_multi_union_pw_aff_get_space(band->mupa);
}

/* Intersect the domain of the band schedule of "band" with "domain".
 */
__isl_give isl_schedule_band *isl_schedule_band_intersect_domain(
	__isl_take isl_schedule_band *band, __isl_take isl_union_set *domain)
{
	band = isl_schedule_band_cow(band);
	if (!band || !domain)
		goto error;

	band->mupa = isl_multi_union_pw_aff_intersect_domain(band->mupa,
								domain);
	if (!band->mupa)
		return isl_schedule_band_free(band);

	return band;
error:
	isl_schedule_band_free(band);
	isl_union_set_free(domain);
	return NULL;
}

/* Return the schedule of the band in isolation.
 */
__isl_give isl_multi_union_pw_aff *isl_schedule_band_get_partial_schedule(
	__isl_keep isl_schedule_band *band)
{
	return band ? isl_multi_union_pw_aff_copy(band->mupa) : NULL;
}

/* Replace the schedule of "band" by "schedule".
 */
__isl_give isl_schedule_band *isl_schedule_band_set_partial_schedule(
	__isl_take isl_schedule_band *band,
	__isl_take isl_multi_union_pw_aff *schedule)
{
	band = isl_schedule_band_cow(band);
	if (!band || !schedule)
		goto error;

	isl_multi_union_pw_aff_free(band->mupa);
	band->mupa = schedule;

	return band;
error:
	isl_schedule_band_free(band);
	isl_multi_union_pw_aff_free(schedule);
	return NULL;
}

/* Return the loop AST generation type for the band member of "band"
 * at position "pos".
 */
enum isl_ast_loop_type isl_schedule_band_member_get_ast_loop_type(
	__isl_keep isl_schedule_band *band, int pos)
{
	if (!band)
		return isl_ast_loop_error;

	if (pos < 0 || pos >= band->n)
		isl_die(isl_schedule_band_get_ctx(band), isl_error_invalid,
			"invalid member position", return -1);

	if (!band->loop_type)
		return isl_ast_loop_default;

	return band->loop_type[pos];
}

/* Set the loop AST generation type for the band member of "band"
 * at position "pos" to "type".
 */
__isl_give isl_schedule_band *isl_schedule_band_member_set_ast_loop_type(
	__isl_take isl_schedule_band *band, int pos,
	enum isl_ast_loop_type type)
{
	if (!band)
		return NULL;
	if (isl_schedule_band_member_get_ast_loop_type(band, pos) == type)
		return band;

	if (pos < 0 || pos >= band->n)
		isl_die(isl_schedule_band_get_ctx(band), isl_error_invalid,
			"invalid member position",
			isl_schedule_band_free(band));

	band = isl_schedule_band_cow(band);
	if (!band)
		return isl_schedule_band_free(band);

	if (!band->loop_type) {
		isl_ctx *ctx;

		ctx = isl_schedule_band_get_ctx(band);
		band->loop_type = isl_calloc_array(ctx,
					    enum isl_ast_loop_type, band->n);
		if (band->n && !band->loop_type)
			return isl_schedule_band_free(band);
	}

	band->loop_type[pos] = type;

	return band;
}

/* Return the loop AST generation type for the band member of "band"
 * at position "pos" for the part that has been isolated by the isolate option.
 */
enum isl_ast_loop_type isl_schedule_band_member_get_isolate_ast_loop_type(
	__isl_keep isl_schedule_band *band, int pos)
{
	if (!band)
		return isl_ast_loop_error;

	if (pos < 0 || pos >= band->n)
		isl_die(isl_schedule_band_get_ctx(band), isl_error_invalid,
			"invalid member position", return -1);

	if (!band->isolate_loop_type)
		return isl_ast_loop_default;

	return band->isolate_loop_type[pos];
}

/* Set the loop AST generation type for the band member of "band"
 * at position "pos" to "type" for the part that has been isolated
 * by the isolate option.
 */
__isl_give isl_schedule_band *
isl_schedule_band_member_set_isolate_ast_loop_type(
	__isl_take isl_schedule_band *band, int pos,
	enum isl_ast_loop_type type)
{
	if (!band)
		return NULL;
	if (isl_schedule_band_member_get_isolate_ast_loop_type(band, pos) ==
									type)
		return band;

	if (pos < 0 || pos >= band->n)
		isl_die(isl_schedule_band_get_ctx(band), isl_error_invalid,
			"invalid member position",
			isl_schedule_band_free(band));

	band = isl_schedule_band_cow(band);
	if (!band)
		return isl_schedule_band_free(band);

	if (!band->isolate_loop_type) {
		isl_ctx *ctx;

		ctx = isl_schedule_band_get_ctx(band);
		band->isolate_loop_type = isl_calloc_array(ctx,
					    enum isl_ast_loop_type, band->n);
		if (band->n && !band->isolate_loop_type)
			return isl_schedule_band_free(band);
	}

	band->isolate_loop_type[pos] = type;

	return band;
}

static const char *option_str[] = {
	[isl_ast_loop_atomic] = "atomic",
	[isl_ast_loop_unroll] = "unroll",
	[isl_ast_loop_separate] = "separate"
};

/* Given a parameter space "space", extend it to a set space
 *
 *	{ type[x] }
 *
 * or
 *
 *	{ [isolate[] -> type[x]] }
 *
 * depending on whether "isolate" is set.
 * These can be used to encode loop AST generation options of the given type.
 */
static __isl_give isl_space *loop_type_space(__isl_take isl_space *space,
	enum isl_ast_loop_type type, int isolate)
{
	const char *name;

	name = option_str[type];
	space = isl_space_set_from_params(space);
	space = isl_space_add_dims(space, isl_dim_set, 1);
	space = isl_space_set_tuple_name(space, isl_dim_set, name);
	if (!isolate)
		return space;
	space = isl_space_from_range(space);
	space = isl_space_set_tuple_name(space, isl_dim_in, "isolate");
	space = isl_space_wrap(space);

	return space;
}

/* Add encodings of the "n" loop AST generation options "type" to "options".
 * If "isolate" is set, then these options refer to the isolated part.
 *
 * In particular, for each sequence of consecutive identical types "t",
 * different from the default, add an option
 *
 *	{ t[x] : first <= x <= last }
 *
 * or
 *
 *	{ [isolate[] -> t[x]] : first <= x <= last }
 */
static __isl_give isl_union_set *add_loop_types(
	__isl_take isl_union_set *options, int n, enum isl_ast_loop_type *type,
	int isolate)
{
	int i;
	isl_ctx *ctx;

	if (!type)
		return options;
	if (!options)
		return NULL;

	ctx = isl_union_set_get_ctx(options);
	for (i = 0; i < n; ++i) {
		int first;
		isl_space *space;
		isl_set *option;

		if (type[i] == isl_ast_loop_default)
			continue;

		first = i;
		while (i + 1 < n && type[i + 1] == type[i])
			++i;

		space = isl_union_set_get_space(options);
		space = loop_type_space(space, type[i], isolate);
		option = isl_set_universe(space);
		option = isl_set_lower_bound_si(option, isl_dim_set, 0, first);
		option = isl_set_upper_bound_si(option, isl_dim_set, 0, i);
		options = isl_union_set_add_set(options, option);
	}

	return options;
}

/* Return the AST build options associated to "band".
 */
__isl_give isl_union_set *isl_schedule_band_get_ast_build_options(
	__isl_keep isl_schedule_band *band)
{
	isl_union_set *options;

	if (!band)
		return NULL;

	options = isl_union_set_copy(band->ast_build_options);
	options = add_loop_types(options, band->n, band->loop_type, 0);
	options = add_loop_types(options, band->n, band->isolate_loop_type, 1);

	return options;
}

/* Does "uset" contain any set that satisfies "is"?
 * "is" is assumed to set its integer argument to 1 if it is satisfied.
 */
static int has_any(__isl_keep isl_union_set *uset,
	isl_stat (*is)(__isl_take isl_set *set, void *user))
{
	int found = 0;

	if (isl_union_set_foreach_set(uset, is, &found) < 0 && !found)
		return -1;

	return found;
}

/* Does "set" live in a space of the form
 *
 *	isolate[[...] -> [...]]
 *
 * ?
 *
 * If so, set *found and abort the search.
 */
static isl_stat is_isolate(__isl_take isl_set *set, void *user)
{
	int *found = user;

	if (isl_set_has_tuple_name(set)) {
		const char *name;
		name = isl_set_get_tuple_name(set);
		if (isl_set_is_wrapping(set) && !strcmp(name, "isolate"))
			*found = 1;
	}
	isl_set_free(set);

	return *found ? isl_stat_error : isl_stat_ok;
}

/* Does "options" include an option of the ofrm
 *
 *	isolate[[...] -> [...]]
 *
 * ?
 */
static int has_isolate_option(__isl_keep isl_union_set *options)
{
	return has_any(options, &is_isolate);
}

/* Does "set" encode a loop AST generation option?
 */
static isl_stat is_loop_type_option(__isl_take isl_set *set, void *user)
{
	int *found = user;

	if (isl_set_dim(set, isl_dim_set) == 1 &&
	    isl_set_has_tuple_name(set)) {
		const char *name;
		enum isl_ast_loop_type type;
		name = isl_set_get_tuple_name(set);
		for (type = isl_ast_loop_atomic;
		    type <= isl_ast_loop_separate; ++type) {
			if (strcmp(name, option_str[type]))
				continue;
			*found = 1;
			break;
		}
	}
	isl_set_free(set);

	return *found ? isl_stat_error : isl_stat_ok;
}

/* Does "set" encode a loop AST generation option for the isolated part?
 * That is, is of the form
 *
 *	{ [isolate[] -> t[x]] }
 *
 * with t equal to "atomic", "unroll" or "separate"?
 */
static isl_stat is_isolate_loop_type_option(__isl_take isl_set *set, void *user)
{
	int *found = user;
	const char *name;
	enum isl_ast_loop_type type;
	isl_map *map;

	if (!isl_set_is_wrapping(set)) {
		isl_set_free(set);
		return isl_stat_ok;
	}
	map = isl_set_unwrap(set);
	if (!isl_map_has_tuple_name(map, isl_dim_in) ||
	    !isl_map_has_tuple_name(map, isl_dim_out)) {
		isl_map_free(map);
		return isl_stat_ok;
	}
	name = isl_map_get_tuple_name(map, isl_dim_in);
	if (!strcmp(name, "isolate")) {
		name = isl_map_get_tuple_name(map, isl_dim_out);
		for (type = isl_ast_loop_atomic;
		    type <= isl_ast_loop_separate; ++type) {
			if (strcmp(name, option_str[type]))
				continue;
			*found = 1;
			break;
		}
	}
	isl_map_free(map);

	return *found ? isl_stat_error : isl_stat_ok;
}

/* Does "options" encode any loop AST generation options
 * for the isolated part?
 */
static int has_isolate_loop_type_options(__isl_keep isl_union_set *options)
{
	return has_any(options, &is_isolate_loop_type_option);
}

/* Does "options" encode any loop AST generation options?
 */
static int has_loop_type_options(__isl_keep isl_union_set *options)
{
	return has_any(options, &is_loop_type_option);
}

/* Extract the loop AST generation type for the band member
 * at position "pos" from "options".
 * If "isolate" is set, then extract the loop types for the isolated part.
 */
static enum isl_ast_loop_type extract_loop_type(
	__isl_keep isl_union_set *options, int pos, int isolate)
{
	isl_ctx *ctx;
	enum isl_ast_loop_type type, res = isl_ast_loop_default;

	ctx = isl_union_set_get_ctx(options);
	for (type = isl_ast_loop_atomic;
	    type <= isl_ast_loop_separate; ++type) {
		isl_space *space;
		isl_set *option;
		int empty;

		space = isl_union_set_get_space(options);
		space = loop_type_space(space, type, isolate);
		option = isl_union_set_extract_set(options, space);
		option = isl_set_fix_si(option, isl_dim_set, 0, pos);
		empty = isl_set_is_empty(option);
		isl_set_free(option);

		if (empty < 0)
			return isl_ast_loop_error;
		if (empty)
			continue;
		if (res != isl_ast_loop_default)
			isl_die(ctx, isl_error_invalid,
				"conflicting loop type options",
				return isl_ast_loop_error);
		res = type;
	}

	return res;
}

/* Extract the loop AST generation types for the members of "band"
 * from "options" and store them in band->loop_type.
 * Return -1 on error.
 */
static int extract_loop_types(__isl_keep isl_schedule_band *band,
	__isl_keep isl_union_set *options)
{
	int i;

	if (!band->loop_type) {
		isl_ctx *ctx = isl_schedule_band_get_ctx(band);
		band->loop_type = isl_alloc_array(ctx,
					    enum isl_ast_loop_type, band->n);
		if (band->n && !band->loop_type)
			return -1;
	}
	for (i = 0; i < band->n; ++i) {
		band->loop_type[i] = extract_loop_type(options, i, 0);
		if (band->loop_type[i] == isl_ast_loop_error)
			return -1;
	}

	return 0;
}

/* Extract the loop AST generation types for the members of "band"
 * from "options" for the isolated part and
 * store them in band->isolate_loop_type.
 * Return -1 on error.
 */
static int extract_isolate_loop_types(__isl_keep isl_schedule_band *band,
	__isl_keep isl_union_set *options)
{
	int i;

	if (!band->isolate_loop_type) {
		isl_ctx *ctx = isl_schedule_band_get_ctx(band);
		band->isolate_loop_type = isl_alloc_array(ctx,
					    enum isl_ast_loop_type, band->n);
		if (band->n && !band->isolate_loop_type)
			return -1;
	}
	for (i = 0; i < band->n; ++i) {
		band->isolate_loop_type[i] = extract_loop_type(options, i, 1);
		if (band->isolate_loop_type[i] == isl_ast_loop_error)
			return -1;
	}

	return 0;
}

/* Construct universe sets of the spaces that encode loop AST generation
 * types (for the isolated part if "isolate" is set).  That is, construct
 *
 *	{ atomic[x]; separate[x]; unroll[x] }
 *
 * or
 *
 *	{ [isolate[] -> atomic[x]]; [isolate[] -> separate[x]];
 *	  [isolate[] -> unroll[x]] }
 */
static __isl_give isl_union_set *loop_types(__isl_take isl_space *space,
	int isolate)
{
	enum isl_ast_loop_type type;
	isl_union_set *types;

	types = isl_union_set_empty(space);
	for (type = isl_ast_loop_atomic;
	    type <= isl_ast_loop_separate; ++type) {
		isl_set *set;

		space = isl_union_set_get_space(types);
		space = loop_type_space(space, type, isolate);
		set = isl_set_universe(space);
		types = isl_union_set_add_set(types, set);
	}

	return types;
}

/* Remove all elements from spaces that encode loop AST generation types
 * from "options".
 */
static __isl_give isl_union_set *clear_loop_types(
	__isl_take isl_union_set *options)
{
	isl_union_set *types;

	types = loop_types(isl_union_set_get_space(options), 0);
	options = isl_union_set_subtract(options, types);

	return options;
}

/* Remove all elements from spaces that encode loop AST generation types
 * for the isolated part from "options".
 */
static __isl_give isl_union_set *clear_isolate_loop_types(
	__isl_take isl_union_set *options)
{
	isl_union_set *types;

	types = loop_types(isl_union_set_get_space(options), 1);
	options = isl_union_set_subtract(options, types);

	return options;
}

/* Replace the AST build options associated to "band" by "options".
 * If there are any loop AST generation type options, then they
 * are extracted and stored in band->loop_type.  Otherwise,
 * band->loop_type is removed to indicate that the default applies
 * to all members.  Similarly for the loop AST generation type options
 * for the isolated part, which are stored in band->isolate_loop_type.
 * The remaining options are stored in band->ast_build_options.
 *
 * Set anchored if the options include an isolate option since the
 * domain of the wrapped map references the outer band node schedules.
 */
__isl_give isl_schedule_band *isl_schedule_band_set_ast_build_options(
	__isl_take isl_schedule_band *band, __isl_take isl_union_set *options)
{
	int has_isolate, has_loop_type, has_isolate_loop_type;

	band = isl_schedule_band_cow(band);
	if (!band || !options)
		goto error;
	has_isolate = has_isolate_option(options);
	if (has_isolate < 0)
		goto error;
	has_loop_type = has_loop_type_options(options);
	if (has_loop_type < 0)
		goto error;
	has_isolate_loop_type = has_isolate_loop_type_options(options);
	if (has_isolate_loop_type < 0)
		goto error;

	if (!has_loop_type) {
		free(band->loop_type);
		band->loop_type = NULL;
	} else {
		if (extract_loop_types(band, options) < 0)
			goto error;
		options = clear_loop_types(options);
		if (!options)
			goto error;
	}

	if (!has_isolate_loop_type) {
		free(band->isolate_loop_type);
		band->isolate_loop_type = NULL;
	} else {
		if (extract_isolate_loop_types(band, options) < 0)
			goto error;
		options = clear_isolate_loop_types(options);
		if (!options)
			goto error;
	}

	isl_union_set_free(band->ast_build_options);
	band->ast_build_options = options;
	band->anchored = has_isolate;

	return band;
error:
	isl_schedule_band_free(band);
	isl_union_set_free(options);
	return NULL;
}

/* Return the "isolate" option associated to "band", assuming
 * it at appears at schedule depth "depth".
 *
 * The isolate option is of the form
 *
 *	isolate[[flattened outer bands] -> band]
 */
__isl_give isl_set *isl_schedule_band_get_ast_isolate_option(
	__isl_keep isl_schedule_band *band, int depth)
{
	isl_space *space;
	isl_set *isolate;

	if (!band)
		return NULL;

	space = isl_schedule_band_get_space(band);
	space = isl_space_from_range(space);
	space = isl_space_add_dims(space, isl_dim_in, depth);
	space = isl_space_wrap(space);
	space = isl_space_set_tuple_name(space, isl_dim_set, "isolate");

	isolate = isl_union_set_extract_set(band->ast_build_options, space);

	return isolate;
}

/* Replace the option "drop" in the AST build options by "add".
 * That is, remove "drop" and add "add".
 */
__isl_give isl_schedule_band *isl_schedule_band_replace_ast_build_option(
	__isl_take isl_schedule_band *band, __isl_take isl_set *drop,
	__isl_take isl_set *add)
{
	isl_union_set *options;

	band = isl_schedule_band_cow(band);
	if (!band)
		return NULL;

	options = band->ast_build_options;
	options = isl_union_set_subtract(options, isl_union_set_from_set(drop));
	options = isl_union_set_union(options, isl_union_set_from_set(add));
	band->ast_build_options = options;

	if (!band->ast_build_options)
		return isl_schedule_band_free(band);

	return band;
}

/* Multiply the partial schedule of "band" with the factors in "mv".
 * Replace the result by its greatest integer part to ensure
 * that the schedule is always integral.
 */
__isl_give isl_schedule_band *isl_schedule_band_scale(
	__isl_take isl_schedule_band *band, __isl_take isl_multi_val *mv)
{
	band = isl_schedule_band_cow(band);
	if (!band || !mv)
		goto error;
	band->mupa = isl_multi_union_pw_aff_scale_multi_val(band->mupa, mv);
	band->mupa = isl_multi_union_pw_aff_floor(band->mupa);
	if (!band->mupa)
		return isl_schedule_band_free(band);
	return band;
error:
	isl_schedule_band_free(band);
	isl_multi_val_free(mv);
	return NULL;
}

/* Divide the partial schedule of "band" by the factors in "mv".
 * Replace the result by its greatest integer part to ensure
 * that the schedule is always integral.
 */
__isl_give isl_schedule_band *isl_schedule_band_scale_down(
	__isl_take isl_schedule_band *band, __isl_take isl_multi_val *mv)
{
	band = isl_schedule_band_cow(band);
	if (!band || !mv)
		goto error;
	band->mupa = isl_multi_union_pw_aff_scale_down_multi_val(band->mupa,
								mv);
	band->mupa = isl_multi_union_pw_aff_floor(band->mupa);
	if (!band->mupa)
		return isl_schedule_band_free(band);
	return band;
error:
	isl_schedule_band_free(band);
	isl_multi_val_free(mv);
	return NULL;
}

/* Reduce the partial schedule of "band" modulo the factors in "mv".
 */
__isl_give isl_schedule_band *isl_schedule_band_mod(
	__isl_take isl_schedule_band *band, __isl_take isl_multi_val *mv)
{
	band = isl_schedule_band_cow(band);
	if (!band || !mv)
		goto error;
	band->mupa = isl_multi_union_pw_aff_mod_multi_val(band->mupa, mv);
	if (!band->mupa)
		return isl_schedule_band_free(band);
	return band;
error:
	isl_schedule_band_free(band);
	isl_multi_val_free(mv);
	return NULL;
}

/* Shift the partial schedule of "band" by "shift" after checking
 * that the domain of the partial schedule would not be affected
 * by this shift.
 */
__isl_give isl_schedule_band *isl_schedule_band_shift(
	__isl_take isl_schedule_band *band,
	__isl_take isl_multi_union_pw_aff *shift)
{
	isl_union_set *dom1, *dom2;
	isl_bool subset;

	band = isl_schedule_band_cow(band);
	if (!band || !shift)
		goto error;
	dom1 = isl_multi_union_pw_aff_domain(
				isl_multi_union_pw_aff_copy(band->mupa));
	dom2 = isl_multi_union_pw_aff_domain(
				isl_multi_union_pw_aff_copy(shift));
	subset = isl_union_set_is_subset(dom1, dom2);
	isl_union_set_free(dom1);
	isl_union_set_free(dom2);
	if (subset < 0)
		goto error;
	if (!subset)
		isl_die(isl_schedule_band_get_ctx(band), isl_error_invalid,
			"domain of shift needs to include domain of "
			"partial schedule", goto error);
	band->mupa = isl_multi_union_pw_aff_add(band->mupa, shift);
	if (!band->mupa)
		return isl_schedule_band_free(band);
	return band;
error:
	isl_schedule_band_free(band);
	isl_multi_union_pw_aff_free(shift);
	return NULL;
}

/* Given the schedule of a band, construct the corresponding
 * schedule for the tile loops based on the given tile sizes
 * and return the result.
 *
 * If the scale tile loops options is set, then the tile loops
 * are scaled by the tile sizes.
 *
 * That is replace each schedule dimension "i" by either
 * "floor(i/s)" or "s * floor(i/s)".
 */
static isl_multi_union_pw_aff *isl_multi_union_pw_aff_tile(
	__isl_take isl_multi_union_pw_aff *sched,
	__isl_take isl_multi_val *sizes)
{
	isl_ctx *ctx;
	int i, n;
	isl_val *v;
	int scale;

	ctx = isl_multi_val_get_ctx(sizes);
	scale = isl_options_get_tile_scale_tile_loops(ctx);

	n = isl_multi_union_pw_aff_dim(sched, isl_dim_set);
	for (i = 0; i < n; ++i) {
		isl_union_pw_aff *upa;

		upa = isl_multi_union_pw_aff_get_union_pw_aff(sched, i);
		v = isl_multi_val_get_val(sizes, i);

		upa = isl_union_pw_aff_scale_down_val(upa, isl_val_copy(v));
		upa = isl_union_pw_aff_floor(upa);
		if (scale)
			upa = isl_union_pw_aff_scale_val(upa, isl_val_copy(v));
		isl_val_free(v);

		sched = isl_multi_union_pw_aff_set_union_pw_aff(sched, i, upa);
	}

	isl_multi_val_free(sizes);
	return sched;
}

/* Replace "band" by a band corresponding to the tile loops of a tiling
 * with the given tile sizes.
 */
__isl_give isl_schedule_band *isl_schedule_band_tile(
	__isl_take isl_schedule_band *band, __isl_take isl_multi_val *sizes)
{
	band = isl_schedule_band_cow(band);
	if (!band || !sizes)
		goto error;
	band->mupa = isl_multi_union_pw_aff_tile(band->mupa, sizes);
	if (!band->mupa)
		return isl_schedule_band_free(band);
	return band;
error:
	isl_schedule_band_free(band);
	isl_multi_val_free(sizes);
	return NULL;
}

/* Replace "band" by a band corresponding to the point loops of a tiling
 * with the given tile sizes.
 * "tile" is the corresponding tile loop band.
 *
 * If the shift point loops option is set, then the point loops
 * are shifted to start at zero.  That is, each schedule dimension "i"
 * is replaced by "i - s * floor(i/s)".
 * The expression "floor(i/s)" (or "s * floor(i/s)") is extracted from
 * the tile band.
 *
 * Otherwise, the band is left untouched.
 */
__isl_give isl_schedule_band *isl_schedule_band_point(
	__isl_take isl_schedule_band *band, __isl_keep isl_schedule_band *tile,
	__isl_take isl_multi_val *sizes)
{
	isl_ctx *ctx;
	isl_multi_union_pw_aff *scaled;

	if (!band || !sizes)
		goto error;

	ctx = isl_schedule_band_get_ctx(band);
	if (!isl_options_get_tile_shift_point_loops(ctx)) {
		isl_multi_val_free(sizes);
		return band;
	}
	band = isl_schedule_band_cow(band);
	if (!band)
		goto error;

	scaled = isl_schedule_band_get_partial_schedule(tile);
	if (!isl_options_get_tile_scale_tile_loops(ctx))
		scaled = isl_multi_union_pw_aff_scale_multi_val(scaled, sizes);
	else
		isl_multi_val_free(sizes);
	band->mupa = isl_multi_union_pw_aff_sub(band->mupa, scaled);
	if (!band->mupa)
		return isl_schedule_band_free(band);
	return band;
error:
	isl_schedule_band_free(band);
	isl_multi_val_free(sizes);
	return NULL;
}

/* Drop the "n" dimensions starting at "pos" from "band".
 *
 * We apply the transformation even if "n" is zero to ensure consistent
 * behavior with respect to changes in the schedule space.
 *
 * The caller is responsible for updating the isolate option.
 */
__isl_give isl_schedule_band *isl_schedule_band_drop(
	__isl_take isl_schedule_band *band, int pos, int n)
{
	int i;

	if (pos < 0 || n < 0 || pos + n > band->n)
		isl_die(isl_schedule_band_get_ctx(band), isl_error_internal,
			"range out of bounds",
			return isl_schedule_band_free(band));

	band = isl_schedule_band_cow(band);
	if (!band)
		return NULL;

	band->mupa = isl_multi_union_pw_aff_drop_dims(band->mupa,
							isl_dim_set, pos, n);
	if (!band->mupa)
		return isl_schedule_band_free(band);

	for (i = pos + n; i < band->n; ++i)
		band->coincident[i - n] = band->coincident[i];
	if (band->loop_type)
		for (i = pos + n; i < band->n; ++i)
			band->loop_type[i - n] = band->loop_type[i];
	if (band->isolate_loop_type)
		for (i = pos + n; i < band->n; ++i)
			band->isolate_loop_type[i - n] =
						    band->isolate_loop_type[i];

	band->n -= n;

	return band;
}

/* Reset the user pointer on all identifiers of parameters and tuples
 * in "band".
 */
__isl_give isl_schedule_band *isl_schedule_band_reset_user(
	__isl_take isl_schedule_band *band)
{
	band = isl_schedule_band_cow(band);
	if (!band)
		return NULL;

	band->mupa = isl_multi_union_pw_aff_reset_user(band->mupa);
	band->ast_build_options =
		isl_union_set_reset_user(band->ast_build_options);
	if (!band->mupa || !band->ast_build_options)
		return isl_schedule_band_free(band);

	return band;
}

/* Align the parameters of "band" to those of "space".
 */
__isl_give isl_schedule_band *isl_schedule_band_align_params(
	__isl_take isl_schedule_band *band, __isl_take isl_space *space)
{
	band = isl_schedule_band_cow(band);
	if (!band || !space)
		goto error;

	band->mupa = isl_multi_union_pw_aff_align_params(band->mupa,
						isl_space_copy(space));
	band->ast_build_options =
		isl_union_set_align_params(band->ast_build_options, space);
	if (!band->mupa || !band->ast_build_options)
		return isl_schedule_band_free(band);

	return band;
error:
	isl_space_free(space);
	isl_schedule_band_free(band);
	return NULL;
}

/* Compute the pullback of "band" by the function represented by "upma".
 * In other words, plug in "upma" in the iteration domains of "band".
 */
__isl_give isl_schedule_band *isl_schedule_band_pullback_union_pw_multi_aff(
	__isl_take isl_schedule_band *band,
	__isl_take isl_union_pw_multi_aff *upma)
{
	band = isl_schedule_band_cow(band);
	if (!band || !upma)
		goto error;

	band->mupa =
		isl_multi_union_pw_aff_pullback_union_pw_multi_aff(band->mupa,
									upma);
	if (!band->mupa)
		return isl_schedule_band_free(band);

	return band;
error:
	isl_union_pw_multi_aff_free(upma);
	isl_schedule_band_free(band);
	return NULL;
}

/* Compute the gist of "band" with respect to "context".
 * In particular, compute the gist of the associated partial schedule.
 */
__isl_give isl_schedule_band *isl_schedule_band_gist(
	__isl_take isl_schedule_band *band, __isl_take isl_union_set *context)
{
	if (!band || !context)
		goto error;
	if (band->n == 0) {
		isl_union_set_free(context);
		return band;
	}
	band = isl_schedule_band_cow(band);
	if (!band)
		goto error;
	band->mupa = isl_multi_union_pw_aff_gist(band->mupa, context);
	if (!band->mupa)
		return isl_schedule_band_free(band);
	return band;
error:
	isl_union_set_free(context);
	isl_schedule_band_free(band);
	return NULL;
}
