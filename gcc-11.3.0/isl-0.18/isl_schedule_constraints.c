/*
 * Copyright 2012      Ecole Normale Superieure
 * Copyright 2015-2016 Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <string.h>

#include <isl_schedule_constraints.h>
#include <isl/schedule.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/union_set.h>
#include <isl/union_map.h>
#include <isl/stream.h>

/* The constraints that need to be satisfied by a schedule on "domain".
 *
 * "context" specifies extra constraints on the parameters.
 *
 * "validity" constraints map domain elements i to domain elements
 * that should be scheduled after i.  (Hard constraint)
 * "proximity" constraints map domain elements i to domains elements
 * that should be scheduled as early as possible after i (or before i).
 * (Soft constraint)
 *
 * "condition" and "conditional_validity" constraints map possibly "tagged"
 * domain elements i -> s to "tagged" domain elements j -> t.
 * The elements of the "conditional_validity" constraints, but without the
 * tags (i.e., the elements i -> j) are treated as validity constraints,
 * except that during the construction of a tilable band,
 * the elements of the "conditional_validity" constraints may be violated
 * provided that all adjacent elements of the "condition" constraints
 * are local within the band.
 * A dependence is local within a band if domain and range are mapped
 * to the same schedule point by the band.
 */
struct isl_schedule_constraints {
	isl_union_set *domain;
	isl_set *context;

	isl_union_map *constraint[isl_edge_last + 1];
};

__isl_give isl_schedule_constraints *isl_schedule_constraints_copy(
	__isl_keep isl_schedule_constraints *sc)
{
	isl_ctx *ctx;
	isl_schedule_constraints *sc_copy;
	enum isl_edge_type i;

	ctx = isl_union_set_get_ctx(sc->domain);
	sc_copy = isl_calloc_type(ctx, struct isl_schedule_constraints);
	if (!sc_copy)
		return NULL;

	sc_copy->domain = isl_union_set_copy(sc->domain);
	sc_copy->context = isl_set_copy(sc->context);
	if (!sc_copy->domain || !sc_copy->context)
		return isl_schedule_constraints_free(sc_copy);

	for (i = isl_edge_first; i <= isl_edge_last; ++i) {
		sc_copy->constraint[i] = isl_union_map_copy(sc->constraint[i]);
		if (!sc_copy->constraint[i])
			return isl_schedule_constraints_free(sc_copy);
	}

	return sc_copy;
}

/* Construct an empty (invalid) isl_schedule_constraints object.
 * The caller is responsible for setting the domain and initializing
 * all the other fields, e.g., by calling isl_schedule_constraints_init.
 */
static __isl_give isl_schedule_constraints *isl_schedule_constraints_alloc(
	isl_ctx *ctx)
{
	return isl_calloc_type(ctx, struct isl_schedule_constraints);
}

/* Initialize all the fields of "sc", except domain, which is assumed
 * to have been set by the caller.
 */
static __isl_give isl_schedule_constraints *isl_schedule_constraints_init(
	__isl_take isl_schedule_constraints *sc)
{
	isl_space *space;
	isl_union_map *empty;
	enum isl_edge_type i;

	if (!sc)
		return NULL;
	if (!sc->domain)
		return isl_schedule_constraints_free(sc);
	space = isl_union_set_get_space(sc->domain);
	if (!sc->context)
		sc->context = isl_set_universe(isl_space_copy(space));
	empty = isl_union_map_empty(space);
	for (i = isl_edge_first; i <= isl_edge_last; ++i) {
		if (sc->constraint[i])
			continue;
		sc->constraint[i] = isl_union_map_copy(empty);
		if (!sc->constraint[i])
			sc->domain = isl_union_set_free(sc->domain);
	}
	isl_union_map_free(empty);

	if (!sc->domain || !sc->context)
		return isl_schedule_constraints_free(sc);

	return sc;
}

/* Construct an isl_schedule_constraints object for computing a schedule
 * on "domain".  The initial object does not impose any constraints.
 */
__isl_give isl_schedule_constraints *isl_schedule_constraints_on_domain(
	__isl_take isl_union_set *domain)
{
	isl_ctx *ctx;
	isl_schedule_constraints *sc;

	if (!domain)
		return NULL;

	ctx = isl_union_set_get_ctx(domain);
	sc = isl_schedule_constraints_alloc(ctx);
	if (!sc)
		goto error;

	sc->domain = domain;
	return isl_schedule_constraints_init(sc);
error:
	isl_union_set_free(domain);
	return NULL;
}

/* Replace the domain of "sc" by "domain".
 */
static __isl_give isl_schedule_constraints *isl_schedule_constraints_set_domain(
	__isl_take isl_schedule_constraints *sc,
	__isl_take isl_union_set *domain)
{
	if (!sc || !domain)
		goto error;

	isl_union_set_free(sc->domain);
	sc->domain = domain;

	return sc;
error:
	isl_schedule_constraints_free(sc);
	isl_union_set_free(domain);
	return NULL;
}

/* Replace the context of "sc" by "context".
 */
__isl_give isl_schedule_constraints *isl_schedule_constraints_set_context(
	__isl_take isl_schedule_constraints *sc, __isl_take isl_set *context)
{
	if (!sc || !context)
		goto error;

	isl_set_free(sc->context);
	sc->context = context;

	return sc;
error:
	isl_schedule_constraints_free(sc);
	isl_set_free(context);
	return NULL;
}

/* Replace the constraints of type "type" in "sc" by "c".
 */
static __isl_give isl_schedule_constraints *isl_schedule_constraints_set(
	__isl_take isl_schedule_constraints *sc, enum isl_edge_type type,
	__isl_take isl_union_map *c)
{
	if (!sc || !c)
		goto error;

	isl_union_map_free(sc->constraint[type]);
	sc->constraint[type] = c;

	return sc;
error:
	isl_schedule_constraints_free(sc);
	isl_union_map_free(c);
	return NULL;
}

/* Replace the validity constraints of "sc" by "validity".
 */
__isl_give isl_schedule_constraints *isl_schedule_constraints_set_validity(
	__isl_take isl_schedule_constraints *sc,
	__isl_take isl_union_map *validity)
{
	return isl_schedule_constraints_set(sc, isl_edge_validity, validity);
}

/* Replace the coincidence constraints of "sc" by "coincidence".
 */
__isl_give isl_schedule_constraints *isl_schedule_constraints_set_coincidence(
	__isl_take isl_schedule_constraints *sc,
	__isl_take isl_union_map *coincidence)
{
	return isl_schedule_constraints_set(sc, isl_edge_coincidence,
						coincidence);
}

/* Replace the proximity constraints of "sc" by "proximity".
 */
__isl_give isl_schedule_constraints *isl_schedule_constraints_set_proximity(
	__isl_take isl_schedule_constraints *sc,
	__isl_take isl_union_map *proximity)
{
	return isl_schedule_constraints_set(sc, isl_edge_proximity, proximity);
}

/* Replace the conditional validity constraints of "sc" by "condition"
 * and "validity".
 */
__isl_give isl_schedule_constraints *
isl_schedule_constraints_set_conditional_validity(
	__isl_take isl_schedule_constraints *sc,
	__isl_take isl_union_map *condition,
	__isl_take isl_union_map *validity)
{
	sc = isl_schedule_constraints_set(sc, isl_edge_condition, condition);
	sc = isl_schedule_constraints_set(sc, isl_edge_conditional_validity,
						validity);
	return sc;
}

__isl_null isl_schedule_constraints *isl_schedule_constraints_free(
	__isl_take isl_schedule_constraints *sc)
{
	enum isl_edge_type i;

	if (!sc)
		return NULL;

	isl_union_set_free(sc->domain);
	isl_set_free(sc->context);
	for (i = isl_edge_first; i <= isl_edge_last; ++i)
		isl_union_map_free(sc->constraint[i]);

	free(sc);

	return NULL;
}

isl_ctx *isl_schedule_constraints_get_ctx(
	__isl_keep isl_schedule_constraints *sc)
{
	return sc ? isl_union_set_get_ctx(sc->domain) : NULL;
}

/* Return the domain of "sc".
 */
__isl_give isl_union_set *isl_schedule_constraints_get_domain(
	__isl_keep isl_schedule_constraints *sc)
{
	if (!sc)
		return NULL;

	return isl_union_set_copy(sc->domain);
}

/* Return the context of "sc".
 */
__isl_give isl_set *isl_schedule_constraints_get_context(
	__isl_keep isl_schedule_constraints *sc)
{
	if (!sc)
		return NULL;

	return isl_set_copy(sc->context);
}

/* Return the constraints of type "type" in "sc".
 */
__isl_give isl_union_map *isl_schedule_constraints_get(
	__isl_keep isl_schedule_constraints *sc, enum isl_edge_type type)
{
	if (!sc)
		return NULL;

	return isl_union_map_copy(sc->constraint[type]);
}

/* Return the validity constraints of "sc".
 */
__isl_give isl_union_map *isl_schedule_constraints_get_validity(
	__isl_keep isl_schedule_constraints *sc)
{
	return isl_schedule_constraints_get(sc, isl_edge_validity);
}

/* Return the coincidence constraints of "sc".
 */
__isl_give isl_union_map *isl_schedule_constraints_get_coincidence(
	__isl_keep isl_schedule_constraints *sc)
{
	return isl_schedule_constraints_get(sc, isl_edge_coincidence);
}

/* Return the proximity constraints of "sc".
 */
__isl_give isl_union_map *isl_schedule_constraints_get_proximity(
	__isl_keep isl_schedule_constraints *sc)
{
	return isl_schedule_constraints_get(sc, isl_edge_proximity);
}

/* Return the conditional validity constraints of "sc".
 */
__isl_give isl_union_map *isl_schedule_constraints_get_conditional_validity(
	__isl_keep isl_schedule_constraints *sc)
{
	return isl_schedule_constraints_get(sc, isl_edge_conditional_validity);
}

/* Return the conditions for the conditional validity constraints of "sc".
 */
__isl_give isl_union_map *
isl_schedule_constraints_get_conditional_validity_condition(
	__isl_keep isl_schedule_constraints *sc)
{
	return isl_schedule_constraints_get(sc, isl_edge_condition);
}

/* Add "c" to the constraints of type "type" in "sc".
 */
__isl_give isl_schedule_constraints *isl_schedule_constraints_add(
	__isl_take isl_schedule_constraints *sc, enum isl_edge_type type,
	__isl_take isl_union_map *c)
{
	if (!sc || !c)
		goto error;

	c = isl_union_map_union(sc->constraint[type], c);
	sc->constraint[type] = c;
	if (!c)
		return isl_schedule_constraints_free(sc);

	return sc;
error:
	isl_schedule_constraints_free(sc);
	isl_union_map_free(c);
	return NULL;
}

/* Can a schedule constraint of type "type" be tagged?
 */
static int may_be_tagged(enum isl_edge_type type)
{
	if (type == isl_edge_condition || type == isl_edge_conditional_validity)
		return 1;
	return 0;
}

/* Apply "umap" to the domains of the wrapped relations
 * inside the domain and range of "c".
 *
 * That is, for each map of the form
 *
 *	[D -> S] -> [E -> T]
 *
 * in "c", apply "umap" to D and E.
 *
 * D is exposed by currying the relation to
 *
 *	D -> [S -> [E -> T]]
 *
 * E is exposed by doing the same to the inverse of "c".
 */
static __isl_give isl_union_map *apply_factor_domain(
	__isl_take isl_union_map *c, __isl_keep isl_union_map *umap)
{
	c = isl_union_map_curry(c);
	c = isl_union_map_apply_domain(c, isl_union_map_copy(umap));
	c = isl_union_map_uncurry(c);

	c = isl_union_map_reverse(c);
	c = isl_union_map_curry(c);
	c = isl_union_map_apply_domain(c, isl_union_map_copy(umap));
	c = isl_union_map_uncurry(c);
	c = isl_union_map_reverse(c);

	return c;
}

/* Apply "umap" to domain and range of "c".
 * If "tag" is set, then "c" may contain tags and then "umap"
 * needs to be applied to the domains of the wrapped relations
 * inside the domain and range of "c".
 */
static __isl_give isl_union_map *apply(__isl_take isl_union_map *c,
	__isl_keep isl_union_map *umap, int tag)
{
	isl_union_map *t;

	if (tag)
		t = isl_union_map_copy(c);
	c = isl_union_map_apply_domain(c, isl_union_map_copy(umap));
	c = isl_union_map_apply_range(c, isl_union_map_copy(umap));
	if (!tag)
		return c;
	t = apply_factor_domain(t, umap);
	c = isl_union_map_union(c, t);
	return c;
}

/* Apply "umap" to the domain of the schedule constraints "sc".
 *
 * The two sides of the various schedule constraints are adjusted
 * accordingly.
 */
__isl_give isl_schedule_constraints *isl_schedule_constraints_apply(
	__isl_take isl_schedule_constraints *sc,
	__isl_take isl_union_map *umap)
{
	enum isl_edge_type i;

	if (!sc || !umap)
		goto error;

	for (i = isl_edge_first; i <= isl_edge_last; ++i) {
		int tag = may_be_tagged(i);

		sc->constraint[i] = apply(sc->constraint[i], umap, tag);
		if (!sc->constraint[i])
			goto error;
	}
	sc->domain = isl_union_set_apply(sc->domain, umap);
	if (!sc->domain)
		return isl_schedule_constraints_free(sc);

	return sc;
error:
	isl_schedule_constraints_free(sc);
	isl_union_map_free(umap);
	return NULL;
}

/* An enumeration of the various keys that may appear in a YAML mapping
 * of an isl_schedule_constraints object.
 * The keys for the edge types are assumed to have the same values
 * as the edge types in isl_edge_type.
 */
enum isl_sc_key {
	isl_sc_key_error = -1,
	isl_sc_key_validity = isl_edge_validity,
	isl_sc_key_coincidence = isl_edge_coincidence,
	isl_sc_key_condition = isl_edge_condition,
	isl_sc_key_conditional_validity = isl_edge_conditional_validity,
	isl_sc_key_proximity = isl_edge_proximity,
	isl_sc_key_domain,
	isl_sc_key_context,
	isl_sc_key_end
};

/* Textual representations of the YAML keys for an isl_schedule_constraints
 * object.
 */
static char *key_str[] = {
	[isl_sc_key_validity] = "validity",
	[isl_sc_key_coincidence] = "coincidence",
	[isl_sc_key_condition] = "condition",
	[isl_sc_key_conditional_validity] = "conditional_validity",
	[isl_sc_key_proximity] = "proximity",
	[isl_sc_key_domain] = "domain",
	[isl_sc_key_context] = "context",
};

/* Print a key, value pair for the edge of type "type" in "sc" to "p".
 */
static __isl_give isl_printer *print_constraint(__isl_take isl_printer *p,
	__isl_keep isl_schedule_constraints *sc, enum isl_edge_type type)
{
	p = isl_printer_print_str(p, key_str[type]);
	p = isl_printer_yaml_next(p);
	p = isl_printer_print_union_map(p, sc->constraint[type]);
	p = isl_printer_yaml_next(p);

	return p;
}

/* Print "sc" to "p"
 *
 * In particular, print the isl_schedule_constraints object as a YAML document.
 */
__isl_give isl_printer *isl_printer_print_schedule_constraints(
	__isl_take isl_printer *p, __isl_keep isl_schedule_constraints *sc)
{
	if (!sc)
		return isl_printer_free(p);

	p = isl_printer_yaml_start_mapping(p);
	p = isl_printer_print_str(p, key_str[isl_sc_key_domain]);
	p = isl_printer_yaml_next(p);
	p = isl_printer_print_union_set(p, sc->domain);
	p = isl_printer_yaml_next(p);
	p = isl_printer_print_str(p, key_str[isl_sc_key_context]);
	p = isl_printer_yaml_next(p);
	p = isl_printer_print_set(p, sc->context);
	p = isl_printer_yaml_next(p);
	p = print_constraint(p, sc, isl_edge_validity);
	p = print_constraint(p, sc, isl_edge_proximity);
	p = print_constraint(p, sc, isl_edge_coincidence);
	p = print_constraint(p, sc, isl_edge_condition);
	p = print_constraint(p, sc, isl_edge_conditional_validity);
	p = isl_printer_yaml_end_mapping(p);

	return p;
}

#undef BASE
#define BASE schedule_constraints
#include <print_templ_yaml.c>

/* Extract a mapping key from the token "tok".
 * Return isl_sc_key_error on error, i.e., if "tok" does not
 * correspond to any known key.
 */
static enum isl_sc_key extract_key(__isl_keep isl_stream *s,
	struct isl_token *tok)
{
	int type;
	char *name;
	isl_ctx *ctx;
	enum isl_sc_key key;

	if (!tok)
		return isl_sc_key_error;
	type = isl_token_get_type(tok);
	if (type != ISL_TOKEN_IDENT && type != ISL_TOKEN_STRING) {
		isl_stream_error(s, tok, "expecting key");
		return isl_sc_key_error;
	}

	ctx = isl_stream_get_ctx(s);
	name = isl_token_get_str(ctx, tok);
	if (!name)
		return isl_sc_key_error;

	for (key = 0; key < isl_sc_key_end; ++key) {
		if (!strcmp(name, key_str[key]))
			break;
	}
	free(name);

	if (key >= isl_sc_key_end)
		isl_die(ctx, isl_error_invalid, "unknown key",
			return isl_sc_key_error);
	return key;
}

/* Read a key from "s" and return the corresponding enum.
 * Return isl_sc_key_error on error, i.e., if the first token
 * on the stream does not correspond to any known key.
 */
static enum isl_sc_key get_key(__isl_keep isl_stream *s)
{
	struct isl_token *tok;
	enum isl_sc_key key;

	tok = isl_stream_next_token(s);
	key = extract_key(s, tok);
	isl_token_free(tok);

	return key;
}

#undef BASE
#define BASE set
#include "read_in_string_templ.c"

#undef BASE
#define BASE union_set
#include "read_in_string_templ.c"

#undef BASE
#define BASE union_map
#include "read_in_string_templ.c"

/* Read an isl_schedule_constraints object from "s".
 *
 * Start off with an empty (invalid) isl_schedule_constraints object and
 * then fill up the fields based on the input.
 * The input needs to contain at least a description of the domain.
 * The other fields are set to defaults by isl_schedule_constraints_init
 * if they are not specified in the input.
 */
__isl_give isl_schedule_constraints *isl_stream_read_schedule_constraints(
	isl_stream *s)
{
	isl_ctx *ctx;
	isl_schedule_constraints *sc;
	int more;
	int domain_set = 0;

	if (isl_stream_yaml_read_start_mapping(s))
		return NULL;

	ctx = isl_stream_get_ctx(s);
	sc = isl_schedule_constraints_alloc(ctx);
	while ((more = isl_stream_yaml_next(s)) > 0) {
		enum isl_sc_key key;
		isl_set *context;
		isl_union_set *domain;
		isl_union_map *constraints;

		key = get_key(s);
		if (isl_stream_yaml_next(s) < 0)
			return isl_schedule_constraints_free(sc);
		switch (key) {
		case isl_sc_key_end:
		case isl_sc_key_error:
			return isl_schedule_constraints_free(sc);
		case isl_sc_key_domain:
			domain_set = 1;
			domain = read_union_set(s);
			sc = isl_schedule_constraints_set_domain(sc, domain);
			if (!sc)
				return NULL;
			break;
		case isl_sc_key_context:
			context = read_set(s);
			sc = isl_schedule_constraints_set_context(sc, context);
			if (!sc)
				return NULL;
			break;
		default:
			constraints = read_union_map(s);
			sc = isl_schedule_constraints_set(sc, key, constraints);
			if (!sc)
				return NULL;
			break;
		}
	}
	if (more < 0)
		return isl_schedule_constraints_free(sc);

	if (isl_stream_yaml_read_end_mapping(s) < 0) {
		isl_stream_error(s, NULL, "unexpected extra elements");
		return isl_schedule_constraints_free(sc);
	}

	if (!domain_set) {
		isl_stream_error(s, NULL, "no domain specified");
		return isl_schedule_constraints_free(sc);
	}

	return isl_schedule_constraints_init(sc);
}

/* Read an isl_schedule_constraints object from the file "input".
 */
__isl_give isl_schedule_constraints *isl_schedule_constraints_read_from_file(
	isl_ctx *ctx, FILE *input)
{
	struct isl_stream *s;
	isl_schedule_constraints *sc;

	s = isl_stream_new_file(ctx, input);
	if (!s)
		return NULL;
	sc = isl_stream_read_schedule_constraints(s);
	isl_stream_free(s);

	return sc;
}

/* Read an isl_schedule_constraints object from the string "str".
 */
__isl_give isl_schedule_constraints *isl_schedule_constraints_read_from_str(
	isl_ctx *ctx, const char *str)
{
	struct isl_stream *s;
	isl_schedule_constraints *sc;

	s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	sc = isl_stream_read_schedule_constraints(s);
	isl_stream_free(s);

	return sc;
}

/* Align the parameters of the fields of "sc".
 */
__isl_give isl_schedule_constraints *
isl_schedule_constraints_align_params(__isl_take isl_schedule_constraints *sc)
{
	isl_space *space;
	enum isl_edge_type i;

	if (!sc)
		return NULL;

	space = isl_union_set_get_space(sc->domain);
	space = isl_space_align_params(space, isl_set_get_space(sc->context));
	for (i = isl_edge_first; i <= isl_edge_last; ++i)
		space = isl_space_align_params(space,
				    isl_union_map_get_space(sc->constraint[i]));

	for (i = isl_edge_first; i <= isl_edge_last; ++i) {
		sc->constraint[i] = isl_union_map_align_params(
				    sc->constraint[i], isl_space_copy(space));
		if (!sc->constraint[i])
			space = isl_space_free(space);
	}
	sc->context = isl_set_align_params(sc->context, isl_space_copy(space));
	sc->domain = isl_union_set_align_params(sc->domain, space);
	if (!sc->context || !sc->domain)
		return isl_schedule_constraints_free(sc);

	return sc;
}

/* Add the number of basic maps in "map" to *n.
 */
static isl_stat add_n_basic_map(__isl_take isl_map *map, void *user)
{
	int *n = user;

	*n += isl_map_n_basic_map(map);
	isl_map_free(map);

	return isl_stat_ok;
}

/* Return the total number of isl_basic_maps in the constraints of "sc".
 * Return -1 on error.
 */
int isl_schedule_constraints_n_basic_map(
	__isl_keep isl_schedule_constraints *sc)
{
	enum isl_edge_type i;
	int n = 0;

	if (!sc)
		return -1;
	for (i = isl_edge_first; i <= isl_edge_last; ++i)
		if (isl_union_map_foreach_map(sc->constraint[i],
						&add_n_basic_map, &n) < 0)
			return -1;

	return n;
}

/* Return the total number of isl_maps in the constraints of "sc".
 */
int isl_schedule_constraints_n_map(__isl_keep isl_schedule_constraints *sc)
{
	enum isl_edge_type i;
	int n = 0;

	for (i = isl_edge_first; i <= isl_edge_last; ++i)
		n += isl_union_map_n_map(sc->constraint[i]);

	return n;
}
