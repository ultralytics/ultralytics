/*
 * Copyright 2010      INRIA Saclay
 * Copyright 2013      Ecole Normale Superieure
 * Copyright 2015      INRIA Paris-Rocquencourt
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 * and INRIA Paris-Rocquencourt, Domaine de Voluceau, Rocquenqourt, B.P. 105,
 * 78153 Le Chesnay Cedex France
 */

#include <isl_hash_private.h>
#include <isl_union_macro.h>

/* A group of expressions defined over the same domain space "domain_space".
 * The entries of "part_table" are the individual expressions,
 * keyed on the entire space of the expression.
 *
 * Each UNION has its own groups, so there can only ever be a single
 * reference to each group.
 */
S(UNION,group) {
	isl_space *domain_space;
	struct isl_hash_table	part_table;
};

/* A union of expressions defined over different disjoint domains.
 * "space" describes the parameters.
 * The entries of "table" are keyed on the domain space of the entry and
 * contain groups of expressions that are defined over the same domain space.
 */
struct UNION {
	int ref;
	isl_space *space;

	struct isl_hash_table	table;
};

/* Internal data structure for isl_union_*_foreach_group.
 * "fn" is the function that needs to be called on each group.
 */
S(UNION,foreach_group_data)
{
	isl_stat (*fn)(__isl_keep S(UNION,group) *group, void *user);
	void *user;
};

/* Call data->fn on the group stored at *entry.
 */
static isl_stat FN(UNION,call_on_group)(void **entry, void *user)
{
	S(UNION,group) *group = *entry;
	S(UNION,foreach_group_data) *data;

	data = (S(UNION,foreach_group_data) *) user;
	return data->fn(group, data->user);
}

/* Call "fn" on each group of expressions in "u".
 */
static isl_stat FN(UNION,foreach_group)(__isl_keep UNION *u,
	isl_stat (*fn)(__isl_keep S(UNION,group) *group, void *user),
	void *user)
{
	S(UNION,foreach_group_data) data = { fn, user };

	if (!u)
		return isl_stat_error;

	return isl_hash_table_foreach(u->space->ctx, &u->table,
				      &FN(UNION,call_on_group), &data);
}

/* A isl_union_*_foreach_group callback for counting the total number
 * of expressions in a UNION.  Add the number of expressions in "group"
 * to *n.
 */
static isl_stat FN(UNION,count_part)(__isl_keep S(UNION,group) *group,
	void *user)
{
	int *n = user;

	if (!group)
		return isl_stat_error;

	*n += group->part_table.n;
	return isl_stat_ok;
}

/* Return the number of base expressions in "u".
 */
int FN(FN(UNION,n),PARTS)(__isl_keep UNION *u)
{
	int n;

	n = 0;
	if (FN(UNION,foreach_group)(u, &FN(UNION,count_part), &n) < 0)
		n = -1;
	return n;
}

/* Free an entry in a group of expressions.
 * Each entry in such a group is a single expression.
 */
static isl_stat FN(UNION,free_group_entry)(void **entry, void *user)
{
	PART *part = *entry;

	FN(PART,free)(part);
	return isl_stat_ok;
}

/* Free all memory allocated for "group" and return NULL.
 */
static __isl_null S(UNION,group) *FN(UNION,group_free)(
	__isl_take S(UNION,group) *group)
{
	isl_ctx *ctx;

	if (!group)
		return NULL;

	ctx = isl_space_get_ctx(group->domain_space);
	isl_hash_table_foreach(ctx, &group->part_table,
				&FN(UNION,free_group_entry), NULL);
	isl_hash_table_clear(&group->part_table);
	isl_space_free(group->domain_space);
	free(group);
	return NULL;
}

/* Allocate a group of expressions defined over the same domain space
 * with domain space "domain_space" and initial size "size".
 */
static __isl_give S(UNION,group) *FN(UNION,group_alloc)(
	__isl_take isl_space *domain_space, int size)
{
	isl_ctx *ctx;
	S(UNION,group) *group;

	if (!domain_space)
		return NULL;
	ctx = isl_space_get_ctx(domain_space);
	group = isl_calloc_type(ctx, S(UNION,group));
	if (!group)
		goto error;
	group->domain_space = domain_space;
	if (isl_hash_table_init(ctx, &group->part_table, size) < 0)
		return FN(UNION,group_free)(group);

	return group;
error:
	isl_space_free(domain_space);
	return NULL;
}

/* Is the space of "entry" equal to "space"?
 */
static int FN(UNION,has_space)(const void *entry, const void *val)
{
	PART *part = (PART *) entry;
	isl_space *space = (isl_space *) val;

	return isl_space_is_equal(part->dim, space);
}

/* Return a group equal to "group", but with a single reference.
 * Since all groups have only a single reference, simply return "group".
 */
static __isl_give S(UNION,group) *FN(UNION,group_cow)(
	__isl_take S(UNION,group) *group)
{
	return group;
}

S(UNION,foreach_data)
{
	isl_stat (*fn)(__isl_take PART *part, void *user);
	void *user;
};

static isl_stat FN(UNION,call_on_copy)(void **entry, void *user)
{
	PART *part = *entry;
	S(UNION,foreach_data) *data = (S(UNION,foreach_data) *) user;

	part = FN(PART,copy)(part);
	if (!part)
		return isl_stat_error;
	return data->fn(part, data->user);
}

/* Call data->fn on a copy of each expression in "group".
 */
static isl_stat FN(UNION,group_call_on_copy)(__isl_keep S(UNION,group) *group,
	void *user)
{
	isl_ctx *ctx;

	if (!group)
		return isl_stat_error;

	ctx = isl_space_get_ctx(group->domain_space);
	return isl_hash_table_foreach(ctx, &group->part_table,
				      &FN(UNION,call_on_copy), user);
}

isl_stat FN(FN(UNION,foreach),PARTS)(__isl_keep UNION *u,
	isl_stat (*fn)(__isl_take PART *part, void *user), void *user)
{
	S(UNION,foreach_data) data = { fn, user };

	if (!u)
		return isl_stat_error;

	return FN(UNION,foreach_group)(u, &FN(UNION,group_call_on_copy), &data);
}

/* Is the domain space of the group of expressions at "entry"
 * equal to "space"?
 */
static int FN(UNION,group_has_domain_space)(const void *entry, const void *val)
{
	S(UNION,group) *group = (S(UNION,group) *) entry;
	isl_space *space = (isl_space *) val;

	return isl_space_is_domain_internal(group->domain_space, space);
}

/* Return the entry, if any, in "u" that lives in "space".
 * If "reserve" is set, then an entry is created if it does not exist yet.
 * Return NULL on error and isl_hash_table_entry_none if no entry was found.
 * Note that when "reserve" is set, the function will never return
 * isl_hash_table_entry_none.
 *
 * First look for the group of expressions with the same domain space,
 * creating one if needed.
 * Then look for the expression living in the specified space in that group.
 */
static struct isl_hash_table_entry *FN(UNION,find_part_entry)(
	__isl_keep UNION *u, __isl_keep isl_space *space, int reserve)
{
	isl_ctx *ctx;
	uint32_t hash;
	struct isl_hash_table_entry *group_entry, *part_entry;
	S(UNION,group) *group;

	if (!u || !space)
		return NULL;

	ctx = FN(UNION,get_ctx)(u);
	hash = isl_space_get_domain_hash(space);
	group_entry = isl_hash_table_find(ctx, &u->table, hash,
			    &FN(UNION,group_has_domain_space), space, reserve);
	if (!group_entry)
		return reserve ? NULL : isl_hash_table_entry_none;
	if (reserve && !group_entry->data) {
		isl_space *domain = isl_space_domain(isl_space_copy(space));
		group = FN(UNION,group_alloc)(domain, 1);
		group_entry->data = group;
	} else {
		group = group_entry->data;
		if (reserve)
			group = FN(UNION,group_cow)(group);
	}
	if (!group)
		return NULL;
	hash = isl_space_get_hash(space);
	part_entry = isl_hash_table_find(ctx, &group->part_table, hash,
				&FN(UNION,has_space), space, reserve);
	if (!reserve && !part_entry)
		return isl_hash_table_entry_none;
	return part_entry;
}

/* Remove "part_entry" from the hash table of "u".
 *
 * First look the group_entry in "u" holding the group that
 * contains "part_entry".  Remove "part_entry" from that group.
 * If the group becomes empty, then also remove the group_entry from "u".
 */
static __isl_give UNION *FN(UNION,remove_part_entry)(__isl_take UNION *u,
	struct isl_hash_table_entry *part_entry)
{
	isl_ctx *ctx;
	uint32_t hash;
	PART *part;
	struct isl_hash_table_entry *group_entry;
	S(UNION,group) *group;

	if (!u || !part_entry)
		return FN(UNION,free)(u);

	part = part_entry->data;
	ctx = FN(UNION,get_ctx)(u);
	hash = isl_space_get_domain_hash(part->dim);
	group_entry = isl_hash_table_find(ctx, &u->table, hash,
			    &FN(UNION,group_has_domain_space), part->dim, 0);
	if (!group_entry)
		isl_die(ctx, isl_error_internal, "missing group",
			return FN(UNION,free)(u));
	group = group_entry->data;
	isl_hash_table_remove(ctx, &group->part_table, part_entry);
	FN(PART,free)(part);

	if (group->part_table.n != 0)
		return u;

	isl_hash_table_remove(ctx, &u->table, group_entry);
	FN(UNION,group_free)(group);

	return u;
}

/* Are the domains of "part1" and "part2" disjoint?
 */
static isl_bool FN(UNION,disjoint_domain)(__isl_keep PART *part1,
	__isl_keep PART *part2)
{
	isl_set *dom1, *dom2;
	isl_bool disjoint;

	if (!part1 || !part2)
		return isl_bool_error;
	dom1 = FN(PART,domain)(FN(PART,copy)(part1));
	dom2 = FN(PART,domain)(FN(PART,copy)(part2));
	disjoint = isl_set_is_disjoint(dom1, dom2);
	isl_set_free(dom1);
	isl_set_free(dom2);

	return disjoint;
}

/* Check that the expression at *entry has a domain that is disjoint
 * from that of "part", unless they also have the same target space.
 */
static isl_stat FN(UNION,check_disjoint_domain_entry)(void **entry, void *user)
{
	PART *part = user;
	PART *other = *entry;
	isl_bool equal;
	isl_bool disjoint;

	equal = isl_space_is_equal(part->dim, other->dim);
	if (equal < 0)
		return isl_stat_error;
	if (equal)
		return isl_stat_ok;

	disjoint = FN(UNION,disjoint_domain)(part, other);
	if (disjoint < 0)
		return isl_stat_error;
	if (!disjoint)
		isl_die(FN(PART,get_ctx)(part), isl_error_invalid,
			"overlapping domain with other part",
			return isl_stat_error);
	return isl_stat_ok;
}

/* Check that the domain of "part" is disjoint from the domain of the entries
 * in "u" that are defined on the same domain space, but have a different
 * target space.
 * If there is no group of expressions in "u" with the same domain space,
 * then everything is fine.  Otherwise, check the individual expressions
 * in that group.
 */
static isl_stat FN(UNION,check_disjoint_domain_other)(__isl_keep UNION *u,
	__isl_keep PART *part)
{
	isl_ctx *ctx;
	uint32_t hash;
	struct isl_hash_table_entry *group_entry;
	S(UNION,group) *group;

	if (!u || !part)
		return isl_stat_error;
	ctx = FN(UNION,get_ctx)(u);
	hash = isl_space_get_domain_hash(part->dim);
	group_entry = isl_hash_table_find(ctx, &u->table, hash,
			    &FN(UNION,group_has_domain_space), part->dim, 0);
	if (!group_entry)
		return isl_stat_ok;
	group = group_entry->data;
	return isl_hash_table_foreach(ctx, &group->part_table,
			      &FN(UNION,check_disjoint_domain_entry), part);
}

/* Check that the domain of "part1" is disjoint from the domain of "part2".
 * This check is performed before "part2" is added to a UNION to ensure
 * that the UNION expression remains a function.
 */
static isl_stat FN(UNION,check_disjoint_domain)(__isl_keep PART *part1,
	__isl_keep PART *part2)
{
	isl_bool disjoint;

	disjoint = FN(UNION,disjoint_domain)(part1, part2);
	if (disjoint < 0)
		return isl_stat_error;
	if (!disjoint)
		isl_die(FN(PART,get_ctx)(part1), isl_error_invalid,
			"domain of additional part should be disjoint",
			return isl_stat_error);
	return isl_stat_ok;
}

/* Internal data structure for isl_union_*_foreach_inplace.
 * "fn" is the function that needs to be called on each entry.
 */
S(UNION,foreach_inplace_data)
{
	isl_stat (*fn)(void **entry, void *user);
	void *user;
};

/* isl_union_*_foreach_group callback for calling data->fn on
 * each part entry in the group.
 */
static isl_stat FN(UNION,group_call_inplace)(__isl_keep S(UNION,group) *group,
	void *user)
{
	isl_ctx *ctx;
	S(UNION,foreach_inplace_data) *data;

	if (!group)
		return isl_stat_error;

	data = (S(UNION,foreach_inplace_data) *) user;
	ctx = isl_space_get_ctx(group->domain_space);
	return isl_hash_table_foreach(ctx, &group->part_table,
				      data->fn, data->user);
}

/* Call "fn" on each part entry of "u".
 */
static isl_stat FN(UNION,foreach_inplace)(__isl_keep UNION *u,
	isl_stat (*fn)(void **part, void *user), void *user)
{
	S(UNION,foreach_inplace_data) data = { fn, user };

	return FN(UNION,foreach_group)(u, &FN(UNION,group_call_inplace), &data);
}

/* Does "u" have a single reference?
 * That is, can we change "u" inplace?
 */
static isl_bool FN(UNION,has_single_reference)(__isl_keep UNION *u)
{
	if (!u)
		return isl_bool_error;
	return u->ref == 1;
}

static isl_stat FN(UNION,free_u_entry)(void **entry, void *user)
{
	S(UNION,group) *group = *entry;
	FN(UNION,group_free)(group);
	return isl_stat_ok;
}

#include <isl_union_templ.c>
