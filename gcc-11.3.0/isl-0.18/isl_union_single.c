/*
 * Copyright 2010      INRIA Saclay
 * Copyright 2013      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl_hash_private.h>
#include <isl_union_macro.h>

/* A union of expressions defined over different domain spaces.
 * "space" describes the parameters.
 * The entries of "table" are keyed on the domain space of the entry.
 */
struct UNION {
	int ref;
#ifdef HAS_TYPE
	enum isl_fold type;
#endif
	isl_space *space;

	struct isl_hash_table	table;
};

/* Return the number of base expressions in "u".
 */
int FN(FN(UNION,n),PARTS)(__isl_keep UNION *u)
{
	return u ? u->table.n : 0;
}

S(UNION,foreach_data)
{
	isl_stat (*fn)(__isl_take PART *part, void *user);
	void *user;
};

static isl_stat FN(UNION,call_on_copy)(void **entry, void *user)
{
	PART *part = *entry;
	S(UNION,foreach_data) *data = (S(UNION,foreach_data) *)user;

	part = FN(PART,copy)(part);
	if (!part)
		return isl_stat_error;
	return data->fn(part, data->user);
}

isl_stat FN(FN(UNION,foreach),PARTS)(__isl_keep UNION *u,
	isl_stat (*fn)(__isl_take PART *part, void *user), void *user)
{
	S(UNION,foreach_data) data = { fn, user };

	if (!u)
		return isl_stat_error;

	return isl_hash_table_foreach(u->space->ctx, &u->table,
				      &FN(UNION,call_on_copy), &data);
}

/* Is the domain space of "entry" equal to the domain of "space"?
 */
static int FN(UNION,has_same_domain_space)(const void *entry, const void *val)
{
	PART *part = (PART *)entry;
	isl_space *space = (isl_space *) val;

	if (isl_space_is_set(space))
		return isl_space_is_set(part->dim);

	return isl_space_tuple_is_equal(part->dim, isl_dim_in,
					space, isl_dim_in);
}

/* Return the entry, if any, in "u" that lives in "space".
 * If "reserve" is set, then an entry is created if it does not exist yet.
 * Return NULL on error and isl_hash_table_entry_none if no entry was found.
 * Note that when "reserve" is set, the function will never return
 * isl_hash_table_entry_none.
 *
 * First look for the entry (if any) with the same domain space.
 * If it exists, then check if the range space also matches.
 */
static struct isl_hash_table_entry *FN(UNION,find_part_entry)(
	__isl_keep UNION *u, __isl_keep isl_space *space, int reserve)
{
	isl_ctx *ctx;
	uint32_t hash;
	struct isl_hash_table_entry *entry;
	isl_bool equal;
	PART *part;

	if (!u || !space)
		return NULL;

	ctx = FN(UNION,get_ctx)(u);
	hash = isl_space_get_domain_hash(space);
	entry = isl_hash_table_find(ctx, &u->table, hash,
			&FN(UNION,has_same_domain_space), space, reserve);
	if (!entry)
		return reserve ? NULL : isl_hash_table_entry_none;
	if (reserve && !entry->data)
		return entry;
	part = entry->data;
	equal = isl_space_tuple_is_equal(part->dim, isl_dim_out,
					    space, isl_dim_out);
	if (equal < 0)
		return NULL;
	if (equal)
		return entry;
	if (!reserve)
		return isl_hash_table_entry_none;
	isl_die(FN(UNION,get_ctx)(u), isl_error_invalid,
		"union expression can only contain a single "
		"expression over a given domain", return NULL);
}

/* Remove "part_entry" from the hash table of "u".
 */
static __isl_give UNION *FN(UNION,remove_part_entry)(__isl_take UNION *u,
	struct isl_hash_table_entry *part_entry)
{
	isl_ctx *ctx;

	if (!u || !part_entry)
		return FN(UNION,free)(u);

	ctx = FN(UNION,get_ctx)(u);
	isl_hash_table_remove(ctx, &u->table, part_entry);
	FN(PART,free)(part_entry->data);

	return u;
}

/* Check that the domain of "part" is disjoint from the domain of the entries
 * in "u" that are defined on the same domain space, but have a different
 * target space.
 * Since a UNION with a single entry per domain space is not allowed
 * to contain two entries with the same domain space, there cannot be
 * any such other entry.
 */
static isl_stat FN(UNION,check_disjoint_domain_other)(__isl_keep UNION *u,
	__isl_keep PART *part)
{
	return isl_stat_ok;
}

/* Check that the domain of "part1" is disjoint from the domain of "part2".
 * This check is performed before "part2" is added to a UNION to ensure
 * that the UNION expression remains a function.
 * Since a UNION with a single entry per domain space is not allowed
 * to contain two entries with the same domain space, fail unconditionally.
 */
static isl_stat FN(UNION,check_disjoint_domain)(__isl_keep PART *part1,
	__isl_keep PART *part2)
{
	isl_die(FN(PART,get_ctx)(part1), isl_error_invalid,
		"additional part should live on separate space",
		return isl_stat_error);
}

/* Call "fn" on each part entry of "u".
 */
static isl_stat FN(UNION,foreach_inplace)(__isl_keep UNION *u,
	isl_stat (*fn)(void **part, void *user), void *user)
{
	isl_ctx *ctx;

	if (!u)
		return isl_stat_error;
	ctx = FN(UNION,get_ctx)(u);
	return isl_hash_table_foreach(ctx, &u->table, fn, user);
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
	PART *part = *entry;
	FN(PART,free)(part);
	return isl_stat_ok;
}

#include <isl_union_templ.c>
