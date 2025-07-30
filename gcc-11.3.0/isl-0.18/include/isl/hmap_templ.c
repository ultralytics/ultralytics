/*
 * Copyright 2011      INRIA Saclay
 * Copyright 2013      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 */

#include <isl/ctx.h>
#include <isl/hash.h>

#define ISL_xCAT(A,B) A ## B
#define ISL_CAT(A,B) ISL_xCAT(A,B)
#define ISL_xFN(TYPE,NAME) TYPE ## _ ## NAME
#define ISL_FN(TYPE,NAME) ISL_xFN(TYPE,NAME)
#define ISL_xS(TYPE1,TYPE2,NAME) struct isl_ ## TYPE1 ## _ ## TYPE2 ## _ ## NAME
#define ISL_yS(TYPE1,TYPE2,NAME) ISL_xS(TYPE1,TYPE2,NAME)
#define ISL_S(NAME) ISL_yS(ISL_KEY,ISL_VAL,NAME)

struct ISL_HMAP {
	int ref;
	isl_ctx *ctx;
	struct isl_hash_table table;
};

ISL_S(pair) {
	ISL_KEY *key;
	ISL_VAL *val;
};

__isl_give ISL_HMAP *ISL_FN(ISL_HMAP,alloc)(isl_ctx *ctx, int min_size)
{
	ISL_HMAP *hmap;

	hmap = isl_calloc_type(ctx, ISL_HMAP);
	if (!hmap)
		return NULL;

	hmap->ctx = ctx;
	isl_ctx_ref(ctx);
	hmap->ref = 1;

	if (isl_hash_table_init(ctx, &hmap->table, min_size) < 0)
		return ISL_FN(ISL_HMAP,free)(hmap);

	return hmap;
}

static isl_stat free_pair(void **entry, void *user)
{
	ISL_S(pair) *pair = *entry;
	ISL_FN(ISL_KEY,free)(pair->key);
	ISL_FN(ISL_VAL,free)(pair->val);
	free(pair);
	*entry = NULL;
	return isl_stat_ok;
}

__isl_null ISL_HMAP *ISL_FN(ISL_HMAP,free)(__isl_take ISL_HMAP *hmap)
{
	if (!hmap)
		return NULL;
	if (--hmap->ref > 0)
		return NULL;
	isl_hash_table_foreach(hmap->ctx, &hmap->table, &free_pair, NULL);
	isl_hash_table_clear(&hmap->table);
	isl_ctx_deref(hmap->ctx);
	free(hmap);
	return NULL;
}

isl_ctx *ISL_FN(ISL_HMAP,get_ctx)(__isl_keep ISL_HMAP *hmap)
{
	return hmap ? hmap->ctx : NULL;
}

/* Add a mapping from "key" to "val" to the associative array
 * pointed to by user.
 */
static isl_stat add_key_val(__isl_take ISL_KEY *key, __isl_take ISL_VAL *val,
	void *user)
{
	ISL_HMAP **hmap = (ISL_HMAP **) user;

	*hmap = ISL_FN(ISL_HMAP,set)(*hmap, key, val);

	if (!*hmap)
		return isl_stat_error;

	return isl_stat_ok;
}

__isl_give ISL_HMAP *ISL_FN(ISL_HMAP,dup)(__isl_keep ISL_HMAP *hmap)
{
	ISL_HMAP *dup;

	if (!hmap)
		return NULL;

	dup = ISL_FN(ISL_HMAP,alloc)(hmap->ctx, hmap->table.n);
	if (ISL_FN(ISL_HMAP,foreach)(hmap, &add_key_val, &dup) < 0)
		return ISL_FN(ISL_HMAP,free)(dup);

	return dup;
}

__isl_give ISL_HMAP *ISL_FN(ISL_HMAP,cow)(__isl_take ISL_HMAP *hmap)
{
	if (!hmap)
		return NULL;

	if (hmap->ref == 1)
		return hmap;
	hmap->ref--;
	return ISL_FN(ISL_HMAP,dup)(hmap);
}

__isl_give ISL_HMAP *ISL_FN(ISL_HMAP,copy)(__isl_keep ISL_HMAP *hmap)
{
	if (!hmap)
		return NULL;

	hmap->ref++;
	return hmap;
}

static int has_key(const void *entry, const void *c_key)
{
	const ISL_S(pair) *pair = entry;
	ISL_KEY *key = (ISL_KEY *) c_key;

	return ISL_KEY_IS_EQUAL(pair->key, key);
}

/* If "hmap" contains a value associated to "key", then return
 * (isl_bool_true, copy of value).
 * Otherwise, return
 * (isl_bool_false, NULL).
 * If an error occurs, then return
 * (isl_bool_error, NULL).
 */
__isl_give ISL_MAYBE(ISL_VAL) ISL_FN(ISL_HMAP,try_get)(
	__isl_keep ISL_HMAP *hmap, __isl_keep ISL_KEY *key)
{
	struct isl_hash_table_entry *entry;
	ISL_S(pair) *pair;
	uint32_t hash;
	ISL_MAYBE(ISL_VAL) res = { isl_bool_false, NULL };

	if (!hmap || !key)
		goto error;

	hash = ISL_FN(ISL_KEY,get_hash)(key);
	entry = isl_hash_table_find(hmap->ctx, &hmap->table, hash,
					&has_key, key, 0);

	if (!entry)
		return res;

	pair = entry->data;

	res.valid = isl_bool_true;
	res.value = ISL_FN(ISL_VAL,copy)(pair->val);
	if (!res.value)
		res.valid = isl_bool_error;
	return res;
error:
	res.valid = isl_bool_error;
	res.value = NULL;
	return res;
}

/* If "hmap" contains a value associated to "key", then return
 * isl_bool_true.  Otherwise, return isl_bool_false.
 * Return isl_bool_error on error.
 */
isl_bool ISL_FN(ISL_HMAP,has)(__isl_keep ISL_HMAP *hmap,
	__isl_keep ISL_KEY *key)
{
	ISL_MAYBE(ISL_VAL) res;

	res = ISL_FN(ISL_HMAP,try_get)(hmap, key);
	ISL_FN(ISL_VAL,free)(res.value);

	return res.valid;
}

/* If "hmap" contains a value associated to "key", then return
 * a copy of that value.  Otherwise, return NULL.
 * Return NULL on error.
 */
__isl_give ISL_VAL *ISL_FN(ISL_HMAP,get)(__isl_keep ISL_HMAP *hmap,
	__isl_take ISL_KEY *key)
{
	ISL_VAL *res;

	res = ISL_FN(ISL_HMAP,try_get)(hmap, key).value;
	ISL_FN(ISL_KEY,free)(key);
	return res;
}

/* Remove the mapping between "key" and its associated value (if any)
 * from "hmap".
 *
 * If "key" is not mapped to anything, then we leave "hmap" untouched"
 */
__isl_give ISL_HMAP *ISL_FN(ISL_HMAP,drop)(__isl_take ISL_HMAP *hmap,
	__isl_take ISL_KEY *key)
{
	struct isl_hash_table_entry *entry;
	ISL_S(pair) *pair;
	uint32_t hash;

	if (!hmap || !key)
		goto error;

	hash = ISL_FN(ISL_KEY,get_hash)(key);
	entry = isl_hash_table_find(hmap->ctx, &hmap->table, hash,
					&has_key, key, 0);
	if (!entry) {
		ISL_FN(ISL_KEY,free)(key);
		return hmap;
	}

	hmap = ISL_FN(ISL_HMAP,cow)(hmap);
	if (!hmap)
		goto error;
	entry = isl_hash_table_find(hmap->ctx, &hmap->table, hash,
					&has_key, key, 0);
	ISL_FN(ISL_KEY,free)(key);

	if (!entry)
		isl_die(hmap->ctx, isl_error_internal,
			"missing entry" , goto error);

	pair = entry->data;
	isl_hash_table_remove(hmap->ctx, &hmap->table, entry);
	ISL_FN(ISL_KEY,free)(pair->key);
	ISL_FN(ISL_VAL,free)(pair->val);
	free(pair);

	return hmap;
error:
	ISL_FN(ISL_KEY,free)(key);
	ISL_FN(ISL_HMAP,free)(hmap);
	return NULL;
}

/* Add a mapping from "key" to "val" to "hmap".
 * If "key" was already mapped to something else, then that mapping
 * is replaced.
 * If key happened to be mapped to "val" already, then we leave
 * "hmap" untouched.
 */
__isl_give ISL_HMAP *ISL_FN(ISL_HMAP,set)(__isl_take ISL_HMAP *hmap,
	__isl_take ISL_KEY *key, __isl_take ISL_VAL *val)
{
	struct isl_hash_table_entry *entry;
	ISL_S(pair) *pair;
	uint32_t hash;

	if (!hmap || !key || !val)
		goto error;

	hash = ISL_FN(ISL_KEY,get_hash)(key);
	entry = isl_hash_table_find(hmap->ctx, &hmap->table, hash,
					&has_key, key, 0);
	if (entry) {
		int equal;
		pair = entry->data;
		equal = ISL_VAL_IS_EQUAL(pair->val, val);
		if (equal < 0)
			goto error;
		if (equal) {
			ISL_FN(ISL_KEY,free)(key);
			ISL_FN(ISL_VAL,free)(val);
			return hmap;
		}
	}

	hmap = ISL_FN(ISL_HMAP,cow)(hmap);
	if (!hmap)
		goto error;

	entry = isl_hash_table_find(hmap->ctx, &hmap->table, hash,
					&has_key, key, 1);

	if (!entry)
		goto error;

	if (entry->data) {
		pair = entry->data;
		ISL_FN(ISL_VAL,free)(pair->val);
		pair->val = val;
		ISL_FN(ISL_KEY,free)(key);
		return hmap;
	}

	pair = isl_alloc_type(hmap->ctx, ISL_S(pair));
	if (!pair)
		goto error;

	entry->data = pair;
	pair->key = key;
	pair->val = val;
	return hmap;
error:
	ISL_FN(ISL_KEY,free)(key);
	ISL_FN(ISL_VAL,free)(val);
	return ISL_FN(ISL_HMAP,free)(hmap);
}

/* Internal data structure for isl_map_to_basic_set_foreach.
 *
 * fn is the function that should be called on each entry.
 * user is the user-specified final argument to fn.
 */
ISL_S(foreach_data) {
	isl_stat (*fn)(__isl_take ISL_KEY *key, __isl_take ISL_VAL *val,
		void *user);
	void *user;
};

/* Call data->fn on a copy of the key and value in *entry.
 */
static isl_stat call_on_copy(void **entry, void *user)
{
	ISL_S(pair) *pair = *entry;
	ISL_S(foreach_data) *data = (ISL_S(foreach_data) *) user;

	return data->fn(ISL_FN(ISL_KEY,copy)(pair->key),
			ISL_FN(ISL_VAL,copy)(pair->val), data->user);
}

/* Call "fn" on each pair of key and value in "hmap".
 */
isl_stat ISL_FN(ISL_HMAP,foreach)(__isl_keep ISL_HMAP *hmap,
	isl_stat (*fn)(__isl_take ISL_KEY *key, __isl_take ISL_VAL *val,
		void *user),
	void *user)
{
	ISL_S(foreach_data) data = { fn, user };

	if (!hmap)
		return isl_stat_error;

	return isl_hash_table_foreach(hmap->ctx, &hmap->table,
				      &call_on_copy, &data);
}

/* Internal data structure for print_pair.
 *
 * p is the printer on which the associative array is being printed.
 * first is set if the current key-value pair is the first to be printed.
 */
ISL_S(print_data) {
	isl_printer *p;
	int first;
};

/* Print the given key-value pair to data->p.
 */
static isl_stat print_pair(__isl_take ISL_KEY *key, __isl_take ISL_VAL *val,
	void *user)
{
	ISL_S(print_data) *data = user;

	if (!data->first)
		data->p = isl_printer_print_str(data->p, ", ");
	data->p = ISL_KEY_PRINT(data->p, key);
	data->p = isl_printer_print_str(data->p, ": ");
	data->p = ISL_VAL_PRINT(data->p, val);
	data->first = 0;

	ISL_FN(ISL_KEY,free)(key);
	ISL_FN(ISL_VAL,free)(val);
	return isl_stat_ok;
}

/* Print the associative array to "p".
 */
__isl_give isl_printer *ISL_FN(isl_printer_print,ISL_HMAP_SUFFIX)(
	__isl_take isl_printer *p, __isl_keep ISL_HMAP *hmap)
{
	ISL_S(print_data) data;

	if (!p || !hmap)
		return isl_printer_free(p);

	p = isl_printer_print_str(p, "{");
	data.p = p;
	data.first = 1;
	if (ISL_FN(ISL_HMAP,foreach)(hmap, &print_pair, &data) < 0)
		data.p = isl_printer_free(data.p);
	p = data.p;
	p = isl_printer_print_str(p, "}");

	return p;
}

void ISL_FN(ISL_HMAP,dump)(__isl_keep ISL_HMAP *hmap)
{
	isl_printer *printer;

	if (!hmap)
		return;

	printer = isl_printer_to_file(ISL_FN(ISL_HMAP,get_ctx)(hmap), stderr);
	printer = ISL_FN(isl_printer_print,ISL_HMAP_SUFFIX)(printer, hmap);
	printer = isl_printer_end_line(printer);

	isl_printer_free(printer);
}
