/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#include <stdlib.h>
#include <isl_hash_private.h>
#include <isl/ctx.h>
#include "isl_config.h"

uint32_t isl_hash_string(uint32_t hash, const char *s)
{
	for (; *s; s++)
		isl_hash_byte(hash, *s);
	return hash;
}

uint32_t isl_hash_mem(uint32_t hash, const void *p, size_t len)
{
	int i;
	const char *s = p;
	for (i = 0; i < len; ++i)
		isl_hash_byte(hash, s[i]);
	return hash;
}

static unsigned int round_up(unsigned int v)
{
	int old_v = v;

	while (v) {
		old_v = v;
		v ^= v & -v;
	}
	return old_v << 1;
}

int isl_hash_table_init(struct isl_ctx *ctx, struct isl_hash_table *table,
			int min_size)
{
	size_t size;

	if (!table)
		return -1;

	if (min_size < 2)
		min_size = 2;
	table->bits = ffs(round_up(4 * (min_size + 1) / 3 - 1)) - 1;
	table->n = 0;

	size = 1 << table->bits;
	table->entries = isl_calloc_array(ctx, struct isl_hash_table_entry,
					  size);
	if (!table->entries)
		return -1;

	return 0;
}

/* Dummy comparison function that always returns false.
 */
static int no(const void *entry, const void *val)
{
	return 0;
}

/* Extend "table" to twice its size.
 * Return 0 on success and -1 on error.
 *
 * We reuse isl_hash_table_find to create entries in the extended table.
 * Since all entries in the original table are assumed to be different,
 * there is no need to compare them against each other.
 */
static int grow_table(struct isl_ctx *ctx, struct isl_hash_table *table)
{
	int n;
	size_t old_size, size;
	struct isl_hash_table_entry *entries;
	uint32_t h;

	entries = table->entries;
	old_size = 1 << table->bits;
	size = 2 * old_size;
	table->entries = isl_calloc_array(ctx, struct isl_hash_table_entry,
					  size);
	if (!table->entries) {
		table->entries = entries;
		return -1;
	}

	n = table->n;
	table->n = 0;
	table->bits++;

	for (h = 0; h < old_size; ++h) {
		struct isl_hash_table_entry *entry;

		if (!entries[h].data)
			continue;

		entry = isl_hash_table_find(ctx, table, entries[h].hash,
					    &no, NULL, 1);
		if (!entry) {
			table->bits--;
			free(table->entries);
			table->entries = entries;
			table->n = n;
			return -1;
		}

		*entry = entries[h];
	}

	free(entries);

	return 0;
}

struct isl_hash_table *isl_hash_table_alloc(struct isl_ctx *ctx, int min_size)
{
	struct isl_hash_table *table = NULL;

	table = isl_alloc_type(ctx, struct isl_hash_table);
	if (isl_hash_table_init(ctx, table, min_size))
		goto error;
	return table;
error:
	isl_hash_table_free(ctx, table);
	return NULL;
}

void isl_hash_table_clear(struct isl_hash_table *table)
{
	if (!table)
		return;
	free(table->entries);
}

void isl_hash_table_free(struct isl_ctx *ctx, struct isl_hash_table *table)
{
	if (!table)
		return;
	isl_hash_table_clear(table);
	free(table);
}

/* A dummy entry that can be used to make a distinction between
 * a missing entry and an error condition.
 * It is used by isl_union_*_find_part_entry.
 */
static struct isl_hash_table_entry none = { 0, NULL };
struct isl_hash_table_entry *isl_hash_table_entry_none = &none;

struct isl_hash_table_entry *isl_hash_table_find(struct isl_ctx *ctx,
				struct isl_hash_table *table,
				uint32_t key_hash,
				int (*eq)(const void *entry, const void *val),
				const void *val, int reserve)
{
	size_t size;
	uint32_t h, key_bits;

	key_bits = isl_hash_bits(key_hash, table->bits);
	size = 1 << table->bits;
	for (h = key_bits; table->entries[h].data; h = (h+1) % size)
		if (table->entries[h].hash == key_hash &&
		    eq(table->entries[h].data, val))
			return &table->entries[h];

	if (!reserve)
		return NULL;

	if (4 * table->n >= 3 * size) {
		if (grow_table(ctx, table) < 0)
			return NULL;
		return isl_hash_table_find(ctx, table, key_hash, eq, val, 1);
	}

	table->n++;
	table->entries[h].hash = key_hash;

	return &table->entries[h];
}

isl_stat isl_hash_table_foreach(isl_ctx *ctx, struct isl_hash_table *table,
	isl_stat (*fn)(void **entry, void *user), void *user)
{
	size_t size;
	uint32_t h;

	if (!table->entries)
		return isl_stat_error;

	size = 1 << table->bits;
	for (h = 0; h < size; ++ h)
		if (table->entries[h].data &&
		    fn(&table->entries[h].data, user) < 0)
			return isl_stat_error;
	
	return isl_stat_ok;
}

void isl_hash_table_remove(struct isl_ctx *ctx,
				struct isl_hash_table *table,
				struct isl_hash_table_entry *entry)
{
	int h, h2;
	size_t size;

	if (!table || !entry)
		return;

	size = 1 << table->bits;
	h = entry - table->entries;
	isl_assert(ctx, h >= 0 && h < size, return);

	for (h2 = h+1; table->entries[h2 % size].data; h2++) {
		uint32_t bits = isl_hash_bits(table->entries[h2 % size].hash,
						table->bits);
		uint32_t offset = (size + bits - (h+1)) % size;
		if (offset <= h2 - (h+1))
			continue;
		*entry = table->entries[h2 % size];
		h = h2;
		entry = &table->entries[h % size];
	}

	entry->hash = 0;
	entry->data = NULL;
	table->n--;
}
