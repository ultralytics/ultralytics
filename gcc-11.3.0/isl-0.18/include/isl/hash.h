/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#ifndef ISL_HASH_H
#define ISL_HASH_H

#include <stdlib.h>
#include <isl/stdint.h>
#include <isl/ctx.h>

#if defined(__cplusplus)
extern "C" {
#endif

#define isl_hash_init()		(2166136261u)
#define isl_hash_byte(h,b)	do {					\
					h *= 16777619;			\
					h ^= b;				\
				} while(0)
#define isl_hash_hash(h,h2)						\
	do {								\
		isl_hash_byte(h, (h2) & 0xFF);				\
		isl_hash_byte(h, ((h2) >> 8) & 0xFF);			\
		isl_hash_byte(h, ((h2) >> 16) & 0xFF);			\
		isl_hash_byte(h, ((h2) >> 24) & 0xFF);			\
	} while(0)
#define isl_hash_bits(h,bits)						\
	((bits) == 32) ? (h) :						\
	((bits) >= 16) ?						\
	      ((h) >> (bits)) ^ ((h) & (((uint32_t)1 << (bits)) - 1)) :	\
	      (((h) >> (bits)) ^ (h)) & (((uint32_t)1 << (bits)) - 1)

uint32_t isl_hash_string(uint32_t hash, const char *s);
uint32_t isl_hash_mem(uint32_t hash, const void *p, size_t len);

#define isl_hash_builtin(h,l)	isl_hash_mem(h, &l, sizeof(l))

struct isl_hash_table_entry
{
	uint32_t  hash;
	void     *data;
};

struct isl_hash_table {
	int    bits;
	int    n;
	struct isl_hash_table_entry *entries;
};

struct isl_hash_table *isl_hash_table_alloc(struct isl_ctx *ctx, int min_size);
void isl_hash_table_free(struct isl_ctx *ctx, struct isl_hash_table *table);

int isl_hash_table_init(struct isl_ctx *ctx, struct isl_hash_table *table,
			int min_size);
void isl_hash_table_clear(struct isl_hash_table *table);
struct isl_hash_table_entry *isl_hash_table_find(struct isl_ctx *ctx,
				struct isl_hash_table *table,
				uint32_t key_hash,
				int (*eq)(const void *entry, const void *val),
				const void *val, int reserve);
isl_stat isl_hash_table_foreach(isl_ctx *ctx, struct isl_hash_table *table,
	isl_stat (*fn)(void **entry, void *user), void *user);
void isl_hash_table_remove(struct isl_ctx *ctx,
				struct isl_hash_table *table,
				struct isl_hash_table_entry *entry);

#if defined(__cplusplus)
}
#endif

#endif
