/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#include <string.h>
#include <isl_ctx_private.h>
#include <isl_id_private.h>

#undef BASE
#define BASE id

#include <isl_list_templ.c>

/* A special, static isl_id to use as domains (and ranges)
 * of sets and parameters domains.
 * The user should never get a hold on this isl_id.
 */
isl_id isl_id_none = {
	.ref = -1,
	.ctx = NULL,
	.name = "#none",
	.user = NULL
};

isl_ctx *isl_id_get_ctx(__isl_keep isl_id *id)
{
	return id ? id->ctx : NULL;
}

void *isl_id_get_user(__isl_keep isl_id *id)
{
	return id ? id->user : NULL;
}

const char *isl_id_get_name(__isl_keep isl_id *id)
{
	return id ? id->name : NULL;
}

static __isl_give isl_id *id_alloc(isl_ctx *ctx, const char *name, void *user)
{
	const char *copy = name ? strdup(name) : NULL;
	isl_id *id;

	if (name && !copy)
		return NULL;
	id = isl_calloc_type(ctx, struct isl_id);
	if (!id)
		goto error;

	id->ctx = ctx;
	isl_ctx_ref(id->ctx);
	id->ref = 1;
	id->name = copy;
	id->user = user;

	id->hash = isl_hash_init();
	if (name)
		id->hash = isl_hash_string(id->hash, name);
	else
		id->hash = isl_hash_builtin(id->hash, user);

	return id;
error:
	free((char *)copy);
	return NULL;
}

uint32_t isl_id_get_hash(__isl_keep isl_id *id)
{
	return id ? id->hash : 0;
}

struct isl_name_and_user {
	const char *name;
	void *user;
};

static int isl_id_has_name_and_user(const void *entry, const void *val)
{
	isl_id *id = (isl_id *)entry;
	struct isl_name_and_user *nu = (struct isl_name_and_user *) val;

	if (id->user != nu->user)
		return 0;
	if (id->name == nu->name)
		return 1;
	if (!id->name || !nu->name)
		return 0;

	return !strcmp(id->name, nu->name);
}

__isl_give isl_id *isl_id_alloc(isl_ctx *ctx, const char *name, void *user)
{
	struct isl_hash_table_entry *entry;
	uint32_t id_hash;
	struct isl_name_and_user nu = { name, user };

	if (!ctx)
		return NULL;

	id_hash = isl_hash_init();
	if (name)
		id_hash = isl_hash_string(id_hash, name);
	else
		id_hash = isl_hash_builtin(id_hash, user);
	entry = isl_hash_table_find(ctx, &ctx->id_table, id_hash,
					isl_id_has_name_and_user, &nu, 1);
	if (!entry)
		return NULL;
	if (entry->data)
		return isl_id_copy(entry->data);
	entry->data = id_alloc(ctx, name, user);
	if (!entry->data)
		ctx->id_table.n--;
	return entry->data;
}

/* If the id has a negative refcount, then it is a static isl_id
 * which should not be changed.
 */
__isl_give isl_id *isl_id_copy(isl_id *id)
{
	if (!id)
		return NULL;

	if (id->ref < 0)
		return id;

	id->ref++;
	return id;
}

/* Compare two isl_ids.
 *
 * The order is fairly arbitrary.  We do keep the comparison of
 * the user pointers as a last resort since these pointer values
 * may not be stable across different systems or even different runs.
 */
int isl_id_cmp(__isl_keep isl_id *id1, __isl_keep isl_id *id2)
{
	if (id1 == id2)
		return 0;
	if (!id1)
		return -1;
	if (!id2)
		return 1;
	if (!id1->name != !id2->name)
		return !id1->name - !id2->name;
	if (id1->name) {
		int cmp = strcmp(id1->name, id2->name);
		if (cmp != 0)
			return cmp;
	}
	if (id1->user < id2->user)
		return -1;
	else
		return 1;
}

static int isl_id_eq(const void *entry, const void *name)
{
	return entry == name;
}

uint32_t isl_hash_id(uint32_t hash, __isl_keep isl_id *id)
{
	if (id)
		isl_hash_hash(hash, id->hash);

	return hash;
}

/* Replace the free_user callback by "free_user".
 */
__isl_give isl_id *isl_id_set_free_user(__isl_take isl_id *id,
	void (*free_user)(void *user))
{
	if (!id)
		return NULL;

	id->free_user = free_user;

	return id;
}

/* If the id has a negative refcount, then it is a static isl_id
 * and should not be freed.
 */
__isl_null isl_id *isl_id_free(__isl_take isl_id *id)
{
	struct isl_hash_table_entry *entry;

	if (!id)
		return NULL;

	if (id->ref < 0)
		return NULL;

	if (--id->ref > 0)
		return NULL;

	entry = isl_hash_table_find(id->ctx, &id->ctx->id_table, id->hash,
					isl_id_eq, id, 0);
	if (!entry)
		isl_die(id->ctx, isl_error_unknown,
			"unable to find id", (void)0);
	else
		isl_hash_table_remove(id->ctx, &id->ctx->id_table, entry);

	if (id->free_user)
		id->free_user(id->user);

	free((char *)id->name);
	isl_ctx_deref(id->ctx);
	free(id);

	return NULL;
}

__isl_give isl_printer *isl_printer_print_id(__isl_take isl_printer *p,
	__isl_keep isl_id *id)
{
	if (!id)
		goto error;

	if (id->name)
		p = isl_printer_print_str(p, id->name);
	if (id->user) {
		char buffer[50];
		snprintf(buffer, sizeof(buffer), "@%p", id->user);
		p = isl_printer_print_str(p, buffer);
	}
	return p;
error:
	isl_printer_free(p);
	return NULL;
}
