/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#ifndef ISL_ID_PRIVATE_H
#define ISL_ID_PRIVATE_H

#include <isl/id.h>

/* Represent a name and/or user pointer.
 *
 * If "free_user" is set, then it will be called on "user" when
 * the last instance of the isl_id is freed.
 */
struct isl_id {
	int ref;
	isl_ctx *ctx;

	const char *name;
	void *user;
	uint32_t hash;

	__isl_give void (*free_user)(void *user);
};

#undef EL
#define EL isl_id

#include <isl_list_templ.h>

uint32_t isl_hash_id(uint32_t hash, __isl_keep isl_id *id);
int isl_id_cmp(__isl_keep isl_id *id1, __isl_keep isl_id *id2);

extern isl_id isl_id_none;

#endif
