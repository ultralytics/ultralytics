/*
 * Copyright 2016      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege
 */

#include <isl_multi_macro.h>
#include <isl/hash.h>

/* Return a hash value that digests "multi".
 */
uint32_t FN(MULTI(BASE),get_hash)(__isl_keep MULTI(BASE) *multi)
{
	int i;
	uint32_t hash;

	if (!multi)
		return 0;

	hash = isl_hash_init();
	for (i = 0; i < multi->n; ++i) {
		uint32_t el_hash;
		el_hash = FN(EL,get_hash)(multi->p[i]);
		isl_hash_hash(hash, el_hash);
	}

	return hash;
}
