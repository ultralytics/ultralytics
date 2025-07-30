/*
 * Copyright 2016      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege
 */

#include <isl_pw_macro.h>
#include <isl/hash.h>

/* Return a hash value that digests "pw".
 */
uint32_t FN(PW,get_hash)(__isl_keep PW *pw)
{
	int i;
	uint32_t hash;

	if (!pw)
		return 0;

	hash = isl_hash_init();
	for (i = 0; i < pw->n; ++i) {
		uint32_t set_hash, el_hash;

		set_hash = isl_set_get_hash(pw->p[i].set);
		isl_hash_hash(hash, set_hash);
		el_hash = FN(EL,get_hash)(pw->p[i].FIELD);
		isl_hash_hash(hash, el_hash);
	}

	return hash;
}
