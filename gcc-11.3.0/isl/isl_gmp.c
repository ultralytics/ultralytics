/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#include <isl_int.h>

uint32_t isl_gmp_hash(mpz_t v, uint32_t hash)
{
	int sa = v[0]._mp_size;
	int abs_sa = sa < 0 ? -sa : sa;
	unsigned char *data = (unsigned char *)v[0]._mp_d;
	unsigned char *end = data + abs_sa * sizeof(v[0]._mp_d[0]);

	if (sa < 0)
		isl_hash_byte(hash, 0xFF);
	for (; data < end; ++data)
		isl_hash_byte(hash, *data);
	return hash;
}
