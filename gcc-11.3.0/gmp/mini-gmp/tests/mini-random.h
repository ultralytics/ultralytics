/*

Copyright 2011, Free Software Foundation, Inc.

This file is part of the GNU MP Library test suite.

The GNU MP Library test suite is free software; you can redistribute it
and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 3 of the License,
or (at your option) any later version.

The GNU MP Library test suite is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the GNU MP Library test suite.  If not, see https://www.gnu.org/licenses/.  */

#include "mini-gmp.h"
#include "hex-random.h"

void mini_urandomb (mpz_t, unsigned long);
void mini_rrandomb (mpz_t, unsigned long);
void mini_rrandomb_export (mpz_t r, void *dst, size_t *countp,
			   int order, size_t size, int endian,
			   unsigned long bits);

void mini_random_op2 (enum hex_random_op,  unsigned long, mpz_t, mpz_t);
void mini_random_op3 (enum hex_random_op,  unsigned long, mpz_t, mpz_t, mpz_t);
void mini_random_op4 (enum hex_random_op, unsigned long, mpz_t, mpz_t, mpz_t, mpz_t);
void mini_random_scan_op (enum hex_random_op, unsigned long, mpz_t, mp_bitcnt_t *, mp_bitcnt_t *);
void mini_random_bit_op (enum hex_random_op, unsigned long, mpz_t, mp_bitcnt_t *, mpz_t);
