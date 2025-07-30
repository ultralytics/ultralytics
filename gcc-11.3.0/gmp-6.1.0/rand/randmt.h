/* Mersenne Twister pseudo-random number generator defines.

Copyright 2002, 2003 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 2 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the GNU MP Library.  If not,
see https://www.gnu.org/licenses/.  */


/* Number of extractions used to warm the buffer up.  */
#define WARM_UP 2000

/* Period parameters.  */
#define N 624
#define M 397
#define MATRIX_A 0x9908B0DF   /* Constant vector a.  */

/* State structure for MT.  */
typedef struct
{
  gmp_uint_least32_t mt[N];    /* State array.  */
  int mti;                     /* Index of current value.  */
} gmp_rand_mt_struct;


void __gmp_mt_recalc_buffer (gmp_uint_least32_t *);
void __gmp_randget_mt (gmp_randstate_t, mp_ptr, unsigned long int);
void __gmp_randclear_mt (gmp_randstate_t);
void __gmp_randiset_mt (gmp_randstate_ptr, gmp_randstate_srcptr);
