/* mpn_add_n_sub_n -- Add and Subtract two limb vectors of equal, non-zero length.

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 1999-2001, 2006 Free Software Foundation, Inc.

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

#include "gmp.h"
#include "gmp-impl.h"

#ifndef L1_CACHE_SIZE
#define L1_CACHE_SIZE 8192	/* only 68040 has less than this */
#endif

#define PART_SIZE (L1_CACHE_SIZE / GMP_LIMB_BYTES / 6)


/* mpn_add_n_sub_n.
   r1[] = s1[] + s2[]
   r2[] = s1[] - s2[]
   All operands have n limbs.
   In-place operations allowed.  */
mp_limb_t
mpn_add_n_sub_n (mp_ptr r1p, mp_ptr r2p, mp_srcptr s1p, mp_srcptr s2p, mp_size_t n)
{
  mp_limb_t acyn, acyo;		/* carry for add */
  mp_limb_t scyn, scyo;		/* carry for subtract */
  mp_size_t off;		/* offset in operands */
  mp_size_t this_n;		/* size of current chunk */

  /* We alternatingly add and subtract in chunks that fit into the (L1)
     cache.  Since the chunks are several hundred limbs, the function call
     overhead is insignificant, but we get much better locality.  */

  /* We have three variant of the inner loop, the proper loop is chosen
     depending on whether r1 or r2 are the same operand as s1 or s2.  */

  if (r1p != s1p && r1p != s2p)
    {
      /* r1 is not identical to either input operand.  We can therefore write
	 to r1 directly, without using temporary storage.  */
      acyo = 0;
      scyo = 0;
      for (off = 0; off < n; off += PART_SIZE)
	{
	  this_n = MIN (n - off, PART_SIZE);
#if HAVE_NATIVE_mpn_add_nc
	  acyo = mpn_add_nc (r1p + off, s1p + off, s2p + off, this_n, acyo);
#else
	  acyn = mpn_add_n (r1p + off, s1p + off, s2p + off, this_n);
	  acyo = acyn + mpn_add_1 (r1p + off, r1p + off, this_n, acyo);
#endif
#if HAVE_NATIVE_mpn_sub_nc
	  scyo = mpn_sub_nc (r2p + off, s1p + off, s2p + off, this_n, scyo);
#else
	  scyn = mpn_sub_n (r2p + off, s1p + off, s2p + off, this_n);
	  scyo = scyn + mpn_sub_1 (r2p + off, r2p + off, this_n, scyo);
#endif
	}
    }
  else if (r2p != s1p && r2p != s2p)
    {
      /* r2 is not identical to either input operand.  We can therefore write
	 to r2 directly, without using temporary storage.  */
      acyo = 0;
      scyo = 0;
      for (off = 0; off < n; off += PART_SIZE)
	{
	  this_n = MIN (n - off, PART_SIZE);
#if HAVE_NATIVE_mpn_sub_nc
	  scyo = mpn_sub_nc (r2p + off, s1p + off, s2p + off, this_n, scyo);
#else
	  scyn = mpn_sub_n (r2p + off, s1p + off, s2p + off, this_n);
	  scyo = scyn + mpn_sub_1 (r2p + off, r2p + off, this_n, scyo);
#endif
#if HAVE_NATIVE_mpn_add_nc
	  acyo = mpn_add_nc (r1p + off, s1p + off, s2p + off, this_n, acyo);
#else
	  acyn = mpn_add_n (r1p + off, s1p + off, s2p + off, this_n);
	  acyo = acyn + mpn_add_1 (r1p + off, r1p + off, this_n, acyo);
#endif
	}
    }
  else
    {
      /* r1 and r2 are identical to s1 and s2 (r1==s1 and r2==s2 or vice versa)
	 Need temporary storage.  */
      mp_limb_t tp[PART_SIZE];
      acyo = 0;
      scyo = 0;
      for (off = 0; off < n; off += PART_SIZE)
	{
	  this_n = MIN (n - off, PART_SIZE);
#if HAVE_NATIVE_mpn_add_nc
	  acyo = mpn_add_nc (tp, s1p + off, s2p + off, this_n, acyo);
#else
	  acyn = mpn_add_n (tp, s1p + off, s2p + off, this_n);
	  acyo = acyn + mpn_add_1 (tp, tp, this_n, acyo);
#endif
#if HAVE_NATIVE_mpn_sub_nc
	  scyo = mpn_sub_nc (r2p + off, s1p + off, s2p + off, this_n, scyo);
#else
	  scyn = mpn_sub_n (r2p + off, s1p + off, s2p + off, this_n);
	  scyo = scyn + mpn_sub_1 (r2p + off, r2p + off, this_n, scyo);
#endif
	  MPN_COPY (r1p + off, tp, this_n);
	}
    }

  return 2 * acyo + scyo;
}

#ifdef MAIN
#include <stdlib.h>
#include <stdio.h>
#include "timing.h"

long cputime ();

int
main (int argc, char **argv)
{
  mp_ptr r1p, r2p, s1p, s2p;
  double t;
  mp_size_t n;

  n = strtol (argv[1], 0, 0);

  r1p = malloc (n * GMP_LIMB_BYTES);
  r2p = malloc (n * GMP_LIMB_BYTES);
  s1p = malloc (n * GMP_LIMB_BYTES);
  s2p = malloc (n * GMP_LIMB_BYTES);
  TIME (t,(mpn_add_n(r1p,s1p,s2p,n),mpn_sub_n(r1p,s1p,s2p,n)));
  printf ("              separate add and sub: %.3f\n", t);
  TIME (t,mpn_add_n_sub_n(r1p,r2p,s1p,s2p,n));
  printf ("combined addsub separate variables: %.3f\n", t);
  TIME (t,mpn_add_n_sub_n(r1p,r2p,r1p,s2p,n));
  printf ("        combined addsub r1 overlap: %.3f\n", t);
  TIME (t,mpn_add_n_sub_n(r1p,r2p,r1p,s2p,n));
  printf ("        combined addsub r2 overlap: %.3f\n", t);
  TIME (t,mpn_add_n_sub_n(r1p,r2p,r1p,r2p,n));
  printf ("          combined addsub in-place: %.3f\n", t);

  return 0;
}
#endif
