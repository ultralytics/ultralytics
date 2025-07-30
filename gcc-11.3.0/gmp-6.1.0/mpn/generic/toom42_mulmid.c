/* mpn_toom42_mulmid -- toom42 middle product

   Contributed by David Harvey.

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2011 Free Software Foundation, Inc.

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



/*
  Middle product of {ap,2n-1} and {bp,n}, output written to {rp,n+2}.

  Neither ap nor bp may overlap rp.

  Must have n >= 4.

  Amount of scratch space required is given by mpn_toom42_mulmid_itch().

  FIXME: this code assumes that n is small compared to GMP_NUMB_MAX. The exact
  requirements should be clarified.
*/
void
mpn_toom42_mulmid (mp_ptr rp, mp_srcptr ap, mp_srcptr bp, mp_size_t n,
                   mp_ptr scratch)
{
  mp_limb_t cy, e[12], zh, zl;
  mp_size_t m;
  int neg;

  ASSERT (n >= 4);
  ASSERT (! MPN_OVERLAP_P (rp, n + 2, ap, 2*n - 1));
  ASSERT (! MPN_OVERLAP_P (rp, n + 2, bp, n));

  ap += n & 1;   /* handle odd row and diagonal later */
  m = n / 2;

  /* (e0h:e0l) etc are correction terms, in 2's complement */
#define e0l (e[0])
#define e0h (e[1])
#define e1l (e[2])
#define e1h (e[3])
#define e2l (e[4])
#define e2h (e[5])
#define e3l (e[6])
#define e3h (e[7])
#define e4l (e[8])
#define e4h (e[9])
#define e5l (e[10])
#define e5h (e[11])

#define s (scratch + 2)
#define t (rp + m + 2)
#define p0 rp
#define p1 scratch
#define p2 (rp + m)
#define next_scratch (scratch + 3*m + 1)

  /*
            rp                            scratch
  |---------|-----------|    |---------|---------|----------|
  0         m         2m+2   0         m         2m        3m+1
            <----p2---->       <-------------s------------->
  <----p0----><---t---->     <----p1---->
  */

  /* compute {s,3m-1} = {a,3m-1} + {a+m,3m-1} and error terms e0, e1, e2, e3 */
  cy = mpn_add_err1_n (s, ap, ap + m, &e0l, bp + m, m - 1, 0);
  cy = mpn_add_err2_n (s + m - 1, ap + m - 1, ap + 2*m - 1, &e1l,
		       bp + m, bp, m, cy);
  mpn_add_err1_n (s + 2*m - 1, ap + 2*m - 1, ap + 3*m - 1, &e3l, bp, m, cy);

  /* compute t = (-1)^neg * ({b,m} - {b+m,m}) and error terms e4, e5 */
  if (mpn_cmp (bp + m, bp, m) < 0)
    {
      ASSERT_NOCARRY (mpn_sub_err2_n (t, bp, bp + m, &e4l,
				      ap + m - 1, ap + 2*m - 1, m, 0));
      neg = 1;
    }
  else
    {
      ASSERT_NOCARRY (mpn_sub_err2_n (t, bp + m, bp, &e4l,
				      ap + m - 1, ap + 2*m - 1, m, 0));
      neg = 0;
    }

  /* recursive middle products. The picture is:

      b[2m-1]   A   A   A   B   B   B   -   -   -   -   -
      ...       -   A   A   A   B   B   B   -   -   -   -
      b[m]      -   -   A   A   A   B   B   B   -   -   -
      b[m-1]    -   -   -   C   C   C   D   D   D   -   -
      ...       -   -   -   -   C   C   C   D   D   D   -
      b[0]      -   -   -   -   -   C   C   C   D   D   D
               a[0]   ...  a[m]  ...  a[2m]    ...    a[4m-2]
  */

  if (m < MULMID_TOOM42_THRESHOLD)
    {
      /* A + B */
      mpn_mulmid_basecase (p0, s, 2*m - 1, bp + m, m);
      /* accumulate high limbs of p0 into e1 */
      ADDC_LIMB (cy, e1l, e1l, p0[m]);
      e1h += p0[m + 1] + cy;
      /* (-1)^neg * (B - C)   (overwrites first m limbs of s) */
      mpn_mulmid_basecase (p1, ap + m, 2*m - 1, t, m);
      /* C + D   (overwrites t) */
      mpn_mulmid_basecase (p2, s + m, 2*m - 1, bp, m);
    }
  else
    {
      /* as above, but use toom42 instead */
      mpn_toom42_mulmid (p0, s, bp + m, m, next_scratch);
      ADDC_LIMB (cy, e1l, e1l, p0[m]);
      e1h += p0[m + 1] + cy;
      mpn_toom42_mulmid (p1, ap + m, t, m, next_scratch);
      mpn_toom42_mulmid (p2, s + m, bp, m, next_scratch);
    }

  /* apply error terms */

  /* -e0 at rp[0] */
  SUBC_LIMB (cy, rp[0], rp[0], e0l);
  SUBC_LIMB (cy, rp[1], rp[1], e0h + cy);
  if (UNLIKELY (cy))
    {
      cy = (m > 2) ? mpn_sub_1 (rp + 2, rp + 2, m - 2, 1) : 1;
      SUBC_LIMB (cy, e1l, e1l, cy);
      e1h -= cy;
    }

  /* z = e1 - e2 + high(p0) */
  SUBC_LIMB (cy, zl, e1l, e2l);
  zh = e1h - e2h - cy;

  /* z at rp[m] */
  ADDC_LIMB (cy, rp[m], rp[m], zl);
  zh = (zh + cy) & GMP_NUMB_MASK;
  ADDC_LIMB (cy, rp[m + 1], rp[m + 1], zh);
  cy -= (zh >> (GMP_NUMB_BITS - 1));
  if (UNLIKELY (cy))
    {
      if (cy == 1)
	mpn_add_1 (rp + m + 2, rp + m + 2, m, 1);
      else /* cy == -1 */
	mpn_sub_1 (rp + m + 2, rp + m + 2, m, 1);
    }

  /* e3 at rp[2*m] */
  ADDC_LIMB (cy, rp[2*m], rp[2*m], e3l);
  rp[2*m + 1] = (rp[2*m + 1] + e3h + cy) & GMP_NUMB_MASK;

  /* e4 at p1[0] */
  ADDC_LIMB (cy, p1[0], p1[0], e4l);
  ADDC_LIMB (cy, p1[1], p1[1], e4h + cy);
  if (UNLIKELY (cy))
    mpn_add_1 (p1 + 2, p1 + 2, m, 1);

  /* -e5 at p1[m] */
  SUBC_LIMB (cy, p1[m], p1[m], e5l);
  p1[m + 1] = (p1[m + 1] - e5h - cy) & GMP_NUMB_MASK;

  /* adjustment if p1 ends up negative */
  cy = (p1[m + 1] >> (GMP_NUMB_BITS - 1));

  /* add (-1)^neg * (p1 - B^m * p1) to output */
  if (neg)
    {
      mpn_sub_1 (rp + m + 2, rp + m + 2, m, cy);
      mpn_add (rp, rp, 2*m + 2, p1, m + 2);             /* A + C */
      mpn_sub_n (rp + m, rp + m, p1, m + 2);            /* B + D */
    }
  else
    {
      mpn_add_1 (rp + m + 2, rp + m + 2, m, cy);
      mpn_sub (rp, rp, 2*m + 2, p1, m + 2);             /* A + C */
      mpn_add_n (rp + m, rp + m, p1, m + 2);            /* B + D */
    }

  /* odd row and diagonal */
  if (n & 1)
    {
      /*
        Products marked E are already done. We need to do products marked O.

        OOOOO----
        -EEEEO---
        --EEEEO--
        ---EEEEO-
        ----EEEEO
       */

      /* first row of O's */
      cy = mpn_addmul_1 (rp, ap - 1, n, bp[n - 1]);
      ADDC_LIMB (rp[n + 1], rp[n], rp[n], cy);

      /* O's on diagonal */
      /* FIXME: should probably define an interface "mpn_mulmid_diag_1"
         that can handle the sum below. Currently we're relying on
         mulmid_basecase being pretty fast for a diagonal sum like this,
	 which is true at least for the K8 asm version, but surely false
	 for the generic version. */
      mpn_mulmid_basecase (e, ap + n - 1, n - 1, bp, n - 1);
      mpn_add_n (rp + n - 1, rp + n - 1, e, 3);
    }
}
