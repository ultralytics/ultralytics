/* Reference floating point routines.

Copyright 1996, 2001, 2004, 2005 Free Software Foundation, Inc.

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

#include <stdio.h>
#include <stdlib.h>

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"


void
refmpf_add (mpf_ptr w, mpf_srcptr u, mpf_srcptr v)
{
  mp_size_t hi, lo, size;
  mp_ptr ut, vt, wt;
  int neg;
  mp_exp_t exp;
  mp_limb_t cy;
  TMP_DECL;

  TMP_MARK;

  if (SIZ (u) == 0)
    {
      size = ABSIZ (v);
      wt = TMP_ALLOC_LIMBS (size + 1);
      MPN_COPY (wt, PTR (v), size);
      exp = EXP (v);
      neg = SIZ (v) < 0;
      goto done;
    }
  if (SIZ (v) == 0)
    {
      size = ABSIZ (u);
      wt = TMP_ALLOC_LIMBS (size + 1);
      MPN_COPY (wt, PTR (u), size);
      exp = EXP (u);
      neg = SIZ (u) < 0;
      goto done;
    }
  if ((SIZ (u) ^ SIZ (v)) < 0)
    {
      mpf_t tmp;
      SIZ (tmp) = -SIZ (v);
      EXP (tmp) = EXP (v);
      PTR (tmp) = PTR (v);
      refmpf_sub (w, u, tmp);
      return;
    }
  neg = SIZ (u) < 0;

  /* Compute the significance of the hi and lo end of the result.  */
  hi = MAX (EXP (u), EXP (v));
  lo = MIN (EXP (u) - ABSIZ (u), EXP (v) - ABSIZ (v));
  size = hi - lo;
  ut = TMP_ALLOC_LIMBS (size + 1);
  vt = TMP_ALLOC_LIMBS (size + 1);
  wt = TMP_ALLOC_LIMBS (size + 1);
  MPN_ZERO (ut, size);
  MPN_ZERO (vt, size);
  {int off;
  off = size + (EXP (u) - hi) - ABSIZ (u);
  MPN_COPY (ut + off, PTR (u), ABSIZ (u));
  off = size + (EXP (v) - hi) - ABSIZ (v);
  MPN_COPY (vt + off, PTR (v), ABSIZ (v));
  }

  cy = mpn_add_n (wt, ut, vt, size);
  wt[size] = cy;
  size += cy;
  exp = hi + cy;

done:
  if (size > PREC (w))
    {
      wt += size - PREC (w);
      size = PREC (w);
    }
  MPN_COPY (PTR (w), wt, size);
  SIZ (w) = neg == 0 ? size : -size;
  EXP (w) = exp;
  TMP_FREE;
}


/* Add 1 "unit in last place" (ie. in the least significant limb) to f.
   f cannot be zero, since that has no well-defined "last place".

   This routine is designed for use in cases where we pay close attention to
   the size of the data value and are using that (and the exponent) to
   indicate the accurate part of a result, or similar.  For this reason, if
   there's a carry out we don't store 1 and adjust the exponent, we just
   leave 100..00.  We don't even adjust if there's a carry out of prec+1
   limbs, but instead give up in that case (which we intend shouldn't arise
   in normal circumstances).  */

void
refmpf_add_ulp (mpf_ptr f)
{
  mp_ptr     fp = PTR(f);
  mp_size_t  fsize = SIZ(f);
  mp_size_t  abs_fsize = ABSIZ(f);
  mp_limb_t  c;

  if (fsize == 0)
    {
      printf ("Oops, refmpf_add_ulp called with f==0\n");
      abort ();
    }

  c = refmpn_add_1 (fp, fp, abs_fsize, CNST_LIMB(1));
  if (c != 0)
    {
      if (abs_fsize >= PREC(f) + 1)
        {
          printf ("Oops, refmpf_add_ulp carried out of prec+1 limbs\n");
          abort ();
        }

      fp[abs_fsize] = c;
      abs_fsize++;
      SIZ(f) = (fsize > 0 ? abs_fsize : - abs_fsize);
      EXP(f)++;
    }
}

/* Fill f with size limbs of the given value, setup as an integer. */
void
refmpf_fill (mpf_ptr f, mp_size_t size, mp_limb_t value)
{
  ASSERT (size >= 0);
  size = MIN (PREC(f) + 1, size);
  SIZ(f) = size;
  EXP(f) = size;
  refmpn_fill (PTR(f), size, value);
}

/* Strip high zero limbs from the f data, adjusting exponent accordingly. */
void
refmpf_normalize (mpf_ptr f)
{
  while (SIZ(f) != 0 && PTR(f)[ABSIZ(f)-1] == 0)
    {
      SIZ(f) = (SIZ(f) >= 0 ? SIZ(f)-1 : SIZ(f)+1);
      EXP(f) --;
    }
  if (SIZ(f) == 0)
    EXP(f) = 0;
}

/* refmpf_set_overlap sets up dst as a copy of src, but with PREC(dst)
   unchanged, in preparation for an overlap test.

   The full value of src is copied, and the space at PTR(dst) is extended as
   necessary.  The way PREC(dst) is unchanged is as per an mpf_set_prec_raw.
   The return value is the new PTR(dst) space precision, in bits, ready for
   a restoring mpf_set_prec_raw before mpf_clear.  */

unsigned long
refmpf_set_overlap (mpf_ptr dst, mpf_srcptr src)
{
  mp_size_t  dprec = PREC(dst);
  mp_size_t  ssize = ABSIZ(src);
  unsigned long  ret;

  refmpf_set_prec_limbs (dst, (unsigned long) MAX (dprec, ssize));
  mpf_set (dst, src);

  ret = mpf_get_prec (dst);
  PREC(dst) = dprec;
  return ret;
}

/* Like mpf_set_prec, but taking a precision in limbs.
   PREC(f) ends up as the given "prec" value.  */
void
refmpf_set_prec_limbs (mpf_ptr f, unsigned long prec)
{
  mpf_set_prec (f, __GMPF_PREC_TO_BITS (prec));
}


void
refmpf_sub (mpf_ptr w, mpf_srcptr u, mpf_srcptr v)
{
  mp_size_t hi, lo, size;
  mp_ptr ut, vt, wt;
  int neg;
  mp_exp_t exp;
  TMP_DECL;

  TMP_MARK;

  if (SIZ (u) == 0)
    {
      size = ABSIZ (v);
      wt = TMP_ALLOC_LIMBS (size + 1);
      MPN_COPY (wt, PTR (v), size);
      exp = EXP (v);
      neg = SIZ (v) > 0;
      goto done;
    }
  if (SIZ (v) == 0)
    {
      size = ABSIZ (u);
      wt = TMP_ALLOC_LIMBS (size + 1);
      MPN_COPY (wt, PTR (u), size);
      exp = EXP (u);
      neg = SIZ (u) < 0;
      goto done;
    }
  if ((SIZ (u) ^ SIZ (v)) < 0)
    {
      mpf_t tmp;
      SIZ (tmp) = -SIZ (v);
      EXP (tmp) = EXP (v);
      PTR (tmp) = PTR (v);
      refmpf_add (w, u, tmp);
      if (SIZ (u) < 0)
	mpf_neg (w, w);
      return;
    }
  neg = SIZ (u) < 0;

  /* Compute the significance of the hi and lo end of the result.  */
  hi = MAX (EXP (u), EXP (v));
  lo = MIN (EXP (u) - ABSIZ (u), EXP (v) - ABSIZ (v));
  size = hi - lo;
  ut = TMP_ALLOC_LIMBS (size + 1);
  vt = TMP_ALLOC_LIMBS (size + 1);
  wt = TMP_ALLOC_LIMBS (size + 1);
  MPN_ZERO (ut, size);
  MPN_ZERO (vt, size);
  {int off;
  off = size + (EXP (u) - hi) - ABSIZ (u);
  MPN_COPY (ut + off, PTR (u), ABSIZ (u));
  off = size + (EXP (v) - hi) - ABSIZ (v);
  MPN_COPY (vt + off, PTR (v), ABSIZ (v));
  }

  if (mpn_cmp (ut, vt, size) >= 0)
    mpn_sub_n (wt, ut, vt, size);
  else
    {
      mpn_sub_n (wt, vt, ut, size);
      neg ^= 1;
    }
  exp = hi;
  while (size != 0 && wt[size - 1] == 0)
    {
      size--;
      exp--;
    }

done:
  if (size > PREC (w))
    {
      wt += size - PREC (w);
      size = PREC (w);
    }
  MPN_COPY (PTR (w), wt, size);
  SIZ (w) = neg == 0 ? size : -size;
  EXP (w) = exp;
  TMP_FREE;
}


/* Validate got by comparing to want.  Return 1 if good, 0 if bad.

   The data in got is compared to that in want, up to either PREC(got) limbs
   or the size of got, whichever is bigger.  Clearly we always demand
   PREC(got) of accuracy, but we go further and say that if got is bigger
   then any extra must be correct too.

   want needs to have enough data to allow this comparison.  The size in
   want doesn't have to be that big though, if it's smaller then further low
   limbs are taken to be zero.

   This validation approach is designed to allow some flexibility in exactly
   how much data is generated by an mpf function, ie. either prec or prec+1
   limbs.  We don't try to make a reference function that emulates that same
   size decision, instead the idea is for a validation function to generate
   at least as much data as the real function, then compare.  */

int
refmpf_validate (const char *name, mpf_srcptr got, mpf_srcptr want)
{
  int  bad = 0;
  mp_size_t  gsize, wsize, cmpsize, i;
  mp_srcptr  gp, wp;
  mp_limb_t  glimb, wlimb;

  MPF_CHECK_FORMAT (got);

  if (EXP (got) != EXP (want))
    {
      printf ("%s: wrong exponent\n", name);
      bad = 1;
    }

  gsize = SIZ (got);
  wsize = SIZ (want);
  if ((gsize < 0 && wsize > 0) || (gsize > 0 && wsize < 0))
    {
      printf ("%s: wrong sign\n", name);
      bad = 1;
    }

  gsize = ABS (gsize);
  wsize = ABS (wsize);

  /* most significant limb of respective data */
  gp = PTR (got) + gsize - 1;
  wp = PTR (want) + wsize - 1;

  /* compare limb data */
  cmpsize = MAX (PREC (got), gsize);
  for (i = 0; i < cmpsize; i++)
    {
      glimb = (i < gsize ? gp[-i] : 0);
      wlimb = (i < wsize ? wp[-i] : 0);

      if (glimb != wlimb)
        {
          printf ("%s: wrong data starting at index %ld from top\n",
                  name, (long) i);
          bad = 1;
          break;
        }
    }

  if (bad)
    {
      printf ("  prec       %d\n", PREC(got));
      printf ("  exp got    %ld\n", (long) EXP(got));
      printf ("  exp want   %ld\n", (long) EXP(want));
      printf ("  size got   %d\n", SIZ(got));
      printf ("  size want  %d\n", SIZ(want));
      printf ("  limbs (high to low)\n");
      printf ("   got  ");
      for (i = ABSIZ(got)-1; i >= 0; i--)
        {
          gmp_printf ("%MX", PTR(got)[i]);
          if (i != 0)
            printf (",");
        }
      printf ("\n");
      printf ("   want ");
      for (i = ABSIZ(want)-1; i >= 0; i--)
        {
          gmp_printf ("%MX", PTR(want)[i]);
          if (i != 0)
            printf (",");
        }
      printf ("\n");
      return 0;
    }

  return 1;
}


int
refmpf_validate_division (const char *name, mpf_srcptr got,
                          mpf_srcptr n, mpf_srcptr d)
{
  mp_size_t  nsize, dsize, sign, prec, qsize, tsize;
  mp_srcptr  np, dp;
  mp_ptr     tp, qp, rp;
  mpf_t      want;
  int        ret;

  nsize = SIZ (n);
  dsize = SIZ (d);
  ASSERT_ALWAYS (dsize != 0);

  sign = nsize ^ dsize;
  nsize = ABS (nsize);
  dsize = ABS (dsize);

  np = PTR (n);
  dp = PTR (d);
  prec = PREC (got);

  EXP (want) = EXP (n) - EXP (d) + 1;

  qsize = prec + 2;            /* at least prec+1 limbs, after high zero */
  tsize = qsize + dsize - 1;   /* dividend size to give desired qsize */

  /* dividend n, extended or truncated */
  tp = refmpn_malloc_limbs (tsize);
  refmpn_copy_extend (tp, tsize, np, nsize);

  qp = refmpn_malloc_limbs (qsize);
  rp = refmpn_malloc_limbs (dsize);  /* remainder, unused */

  ASSERT_ALWAYS (qsize == tsize - dsize + 1);
  refmpn_tdiv_qr (qp, rp, (mp_size_t) 0, tp, tsize, dp, dsize);

  PTR (want) = qp;
  SIZ (want) = (sign >= 0 ? qsize : -qsize);
  refmpf_normalize (want);

  ret = refmpf_validate (name, got, want);

  free (tp);
  free (qp);
  free (rp);

  return ret;
}
