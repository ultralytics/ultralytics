/* Generate perfect square testing data.

Copyright 2002-2004, 2012, 2014 Free Software Foundation, Inc.

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

#include <stdio.h>
#include <stdlib.h>

#include "bootstrap.c"


/* The aim of this program is to choose either mpn_mod_34lsub1 or mpn_mod_1
   (plus a PERFSQR_PP modulus), and generate tables indicating quadratic
   residues and non-residues modulo small factors of that modulus.

   For the usual 32 or 64 bit cases mpn_mod_34lsub1 gets used.  That
   function exists specifically because 2^24-1 and 2^48-1 have nice sets of
   prime factors.  For other limb sizes it's considered, but if it doesn't
   have good factors then mpn_mod_1 will be used instead.

   When mpn_mod_1 is used, the modulus PERFSQR_PP is created from a
   selection of small primes, chosen to fill PERFSQR_MOD_BITS of a limb,
   with that bit count chosen so (2*GMP_LIMB_BITS)*2^PERFSQR_MOD_BITS <=
   GMP_LIMB_MAX, allowing PERFSQR_MOD_IDX in mpn/generic/perfsqr.c to do its
   calculation within a single limb.

   In either case primes can be combined to make divisors.  The table data
   then effectively indicates remainders which are quadratic residues mod
   all the primes.  This sort of combining reduces the number of steps
   needed after mpn_mod_34lsub1 or mpn_mod_1, saving code size and time.
   Nothing is gained or lost in terms of detections, the same total fraction
   of non-residues will be identified.

   Nothing particularly sophisticated is attempted for combining factors to
   make divisors.  This is probably a kind of knapsack problem so it'd be
   too hard to attempt anything completely general.  For the usual 32 and 64
   bit limbs we get a good enough result just pairing the biggest and
   smallest which fit together, repeatedly.

   Another aim is to get powerful combinations, ie. divisors which identify
   biggest fraction of non-residues, and have those run first.  Again for
   the usual 32 and 64 bits it seems good enough just to pair for big
   divisors then sort according to the resulting fraction of non-residues
   identified.

   Also in this program, a table sq_res_0x100 of residues modulo 256 is
   generated.  This simply fills bits into limbs of the appropriate
   build-time GMP_LIMB_BITS each.

*/


/* Normally we aren't using const in gen*.c programs, so as not to have to
   bother figuring out if it works, but using it with f_cmp_divisor and
   f_cmp_fraction avoids warnings from the qsort calls. */

/* Same tests as gmp.h. */
#if  defined (__STDC__)                                 \
  || defined (__cplusplus)                              \
  || defined (_AIX)                                     \
  || defined (__DECC)                                   \
  || (defined (__mips) && defined (_SYSTYPE_SVR4))      \
  || defined (_MSC_VER)                                 \
  || defined (_WIN32)
#define HAVE_CONST        1
#endif

#if ! HAVE_CONST
#define const
#endif


mpz_t  *sq_res_0x100;          /* table of limbs */
int    nsq_res_0x100;          /* elements in sq_res_0x100 array */
int    sq_res_0x100_num;       /* squares in sq_res_0x100 */
double sq_res_0x100_fraction;  /* sq_res_0x100_num / 256 */

int     mod34_bits;        /* 3*GMP_NUMB_BITS/4 */
int     mod_bits;          /* bits from PERFSQR_MOD_34 or MOD_PP */
int     max_divisor;       /* all divisors <= max_divisor */
int     max_divisor_bits;  /* ceil(log2(max_divisor)) */
double  total_fraction;    /* of squares */
mpz_t   pp;                /* product of primes, or 0 if mod_34lsub1 used */
mpz_t   pp_norm;           /* pp shifted so NUMB high bit set */
mpz_t   pp_inverted;       /* invert_limb style inverse */
mpz_t   mod_mask;          /* 2^mod_bits-1 */
char    mod34_excuse[128]; /* why mod_34lsub1 not used (if it's not) */

/* raw list of divisors of 2^mod34_bits-1 or pp, just to show in a comment */
struct rawfactor_t {
  int     divisor;
  int     multiplicity;
};
struct rawfactor_t  *rawfactor;
int                 nrawfactor;

/* factors of 2^mod34_bits-1 or pp and associated data, after combining etc */
struct factor_t {
  int     divisor;
  mpz_t   inverse;   /* 1/divisor mod 2^mod_bits */
  mpz_t   mask;      /* indicating squares mod divisor */
  double  fraction;  /* squares/total */
};
struct factor_t  *factor;
int              nfactor;       /* entries in use in factor array */
int              factor_alloc;  /* entries allocated to factor array */


int
f_cmp_divisor (const void *parg, const void *qarg)
{
  const struct factor_t *p, *q;
  p = (const struct factor_t *) parg;
  q = (const struct factor_t *) qarg;
  if (p->divisor > q->divisor)
    return 1;
  else if (p->divisor < q->divisor)
    return -1;
  else
    return 0;
}

int
f_cmp_fraction (const void *parg, const void *qarg)
{
  const struct factor_t *p, *q;
  p = (const struct factor_t *) parg;
  q = (const struct factor_t *) qarg;
  if (p->fraction > q->fraction)
    return 1;
  else if (p->fraction < q->fraction)
    return -1;
  else
    return 0;
}

/* Remove array[idx] by copying the remainder down, and adjust narray
   accordingly.  */
#define COLLAPSE_ELEMENT(array, idx, narray)                    \
  do {                                                          \
    memmove (&(array)[idx],					\
	     &(array)[idx+1],					\
	     ((narray)-((idx)+1)) * sizeof (array[0]));		\
    (narray)--;                                                 \
  } while (0)


/* return n*2^p mod m */
int
mul_2exp_mod (int n, int p, int m)
{
  while (--p >= 0)
    n = (2 * n) % m;
  return n;
}

/* return -n mod m */
int
neg_mod (int n, int m)
{
  assert (n >= 0 && n < m);
  return (n == 0 ? 0 : m-n);
}

/* Set "mask" to a value such that "mask & (1<<idx)" is non-zero if
   "-(idx<<mod_bits)" can be a square modulo m.  */
void
square_mask (mpz_t mask, int m)
{
  int    p, i, r, idx;

  p = mul_2exp_mod (1, mod_bits, m);
  p = neg_mod (p, m);

  mpz_set_ui (mask, 0L);
  for (i = 0; i < m; i++)
    {
      r = (i * i) % m;
      idx = (r * p) % m;
      mpz_setbit (mask, (unsigned long) idx);
    }
}

void
generate_sq_res_0x100 (int limb_bits)
{
  int  i, res;

  nsq_res_0x100 = (0x100 + limb_bits - 1) / limb_bits;
  sq_res_0x100 = (mpz_t *) xmalloc (nsq_res_0x100 * sizeof (*sq_res_0x100));

  for (i = 0; i < nsq_res_0x100; i++)
    mpz_init_set_ui (sq_res_0x100[i], 0L);

  for (i = 0; i < 0x100; i++)
    {
      res = (i * i) % 0x100;
      mpz_setbit (sq_res_0x100[res / limb_bits],
                  (unsigned long) (res % limb_bits));
    }

  sq_res_0x100_num = 0;
  for (i = 0; i < nsq_res_0x100; i++)
    sq_res_0x100_num += mpz_popcount (sq_res_0x100[i]);
  sq_res_0x100_fraction = (double) sq_res_0x100_num / 256.0;
}

void
generate_mod (int limb_bits, int nail_bits)
{
  int    numb_bits = limb_bits - nail_bits;
  int    i, divisor;

  mpz_init_set_ui (pp, 0L);
  mpz_init_set_ui (pp_norm, 0L);
  mpz_init_set_ui (pp_inverted, 0L);

  /* no more than limb_bits many factors in a one limb modulus (and of
     course in reality nothing like that many) */
  factor_alloc = limb_bits;
  factor = (struct factor_t *) xmalloc (factor_alloc * sizeof (*factor));
  rawfactor = (struct rawfactor_t *) xmalloc (factor_alloc * sizeof (*rawfactor));

  if (numb_bits % 4 != 0)
    {
      strcpy (mod34_excuse, "GMP_NUMB_BITS % 4 != 0");
      goto use_pp;
    }

  max_divisor = 2*limb_bits;
  max_divisor_bits = log2_ceil (max_divisor);

  if (numb_bits / 4 < max_divisor_bits)
    {
      /* Wind back to one limb worth of max_divisor, if that will let us use
         mpn_mod_34lsub1.  */
      max_divisor = limb_bits;
      max_divisor_bits = log2_ceil (max_divisor);

      if (numb_bits / 4 < max_divisor_bits)
        {
          strcpy (mod34_excuse, "GMP_NUMB_BITS / 4 too small");
          goto use_pp;
        }
    }

  {
    /* Can use mpn_mod_34lsub1, find small factors of 2^mod34_bits-1. */
    mpz_t  m, q, r;
    int    multiplicity;

    mod34_bits = (numb_bits / 4) * 3;

    /* mpn_mod_34lsub1 returns a full limb value, PERFSQR_MOD_34 folds it at
       the mod34_bits mark, adding the two halves for a remainder of at most
       mod34_bits+1 many bits */
    mod_bits = mod34_bits + 1;

    mpz_init_set_ui (m, 1L);
    mpz_mul_2exp (m, m, mod34_bits);
    mpz_sub_ui (m, m, 1L);

    mpz_init (q);
    mpz_init (r);

    for (i = 3; i <= max_divisor; i+=2)
      {
        if (! isprime (i))
          continue;

        mpz_tdiv_qr_ui (q, r, m, (unsigned long) i);
        if (mpz_sgn (r) != 0)
          continue;

        /* if a repeated prime is found it's used as an i^n in one factor */
        divisor = 1;
        multiplicity = 0;
        do
          {
            if (divisor > max_divisor / i)
              break;
            multiplicity++;
            mpz_set (m, q);
            mpz_tdiv_qr_ui (q, r, m, (unsigned long) i);
          }
        while (mpz_sgn (r) == 0);

        assert (nrawfactor < factor_alloc);
        rawfactor[nrawfactor].divisor = i;
        rawfactor[nrawfactor].multiplicity = multiplicity;
        nrawfactor++;
      }

    mpz_clear (m);
    mpz_clear (q);
    mpz_clear (r);
  }

  if (nrawfactor <= 2)
    {
      mpz_t  new_pp;

      sprintf (mod34_excuse, "only %d small factor%s",
               nrawfactor, nrawfactor == 1 ? "" : "s");

    use_pp:
      /* reset to two limbs of max_divisor, in case the mpn_mod_34lsub1 code
         tried with just one */
      max_divisor = 2*limb_bits;
      max_divisor_bits = log2_ceil (max_divisor);

      mpz_init (new_pp);
      nrawfactor = 0;
      mod_bits = MIN (numb_bits, limb_bits - max_divisor_bits);

      /* one copy of each small prime */
      mpz_set_ui (pp, 1L);
      for (i = 3; i <= max_divisor; i+=2)
        {
          if (! isprime (i))
            continue;

          mpz_mul_ui (new_pp, pp, (unsigned long) i);
          if (mpz_sizeinbase (new_pp, 2) > mod_bits)
            break;
          mpz_set (pp, new_pp);

          assert (nrawfactor < factor_alloc);
          rawfactor[nrawfactor].divisor = i;
          rawfactor[nrawfactor].multiplicity = 1;
          nrawfactor++;
        }

      /* Plus an extra copy of one or more of the primes selected, if that
         still fits in max_divisor and the total in mod_bits.  Usually only
         3 or 5 will be candidates */
      for (i = nrawfactor-1; i >= 0; i--)
        {
          if (rawfactor[i].divisor > max_divisor / rawfactor[i].divisor)
            continue;
          mpz_mul_ui (new_pp, pp, (unsigned long) rawfactor[i].divisor);
          if (mpz_sizeinbase (new_pp, 2) > mod_bits)
            continue;
          mpz_set (pp, new_pp);

          rawfactor[i].multiplicity++;
        }

      mod_bits = mpz_sizeinbase (pp, 2);

      mpz_set (pp_norm, pp);
      while (mpz_sizeinbase (pp_norm, 2) < numb_bits)
        mpz_add (pp_norm, pp_norm, pp_norm);

      mpz_preinv_invert (pp_inverted, pp_norm, numb_bits);

      mpz_clear (new_pp);
    }

  /* start the factor array */
  for (i = 0; i < nrawfactor; i++)
    {
      int  j;
      assert (nfactor < factor_alloc);
      factor[nfactor].divisor = 1;
      for (j = 0; j < rawfactor[i].multiplicity; j++)
        factor[nfactor].divisor *= rawfactor[i].divisor;
      nfactor++;
    }

 combine:
  /* Combine entries in the factor array.  Combine the smallest entry with
     the biggest one that will fit with it (ie. under max_divisor), then
     repeat that with the new smallest entry. */
  qsort (factor, nfactor, sizeof (factor[0]), f_cmp_divisor);
  for (i = nfactor-1; i >= 1; i--)
    {
      if (factor[i].divisor <= max_divisor / factor[0].divisor)
        {
          factor[0].divisor *= factor[i].divisor;
          COLLAPSE_ELEMENT (factor, i, nfactor);
          goto combine;
        }
    }

  total_fraction = 1.0;
  for (i = 0; i < nfactor; i++)
    {
      mpz_init (factor[i].inverse);
      mpz_invert_ui_2exp (factor[i].inverse,
                          (unsigned long) factor[i].divisor,
                          (unsigned long) mod_bits);

      mpz_init (factor[i].mask);
      square_mask (factor[i].mask, factor[i].divisor);

      /* fraction of possible squares */
      factor[i].fraction = (double) mpz_popcount (factor[i].mask)
        / factor[i].divisor;

      /* total fraction of possible squares */
      total_fraction *= factor[i].fraction;
    }

  /* best tests first (ie. smallest fraction) */
  qsort (factor, nfactor, sizeof (factor[0]), f_cmp_fraction);
}

void
print (int limb_bits, int nail_bits)
{
  int    i;
  mpz_t  mhi, mlo;

  printf ("/* This file generated by gen-psqr.c - DO NOT EDIT. */\n");
  printf ("\n");

  printf ("#if GMP_LIMB_BITS != %d || GMP_NAIL_BITS != %d\n",
          limb_bits, nail_bits);
  printf ("Error, error, this data is for %d bit limb and %d bit nail\n",
          limb_bits, nail_bits);
  printf ("#endif\n");
  printf ("\n");

  printf ("/* Non-zero bit indicates a quadratic residue mod 0x100.\n");
  printf ("   This test identifies %.2f%% as non-squares (%d/256). */\n",
          (1.0 - sq_res_0x100_fraction) * 100.0,
          0x100 - sq_res_0x100_num);
  printf ("static const mp_limb_t\n");
  printf ("sq_res_0x100[%d] = {\n", nsq_res_0x100);
  for (i = 0; i < nsq_res_0x100; i++)
    {
      printf ("  CNST_LIMB(0x");
      mpz_out_str (stdout, 16, sq_res_0x100[i]);
      printf ("),\n");
    }
  printf ("};\n");
  printf ("\n");

  if (mpz_sgn (pp) != 0)
    {
      printf ("/* mpn_mod_34lsub1 not used due to %s */\n", mod34_excuse);
      printf ("/* PERFSQR_PP = ");
    }
  else
    printf ("/* 2^%d-1 = ", mod34_bits);
  for (i = 0; i < nrawfactor; i++)
    {
      if (i != 0)
        printf (" * ");
      printf ("%d", rawfactor[i].divisor);
      if (rawfactor[i].multiplicity != 1)
        printf ("^%d", rawfactor[i].multiplicity);
    }
  printf (" %s*/\n", mpz_sgn (pp) == 0 ? "... " : "");

  printf ("#define PERFSQR_MOD_BITS  %d\n", mod_bits);
  if (mpz_sgn (pp) != 0)
    {
      printf ("#define PERFSQR_PP            CNST_LIMB(0x");
      mpz_out_str (stdout, 16, pp);
      printf (")\n");
      printf ("#define PERFSQR_PP_NORM       CNST_LIMB(0x");
      mpz_out_str (stdout, 16, pp_norm);
      printf (")\n");
      printf ("#define PERFSQR_PP_INVERTED   CNST_LIMB(0x");
      mpz_out_str (stdout, 16, pp_inverted);
      printf (")\n");
    }
  printf ("\n");

  mpz_init (mhi);
  mpz_init (mlo);

  printf ("/* This test identifies %.2f%% as non-squares. */\n",
          (1.0 - total_fraction) * 100.0);
  printf ("#define PERFSQR_MOD_TEST(up, usize) \\\n");
  printf ("  do {                              \\\n");
  printf ("    mp_limb_t  r;                   \\\n");
  if (mpz_sgn (pp) != 0)
    printf ("    PERFSQR_MOD_PP (r, up, usize);  \\\n");
  else
    printf ("    PERFSQR_MOD_34 (r, up, usize);  \\\n");

  for (i = 0; i < nfactor; i++)
    {
      printf ("                                    \\\n");
      printf ("    /* %5.2f%% */                    \\\n",
              (1.0 - factor[i].fraction) * 100.0);

      printf ("    PERFSQR_MOD_%d (r, CNST_LIMB(%2d), CNST_LIMB(0x",
              factor[i].divisor <= limb_bits ? 1 : 2,
              factor[i].divisor);
      mpz_out_str (stdout, 16, factor[i].inverse);
      printf ("), \\\n");
      printf ("                   CNST_LIMB(0x");

      if ( factor[i].divisor <= limb_bits)
        {
          mpz_out_str (stdout, 16, factor[i].mask);
        }
      else
        {
          mpz_tdiv_r_2exp (mlo, factor[i].mask, (unsigned long) limb_bits);
          mpz_tdiv_q_2exp (mhi, factor[i].mask, (unsigned long) limb_bits);
          mpz_out_str (stdout, 16, mhi);
          printf ("), CNST_LIMB(0x");
          mpz_out_str (stdout, 16, mlo);
        }
      printf (")); \\\n");
    }

  printf ("  } while (0)\n");
  printf ("\n");

  printf ("/* Grand total sq_res_0x100 and PERFSQR_MOD_TEST, %.2f%% non-squares. */\n",
          (1.0 - (total_fraction * 44.0/256.0)) * 100.0);
  printf ("\n");

  printf ("/* helper for tests/mpz/t-perfsqr.c */\n");
  printf ("#define PERFSQR_DIVISORS  { 256,");
  for (i = 0; i < nfactor; i++)
      printf (" %d,", factor[i].divisor);
  printf (" }\n");


  mpz_clear (mhi);
  mpz_clear (mlo);
}

int
main (int argc, char *argv[])
{
  int  limb_bits, nail_bits;

  if (argc != 3)
    {
      fprintf (stderr, "Usage: gen-psqr <limbbits> <nailbits>\n");
      exit (1);
    }

  limb_bits = atoi (argv[1]);
  nail_bits = atoi (argv[2]);

  if (limb_bits <= 0
      || nail_bits < 0
      || nail_bits >= limb_bits)
    {
      fprintf (stderr, "Invalid limb/nail bits: %d %d\n",
               limb_bits, nail_bits);
      exit (1);
    }

  generate_sq_res_0x100 (limb_bits);
  generate_mod (limb_bits, nail_bits);

  print (limb_bits, nail_bits);

  return 0;
}
