/* Test file for mpfr_add_[q,z], mpfr_sub_[q,z], mpfr_div_[q,z],
   mpfr_mul_[q,z], mpfr_cmp_[f,q,z]

Copyright 2004-2017 Free Software Foundation, Inc.
Contributed by the AriC and Caramba projects, INRIA.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

#include <stdio.h>
#include <stdlib.h>
#include "mpfr-test.h"

#define CHECK_FOR(str, cond)                                            \
  if ((cond) == 0) {                                                    \
    printf ("Special case error %s. Ternary value = %d, flags = %u\n",  \
            str, res, __gmpfr_flags);                                   \
    printf ("Got "); mpfr_dump (y);                                     \
    printf ("X = "); mpfr_dump (x);                                     \
    printf ("Q = "); mpz_dump (mpq_numref(q));                          \
    printf ("   /"); mpz_dump (mpq_denref(q));                          \
    exit (1);                                                           \
  }

#define CHECK_FORZ(str, cond)                                           \
  if ((cond) == 0) {                                                    \
    printf ("Special case error %s. Ternary value = %d, flags = %u\n",  \
            str, res, __gmpfr_flags);                                   \
    printf ("Got "); mpfr_dump (y);                                     \
    printf ("X = "); mpfr_dump (x);                                     \
    printf ("Z = "); mpz_dump (z);                                      \
    exit (1);                                                           \
  }

static void
special (void)
{
  mpfr_t x, y;
  mpq_t q;
  mpz_t z;
  int res = 0;

  mpfr_init (x);
  mpfr_init (y);
  mpq_init (q);
  mpz_init (z);

  /* cancellation in mpfr_add_q */
  mpfr_set_prec (x, 60);
  mpfr_set_prec (y, 20);
  mpz_set_str (mpq_numref (q), "-187207494", 10);
  mpz_set_str (mpq_denref (q), "5721", 10);
  mpfr_set_str_binary (x, "11111111101001011011100101100011011110010011100010000100001E-44");
  mpfr_add_q (y, x, q, MPFR_RNDN);
  CHECK_FOR ("cancelation in add_q", mpfr_cmp_ui_2exp (y, 256783, -64) == 0);

  mpfr_set_prec (x, 19);
  mpfr_set_str_binary (x, "0.1011110101110011100E0");
  mpz_set_str (mpq_numref (q), "187207494", 10);
  mpz_set_str (mpq_denref (q), "5721", 10);
  mpfr_set_prec (y, 29);
  mpfr_add_q (y, x, q, MPFR_RNDD);
  mpfr_set_prec (x, 29);
  mpfr_set_str_binary (x, "11111111101001110011010001001E-14");
  CHECK_FOR ("cancelation in add_q", mpfr_cmp (x,y) == 0);

  /* Inf */
  mpfr_set_inf (x, 1);
  mpz_set_str (mpq_numref (q), "395877315", 10);
  mpz_set_str (mpq_denref (q), "3508975966", 10);
  mpfr_set_prec (y, 118);
  mpfr_add_q (y, x, q, MPFR_RNDU);
  CHECK_FOR ("inf", mpfr_inf_p (y) && mpfr_sgn (y) > 0);
  mpfr_sub_q (y, x, q, MPFR_RNDU);
  CHECK_FOR ("inf", mpfr_inf_p (y) && mpfr_sgn (y) > 0);

  /* Nan */
  MPFR_SET_NAN (x);
  mpfr_add_q (y, x, q, MPFR_RNDU);
  CHECK_FOR ("nan", mpfr_nan_p (y));
  mpfr_sub_q (y, x, q, MPFR_RNDU);
  CHECK_FOR ("nan", mpfr_nan_p (y));

  /* Exact value */
  mpfr_set_prec (x, 60);
  mpfr_set_prec (y, 60);
  mpfr_set_str1 (x, "0.5");
  mpz_set_str (mpq_numref (q), "3", 10);
  mpz_set_str (mpq_denref (q), "2", 10);
  res = mpfr_add_q (y, x, q, MPFR_RNDU);
  CHECK_FOR ("0.5+3/2", mpfr_cmp_ui(y, 2)==0 && res==0);
  res = mpfr_sub_q (y, x, q, MPFR_RNDU);
  CHECK_FOR ("0.5-3/2", mpfr_cmp_si(y, -1)==0 && res==0);

  /* Inf Rationnal */
  mpq_set_ui (q, 1, 0);
  mpfr_set_str1 (x, "0.5");
  res = mpfr_add_q (y, x, q, MPFR_RNDN);
  CHECK_FOR ("0.5+1/0", mpfr_inf_p (y) && MPFR_SIGN (y) > 0 && res == 0);
  res = mpfr_sub_q (y, x, q, MPFR_RNDN);
  CHECK_FOR ("0.5-1/0", mpfr_inf_p (y) && MPFR_SIGN (y) < 0 && res == 0);
  mpq_set_si (q, -1, 0);
  res = mpfr_add_q (y, x, q, MPFR_RNDN);
  CHECK_FOR ("0.5+ -1/0", mpfr_inf_p (y) && MPFR_SIGN (y) < 0 && res == 0);
  res = mpfr_sub_q (y, x, q, MPFR_RNDN);
  CHECK_FOR ("0.5- -1/0", mpfr_inf_p (y) && MPFR_SIGN (y) > 0 && res == 0);
  res = mpfr_div_q (y, x, q, MPFR_RNDN);
  CHECK_FOR ("0.5 / (-1/0)", mpfr_zero_p (y) && MPFR_SIGN (y) < 0 && res == 0);
  mpq_set_ui (q, 1, 0);
  mpfr_set_inf (x, 1);
  res = mpfr_add_q (y, x, q, MPFR_RNDN);
  CHECK_FOR ("+Inf + +Inf", mpfr_inf_p (y) && MPFR_SIGN (y) > 0 && res == 0);
  res = mpfr_sub_q (y, x, q, MPFR_RNDN);
  CHECK_FOR ("+Inf - +Inf", MPFR_IS_NAN (y) && res == 0);
  mpfr_set_inf (x, -1);
  res = mpfr_add_q (y, x, q, MPFR_RNDN);
  CHECK_FOR ("-Inf + +Inf", MPFR_IS_NAN (y) && res == 0);
  res = mpfr_sub_q (y, x, q, MPFR_RNDN);
  CHECK_FOR ("-Inf - +Inf", mpfr_inf_p (y) && MPFR_SIGN (y) < 0 && res == 0);
  mpq_set_si (q, -1, 0);
  mpfr_set_inf (x, 1);
  res = mpfr_add_q (y, x, q, MPFR_RNDN);
  CHECK_FOR ("+Inf + -Inf", MPFR_IS_NAN (y) && res == 0);
  res = mpfr_sub_q (y, x, q, MPFR_RNDN);
  CHECK_FOR ("+Inf - -Inf", mpfr_inf_p (y) && MPFR_SIGN (y) > 0 && res == 0);
  mpfr_set_inf (x, -1);
  res = mpfr_add_q (y, x, q, MPFR_RNDN);
  CHECK_FOR ("-Inf + -Inf", mpfr_inf_p (y) && MPFR_SIGN (y) < 0 && res == 0);
  res = mpfr_sub_q (y, x, q, MPFR_RNDN);
  CHECK_FOR ("-Inf - -Inf", MPFR_IS_NAN (y) && res == 0);

  /* 0 */
  mpq_set_ui (q, 0, 1);
  mpfr_set_ui (x, 42, MPFR_RNDN);
  res = mpfr_add_q (y, x, q, MPFR_RNDN);
  CHECK_FOR ("42+0/1", mpfr_cmp_ui (y, 42) == 0 && res == 0);
  res = mpfr_sub_q (y, x, q, MPFR_RNDN);
  CHECK_FOR ("42-0/1", mpfr_cmp_ui (y, 42) == 0 && res == 0);
  res = mpfr_mul_q (y, x, q, MPFR_RNDN);
  CHECK_FOR ("42*0/1", mpfr_zero_p (y) && MPFR_SIGN (y) > 0 && res == 0);
  mpfr_clear_flags ();
  res = mpfr_div_q (y, x, q, MPFR_RNDN);
  CHECK_FOR ("42/(0/1)", mpfr_inf_p (y) && MPFR_SIGN (y) > 0 && res == 0
             && mpfr_divby0_p ());
  mpz_set_ui (z, 0);
  mpfr_clear_flags ();
  res = mpfr_div_z (y, x, z, MPFR_RNDN);
  CHECK_FORZ ("42/0", mpfr_inf_p (y) && MPFR_SIGN (y) > 0 && res == 0
              && mpfr_divby0_p ());

  mpz_clear (z);
  mpq_clear (q);
  mpfr_clear (x);
  mpfr_clear (y);
}

static void
check_for_zero (void)
{
  /* Check that 0 is unsigned! */
  mpq_t q;
  mpz_t z;
  mpfr_t x;
  int r;
  mpfr_sign_t i;

  mpfr_init (x);
  mpz_init (z);
  mpq_init (q);

  mpz_set_ui (z, 0);
  mpq_set_ui (q, 0, 1);

  MPFR_SET_ZERO (x);
  RND_LOOP (r)
    {
      for (i = MPFR_SIGN_NEG ; i <= MPFR_SIGN_POS ;
           i+=MPFR_SIGN_POS-MPFR_SIGN_NEG)
        {
          MPFR_SET_SIGN(x, i);
          mpfr_add_z (x, x, z, (mpfr_rnd_t) r);
          if (!MPFR_IS_ZERO(x) || MPFR_SIGN(x)!=i)
            {
              printf("GMP Zero errors for add_z & rnd=%s & s=%d\n",
                     mpfr_print_rnd_mode ((mpfr_rnd_t) r), i);
              mpfr_dump (x);
              exit (1);
            }
          mpfr_sub_z (x, x, z, (mpfr_rnd_t) r);
          if (!MPFR_IS_ZERO(x) || MPFR_SIGN(x)!=i)
            {
              printf("GMP Zero errors for sub_z & rnd=%s & s=%d\n",
                     mpfr_print_rnd_mode ((mpfr_rnd_t) r), i);
              mpfr_dump (x);
              exit (1);
            }
          mpfr_mul_z (x, x, z, (mpfr_rnd_t) r);
          if (!MPFR_IS_ZERO(x) || MPFR_SIGN(x)!=i)
            {
              printf("GMP Zero errors for mul_z & rnd=%s & s=%d\n",
                     mpfr_print_rnd_mode ((mpfr_rnd_t) r), i);
              mpfr_dump (x);
              exit (1);
            }
          mpfr_add_q (x, x, q, (mpfr_rnd_t) r);
          if (!MPFR_IS_ZERO(x) || MPFR_SIGN(x)!=i)
            {
              printf("GMP Zero errors for add_q & rnd=%s & s=%d\n",
                     mpfr_print_rnd_mode ((mpfr_rnd_t) r), i);
              mpfr_dump (x);
              exit (1);
            }
          mpfr_sub_q (x, x, q, (mpfr_rnd_t) r);
          if (!MPFR_IS_ZERO(x) || MPFR_SIGN(x)!=i)
            {
              printf("GMP Zero errors for sub_q & rnd=%s & s=%d\n",
                     mpfr_print_rnd_mode ((mpfr_rnd_t) r), i);
              mpfr_dump (x);
              exit (1);
             }
        }
    }

  mpq_clear (q);
  mpz_clear (z);
  mpfr_clear (x);
}

static void
test_cmp_z (mpfr_prec_t pmin, mpfr_prec_t pmax, int nmax)
{
  mpfr_t x, z;
  mpz_t  y;
  mpfr_prec_t p;
  int res1, res2;
  int n;

  mpfr_init (x);
  mpfr_init2 (z, MPFR_PREC_MIN);
  mpz_init (y);

  /* check the erange flag when x is NaN */
  mpfr_set_nan (x);
  mpz_set_ui (y, 17);
  mpfr_clear_erangeflag ();
  res1 = mpfr_cmp_z (x, y);
  if (res1 != 0 || mpfr_erangeflag_p () == 0)
    {
      printf ("Error for mpfr_cmp_z (NaN, 17)\n");
      printf ("Return value: expected 0, got %d\n", res1);
      printf ("Erange flag: expected set, got %d\n", mpfr_erangeflag_p ());
      exit (1);
    }

  for(p=pmin ; p < pmax ; p++)
    {
      mpfr_set_prec (x, p);
      for ( n = 0; n < nmax ; n++)
        {
          mpfr_urandomb (x, RANDS);
          mpz_urandomb  (y, RANDS, 1024);
          if (!MPFR_IS_SINGULAR (x))
            {
              mpfr_sub_z (z, x, y, MPFR_RNDN);
              res1 = mpfr_sgn (z);
              res2 = mpfr_cmp_z (x, y);
              if (res1 != res2)
                {
                  printf("Error for mpfr_cmp_z: res=%d sub_z gives %d\n",
                         res2, res1);
                  exit (1);
                }
            }
        }
    }
  mpz_clear (y);
  mpfr_clear (x);
  mpfr_clear (z);
}

static void
test_cmp_q (mpfr_prec_t pmin, mpfr_prec_t pmax, int nmax)
{
  mpfr_t x, z;
  mpq_t  y;
  mpfr_prec_t p;
  int res1, res2;
  int n;

  mpfr_init (x);
  mpfr_init2 (z, MPFR_PREC_MIN);
  mpq_init (y);

  /* check the erange flag when x is NaN */
  mpfr_set_nan (x);
  mpq_set_ui (y, 17, 1);
  mpfr_clear_erangeflag ();
  res1 = mpfr_cmp_q (x, y);
  if (res1 != 0 || mpfr_erangeflag_p () == 0)
    {
      printf ("Error for mpfr_cmp_q (NaN, 17)\n");
      printf ("Return value: expected 0, got %d\n", res1);
      printf ("Erange flag: expected set, got %d\n", mpfr_erangeflag_p ());
      exit (1);
    }

  for(p=pmin ; p < pmax ; p++)
    {
      mpfr_set_prec (x, p);
      for (n = 0 ; n < nmax ; n++)
        {
          mpfr_urandomb (x, RANDS);
          mpq_set_ui (y, randlimb (), randlimb() );
          if (!MPFR_IS_SINGULAR (x))
            {
              mpfr_sub_q (z, x, y, MPFR_RNDN);
              res1 = mpfr_sgn (z);
              res2 = mpfr_cmp_q (x, y);
              if (res1 != res2)
                {
                  printf("Error for mpfr_cmp_q: res=%d sub_z gives %d\n",
                         res2, res1);
                  exit (1);
                }
            }
        }
    }
  mpq_clear (y);
  mpfr_clear (x);
  mpfr_clear (z);
}

static void
test_cmp_f (mpfr_prec_t pmin, mpfr_prec_t pmax, int nmax)
{
  mpfr_t x, z;
  mpf_t  y;
  mpfr_prec_t p;
  int res1, res2;
  int n;

  mpfr_init (x);
  mpfr_init2 (z, pmax+GMP_NUMB_BITS);
  mpf_init2 (y, MPFR_PREC_MIN);

  /* check the erange flag when x is NaN */
  mpfr_set_nan (x);
  mpf_set_ui (y, 17);
  mpfr_clear_erangeflag ();
  res1 = mpfr_cmp_f (x, y);
  if (res1 != 0 || mpfr_erangeflag_p () == 0)
    {
      printf ("Error for mpfr_cmp_f (NaN, 17)\n");
      printf ("Return value: expected 0, got %d\n", res1);
      printf ("Erange flag: expected set, got %d\n", mpfr_erangeflag_p ());
      exit (1);
    }

  for(p=pmin ; p < pmax ; p+=3)
    {
      mpfr_set_prec (x, p);
      mpf_set_prec (y, p);
      for ( n = 0; n < nmax ; n++)
        {
          mpfr_urandomb (x, RANDS);
          mpf_urandomb  (y, RANDS, p);
          if (!MPFR_IS_SINGULAR (x))
            {
              mpfr_set_f (z, y, MPFR_RNDN);
              mpfr_sub   (z, x, z, MPFR_RNDN);
              res1 = mpfr_sgn (z);
              res2 = mpfr_cmp_f (x, y);
              if (res1 != res2)
                {
                  printf("Error for mpfr_cmp_f: res=%d sub gives %d\n",
                         res2, res1);
                  exit (1);
                }
            }
        }
    }
  mpf_clear (y);
  mpfr_clear (x);
  mpfr_clear (z);
}

static void
test_specialz (int (*mpfr_func)(mpfr_ptr, mpfr_srcptr, mpz_srcptr, mpfr_rnd_t),
               void (*mpz_func)(mpz_ptr, mpz_srcptr, mpz_srcptr),
               const char *op)
{
  mpfr_t x1, x2;
  mpz_t  z1, z2;
  int res;

  mpfr_inits2 (128, x1, x2, (mpfr_ptr) 0);
  mpz_init (z1); mpz_init(z2);
  mpz_fac_ui (z1, 19); /* 19!+1 fits perfectly in a 128 bits mantissa */
  mpz_add_ui (z1, z1, 1);
  mpz_fac_ui (z2, 20); /* 20!+1 fits perfectly in a 128 bits mantissa */
  mpz_add_ui (z2, z2, 1);

  res = mpfr_set_z(x1, z1, MPFR_RNDN);
  if (res)
    {
      printf("Specialz %s: set_z1 error\n", op);
      exit(1);
    }
  mpfr_set_z (x2, z2, MPFR_RNDN);
  if (res)
    {
      printf("Specialz %s: set_z2 error\n", op);
      exit(1);
    }

  /* (19!+1) * (20!+1) fits in a 128 bits number */
  res = mpfr_func(x1, x1, z2, MPFR_RNDN);
  if (res)
    {
      printf("Specialz %s: wrong inexact flag.\n", op);
      exit(1);
    }
  mpz_func(z1, z1, z2);
  res = mpfr_set_z (x2, z1, MPFR_RNDN);
  if (res)
    {
      printf("Specialz %s: set_z2 error\n", op);
      exit(1);
    }
  if (mpfr_cmp(x1, x2))
    {
      printf("Specialz %s: results differ.\nx1=", op);
      mpfr_print_binary(x1);
      printf("\nx2=");
      mpfr_print_binary(x2);
      printf ("\nZ2=");
      mpz_out_str (stdout, 2, z1);
      putchar('\n');
      exit(1);
    }

  mpz_set_ui (z1, 1);
  mpz_set_ui (z2, 0);
  mpfr_set_ui (x1, 1, MPFR_RNDN);
  mpz_func (z1, z1, z2);
  res = mpfr_func(x1, x1, z2, MPFR_RNDN);
  mpfr_set_z (x2, z1, MPFR_RNDN);
  if (mpfr_cmp(x1, x2))
    {
      printf("Specialz %s: results differ(2).\nx1=", op);
      mpfr_print_binary(x1);
      printf("\nx2=");
      mpfr_print_binary(x2);
      putchar('\n');
      exit(1);
    }

  mpz_clear (z1); mpz_clear(z2);
  mpfr_clears (x1, x2, (mpfr_ptr) 0);
}

static void
test_special2z (int (*mpfr_func)(mpfr_ptr, mpz_srcptr, mpfr_srcptr, mpfr_rnd_t),
               void (*mpz_func)(mpz_ptr, mpz_srcptr, mpz_srcptr),
               const char *op)
{
  mpfr_t x1, x2;
  mpz_t  z1, z2;
  int res;

  mpfr_inits2 (128, x1, x2, (mpfr_ptr) 0);
  mpz_init (z1); mpz_init(z2);
  mpz_fac_ui (z1, 19); /* 19!+1 fits perfectly in a 128 bits mantissa */
  mpz_add_ui (z1, z1, 1);
  mpz_fac_ui (z2, 20); /* 20!+1 fits perfectly in a 128 bits mantissa */
  mpz_add_ui (z2, z2, 1);

  res = mpfr_set_z(x1, z1, MPFR_RNDN);
  if (res)
    {
      printf("Special2z %s: set_z1 error\n", op);
      exit(1);
    }
  mpfr_set_z (x2, z2, MPFR_RNDN);
  if (res)
    {
      printf("Special2z %s: set_z2 error\n", op);
      exit(1);
    }

  /* (19!+1) * (20!+1) fits in a 128 bits number */
  res = mpfr_func(x1, z1, x2, MPFR_RNDN);
  if (res)
    {
      printf("Special2z %s: wrong inexact flag.\n", op);
      exit(1);
    }
  mpz_func(z1, z1, z2);
  res = mpfr_set_z (x2, z1, MPFR_RNDN);
  if (res)
    {
      printf("Special2z %s: set_z2 error\n", op);
      exit(1);
    }
  if (mpfr_cmp(x1, x2))
    {
      printf("Special2z %s: results differ.\nx1=", op);
      mpfr_print_binary(x1);
      printf("\nx2=");
      mpfr_print_binary(x2);
      printf ("\nZ2=");
      mpz_out_str (stdout, 2, z1);
      putchar('\n');
      exit(1);
    }

  mpz_set_ui (z1, 0);
  mpz_set_ui (z2, 1);
  mpfr_set_ui (x2, 1, MPFR_RNDN);
  res = mpfr_func(x1, z1, x2, MPFR_RNDN);
  mpz_func (z1, z1, z2);
  mpfr_set_z (x2, z1, MPFR_RNDN);
  if (mpfr_cmp(x1, x2))
    {
      printf("Special2z %s: results differ(2).\nx1=", op);
      mpfr_print_binary(x1);
      printf("\nx2=");
      mpfr_print_binary(x2);
      putchar('\n');
      exit(1);
    }

  mpz_clear (z1); mpz_clear(z2);
  mpfr_clears (x1, x2, (mpfr_ptr) 0);
}

static void
test_genericz (mpfr_prec_t p0, mpfr_prec_t p1, unsigned int N,
               int (*func)(mpfr_ptr, mpfr_srcptr, mpz_srcptr, mpfr_rnd_t),
               const char *op)
{
  mpfr_prec_t prec;
  mpfr_t arg1, dst_big, dst_small, tmp;
  mpz_t  arg2;
  mpfr_rnd_t rnd;
  int inexact, compare, compare2;
  unsigned int n;

  mpfr_inits (arg1, dst_big, dst_small, tmp, (mpfr_ptr) 0);
  mpz_init (arg2);

  for (prec = p0; prec <= p1; prec++)
    {
      mpfr_set_prec (arg1, prec);
      mpfr_set_prec (tmp, prec);
      mpfr_set_prec (dst_small, prec);

      for (n=0; n<N; n++)
        {
          mpfr_urandomb (arg1, RANDS);
          mpz_urandomb (arg2, RANDS, 1024);
          rnd = RND_RAND ();
          mpfr_set_prec (dst_big, 2*prec);
          compare = func(dst_big, arg1, arg2, rnd);
          if (mpfr_can_round (dst_big, 2*prec, rnd, rnd, prec))
            {
              mpfr_set (tmp, dst_big, rnd);
              inexact = func(dst_small, arg1, arg2, rnd);
              if (mpfr_cmp (tmp, dst_small))
                {
                  printf ("Results differ for prec=%u rnd_mode=%s and %s_z:\n"
                          "arg1=",
                          (unsigned) prec, mpfr_print_rnd_mode (rnd), op);
                  mpfr_print_binary (arg1);
                  printf("\narg2=");
                  mpz_out_str (stdout, 10, arg2);
                  printf ("\ngot      ");
                  mpfr_dump (dst_small);
                  printf ("expected ");
                  mpfr_dump (tmp);
                  printf ("approx   ");
                  mpfr_dump (dst_big);
                  exit (1);
                }
              compare2 = mpfr_cmp (tmp, dst_big);
              /* if rounding to nearest, cannot know the sign of t - f(x)
                 because of composed rounding: y = o(f(x)) and t = o(y) */
              if (compare * compare2 >= 0)
                compare = compare + compare2;
              else
                compare = inexact; /* cannot determine sign(t-f(x)) */
              if (((inexact == 0) && (compare != 0)) ||
                  ((inexact > 0) && (compare <= 0)) ||
                  ((inexact < 0) && (compare >= 0)))
                {
                  printf ("Wrong inexact flag for rnd=%s and %s_z:\n"
                          "expected %d, got %d\n",
                          mpfr_print_rnd_mode (rnd), op, compare, inexact);
                  printf ("\narg1="); mpfr_print_binary (arg1);
                  printf ("\narg2="); mpz_out_str(stdout, 2, arg2);
                  printf ("\ndstl="); mpfr_print_binary (dst_big);
                  printf ("\ndsts="); mpfr_print_binary (dst_small);
                  printf ("\ntmp ="); mpfr_dump (tmp);
                  exit (1);
                }
            }
        }
    }

  mpz_clear (arg2);
  mpfr_clears (arg1, dst_big, dst_small, tmp, (mpfr_ptr) 0);
}

static void
test_generic2z (mpfr_prec_t p0, mpfr_prec_t p1, unsigned int N,
               int (*func)(mpfr_ptr, mpz_srcptr, mpfr_srcptr, mpfr_rnd_t),
               const char *op)
{
  mpfr_prec_t prec;
  mpfr_t arg1, dst_big, dst_small, tmp;
  mpz_t  arg2;
  mpfr_rnd_t rnd;
  int inexact, compare, compare2;
  unsigned int n;

  mpfr_inits (arg1, dst_big, dst_small, tmp, (mpfr_ptr) 0);
  mpz_init (arg2);

  for (prec = p0; prec <= p1; prec++)
    {
      mpfr_set_prec (arg1, prec);
      mpfr_set_prec (tmp, prec);
      mpfr_set_prec (dst_small, prec);

      for (n=0; n<N; n++)
        {
          mpfr_urandomb (arg1, RANDS);
          mpz_urandomb (arg2, RANDS, 1024);
          rnd = RND_RAND ();
          mpfr_set_prec (dst_big, 2*prec);
          compare = func(dst_big, arg2, arg1, rnd);
          if (mpfr_can_round (dst_big, 2*prec, rnd, rnd, prec))
            {
              mpfr_set (tmp, dst_big, rnd);
              inexact = func(dst_small, arg2, arg1, rnd);
              if (mpfr_cmp (tmp, dst_small))
                {
                  printf ("Results differ for prec=%u rnd_mode=%s and %s_z:\n"
                          "arg1=",
                          (unsigned) prec, mpfr_print_rnd_mode (rnd), op);
                  mpfr_print_binary (arg1);
                  printf("\narg2=");
                  mpz_out_str (stdout, 10, arg2);
                  printf ("\ngot      ");
                  mpfr_dump (dst_small);
                  printf ("expected ");
                  mpfr_dump (tmp);
                  printf ("approx   ");
                  mpfr_dump (dst_big);
                  exit (1);
                }
              compare2 = mpfr_cmp (tmp, dst_big);
              /* if rounding to nearest, cannot know the sign of t - f(x)
                 because of composed rounding: y = o(f(x)) and t = o(y) */
              if (compare * compare2 >= 0)
                compare = compare + compare2;
              else
                compare = inexact; /* cannot determine sign(t-f(x)) */
              if (((inexact == 0) && (compare != 0)) ||
                  ((inexact > 0) && (compare <= 0)) ||
                  ((inexact < 0) && (compare >= 0)))
                {
                  printf ("Wrong inexact flag for rnd=%s and %s_z:\n"
                          "expected %d, got %d\n",
                          mpfr_print_rnd_mode (rnd), op, compare, inexact);
                  printf ("\narg1="); mpfr_print_binary (arg1);
                  printf ("\narg2="); mpz_out_str(stdout, 2, arg2);
                  printf ("\ndstl="); mpfr_print_binary (dst_big);
                  printf ("\ndsts="); mpfr_print_binary (dst_small);
                  printf ("\ntmp ="); mpfr_dump (tmp);
                  exit (1);
                }
            }
        }
    }

  mpz_clear (arg2);
  mpfr_clears (arg1, dst_big, dst_small, tmp, (mpfr_ptr) 0);
}

static void
test_genericq (mpfr_prec_t p0, mpfr_prec_t p1, unsigned int N,
               int (*func)(mpfr_ptr, mpfr_srcptr, mpq_srcptr, mpfr_rnd_t),
               const char *op)
{
  mpfr_prec_t prec;
  mpfr_t arg1, dst_big, dst_small, tmp;
  mpq_t  arg2;
  mpfr_rnd_t rnd;
  int inexact, compare, compare2;
  unsigned int n;

  mpfr_inits (arg1, dst_big, dst_small, tmp, (mpfr_ptr) 0);
  mpq_init (arg2);

  for (prec = p0; prec <= p1; prec++)
    {
      mpfr_set_prec (arg1, prec);
      mpfr_set_prec (tmp, prec);
      mpfr_set_prec (dst_small, prec);

      for (n=0; n<N; n++)
        {
          mpfr_urandomb (arg1, RANDS);
          mpq_set_ui (arg2, randlimb (), randlimb() );
          mpq_canonicalize (arg2);
          rnd = RND_RAND ();
          mpfr_set_prec (dst_big, prec+10);
          compare = func(dst_big, arg1, arg2, rnd);
          if (mpfr_can_round (dst_big, prec+10, rnd, rnd, prec))
            {
              mpfr_set (tmp, dst_big, rnd);
              inexact = func(dst_small, arg1, arg2, rnd);
              if (mpfr_cmp (tmp, dst_small))
                {
                  printf ("Results differ for prec=%u rnd_mode=%s and %s_q:\n"
                          "arg1=",
                          (unsigned) prec, mpfr_print_rnd_mode (rnd), op);
                  mpfr_print_binary (arg1);
                  printf("\narg2=");
                  mpq_out_str(stdout, 2, arg2);
                  printf ("\ngot      ");
                  mpfr_print_binary (dst_small);
                  printf ("\nexpected ");
                  mpfr_print_binary (tmp);
                  printf ("\napprox  ");
                  mpfr_print_binary (dst_big);
                  putchar('\n');
                  exit (1);
                }
              compare2 = mpfr_cmp (tmp, dst_big);
              /* if rounding to nearest, cannot know the sign of t - f(x)
                 because of composed rounding: y = o(f(x)) and t = o(y) */
              if (compare * compare2 >= 0)
                compare = compare + compare2;
              else
                compare = inexact; /* cannot determine sign(t-f(x)) */
              if (((inexact == 0) && (compare != 0)) ||
                  ((inexact > 0) && (compare <= 0)) ||
                  ((inexact < 0) && (compare >= 0)))
                {
                  printf ("Wrong inexact flag for rnd=%s and %s_q:\n"
                          "expected %d, got %d",
                          mpfr_print_rnd_mode (rnd), op, compare, inexact);
                  printf ("\narg1="); mpfr_print_binary (arg1);
                  printf ("\narg2="); mpq_out_str(stdout, 2, arg2);
                  printf ("\ndstl="); mpfr_print_binary (dst_big);
                  printf ("\ndsts="); mpfr_print_binary (dst_small);
                  printf ("\ntmp ="); mpfr_print_binary (tmp);
                  putchar('\n');
                  exit (1);
                }
            }
        }
    }

  mpq_clear (arg2);
  mpfr_clears (arg1, dst_big, dst_small, tmp, (mpfr_ptr) 0);
}

static void
test_specialq (mpfr_prec_t p0, mpfr_prec_t p1, unsigned int N,
               int (*mpfr_func)(mpfr_ptr, mpfr_srcptr, mpq_srcptr, mpfr_rnd_t),
               void (*mpq_func)(mpq_ptr, mpq_srcptr, mpq_srcptr),
               const char *op)
{
  mpfr_t fra, frb, frq;
  mpq_t  q1, q2, qr;
  unsigned int n;
  mpfr_prec_t prec;

  for (prec = p0 ; prec < p1 ; prec++)
    {
      mpfr_inits2 (prec, fra, frb, frq, (mpfr_ptr) 0);
      mpq_init (q1); mpq_init(q2); mpq_init (qr);

      for( n = 0 ; n < N ; n++)
        {
          mpq_set_ui(q1, randlimb(), randlimb() );
          mpq_set_ui(q2, randlimb(), randlimb() );
          mpq_canonicalize (q1);
          mpq_canonicalize (q2);
          mpq_func (qr, q1, q2);
          mpfr_set_q (fra, q1, MPFR_RNDD);
          mpfr_func (fra, fra, q2, MPFR_RNDD);
          mpfr_set_q (frb, q1, MPFR_RNDU);
          mpfr_func (frb, frb, q2, MPFR_RNDU);
          mpfr_set_q (frq, qr, MPFR_RNDN);
          /* We should have fra <= qr <= frb */
          if ( (mpfr_cmp(fra, frq) > 0) || (mpfr_cmp (frq, frb) > 0))
            {
              printf("Range error for prec=%lu and %s",
                     (unsigned long) prec, op);
              printf ("\nq1="); mpq_out_str(stdout, 2, q1);
              printf ("\nq2="); mpq_out_str(stdout, 2, q2);
              printf ("\nfr_dn="); mpfr_print_binary (fra);
              printf ("\nfr_q ="); mpfr_print_binary (frq);
              printf ("\nfr_up="); mpfr_print_binary (frb);
              putchar('\n');
              exit (1);
            }
        }

      mpq_clear (q1); mpq_clear (q2); mpq_clear (qr);
      mpfr_clears (fra, frb, frq, (mpfr_ptr) 0);
    }
}

static void
bug_mul_q_20100810 (void)
{
  mpfr_t x;
  mpfr_t y;
  mpq_t q;
  int inexact;

  mpfr_init (x);
  mpfr_init (y);
  mpq_init (q);

  /* mpfr_mul_q: the inexact value must be set in case of overflow */
  mpq_set_ui (q, 4096, 3);
  mpfr_set_inf (x, +1);
  mpfr_nextbelow (x);
  inexact = mpfr_mul_q (y, x, q, MPFR_RNDU);

  if (inexact <= 0)
    {
      printf ("Overflow error in mpfr_mul_q. ");
      printf ("Wrong inexact flag: got %d, should be positive.\n", inexact);

      exit (1);
    }
  if (!mpfr_inf_p (y))
    {
      printf ("Overflow error in mpfr_mul_q (y, x, q, MPFR_RNDD). ");
      printf ("\nx = ");
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDD);
      printf ("\nq = ");
      mpq_out_str (stdout, 10, q);
      printf ("\ny = ");
      mpfr_out_str (stdout, 10, 0, y, MPFR_RNDD);
      printf (" (should be +infinity)\n");

      exit (1);
    }

  mpq_clear (q);
  mpfr_clear (y);
  mpfr_clear (x);
}

static void
bug_div_q_20100810 (void)
{
  mpfr_t x;
  mpfr_t y;
  mpq_t q;
  int inexact;

  mpfr_init (x);
  mpfr_init (y);
  mpq_init (q);

  /* mpfr_div_q: the inexact value must be set in case of overflow */
  mpq_set_ui (q, 3, 4096);
  mpfr_set_inf (x, +1);
  mpfr_nextbelow (x);
  inexact = mpfr_div_q (y, x, q, MPFR_RNDU);

  if (inexact <= 0)
    {
      printf ("Overflow error in mpfr_div_q. ");
      printf ("Wrong inexact flag: got %d, should be positive.\n", inexact);

      exit (1);
    }
  if (!mpfr_inf_p (y))
    {
      printf ("Overflow error in mpfr_div_q (y, x, q, MPFR_RNDD). ");
      printf ("\nx = ");
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDD);
      printf ("\nq = ");
      mpq_out_str (stdout, 10, q);
      printf ("\ny = ");
      mpfr_out_str (stdout, 10, 0, y, MPFR_RNDD);
      printf (" (should be +infinity)\n");

      exit (1);
    }

  mpq_clear (q);
  mpfr_clear (y);
  mpfr_clear (x);
}

static void
bug_mul_div_q_20100818 (void)
{
  mpq_t qa, qb;
  mpfr_t x1, x2, y1, y2, y3;
  mpfr_exp_t emin, emax, e;
  int inex;
  int rnd;

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();
  set_emin (MPFR_EMIN_MIN);
  set_emax (MPFR_EMAX_MAX);

  mpq_init (qa);
  mpq_init (qb);
  mpfr_inits2 (32, x1, x2, y1, y2, y3, (mpfr_ptr) 0);

  mpq_set_ui (qa, 3, 17);
  mpq_set_ui (qb, 17, 3);
  inex = mpfr_set_ui (x1, 7, MPFR_RNDN);
  MPFR_ASSERTN (inex == 0);

  e = MPFR_EMAX_MAX - 3;
  inex = mpfr_set_ui_2exp (x2, 7, e, MPFR_RNDN);  /* x2 = x1 * 2^e */
  MPFR_ASSERTN (inex == 0);

  RND_LOOP(rnd)
    {
      mpfr_mul_q (y1, x1, qa, (mpfr_rnd_t) rnd);
      mpfr_div_q (y3, x1, qb, (mpfr_rnd_t) rnd);
      MPFR_ASSERTN (mpfr_equal_p (y1, y3));
      inex = mpfr_set_ui_2exp (y3, 1, e, MPFR_RNDN);
      MPFR_ASSERTN (inex == 0);
      inex = mpfr_mul (y3, y3, y1, MPFR_RNDN);  /* y3 = y1 * 2^e */
      MPFR_ASSERTN (inex == 0);
      mpfr_mul_q (y2, x2, qa, (mpfr_rnd_t) rnd);
      if (! mpfr_equal_p (y2, y3))
        {
          printf ("Error 1 in bug_mul_div_q_20100818 (rnd = %d)\n", rnd);
          printf ("Expected "); mpfr_dump (y3);
          printf ("Got      "); mpfr_dump (y2);
          exit (1);
        }
      mpfr_div_q (y2, x2, qb, (mpfr_rnd_t) rnd);
      if (! mpfr_equal_p (y2, y3))
        {
          printf ("Error 2 in bug_mul_div_q_20100818 (rnd = %d)\n", rnd);
          printf ("Expected "); mpfr_dump (y3);
          printf ("Got      "); mpfr_dump (y2);
          exit (1);
        }
    }

  e = MPFR_EMIN_MIN;
  inex = mpfr_set_ui_2exp (x2, 7, e, MPFR_RNDN);  /* x2 = x1 * 2^e */
  MPFR_ASSERTN (inex == 0);

  RND_LOOP(rnd)
    {
      mpfr_div_q (y1, x1, qa, (mpfr_rnd_t) rnd);
      mpfr_mul_q (y3, x1, qb, (mpfr_rnd_t) rnd);
      MPFR_ASSERTN (mpfr_equal_p (y1, y3));
      inex = mpfr_set_ui_2exp (y3, 1, e, MPFR_RNDN);
      MPFR_ASSERTN (inex == 0);
      inex = mpfr_mul (y3, y3, y1, MPFR_RNDN);  /* y3 = y1 * 2^e */
      MPFR_ASSERTN (inex == 0);
      mpfr_div_q (y2, x2, qa, (mpfr_rnd_t) rnd);
      if (! mpfr_equal_p (y2, y3))
        {
          printf ("Error 3 in bug_mul_div_q_20100818 (rnd = %d)\n", rnd);
          printf ("Expected "); mpfr_dump (y3);
          printf ("Got      "); mpfr_dump (y2);
          exit (1);
        }
      mpfr_mul_q (y2, x2, qb, (mpfr_rnd_t) rnd);
      if (! mpfr_equal_p (y2, y3))
        {
          printf ("Error 4 in bug_mul_div_q_20100818 (rnd = %d)\n", rnd);
          printf ("Expected "); mpfr_dump (y3);
          printf ("Got      "); mpfr_dump (y2);
          exit (1);
        }
    }

  mpq_clear (qa);
  mpq_clear (qb);
  mpfr_clears (x1, x2, y1, y2, y3, (mpfr_ptr) 0);

  set_emin (emin);
  set_emax (emax);
}

static void
reduced_expo_range (void)
{
  mpfr_t x;
  mpz_t z;
  mpq_t q;
  mpfr_exp_t emin;
  int inex;

  emin = mpfr_get_emin ();
  set_emin (4);

  mpfr_init2 (x, 32);

  mpz_init (z);
  mpfr_clear_flags ();
  inex = mpfr_set_ui (x, 17, MPFR_RNDN);
  MPFR_ASSERTN (inex == 0);
  mpz_set_ui (z, 3);
  inex = mpfr_mul_z (x, x, z, MPFR_RNDN);
  if (inex != 0 || MPFR_IS_NAN (x) || mpfr_cmp_ui (x, 51) != 0)
    {
      printf ("Error 1 in reduce_expo_range: expected 51 with inex = 0,"
              " got\n");
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
      printf ("with inex = %d\n", inex);
      exit (1);
    }
  inex = mpfr_div_z (x, x, z, MPFR_RNDN);
  if (inex != 0 || MPFR_IS_NAN (x) || mpfr_cmp_ui (x, 17) != 0)
    {
      printf ("Error 2 in reduce_expo_range: expected 17 with inex = 0,"
              " got\n");
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
      printf ("with inex = %d\n", inex);
      exit (1);
    }
  inex = mpfr_add_z (x, x, z, MPFR_RNDN);
  if (inex != 0 || MPFR_IS_NAN (x) || mpfr_cmp_ui (x, 20) != 0)
    {
      printf ("Error 3 in reduce_expo_range: expected 20 with inex = 0,"
              " got\n");
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
      printf ("with inex = %d\n", inex);
      exit (1);
    }
  inex = mpfr_sub_z (x, x, z, MPFR_RNDN);
  if (inex != 0 || MPFR_IS_NAN (x) || mpfr_cmp_ui (x, 17) != 0)
    {
      printf ("Error 4 in reduce_expo_range: expected 17 with inex = 0,"
              " got\n");
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
      printf ("with inex = %d\n", inex);
      exit (1);
    }
  MPFR_ASSERTN (__gmpfr_flags == 0);
  if (mpfr_cmp_z (x, z) <= 0)
    {
      printf ("Error 5 in reduce_expo_range: expected a positive value.\n");
      exit (1);
    }
  mpz_clear (z);

  mpq_init (q);
  mpq_set_ui (q, 1, 1);
  mpfr_set_ui (x, 16, MPFR_RNDN);
  inex = mpfr_add_q (x, x, q, MPFR_RNDN);
  if (inex != 0 || MPFR_IS_NAN (x) || mpfr_cmp_ui (x, 17) != 0)
    {
      printf ("Error in reduce_expo_range for 16 + 1/1,"
              " got inex = %d and\nx = ", inex);
      mpfr_dump (x);
      exit (1);
    }
  inex = mpfr_sub_q (x, x, q, MPFR_RNDN);
  if (inex != 0 || MPFR_IS_NAN (x) || mpfr_cmp_ui (x, 16) != 0)
    {
      printf ("Error in reduce_expo_range for 17 - 1/1,"
              " got inex = %d and\nx = ", inex);
      mpfr_dump (x);
      exit (1);
    }
  mpq_clear (q);

  mpfr_clear (x);

  set_emin (emin);
}

static void
addsubq_overflow_aux (mpfr_exp_t e)
{
  mpfr_t x, y;
  mpq_t q;
  mpfr_exp_t emax;
  int inex;
  int rnd;
  int sign, sub;

  MPFR_ASSERTN (e <= LONG_MAX);
  emax = mpfr_get_emax ();
  set_emax (e);
  mpfr_inits2 (16, x, y, (mpfr_ptr) 0);
  mpq_init (q);

  mpfr_set_inf (x, 1);
  mpfr_nextbelow (x);
  mpq_set_ui (q, 1, 1);

  for (sign = 0; sign <= 1; sign++)
    {
      for (sub = 0; sub <= 1; sub++)
        {
          RND_LOOP(rnd)
            {
              unsigned int flags, ex_flags;
              int inf;

              inf = rnd == MPFR_RNDA ||
                    rnd == (sign ? MPFR_RNDD : MPFR_RNDU);
              ex_flags = MPFR_FLAGS_INEXACT | (inf ? MPFR_FLAGS_OVERFLOW : 0);
              mpfr_clear_flags ();
              inex = sub ?
                mpfr_sub_q (y, x, q, (mpfr_rnd_t) rnd) :
                mpfr_add_q (y, x, q, (mpfr_rnd_t) rnd);
              flags = __gmpfr_flags;
              if (inex == 0 || flags != ex_flags ||
                  (inf ? ! mpfr_inf_p (y) : ! mpfr_equal_p (x, y)))
                {
                  printf ("Error in addsubq_overflow_aux(%ld),"
                          " sign = %d, %s\n", (long) e, sign,
                          mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                  printf ("Got inex = %d, y = ", inex);
                  mpfr_dump (y);
                  printf ("Expected flags:");
                  flags_out (ex_flags);
                  printf ("Got flags:     ");
                  flags_out (flags);
                  exit (1);
                }
            }
          mpq_neg (q, q);
        }
      mpfr_neg (x, x, MPFR_RNDN);
      mpq_neg (q, q);
    }

  mpq_clear (q);
  mpfr_clears (x, y, (mpfr_ptr) 0);
  set_emax (emax);
}

static void
addsubq_overflow (void)
{
  addsubq_overflow_aux (4913);
  addsubq_overflow_aux (MPFR_EMAX_MAX);
}

static void
coverage_mpfr_mul_q_20110218 (void)
{
  mpfr_t cmp, res, op1;
  mpq_t op2;
  int status;

  mpfr_init2 (cmp, MPFR_PREC_MIN);
  mpfr_init2 (res, MPFR_PREC_MIN);
  mpfr_init_set_si (op1, 1, MPFR_RNDN);

  mpq_init (op2);
  mpq_set_si (op2, 0, 0);
  mpz_set_si (mpq_denref (op2), 0);

  status = mpfr_mul_q (res, op1, op2, MPFR_RNDN);

  if ((status != 0) || (mpfr_cmp (cmp, res) != 0))
    {
      printf ("Results differ %d.\nres=", status);
      mpfr_print_binary (res);
      printf ("\ncmp=");
      mpfr_print_binary (cmp);
      putchar ('\n');
      exit (1);
    }

  mpfr_set_si (op1, 1, MPFR_RNDN);
  mpq_set_si (op2, -1, 0);

  status = mpfr_mul_q (res, op1, op2, MPFR_RNDN);

  mpfr_set_inf (cmp, -1);
  if ((status != 0) || (mpfr_cmp(res, cmp) != 0))
    {
      printf ("mpfr_mul_q 1 * (-1/0) returned a wrong value :\n waiting for ");
      mpfr_print_binary (cmp);
      printf (" got ");
      mpfr_print_binary (res);
      printf ("\n trinary value is %d\n", status);
      exit (1);
    }

  mpq_clear (op2);
  mpfr_clear (op1);
  mpfr_clear (res);
  mpfr_clear (cmp);
}

int
main (int argc, char *argv[])
{
  tests_start_mpfr ();

  special ();

  test_specialz (mpfr_add_z, mpz_add, "add");
  test_specialz (mpfr_sub_z, mpz_sub, "sub");
  test_specialz (mpfr_mul_z, mpz_mul, "mul");
  test_genericz (2, 100, 100, mpfr_add_z, "add");
  test_genericz (2, 100, 100, mpfr_sub_z, "sub");
  test_genericz (2, 100, 100, mpfr_mul_z, "mul");
  test_genericz (2, 100, 100, mpfr_div_z, "div");
  test_special2z (mpfr_z_sub, mpz_sub, "sub");
  test_generic2z (2, 100, 100, mpfr_z_sub, "sub");

  test_genericq (2, 100, 100, mpfr_add_q, "add");
  test_genericq (2, 100, 100, mpfr_sub_q, "sub");
  test_genericq (2, 100, 100, mpfr_mul_q, "mul");
  test_genericq (2, 100, 100, mpfr_div_q, "div");
  test_specialq (2, 100, 100, mpfr_mul_q, mpq_mul, "mul");
  test_specialq (2, 100, 100, mpfr_div_q, mpq_div, "div");
  test_specialq (2, 100, 100, mpfr_add_q, mpq_add, "add");
  test_specialq (2, 100, 100, mpfr_sub_q, mpq_sub, "sub");

  test_cmp_z (2, 100, 100);
  test_cmp_q (2, 100, 100);
  test_cmp_f (2, 100, 100);

  check_for_zero ();

  bug_mul_q_20100810 ();
  bug_div_q_20100810 ();
  bug_mul_div_q_20100818 ();
  reduced_expo_range ();
  addsubq_overflow ();

  coverage_mpfr_mul_q_20110218 ();

  tests_end_mpfr ();
  return 0;
}

