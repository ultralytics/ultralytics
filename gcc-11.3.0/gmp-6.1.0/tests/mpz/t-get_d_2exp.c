/* Test mpz_get_d_2exp.

Copyright 2002, 2003, 2012 Free Software Foundation, Inc.

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


static void
check_zero (void)
{
  mpz_t   z;
  double  got, want;
  long    got_exp, want_exp;

  mpz_init_set_ui (z, 0);

  want = 0.0;
  want_exp = 0;
  got = mpz_get_d_2exp (&got_exp, z);
  if (got != want || got_exp != want_exp)
    {
      printf    ("mpz_get_d_2exp wrong on zero\n");
      mpz_trace ("   z    ", z);
      d_trace   ("   want ", want);
      d_trace   ("   got  ", got);
      printf    ("   want exp %ld\n", want_exp);
      printf    ("   got exp  %ld\n", got_exp);
      abort();
    }

  mpz_clear (z);
}

static void
check_onebit (void)
{
  static const unsigned long data[] = {
    1, 32, 52, 53, 54, 63, 64, 65, 128, 256, 511, 512, 513
  };
  mpz_t   z;
  double  got, want;
  long    got_exp, want_exp;
  int     i;

  mpz_init (z);

  for (i = 0; i < numberof (data); i++)
    {
      mpz_set_ui (z, 1L);
      mpz_mul_2exp (z, z, data[i]);
      want = 0.5;
      want_exp = data[i] + 1;
      got = mpz_get_d_2exp (&got_exp, z);
      if (got != want || got_exp != want_exp)
        {
          printf    ("mpz_get_d_2exp wrong on 2**%ld\n", data[i]);
          mpz_trace ("   z    ", z);
          d_trace   ("   want ", want);
          d_trace   ("   got  ", got);
          printf    ("   want exp %ld\n", want_exp);
          printf    ("   got exp  %ld\n", got_exp);
          abort();
        }

      mpz_set_si (z, -1L);
      mpz_mul_2exp (z, z, data[i]);
      want = -0.5;
      want_exp = data[i] + 1;
      got = mpz_get_d_2exp (&got_exp, z);
      if (got != want || got_exp != want_exp)
        {
          printf    ("mpz_get_d_2exp wrong on -2**%ld\n", data[i]);
          mpz_trace ("   z    ", z);
          d_trace   ("   want ", want);
          d_trace   ("   got  ", got);
          printf    ("   want exp %ld\n", want_exp);
          printf    ("   got exp  %ld\n", got_exp);
          abort();
        }
    }
  mpz_clear (z);
}

/* Check that hardware rounding doesn't make mpz_get_d_2exp return a value
   outside its defined range. */
static void
check_round (void)
{
  static const unsigned long data[] = { 1, 32, 53, 54, 64, 128, 256, 512 };
  mpz_t   z;
  double  got;
  long    got_exp;
  int     i, rnd_mode, old_rnd_mode;

  mpz_init (z);
  old_rnd_mode = tests_hardware_getround ();

  for (rnd_mode = 0; rnd_mode < 4; rnd_mode++)
    {
      tests_hardware_setround (rnd_mode);

      for (i = 0; i < numberof (data); i++)
        {
          mpz_set_ui (z, 1L);
          mpz_mul_2exp (z, z, data[i]);
          mpz_sub_ui (z, z, 1L);

          got = mpz_get_d_2exp (&got_exp, z);
          if (got < 0.5 || got >= 1.0)
            {
              printf    ("mpz_get_d_2exp wrong on 2**%lu-1\n", data[i]);
              printf    ("result out of range, expect 0.5 <= got < 1.0\n");
              printf    ("   rnd_mode = %d\n", rnd_mode);
              printf    ("   data[i]  = %lu\n", data[i]);
              mpz_trace ("   z    ", z);
              d_trace   ("   got  ", got);
              printf    ("   got exp  %ld\n", got_exp);
              abort();
            }

          mpz_neg (z, z);
          got = mpz_get_d_2exp (&got_exp, z);
          if (got <= -1.0 || got > -0.5)
            {
              printf    ("mpz_get_d_2exp wrong on -2**%lu-1\n", data[i]);
              printf    ("result out of range, expect -1.0 < got <= -0.5\n");
              printf    ("   rnd_mode = %d\n", rnd_mode);
              printf    ("   data[i]  = %lu\n", data[i]);
              mpz_trace ("   z    ", z);
              d_trace   ("   got  ", got);
              printf    ("   got exp  %ld\n", got_exp);
              abort();
            }
        }
    }

  mpz_clear (z);
  tests_hardware_setround (old_rnd_mode);
}

static void
check_rand (void)
{
  gmp_randstate_ptr rands = RANDS;
  int     i;
  mpz_t   z;
  double  got;
  long    got_exp;
  unsigned long  bits;

  mpz_init (z);

  for (i = 0; i < 200; i++)
    {
      bits = gmp_urandomm_ui (rands, 512L);
      mpz_urandomb (z, rands, bits);

      got = mpz_get_d_2exp (&got_exp, z);
      if (mpz_sgn (z) == 0)
        continue;
      bits = mpz_sizeinbase (z, 2);

      if (got < 0.5 || got >= 1.0)
        {
          printf    ("mpz_get_d_2exp out of range, expect 0.5 <= got < 1.0\n");
          mpz_trace ("   z    ", z);
          d_trace   ("   got  ", got);
          printf    ("   got exp  %ld\n", got_exp);
          abort();
        }

      /* FIXME: If mpz_get_d_2exp rounds upwards we might have got_exp ==
         bits+1, so leave this test disabled until we decide if that's what
         should happen, or not.  */
#if 0
      if (got_exp != bits)
        {
          printf    ("mpz_get_d_2exp wrong exponent\n", i);
          mpz_trace ("   z    ", z);
          d_trace   ("   bits ", bits);
          d_trace   ("   got  ", got);
          printf    ("   got exp  %ld\n", got_exp);
          abort();
        }
#endif
    }
  mpz_clear (z);
}


int
main (void)
{
  tests_start ();
  mp_trace_base = -16;

  check_zero ();
  check_onebit ();
  check_round ();
  check_rand ();

  tests_end ();
  exit (0);
}
