/* Test compatibility mpf-mpfr.

Copyright 2003-2017 Free Software Foundation, Inc.
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

#include <stdlib.h>

#include "mpfr-test.h"

#ifdef MPFR
#include "mpf2mpfr.h"
#endif

int
main (void)
{
  unsigned long int prec;
  unsigned long int prec2;
  mpf_t x, y;
  mpz_t z;
  mpq_t q;
  double d;
  signed long int exp;
  long l;
  unsigned long u;
  char *s;
  int i;
  FILE *f;
  gmp_randstate_t state;

  /* Initialization Functions */
  prec = 53;
  mpf_set_default_prec (prec);
  prec2 = mpf_get_default_prec ();
  if (prec2 < prec)
    {
      printf ("Error in get_default_prec: %lu < %lu\n", prec2, prec);
      exit (1);
    }

  mpf_init (y);

  mpf_init2 (x, prec);
  prec2 = mpf_get_prec (x);
  if (prec2 < prec)
    {
      printf ("Error in get_prec: %lu < %lu\n", prec2, prec);
      mpf_clear (y);
      mpf_clear (x);
      exit (1);
    }

  mpf_set_prec (x, 2 * prec);
  prec2 = mpf_get_prec (x);
  if (prec2 < 2 * prec)
    {
      printf ("Error in set_prec: %lu < %lu\n", prec2, 2 * prec);
      mpf_clear (y);
      mpf_clear (x);
      exit (1);
    }

  mpf_set_prec_raw (x, prec);
  prec2 = mpf_get_prec (x);
  if (prec2 < prec)
    {
      printf ("Error in set_prec_raw: %lu < %lu\n", prec2, prec);
      mpf_clear (y);
      mpf_clear (x);
      exit (1);
    }

  /* Assignment Functions */

  mpf_set (y, x);
  mpf_set_ui (x, 1);
  mpf_set_si (x, -1);
  mpf_set_d (x, 1.0);

  mpz_init_set_ui (z, 17);
  mpf_set_z (x, z);
  mpz_clear (z);

  mpq_init (q);
  mpq_set_ui (q, 2, 3);
  mpf_set_q (x, q);
  mpq_clear (q);

  mpf_set_str (x, "3.1415e1", 10);
  mpf_swap (x, y);

  /* Combined Initialization and Assignment Functions */

  mpf_clear (x);
  mpf_init_set (x, y);
  mpf_clear (x);
  mpf_init_set_ui (x, 17);
  mpf_clear (x);
  mpf_init_set_si (x, -17);
  mpf_clear (x);
  mpf_init_set_d (x, 17.0);
  mpf_clear (x);
  mpf_init_set_str (x, "3.1415e1", 10);

  /* Conversion Functions */

  d = mpf_get_d (x);
  d = mpf_get_d_2exp (&exp, x);
  l = mpf_get_si (x);
  u = mpf_get_ui (x);
  s = mpf_get_str (NULL, &exp, 10, 10, x);
  /* MPF doen't have mpf_free_str */
  mpfr_free_str (s);

  /* Use d, l and u to avoid a warning with -Wunused-but-set-variable
     from GCC 4.6. The variables above were mainly used for prototype
     checking. */
  (void) d;  (void) l;  (void)  u;

  /* Arithmetic Functions */

  mpf_add (y, x, x);
  mpf_add_ui (y, x, 1);
  mpf_sub (y, x, x);
  mpf_ui_sub (y, 1, x);
  mpf_sub_ui (y, x, 1);
  mpf_mul (y, x, x);
  mpf_mul_ui (y, x, 17);
  mpf_div (y, x, x);
  mpf_ui_div (y, 17, x);
  mpf_div_ui (y, x, 17);
  mpf_sqrt (y, x);
  mpf_sqrt_ui (y, 17);
  mpf_pow_ui (y, x, 2);
  mpf_neg (y, x);
  mpf_abs (y, x);
  mpf_mul_2exp (y, x, 17);
  mpf_div_2exp (y, x, 17);

  /* Comparison Functions */

  i = mpf_cmp (y, x);
  i = mpf_cmp_d (y, 1.7);
  i = mpf_cmp_ui (y, 17);
  i = mpf_cmp_si (y, -17);
  i = mpf_eq (y, x, 17);
  mpf_reldiff (y, y, x);
  i = mpf_sgn (x);

  /* Input and Output Functions */

  f = fopen ("/dev/null", "w");
  if (f != NULL)
    {
      mpf_out_str (f, 10, 10, x);
      fclose (f);
    }

  mpf_set_prec (x, 15);
  mpf_set_prec (y, 15);
  /* We may use src_fopen instead of fopen, but it is defined
     in mpfr-test, and not in mpfr.h and gmp.h, and we want
     to test theses includes files. */
  f = fopen ("inp_str.data", "r");
  if (f != NULL)
    {
      i = mpf_inp_str (x, f, 10);
      if ((i == 0) || mpf_cmp_ui (x, 31415))
        {
          printf ("Error in reading 1st line from file inp_str.data\n");
          exit (1);
        }
      fclose (f);
    }

  /* Miscellaneous Functions */

  mpf_ceil (y, x);
  mpf_floor (y, x);
  mpf_trunc (y, x);

  i = mpf_integer_p (x);

  i = mpf_fits_ulong_p (x);
  i = mpf_fits_slong_p (x);
  i = mpf_fits_uint_p (x);
  i = mpf_fits_sint_p (x);
  i = mpf_fits_ushort_p (x);
  i = mpf_fits_sshort_p (x);

  gmp_randinit_lc_2exp_size (state, 128);
  mpf_urandomb (x, state, 10);
  gmp_randclear (state);

  /* Conversion to mpz */
  mpz_init (z);
  mpf_set_ui (x, 17);
  mpz_set_f (z, x);
  mpf_set_z (x, z);
  mpz_clear (z);
  if (mpf_cmp_ui (x, 17) != 0)
    {
      fprintf (stderr, "Error in conversion to/from mpz\n");
      fprintf (stderr, "expected 17, got %1.16e\n", mpf_get_d (x));
      exit (1);
    }

  /* non-regression tests for bugs fixed in revision 11565 */
  mpf_set_si (x, -1);
  MPFR_ASSERTN(mpf_fits_ulong_p (x) == 0);
  MPFR_ASSERTN(mpf_fits_slong_p (x) != 0);
  MPFR_ASSERTN(mpf_fits_uint_p (x) == 0);
  MPFR_ASSERTN(mpf_fits_sint_p (x) != 0);
  MPFR_ASSERTN(mpf_fits_ushort_p (x) == 0);
  MPFR_ASSERTN(mpf_fits_sshort_p (x) != 0);
  MPFR_ASSERTN(mpf_get_si (x) == -1);

  /* clear all variables */
  mpf_clear (y);
  mpf_clear (x);

  return 0;
}
