/* mpfr_print_binary -- print the internal binary representation of a
                     floating-point number

Copyright 1999-2017 Free Software Foundation, Inc.
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

#include "mpfr-impl.h"

void
mpfr_fprint_binary (FILE *stream, mpfr_srcptr x)
{
  if (MPFR_IS_NAN (x))
    {
      fprintf (stream, "@NaN@");
      return;
    }

  if (MPFR_SIGN (x) < 0)
    fprintf (stream, "-");

  if (MPFR_IS_INF (x))
    fprintf (stream, "@Inf@");
  else if (MPFR_IS_ZERO (x))
    fprintf (stream, "0");
  else
    {
      mp_limb_t *mx;
      mpfr_prec_t px;
      mp_size_t n;

      mx = MPFR_MANT (x);
      px = MPFR_PREC (x);

      fprintf (stream, "0.");
      for (n = (px - 1) / GMP_NUMB_BITS; ; n--)
        {
          mp_limb_t wd, t;

          MPFR_ASSERTN (n >= 0);
          wd = mx[n];
          for (t = MPFR_LIMB_HIGHBIT; t != 0; t >>= 1)
            {
              putc ((wd & t) == 0 ? '0' : '1', stream);
              if (--px == 0)
                {
                  mpfr_exp_t ex;

                  ex = MPFR_GET_EXP (x);
                  MPFR_ASSERTN (ex >= LONG_MIN && ex <= LONG_MAX);
                  fprintf (stream, "E%ld", (long) ex);
                  return;
                }
            }
        }
    }
}

void
mpfr_print_binary (mpfr_srcptr x)
{
  mpfr_fprint_binary (stdout, x);
}

void
mpfr_print_mant_binary(const char *str, const mp_limb_t *p, mpfr_prec_t r)
{
  int i;
  mpfr_prec_t count = 0;
  char c;
  mp_size_t n = MPFR_PREC2LIMBS (r);

  printf("%s ", str);
  for(n-- ; n>=0 ; n--)
    {
      for(i = GMP_NUMB_BITS-1 ; i >=0 ; i--)
        {
          c = (p[n] & (((mp_limb_t)1L)<<i)) ? '1' : '0';
          putchar(c);
          count++;
          if (count == r)
            putchar('[');
        }
      putchar('.');
    }
  putchar('\n');
}

void
mpfr_dump_mant (const mp_limb_t *p, mpfr_prec_t r, mpfr_prec_t precx,
                mpfr_prec_t error)
{
  int i;
  mpfr_prec_t count = 0;
  char c;
  mp_size_t n = MPFR_PREC2LIMBS (r);

  for(n-- ; n>=0 ; n--)
    {
      for(i = GMP_NUMB_BITS-1 ; i >=0 ; i--)
        {
          c = (p[n] & (((mp_limb_t)1L)<<i)) ? '1' : '0';
          putchar(c);
          count++;
          if (count == precx)
            putchar (',');
          if (count == error)
            putchar('[');
        }
      putchar('.');
    }
  putchar('\n');
}
