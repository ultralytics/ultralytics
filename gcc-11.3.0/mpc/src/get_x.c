/* mpc_get_dc, mpc_get_ldc -- Transform mpc number into C complex number
   mpc_get_str -- Convert a complex number into a string.

Copyright (C) 2009, 2010, 2011 INRIA

This file is part of GNU MPC.

GNU MPC is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

GNU MPC is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see http://www.gnu.org/licenses/ .
*/

#include "config.h"

#ifdef HAVE_COMPLEX_H
#include <complex.h>
#endif

#ifdef HAVE_LOCALE_H
#include <locale.h>
#endif

#include <stdio.h> /* for sprintf, fprintf */
#include <ctype.h>
#include <string.h>
#include "mpc-impl.h"

#ifdef HAVE_COMPLEX_H
double _Complex
mpc_get_dc (mpc_srcptr op, mpc_rnd_t rnd) {
   return I * mpfr_get_d (mpc_imagref (op), MPC_RND_IM (rnd))
          + mpfr_get_d (mpc_realref (op), MPC_RND_RE (rnd));
}

long double _Complex
mpc_get_ldc (mpc_srcptr op, mpc_rnd_t rnd) {
   return I * mpfr_get_ld (mpc_imagref (op), MPC_RND_IM (rnd))
          + mpfr_get_ld (mpc_realref (op), MPC_RND_RE (rnd));
}
#endif


/* Code for mpc_get_str. The output format is "(real imag)", the decimal point
   of the locale is used. */

/* mpfr_prec_t can be either int or long int */
#if (__GMP_MP_SIZE_T_INT == 1)
#define MPC_EXP_FORMAT_SPEC "i"
#elif (__GMP_MP_SIZE_T_INT == 0)
#define MPC_EXP_FORMAT_SPEC "li"
#else
#error "mpfr_exp_t size not supported"
#endif

static char *
pretty_zero (mpfr_srcptr zero)
{
  char *pretty;

  pretty = mpc_alloc_str (3);

  pretty[0] = mpfr_signbit (zero) ? '-' : '+';
  pretty[1] = '0';
  pretty[2] = '\0';

  return pretty;
}

static char *
prettify (const char *str, const mp_exp_t expo, int base, int special)
{
  size_t sz;
  char *pretty;
  char *p;
  const char *s;
  mp_exp_t x;
  int sign;

  sz = strlen (str) + 1; /* + terminal '\0' */

  if (special)
    {
      /* special number: nan or inf */
      pretty = mpc_alloc_str (sz);
      strcpy (pretty, str);

      return pretty;
    }

  /* regular number */

  sign = (str[0] == '-' || str[0] == '+');

  x = expo - 1; /* expo is the exponent value with decimal point BEFORE
                   the first digit, we wants decimal point AFTER the first
                   digit */
  if (base == 16)
    x <<= 2; /* the output exponent is a binary exponent */

  ++sz; /* + decimal point */

  if (x != 0)
    {
      /* augment sz with the size needed for an exponent written in base
         ten */
      mp_exp_t xx;

      sz += 3; /* + exponent char + sign + 1 digit */

      if (x < 0)
        {
          /* avoid overflow when changing sign (assuming that, for the
             mp_exp_t type, (max value) is greater than (- min value / 10)) */
          if (x < -10)
            {
              xx = - (x / 10);
              sz++;
            }
          else
            xx = -x;
        }
      else
        xx = x;

      /* compute sz += floor(log(expo)/log(10)) without using libm
         functions */
      while (xx > 9)
        {
          sz++;
          xx /= 10;
        }
    }

  pretty = mpc_alloc_str (sz);
  p = pretty;

  /* 1. optional sign plus first digit */
  s = str;
  *p++ = *s++;
  if (sign)
    *p++ = *s++;

  /* 2. decimal point */
#ifdef HAVE_LOCALECONV
  *p++ = *localeconv ()->decimal_point;
#else
  *p++ = '.';
#endif
  *p = '\0';

  /* 3. other significant digits */
  strcat (pretty, s);

  /* 4. exponent (in base ten) */
  if (x == 0)
    return pretty;

  p = pretty + strlen (str) + 1;

  switch (base)
    {
    case 10:
      *p++ = 'e';
      break;
    case 2:
    case 16:
      *p++ = 'p';
      break;
    default:
      *p++ = '@';
    }

  *p = '\0';

  sprintf (p, "%+"MPC_EXP_FORMAT_SPEC, x);

  return pretty;
}

static char *
get_pretty_str (const int base, const size_t n, mpfr_srcptr x, mpfr_rnd_t rnd)
{
  mp_exp_t expo;
  char *ugly;
  char *pretty;

  if (mpfr_zero_p (x))
    return pretty_zero (x);

  ugly = mpfr_get_str (NULL, &expo, base, n, x, rnd);
  MPC_ASSERT (ugly != NULL);
  pretty = prettify (ugly, expo, base, !mpfr_number_p (x));
  mpfr_free_str (ugly);

  return pretty;
}

char *
mpc_get_str (int base, size_t n, mpc_srcptr op, mpc_rnd_t rnd)
{
  size_t needed_size;
  char *real_str;
  char *imag_str;
  char *complex_str = NULL;

  if (base < 2 || base > 36)
    return NULL;

  real_str = get_pretty_str (base, n, mpc_realref (op), MPC_RND_RE (rnd));
  imag_str = get_pretty_str (base, n, mpc_imagref (op), MPC_RND_IM (rnd));

  needed_size = strlen (real_str) + strlen (imag_str) + 4;

  complex_str = mpc_alloc_str (needed_size);
MPC_ASSERT (complex_str != NULL);

  strcpy (complex_str, "(");
  strcat (complex_str, real_str);
  strcat (complex_str, " ");
  strcat (complex_str, imag_str);
  strcat (complex_str, ")");

  mpc_free_str (real_str);
  mpc_free_str (imag_str);

  return complex_str;
}
