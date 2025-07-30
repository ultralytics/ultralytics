/* mpfr_cache -- cache interface for multiple-precision constants in MPFR.

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

#include "mpfr-impl.h"

#if 0 /* this function is not used/documented/tested so far, it could be
         useful if some user wants to add a new constant to mpfr, and
         implement a cache mechanism for that constant */
void
mpfr_init_cache (mpfr_cache_t cache, int (*func)(mpfr_ptr, mpfr_rnd_t))
{
  MPFR_PREC (cache->x) = 0; /* Invalid prec to detect that the cache is not
                               valid. Maybe add a flag? */
  cache->func = func;
}
#endif

void
mpfr_clear_cache (mpfr_cache_t cache)
{
  if (MPFR_PREC (cache->x) != 0)
    mpfr_clear (cache->x);
  MPFR_PREC (cache->x) = 0;
}

int
mpfr_cache (mpfr_ptr dest, mpfr_cache_t cache, mpfr_rnd_t rnd)
{
  mpfr_prec_t prec = MPFR_PREC (dest);
  mpfr_prec_t pold = MPFR_PREC (cache->x);
  int inexact, sign;
  MPFR_SAVE_EXPO_DECL (expo);

  MPFR_SAVE_EXPO_MARK (expo);

  if (MPFR_UNLIKELY (prec > pold))
    {
      /* No previous result in the cache or the precision of the
         previous result is not sufficient. */

      if (MPFR_UNLIKELY (pold == 0))  /* No previous result. */
        mpfr_init2 (cache->x, prec);

      /* Update the cache. */
      pold = prec;
      /* no need to keep the previous value */
      mpfr_set_prec (cache->x, pold);
      cache->inexact = (*cache->func) (cache->x, MPFR_RNDN);
    }

  /* now pold >= prec is the precision of cache->x */

  /* First, check if the cache has the exact value (unlikely).
     Else the exact value is between (assuming x=cache->x > 0):
       x and x+ulp(x) if cache->inexact < 0,
       x-ulp(x) and x if cache->inexact > 0,
     and abs(x-exact) <= ulp(x)/2. */

  /* we assume all cached constants are positive */
  MPFR_ASSERTN (MPFR_IS_POS (cache->x)); /* TODO... */
  sign = MPFR_SIGN (cache->x);
  MPFR_SET_EXP (dest, MPFR_GET_EXP (cache->x));
  MPFR_SET_SIGN (dest, sign);

  /* round cache->x from precision pold down to precision prec */
  MPFR_RNDRAW_GEN (inexact, dest,
                   MPFR_MANT (cache->x), pold, rnd, sign,
                   if (MPFR_UNLIKELY (cache->inexact == 0))
                     {
                       if ((_sp[0] & _ulp) == 0)
                         {
                           inexact = -sign;
                           goto trunc_doit;
                         }
                       else
                         goto addoneulp;
                     }
                   else if (cache->inexact < 0)
                     goto addoneulp;
                   else /* cache->inexact > 0 */
                     {
                       inexact = -sign;
                       goto trunc_doit;
                     },
                   if (MPFR_UNLIKELY (++MPFR_EXP (dest) > __gmpfr_emax))
                     mpfr_overflow (dest, rnd, sign);
                  );

  if (MPFR_LIKELY (cache->inexact != 0))
    {
      switch (rnd)
        {
        case MPFR_RNDZ:
        case MPFR_RNDD:
          if (MPFR_UNLIKELY (inexact == 0))
            {
              inexact = cache->inexact;
              if (inexact > 0)
                {
                  mpfr_nextbelow (dest);
                  inexact = -inexact;
                }
            }
          break;
        case MPFR_RNDU:
        case MPFR_RNDA:
          if (MPFR_UNLIKELY (inexact == 0))
            {
              inexact = cache->inexact;
              if (inexact < 0)
                {
                  mpfr_nextabove (dest);
                  inexact = -inexact;
                }
            }
          break;
        default: /* MPFR_RNDN */
          if (MPFR_UNLIKELY(inexact == 0))
            inexact = cache->inexact;
          break;
        }
    }

  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (dest, inexact, rnd);
}
