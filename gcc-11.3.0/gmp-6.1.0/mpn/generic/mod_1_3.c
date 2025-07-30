/* mpn_mod_1s_3p (ap, n, b, cps)
   Divide (ap,,n) by b.  Return the single-limb remainder.
   Requires that d < B / 3.

   Contributed to the GNU project by Torbjorn Granlund.
   Based on a suggestion by Peter L. Montgomery.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY WILL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2008-2010, 2013 Free Software Foundation, Inc.

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
#include "longlong.h"

void
mpn_mod_1s_3p_cps (mp_limb_t cps[6], mp_limb_t b)
{
  mp_limb_t bi;
  mp_limb_t B1modb, B2modb, B3modb, B4modb;
  int cnt;

  ASSERT (b <= (~(mp_limb_t) 0) / 3);

  count_leading_zeros (cnt, b);

  b <<= cnt;
  invert_limb (bi, b);

  cps[0] = bi;
  cps[1] = cnt;

  B1modb = -b * ((bi >> (GMP_LIMB_BITS-cnt)) | (CNST_LIMB(1) << cnt));
  ASSERT (B1modb <= b);		/* NB: not fully reduced mod b */
  cps[2] = B1modb >> cnt;

  udiv_rnnd_preinv (B2modb, B1modb, CNST_LIMB(0), b, bi);
  cps[3] = B2modb >> cnt;

  udiv_rnnd_preinv (B3modb, B2modb, CNST_LIMB(0), b, bi);
  cps[4] = B3modb >> cnt;

  udiv_rnnd_preinv (B4modb, B3modb, CNST_LIMB(0), b, bi);
  cps[5] = B4modb >> cnt;

#if WANT_ASSERT
  {
    int i;
    b = cps[2];
    for (i = 3; i <= 5; i++)
      {
	b += cps[i];
	ASSERT (b >= cps[i]);
      }
  }
#endif
}

mp_limb_t
mpn_mod_1s_3p (mp_srcptr ap, mp_size_t n, mp_limb_t b, const mp_limb_t cps[6])
{
  mp_limb_t rh, rl, bi, ph, pl, ch, cl, r;
  mp_limb_t B1modb, B2modb, B3modb, B4modb;
  mp_size_t i;
  int cnt;

  ASSERT (n >= 1);

  B1modb = cps[2];
  B2modb = cps[3];
  B3modb = cps[4];
  B4modb = cps[5];

  /* We compute n mod 3 in a tricky way, which works except for when n is so
     close to the maximum size that we don't need to support it.  The final
     cast to int is a workaround for HP cc.  */
  switch ((int) ((mp_limb_t) n * MODLIMB_INVERSE_3 >> (GMP_NUMB_BITS - 2)))
    {
    case 0:
      umul_ppmm (ph, pl, ap[n - 2], B1modb);
      add_ssaaaa (ph, pl, ph, pl, CNST_LIMB(0), ap[n - 3]);
      umul_ppmm (rh, rl, ap[n - 1], B2modb);
      add_ssaaaa (rh, rl, rh, rl, ph, pl);
      n -= 3;
      break;
    case 2:	/* n mod 3 = 1 */
      rh = 0;
      rl = ap[n - 1];
      n -= 1;
      break;
    case 1:	/* n mod 3 = 2 */
      rh = ap[n - 1];
      rl = ap[n - 2];
      n -= 2;
      break;
    }

  for (i = n - 3; i >= 0; i -= 3)
    {
      /* rr = ap[i]				< B
	    + ap[i+1] * (B mod b)		<= (B-1)(b-1)
	    + ap[i+2] * (B^2 mod b)		<= (B-1)(b-1)
	    + LO(rr)  * (B^3 mod b)		<= (B-1)(b-1)
	    + HI(rr)  * (B^4 mod b)		<= (B-1)(b-1)
      */
      umul_ppmm (ph, pl, ap[i + 1], B1modb);
      add_ssaaaa (ph, pl, ph, pl, CNST_LIMB(0), ap[i + 0]);

      umul_ppmm (ch, cl, ap[i + 2], B2modb);
      add_ssaaaa (ph, pl, ph, pl, ch, cl);

      umul_ppmm (ch, cl, rl, B3modb);
      add_ssaaaa (ph, pl, ph, pl, ch, cl);

      umul_ppmm (rh, rl, rh, B4modb);
      add_ssaaaa (rh, rl, rh, rl, ph, pl);
    }

  umul_ppmm (rh, cl, rh, B1modb);
  add_ssaaaa (rh, rl, rh, rl, CNST_LIMB(0), cl);

  cnt = cps[1];
  bi = cps[0];

  r = (rh << cnt) | (rl >> (GMP_LIMB_BITS - cnt));
  udiv_rnnd_preinv (r, r, rl << cnt, b, bi);

  return r >> cnt;
}
