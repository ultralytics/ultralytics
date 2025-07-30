/* Test file for mpfr_prec_round.

Copyright 1999-2004, 2006-2017 Free Software Foundation, Inc.
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

int
main (void)
{
   mpfr_t x;
   mpfr_exp_t emax;

   tests_start_mpfr ();

   mpfr_init (x);

   mpfr_set_nan (x);
   mpfr_prec_round (x, 2, MPFR_RNDN);
   MPFR_ASSERTN(mpfr_nan_p (x));

   mpfr_set_inf (x, 1);
   mpfr_prec_round (x, 2, MPFR_RNDN);
   MPFR_ASSERTN(mpfr_inf_p (x) && mpfr_sgn (x) > 0);

   mpfr_set_inf (x, -1);
   mpfr_prec_round (x, 2, MPFR_RNDN);
   MPFR_ASSERTN(mpfr_inf_p (x) && mpfr_sgn (x) < 0);

   mpfr_set_ui (x, 0, MPFR_RNDN);
   mpfr_prec_round (x, 2, MPFR_RNDN);
   MPFR_ASSERTN(mpfr_cmp_ui (x, 0) == 0 && MPFR_IS_POS(x));

   mpfr_set_ui (x, 0, MPFR_RNDN);
   mpfr_neg (x, x, MPFR_RNDN);
   mpfr_prec_round (x, 2, MPFR_RNDN);
   MPFR_ASSERTN(mpfr_cmp_ui (x, 0) == 0 && MPFR_IS_NEG(x));

   emax = mpfr_get_emax ();
   set_emax (0);
   mpfr_set_prec (x, 3);
   mpfr_set_str_binary (x, "0.111");
   mpfr_prec_round (x, 2, MPFR_RNDN);
   MPFR_ASSERTN(mpfr_inf_p (x) && mpfr_sgn (x) > 0);
   set_emax (emax);

   mpfr_set_prec (x, mp_bits_per_limb + 2);
   mpfr_set_ui (x, 1, MPFR_RNDN);
   mpfr_nextbelow (x);
   mpfr_prec_round (x, mp_bits_per_limb + 1, MPFR_RNDN);
   MPFR_ASSERTN(mpfr_cmp_ui (x, 1) == 0);

   mpfr_set_prec (x, 3);
   mpfr_set_ui (x, 5, MPFR_RNDN);
   mpfr_prec_round (x, 2, MPFR_RNDN);
   if (mpfr_cmp_ui(x, 4))
     {
       printf ("Error in tround: got ");
       mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
       printf (" instead of 4\n");
       exit (1);
     }

   /* check case when reallocation is needed */
   mpfr_set_prec (x, 3);
   mpfr_set_ui (x, 5, MPFR_RNDN); /* exact */
   mpfr_prec_round (x, mp_bits_per_limb + 1, MPFR_RNDN);
   if (mpfr_cmp_ui(x, 5))
     {
       printf ("Error in tround: got ");
       mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
       printf (" instead of 5\n");
       exit (1);
     }

   mpfr_clear(x);
   mpfr_init2 (x, 3);
   mpfr_set_si (x, -5, MPFR_RNDN); /* exact */
   mpfr_prec_round (x, mp_bits_per_limb + 1, MPFR_RNDN);
   if (mpfr_cmp_si(x, -5))
     {
       printf ("Error in tround: got ");
       mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
       printf (" instead of -5\n");
       exit (1);
     }

   /* check case when new precision needs less limbs */
   mpfr_set_prec (x, mp_bits_per_limb + 1);
   mpfr_set_ui (x, 5, MPFR_RNDN); /* exact */
   mpfr_prec_round (x, 3, MPFR_RNDN); /* exact */
   if (mpfr_cmp_ui(x, 5))
     {
       printf ("Error in tround: got ");
       mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
       printf (" instead of 5\n");
       exit (1);
     }

   mpfr_clear(x);

   tests_end_mpfr ();
   return 0;
}
