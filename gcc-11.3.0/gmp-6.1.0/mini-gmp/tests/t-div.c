/*

Copyright 2012, 2013 Free Software Foundation, Inc.

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

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include "testutils.h"

#define MAXBITS 400
#define COUNT 10000

typedef void div_qr_func (mpz_t, mpz_t, const mpz_t, const mpz_t);
typedef unsigned long div_qr_ui_func (mpz_t, mpz_t, const mpz_t, unsigned long);
typedef void div_func (mpz_t, const mpz_t, const mpz_t);
typedef unsigned long div_x_ui_func (mpz_t, const mpz_t, unsigned long);
typedef unsigned long div_ui_func (const mpz_t, unsigned long);

void
testmain (int argc, char **argv)
{
  unsigned i;
  mpz_t a, b, q, r, rq, rr;
  int div_p;

  mpz_init (a);
  mpz_init (b);
  mpz_init (r);
  mpz_init (q);
  mpz_init (rr);
  mpz_init (rq);

  for (i = 0; i < COUNT; i++)
    {
      unsigned j;
      for (j = 0; j < 3; j++)
	{
	  static const enum hex_random_op ops[3] = { OP_CDIV, OP_FDIV, OP_TDIV };
	  static const char name[3] = { 'c', 'f', 't'};
	  static div_qr_func * const div_qr [3] =
	    {
	      mpz_cdiv_qr, mpz_fdiv_qr, mpz_tdiv_qr
	    };
	  static div_qr_ui_func  *div_qr_ui[3] =
	    {
	      mpz_cdiv_qr_ui, mpz_fdiv_qr_ui, mpz_tdiv_qr_ui
	    };
	  static div_func * const div_q [3] =
	    {
	      mpz_cdiv_q, mpz_fdiv_q, mpz_tdiv_q
	    };
	  static div_x_ui_func  *div_q_ui[3] =
	    {
	      mpz_cdiv_q_ui, mpz_fdiv_q_ui, mpz_tdiv_q_ui
	    };
	  static div_func * const div_r [3] =
	    {
	      mpz_cdiv_r, mpz_fdiv_r, mpz_tdiv_r
	    };
	  static div_x_ui_func  *div_r_ui[3] =
	    {
	      mpz_cdiv_r_ui, mpz_fdiv_r_ui, mpz_tdiv_r_ui
	    };
	  static div_ui_func  *div_ui[3] =
	    {
	      mpz_cdiv_ui, mpz_fdiv_ui, mpz_tdiv_ui
	    };

	  mini_random_op4 (ops[j], MAXBITS, a, b, rq, rr);
	  div_qr[j] (q, r, a, b);
	  if (mpz_cmp (r, rr) || mpz_cmp (q, rq))
	    {
	      fprintf (stderr, "mpz_%cdiv_qr failed:\n", name[j]);
	      dump ("a", a);
	      dump ("b", b);
	      dump ("r   ", r);
	      dump ("rref", rr);
	      dump ("q   ", q);
	      dump ("qref", rq);
	      abort ();
	    }
	  mpz_set_si (q, -5);
	  div_q[j] (q, a, b);
	  if (mpz_cmp (q, rq))
	    {
	      fprintf (stderr, "mpz_%cdiv_q failed:\n", name[j]);
	      dump ("a", a);
	      dump ("b", b);
	      dump ("q   ", q);
	      dump ("qref", rq);
	      abort ();
	    }
	  mpz_set_ui (r, ~5);
	  div_r[j] (r, a, b);
	  if (mpz_cmp (r, rr))
	    {
	      fprintf (stderr, "mpz_%cdiv_r failed:\n", name[j]);
	      dump ("a", a);
	      dump ("b", b);
	      dump ("r   ", r);
	      dump ("rref", rr);
	      abort ();
	    }

	  if (j == 0)		/* do this once, not for all roundings */
	    {
	      div_p = mpz_divisible_p (a, b);
	      if ((mpz_sgn (r) == 0) ^ (div_p != 0))
		{
		  fprintf (stderr, "mpz_divisible_p failed:\n");
		  dump ("a", a);
		  dump ("b", b);
		  dump ("r   ", r);
		  abort ();
		}
	    }

	  if (j == 0 && mpz_sgn (b) < 0)  /* ceil, negative divisor */
	    {
	      mpz_mod (r, a, b);
	      if (mpz_cmp (r, rr))
		{
		  fprintf (stderr, "mpz_mod failed:\n");
		  dump ("a", a);
		  dump ("b", b);
		  dump ("r   ", r);
		  dump ("rref", rr);
		  abort ();
		}
	    }

	  if (j == 1 && mpz_sgn (b) > 0) /* floor, positive divisor */
	    {
	      mpz_mod (r, a, b);
	      if (mpz_cmp (r, rr))
		{
		  fprintf (stderr, "mpz_mod failed:\n");
		  dump ("a", a);
		  dump ("b", b);
		  dump ("r   ", r);
		  dump ("rref", rr);
		  abort ();
		}
	    }

	  if (mpz_fits_ulong_p (b))
	    {
	      mp_limb_t rl;

	      rl = div_qr_ui[j] (q, r, a, mpz_get_ui (b));
	      if (rl != mpz_get_ui (rr)
		  || mpz_cmp (r, rr) || mpz_cmp (q, rq))
		{
		  fprintf (stderr, "mpz_%cdiv_qr_ui failed:\n", name[j]);
		  dump ("a", a);
		  dump ("b", b);
		  fprintf(stderr, "rl   = %lx\n", rl);
		  dump ("r   ", r);
		  dump ("rref", rr);
		  dump ("q   ", q);
		  dump ("qref", rq);
		  abort ();
		}

	      mpz_set_si (q, 3);
	      rl = div_q_ui[j] (q, a, mpz_get_ui (b));
	      if (rl != mpz_get_ui (rr) || mpz_cmp (q, rq))
		{
		  fprintf (stderr, "mpz_%cdiv_q_ui failed:\n", name[j]);
		  dump ("a", a);
		  dump ("b", b);
		  fprintf(stderr, "rl   = %lx\n", rl);
		  dump ("rref", rr);
		  dump ("q   ", q);
		  dump ("qref", rq);
		  abort ();
		}

	      mpz_set_ui (r, 7);
	      rl = div_r_ui[j] (r, a, mpz_get_ui (b));
	      if (rl != mpz_get_ui (rr) || mpz_cmp (r, rr))
		{
		  fprintf (stderr, "mpz_%cdiv_qr_ui failed:\n", name[j]);
		  dump ("a", a);
		  dump ("b", b);
		  fprintf(stderr, "rl   = %lx\n", rl);
		  dump ("r   ", r);
		  dump ("rref", rr);
		  abort ();
		}

	      rl = div_ui[j] (a, mpz_get_ui (b));
	      if (rl != mpz_get_ui (rr))
		{
		  fprintf (stderr, "mpz_%cdiv_qr_ui failed:\n", name[j]);
		  dump ("a", a);
		  dump ("b", b);
		  fprintf(stderr, "rl   = %lx\n", rl);
		  dump ("rref", rr);
		  abort ();
		}

	      if (j == 0)	/* do this once, not for all roundings */
		{
		  div_p = mpz_divisible_ui_p (a, mpz_get_ui (b));
		  if ((mpz_sgn (r) == 0) ^ (div_p != 0))
		    {
		      fprintf (stderr, "mpz_divisible_ui_p failed:\n");
		      dump ("a", a);
		      dump ("b", b);
		      dump ("r   ", r);
		      abort ();
		    }
		}

	      if (j == 1)	/* floor */
		{
		  mpz_mod_ui (r, a, mpz_get_ui (b));
		  if (mpz_cmp (r, rr))
		    {
		      fprintf (stderr, "mpz_mod failed:\n");
		      dump ("a", a);
		      dump ("b", b);
		      dump ("r   ", r);
		      dump ("rref", rr);
		      abort ();
		    }
		}
	    }
	}
    }
  mpz_clear (a);
  mpz_clear (b);
  mpz_clear (r);
  mpz_clear (q);
  mpz_clear (rr);
  mpz_clear (rq);
}
