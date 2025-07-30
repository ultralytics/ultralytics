/* gen-trialdivtab.c

   Contributed to the GNU project by Torbjorn Granlund.

Copyright 2009, 2012, 2013 Free Software Foundation, Inc.

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

/*
  Generate tables for fast, division-free trial division for GMP.

  There is one main table, ptab.  It contains primes, multiplied together, and
  several types of pre-computed inverses.  It refers to tables of the type
  dtab, via the last two indices.  That table contains the individual primes in
  the range, except that the primes are not actually included in the table (see
  the P macro; it sneakingly excludes the primes themselves).  Instead, the
  dtab tables contains tuples for each prime (modular-inverse, limit) used for
  divisibility checks.

  This interface is not intended for division of very many primes, since then
  other algorithms apply.
*/

#include <stdlib.h>
#include <stdio.h>
#include "bootstrap.c"

int sumspills (mpz_t, mpz_t *, int);
void mpn_mod_1s_4p_cps (mpz_t [7], mpz_t);

int limb_bits;

mpz_t B;

int
main (int argc, char *argv[])
{
  unsigned long t, p;
  mpz_t ppp, acc, inv, gmp_numb_max, tmp, Bhalf;
  mpz_t pre[7];
  int i;
  int start_p, end_p, interval_start, interval_end, omitted_p;
  const char *endtok;
  int stop;
  int np, start_idx;

  if (argc < 2)
    {
      fprintf (stderr, "usage: %s bits endprime\n", argv[0]);
      exit (1);
    }

  limb_bits = atoi (argv[1]);

  end_p = 1290;			/* default end prime */
  if (argc == 3)
    end_p = atoi (argv[2]);

  printf ("#if GMP_LIMB_BITS != %d\n", limb_bits);
  printf ("#error This table is for GMP_LIMB_BITS = %d\n", limb_bits);
  printf ("#endif\n\n");

  printf ("#if GMP_NAIL_BITS != 0\n");
  printf ("#error This table does not support nails\n");
  printf ("#endif\n\n");

  for (i = 0; i < 7; i++)
    mpz_init (pre[i]);

  mpz_init_set_ui (gmp_numb_max, 1);
  mpz_mul_2exp (gmp_numb_max, gmp_numb_max, limb_bits);
  mpz_sub_ui (gmp_numb_max, gmp_numb_max, 1);

  mpz_init (tmp);
  mpz_init (inv);

  mpz_init_set_ui (B, 1);  mpz_mul_2exp (B, B, limb_bits);
  mpz_init_set_ui (Bhalf, 1);  mpz_mul_2exp (Bhalf, Bhalf, limb_bits - 1);

  start_p = 3;

  mpz_init_set_ui (ppp, 1);
  mpz_init (acc);
  interval_start = start_p;
  omitted_p = 3;
  interval_end = 0;

/*  printf ("static struct gmp_primes_dtab gmp_primes_dtab[] = {\n"); */

  printf ("#ifdef WANT_dtab\n");

  for (t = start_p; t <= end_p; t += 2)
    {
      if (! isprime (t))
	continue;

      mpz_mul_ui (acc, ppp, t);
      stop = mpz_cmp (acc, Bhalf) >= 0;
      if (!stop)
	{
	  mpn_mod_1s_4p_cps (pre, acc);
	  stop = sumspills (acc, pre + 2, 5);
	}

      if (stop)
	{
	  for (p = interval_start; p <= interval_end; p += 2)
	    {
	      if (! isprime (p))
		continue;

	      printf ("  P(%d,", (int) p);
	      mpz_invert_ui_2exp (inv, p, limb_bits);
	      printf ("CNST_LIMB(0x");  mpz_out_str (stdout, 16, inv);  printf ("),");

	      mpz_tdiv_q_ui (tmp, gmp_numb_max, p);
	      printf ("CNST_LIMB(0x");  mpz_out_str (stdout, 16, tmp);
	      printf (")),\n");
	    }
	  mpz_set_ui (ppp, t);
	  interval_start = t;
	  omitted_p = t;
	}
      else
	{
	  mpz_set (ppp, acc);
	}
      interval_end = t;
    }
  printf ("#define SMALLEST_OMITTED_PRIME %d\n", (int) omitted_p);
  printf ("#endif\n");

  printf ("#ifdef WANT_ptab\n");

/*  printf ("static struct gmp_primes_ptab gmp_primes_ptab[] = {\n"); */

  endtok = "";

  mpz_set_ui (ppp, 1);
  interval_start = start_p;
  interval_end = 0;
  np = 0;
  start_idx = 0;
  for (t = start_p; t <= end_p; t += 2)
    {
      if (! isprime (t))
	continue;

      mpz_mul_ui (acc, ppp, t);

      stop = mpz_cmp (acc, Bhalf) >= 0;
      if (!stop)
	{
	  mpn_mod_1s_4p_cps (pre, acc);
	  stop = sumspills (acc, pre + 2, 5);
	}

      if (stop)
	{
	  mpn_mod_1s_4p_cps (pre, ppp);
	  printf ("%s", endtok);
	  printf ("  {CNST_LIMB(0x");  mpz_out_str (stdout, 16, ppp);
	  printf ("),{CNST_LIMB(0x");  mpz_out_str (stdout, 16, pre[0]);
	  printf ("),%d", (int) PTR(pre[1])[0]);
	  for (i = 0; i < 5; i++)
	    {
	      printf (",");
	      printf ("CNST_LIMB(0x");  mpz_out_str (stdout, 16, pre[2 + i]);
	      printf (")");
	    }
	  printf ("},");
	  printf ("%d,", start_idx);
	  printf ("%d}", np - start_idx);

	  endtok = ",\n";
	  mpz_set_ui (ppp, t);
	  interval_start = t;
	  start_idx = np;
	}
      else
	{
	  mpz_set (ppp, acc);
	}
      interval_end = t;
      np++;
    }

  printf ("\n");
  printf ("#endif\n");

  return 0;
}

unsigned long
mpz_log2 (mpz_t x)
{
  return mpz_sgn (x) ? mpz_sizeinbase (x, 2) : 0;
}

void
mpn_mod_1s_4p_cps (mpz_t cps[7], mpz_t bparm)
{
  mpz_t b, bi;
  mpz_t B1modb, B2modb, B3modb, B4modb, B5modb;
  mpz_t t;
  int cnt;

  mpz_init_set (b, bparm);

  cnt = limb_bits - mpz_log2 (b);

  mpz_init (bi);
  mpz_init (t);
  mpz_init (B1modb);
  mpz_init (B2modb);
  mpz_init (B3modb);
  mpz_init (B4modb);
  mpz_init (B5modb);

  mpz_set_ui (t, 1);
  mpz_mul_2exp (t, t, limb_bits - cnt);
  mpz_sub (t, t, b);
  mpz_mul_2exp (t, t, limb_bits);
  mpz_tdiv_q (bi, t, b);		/* bi = B^2/b, except msb */

  mpz_set_ui (t, 1);
  mpz_mul_2exp (t, t, limb_bits);	/* t = B */
  mpz_tdiv_r (B1modb, t, b);

  mpz_mul_2exp (t, B1modb, limb_bits);
  mpz_tdiv_r (B2modb, t, b);

  mpz_mul_2exp (t, B2modb, limb_bits);
  mpz_tdiv_r (B3modb, t, b);

  mpz_mul_2exp (t, B3modb, limb_bits);
  mpz_tdiv_r (B4modb, t, b);

  mpz_mul_2exp (t, B4modb, limb_bits);
  mpz_tdiv_r (B5modb, t, b);

  mpz_set (cps[0], bi);
  mpz_set_ui (cps[1], cnt);
  mpz_tdiv_q_2exp (cps[2], B1modb, 0);
  mpz_tdiv_q_2exp (cps[3], B2modb, 0);
  mpz_tdiv_q_2exp (cps[4], B3modb, 0);
  mpz_tdiv_q_2exp (cps[5], B4modb, 0);
  mpz_tdiv_q_2exp (cps[6], B5modb, 0);

  mpz_clear (b);
  mpz_clear (bi);
  mpz_clear (t);
  mpz_clear (B1modb);
  mpz_clear (B2modb);
  mpz_clear (B3modb);
  mpz_clear (B4modb);
  mpz_clear (B5modb);
}

int
sumspills (mpz_t ppp, mpz_t *a, int n)
{
  mpz_t s;
  int i, ret;

  mpz_init_set (s, a[0]);

  for (i = 1; i < n; i++)
    {
      mpz_add (s, s, a[i]);
    }
  ret = mpz_cmp (s, B) >= 0;
  mpz_clear (s);

  return ret;
}
