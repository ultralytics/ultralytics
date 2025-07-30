/*
Copyright 2000 Free Software Foundation, Inc.

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
#include <unistd.h>
#include <signal.h>
#include <math.h>
#include "gmp.h"
#include "gmpstat.h"

#define RCSID(msg) \
static /**/const char *const rcsid[] = { (char *)rcsid, "\100(#)" msg }

RCSID("$Id$");

int g_debug = 0;

static mpz_t a;

static void
sh_status (int sig)
{
  printf ("sh_status: signal %d caught. dumping status.\n", sig);

  printf ("  a = ");
  mpz_out_str (stdout, 10, a);
  printf ("\n");
  fflush (stdout);

  if (SIGSEGV == sig)		/* remove SEGV handler */
    signal (SIGSEGV, SIG_DFL);
}

/* Input is a modulus (m).  We shall find multiplier (a) and adder (c)
   conforming to the rules found in the first comment block in file
   mpz/urandom.c.

   Then run a spectral test on the generator and discard any
   multipliers not passing.  */

/* TODO:

   . find a better algorithm than a+=8; bigger jumps perhaps?

*/

void
mpz_true_random (mpz_t s, unsigned long int nbits)
{
#if __FreeBSD__
  FILE *fs;
  char c[1];
  int i;

  mpz_set_ui (s, 0);
  for (i = 0; i < nbits; i += 8)
    {
      for (;;)
	{
	  int nread;
	  fs = fopen ("/dev/random", "r");
	  nread = fread (c, 1, 1, fs);
	  fclose (fs);
	  if (nread != 0)
	    break;
	  sleep (1);
	}
      mpz_mul_2exp (s, s, 8);
      mpz_add_ui (s, s, ((unsigned long int) c[0]) & 0xff);
      printf ("%d random bits\n", i + 8);
    }
  if (nbits % 8 != 0)
    mpz_mod_2exp (s, s, nbits);
#endif
}

int
main (int argc, char *argv[])
{
  const char usage[] = "usage: findlc [-dv] m2exp [low_merit [high_merit]]\n";
  int f;
  int v_lose, m_lose, v_best, m_best;
  int c;
  int debug = 1;
  int cnt_high_merit;
  mpz_t m;
  unsigned long int m2exp;
#define DIMS 6			/* dimensions run in spectral test */
  mpf_t v[DIMS-1];		/* spectral test result (there's no v
				   for 1st dimension */
  mpf_t f_merit, low_merit, high_merit;
  mpz_t acc, minus8;
  mpz_t min, max;
  mpz_t s;


  mpz_init (m);
  mpz_init (a);
  for (f = 0; f < DIMS-1; f++)
    mpf_init (v[f]);
  mpf_init (f_merit);
  mpf_init_set_d (low_merit, .1);
  mpf_init_set_d (high_merit, .1);

  while ((c = getopt (argc, argv, "a:di:hv")) != -1)
    switch (c)
      {
      case 'd':			/* debug */
	g_debug++;
	break;

      case 'v':			/* print version */
	puts (rcsid[1]);
	exit (0);

      case 'h':
      case '?':
      default:
	fputs (usage, stderr);
	exit (1);
      }

  argc -= optind;
  argv += optind;

  if (argc < 1)
    {
      fputs (usage, stderr);
      exit (1);
    }

  /* Install signal handler. */
  if (SIG_ERR == signal (SIGSEGV, sh_status))
    {
      perror ("signal (SIGSEGV)");
      exit (1);
    }
  if (SIG_ERR == signal (SIGHUP, sh_status))
    {
      perror ("signal (SIGHUP)");
      exit (1);
    }

  printf ("findlc: version: %s\n", rcsid[1]);
  m2exp = atol (argv[0]);
  mpz_init_set_ui (m, 1);
  mpz_mul_2exp (m, m, m2exp);
  printf ("m = 0x");
  mpz_out_str (stdout, 16, m);
  puts ("");

  if (argc > 1)			/* have low_merit */
    mpf_set_str (low_merit, argv[1], 0);
  if (argc > 2)			/* have high_merit */
    mpf_set_str (high_merit, argv[2], 0);

  if (debug)
    {
      fprintf (stderr, "low_merit = ");
      mpf_out_str (stderr, 10, 2, low_merit);
      fprintf (stderr, "; high_merit = ");
      mpf_out_str (stderr, 10, 2, high_merit);
      fputs ("\n", stderr);
    }

  mpz_init (minus8);
  mpz_set_si (minus8, -8L);
  mpz_init_set_ui (acc, 0);
  mpz_init (s);
  mpz_init_set_d (min, 0.01 * pow (2.0, (double) m2exp));
  mpz_init_set_d (max, 0.99 * pow (2.0, (double) m2exp));

  mpz_true_random (s, m2exp);	/* Start.  */
  mpz_setbit (s, 0);		/* Make it odd.  */

  v_best = m_best = 2*(DIMS-1);
  for (;;)
    {
      mpz_add (acc, acc, s);
      mpz_mod_2exp (acc, acc, m2exp);
#if later
      mpz_and_si (a, acc, -8L);
#else
      mpz_and (a, acc, minus8);
#endif
      mpz_add_ui (a, a, 5);
      if (mpz_cmp (a, min) <= 0 || mpz_cmp (a, max) >= 0)
	continue;

      spectral_test (v, DIMS, a, m);
      for (f = 0, v_lose = m_lose = 0, cnt_high_merit = DIMS-1;
	   f < DIMS-1; f++)
	{
	  merit (f_merit, f + 2, v[f], m);

	  if (mpf_cmp_ui (v[f], 1 << (30 / (f + 2) + (f == 2))) < 0)
	    v_lose++;

	  if (mpf_cmp (f_merit, low_merit) < 0)
	    m_lose++;

	  if (mpf_cmp (f_merit, high_merit) >= 0)
	    cnt_high_merit--;
	}

      if (0 == v_lose && 0 == m_lose)
	{
	  mpz_out_str (stdout, 10, a); puts (""); fflush (stdout);
	  if (0 == cnt_high_merit)
	    break;		/* leave loop */
	}
      if (v_lose < v_best)
	{
	  v_best = v_lose;
	  printf ("best (v_lose=%d; m_lose=%d): ", v_lose, m_lose);
	  mpz_out_str (stdout, 10, a); puts (""); fflush (stdout);
	}
      if (m_lose < m_best)
	{
	  m_best = m_lose;
	  printf ("best (v_lose=%d; m_lose=%d): ", v_lose, m_lose);
	  mpz_out_str (stdout, 10, a); puts (""); fflush (stdout);
	}
    }

  mpz_clear (m);
  mpz_clear (a);
  for (f = 0; f < DIMS-1; f++)
    mpf_clear (v[f]);
  mpf_clear (f_merit);
  mpf_clear (low_merit);
  mpf_clear (high_merit);

  printf ("done.\n");
  return 0;
}
