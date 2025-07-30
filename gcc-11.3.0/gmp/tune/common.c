/* Shared speed subroutines.

Copyright 1999-2006, 2008-2015 Free Software Foundation, Inc.

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

#define __GMP_NO_ATTRIBUTE_CONST_PURE

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h> /* for qsort */
#include <string.h>
#include <unistd.h>
#if 0
#include <sys/ioctl.h>
#endif

#include "gmp.h"
#include "gmp-impl.h"
#include "longlong.h"

#include "tests.h"
#include "speed.h"


int   speed_option_addrs = 0;
int   speed_option_verbose = 0;
int   speed_option_cycles_broken = 0;


/* Provide __clz_tab even if it's not required, for the benefit of new code
   being tested with many.pl. */
#ifndef COUNT_LEADING_ZEROS_NEED_CLZ_TAB
#define COUNT_LEADING_ZEROS_NEED_CLZ_TAB
#include "mp_clz_tab.c"
#undef COUNT_LEADING_ZEROS_NEED_CLZ_TAB
#endif


void
pentium_wbinvd(void)
{
#if 0
  {
    static int  fd = -2;

    if (fd == -2)
      {
	fd = open ("/dev/wbinvd", O_RDWR);
	if (fd == -1)
	  perror ("open /dev/wbinvd");
      }

    if (fd != -1)
      ioctl (fd, 0, 0);
  }
#endif

#if 0
#define WBINVDSIZE  1024*1024*2
  {
    static char  *p = NULL;
    int   i, sum;

    if (p == NULL)
      p = malloc (WBINVDSIZE);

#if 0
    for (i = 0; i < WBINVDSIZE; i++)
      p[i] = i & 0xFF;
#endif

    sum = 0;
    for (i = 0; i < WBINVDSIZE; i++)
      sum += p[i];

    mpn_cache_fill_dummy (sum);
  }
#endif
}


int
double_cmp_ptr (const double *p, const double *q)
{
  if (*p > *q)  return 1;
  if (*p < *q)  return -1;
  return 0;
}


/* Measure the speed of a given routine.

   The routine is run with enough repetitions to make it take at least
   speed_precision * speed_unittime.  This aims to minimize the effects of a
   limited accuracy time base and the overhead of the measuring itself.

   Measurements are made looking for 4 results within TOLERANCE of each
   other (or 3 for routines taking longer than 2 seconds).  This aims to get
   an accurate reading even if some runs are bloated by interrupts or task
   switches or whatever.

   The given (*fun)() is expected to run its function "s->reps" many times
   and return the total elapsed time measured using speed_starttime() and
   speed_endtime().  If the function doesn't support the given s->size or
   s->r, -1.0 should be returned.  See the various base routines below.  */

double
speed_measure (double (*fun) (struct speed_params *s), struct speed_params *s)
{
#define TOLERANCE    1.01  /* 1% */
  const int max_zeros = 10;

  struct speed_params  s_dummy;
  int     i, j, e;
  double  t[30];
  double  t_unsorted[30];
  double  reps_d;
  int     zeros = 0;

  /* Use dummy parameters if caller doesn't provide any.  Only a few special
     "fun"s will cope with this, speed_noop() is one.  */
  if (s == NULL)
    {
      memset (&s_dummy, '\0', sizeof (s_dummy));
      s = &s_dummy;
    }

  s->reps = 1;
  s->time_divisor = 1.0;
  for (i = 0; i < numberof (t); i++)
    {
      for (;;)
	{
	  s->src_num = 0;
	  s->dst_num = 0;

	  t[i] = (*fun) (s);

	  if (speed_option_verbose >= 3)
	    gmp_printf("size=%ld reps=%u r=%Md attempt=%d  %.9f\n",
		       (long) s->size, s->reps, s->r, i, t[i]);

	  if (t[i] == 0.0)
	    {
	      zeros++;
	      if (zeros > max_zeros)
		{
		  fprintf (stderr, "Fatal error: too many (%d) failed measurements (0.0)\n", zeros);
		  abort ();
		}
	     if (s->reps < 10000)
	       s->reps *= 2;

	      continue;
	    }

	  if (t[i] == -1.0)
	    return -1.0;

	  if (t[i] >= speed_unittime * speed_precision)
	    break;

	  /* go to a value of reps to make t[i] >= precision */
	  reps_d = ceil (1.1 * s->reps
			 * speed_unittime * speed_precision
			 / MAX (t[i], speed_unittime));
	  if (reps_d > 2e9 || reps_d < 1.0)
	    {
	      fprintf (stderr, "Fatal error: new reps bad: %.2f\n", reps_d);
	      fprintf (stderr, "  (old reps %u, unittime %.4g, precision %d, t[i] %.4g)\n",
		       s->reps, speed_unittime, speed_precision, t[i]);
	      abort ();
	    }
	  s->reps = (unsigned) reps_d;
	}
      t[i] /= s->reps;
      t_unsorted[i] = t[i];

      if (speed_precision == 0)
	return t[i];

      /* require 3 values within TOLERANCE when >= 2 secs, 4 when below */
      if (t[0] >= 2.0)
	e = 3;
      else
	e = 4;

      /* Look for e many t[]'s within TOLERANCE of each other to consider a
	 valid measurement.  Return smallest among them.  */
      if (i >= e)
	{
	  qsort (t, i+1, sizeof(t[0]), (qsort_function_t) double_cmp_ptr);
	  for (j = e-1; j < i; j++)
	    if (t[j] <= t[j-e+1] * TOLERANCE)
	      return t[j-e+1] / s->time_divisor;
	}
    }

  fprintf (stderr, "speed_measure() could not get %d results within %.1f%%\n",
	   e, (TOLERANCE-1.0)*100.0);
  fprintf (stderr, "    unsorted         sorted\n");
  fprintf (stderr, "  %.12f    %.12f    is about 0.5%%\n",
	   t_unsorted[0]*(TOLERANCE-1.0), t[0]*(TOLERANCE-1.0));
  for (i = 0; i < numberof (t); i++)
    fprintf (stderr, "  %.09f       %.09f\n", t_unsorted[i], t[i]);

  return -1.0;
}


/* Read all of ptr,size to get it into the CPU memory cache.

   A call to mpn_cache_fill_dummy() is used to make sure the compiler
   doesn't optimize away the whole loop.  Using "volatile mp_limb_t sum"
   would work too, but the function call means we don't rely on every
   compiler actually implementing volatile properly.

   mpn_cache_fill_dummy() is in a separate source file to stop gcc thinking
   it can inline it.  */

void
mpn_cache_fill (mp_srcptr ptr, mp_size_t size)
{
  mp_limb_t  sum = 0;
  mp_size_t  i;

  for (i = 0; i < size; i++)
    sum += ptr[i];

  mpn_cache_fill_dummy(sum);
}


void
mpn_cache_fill_write (mp_ptr ptr, mp_size_t size)
{
  mpn_cache_fill (ptr, size);

#if 0
  mpn_random (ptr, size);
#endif

#if 0
  mp_size_t  i;

  for (i = 0; i < size; i++)
    ptr[i] = i;
#endif
}


void
speed_operand_src (struct speed_params *s, mp_ptr ptr, mp_size_t size)
{
  if (s->src_num >= numberof (s->src))
    {
      fprintf (stderr, "speed_operand_src: no room left in s->src[]\n");
      abort ();
    }
  s->src[s->src_num].ptr = ptr;
  s->src[s->src_num].size = size;
  s->src_num++;
}


void
speed_operand_dst (struct speed_params *s, mp_ptr ptr, mp_size_t size)
{
  if (s->dst_num >= numberof (s->dst))
    {
      fprintf (stderr, "speed_operand_dst: no room left in s->dst[]\n");
      abort ();
    }
  s->dst[s->dst_num].ptr = ptr;
  s->dst[s->dst_num].size = size;
  s->dst_num++;
}


void
speed_cache_fill (struct speed_params *s)
{
  static struct speed_params  prev;
  int  i;

  /* FIXME: need a better way to get the format string for a pointer */

  if (speed_option_addrs)
    {
      int  different;

      different = (s->dst_num != prev.dst_num || s->src_num != prev.src_num);
      for (i = 0; i < s->dst_num; i++)
	different |= (s->dst[i].ptr != prev.dst[i].ptr);
      for (i = 0; i < s->src_num; i++)
	different |= (s->src[i].ptr != prev.src[i].ptr);

      if (different)
	{
	  if (s->dst_num != 0)
	    {
	      printf ("dst");
	      for (i = 0; i < s->dst_num; i++)
		printf (" %08lX", (unsigned long) s->dst[i].ptr);
	      printf (" ");
	    }

	  if (s->src_num != 0)
	    {
	      printf ("src");
	      for (i = 0; i < s->src_num; i++)
		printf (" %08lX", (unsigned long) s->src[i].ptr);
	      printf (" ");
	    }
	  printf ("  (cf sp approx %08lX)\n", (unsigned long) &different);

	}

      memcpy (&prev, s, sizeof(prev));
    }

  switch (s->cache) {
  case 0:
    for (i = 0; i < s->dst_num; i++)
      mpn_cache_fill_write (s->dst[i].ptr, s->dst[i].size);
    for (i = 0; i < s->src_num; i++)
      mpn_cache_fill (s->src[i].ptr, s->src[i].size);
    break;
  case 1:
    pentium_wbinvd();
    break;
  }
}


/* Miscellaneous options accepted by tune and speed programs under -o. */

void
speed_option_set (const char *s)
{
  int  n;

  if (strcmp (s, "addrs") == 0)
    {
      speed_option_addrs = 1;
    }
  else if (strcmp (s, "verbose") == 0)
    {
      speed_option_verbose++;
    }
  else if (sscanf (s, "verbose=%d", &n) == 1)
    {
      speed_option_verbose = n;
    }
  else if (strcmp (s, "cycles-broken") == 0)
    {
      speed_option_cycles_broken = 1;
    }
  else
    {
      printf ("Unrecognised -o option: %s\n", s);
      exit (1);
    }
}


/* The following are basic speed running routines for various gmp functions.
   Many are very similar and use speed.h macros.

   Each routine allocates it's own destination space for the result of the
   function, because only it can know what the function needs.

   speed_starttime() and speed_endtime() are put tight around the code to be
   measured.  Any setups are done outside the timed portion.

   Each routine is responsible for its own cache priming.
   speed_cache_fill() is a good way to do this, see examples in speed.h.
   One cache priming possibility, for CPUs with write-allocate cache, and
   functions that don't take too long, is to do one dummy call before timing
   so as to cache everything that gets used.  But speed_measure() runs a
   routine at least twice and will take the smaller time, so this might not
   be necessary.

   Data alignment will be important, for source, destination and temporary
   workspace.  A routine can align its destination and workspace.  Programs
   using the routines will ensure s->xp and s->yp are aligned.  Aligning
   onto a CACHE_LINE_SIZE boundary is suggested.  s->align_wp and
   s->align_wp2 should be respected where it makes sense to do so.
   SPEED_TMP_ALLOC_LIMBS is a good way to do this.

   A loop of the following form can be expected to turn into good assembler
   code on most CPUs, thereby minimizing overhead in the measurement.  It
   can always be assumed s->reps >= 1.

	  i = s->reps
	  do
	    foo();
	  while (--i != 0);

   Additional parameters might be added to "struct speed_params" in the
   future.  Routines should ignore anything they don't use.

   s->size can be used creatively, and s->xp and s->yp can be ignored.  For
   example, speed_mpz_fac_ui() uses s->size as n for the factorial.  s->r is
   just a user-supplied parameter.  speed_mpn_lshift() uses it as a shift,
   speed_mpn_mul_1() uses it as a multiplier.  */


/* MPN_COPY etc can be macros, so the _CALL forms are necessary */
double
speed_MPN_COPY (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_COPY (MPN_COPY);
}
double
speed_MPN_COPY_INCR (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_COPY (MPN_COPY_INCR);
}
double
speed_MPN_COPY_DECR (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_COPY (MPN_COPY_DECR);
}
#if HAVE_NATIVE_mpn_copyi
double
speed_mpn_copyi (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_COPY (mpn_copyi);
}
#endif
#if HAVE_NATIVE_mpn_copyd
double
speed_mpn_copyd (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_COPY (mpn_copyd);
}
#endif
double
speed_memcpy (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_COPY_BYTES (memcpy);
}
double
speed_mpn_com (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_COPY (mpn_com);
}
double
speed_mpn_neg (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_COPY (mpn_neg);
}
double
speed_mpn_sec_tabselect (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_TABSELECT (mpn_sec_tabselect);
}


double
speed_mpn_addmul_1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_UNARY_1 (mpn_addmul_1);
}
double
speed_mpn_submul_1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_UNARY_1 (mpn_submul_1);
}

#if HAVE_NATIVE_mpn_addmul_2
double
speed_mpn_addmul_2 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_UNARY_2 (mpn_addmul_2);
}
#endif
#if HAVE_NATIVE_mpn_addmul_3
double
speed_mpn_addmul_3 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_UNARY_3 (mpn_addmul_3);
}
#endif
#if HAVE_NATIVE_mpn_addmul_4
double
speed_mpn_addmul_4 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_UNARY_4 (mpn_addmul_4);
}
#endif
#if HAVE_NATIVE_mpn_addmul_5
double
speed_mpn_addmul_5 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_UNARY_5 (mpn_addmul_5);
}
#endif
#if HAVE_NATIVE_mpn_addmul_6
double
speed_mpn_addmul_6 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_UNARY_6 (mpn_addmul_6);
}
#endif
#if HAVE_NATIVE_mpn_addmul_7
double
speed_mpn_addmul_7 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_UNARY_7 (mpn_addmul_7);
}
#endif
#if HAVE_NATIVE_mpn_addmul_8
double
speed_mpn_addmul_8 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_UNARY_8 (mpn_addmul_8);
}
#endif

double
speed_mpn_mul_1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_UNARY_1 (mpn_mul_1);
}
double
speed_mpn_mul_1_inplace (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_UNARY_1_INPLACE (mpn_mul_1);
}

#if HAVE_NATIVE_mpn_mul_2
double
speed_mpn_mul_2 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_UNARY_2 (mpn_mul_2);
}
#endif
#if HAVE_NATIVE_mpn_mul_3
double
speed_mpn_mul_3 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_UNARY_3 (mpn_mul_3);
}
#endif
#if HAVE_NATIVE_mpn_mul_4
double
speed_mpn_mul_4 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_UNARY_4 (mpn_mul_4);
}
#endif
#if HAVE_NATIVE_mpn_mul_5
double
speed_mpn_mul_5 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_UNARY_5 (mpn_mul_5);
}
#endif
#if HAVE_NATIVE_mpn_mul_6
double
speed_mpn_mul_6 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_UNARY_6 (mpn_mul_6);
}
#endif


double
speed_mpn_lshift (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_UNARY_1 (mpn_lshift);
}
double
speed_mpn_lshiftc (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_UNARY_1 (mpn_lshiftc);
}
double
speed_mpn_rshift (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_UNARY_1 (mpn_rshift);
}


/* The carry-in variants (if available) are good for measuring because they
   won't skip a division if high<divisor.  Alternately, use -1 as a divisor
   with the plain _1 forms. */
double
speed_mpn_divrem_1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_DIVREM_1 (mpn_divrem_1);
}
double
speed_mpn_divrem_1f (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_DIVREM_1F (mpn_divrem_1);
}
#if HAVE_NATIVE_mpn_divrem_1c
double
speed_mpn_divrem_1c (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_DIVREM_1C (mpn_divrem_1c);
}
double
speed_mpn_divrem_1cf (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_DIVREM_1CF (mpn_divrem_1c);
}
#endif

double
speed_mpn_divrem_1_div (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_DIVREM_1 (mpn_divrem_1_div);
}
double
speed_mpn_divrem_1f_div (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_DIVREM_1F (mpn_divrem_1_div);
}
double
speed_mpn_divrem_1_inv (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_DIVREM_1 (mpn_divrem_1_inv);
}
double
speed_mpn_divrem_1f_inv (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_DIVREM_1F (mpn_divrem_1_inv);
}
double
speed_mpn_mod_1_div (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MOD_1 (mpn_mod_1_div);
}
double
speed_mpn_mod_1_inv (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MOD_1 (mpn_mod_1_inv);
}

double
speed_mpn_preinv_divrem_1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_PREINV_DIVREM_1 (mpn_preinv_divrem_1);
}
double
speed_mpn_preinv_divrem_1f (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_PREINV_DIVREM_1F (mpn_preinv_divrem_1);
}

#if GMP_NUMB_BITS % 4 == 0
double
speed_mpn_mod_34lsub1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MOD_34LSUB1 (mpn_mod_34lsub1);
}
#endif

double
speed_mpn_divrem_2 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_DIVREM_2 (mpn_divrem_2);
}
double
speed_mpn_divrem_2_div (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_DIVREM_2 (mpn_divrem_2_div);
}
double
speed_mpn_divrem_2_inv (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_DIVREM_2 (mpn_divrem_2_inv);
}

double
speed_mpn_div_qr_1n_pi1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_DIV_QR_1N_PI1 (mpn_div_qr_1n_pi1);
}
double
speed_mpn_div_qr_1n_pi1_1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_DIV_QR_1N_PI1 (mpn_div_qr_1n_pi1_1);
}
double
speed_mpn_div_qr_1n_pi1_2 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_DIV_QR_1N_PI1 (mpn_div_qr_1n_pi1_2);
}

double
speed_mpn_div_qr_1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_DIV_QR_1 (mpn_div_qr_1);
}

double
speed_mpn_div_qr_2n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_DIV_QR_2 (mpn_div_qr_2, 1);
}
double
speed_mpn_div_qr_2u (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_DIV_QR_2 (mpn_div_qr_2, 0);
}

double
speed_mpn_mod_1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MOD_1 (mpn_mod_1);
}
#if HAVE_NATIVE_mpn_mod_1c
double
speed_mpn_mod_1c (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MOD_1C (mpn_mod_1c);
}
#endif
double
speed_mpn_preinv_mod_1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_PREINV_MOD_1 (mpn_preinv_mod_1);
}
double
speed_mpn_mod_1_1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MOD_1_1 (mpn_mod_1_1p,mpn_mod_1_1p_cps);
}
double
speed_mpn_mod_1_1_1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MOD_1_1 (mpn_mod_1_1p_1,mpn_mod_1_1p_cps_1);
}
double
speed_mpn_mod_1_1_2 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MOD_1_1 (mpn_mod_1_1p_2,mpn_mod_1_1p_cps_2);
}
double
speed_mpn_mod_1_2 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MOD_1_N (mpn_mod_1s_2p,mpn_mod_1s_2p_cps,2);
}
double
speed_mpn_mod_1_3 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MOD_1_N (mpn_mod_1s_3p,mpn_mod_1s_3p_cps,3);
}
double
speed_mpn_mod_1_4 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MOD_1_N (mpn_mod_1s_4p,mpn_mod_1s_4p_cps,4);
}

double
speed_mpn_divexact_1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_DIVEXACT_1 (mpn_divexact_1);
}

double
speed_mpn_divexact_by3 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_COPY (mpn_divexact_by3);
}

double
speed_mpn_bdiv_dbm1c (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BDIV_DBM1C (mpn_bdiv_dbm1c);
}

double
speed_mpn_bdiv_q_1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BDIV_Q_1 (mpn_bdiv_q_1);
}

double
speed_mpn_pi1_bdiv_q_1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_PI1_BDIV_Q_1 (mpn_pi1_bdiv_q_1);
}

#if HAVE_NATIVE_mpn_modexact_1_odd
double
speed_mpn_modexact_1_odd (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MODEXACT_1_ODD (mpn_modexact_1_odd);
}
#endif

double
speed_mpn_modexact_1c_odd (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MODEXACT_1C_ODD (mpn_modexact_1c_odd);
}

double
speed_mpz_mod (struct speed_params *s)
{
  SPEED_ROUTINE_MPZ_MOD (mpz_mod);
}

double
speed_mpn_sbpi1_div_qr (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_PI1_DIV (mpn_sbpi1_div_qr, inv.inv32, 2,0);
}
double
speed_mpn_dcpi1_div_qr (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_PI1_DIV (mpn_dcpi1_div_qr, &inv, 6,3);
}
double
speed_mpn_sbpi1_divappr_q (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_PI1_DIV (mpn_sbpi1_divappr_q, inv.inv32, 2,0);
}
double
speed_mpn_dcpi1_divappr_q (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_PI1_DIV (mpn_dcpi1_divappr_q, &inv, 6,3);
}
double
speed_mpn_mu_div_qr (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MU_DIV_QR (mpn_mu_div_qr, mpn_mu_div_qr_itch);
}
double
speed_mpn_mu_divappr_q (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MU_DIV_Q (mpn_mu_divappr_q, mpn_mu_divappr_q_itch);
}
double
speed_mpn_mu_div_q (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MU_DIV_Q (mpn_mu_div_q, mpn_mu_div_q_itch);
}
double
speed_mpn_mupi_div_qr (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MUPI_DIV_QR (mpn_preinv_mu_div_qr, mpn_preinv_mu_div_qr_itch);
}

double
speed_mpn_sbpi1_bdiv_qr (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_PI1_BDIV_QR (mpn_sbpi1_bdiv_qr);
}
double
speed_mpn_dcpi1_bdiv_qr (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_PI1_BDIV_QR (mpn_dcpi1_bdiv_qr);
}
double
speed_mpn_sbpi1_bdiv_q (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_PI1_BDIV_Q (mpn_sbpi1_bdiv_q);
}
double
speed_mpn_dcpi1_bdiv_q (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_PI1_BDIV_Q (mpn_dcpi1_bdiv_q);
}
double
speed_mpn_mu_bdiv_q (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MU_BDIV_Q (mpn_mu_bdiv_q, mpn_mu_bdiv_q_itch);
}
double
speed_mpn_mu_bdiv_qr (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MU_BDIV_QR (mpn_mu_bdiv_qr, mpn_mu_bdiv_qr_itch);
}

double
speed_mpn_broot (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BROOT (mpn_broot);
}
double
speed_mpn_broot_invm1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BROOT (mpn_broot_invm1);
}
double
speed_mpn_brootinv (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BROOTINV (mpn_brootinv, 5*s->size);
}

double
speed_mpn_binvert (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINVERT (mpn_binvert, mpn_binvert_itch);
}

double
speed_mpn_invert (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_INVERT (mpn_invert, mpn_invert_itch);
}

double
speed_mpn_invertappr (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_INVERTAPPR (mpn_invertappr, mpn_invertappr_itch);
}

double
speed_mpn_ni_invertappr (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_INVERTAPPR (mpn_ni_invertappr, mpn_invertappr_itch);
}

double
speed_mpn_sec_invert (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_SEC_INVERT (mpn_sec_invert, mpn_sec_invert_itch);
}

double
speed_mpn_redc_1 (struct speed_params *s)
{
  SPEED_ROUTINE_REDC_1 (mpn_redc_1);
}
double
speed_mpn_redc_2 (struct speed_params *s)
{
  SPEED_ROUTINE_REDC_2 (mpn_redc_2);
}
double
speed_mpn_redc_n (struct speed_params *s)
{
  SPEED_ROUTINE_REDC_N (mpn_redc_n);
}


double
speed_mpn_popcount (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_POPCOUNT (mpn_popcount);
}
double
speed_mpn_hamdist (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_HAMDIST (mpn_hamdist);
}


double
speed_mpn_add_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_N (mpn_add_n);
}
double
speed_mpn_sub_n (struct speed_params *s)
{
SPEED_ROUTINE_MPN_BINARY_N (mpn_sub_n);
}
double
speed_mpn_add_1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_UNARY_1 (mpn_add_1);
}
double
speed_mpn_add_1_inplace (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_UNARY_1_INPLACE (mpn_add_1);
}
double
speed_mpn_sub_1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_UNARY_1 (mpn_sub_1);
}
double
speed_mpn_sub_1_inplace (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_UNARY_1_INPLACE (mpn_sub_1);
}

double
speed_mpn_add_err1_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_ERR1_N (mpn_add_err1_n);
}
double
speed_mpn_sub_err1_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_ERR1_N (mpn_sub_err1_n);
}
double
speed_mpn_add_err2_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_ERR2_N (mpn_add_err2_n);
}
double
speed_mpn_sub_err2_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_ERR2_N (mpn_sub_err2_n);
}
double
speed_mpn_add_err3_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_ERR3_N (mpn_add_err3_n);
}
double
speed_mpn_sub_err3_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_ERR3_N (mpn_sub_err3_n);
}


#if HAVE_NATIVE_mpn_add_n_sub_n
double
speed_mpn_add_n_sub_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_ADDSUB_N_CALL (mpn_add_n_sub_n (ap, sp, s->xp, s->yp, s->size));
}
#endif

#if HAVE_NATIVE_mpn_addlsh1_n == 1
double
speed_mpn_addlsh1_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_N (mpn_addlsh1_n);
}
#endif
#if HAVE_NATIVE_mpn_sublsh1_n == 1
double
speed_mpn_sublsh1_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_N (mpn_sublsh1_n);
}
#endif
#if HAVE_NATIVE_mpn_addlsh1_n_ip1
double
speed_mpn_addlsh1_n_ip1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_COPY (mpn_addlsh1_n_ip1);
}
#endif
#if HAVE_NATIVE_mpn_addlsh1_n_ip2
double
speed_mpn_addlsh1_n_ip2 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_COPY (mpn_addlsh1_n_ip2);
}
#endif
#if HAVE_NATIVE_mpn_sublsh1_n_ip1
double
speed_mpn_sublsh1_n_ip1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_COPY (mpn_sublsh1_n_ip1);
}
#endif
#if HAVE_NATIVE_mpn_rsblsh1_n == 1
double
speed_mpn_rsblsh1_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_N (mpn_rsblsh1_n);
}
#endif
#if HAVE_NATIVE_mpn_addlsh2_n == 1
double
speed_mpn_addlsh2_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_N (mpn_addlsh2_n);
}
#endif
#if HAVE_NATIVE_mpn_sublsh2_n == 1
double
speed_mpn_sublsh2_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_N (mpn_sublsh2_n);
}
#endif
#if HAVE_NATIVE_mpn_addlsh2_n_ip1
double
speed_mpn_addlsh2_n_ip1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_COPY (mpn_addlsh2_n_ip1);
}
#endif
#if HAVE_NATIVE_mpn_addlsh2_n_ip2
double
speed_mpn_addlsh2_n_ip2 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_COPY (mpn_addlsh2_n_ip2);
}
#endif
#if HAVE_NATIVE_mpn_sublsh2_n_ip1
double
speed_mpn_sublsh2_n_ip1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_COPY (mpn_sublsh2_n_ip1);
}
#endif
#if HAVE_NATIVE_mpn_rsblsh2_n == 1
double
speed_mpn_rsblsh2_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_N (mpn_rsblsh2_n);
}
#endif
#if HAVE_NATIVE_mpn_addlsh_n
double
speed_mpn_addlsh_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_N_CALL (mpn_addlsh_n (wp, xp, yp, s->size, 7));
}
#endif
#if HAVE_NATIVE_mpn_sublsh_n
double
speed_mpn_sublsh_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_N_CALL (mpn_sublsh_n (wp, xp, yp, s->size, 7));
}
#endif
#if HAVE_NATIVE_mpn_addlsh_n_ip1
double
speed_mpn_addlsh_n_ip1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_UNARY_1_CALL (mpn_addlsh_n_ip1 (wp, s->xp, s->size, 7));
}
#endif
#if HAVE_NATIVE_mpn_addlsh_n_ip2
double
speed_mpn_addlsh_n_ip2 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_UNARY_1_CALL (mpn_addlsh_n_ip2 (wp, s->xp, s->size, 7));
}
#endif
#if HAVE_NATIVE_mpn_sublsh_n_ip1
double
speed_mpn_sublsh_n_ip1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_UNARY_1_CALL (mpn_sublsh_n_ip1 (wp, s->xp, s->size, 7));
}
#endif
#if HAVE_NATIVE_mpn_rsblsh_n
double
speed_mpn_rsblsh_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_N_CALL (mpn_rsblsh_n (wp, xp, yp, s->size, 7));
}
#endif
#if HAVE_NATIVE_mpn_rsh1add_n
double
speed_mpn_rsh1add_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_N (mpn_rsh1add_n);
}
#endif
#if HAVE_NATIVE_mpn_rsh1sub_n
double
speed_mpn_rsh1sub_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_N (mpn_rsh1sub_n);
}
#endif

double
speed_mpn_cnd_add_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_N_CALL (mpn_cnd_add_n (1, wp, xp, yp, s->size));
}
double
speed_mpn_cnd_sub_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_N_CALL (mpn_cnd_sub_n (1, wp, xp, yp, s->size));
}

/* mpn_and_n etc can be macros and so have to be handled with
   SPEED_ROUTINE_MPN_BINARY_N_CALL forms */
double
speed_mpn_and_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_N_CALL (mpn_and_n (wp, xp, yp, s->size));
}
double
speed_mpn_andn_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_N_CALL (mpn_andn_n (wp, xp, yp, s->size));
}
double
speed_mpn_nand_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_N_CALL (mpn_nand_n (wp, xp, yp, s->size));
}
double
speed_mpn_ior_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_N_CALL (mpn_ior_n (wp, xp, yp, s->size));
}
double
speed_mpn_iorn_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_N_CALL (mpn_iorn_n (wp, xp, yp, s->size));
}
double
speed_mpn_nior_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_N_CALL (mpn_nior_n (wp, xp, yp, s->size));
}
double
speed_mpn_xor_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_N_CALL (mpn_xor_n (wp, xp, yp, s->size));
}
double
speed_mpn_xnor_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_BINARY_N_CALL (mpn_xnor_n (wp, xp, yp, s->size));
}


double
speed_mpn_mul_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MUL_N (mpn_mul_n);
}
double
speed_mpn_sqr (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_SQR (mpn_sqr);
}
double
speed_mpn_mul_n_sqr (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_SQR_CALL (mpn_mul_n (wp, s->xp, s->xp, s->size));
}

double
speed_mpn_mul_basecase (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MUL(mpn_mul_basecase);
}
double
speed_mpn_mul (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MUL(mpn_mul);
}
double
speed_mpn_sqr_basecase (struct speed_params *s)
{
  /* FIXME: size restrictions on some versions of sqr_basecase */
  SPEED_ROUTINE_MPN_SQR (mpn_sqr_basecase);
}

#if HAVE_NATIVE_mpn_sqr_diagonal
double
speed_mpn_sqr_diagonal (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_SQR (mpn_sqr_diagonal);
}
#endif

#if HAVE_NATIVE_mpn_sqr_diag_addlsh1
double
speed_mpn_sqr_diag_addlsh1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_SQR_DIAG_ADDLSH1_CALL (mpn_sqr_diag_addlsh1 (wp, tp, s->xp, s->size));
}
#endif

double
speed_mpn_toom2_sqr (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_TOOM2_SQR (mpn_toom2_sqr);
}
double
speed_mpn_toom3_sqr (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_TOOM3_SQR (mpn_toom3_sqr);
}
double
speed_mpn_toom4_sqr (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_TOOM4_SQR (mpn_toom4_sqr);
}
double
speed_mpn_toom6_sqr (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_TOOM6_SQR (mpn_toom6_sqr);
}
double
speed_mpn_toom8_sqr (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_TOOM8_SQR (mpn_toom8_sqr);
}
double
speed_mpn_toom22_mul (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_TOOM22_MUL_N (mpn_toom22_mul);
}
double
speed_mpn_toom33_mul (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_TOOM33_MUL_N (mpn_toom33_mul);
}
double
speed_mpn_toom44_mul (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_TOOM44_MUL_N (mpn_toom44_mul);
}
double
speed_mpn_toom6h_mul (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_TOOM6H_MUL_N (mpn_toom6h_mul);
}
double
speed_mpn_toom8h_mul (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_TOOM8H_MUL_N (mpn_toom8h_mul);
}

double
speed_mpn_toom32_mul (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_TOOM32_MUL (mpn_toom32_mul);
}
double
speed_mpn_toom42_mul (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_TOOM42_MUL (mpn_toom42_mul);
}
double
speed_mpn_toom43_mul (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_TOOM43_MUL (mpn_toom43_mul);
}
double
speed_mpn_toom63_mul (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_TOOM63_MUL (mpn_toom63_mul);
}
double
speed_mpn_toom32_for_toom43_mul (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_TOOM32_FOR_TOOM43_MUL (mpn_toom32_mul);
}
double
speed_mpn_toom43_for_toom32_mul (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_TOOM43_FOR_TOOM32_MUL (mpn_toom43_mul);
}
double
speed_mpn_toom32_for_toom53_mul (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_TOOM32_FOR_TOOM53_MUL (mpn_toom32_mul);
}
double
speed_mpn_toom53_for_toom32_mul (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_TOOM53_FOR_TOOM32_MUL (mpn_toom53_mul);
}
double
speed_mpn_toom42_for_toom53_mul (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_TOOM42_FOR_TOOM53_MUL (mpn_toom42_mul);
}
double
speed_mpn_toom53_for_toom42_mul (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_TOOM53_FOR_TOOM42_MUL (mpn_toom53_mul);
}
double
speed_mpn_toom43_for_toom54_mul (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_TOOM43_FOR_TOOM54_MUL (mpn_toom43_mul);
}
double
speed_mpn_toom54_for_toom43_mul (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_TOOM54_FOR_TOOM43_MUL (mpn_toom54_mul);
}

double
speed_mpn_nussbaumer_mul (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MUL_N_CALL
    (mpn_nussbaumer_mul (wp, s->xp, s->size, s->yp, s->size));
}
double
speed_mpn_nussbaumer_mul_sqr (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_SQR_CALL
    (mpn_nussbaumer_mul (wp, s->xp, s->size, s->xp, s->size));
}

#if WANT_OLD_FFT_FULL
double
speed_mpn_mul_fft_full (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MUL_N_CALL
    (mpn_mul_fft_full (wp, s->xp, s->size, s->yp, s->size));
}
double
speed_mpn_mul_fft_full_sqr (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_SQR_CALL
    (mpn_mul_fft_full (wp, s->xp, s->size, s->xp, s->size));
}
#endif

/* These are mod 2^N+1 multiplies and squares.  If s->r is supplied it's
   used as k, otherwise the best k for the size is used.  If s->size isn't a
   multiple of 2^k it's rounded up to make the effective operation size.  */

#define SPEED_ROUTINE_MPN_MUL_FFT_CALL(call, sqr)       \
  {                                                     \
    mp_ptr     wp;                                      \
    mp_size_t  pl;                                      \
    int        k;                                       \
    unsigned   i;                                       \
    double     t;                                       \
    TMP_DECL;                                           \
							\
    SPEED_RESTRICT_COND (s->size >= 1);                 \
							\
    if (s->r != 0)                                      \
      k = s->r;                                         \
    else                                                \
      k = mpn_fft_best_k (s->size, sqr);                \
							\
    TMP_MARK;                                           \
    pl = mpn_fft_next_size (s->size, k);                \
    SPEED_TMP_ALLOC_LIMBS (wp, pl+1, s->align_wp);      \
							\
    speed_operand_src (s, s->xp, s->size);              \
    if (!sqr)                                           \
      speed_operand_src (s, s->yp, s->size);            \
    speed_operand_dst (s, wp, pl+1);                    \
    speed_cache_fill (s);                               \
							\
    speed_starttime ();                                 \
    i = s->reps;                                        \
    do                                                  \
      call;                                             \
    while (--i != 0);                                   \
    t = speed_endtime ();                               \
							\
    TMP_FREE;                                           \
    return t;                                           \
  }

double
speed_mpn_mul_fft (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MUL_FFT_CALL
    (mpn_mul_fft (wp, pl, s->xp, s->size, s->yp, s->size, k), 0);
}

double
speed_mpn_mul_fft_sqr (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MUL_FFT_CALL
    (mpn_mul_fft (wp, pl, s->xp, s->size, s->xp, s->size, k), 1);
}

double
speed_mpn_fft_mul (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MUL_N_CALL (mpn_fft_mul (wp, s->xp, s->size, s->yp, s->size));
}

double
speed_mpn_fft_sqr (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_SQR_CALL (mpn_fft_mul (wp, s->xp, s->size, s->xp, s->size));
}

double
speed_mpn_sqrlo (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_SQRLO (mpn_sqrlo);
}
double
speed_mpn_sqrlo_basecase (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_SQRLO (mpn_sqrlo_basecase);
}
double
speed_mpn_mullo_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MULLO_N (mpn_mullo_n);
}
double
speed_mpn_mullo_basecase (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MULLO_BASECASE (mpn_mullo_basecase);
}

double
speed_mpn_mulmid_basecase (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MULMID (mpn_mulmid_basecase);
}

double
speed_mpn_mulmid (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MULMID (mpn_mulmid);
}

double
speed_mpn_mulmid_n (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MULMID_N (mpn_mulmid_n);
}

double
speed_mpn_toom42_mulmid (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_TOOM42_MULMID (mpn_toom42_mulmid);
}

double
speed_mpn_mulmod_bnm1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MULMOD_BNM1_CALL (mpn_mulmod_bnm1 (wp, s->size, s->xp, s->size, s->yp, s->size, tp));
}

double
speed_mpn_bc_mulmod_bnm1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MULMOD_BNM1_CALL (mpn_bc_mulmod_bnm1 (wp, s->xp, s->yp, s->size, tp));
}

double
speed_mpn_mulmod_bnm1_rounded (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MULMOD_BNM1_ROUNDED (mpn_mulmod_bnm1);
}

double
speed_mpn_sqrmod_bnm1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MULMOD_BNM1_CALL (mpn_sqrmod_bnm1 (wp, s->size, s->xp, s->size, tp));
}

double
speed_mpn_matrix22_mul (struct speed_params *s)
{
  /* Speed params only includes 2 inputs, so we have to invent the
     other 6. */

  mp_ptr a;
  mp_ptr r;
  mp_ptr b;
  mp_ptr tp;
  mp_size_t itch;
  unsigned i;
  double t;
  TMP_DECL;

  TMP_MARK;
  SPEED_TMP_ALLOC_LIMBS (a, 4 * s->size, s->align_xp);
  SPEED_TMP_ALLOC_LIMBS (b, 4 * s->size, s->align_yp);
  SPEED_TMP_ALLOC_LIMBS (r, 8 * s->size + 4, s->align_wp);

  MPN_COPY (a, s->xp, s->size);
  mpn_random (a + s->size, 3 * s->size);
  MPN_COPY (b, s->yp, s->size);
  mpn_random (b + s->size, 3 * s->size);

  itch = mpn_matrix22_mul_itch (s->size, s->size);
  SPEED_TMP_ALLOC_LIMBS (tp, itch, s->align_wp2);

  speed_operand_src (s, a, 4 * s->size);
  speed_operand_src (s, b, 4 * s->size);
  speed_operand_dst (s, r, 8 * s->size + 4);
  speed_operand_dst (s, tp, itch);
  speed_cache_fill (s);

  speed_starttime ();
  i = s->reps;
  do
    {
      mp_size_t sz = s->size;
      MPN_COPY (r + 0 * sz + 0, a + 0 * sz, sz);
      MPN_COPY (r + 2 * sz + 1, a + 1 * sz, sz);
      MPN_COPY (r + 4 * sz + 2, a + 2 * sz, sz);
      MPN_COPY (r + 6 * sz + 3, a + 3 * sz, sz);
      mpn_matrix22_mul (r, r + 2 * sz + 1, r + 4 * sz + 2, r + 6 * sz + 3, sz,
			b, b + 1 * sz,     b + 2 * sz,     b + 3 * sz,     sz,
			tp);
    }
  while (--i != 0);
  t = speed_endtime();
  TMP_FREE;
  return t;
}

double
speed_mpn_hgcd (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_HGCD_CALL (mpn_hgcd, mpn_hgcd_itch);
}

double
speed_mpn_hgcd_lehmer (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_HGCD_CALL (mpn_hgcd_lehmer, mpn_hgcd_lehmer_itch);
}

double
speed_mpn_hgcd_appr (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_HGCD_CALL (mpn_hgcd_appr, mpn_hgcd_appr_itch);
}

double
speed_mpn_hgcd_appr_lehmer (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_HGCD_CALL (mpn_hgcd_appr_lehmer, mpn_hgcd_appr_lehmer_itch);
}

double
speed_mpn_hgcd_reduce (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_HGCD_REDUCE_CALL (mpn_hgcd_reduce, mpn_hgcd_reduce_itch);
}
double
speed_mpn_hgcd_reduce_1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_HGCD_REDUCE_CALL (mpn_hgcd_reduce_1, mpn_hgcd_reduce_1_itch);
}
double
speed_mpn_hgcd_reduce_2 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_HGCD_REDUCE_CALL (mpn_hgcd_reduce_2, mpn_hgcd_reduce_2_itch);
}

double
speed_mpn_gcd (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_GCD (mpn_gcd);
}

double
speed_mpn_gcdext (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_GCDEXT (mpn_gcdext);
}
#if 0
double
speed_mpn_gcdext_lehmer (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_GCDEXT (__gmpn_gcdext_lehmer);
}
#endif
double
speed_mpn_gcdext_single (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_GCDEXT (mpn_gcdext_single);
}
double
speed_mpn_gcdext_double (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_GCDEXT (mpn_gcdext_double);
}
double
speed_mpn_gcdext_one_single (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_GCDEXT_ONE (mpn_gcdext_one_single);
}
double
speed_mpn_gcdext_one_double (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_GCDEXT_ONE (mpn_gcdext_one_double);
}
double
speed_mpn_gcd_1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_GCD_1 (mpn_gcd_1);
}
double
speed_mpn_gcd_1N (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_GCD_1N (mpn_gcd_1);
}


double
speed_mpz_jacobi (struct speed_params *s)
{
  SPEED_ROUTINE_MPZ_JACOBI (mpz_jacobi);
}
double
speed_mpn_jacobi_base (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_JACBASE (mpn_jacobi_base);
}
double
speed_mpn_jacobi_base_1 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_JACBASE (mpn_jacobi_base_1);
}
double
speed_mpn_jacobi_base_2 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_JACBASE (mpn_jacobi_base_2);
}
double
speed_mpn_jacobi_base_3 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_JACBASE (mpn_jacobi_base_3);
}
double
speed_mpn_jacobi_base_4 (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_JACBASE (mpn_jacobi_base_4);
}


double
speed_mpn_sqrtrem (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_SQRTROOT_CALL (mpn_sqrtrem (wp, wp2, s->xp, s->size));
}

double
speed_mpn_sqrt (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_SQRTROOT_CALL (mpn_sqrtrem (wp, NULL, s->xp, s->size));
}

double
speed_mpn_rootrem (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_SQRTROOT_CALL (mpn_rootrem (wp, wp2, s->xp, s->size, s->r));
}

double
speed_mpn_root (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_SQRTROOT_CALL (mpn_rootrem (wp, NULL, s->xp, s->size, s->r));
}


double
speed_mpz_fac_ui (struct speed_params *s)
{
  SPEED_ROUTINE_MPZ_FAC_UI (mpz_fac_ui);
}

double
speed_mpz_2fac_ui (struct speed_params *s)
{
  SPEED_ROUTINE_MPZ_UI (mpz_2fac_ui);
}


double
speed_mpn_fib2_ui (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_FIB2_UI (mpn_fib2_ui);
}
double
speed_mpz_fib_ui (struct speed_params *s)
{
  SPEED_ROUTINE_MPZ_FIB_UI (mpz_fib_ui);
}
double
speed_mpz_fib2_ui (struct speed_params *s)
{
  SPEED_ROUTINE_MPZ_FIB2_UI (mpz_fib2_ui);
}
double
speed_mpz_lucnum_ui (struct speed_params *s)
{
  SPEED_ROUTINE_MPZ_LUCNUM_UI (mpz_lucnum_ui);
}
double
speed_mpz_lucnum2_ui (struct speed_params *s)
{
  SPEED_ROUTINE_MPZ_LUCNUM2_UI (mpz_lucnum2_ui);
}


double
speed_mpz_powm (struct speed_params *s)
{
  SPEED_ROUTINE_MPZ_POWM (mpz_powm);
}
double
speed_mpz_powm_mod (struct speed_params *s)
{
  SPEED_ROUTINE_MPZ_POWM (mpz_powm_mod);
}
double
speed_mpz_powm_redc (struct speed_params *s)
{
  SPEED_ROUTINE_MPZ_POWM (mpz_powm_redc);
}
double
speed_mpz_powm_sec (struct speed_params *s)
{
  SPEED_ROUTINE_MPZ_POWM (mpz_powm_sec);
}
double
speed_mpz_powm_ui (struct speed_params *s)
{
  SPEED_ROUTINE_MPZ_POWM_UI (mpz_powm_ui);
}


double
speed_binvert_limb (struct speed_params *s)
{
  SPEED_ROUTINE_MODLIMB_INVERT (binvert_limb);
}


double
speed_noop (struct speed_params *s)
{
  unsigned  i;

  speed_starttime ();
  i = s->reps;
  do
    noop ();
  while (--i != 0);
  return speed_endtime ();
}

double
speed_noop_wxs (struct speed_params *s)
{
  mp_ptr   wp;
  unsigned i;
  double   t;
  TMP_DECL;

  TMP_MARK;
  wp = TMP_ALLOC_LIMBS (1);

  speed_starttime ();
  i = s->reps;
  do
    noop_wxs (wp, s->xp, s->size);
  while (--i != 0);
  t = speed_endtime ();

  TMP_FREE;
  return t;
}

double
speed_noop_wxys (struct speed_params *s)
{
  mp_ptr   wp;
  unsigned i;
  double   t;
  TMP_DECL;

  TMP_MARK;
  wp = TMP_ALLOC_LIMBS (1);

  speed_starttime ();
  i = s->reps;
  do
    noop_wxys (wp, s->xp, s->yp, s->size);
  while (--i != 0);
  t = speed_endtime ();

  TMP_FREE;
  return t;
}


#define SPEED_ROUTINE_ALLOC_FREE(variables, calls)      \
  {                                                     \
    unsigned  i;                                        \
    variables;                                          \
							\
    speed_starttime ();                                 \
    i = s->reps;                                        \
    do                                                  \
      {                                                 \
	calls;                                          \
      }                                                 \
    while (--i != 0);                                   \
    return speed_endtime ();                            \
  }


/* Compare these to see how much malloc/free costs and then how much
   __gmp_default_allocate/free and mpz_init/clear add.  mpz_init/clear or
   mpq_init/clear will be doing a 1 limb allocate, so use that as the size
   when including them in comparisons.  */

double
speed_malloc_free (struct speed_params *s)
{
  size_t  bytes = s->size * GMP_LIMB_BYTES;
  SPEED_ROUTINE_ALLOC_FREE (void *p,
			    p = malloc (bytes);
			    free (p));
}

double
speed_malloc_realloc_free (struct speed_params *s)
{
  size_t  bytes = s->size * GMP_LIMB_BYTES;
  SPEED_ROUTINE_ALLOC_FREE (void *p,
			    p = malloc (GMP_LIMB_BYTES);
			    p = realloc (p, bytes);
			    free (p));
}

double
speed_gmp_allocate_free (struct speed_params *s)
{
  size_t  bytes = s->size * GMP_LIMB_BYTES;
  SPEED_ROUTINE_ALLOC_FREE (void *p,
			    p = (*__gmp_allocate_func) (bytes);
			    (*__gmp_free_func) (p, bytes));
}

double
speed_gmp_allocate_reallocate_free (struct speed_params *s)
{
  size_t  bytes = s->size * GMP_LIMB_BYTES;
  SPEED_ROUTINE_ALLOC_FREE
    (void *p,
     p = (*__gmp_allocate_func) (GMP_LIMB_BYTES);
     p = (*__gmp_reallocate_func) (p, bytes, GMP_LIMB_BYTES);
     (*__gmp_free_func) (p, bytes));
}

double
speed_mpz_init_clear (struct speed_params *s)
{
  SPEED_ROUTINE_ALLOC_FREE (mpz_t z,
			    mpz_init (z);
			    mpz_clear (z));
}

double
speed_mpz_init_realloc_clear (struct speed_params *s)
{
  SPEED_ROUTINE_ALLOC_FREE (mpz_t z,
			    mpz_init (z);
			    _mpz_realloc (z, s->size);
			    mpz_clear (z));
}

double
speed_mpq_init_clear (struct speed_params *s)
{
  SPEED_ROUTINE_ALLOC_FREE (mpq_t q,
			    mpq_init (q);
			    mpq_clear (q));
}

double
speed_mpf_init_clear (struct speed_params *s)
{
  SPEED_ROUTINE_ALLOC_FREE (mpf_t f,
			    mpf_init (f);
			    mpf_clear (f));
}


/* Compare this to mpn_add_n to see how much overhead mpz_add adds.  Note
   that repeatedly calling mpz_add with the same data gives branch prediction
   in it an advantage.  */

double
speed_mpz_add (struct speed_params *s)
{
  mpz_t     w, x, y;
  unsigned  i;
  double    t;

  mpz_init (w);
  mpz_init (x);
  mpz_init (y);

  mpz_set_n (x, s->xp, s->size);
  mpz_set_n (y, s->yp, s->size);
  mpz_add (w, x, y);

  speed_starttime ();
  i = s->reps;
  do
    {
      mpz_add (w, x, y);
    }
  while (--i != 0);
  t = speed_endtime ();

  mpz_clear (w);
  mpz_clear (x);
  mpz_clear (y);
  return t;
}


/* If r==0, calculate (size,size/2),
   otherwise calculate (size,r). */

double
speed_mpz_bin_uiui (struct speed_params *s)
{
  mpz_t          w;
  unsigned long  k;
  unsigned  i;
  double    t;

  mpz_init (w);
  if (s->r != 0)
    k = s->r;
  else
    k = s->size/2;

  speed_starttime ();
  i = s->reps;
  do
    {
      mpz_bin_uiui (w, s->size, k);
    }
  while (--i != 0);
  t = speed_endtime ();

  mpz_clear (w);
  return t;
}

/* If r==0, calculate binomial(2^size,size),
   otherwise calculate binomial(2^size,r). */

double
speed_mpz_bin_ui (struct speed_params *s)
{
  mpz_t          w, x;
  unsigned long  k;
  unsigned  i;
  double    t;

  mpz_init (w);
  mpz_init_set_ui (x, 0);

  mpz_setbit (x, s->size);

  if (s->r != 0)
    k = s->r;
  else
    k = s->size;

  speed_starttime ();
  i = s->reps;
  do
    {
      mpz_bin_ui (w, x, k);
    }
  while (--i != 0);
  t = speed_endtime ();

  mpz_clear (w);
  mpz_clear (x);
  return t;
}

/* The multiplies are successively dependent so the latency is measured, not
   the issue rate.  There's only 10 per loop so the code doesn't get too big
   since umul_ppmm is several instructions on some cpus.

   Putting the arguments as "h,l,l,h" gets slightly better code from gcc
   2.95.2 on x86, it puts only one mov between each mul, not two.  That mov
   though will probably show up as a bogus extra cycle though.

   The measuring function macros are into three parts to avoid overflowing
   preprocessor expansion space if umul_ppmm is big.

   Limitations:

   Don't blindly use this to set UMUL_TIME in gmp-mparam.h, check the code
   generated first, especially on CPUs with low latency multipliers.

   The default umul_ppmm doing h*l will be getting increasing numbers of
   high zero bits in the calculation.  CPUs with data-dependent multipliers
   will want to use umul_ppmm.1 to get some randomization into the
   calculation.  The extra xors and fetches will be a slowdown of course.  */

#define SPEED_MACRO_UMUL_PPMM_A \
  {                             \
    mp_limb_t  h, l;            \
    unsigned   i;               \
    double     t;               \
				\
    s->time_divisor = 10;       \
				\
    h = s->xp[0];               \
    l = s->yp[0];               \
				\
    if (s->r == 1)              \
      {                         \
	speed_starttime ();     \
	i = s->reps;            \
	do                      \
	  {

#define SPEED_MACRO_UMUL_PPMM_B \
	  }                     \
	while (--i != 0);       \
	t = speed_endtime ();   \
      }                         \
    else                        \
      {                         \
	speed_starttime ();     \
	i = s->reps;            \
	do                      \
	  {

#define SPEED_MACRO_UMUL_PPMM_C                                         \
	  }                                                             \
	while (--i != 0);                                               \
	t = speed_endtime ();                                           \
      }                                                                 \
									\
    /* stop the compiler optimizing away the whole calculation! */      \
    noop_1 (h);                                                         \
    noop_1 (l);                                                         \
									\
    return t;                                                           \
  }


double
speed_umul_ppmm (struct speed_params *s)
{
  SPEED_MACRO_UMUL_PPMM_A;
  {
    umul_ppmm (h, l, l, h);  h ^= s->xp_block[0]; l ^= s->yp_block[0];
     umul_ppmm (h, l, l, h); h ^= s->xp_block[1]; l ^= s->yp_block[1];
     umul_ppmm (h, l, l, h); h ^= s->xp_block[2]; l ^= s->yp_block[2];
    umul_ppmm (h, l, l, h);  h ^= s->xp_block[3]; l ^= s->yp_block[3];
     umul_ppmm (h, l, l, h); h ^= s->xp_block[4]; l ^= s->yp_block[4];
     umul_ppmm (h, l, l, h); h ^= s->xp_block[5]; l ^= s->yp_block[5];
    umul_ppmm (h, l, l, h);  h ^= s->xp_block[6]; l ^= s->yp_block[6];
     umul_ppmm (h, l, l, h); h ^= s->xp_block[7]; l ^= s->yp_block[7];
     umul_ppmm (h, l, l, h); h ^= s->xp_block[8]; l ^= s->yp_block[8];
    umul_ppmm (h, l, l, h);  h ^= s->xp_block[9]; l ^= s->yp_block[9];
  }
  SPEED_MACRO_UMUL_PPMM_B;
  {
    umul_ppmm (h, l, l, h);
     umul_ppmm (h, l, l, h);
     umul_ppmm (h, l, l, h);
    umul_ppmm (h, l, l, h);
     umul_ppmm (h, l, l, h);
     umul_ppmm (h, l, l, h);
    umul_ppmm (h, l, l, h);
     umul_ppmm (h, l, l, h);
     umul_ppmm (h, l, l, h);
    umul_ppmm (h, l, l, h);
  }
  SPEED_MACRO_UMUL_PPMM_C;
}


#if HAVE_NATIVE_mpn_umul_ppmm
double
speed_mpn_umul_ppmm (struct speed_params *s)
{
  SPEED_MACRO_UMUL_PPMM_A;
  {
    h = mpn_umul_ppmm (&l, h, l);  h ^= s->xp_block[0]; l ^= s->yp_block[0];
     h = mpn_umul_ppmm (&l, h, l); h ^= s->xp_block[1]; l ^= s->yp_block[1];
     h = mpn_umul_ppmm (&l, h, l); h ^= s->xp_block[2]; l ^= s->yp_block[2];
    h = mpn_umul_ppmm (&l, h, l);  h ^= s->xp_block[3]; l ^= s->yp_block[3];
     h = mpn_umul_ppmm (&l, h, l); h ^= s->xp_block[4]; l ^= s->yp_block[4];
     h = mpn_umul_ppmm (&l, h, l); h ^= s->xp_block[5]; l ^= s->yp_block[5];
    h = mpn_umul_ppmm (&l, h, l);  h ^= s->xp_block[6]; l ^= s->yp_block[6];
     h = mpn_umul_ppmm (&l, h, l); h ^= s->xp_block[7]; l ^= s->yp_block[7];
     h = mpn_umul_ppmm (&l, h, l); h ^= s->xp_block[8]; l ^= s->yp_block[8];
    h = mpn_umul_ppmm (&l, h, l);  h ^= s->xp_block[9]; l ^= s->yp_block[9];
  }
  SPEED_MACRO_UMUL_PPMM_B;
  {
    h = mpn_umul_ppmm (&l, h, l);
     h = mpn_umul_ppmm (&l, h, l);
     h = mpn_umul_ppmm (&l, h, l);
    h = mpn_umul_ppmm (&l, h, l);
     h = mpn_umul_ppmm (&l, h, l);
     h = mpn_umul_ppmm (&l, h, l);
    h = mpn_umul_ppmm (&l, h, l);
     h = mpn_umul_ppmm (&l, h, l);
     h = mpn_umul_ppmm (&l, h, l);
    h = mpn_umul_ppmm (&l, h, l);
  }
  SPEED_MACRO_UMUL_PPMM_C;
}
#endif

#if HAVE_NATIVE_mpn_umul_ppmm_r
double
speed_mpn_umul_ppmm_r (struct speed_params *s)
{
  SPEED_MACRO_UMUL_PPMM_A;
  {
    h = mpn_umul_ppmm_r (h, l, &l);  h ^= s->xp_block[0]; l ^= s->yp_block[0];
     h = mpn_umul_ppmm_r (h, l, &l); h ^= s->xp_block[1]; l ^= s->yp_block[1];
     h = mpn_umul_ppmm_r (h, l, &l); h ^= s->xp_block[2]; l ^= s->yp_block[2];
    h = mpn_umul_ppmm_r (h, l, &l);  h ^= s->xp_block[3]; l ^= s->yp_block[3];
     h = mpn_umul_ppmm_r (h, l, &l); h ^= s->xp_block[4]; l ^= s->yp_block[4];
     h = mpn_umul_ppmm_r (h, l, &l); h ^= s->xp_block[5]; l ^= s->yp_block[5];
    h = mpn_umul_ppmm_r (h, l, &l);  h ^= s->xp_block[6]; l ^= s->yp_block[6];
     h = mpn_umul_ppmm_r (h, l, &l); h ^= s->xp_block[7]; l ^= s->yp_block[7];
     h = mpn_umul_ppmm_r (h, l, &l); h ^= s->xp_block[8]; l ^= s->yp_block[8];
    h = mpn_umul_ppmm_r (h, l, &l);  h ^= s->xp_block[9]; l ^= s->yp_block[9];
  }
  SPEED_MACRO_UMUL_PPMM_B;
  {
    h = mpn_umul_ppmm_r (h, l, &l);
     h = mpn_umul_ppmm_r (h, l, &l);
     h = mpn_umul_ppmm_r (h, l, &l);
    h = mpn_umul_ppmm_r (h, l, &l);
     h = mpn_umul_ppmm_r (h, l, &l);
     h = mpn_umul_ppmm_r (h, l, &l);
    h = mpn_umul_ppmm_r (h, l, &l);
     h = mpn_umul_ppmm_r (h, l, &l);
     h = mpn_umul_ppmm_r (h, l, &l);
    h = mpn_umul_ppmm_r (h, l, &l);
  }
  SPEED_MACRO_UMUL_PPMM_C;
}
#endif


/* The divisions are successively dependent so latency is measured, not
   issue rate.  There's only 10 per loop so the code doesn't get too big,
   especially for udiv_qrnnd_preinv and preinv2norm, which are several
   instructions each.

   Note that it's only the division which is measured here, there's no data
   fetching and no shifting if the divisor gets normalized.

   In speed_udiv_qrnnd with gcc 2.95.2 on x86 the parameters "q,r,r,q,d"
   generate x86 div instructions with nothing in between.

   The measuring function macros are in two parts to avoid overflowing
   preprocessor expansion space if udiv_qrnnd etc are big.

   Limitations:

   Don't blindly use this to set UDIV_TIME in gmp-mparam.h, check the code
   generated first.

   CPUs with data-dependent divisions may want more attention paid to the
   randomness of the data used.  Probably the measurement wanted is over
   uniformly distributed numbers, but what's here might not be giving that.  */

#define SPEED_ROUTINE_UDIV_QRNND_A(normalize)           \
  {                                                     \
    double     t;                                       \
    unsigned   i;                                       \
    mp_limb_t  q, r, d;                                 \
    mp_limb_t  dinv;                                    \
							\
    s->time_divisor = 10;                               \
							\
    /* divisor from "r" parameter, or a default */      \
    d = s->r;                                           \
    if (d == 0)                                         \
      d = mp_bases[10].big_base;                        \
							\
    if (normalize)                                      \
      {                                                 \
	unsigned  norm;                                 \
	count_leading_zeros (norm, d);                  \
	d <<= norm;                                     \
	invert_limb (dinv, d);                          \
      }                                                 \
							\
    q = s->xp[0];                                       \
    r = s->yp[0] % d;                                   \
							\
    speed_starttime ();                                 \
    i = s->reps;                                        \
    do                                                  \
      {

#define SPEED_ROUTINE_UDIV_QRNND_B                                      \
      }                                                                 \
    while (--i != 0);                                                   \
    t = speed_endtime ();                                               \
									\
    /* stop the compiler optimizing away the whole calculation! */      \
    noop_1 (q);                                                         \
    noop_1 (r);                                                         \
									\
    return t;                                                           \
  }

double
speed_udiv_qrnnd (struct speed_params *s)
{
  SPEED_ROUTINE_UDIV_QRNND_A (UDIV_NEEDS_NORMALIZATION);
  {
    udiv_qrnnd (q, r, r, q, d);
     udiv_qrnnd (q, r, r, q, d);
     udiv_qrnnd (q, r, r, q, d);
    udiv_qrnnd (q, r, r, q, d);
     udiv_qrnnd (q, r, r, q, d);
     udiv_qrnnd (q, r, r, q, d);
    udiv_qrnnd (q, r, r, q, d);
     udiv_qrnnd (q, r, r, q, d);
     udiv_qrnnd (q, r, r, q, d);
    udiv_qrnnd (q, r, r, q, d);
  }
  SPEED_ROUTINE_UDIV_QRNND_B;
}

double
speed_udiv_qrnnd_c (struct speed_params *s)
{
  SPEED_ROUTINE_UDIV_QRNND_A (1);
  {
    __udiv_qrnnd_c (q, r, r, q, d);
     __udiv_qrnnd_c (q, r, r, q, d);
     __udiv_qrnnd_c (q, r, r, q, d);
    __udiv_qrnnd_c (q, r, r, q, d);
     __udiv_qrnnd_c (q, r, r, q, d);
     __udiv_qrnnd_c (q, r, r, q, d);
    __udiv_qrnnd_c (q, r, r, q, d);
     __udiv_qrnnd_c (q, r, r, q, d);
     __udiv_qrnnd_c (q, r, r, q, d);
    __udiv_qrnnd_c (q, r, r, q, d);
  }
  SPEED_ROUTINE_UDIV_QRNND_B;
}

#if HAVE_NATIVE_mpn_udiv_qrnnd
double
speed_mpn_udiv_qrnnd (struct speed_params *s)
{
  SPEED_ROUTINE_UDIV_QRNND_A (1);
  {
    q = mpn_udiv_qrnnd (&r, r, q, d);
     q = mpn_udiv_qrnnd (&r, r, q, d);
     q = mpn_udiv_qrnnd (&r, r, q, d);
    q = mpn_udiv_qrnnd (&r, r, q, d);
     q = mpn_udiv_qrnnd (&r, r, q, d);
     q = mpn_udiv_qrnnd (&r, r, q, d);
    q = mpn_udiv_qrnnd (&r, r, q, d);
     q = mpn_udiv_qrnnd (&r, r, q, d);
     q = mpn_udiv_qrnnd (&r, r, q, d);
    q = mpn_udiv_qrnnd (&r, r, q, d);
  }
  SPEED_ROUTINE_UDIV_QRNND_B;
}
#endif

#if HAVE_NATIVE_mpn_udiv_qrnnd_r
double
speed_mpn_udiv_qrnnd_r (struct speed_params *s)
{
  SPEED_ROUTINE_UDIV_QRNND_A (1);
  {
    q = mpn_udiv_qrnnd_r (r, q, d, &r);
     q = mpn_udiv_qrnnd_r (r, q, d, &r);
     q = mpn_udiv_qrnnd_r (r, q, d, &r);
    q = mpn_udiv_qrnnd_r (r, q, d, &r);
     q = mpn_udiv_qrnnd_r (r, q, d, &r);
     q = mpn_udiv_qrnnd_r (r, q, d, &r);
    q = mpn_udiv_qrnnd_r (r, q, d, &r);
     q = mpn_udiv_qrnnd_r (r, q, d, &r);
     q = mpn_udiv_qrnnd_r (r, q, d, &r);
    q = mpn_udiv_qrnnd_r (r, q, d, &r);
  }
  SPEED_ROUTINE_UDIV_QRNND_B;
}
#endif


double
speed_invert_limb (struct speed_params *s)
{
  SPEED_ROUTINE_INVERT_LIMB_CALL (invert_limb (dinv, d));
}


/* xp[0] might not be particularly random, but should give an indication how
   "/" runs.  Same for speed_operator_mod below.  */
double
speed_operator_div (struct speed_params *s)
{
  double     t;
  unsigned   i;
  mp_limb_t  x, q, d;

  s->time_divisor = 10;

  /* divisor from "r" parameter, or a default */
  d = s->r;
  if (d == 0)
    d = mp_bases[10].big_base;

  x = s->xp[0];
  q = 0;

  speed_starttime ();
  i = s->reps;
  do
    {
      q ^= x; q /= d;
       q ^= x; q /= d;
       q ^= x; q /= d;
      q ^= x; q /= d;
       q ^= x; q /= d;
       q ^= x; q /= d;
      q ^= x; q /= d;
       q ^= x; q /= d;
       q ^= x; q /= d;
      q ^= x; q /= d;
    }
  while (--i != 0);
  t = speed_endtime ();

  /* stop the compiler optimizing away the whole calculation! */
  noop_1 (q);

  return t;
}

double
speed_operator_mod (struct speed_params *s)
{
  double     t;
  unsigned   i;
  mp_limb_t  x, r, d;

  s->time_divisor = 10;

  /* divisor from "r" parameter, or a default */
  d = s->r;
  if (d == 0)
    d = mp_bases[10].big_base;

  x = s->xp[0];
  r = 0;

  speed_starttime ();
  i = s->reps;
  do
    {
      r ^= x; r %= d;
       r ^= x; r %= d;
       r ^= x; r %= d;
      r ^= x; r %= d;
       r ^= x; r %= d;
       r ^= x; r %= d;
      r ^= x; r %= d;
       r ^= x; r %= d;
       r ^= x; r %= d;
      r ^= x; r %= d;
    }
  while (--i != 0);
  t = speed_endtime ();

  /* stop the compiler optimizing away the whole calculation! */
  noop_1 (r);

  return t;
}


/* r==0 measures on data with the values uniformly distributed.  This will
   be typical for count_trailing_zeros in a GCD etc.

   r==1 measures on data with the resultant count uniformly distributed
   between 0 and GMP_LIMB_BITS-1.  This is probably sensible for
   count_leading_zeros on the high limbs of divisors.  */

int
speed_routine_count_zeros_setup (struct speed_params *s,
				 mp_ptr xp, int leading, int zero)
{
  int        i, c;
  mp_limb_t  n;

  if (s->r == 0)
    {
      /* Make uniformly distributed data.  If zero isn't allowed then change
	 it to 1 for leading, or 0x800..00 for trailing.  */
      MPN_COPY (xp, s->xp_block, SPEED_BLOCK_SIZE);
      if (! zero)
	for (i = 0; i < SPEED_BLOCK_SIZE; i++)
	  if (xp[i] == 0)
	    xp[i] = leading ? 1 : GMP_LIMB_HIGHBIT;
    }
  else if (s->r == 1)
    {
      /* Make counts uniformly distributed.  A randomly chosen bit is set, and
	 for leading the rest above it are cleared, or for trailing then the
	 rest below.  */
      for (i = 0; i < SPEED_BLOCK_SIZE; i++)
	{
	  mp_limb_t  set = CNST_LIMB(1) << (s->yp_block[i] % GMP_LIMB_BITS);
	  mp_limb_t  keep_below = set-1;
	  mp_limb_t  keep_above = MP_LIMB_T_MAX ^ keep_below;
	  mp_limb_t  keep = (leading ? keep_below : keep_above);
	  xp[i] = (s->xp_block[i] & keep) | set;
	}
    }
  else
    {
      return 0;
    }

  /* Account for the effect of n^=c. */
  c = 0;
  for (i = 0; i < SPEED_BLOCK_SIZE; i++)
    {
      n = xp[i];
      xp[i] ^= c;

      if (leading)
	count_leading_zeros (c, n);
      else
	count_trailing_zeros (c, n);
    }

  return 1;
}

double
speed_count_leading_zeros (struct speed_params *s)
{
#ifdef COUNT_LEADING_ZEROS_0
#define COUNT_LEADING_ZEROS_0_ALLOWED   1
#else
#define COUNT_LEADING_ZEROS_0_ALLOWED   0
#endif

  SPEED_ROUTINE_COUNT_ZEROS_A (1, COUNT_LEADING_ZEROS_0_ALLOWED);
  count_leading_zeros (c, n);
  SPEED_ROUTINE_COUNT_ZEROS_B ();
}
double
speed_count_trailing_zeros (struct speed_params *s)
{
  SPEED_ROUTINE_COUNT_ZEROS_A (0, 0);
  count_trailing_zeros (c, n);
  SPEED_ROUTINE_COUNT_ZEROS_B ();
}


double
speed_mpn_get_str (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_GET_STR (mpn_get_str);
}

double
speed_mpn_set_str (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_SET_STR_CALL (mpn_set_str (wp, xp, s->size, base));
}
double
speed_mpn_bc_set_str (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_SET_STR_CALL (mpn_bc_set_str (wp, xp, s->size, base));
}

double
speed_MPN_ZERO (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_ZERO_CALL (MPN_ZERO (wp, s->size));
}


int
speed_randinit (struct speed_params *s, gmp_randstate_ptr rstate)
{
  if (s->r == 0)
    gmp_randinit_default (rstate);
  else if (s->r == 1)
    gmp_randinit_mt (rstate);
  else
    {
      return gmp_randinit_lc_2exp_size (rstate, s->r);
    }
  return 1;
}

double
speed_gmp_randseed (struct speed_params *s)
{
  gmp_randstate_t  rstate;
  unsigned  i;
  double    t;
  mpz_t     x;

  SPEED_RESTRICT_COND (s->size >= 1);
  SPEED_RESTRICT_COND (speed_randinit (s, rstate));

  /* s->size bits of seed */
  mpz_init_set_n (x, s->xp, s->size);
  mpz_fdiv_r_2exp (x, x, (unsigned long) s->size);

  /* cache priming */
  gmp_randseed (rstate, x);

  speed_starttime ();
  i = s->reps;
  do
    gmp_randseed (rstate, x);
  while (--i != 0);
  t = speed_endtime ();

  gmp_randclear (rstate);
  mpz_clear (x);
  return t;
}

double
speed_gmp_randseed_ui (struct speed_params *s)
{
  gmp_randstate_t  rstate;
  unsigned  i, j;
  double    t;

  SPEED_RESTRICT_COND (speed_randinit (s, rstate));

  /* cache priming */
  gmp_randseed_ui (rstate, 123L);

  speed_starttime ();
  i = s->reps;
  j = 0;
  do
    {
      gmp_randseed_ui (rstate, (unsigned long) s->xp_block[j]);
      j++;
      if (j >= SPEED_BLOCK_SIZE)
	j = 0;
    }
  while (--i != 0);
  t = speed_endtime ();

  gmp_randclear (rstate);
  return t;
}

double
speed_mpz_urandomb (struct speed_params *s)
{
  gmp_randstate_t  rstate;
  mpz_t     z;
  unsigned  i;
  double    t;

  SPEED_RESTRICT_COND (s->size >= 0);
  SPEED_RESTRICT_COND (speed_randinit (s, rstate));

  mpz_init (z);

  /* cache priming */
  mpz_urandomb (z, rstate, (unsigned long) s->size);
  mpz_urandomb (z, rstate, (unsigned long) s->size);

  speed_starttime ();
  i = s->reps;
  do
    mpz_urandomb (z, rstate, (unsigned long) s->size);
  while (--i != 0);
  t = speed_endtime ();

  mpz_clear (z);
  gmp_randclear (rstate);
  return t;
}
