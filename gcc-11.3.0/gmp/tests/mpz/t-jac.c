/* Exercise mpz_*_kronecker_*() and mpz_jacobi() functions.

Copyright 1999-2004, 2013 Free Software Foundation, Inc.

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


/* With no arguments the various Kronecker/Jacobi symbol routines are
   checked against some test data and a lot of derived data.

   To check the test data against PARI-GP, run

	   t-jac -p | gp -q

   It takes a while because the output from "t-jac -p" is big.


   Enhancements:

   More big test cases than those given by check_squares_zi would be good.  */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"

#ifdef _LONG_LONG_LIMB
#define LL(l,ll)  ll
#else
#define LL(l,ll)  l
#endif


int option_pari = 0;


unsigned long
mpz_mod4 (mpz_srcptr z)
{
  mpz_t          m;
  unsigned long  ret;

  mpz_init (m);
  mpz_fdiv_r_2exp (m, z, 2);
  ret = mpz_get_ui (m);
  mpz_clear (m);
  return ret;
}

int
mpz_fits_ulimb_p (mpz_srcptr z)
{
  return (SIZ(z) == 1 || SIZ(z) == 0);
}

mp_limb_t
mpz_get_ulimb (mpz_srcptr z)
{
  if (SIZ(z) == 0)
    return 0;
  else
    return PTR(z)[0];
}


void
try_base (mp_limb_t a, mp_limb_t b, int answer)
{
  int  got;

  if ((b & 1) == 0 || b == 1 || a > b)
    return;

  got = mpn_jacobi_base (a, b, 0);
  if (got != answer)
    {
      printf (LL("mpn_jacobi_base (%lu, %lu) is %d should be %d\n",
		 "mpn_jacobi_base (%llu, %llu) is %d should be %d\n"),
	      a, b, got, answer);
      abort ();
    }
}


void
try_zi_ui (mpz_srcptr a, unsigned long b, int answer)
{
  int  got;

  got = mpz_kronecker_ui (a, b);
  if (got != answer)
    {
      printf ("mpz_kronecker_ui (");
      mpz_out_str (stdout, 10, a);
      printf (", %lu) is %d should be %d\n", b, got, answer);
      abort ();
    }
}


void
try_zi_si (mpz_srcptr a, long b, int answer)
{
  int  got;

  got = mpz_kronecker_si (a, b);
  if (got != answer)
    {
      printf ("mpz_kronecker_si (");
      mpz_out_str (stdout, 10, a);
      printf (", %ld) is %d should be %d\n", b, got, answer);
      abort ();
    }
}


void
try_ui_zi (unsigned long a, mpz_srcptr b, int answer)
{
  int  got;

  got = mpz_ui_kronecker (a, b);
  if (got != answer)
    {
      printf ("mpz_ui_kronecker (%lu, ", a);
      mpz_out_str (stdout, 10, b);
      printf (") is %d should be %d\n", got, answer);
      abort ();
    }
}


void
try_si_zi (long a, mpz_srcptr b, int answer)
{
  int  got;

  got = mpz_si_kronecker (a, b);
  if (got != answer)
    {
      printf ("mpz_si_kronecker (%ld, ", a);
      mpz_out_str (stdout, 10, b);
      printf (") is %d should be %d\n", got, answer);
      abort ();
    }
}


/* Don't bother checking mpz_jacobi, since it only differs for b even, and
   we don't have an actual expected answer for it.  tests/devel/try.c does
   some checks though.  */
void
try_zi_zi (mpz_srcptr a, mpz_srcptr b, int answer)
{
  int  got;

  got = mpz_kronecker (a, b);
  if (got != answer)
    {
      printf ("mpz_kronecker (");
      mpz_out_str (stdout, 10, a);
      printf (", ");
      mpz_out_str (stdout, 10, b);
      printf (") is %d should be %d\n", got, answer);
      abort ();
    }
}


void
try_pari (mpz_srcptr a, mpz_srcptr b, int answer)
{
  printf ("try(");
  mpz_out_str (stdout, 10, a);
  printf (",");
  mpz_out_str (stdout, 10, b);
  printf (",%d)\n", answer);
}


void
try_each (mpz_srcptr a, mpz_srcptr b, int answer)
{
#if 0
  fprintf(stderr, "asize = %d, bsize = %d\n",
	  mpz_sizeinbase (a, 2), mpz_sizeinbase (b, 2));
#endif
  if (option_pari)
    {
      try_pari (a, b, answer);
      return;
    }

  if (mpz_fits_ulimb_p (a) && mpz_fits_ulimb_p (b))
    try_base (mpz_get_ulimb (a), mpz_get_ulimb (b), answer);

  if (mpz_fits_ulong_p (b))
    try_zi_ui (a, mpz_get_ui (b), answer);

  if (mpz_fits_slong_p (b))
    try_zi_si (a, mpz_get_si (b), answer);

  if (mpz_fits_ulong_p (a))
    try_ui_zi (mpz_get_ui (a), b, answer);

  if (mpz_fits_sint_p (a))
    try_si_zi (mpz_get_si (a), b, answer);

  try_zi_zi (a, b, answer);
}


/* Try (a/b) and (a/-b). */
void
try_pn (mpz_srcptr a, mpz_srcptr b_orig, int answer)
{
  mpz_t  b;

  mpz_init_set (b, b_orig);
  try_each (a, b, answer);

  mpz_neg (b, b);
  if (mpz_sgn (a) < 0)
    answer = -answer;

  try_each (a, b, answer);

  mpz_clear (b);
}


/* Try (a+k*p/b) for various k, using the fact (a/b) is periodic in a with
   period p.  For b>0, p=b if b!=2mod4 or p=4*b if b==2mod4. */

void
try_periodic_num (mpz_srcptr a_orig, mpz_srcptr b, int answer)
{
  mpz_t  a, a_period;
  int    i;

  if (mpz_sgn (b) <= 0)
    return;

  mpz_init_set (a, a_orig);
  mpz_init_set (a_period, b);
  if (mpz_mod4 (b) == 2)
    mpz_mul_ui (a_period, a_period, 4);

  /* don't bother with these tests if they're only going to produce
     even/even */
  if (mpz_even_p (a) && mpz_even_p (b) && mpz_even_p (a_period))
    goto done;

  for (i = 0; i < 6; i++)
    {
      mpz_add (a, a, a_period);
      try_pn (a, b, answer);
    }

  mpz_set (a, a_orig);
  for (i = 0; i < 6; i++)
    {
      mpz_sub (a, a, a_period);
      try_pn (a, b, answer);
    }

 done:
  mpz_clear (a);
  mpz_clear (a_period);
}


/* Try (a/b+k*p) for various k, using the fact (a/b) is periodic in b of
   period p.

			       period p
	   a==0,1mod4             a
	   a==2mod4              4*a
	   a==3mod4 and b odd    4*a
	   a==3mod4 and b even   8*a

   In Henri Cohen's book the period is given as 4*a for all a==2,3mod4, but
   a counterexample would seem to be (3/2)=-1 which with (3/14)=+1 doesn't
   have period 4*a (but rather 8*a with (3/26)=-1).  Maybe the plain 4*a is
   to be read as applying to a plain Jacobi symbol with b odd, rather than
   the Kronecker extension to b even. */

void
try_periodic_den (mpz_srcptr a, mpz_srcptr b_orig, int answer)
{
  mpz_t  b, b_period;
  int    i;

  if (mpz_sgn (a) == 0 || mpz_sgn (b_orig) == 0)
    return;

  mpz_init_set (b, b_orig);

  mpz_init_set (b_period, a);
  if (mpz_mod4 (a) == 3 && mpz_even_p (b))
    mpz_mul_ui (b_period, b_period, 8L);
  else if (mpz_mod4 (a) >= 2)
    mpz_mul_ui (b_period, b_period, 4L);

  /* don't bother with these tests if they're only going to produce
     even/even */
  if (mpz_even_p (a) && mpz_even_p (b) && mpz_even_p (b_period))
    goto done;

  for (i = 0; i < 6; i++)
    {
      mpz_add (b, b, b_period);
      try_pn (a, b, answer);
    }

  mpz_set (b, b_orig);
  for (i = 0; i < 6; i++)
    {
      mpz_sub (b, b, b_period);
      try_pn (a, b, answer);
    }

 done:
  mpz_clear (b);
  mpz_clear (b_period);
}


static const unsigned long  ktable[] = {
  0, 1, 2, 3, 4, 5, 6, 7,
  GMP_NUMB_BITS-1, GMP_NUMB_BITS, GMP_NUMB_BITS+1,
  2*GMP_NUMB_BITS-1, 2*GMP_NUMB_BITS, 2*GMP_NUMB_BITS+1,
  3*GMP_NUMB_BITS-1, 3*GMP_NUMB_BITS, 3*GMP_NUMB_BITS+1
};


/* Try (a/b*2^k) for various k. */
void
try_2den (mpz_srcptr a, mpz_srcptr b_orig, int answer)
{
  mpz_t  b;
  int    kindex;
  int    answer_a2, answer_k;
  unsigned long k;

  /* don't bother when b==0 */
  if (mpz_sgn (b_orig) == 0)
    return;

  mpz_init_set (b, b_orig);

  /* (a/2) is 0 if a even, 1 if a==1 or 7 mod 8, -1 if a==3 or 5 mod 8 */
  answer_a2 = (mpz_even_p (a) ? 0
	       : (((SIZ(a) >= 0 ? PTR(a)[0] : -PTR(a)[0]) + 2) & 7) < 4 ? 1
	       : -1);

  for (kindex = 0; kindex < numberof (ktable); kindex++)
    {
      k = ktable[kindex];

      /* answer_k = answer*(answer_a2^k) */
      answer_k = (answer_a2 == 0 && k != 0 ? 0
		  : (k & 1) == 1 && answer_a2 == -1 ? -answer
		  : answer);

      mpz_mul_2exp (b, b_orig, k);
      try_pn (a, b, answer_k);
    }

  mpz_clear (b);
}


/* Try (a*2^k/b) for various k.  If it happens mpz_ui_kronecker() gets (2/b)
   wrong it will show up as wrong answers demanded. */
void
try_2num (mpz_srcptr a_orig, mpz_srcptr b, int answer)
{
  mpz_t  a;
  int    kindex;
  int    answer_2b, answer_k;
  unsigned long  k;

  /* don't bother when a==0 */
  if (mpz_sgn (a_orig) == 0)
    return;

  mpz_init (a);

  /* (2/b) is 0 if b even, 1 if b==1 or 7 mod 8, -1 if b==3 or 5 mod 8 */
  answer_2b = (mpz_even_p (b) ? 0
	       : (((SIZ(b) >= 0 ? PTR(b)[0] : -PTR(b)[0]) + 2) & 7) < 4 ? 1
	       : -1);

  for (kindex = 0; kindex < numberof (ktable); kindex++)
    {
      k = ktable[kindex];

      /* answer_k = answer*(answer_2b^k) */
      answer_k = (answer_2b == 0 && k != 0 ? 0
		  : (k & 1) == 1 && answer_2b == -1 ? -answer
		  : answer);

	mpz_mul_2exp (a, a_orig, k);
      try_pn (a, b, answer_k);
    }

  mpz_clear (a);
}


/* The try_2num() and try_2den() routines don't in turn call
   try_periodic_num() and try_periodic_den() because it hugely increases the
   number of tests performed, without obviously increasing coverage.

   Useful extra derived cases can be added here. */

void
try_all (mpz_t a, mpz_t b, int answer)
{
  try_pn (a, b, answer);
  try_periodic_num (a, b, answer);
  try_periodic_den (a, b, answer);
  try_2num (a, b, answer);
  try_2den (a, b, answer);
}


void
check_data (void)
{
  static const struct {
    const char  *a;
    const char  *b;
    int         answer;

  } data[] = {

    /* Note that the various derived checks in try_all() reduce the cases
       that need to be given here.  */

    /* some zeros */
    {  "0",  "0", 0 },
    {  "0",  "2", 0 },
    {  "0",  "6", 0 },
    {  "5",  "0", 0 },
    { "24", "60", 0 },

    /* (a/1) = 1, any a
       In particular note (0/1)=1 so that (a/b)=(a mod b/b). */
    { "0", "1", 1 },
    { "1", "1", 1 },
    { "2", "1", 1 },
    { "3", "1", 1 },
    { "4", "1", 1 },
    { "5", "1", 1 },

    /* (0/b) = 0, b != 1 */
    { "0",  "3", 0 },
    { "0",  "5", 0 },
    { "0",  "7", 0 },
    { "0",  "9", 0 },
    { "0", "11", 0 },
    { "0", "13", 0 },
    { "0", "15", 0 },

    /* (1/b) = 1 */
    { "1",  "1", 1 },
    { "1",  "3", 1 },
    { "1",  "5", 1 },
    { "1",  "7", 1 },
    { "1",  "9", 1 },
    { "1", "11", 1 },

    /* (-1/b) = (-1)^((b-1)/2) which is -1 for b==3 mod 4 */
    { "-1",  "1",  1 },
    { "-1",  "3", -1 },
    { "-1",  "5",  1 },
    { "-1",  "7", -1 },
    { "-1",  "9",  1 },
    { "-1", "11", -1 },
    { "-1", "13",  1 },
    { "-1", "15", -1 },
    { "-1", "17",  1 },
    { "-1", "19", -1 },

    /* (2/b) = (-1)^((b^2-1)/8) which is -1 for b==3,5 mod 8.
       try_2num() will exercise multiple powers of 2 in the numerator.  */
    { "2",  "1",  1 },
    { "2",  "3", -1 },
    { "2",  "5", -1 },
    { "2",  "7",  1 },
    { "2",  "9",  1 },
    { "2", "11", -1 },
    { "2", "13", -1 },
    { "2", "15",  1 },
    { "2", "17",  1 },

    /* (-2/b) = (-1)^((b^2-1)/8)*(-1)^((b-1)/2) which is -1 for b==5,7mod8.
       try_2num() will exercise multiple powers of 2 in the numerator, which
       will test that the shift in mpz_si_kronecker() uses unsigned not
       signed.  */
    { "-2",  "1",  1 },
    { "-2",  "3",  1 },
    { "-2",  "5", -1 },
    { "-2",  "7", -1 },
    { "-2",  "9",  1 },
    { "-2", "11",  1 },
    { "-2", "13", -1 },
    { "-2", "15", -1 },
    { "-2", "17",  1 },

    /* (a/2)=(2/a).
       try_2den() will exercise multiple powers of 2 in the denominator. */
    {  "3",  "2", -1 },
    {  "5",  "2", -1 },
    {  "7",  "2",  1 },
    {  "9",  "2",  1 },
    {  "11", "2", -1 },

    /* Harriet Griffin, "Elementary Theory of Numbers", page 155, various
       examples.  */
    {   "2", "135",  1 },
    { "135",  "19", -1 },
    {   "2",  "19", -1 },
    {  "19", "135",  1 },
    { "173", "135",  1 },
    {  "38", "135",  1 },
    { "135", "173",  1 },
    { "173",   "5", -1 },
    {   "3",   "5", -1 },
    {   "5", "173", -1 },
    { "173",   "3", -1 },
    {   "2",   "3", -1 },
    {   "3", "173", -1 },
    { "253",  "21",  1 },
    {   "1",  "21",  1 },
    {  "21", "253",  1 },
    {  "21",  "11", -1 },
    {  "-1",  "11", -1 },

    /* Griffin page 147 */
    {  "-1",  "17",  1 },
    {   "2",  "17",  1 },
    {  "-2",  "17",  1 },
    {  "-1",  "89",  1 },
    {   "2",  "89",  1 },

    /* Griffin page 148 */
    {  "89",  "11",  1 },
    {   "1",  "11",  1 },
    {  "89",   "3", -1 },
    {   "2",   "3", -1 },
    {   "3",  "89", -1 },
    {  "11",  "89",  1 },
    {  "33",  "89", -1 },

    /* H. Davenport, "The Higher Arithmetic", page 65, the quadratic
       residues and non-residues mod 19.  */
    {  "1", "19",  1 },
    {  "4", "19",  1 },
    {  "5", "19",  1 },
    {  "6", "19",  1 },
    {  "7", "19",  1 },
    {  "9", "19",  1 },
    { "11", "19",  1 },
    { "16", "19",  1 },
    { "17", "19",  1 },
    {  "2", "19", -1 },
    {  "3", "19", -1 },
    {  "8", "19", -1 },
    { "10", "19", -1 },
    { "12", "19", -1 },
    { "13", "19", -1 },
    { "14", "19", -1 },
    { "15", "19", -1 },
    { "18", "19", -1 },

    /* Residues and non-residues mod 13 */
    {  "0",  "13",  0 },
    {  "1",  "13",  1 },
    {  "2",  "13", -1 },
    {  "3",  "13",  1 },
    {  "4",  "13",  1 },
    {  "5",  "13", -1 },
    {  "6",  "13", -1 },
    {  "7",  "13", -1 },
    {  "8",  "13", -1 },
    {  "9",  "13",  1 },
    { "10",  "13",  1 },
    { "11",  "13", -1 },
    { "12",  "13",  1 },

    /* various */
    {  "5",   "7", -1 },
    { "15",  "17",  1 },
    { "67",  "89",  1 },

    /* special values inducing a==b==1 at the end of jac_or_kron() */
    { "0x10000000000000000000000000000000000000000000000001",
      "0x10000000000000000000000000000000000000000000000003", 1 },

    /* Test for previous bugs in jacobi_2. */
    { "0x43900000000", "0x42400000439", -1 }, /* 32-bit limbs */
    { "0x4390000000000000000", "0x4240000000000000439", -1 }, /* 64-bit limbs */

    { "198158408161039063", "198158360916398807", -1 },

    /* Some tests involving large quotients in the continued fraction
       expansion. */
    { "37200210845139167613356125645445281805",
      "451716845976689892447895811408978421929", -1 },
    { "67674091930576781943923596701346271058970643542491743605048620644676477275152701774960868941561652032482173612421015",
      "4902678867794567120224500687210807069172039735", 0 },
    { "2666617146103764067061017961903284334497474492754652499788571378062969111250584288683585223600172138551198546085281683283672592", "2666617146103764067061017961903284334497474492754652499788571378062969111250584288683585223600172138551198546085281683290481773", 1 },

    /* Exercises the case asize == 1, btwos > 0 in mpz_jacobi. */
    { "804609", "421248363205206617296534688032638102314410556521742428832362659824", 1 } ,
    { "4190209", "2239744742177804210557442048984321017460028974602978995388383905961079286530650825925074203175536427000", 1 },

    /* Exercises the case asize == 1, btwos = 63 in mpz_jacobi
       (relevant when GMP_LIMB_BITS == 64). */
    { "17311973299000934401", "1675975991242824637446753124775689449936871337036614677577044717424700351103148799107651171694863695242089956242888229458836426332300124417011114380886016", 1 },
    { "3220569220116583677", "41859917623035396746", -1 },

    /* Other test cases that triggered bugs during development. */
    { "37200210845139167613356125645445281805", "340116213441272389607827434472642576514", -1 },
    { "74400421690278335226712251290890563610", "451716845976689892447895811408978421929", -1 },
  };

  int    i;
  mpz_t  a, b;

  mpz_init (a);
  mpz_init (b);

  for (i = 0; i < numberof (data); i++)
    {
      mpz_set_str_or_abort (a, data[i].a, 0);
      mpz_set_str_or_abort (b, data[i].b, 0);
      try_all (a, b, data[i].answer);
    }

  mpz_clear (a);
  mpz_clear (b);
}


/* (a^2/b)=1 if gcd(a,b)=1, or (a^2/b)=0 if gcd(a,b)!=1.
   This includes when a=0 or b=0. */
void
check_squares_zi (void)
{
  gmp_randstate_ptr rands = RANDS;
  mpz_t  a, b, g;
  int    i, answer;
  mp_size_t size_range, an, bn;
  mpz_t bs;

  mpz_init (bs);
  mpz_init (a);
  mpz_init (b);
  mpz_init (g);

  for (i = 0; i < 50; i++)
    {
      mpz_urandomb (bs, rands, 32);
      size_range = mpz_get_ui (bs) % 10 + i/8 + 2;

      mpz_urandomb (bs, rands, size_range);
      an = mpz_get_ui (bs);
      mpz_rrandomb (a, rands, an);

      mpz_urandomb (bs, rands, size_range);
      bn = mpz_get_ui (bs);
      mpz_rrandomb (b, rands, bn);

      mpz_gcd (g, a, b);
      if (mpz_cmp_ui (g, 1L) == 0)
	answer = 1;
      else
	answer = 0;

      mpz_mul (a, a, a);

      try_all (a, b, answer);
    }

  mpz_clear (bs);
  mpz_clear (a);
  mpz_clear (b);
  mpz_clear (g);
}


/* Check the handling of asize==0, make sure it isn't affected by the low
   limb. */
void
check_a_zero (void)
{
  mpz_t  a, b;

  mpz_init_set_ui (a, 0);
  mpz_init (b);

  mpz_set_ui (b, 1L);
  PTR(a)[0] = 0;
  try_all (a, b, 1);   /* (0/1)=1 */
  PTR(a)[0] = 1;
  try_all (a, b, 1);   /* (0/1)=1 */

  mpz_set_si (b, -1L);
  PTR(a)[0] = 0;
  try_all (a, b, 1);   /* (0/-1)=1 */
  PTR(a)[0] = 1;
  try_all (a, b, 1);   /* (0/-1)=1 */

  mpz_set_ui (b, 0);
  PTR(a)[0] = 0;
  try_all (a, b, 0);   /* (0/0)=0 */
  PTR(a)[0] = 1;
  try_all (a, b, 0);   /* (0/0)=0 */

  mpz_set_ui (b, 2);
  PTR(a)[0] = 0;
  try_all (a, b, 0);   /* (0/2)=0 */
  PTR(a)[0] = 1;
  try_all (a, b, 0);   /* (0/2)=0 */

  mpz_clear (a);
  mpz_clear (b);
}


/* Assumes that b = prod p_k^e_k */
int
ref_jacobi (mpz_srcptr a, mpz_srcptr b, unsigned nprime,
	    mpz_t prime[], unsigned *exp)
{
  unsigned i;
  int res;

  for (i = 0, res = 1; i < nprime; i++)
    if (exp[i])
      {
	int legendre = refmpz_legendre (a, prime[i]);
	if (!legendre)
	  return 0;
	if (exp[i] & 1)
	  res *= legendre;
      }
  return res;
}

void
check_jacobi_factored (void)
{
#define PRIME_N 10
#define PRIME_MAX_SIZE 50
#define PRIME_MAX_EXP 4
#define PRIME_A_COUNT 10
#define PRIME_B_COUNT 5
#define PRIME_MAX_B_SIZE 2000

  gmp_randstate_ptr rands = RANDS;
  mpz_t prime[PRIME_N];
  unsigned exp[PRIME_N];
  mpz_t a, b, t, bs;
  unsigned i;

  mpz_init (a);
  mpz_init (b);
  mpz_init (t);
  mpz_init (bs);

  /* Generate primes */
  for (i = 0; i < PRIME_N; i++)
    {
      mp_size_t size;
      mpz_init (prime[i]);
      mpz_urandomb (bs, rands, 32);
      size = mpz_get_ui (bs) % PRIME_MAX_SIZE + 2;
      mpz_rrandomb (prime[i], rands, size);
      if (mpz_cmp_ui (prime[i], 3) <= 0)
	mpz_set_ui (prime[i], 3);
      else
	mpz_nextprime (prime[i], prime[i]);
    }

  for (i = 0; i < PRIME_B_COUNT; i++)
    {
      unsigned j, k;
      mp_bitcnt_t bsize;

      mpz_set_ui (b, 1);
      bsize = 1;

      for (j = 0; j < PRIME_N && bsize < PRIME_MAX_B_SIZE; j++)
	{
	  mpz_urandomb (bs, rands, 32);
	  exp[j] = mpz_get_ui (bs) % PRIME_MAX_EXP;
	  mpz_pow_ui (t, prime[j], exp[j]);
	  mpz_mul (b, b, t);
	  bsize = mpz_sizeinbase (b, 2);
	}
      for (k = 0; k < PRIME_A_COUNT; k++)
	{
	  int answer;
	  mpz_rrandomb (a, rands, bsize + 2);
	  answer = ref_jacobi (a, b, j, prime, exp);
	  try_all (a, b, answer);
	}
    }
  for (i = 0; i < PRIME_N; i++)
    mpz_clear (prime[i]);

  mpz_clear (a);
  mpz_clear (b);
  mpz_clear (t);
  mpz_clear (bs);

#undef PRIME_N
#undef PRIME_MAX_SIZE
#undef PRIME_MAX_EXP
#undef PRIME_A_COUNT
#undef PRIME_B_COUNT
#undef PRIME_MAX_B_SIZE
}

/* These tests compute (a|n), where the quotient sequence includes
   large quotients, and n has a known factorization. Such inputs are
   generated as follows. First, construct a large n, as a power of a
   prime p of moderate size.

   Next, compute a matrix from factors (q,1;1,0), with q chosen with
   uniformly distributed size. We must stop with matrix elements of
   roughly half the size of n. Denote elements of M as M = (m00, m01;
   m10, m11).

   We now look for solutions to

     n = m00 x + m01 y
     a = m10 x + m11 y

   with x,y > 0. Since n >= m00 * m01, there exists a positive
   solution to the first equation. Find those x, y, and substitute in
   the second equation to get a. Then the quotient sequence for (a|n)
   is precisely the quotients used when constructing M, followed by
   the quotient sequence for (x|y).

   Numbers should also be large enough that we exercise hgcd_jacobi,
   which means that they should be larger than

     max (GCD_DC_THRESHOLD, 3 * HGCD_THRESHOLD)

   With an n of roughly 40000 bits, this should hold on most machines.
*/

void
check_large_quotients (void)
{
#define COUNT 50
#define PBITS 200
#define PPOWER 201
#define MAX_QBITS 500

  gmp_randstate_ptr rands = RANDS;

  mpz_t p, n, q, g, s, t, x, y, bs;
  mpz_t M[2][2];
  mp_bitcnt_t nsize;
  unsigned i;

  mpz_init (p);
  mpz_init (n);
  mpz_init (q);
  mpz_init (g);
  mpz_init (s);
  mpz_init (t);
  mpz_init (x);
  mpz_init (y);
  mpz_init (bs);
  mpz_init (M[0][0]);
  mpz_init (M[0][1]);
  mpz_init (M[1][0]);
  mpz_init (M[1][1]);

  /* First generate a number with known factorization, as a random
     smallish prime raised to an odd power. Then (a|n) = (a|p). */
  mpz_rrandomb (p, rands, PBITS);
  mpz_nextprime (p, p);
  mpz_pow_ui (n, p, PPOWER);

  nsize = mpz_sizeinbase (n, 2);

  for (i = 0; i < COUNT; i++)
    {
      int answer;
      mp_bitcnt_t msize;

      mpz_set_ui (M[0][0], 1);
      mpz_set_ui (M[0][1], 0);
      mpz_set_ui (M[1][0], 0);
      mpz_set_ui (M[1][1], 1);

      for (msize = 1; 2*(msize + MAX_QBITS) + 1 < nsize ;)
	{
	  unsigned i;
	  mpz_rrandomb (bs, rands, 32);
	  mpz_rrandomb (q, rands, 1 + mpz_get_ui (bs) % MAX_QBITS);

	  /* Multiply by (q, 1; 1,0) from the right */
	  for (i = 0; i < 2; i++)
	    {
	      mp_bitcnt_t size;
	      mpz_swap (M[i][0], M[i][1]);
	      mpz_addmul (M[i][0], M[i][1], q);
	      size = mpz_sizeinbase (M[i][0], 2);
	      if (size > msize)
		msize = size;
	    }
	}
      mpz_gcdext (g, s, t, M[0][0], M[0][1]);
      ASSERT_ALWAYS (mpz_cmp_ui (g, 1) == 0);

      /* Solve n = M[0][0] * x + M[0][1] * y */
      if (mpz_sgn (s) > 0)
	{
	  mpz_mul (x, n, s);
	  mpz_fdiv_qr (q, x, x, M[0][1]);
	  mpz_mul (y, q, M[0][0]);
	  mpz_addmul (y, t, n);
	  ASSERT_ALWAYS (mpz_sgn (y) > 0);
	}
      else
	{
	  mpz_mul (y, n, t);
	  mpz_fdiv_qr (q, y, y, M[0][0]);
	  mpz_mul (x, q, M[0][1]);
	  mpz_addmul (x, s, n);
	  ASSERT_ALWAYS (mpz_sgn (x) > 0);
	}
      mpz_mul (x, x, M[1][0]);
      mpz_addmul (x, y, M[1][1]);

      /* Now (x|n) has the selected large quotients */
      answer = refmpz_legendre (x, p);
      try_zi_zi (x, n, answer);
    }
  mpz_clear (p);
  mpz_clear (n);
  mpz_clear (q);
  mpz_clear (g);
  mpz_clear (s);
  mpz_clear (t);
  mpz_clear (x);
  mpz_clear (y);
  mpz_clear (bs);
  mpz_clear (M[0][0]);
  mpz_clear (M[0][1]);
  mpz_clear (M[1][0]);
  mpz_clear (M[1][1]);
#undef COUNT
#undef PBITS
#undef PPOWER
#undef MAX_QBITS
}

int
main (int argc, char *argv[])
{
  tests_start ();

  if (argc >= 2 && strcmp (argv[1], "-p") == 0)
    {
      option_pari = 1;

      printf ("\
try(a,b,answer) =\n\
{\n\
  if (kronecker(a,b) != answer,\n\
    print(\"wrong at \", a, \",\", b,\n\
      \" expected \", answer,\n\
      \" pari says \", kronecker(a,b)))\n\
}\n");
    }

  check_data ();
  check_squares_zi ();
  check_a_zero ();
  check_jacobi_factored ();
  check_large_quotients ();
  tests_end ();
  exit (0);
}
