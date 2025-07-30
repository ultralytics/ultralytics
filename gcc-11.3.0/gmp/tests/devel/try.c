/* Run some tests on various mpn routines.

   THIS IS A TEST PROGRAM USED ONLY FOR DEVELOPMENT.  IT'S ALMOST CERTAIN TO
   BE SUBJECT TO INCOMPATIBLE CHANGES IN FUTURE VERSIONS OF GMP.

Copyright 2000-2006, 2008, 2009, 2011, 2012 Free Software Foundation, Inc.

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


/* Usage: try [options] <function>...

   For example, "./try mpn_add_n" to run tests of that function.

   Combinations of alignments and overlaps are tested, with redzones above
   or below the destinations, and with the sources write-protected.

   The number of tests performed becomes ridiculously large with all the
   combinations, and for that reason this can't be a part of a "make check",
   it's meant only for development.  The code isn't very pretty either.

   During development it can help to disable the redzones, since seeing the
   rest of the destination written can show where the wrong part is, or if
   the dst pointers are off by 1 or whatever.  The magic DEADVAL initial
   fill (see below) will show locations never written.

   The -s option can be used to test only certain size operands, which is
   useful if some new code doesn't yet support say sizes less than the
   unrolling, or whatever.

   When a problem occurs it'll of course be necessary to run the program
   under gdb to find out quite where, how and why it's going wrong.  Disable
   the spinner with the -W option when doing this, or single stepping won't
   work.  Using the "-1" option to run with simple data can be useful.

   New functions to test can be added in try_array[].  If a new TYPE is
   required then add it to the existing constants, set up its parameters in
   param_init(), and add it to the call() function.  Extra parameter fields
   can be added if necessary, or further interpretations given to existing
   fields.


   Portability:

   This program is not designed for use on Cray vector systems under Unicos,
   it will fail to compile due to missing _SC_PAGE_SIZE.  Those systems
   don't really have pages or mprotect.  We could arrange to run the tests
   without the redzones, but we haven't bothered currently.


   Enhancements:

   umul_ppmm support is not very good, lots of source data is generated
   whereas only two limbs are needed.

   Make a little scheme for interpreting the "SIZE" selections uniformly.

   Make tr->size==SIZE_2 work, for the benefit of find_a which wants just 2
   source limbs.  Possibly increase the default repetitions in that case.

   Automatically detect gdb and disable the spinner (use -W for now).

   Make a way to re-run a failing case in the debugger.  Have an option to
   snapshot each test case before it's run so the data is available if a
   segv occurs.  (This should be more reliable than the current print_all()
   in the signal handler.)

   When alignment means a dst isn't hard against the redzone, check the
   space in between remains unchanged.

   When a source overlaps a destination, don't run both s[i].high 0 and 1,
   as s[i].high has no effect.  Maybe encode s[i].high into overlap->s[i].

   When partial overlaps aren't done, don't loop over source alignments
   during overlaps.

   Try to make the looping code a bit less horrible.  Right now it's pretty
   hard to see what iterations are actually done.

   Perhaps specific setups and loops for each style of function under test
   would be clearer than a parameterized general loop.  There's lots of
   stuff common to all functions, but the exceptions get messy.

   When there's no overlap, run with both src>dst and src<dst.  A subtle
   calling-conventions violation occurred in a P6 copy which depended on the
   relative location of src and dst.

   multiplier_N is more or less a third source region for the addmul_N
   routines, and could be done with the redzoned region scheme.

*/


/* always do assertion checking */
#define WANT_ASSERT 1

#include "config.h"

#include <errno.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if HAVE_UNISTD_H
#include <unistd.h>
#endif

#if HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif

#include "gmp.h"
#include "gmp-impl.h"
#include "longlong.h"
#include "tests.h"


#if !HAVE_DECL_OPTARG
extern char *optarg;
extern int optind, opterr;
#endif

#if ! HAVE_DECL_SYS_NERR
extern int sys_nerr;
#endif

#if ! HAVE_DECL_SYS_ERRLIST
extern char *sys_errlist[];
#endif

#if ! HAVE_STRERROR
char *
strerror (int n)
{
  if (n < 0 || n >= sys_nerr)
    return "errno out of range";
  else
    return sys_errlist[n];
}
#endif

/* Rumour has it some systems lack a define of PROT_NONE. */
#ifndef PROT_NONE
#define PROT_NONE   0
#endif

/* Dummy defines for when mprotect doesn't exist. */
#ifndef PROT_READ
#define PROT_READ   0
#endif
#ifndef PROT_WRITE
#define PROT_WRITE  0
#endif

/* _SC_PAGESIZE is standard, but hpux 9 and possibly other systems have
   _SC_PAGE_SIZE instead. */
#if defined (_SC_PAGE_SIZE) && ! defined (_SC_PAGESIZE)
#define _SC_PAGESIZE  _SC_PAGE_SIZE
#endif


#ifdef EXTRA_PROTOS
EXTRA_PROTOS
#endif
#ifdef EXTRA_PROTOS2
EXTRA_PROTOS2
#endif


#define DEFAULT_REPETITIONS  10

int  option_repetitions = DEFAULT_REPETITIONS;
int  option_spinner = 1;
int  option_redzones = 1;
int  option_firstsize = 0;
int  option_lastsize = 500;
int  option_firstsize2 = 0;

#define ALIGNMENTS          4
#define OVERLAPS            4
#define CARRY_RANDOMS       5
#define MULTIPLIER_RANDOMS  5
#define DIVISOR_RANDOMS     5
#define FRACTION_COUNT      4

int  option_print = 0;

#define DATA_TRAND  0
#define DATA_ZEROS  1
#define DATA_SEQ    2
#define DATA_FFS    3
#define DATA_2FD    4
int  option_data = DATA_TRAND;


mp_size_t  pagesize;
#define PAGESIZE_LIMBS  (pagesize / GMP_LIMB_BYTES)

/* must be a multiple of the page size */
#define REDZONE_BYTES   (pagesize * 16)
#define REDZONE_LIMBS   (REDZONE_BYTES / GMP_LIMB_BYTES)


#define MAX3(x,y,z)   (MAX (x, MAX (y, z)))

#if GMP_LIMB_BITS == 32
#define DEADVAL  CNST_LIMB(0xDEADBEEF)
#else
#define DEADVAL  CNST_LIMB(0xDEADBEEFBADDCAFE)
#endif


struct region_t {
  mp_ptr     ptr;
  mp_size_t  size;
};


#define TRAP_NOWHERE 0
#define TRAP_REF     1
#define TRAP_FUN     2
#define TRAP_SETUPS  3
int trap_location = TRAP_NOWHERE;


#define NUM_SOURCES  5
#define NUM_DESTS    2

struct source_t {
  struct region_t  region;
  int        high;
  mp_size_t  align;
  mp_ptr     p;
};

struct source_t  s[NUM_SOURCES];

struct dest_t {
  int        high;
  mp_size_t  align;
  mp_size_t  size;
};

struct dest_t  d[NUM_DESTS];

struct source_each_t {
  mp_ptr     p;
};

struct dest_each_t {
  struct region_t  region;
  mp_ptr     p;
};

mp_size_t       size;
mp_size_t       size2;
unsigned long   shift;
mp_limb_t       carry;
mp_limb_t       divisor;
mp_limb_t       multiplier;
mp_limb_t       multiplier_N[8];

struct each_t {
  const char  *name;
  struct dest_each_t    d[NUM_DESTS];
  struct source_each_t  s[NUM_SOURCES];
  mp_limb_t  retval;
};

struct each_t  ref = { "Ref" };
struct each_t  fun = { "Fun" };

#define SRC_SIZE(n)  ((n) == 1 && tr->size2 ? size2 : size)

void validate_fail (void);


#if HAVE_TRY_NEW_C
#include "try-new.c"
#endif


typedef mp_limb_t (*tryfun_t) (ANYARGS);

struct try_t {
  char  retval;

  char  src[NUM_SOURCES];
  char  dst[NUM_DESTS];

#define SIZE_YES          1
#define SIZE_ALLOW_ZERO   2
#define SIZE_1            3  /* 1 limb  */
#define SIZE_2            4  /* 2 limbs */
#define SIZE_3            5  /* 3 limbs */
#define SIZE_4            6  /* 4 limbs */
#define SIZE_6            7  /* 6 limbs */
#define SIZE_FRACTION     8  /* size2 is fraction for divrem etc */
#define SIZE_SIZE2        9
#define SIZE_PLUS_1      10
#define SIZE_SUM         11
#define SIZE_DIFF        12
#define SIZE_DIFF_PLUS_1 13
#define SIZE_DIFF_PLUS_3 14
#define SIZE_RETVAL      15
#define SIZE_CEIL_HALF   16
#define SIZE_GET_STR     17
#define SIZE_PLUS_MSIZE_SUB_1 18  /* size+msize-1 */
#define SIZE_ODD         19
  char  size;
  char  size2;
  char  dst_size[NUM_DESTS];

  /* multiplier_N size in limbs */
  mp_size_t  msize;

  char  dst_bytes[NUM_DESTS];

  char  dst0_from_src1;

#define CARRY_BIT     1  /* single bit 0 or 1 */
#define CARRY_3       2  /* 0, 1, 2 */
#define CARRY_4       3  /* 0 to 3 */
#define CARRY_LIMB    4  /* any limb value */
#define CARRY_DIVISOR 5  /* carry<divisor */
  char  carry;

  /* a fudge to tell the output when to print negatives */
  char  carry_sign;

  char  multiplier;
  char  shift;

#define DIVISOR_LIMB  1
#define DIVISOR_NORM  2
#define DIVISOR_ODD   3
  char  divisor;

#define DATA_NON_ZERO         1
#define DATA_GCD              2
#define DATA_SRC0_ODD         3
#define DATA_SRC0_HIGHBIT     4
#define DATA_SRC1_ODD         5
#define DATA_SRC1_ODD_PRIME   6
#define DATA_SRC1_HIGHBIT     7
#define DATA_MULTIPLE_DIVISOR 8
#define DATA_UDIV_QRNND       9
#define DATA_DIV_QR_1        10
  char  data;

/* Default is allow full overlap. */
#define OVERLAP_NONE         1
#define OVERLAP_LOW_TO_HIGH  2
#define OVERLAP_HIGH_TO_LOW  3
#define OVERLAP_NOT_SRCS     4
#define OVERLAP_NOT_SRC2     8
#define OVERLAP_NOT_DST2     16
  char  overlap;

  tryfun_t    reference;
  const char  *reference_name;

  void        (*validate) (void);
  const char  *validate_name;
};

struct try_t  *tr;


void
validate_mod_34lsub1 (void)
{
#define CNST_34LSUB1   ((CNST_LIMB(1) << (3 * (GMP_NUMB_BITS / 4))) - 1)

  mp_srcptr  ptr = s[0].p;
  int        error = 0;
  mp_limb_t  got, got_mod, want, want_mod;

  ASSERT (size >= 1);

  got = fun.retval;
  got_mod = got % CNST_34LSUB1;

  want = refmpn_mod_34lsub1 (ptr, size);
  want_mod = want % CNST_34LSUB1;

  if (got_mod != want_mod)
    {
      gmp_printf ("got   0x%MX reduced from 0x%MX\n", got_mod, got);
      gmp_printf ("want  0x%MX reduced from 0x%MX\n", want_mod, want);
      error = 1;
    }

  if (error)
    validate_fail ();
}

void
validate_divexact_1 (void)
{
  mp_srcptr  src = s[0].p;
  mp_srcptr  dst = fun.d[0].p;
  int  error = 0;

  ASSERT (size >= 1);

  {
    mp_ptr     tp = refmpn_malloc_limbs (size);
    mp_limb_t  rem;

    rem = refmpn_divrem_1 (tp, 0, src, size, divisor);
    if (rem != 0)
      {
	gmp_printf ("Remainder a%%d == 0x%MX, mpn_divexact_1 undefined\n", rem);
	error = 1;
      }
    if (! refmpn_equal_anynail (tp, dst, size))
      {
	printf ("Quotient a/d wrong\n");
	mpn_trace ("fun ", dst, size);
	mpn_trace ("want", tp, size);
	error = 1;
      }
    free (tp);
  }

  if (error)
    validate_fail ();
}

void
validate_bdiv_q_1
 (void)
{
  mp_srcptr  src = s[0].p;
  mp_srcptr  dst = fun.d[0].p;
  int  error = 0;

  ASSERT (size >= 1);

  {
    mp_ptr     tp = refmpn_malloc_limbs (size + 1);

    refmpn_mul_1 (tp, dst, size, divisor);
    /* Set ignored low bits */
    tp[0] |= (src[0] & LOW_ZEROS_MASK (divisor));
    if (! refmpn_equal_anynail (tp, src, size))
      {
	printf ("Bdiv wrong: res * divisor != src (mod B^size)\n");
	mpn_trace ("res ", dst, size);
	mpn_trace ("src ", src, size);
	error = 1;
      }
    free (tp);
  }

  if (error)
    validate_fail ();
}


void
validate_modexact_1c_odd (void)
{
  mp_srcptr  ptr = s[0].p;
  mp_limb_t  r = fun.retval;
  int  error = 0;

  ASSERT (size >= 1);
  ASSERT (divisor & 1);

  if ((r & GMP_NAIL_MASK) != 0)
    printf ("r has non-zero nail\n");

  if (carry < divisor)
    {
      if (! (r < divisor))
	{
	  printf ("Don't have r < divisor\n");
	  error = 1;
	}
    }
  else /* carry >= divisor */
    {
      if (! (r <= divisor))
	{
	  printf ("Don't have r <= divisor\n");
	  error = 1;
	}
    }

  {
    mp_limb_t  c = carry % divisor;
    mp_ptr     tp = refmpn_malloc_limbs (size+1);
    mp_size_t  k;

    for (k = size-1; k <= size; k++)
      {
	/* set {tp,size+1} to r*b^k + a - c */
	refmpn_copyi (tp, ptr, size);
	tp[size] = 0;
	ASSERT_NOCARRY (refmpn_add_1 (tp+k, tp+k, size+1-k, r));
	if (refmpn_sub_1 (tp, tp, size+1, c))
	  ASSERT_CARRY (mpn_add_1 (tp, tp, size+1, divisor));

	if (refmpn_mod_1 (tp, size+1, divisor) == 0)
	  goto good_remainder;
      }
    printf ("Remainder matches neither r*b^(size-1) nor r*b^size\n");
    error = 1;

  good_remainder:
    free (tp);
  }

  if (error)
    validate_fail ();
}

void
validate_modexact_1_odd (void)
{
  carry = 0;
  validate_modexact_1c_odd ();
}

void
validate_div_qr_1_pi1 (void)
{
  mp_srcptr up = ref.s[0].p;
  mp_size_t un = size;
  mp_size_t uh = ref.s[1].p[0];
  mp_srcptr qp = fun.d[0].p;
  mp_limb_t r = fun.retval;
  mp_limb_t cy;
  int cmp;
  mp_ptr tp;
  if (r >= divisor)
    {
      gmp_printf ("Bad remainder %Md, d = %Md\n", r, divisor);
      validate_fail ();
    }
  tp = refmpn_malloc_limbs (un);
  cy = refmpn_mul_1 (tp, qp, un, divisor);
  cy += refmpn_add_1 (tp, tp, un, r);
  if (cy != uh || refmpn_cmp (tp, up, un) != 0)
    {
      gmp_printf ("Incorrect result, size %ld.\n"
		  "d = %Mx, u = %Mx, %Nx\n"
		  "got: r = %Mx, q = %Nx\n"
		  "q d + r = %Mx, %Nx",
		  (long) un,
		  divisor, uh, up, un,
		  r, qp, un,
		  cy, tp, un);
      validate_fail ();
    }
  free (tp);
}


void
validate_sqrtrem (void)
{
  mp_srcptr  orig_ptr = s[0].p;
  mp_size_t  orig_size = size;
  mp_size_t  root_size = (size+1)/2;
  mp_srcptr  root_ptr = fun.d[0].p;
  mp_size_t  rem_size = fun.retval;
  mp_srcptr  rem_ptr = fun.d[1].p;
  mp_size_t  prod_size = 2*root_size;
  mp_ptr     p;
  int  error = 0;

  if (rem_size < 0 || rem_size > size)
    {
      printf ("Bad remainder size retval %ld\n", (long) rem_size);
      validate_fail ();
    }

  p = refmpn_malloc_limbs (prod_size);

  p[root_size] = refmpn_lshift (p, root_ptr, root_size, 1);
  if (refmpn_cmp_twosizes (p,root_size+1, rem_ptr,rem_size) < 0)
    {
      printf ("Remainder bigger than 2*root\n");
      error = 1;
    }

  refmpn_sqr (p, root_ptr, root_size);
  if (rem_size != 0)
    refmpn_add (p, p, prod_size, rem_ptr, rem_size);
  if (refmpn_cmp_twosizes (p,prod_size, orig_ptr,orig_size) != 0)
    {
      printf ("root^2+rem != original\n");
      mpn_trace ("prod", p, prod_size);
      error = 1;
    }
  free (p);

  if (error)
    validate_fail ();
}

void
validate_sqrt (void)
{
  mp_srcptr  orig_ptr = s[0].p;
  mp_size_t  orig_size = size;
  mp_size_t  root_size = (size+1)/2;
  mp_srcptr  root_ptr = fun.d[0].p;
  int        perf_pow = (fun.retval == 0);
  mp_size_t  prod_size = 2*root_size;
  mp_ptr     p;
  int  error = 0;

  p = refmpn_malloc_limbs (prod_size);

  refmpn_sqr (p, root_ptr, root_size);
  MPN_NORMALIZE (p, prod_size);
  if (refmpn_cmp_twosizes (p,prod_size, orig_ptr,orig_size) != - !perf_pow)
    {
      printf ("root^2 bigger than original, or wrong return value.\n");
      mpn_trace ("prod...", p, prod_size);
      error = 1;
    }

  refmpn_sub (p, orig_ptr,orig_size, p,prod_size);
  MPN_NORMALIZE (p, prod_size);
  if (prod_size >= root_size &&
      refmpn_sub (p, p,prod_size, root_ptr, root_size) == 0 &&
      refmpn_cmp_twosizes (p, prod_size, root_ptr, root_size) > 0)
    {
      printf ("(root+1)^2 smaller than original.\n");
      mpn_trace ("prod", p, prod_size);
      error = 1;
    }
  free (p);

  if (error)
    validate_fail ();
}


/* These types are indexes into the param[] array and are arbitrary so long
   as they're all distinct and within the size of param[].  Renumber
   whenever necessary or desired.  */

enum {
  TYPE_ADD = 1, TYPE_ADD_N, TYPE_ADD_NC, TYPE_SUB, TYPE_SUB_N, TYPE_SUB_NC,

  TYPE_ADD_ERR1_N, TYPE_ADD_ERR2_N, TYPE_ADD_ERR3_N,
  TYPE_SUB_ERR1_N, TYPE_SUB_ERR2_N, TYPE_SUB_ERR3_N,

  TYPE_MUL_1, TYPE_MUL_1C,

  TYPE_MUL_2, TYPE_MUL_3, TYPE_MUL_4, TYPE_MUL_5, TYPE_MUL_6,

  TYPE_ADDMUL_1, TYPE_ADDMUL_1C, TYPE_SUBMUL_1, TYPE_SUBMUL_1C,

  TYPE_ADDMUL_2, TYPE_ADDMUL_3, TYPE_ADDMUL_4, TYPE_ADDMUL_5, TYPE_ADDMUL_6,
  TYPE_ADDMUL_7, TYPE_ADDMUL_8,

  TYPE_ADDSUB_N, TYPE_ADDSUB_NC,

  TYPE_RSHIFT, TYPE_LSHIFT, TYPE_LSHIFTC,

  TYPE_COPY, TYPE_COPYI, TYPE_COPYD, TYPE_COM,

  TYPE_ADDLSH1_N, TYPE_ADDLSH2_N, TYPE_ADDLSH_N,
  TYPE_ADDLSH1_N_IP1, TYPE_ADDLSH2_N_IP1, TYPE_ADDLSH_N_IP1,
  TYPE_ADDLSH1_N_IP2, TYPE_ADDLSH2_N_IP2, TYPE_ADDLSH_N_IP2,
  TYPE_SUBLSH1_N, TYPE_SUBLSH2_N, TYPE_SUBLSH_N,
  TYPE_SUBLSH1_N_IP1, TYPE_SUBLSH2_N_IP1, TYPE_SUBLSH_N_IP1,
  TYPE_RSBLSH1_N, TYPE_RSBLSH2_N, TYPE_RSBLSH_N,
  TYPE_RSH1ADD_N, TYPE_RSH1SUB_N,

  TYPE_ADDLSH1_NC, TYPE_ADDLSH2_NC, TYPE_ADDLSH_NC,
  TYPE_SUBLSH1_NC, TYPE_SUBLSH2_NC, TYPE_SUBLSH_NC,
  TYPE_RSBLSH1_NC, TYPE_RSBLSH2_NC, TYPE_RSBLSH_NC,

  TYPE_ADDCND_N, TYPE_SUBCND_N,

  TYPE_MOD_1, TYPE_MOD_1C, TYPE_DIVMOD_1, TYPE_DIVMOD_1C, TYPE_DIVREM_1,
  TYPE_DIVREM_1C, TYPE_PREINV_DIVREM_1, TYPE_DIVREM_2, TYPE_PREINV_MOD_1,
  TYPE_DIV_QR_1N_PI1,
  TYPE_MOD_34LSUB1, TYPE_UDIV_QRNND, TYPE_UDIV_QRNND_R,

  TYPE_DIVEXACT_1, TYPE_BDIV_Q_1, TYPE_DIVEXACT_BY3, TYPE_DIVEXACT_BY3C,
  TYPE_MODEXACT_1_ODD, TYPE_MODEXACT_1C_ODD,

  TYPE_INVERT, TYPE_BINVERT,

  TYPE_GCD, TYPE_GCD_1, TYPE_GCD_FINDA, TYPE_MPZ_JACOBI, TYPE_MPZ_KRONECKER,
  TYPE_MPZ_KRONECKER_UI, TYPE_MPZ_KRONECKER_SI, TYPE_MPZ_UI_KRONECKER,
  TYPE_MPZ_SI_KRONECKER, TYPE_MPZ_LEGENDRE,

  TYPE_AND_N, TYPE_NAND_N, TYPE_ANDN_N, TYPE_IOR_N, TYPE_IORN_N, TYPE_NIOR_N,
  TYPE_XOR_N, TYPE_XNOR_N,

  TYPE_MUL_MN, TYPE_MUL_N, TYPE_SQR, TYPE_UMUL_PPMM, TYPE_UMUL_PPMM_R,
  TYPE_MULLO_N, TYPE_SQRLO, TYPE_MULMID_MN, TYPE_MULMID_N,

  TYPE_SBPI1_DIV_QR, TYPE_TDIV_QR,

  TYPE_SQRTREM, TYPE_SQRT, TYPE_ZERO, TYPE_GET_STR, TYPE_POPCOUNT, TYPE_HAMDIST,

  TYPE_EXTRA
};

struct try_t  param[TYPE_EXTRA];


void
param_init (void)
{
  struct try_t  *p;

#define COPY(index)  memcpy (p, &param[index], sizeof (*p))

#define REFERENCE(fun)                  \
  p->reference = (tryfun_t) fun;        \
  p->reference_name = #fun
#define VALIDATE(fun)           \
  p->validate = fun;            \
  p->validate_name = #fun


  p = &param[TYPE_ADD_N];
  p->retval = 1;
  p->dst[0] = 1;
  p->src[0] = 1;
  p->src[1] = 1;
  REFERENCE (refmpn_add_n);

  p = &param[TYPE_ADD_NC];
  COPY (TYPE_ADD_N);
  p->carry = CARRY_BIT;
  REFERENCE (refmpn_add_nc);

  p = &param[TYPE_SUB_N];
  COPY (TYPE_ADD_N);
  REFERENCE (refmpn_sub_n);

  p = &param[TYPE_SUB_NC];
  COPY (TYPE_ADD_NC);
  REFERENCE (refmpn_sub_nc);

  p = &param[TYPE_ADD];
  COPY (TYPE_ADD_N);
  p->size = SIZE_ALLOW_ZERO;
  p->size2 = 1;
  REFERENCE (refmpn_add);

  p = &param[TYPE_SUB];
  COPY (TYPE_ADD);
  REFERENCE (refmpn_sub);


  p = &param[TYPE_ADD_ERR1_N];
  p->retval = 1;
  p->dst[0] = 1;
  p->dst[1] = 1;
  p->src[0] = 1;
  p->src[1] = 1;
  p->src[2] = 1;
  p->dst_size[1] = SIZE_2;
  p->carry = CARRY_BIT;
  p->overlap = OVERLAP_NOT_DST2;
  REFERENCE (refmpn_add_err1_n);

  p = &param[TYPE_SUB_ERR1_N];
  COPY (TYPE_ADD_ERR1_N);
  REFERENCE (refmpn_sub_err1_n);

  p = &param[TYPE_ADD_ERR2_N];
  COPY (TYPE_ADD_ERR1_N);
  p->src[3] = 1;
  p->dst_size[1] = SIZE_4;
  REFERENCE (refmpn_add_err2_n);

  p = &param[TYPE_SUB_ERR2_N];
  COPY (TYPE_ADD_ERR2_N);
  REFERENCE (refmpn_sub_err2_n);

  p = &param[TYPE_ADD_ERR3_N];
  COPY (TYPE_ADD_ERR2_N);
  p->src[4] = 1;
  p->dst_size[1] = SIZE_6;
  REFERENCE (refmpn_add_err3_n);

  p = &param[TYPE_SUB_ERR3_N];
  COPY (TYPE_ADD_ERR3_N);
  REFERENCE (refmpn_sub_err3_n);

  p = &param[TYPE_ADDCND_N];
  COPY (TYPE_ADD_N);
  p->carry = CARRY_BIT;
  REFERENCE (refmpn_cnd_add_n);

  p = &param[TYPE_SUBCND_N];
  COPY (TYPE_ADD_N);
  p->carry = CARRY_BIT;
  REFERENCE (refmpn_cnd_sub_n);


  p = &param[TYPE_MUL_1];
  p->retval = 1;
  p->dst[0] = 1;
  p->src[0] = 1;
  p->multiplier = 1;
  p->overlap = OVERLAP_LOW_TO_HIGH;
  REFERENCE (refmpn_mul_1);

  p = &param[TYPE_MUL_1C];
  COPY (TYPE_MUL_1);
  p->carry = CARRY_LIMB;
  REFERENCE (refmpn_mul_1c);


  p = &param[TYPE_MUL_2];
  p->retval = 1;
  p->dst[0] = 1;
  p->dst_size[0] = SIZE_PLUS_MSIZE_SUB_1;
  p->src[0] = 1;
  p->src[1] = 1;
  p->msize = 2;
  p->overlap = OVERLAP_NOT_SRC2;
  REFERENCE (refmpn_mul_2);

  p = &param[TYPE_MUL_3];
  COPY (TYPE_MUL_2);
  p->msize = 3;
  REFERENCE (refmpn_mul_3);

  p = &param[TYPE_MUL_4];
  COPY (TYPE_MUL_2);
  p->msize = 4;
  REFERENCE (refmpn_mul_4);

  p = &param[TYPE_MUL_5];
  COPY (TYPE_MUL_2);
  p->msize = 5;
  REFERENCE (refmpn_mul_5);

  p = &param[TYPE_MUL_6];
  COPY (TYPE_MUL_2);
  p->msize = 6;
  REFERENCE (refmpn_mul_6);


  p = &param[TYPE_ADDMUL_1];
  p->retval = 1;
  p->dst[0] = 1;
  p->src[0] = 1;
  p->multiplier = 1;
  p->dst0_from_src1 = 1;
  REFERENCE (refmpn_addmul_1);

  p = &param[TYPE_ADDMUL_1C];
  COPY (TYPE_ADDMUL_1);
  p->carry = CARRY_LIMB;
  REFERENCE (refmpn_addmul_1c);

  p = &param[TYPE_SUBMUL_1];
  COPY (TYPE_ADDMUL_1);
  REFERENCE (refmpn_submul_1);

  p = &param[TYPE_SUBMUL_1C];
  COPY (TYPE_ADDMUL_1C);
  REFERENCE (refmpn_submul_1c);


  p = &param[TYPE_ADDMUL_2];
  p->retval = 1;
  p->dst[0] = 1;
  p->dst_size[0] = SIZE_PLUS_MSIZE_SUB_1;
  p->src[0] = 1;
  p->src[1] = 1;
  p->msize = 2;
  p->dst0_from_src1 = 1;
  p->overlap = OVERLAP_NONE;
  REFERENCE (refmpn_addmul_2);

  p = &param[TYPE_ADDMUL_3];
  COPY (TYPE_ADDMUL_2);
  p->msize = 3;
  REFERENCE (refmpn_addmul_3);

  p = &param[TYPE_ADDMUL_4];
  COPY (TYPE_ADDMUL_2);
  p->msize = 4;
  REFERENCE (refmpn_addmul_4);

  p = &param[TYPE_ADDMUL_5];
  COPY (TYPE_ADDMUL_2);
  p->msize = 5;
  REFERENCE (refmpn_addmul_5);

  p = &param[TYPE_ADDMUL_6];
  COPY (TYPE_ADDMUL_2);
  p->msize = 6;
  REFERENCE (refmpn_addmul_6);

  p = &param[TYPE_ADDMUL_7];
  COPY (TYPE_ADDMUL_2);
  p->msize = 7;
  REFERENCE (refmpn_addmul_7);

  p = &param[TYPE_ADDMUL_8];
  COPY (TYPE_ADDMUL_2);
  p->msize = 8;
  REFERENCE (refmpn_addmul_8);


  p = &param[TYPE_AND_N];
  p->dst[0] = 1;
  p->src[0] = 1;
  p->src[1] = 1;
  REFERENCE (refmpn_and_n);

  p = &param[TYPE_ANDN_N];
  COPY (TYPE_AND_N);
  REFERENCE (refmpn_andn_n);

  p = &param[TYPE_NAND_N];
  COPY (TYPE_AND_N);
  REFERENCE (refmpn_nand_n);

  p = &param[TYPE_IOR_N];
  COPY (TYPE_AND_N);
  REFERENCE (refmpn_ior_n);

  p = &param[TYPE_IORN_N];
  COPY (TYPE_AND_N);
  REFERENCE (refmpn_iorn_n);

  p = &param[TYPE_NIOR_N];
  COPY (TYPE_AND_N);
  REFERENCE (refmpn_nior_n);

  p = &param[TYPE_XOR_N];
  COPY (TYPE_AND_N);
  REFERENCE (refmpn_xor_n);

  p = &param[TYPE_XNOR_N];
  COPY (TYPE_AND_N);
  REFERENCE (refmpn_xnor_n);


  p = &param[TYPE_ADDSUB_N];
  p->retval = 1;
  p->dst[0] = 1;
  p->dst[1] = 1;
  p->src[0] = 1;
  p->src[1] = 1;
  REFERENCE (refmpn_add_n_sub_n);

  p = &param[TYPE_ADDSUB_NC];
  COPY (TYPE_ADDSUB_N);
  p->carry = CARRY_4;
  REFERENCE (refmpn_add_n_sub_nc);


  p = &param[TYPE_COPY];
  p->dst[0] = 1;
  p->src[0] = 1;
  p->overlap = OVERLAP_NONE;
  p->size = SIZE_ALLOW_ZERO;
  REFERENCE (refmpn_copy);

  p = &param[TYPE_COPYI];
  p->dst[0] = 1;
  p->src[0] = 1;
  p->overlap = OVERLAP_LOW_TO_HIGH;
  p->size = SIZE_ALLOW_ZERO;
  REFERENCE (refmpn_copyi);

  p = &param[TYPE_COPYD];
  p->dst[0] = 1;
  p->src[0] = 1;
  p->overlap = OVERLAP_HIGH_TO_LOW;
  p->size = SIZE_ALLOW_ZERO;
  REFERENCE (refmpn_copyd);

  p = &param[TYPE_COM];
  p->dst[0] = 1;
  p->src[0] = 1;
  REFERENCE (refmpn_com);


  p = &param[TYPE_ADDLSH1_N];
  COPY (TYPE_ADD_N);
  REFERENCE (refmpn_addlsh1_n);

  p = &param[TYPE_ADDLSH2_N];
  COPY (TYPE_ADD_N);
  REFERENCE (refmpn_addlsh2_n);

  p = &param[TYPE_ADDLSH_N];
  COPY (TYPE_ADD_N);
  p->shift = 1;
  REFERENCE (refmpn_addlsh_n);

  p = &param[TYPE_ADDLSH1_N_IP1];
  p->retval = 1;
  p->dst[0] = 1;
  p->src[0] = 1;
  p->dst0_from_src1 = 1;
  REFERENCE (refmpn_addlsh1_n_ip1);

  p = &param[TYPE_ADDLSH2_N_IP1];
  COPY (TYPE_ADDLSH1_N_IP1);
  REFERENCE (refmpn_addlsh2_n_ip1);

  p = &param[TYPE_ADDLSH_N_IP1];
  COPY (TYPE_ADDLSH1_N_IP1);
  p->shift = 1;
  REFERENCE (refmpn_addlsh_n_ip1);

  p = &param[TYPE_ADDLSH1_N_IP2];
  COPY (TYPE_ADDLSH1_N_IP1);
  REFERENCE (refmpn_addlsh1_n_ip2);

  p = &param[TYPE_ADDLSH2_N_IP2];
  COPY (TYPE_ADDLSH1_N_IP1);
  REFERENCE (refmpn_addlsh2_n_ip2);

  p = &param[TYPE_ADDLSH_N_IP2];
  COPY (TYPE_ADDLSH_N_IP1);
  REFERENCE (refmpn_addlsh_n_ip2);

  p = &param[TYPE_SUBLSH1_N];
  COPY (TYPE_ADD_N);
  REFERENCE (refmpn_sublsh1_n);

  p = &param[TYPE_SUBLSH2_N];
  COPY (TYPE_ADD_N);
  REFERENCE (refmpn_sublsh2_n);

  p = &param[TYPE_SUBLSH_N];
  COPY (TYPE_ADDLSH_N);
  REFERENCE (refmpn_sublsh_n);

  p = &param[TYPE_SUBLSH1_N_IP1];
  COPY (TYPE_ADDLSH1_N_IP1);
  REFERENCE (refmpn_sublsh1_n_ip1);

  p = &param[TYPE_SUBLSH2_N_IP1];
  COPY (TYPE_ADDLSH1_N_IP1);
  REFERENCE (refmpn_sublsh2_n_ip1);

  p = &param[TYPE_SUBLSH_N_IP1];
  COPY (TYPE_ADDLSH_N_IP1);
  REFERENCE (refmpn_sublsh_n_ip1);

  p = &param[TYPE_RSBLSH1_N];
  COPY (TYPE_ADD_N);
  REFERENCE (refmpn_rsblsh1_n);

  p = &param[TYPE_RSBLSH2_N];
  COPY (TYPE_ADD_N);
  REFERENCE (refmpn_rsblsh2_n);

  p = &param[TYPE_RSBLSH_N];
  COPY (TYPE_ADDLSH_N);
  REFERENCE (refmpn_rsblsh_n);

  p = &param[TYPE_RSH1ADD_N];
  COPY (TYPE_ADD_N);
  REFERENCE (refmpn_rsh1add_n);

  p = &param[TYPE_RSH1SUB_N];
  COPY (TYPE_ADD_N);
  REFERENCE (refmpn_rsh1sub_n);


  p = &param[TYPE_ADDLSH1_NC];
  COPY (TYPE_ADDLSH1_N);
  p->carry = CARRY_3;
  REFERENCE (refmpn_addlsh1_nc);

  p = &param[TYPE_ADDLSH2_NC];
  COPY (TYPE_ADDLSH2_N);
  p->carry = CARRY_4; /* FIXME */
  REFERENCE (refmpn_addlsh2_nc);

  p = &param[TYPE_ADDLSH_NC];
  COPY (TYPE_ADDLSH_N);
  p->carry = CARRY_BIT; /* FIXME */
  REFERENCE (refmpn_addlsh_nc);

  p = &param[TYPE_SUBLSH1_NC];
  COPY (TYPE_ADDLSH1_NC);
  REFERENCE (refmpn_sublsh1_nc);

  p = &param[TYPE_SUBLSH2_NC];
  COPY (TYPE_ADDLSH2_NC);
  REFERENCE (refmpn_sublsh2_nc);

  p = &param[TYPE_SUBLSH_NC];
  COPY (TYPE_ADDLSH_NC);
  REFERENCE (refmpn_sublsh_nc);

  p = &param[TYPE_RSBLSH1_NC];
  COPY (TYPE_RSBLSH1_N);
  p->carry = CARRY_BIT; /* FIXME */
  REFERENCE (refmpn_rsblsh1_nc);

  p = &param[TYPE_RSBLSH2_NC];
  COPY (TYPE_RSBLSH2_N);
  p->carry = CARRY_4; /* FIXME */
  REFERENCE (refmpn_rsblsh2_nc);

  p = &param[TYPE_RSBLSH_NC];
  COPY (TYPE_RSBLSH_N);
  p->carry = CARRY_BIT; /* FIXME */
  REFERENCE (refmpn_rsblsh_nc);


  p = &param[TYPE_MOD_1];
  p->retval = 1;
  p->src[0] = 1;
  p->size = SIZE_ALLOW_ZERO;
  p->divisor = DIVISOR_LIMB;
  REFERENCE (refmpn_mod_1);

  p = &param[TYPE_MOD_1C];
  COPY (TYPE_MOD_1);
  p->carry = CARRY_DIVISOR;
  REFERENCE (refmpn_mod_1c);

  p = &param[TYPE_DIVMOD_1];
  COPY (TYPE_MOD_1);
  p->dst[0] = 1;
  REFERENCE (refmpn_divmod_1);

  p = &param[TYPE_DIVMOD_1C];
  COPY (TYPE_DIVMOD_1);
  p->carry = CARRY_DIVISOR;
  REFERENCE (refmpn_divmod_1c);

  p = &param[TYPE_DIVREM_1];
  COPY (TYPE_DIVMOD_1);
  p->size2 = SIZE_FRACTION;
  p->dst_size[0] = SIZE_SUM;
  REFERENCE (refmpn_divrem_1);

  p = &param[TYPE_DIVREM_1C];
  COPY (TYPE_DIVREM_1);
  p->carry = CARRY_DIVISOR;
  REFERENCE (refmpn_divrem_1c);

  p = &param[TYPE_PREINV_DIVREM_1];
  COPY (TYPE_DIVREM_1);
  p->size = SIZE_YES; /* ie. no size==0 */
  REFERENCE (refmpn_preinv_divrem_1);

  p = &param[TYPE_DIV_QR_1N_PI1];
  p->retval = 1;
  p->src[0] = 1;
  p->src[1] = 1;
  /* SIZE_1 not supported. Always uses low limb only. */
  p->size2 = 1;
  p->dst[0] = 1;
  p->divisor = DIVISOR_NORM;
  p->data = DATA_DIV_QR_1;
  VALIDATE (validate_div_qr_1_pi1);

  p = &param[TYPE_PREINV_MOD_1];
  p->retval = 1;
  p->src[0] = 1;
  p->divisor = DIVISOR_NORM;
  REFERENCE (refmpn_preinv_mod_1);

  p = &param[TYPE_MOD_34LSUB1];
  p->retval = 1;
  p->src[0] = 1;
  VALIDATE (validate_mod_34lsub1);

  p = &param[TYPE_UDIV_QRNND];
  p->retval = 1;
  p->src[0] = 1;
  p->dst[0] = 1;
  p->dst_size[0] = SIZE_1;
  p->divisor = UDIV_NEEDS_NORMALIZATION ? DIVISOR_NORM : DIVISOR_LIMB;
  p->data = DATA_UDIV_QRNND;
  p->overlap = OVERLAP_NONE;
  REFERENCE (refmpn_udiv_qrnnd);

  p = &param[TYPE_UDIV_QRNND_R];
  COPY (TYPE_UDIV_QRNND);
  REFERENCE (refmpn_udiv_qrnnd_r);


  p = &param[TYPE_DIVEXACT_1];
  p->dst[0] = 1;
  p->src[0] = 1;
  p->divisor = DIVISOR_LIMB;
  p->data = DATA_MULTIPLE_DIVISOR;
  VALIDATE (validate_divexact_1);
  REFERENCE (refmpn_divmod_1);

  p = &param[TYPE_BDIV_Q_1];
  p->dst[0] = 1;
  p->src[0] = 1;
  p->divisor = DIVISOR_LIMB;
  VALIDATE (validate_bdiv_q_1);

  p = &param[TYPE_DIVEXACT_BY3];
  p->retval = 1;
  p->dst[0] = 1;
  p->src[0] = 1;
  REFERENCE (refmpn_divexact_by3);

  p = &param[TYPE_DIVEXACT_BY3C];
  COPY (TYPE_DIVEXACT_BY3);
  p->carry = CARRY_3;
  REFERENCE (refmpn_divexact_by3c);


  p = &param[TYPE_MODEXACT_1_ODD];
  p->retval = 1;
  p->src[0] = 1;
  p->divisor = DIVISOR_ODD;
  VALIDATE (validate_modexact_1_odd);

  p = &param[TYPE_MODEXACT_1C_ODD];
  COPY (TYPE_MODEXACT_1_ODD);
  p->carry = CARRY_LIMB;
  VALIDATE (validate_modexact_1c_odd);


  p = &param[TYPE_GCD_1];
  p->retval = 1;
  p->src[0] = 1;
  p->data = DATA_NON_ZERO;
  p->divisor = DIVISOR_LIMB;
  REFERENCE (refmpn_gcd_1);

  p = &param[TYPE_GCD];
  p->retval = 1;
  p->dst[0] = 1;
  p->src[0] = 1;
  p->src[1] = 1;
  p->size2 = 1;
  p->dst_size[0] = SIZE_RETVAL;
  p->overlap = OVERLAP_NOT_SRCS;
  p->data = DATA_GCD;
  REFERENCE (refmpn_gcd);


  p = &param[TYPE_MPZ_LEGENDRE];
  p->retval = 1;
  p->src[0] = 1;
  p->size = SIZE_ALLOW_ZERO;
  p->src[1] = 1;
  p->data = DATA_SRC1_ODD_PRIME;
  p->size2 = 1;
  p->carry = CARRY_BIT;
  p->carry_sign = 1;
  REFERENCE (refmpz_legendre);

  p = &param[TYPE_MPZ_JACOBI];
  p->retval = 1;
  p->src[0] = 1;
  p->size = SIZE_ALLOW_ZERO;
  p->src[1] = 1;
  p->data = DATA_SRC1_ODD;
  p->size2 = 1;
  p->carry = CARRY_BIT;
  p->carry_sign = 1;
  REFERENCE (refmpz_jacobi);

  p = &param[TYPE_MPZ_KRONECKER];
  p->retval = 1;
  p->src[0] = 1;
  p->size = SIZE_ALLOW_ZERO;
  p->src[1] = 1;
  p->data = 0;
  p->size2 = 1;
  p->carry = CARRY_4;
  p->carry_sign = 1;
  REFERENCE (refmpz_kronecker);


  p = &param[TYPE_MPZ_KRONECKER_UI];
  p->retval = 1;
  p->src[0] = 1;
  p->size = SIZE_ALLOW_ZERO;
  p->multiplier = 1;
  p->carry = CARRY_BIT;
  REFERENCE (refmpz_kronecker_ui);

  p = &param[TYPE_MPZ_KRONECKER_SI];
  COPY (TYPE_MPZ_KRONECKER_UI);
  REFERENCE (refmpz_kronecker_si);

  p = &param[TYPE_MPZ_UI_KRONECKER];
  COPY (TYPE_MPZ_KRONECKER_UI);
  REFERENCE (refmpz_ui_kronecker);

  p = &param[TYPE_MPZ_SI_KRONECKER];
  COPY (TYPE_MPZ_KRONECKER_UI);
  REFERENCE (refmpz_si_kronecker);


  p = &param[TYPE_SQR];
  p->dst[0] = 1;
  p->src[0] = 1;
  p->dst_size[0] = SIZE_SUM;
  p->overlap = OVERLAP_NONE;
  REFERENCE (refmpn_sqr);

  p = &param[TYPE_MUL_N];
  COPY (TYPE_SQR);
  p->src[1] = 1;
  REFERENCE (refmpn_mul_n);

  p = &param[TYPE_MULLO_N];
  COPY (TYPE_MUL_N);
  p->dst_size[0] = 0;
  REFERENCE (refmpn_mullo_n);

  p = &param[TYPE_SQRLO];
  COPY (TYPE_SQR);
  p->dst_size[0] = 0;
  REFERENCE (refmpn_sqrlo);

  p = &param[TYPE_MUL_MN];
  COPY (TYPE_MUL_N);
  p->size2 = 1;
  REFERENCE (refmpn_mul_basecase);

  p = &param[TYPE_MULMID_MN];
  COPY (TYPE_MUL_MN);
  p->dst_size[0] = SIZE_DIFF_PLUS_3;
  REFERENCE (refmpn_mulmid_basecase);

  p = &param[TYPE_MULMID_N];
  COPY (TYPE_MUL_N);
  p->size = SIZE_ODD;
  p->size2 = SIZE_CEIL_HALF;
  p->dst_size[0] = SIZE_DIFF_PLUS_3;
  REFERENCE (refmpn_mulmid_n);

  p = &param[TYPE_UMUL_PPMM];
  p->retval = 1;
  p->src[0] = 1;
  p->dst[0] = 1;
  p->dst_size[0] = SIZE_1;
  p->overlap = OVERLAP_NONE;
  REFERENCE (refmpn_umul_ppmm);

  p = &param[TYPE_UMUL_PPMM_R];
  COPY (TYPE_UMUL_PPMM);
  REFERENCE (refmpn_umul_ppmm_r);


  p = &param[TYPE_RSHIFT];
  p->retval = 1;
  p->dst[0] = 1;
  p->src[0] = 1;
  p->shift = 1;
  p->overlap = OVERLAP_LOW_TO_HIGH;
  REFERENCE (refmpn_rshift);

  p = &param[TYPE_LSHIFT];
  COPY (TYPE_RSHIFT);
  p->overlap = OVERLAP_HIGH_TO_LOW;
  REFERENCE (refmpn_lshift);

  p = &param[TYPE_LSHIFTC];
  COPY (TYPE_RSHIFT);
  p->overlap = OVERLAP_HIGH_TO_LOW;
  REFERENCE (refmpn_lshiftc);


  p = &param[TYPE_POPCOUNT];
  p->retval = 1;
  p->src[0] = 1;
  REFERENCE (refmpn_popcount);

  p = &param[TYPE_HAMDIST];
  COPY (TYPE_POPCOUNT);
  p->src[1] = 1;
  REFERENCE (refmpn_hamdist);


  p = &param[TYPE_SBPI1_DIV_QR];
  p->retval = 1;
  p->dst[0] = 1;
  p->dst[1] = 1;
  p->src[0] = 1;
  p->src[1] = 1;
  p->data = DATA_SRC1_HIGHBIT;
  p->size2 = 1;
  p->dst_size[0] = SIZE_DIFF;
  p->overlap = OVERLAP_NONE;
  REFERENCE (refmpn_sb_div_qr);

  p = &param[TYPE_TDIV_QR];
  p->dst[0] = 1;
  p->dst[1] = 1;
  p->src[0] = 1;
  p->src[1] = 1;
  p->size2 = 1;
  p->dst_size[0] = SIZE_DIFF_PLUS_1;
  p->dst_size[1] = SIZE_SIZE2;
  p->overlap = OVERLAP_NONE;
  REFERENCE (refmpn_tdiv_qr);

  p = &param[TYPE_SQRTREM];
  p->retval = 1;
  p->dst[0] = 1;
  p->dst[1] = 1;
  p->src[0] = 1;
  p->dst_size[0] = SIZE_CEIL_HALF;
  p->dst_size[1] = SIZE_RETVAL;
  p->overlap = OVERLAP_NONE;
  VALIDATE (validate_sqrtrem);
  REFERENCE (refmpn_sqrtrem);

  p = &param[TYPE_SQRT];
  p->retval = 1;
  p->dst[0] = 1;
  p->dst[1] = 0;
  p->src[0] = 1;
  p->dst_size[0] = SIZE_CEIL_HALF;
  p->overlap = OVERLAP_NONE;
  VALIDATE (validate_sqrt);

  p = &param[TYPE_ZERO];
  p->dst[0] = 1;
  p->size = SIZE_ALLOW_ZERO;
  REFERENCE (refmpn_zero);

  p = &param[TYPE_GET_STR];
  p->retval = 1;
  p->src[0] = 1;
  p->size = SIZE_ALLOW_ZERO;
  p->dst[0] = 1;
  p->dst[1] = 1;
  p->dst_size[0] = SIZE_GET_STR;
  p->dst_bytes[0] = 1;
  p->overlap = OVERLAP_NONE;
  REFERENCE (refmpn_get_str);

  p = &param[TYPE_BINVERT];
  p->dst[0] = 1;
  p->src[0] = 1;
  p->data = DATA_SRC0_ODD;
  p->overlap = OVERLAP_NONE;
  REFERENCE (refmpn_binvert);

  p = &param[TYPE_INVERT];
  p->dst[0] = 1;
  p->src[0] = 1;
  p->data = DATA_SRC0_HIGHBIT;
  p->overlap = OVERLAP_NONE;
  REFERENCE (refmpn_invert);

#ifdef EXTRA_PARAM_INIT
  EXTRA_PARAM_INIT
#endif
}


/* The following are macros if there's no native versions, so wrap them in
   functions that can be in try_array[]. */

void
MPN_COPY_fun (mp_ptr rp, mp_srcptr sp, mp_size_t size)
{ MPN_COPY (rp, sp, size); }

void
MPN_COPY_INCR_fun (mp_ptr rp, mp_srcptr sp, mp_size_t size)
{ MPN_COPY_INCR (rp, sp, size); }

void
MPN_COPY_DECR_fun (mp_ptr rp, mp_srcptr sp, mp_size_t size)
{ MPN_COPY_DECR (rp, sp, size); }

void
__GMPN_COPY_fun (mp_ptr rp, mp_srcptr sp, mp_size_t size)
{ __GMPN_COPY (rp, sp, size); }

#ifdef __GMPN_COPY_INCR
void
__GMPN_COPY_INCR_fun (mp_ptr rp, mp_srcptr sp, mp_size_t size)
{ __GMPN_COPY_INCR (rp, sp, size); }
#endif

void
mpn_com_fun (mp_ptr rp, mp_srcptr sp, mp_size_t size)
{ mpn_com (rp, sp, size); }

void
mpn_and_n_fun (mp_ptr rp, mp_srcptr s1, mp_srcptr s2, mp_size_t size)
{ mpn_and_n (rp, s1, s2, size); }

void
mpn_andn_n_fun (mp_ptr rp, mp_srcptr s1, mp_srcptr s2, mp_size_t size)
{ mpn_andn_n (rp, s1, s2, size); }

void
mpn_nand_n_fun (mp_ptr rp, mp_srcptr s1, mp_srcptr s2, mp_size_t size)
{ mpn_nand_n (rp, s1, s2, size); }

void
mpn_ior_n_fun (mp_ptr rp, mp_srcptr s1, mp_srcptr s2, mp_size_t size)
{ mpn_ior_n (rp, s1, s2, size); }

void
mpn_iorn_n_fun (mp_ptr rp, mp_srcptr s1, mp_srcptr s2, mp_size_t size)
{ mpn_iorn_n (rp, s1, s2, size); }

void
mpn_nior_n_fun (mp_ptr rp, mp_srcptr s1, mp_srcptr s2, mp_size_t size)
{ mpn_nior_n (rp, s1, s2, size); }

void
mpn_xor_n_fun (mp_ptr rp, mp_srcptr s1, mp_srcptr s2, mp_size_t size)
{ mpn_xor_n (rp, s1, s2, size); }

void
mpn_xnor_n_fun (mp_ptr rp, mp_srcptr s1, mp_srcptr s2, mp_size_t size)
{ mpn_xnor_n (rp, s1, s2, size); }

mp_limb_t
udiv_qrnnd_fun (mp_limb_t *remptr, mp_limb_t n1, mp_limb_t n0, mp_limb_t d)
{
  mp_limb_t  q;
  udiv_qrnnd (q, *remptr, n1, n0, d);
  return q;
}

mp_limb_t
mpn_divexact_by3_fun (mp_ptr rp, mp_srcptr sp, mp_size_t size)
{
  return mpn_divexact_by3 (rp, sp, size);
}

#if HAVE_NATIVE_mpn_addlsh1_n_ip1
mp_limb_t
mpn_addlsh1_n_ip1_fun (mp_ptr rp, mp_srcptr sp, mp_size_t size)
{
  return mpn_addlsh1_n_ip1 (rp, sp, size);
}
#endif
#if HAVE_NATIVE_mpn_addlsh2_n_ip1
mp_limb_t
mpn_addlsh2_n_ip1_fun (mp_ptr rp, mp_srcptr sp, mp_size_t size)
{
  return mpn_addlsh2_n_ip1 (rp, sp, size);
}
#endif
#if HAVE_NATIVE_mpn_addlsh_n_ip1
mp_limb_t
mpn_addlsh_n_ip1_fun (mp_ptr rp, mp_srcptr sp, mp_size_t size, unsigned int sh)
{
  return mpn_addlsh_n_ip1 (rp, sp, size, sh);
}
#endif
#if HAVE_NATIVE_mpn_addlsh1_n_ip2
mp_limb_t
mpn_addlsh1_n_ip2_fun (mp_ptr rp, mp_srcptr sp, mp_size_t size)
{
  return mpn_addlsh1_n_ip2 (rp, sp, size);
}
#endif
#if HAVE_NATIVE_mpn_addlsh2_n_ip2
mp_limb_t
mpn_addlsh2_n_ip2_fun (mp_ptr rp, mp_srcptr sp, mp_size_t size)
{
  return mpn_addlsh2_n_ip2 (rp, sp, size);
}
#endif
#if HAVE_NATIVE_mpn_addlsh_n_ip2
mp_limb_t
mpn_addlsh_n_ip2_fun (mp_ptr rp, mp_srcptr sp, mp_size_t size, unsigned int sh)
{
  return mpn_addlsh_n_ip2 (rp, sp, size, sh);
}
#endif
#if HAVE_NATIVE_mpn_sublsh1_n_ip1
mp_limb_t
mpn_sublsh1_n_ip1_fun (mp_ptr rp, mp_srcptr sp, mp_size_t size)
{
  return mpn_sublsh1_n_ip1 (rp, sp, size);
}
#endif
#if HAVE_NATIVE_mpn_sublsh2_n_ip1
mp_limb_t
mpn_sublsh2_n_ip1_fun (mp_ptr rp, mp_srcptr sp, mp_size_t size)
{
  return mpn_sublsh2_n_ip1 (rp, sp, size);
}
#endif
#if HAVE_NATIVE_mpn_sublsh_n_ip1
mp_limb_t
mpn_sublsh_n_ip1_fun (mp_ptr rp, mp_srcptr sp, mp_size_t size, unsigned int sh)
{
  return mpn_sublsh_n_ip1 (rp, sp, size, sh);
}
#endif

mp_limb_t
mpn_modexact_1_odd_fun (mp_srcptr ptr, mp_size_t size, mp_limb_t divisor)
{
  return mpn_modexact_1_odd (ptr, size, divisor);
}

void
mpn_toom22_mul_fun (mp_ptr dst, mp_srcptr src1, mp_srcptr src2, mp_size_t size)
{
  mp_ptr  tspace;
  TMP_DECL;
  TMP_MARK;
  tspace = TMP_ALLOC_LIMBS (mpn_toom22_mul_itch (size, size));
  mpn_toom22_mul (dst, src1, size, src2, size, tspace);
  TMP_FREE;
}
void
mpn_toom2_sqr_fun (mp_ptr dst, mp_srcptr src, mp_size_t size)
{
  mp_ptr tspace;
  TMP_DECL;
  TMP_MARK;
  tspace = TMP_ALLOC_LIMBS (mpn_toom2_sqr_itch (size));
  mpn_toom2_sqr (dst, src, size, tspace);
  TMP_FREE;
}
void
mpn_toom33_mul_fun (mp_ptr dst, mp_srcptr src1, mp_srcptr src2, mp_size_t size)
{
  mp_ptr  tspace;
  TMP_DECL;
  TMP_MARK;
  tspace = TMP_ALLOC_LIMBS (mpn_toom33_mul_itch (size, size));
  mpn_toom33_mul (dst, src1, size, src2, size, tspace);
  TMP_FREE;
}
void
mpn_toom3_sqr_fun (mp_ptr dst, mp_srcptr src, mp_size_t size)
{
  mp_ptr tspace;
  TMP_DECL;
  TMP_MARK;
  tspace = TMP_ALLOC_LIMBS (mpn_toom3_sqr_itch (size));
  mpn_toom3_sqr (dst, src, size, tspace);
  TMP_FREE;
}
void
mpn_toom44_mul_fun (mp_ptr dst, mp_srcptr src1, mp_srcptr src2, mp_size_t size)
{
  mp_ptr  tspace;
  TMP_DECL;
  TMP_MARK;
  tspace = TMP_ALLOC_LIMBS (mpn_toom44_mul_itch (size, size));
  mpn_toom44_mul (dst, src1, size, src2, size, tspace);
  TMP_FREE;
}
void
mpn_toom4_sqr_fun (mp_ptr dst, mp_srcptr src, mp_size_t size)
{
  mp_ptr tspace;
  TMP_DECL;
  TMP_MARK;
  tspace = TMP_ALLOC_LIMBS (mpn_toom4_sqr_itch (size));
  mpn_toom4_sqr (dst, src, size, tspace);
  TMP_FREE;
}

void
mpn_toom42_mulmid_fun (mp_ptr dst, mp_srcptr src1, mp_srcptr src2,
		       mp_size_t size)
{
  mp_ptr  tspace;
  mp_size_t n;
  TMP_DECL;
  TMP_MARK;
  tspace = TMP_ALLOC_LIMBS (mpn_toom42_mulmid_itch (size));
  mpn_toom42_mulmid (dst, src1, src2, size, tspace);
  TMP_FREE;
}

mp_limb_t
umul_ppmm_fun (mp_limb_t *lowptr, mp_limb_t m1, mp_limb_t m2)
{
  mp_limb_t  high;
  umul_ppmm (high, *lowptr, m1, m2);
  return high;
}

void
MPN_ZERO_fun (mp_ptr ptr, mp_size_t size)
{ MPN_ZERO (ptr, size); }

mp_size_t
mpn_sqrt_fun (mp_ptr dst, mp_srcptr src, mp_size_t size)
{ return mpn_sqrtrem (dst, NULL, src, size); }

struct choice_t {
  const char  *name;
  tryfun_t    function;
  int         type;
  mp_size_t   minsize;
};

#define TRY(fun)        #fun, (tryfun_t) fun
#define TRY_FUNFUN(fun) #fun, (tryfun_t) fun##_fun

const struct choice_t choice_array[] = {
  { TRY(mpn_add),       TYPE_ADD    },
  { TRY(mpn_sub),       TYPE_SUB    },

  { TRY(mpn_add_n),     TYPE_ADD_N  },
  { TRY(mpn_sub_n),     TYPE_SUB_N  },

#if HAVE_NATIVE_mpn_add_nc
  { TRY(mpn_add_nc),    TYPE_ADD_NC },
#endif
#if HAVE_NATIVE_mpn_sub_nc
  { TRY(mpn_sub_nc),    TYPE_SUB_NC },
#endif

#if HAVE_NATIVE_mpn_add_n_sub_n
  { TRY(mpn_add_n_sub_n),  TYPE_ADDSUB_N  },
#endif
#if HAVE_NATIVE_mpn_add_n_sub_nc
  { TRY(mpn_add_n_sub_nc), TYPE_ADDSUB_NC },
#endif

  { TRY(mpn_add_err1_n),  TYPE_ADD_ERR1_N  },
  { TRY(mpn_sub_err1_n),  TYPE_SUB_ERR1_N  },
  { TRY(mpn_add_err2_n),  TYPE_ADD_ERR2_N  },
  { TRY(mpn_sub_err2_n),  TYPE_SUB_ERR2_N  },
  { TRY(mpn_add_err3_n),  TYPE_ADD_ERR3_N  },
  { TRY(mpn_sub_err3_n),  TYPE_SUB_ERR3_N  },

  { TRY(mpn_addmul_1),  TYPE_ADDMUL_1  },
  { TRY(mpn_submul_1),  TYPE_SUBMUL_1  },
#if HAVE_NATIVE_mpn_addmul_1c
  { TRY(mpn_addmul_1c), TYPE_ADDMUL_1C },
#endif
#if HAVE_NATIVE_mpn_submul_1c
  { TRY(mpn_submul_1c), TYPE_SUBMUL_1C },
#endif

#if HAVE_NATIVE_mpn_addmul_2
  { TRY(mpn_addmul_2), TYPE_ADDMUL_2, 2 },
#endif
#if HAVE_NATIVE_mpn_addmul_3
  { TRY(mpn_addmul_3), TYPE_ADDMUL_3, 3 },
#endif
#if HAVE_NATIVE_mpn_addmul_4
  { TRY(mpn_addmul_4), TYPE_ADDMUL_4, 4 },
#endif
#if HAVE_NATIVE_mpn_addmul_5
  { TRY(mpn_addmul_5), TYPE_ADDMUL_5, 5 },
#endif
#if HAVE_NATIVE_mpn_addmul_6
  { TRY(mpn_addmul_6), TYPE_ADDMUL_6, 6 },
#endif
#if HAVE_NATIVE_mpn_addmul_7
  { TRY(mpn_addmul_7), TYPE_ADDMUL_7, 7 },
#endif
#if HAVE_NATIVE_mpn_addmul_8
  { TRY(mpn_addmul_8), TYPE_ADDMUL_8, 8 },
#endif

  { TRY_FUNFUN(mpn_com),  TYPE_COM },

  { TRY_FUNFUN(MPN_COPY),      TYPE_COPY },
  { TRY_FUNFUN(MPN_COPY_INCR), TYPE_COPYI },
  { TRY_FUNFUN(MPN_COPY_DECR), TYPE_COPYD },

  { TRY_FUNFUN(__GMPN_COPY),      TYPE_COPY },
#ifdef __GMPN_COPY_INCR
  { TRY_FUNFUN(__GMPN_COPY_INCR), TYPE_COPYI },
#endif

#if HAVE_NATIVE_mpn_copyi
  { TRY(mpn_copyi), TYPE_COPYI },
#endif
#if HAVE_NATIVE_mpn_copyd
  { TRY(mpn_copyd), TYPE_COPYD },
#endif

  { TRY(mpn_cnd_add_n), TYPE_ADDCND_N },
  { TRY(mpn_cnd_sub_n), TYPE_SUBCND_N },
#if HAVE_NATIVE_mpn_addlsh1_n == 1
  { TRY(mpn_addlsh1_n), TYPE_ADDLSH1_N },
#endif
#if HAVE_NATIVE_mpn_addlsh2_n == 1
  { TRY(mpn_addlsh2_n), TYPE_ADDLSH2_N },
#endif
#if HAVE_NATIVE_mpn_addlsh_n
  { TRY(mpn_addlsh_n), TYPE_ADDLSH_N },
#endif
#if HAVE_NATIVE_mpn_addlsh1_n_ip1
  { TRY_FUNFUN(mpn_addlsh1_n_ip1), TYPE_ADDLSH1_N_IP1 },
#endif
#if HAVE_NATIVE_mpn_addlsh2_n_ip1
  { TRY_FUNFUN(mpn_addlsh2_n_ip1), TYPE_ADDLSH2_N_IP1 },
#endif
#if HAVE_NATIVE_mpn_addlsh_n_ip1
  { TRY_FUNFUN(mpn_addlsh_n_ip1), TYPE_ADDLSH_N_IP1 },
#endif
#if HAVE_NATIVE_mpn_addlsh1_n_ip2
  { TRY_FUNFUN(mpn_addlsh1_n_ip2), TYPE_ADDLSH1_N_IP2 },
#endif
#if HAVE_NATIVE_mpn_addlsh2_n_ip2
  { TRY_FUNFUN(mpn_addlsh2_n_ip2), TYPE_ADDLSH2_N_IP2 },
#endif
#if HAVE_NATIVE_mpn_addlsh_n_ip2
  { TRY_FUNFUN(mpn_addlsh_n_ip2), TYPE_ADDLSH_N_IP2 },
#endif
#if HAVE_NATIVE_mpn_sublsh1_n == 1
  { TRY(mpn_sublsh1_n), TYPE_SUBLSH1_N },
#endif
#if HAVE_NATIVE_mpn_sublsh2_n == 1
  { TRY(mpn_sublsh2_n), TYPE_SUBLSH2_N },
#endif
#if HAVE_NATIVE_mpn_sublsh_n
  { TRY(mpn_sublsh_n), TYPE_SUBLSH_N },
#endif
#if HAVE_NATIVE_mpn_sublsh1_n_ip1
  { TRY_FUNFUN(mpn_sublsh1_n_ip1), TYPE_SUBLSH1_N_IP1 },
#endif
#if HAVE_NATIVE_mpn_sublsh2_n_ip1
  { TRY_FUNFUN(mpn_sublsh2_n_ip1), TYPE_SUBLSH2_N_IP1 },
#endif
#if HAVE_NATIVE_mpn_sublsh_n_ip1
  { TRY_FUNFUN(mpn_sublsh_n_ip1), TYPE_SUBLSH_N_IP1 },
#endif
#if HAVE_NATIVE_mpn_rsblsh1_n == 1
  { TRY(mpn_rsblsh1_n), TYPE_RSBLSH1_N },
#endif
#if HAVE_NATIVE_mpn_rsblsh2_n == 1
  { TRY(mpn_rsblsh2_n), TYPE_RSBLSH2_N },
#endif
#if HAVE_NATIVE_mpn_rsblsh_n
  { TRY(mpn_rsblsh_n), TYPE_RSBLSH_N },
#endif
#if HAVE_NATIVE_mpn_rsh1add_n
  { TRY(mpn_rsh1add_n), TYPE_RSH1ADD_N },
#endif
#if HAVE_NATIVE_mpn_rsh1sub_n
  { TRY(mpn_rsh1sub_n), TYPE_RSH1SUB_N },
#endif

#if HAVE_NATIVE_mpn_addlsh1_nc
  { TRY(mpn_addlsh1_nc), TYPE_ADDLSH1_NC },
#endif
#if HAVE_NATIVE_mpn_addlsh2_nc
  { TRY(mpn_addlsh2_nc), TYPE_ADDLSH2_NC },
#endif
#if HAVE_NATIVE_mpn_addlsh_nc
  { TRY(mpn_addlsh_nc), TYPE_ADDLSH_NC },
#endif
#if HAVE_NATIVE_mpn_sublsh1_nc
  { TRY(mpn_sublsh1_nc), TYPE_SUBLSH1_NC },
#endif
#if HAVE_NATIVE_mpn_sublsh2_nc
  { TRY(mpn_sublsh2_nc), TYPE_SUBLSH2_NC },
#endif
#if HAVE_NATIVE_mpn_sublsh_nc
  { TRY(mpn_sublsh_nc), TYPE_SUBLSH_NC },
#endif
#if HAVE_NATIVE_mpn_rsblsh1_nc
  { TRY(mpn_rsblsh1_nc), TYPE_RSBLSH1_NC },
#endif
#if HAVE_NATIVE_mpn_rsblsh2_nc
  { TRY(mpn_rsblsh2_nc), TYPE_RSBLSH2_NC },
#endif
#if HAVE_NATIVE_mpn_rsblsh_nc
  { TRY(mpn_rsblsh_nc), TYPE_RSBLSH_NC },
#endif

  { TRY_FUNFUN(mpn_and_n),  TYPE_AND_N  },
  { TRY_FUNFUN(mpn_andn_n), TYPE_ANDN_N },
  { TRY_FUNFUN(mpn_nand_n), TYPE_NAND_N },
  { TRY_FUNFUN(mpn_ior_n),  TYPE_IOR_N  },
  { TRY_FUNFUN(mpn_iorn_n), TYPE_IORN_N },
  { TRY_FUNFUN(mpn_nior_n), TYPE_NIOR_N },
  { TRY_FUNFUN(mpn_xor_n),  TYPE_XOR_N  },
  { TRY_FUNFUN(mpn_xnor_n), TYPE_XNOR_N },

  { TRY(mpn_divrem_1),     TYPE_DIVREM_1 },
#if USE_PREINV_DIVREM_1
  { TRY(mpn_preinv_divrem_1), TYPE_PREINV_DIVREM_1 },
#endif
  { TRY(mpn_mod_1),        TYPE_MOD_1 },
#if USE_PREINV_MOD_1
  { TRY(mpn_preinv_mod_1), TYPE_PREINV_MOD_1 },
#endif
#if HAVE_NATIVE_mpn_divrem_1c
  { TRY(mpn_divrem_1c),    TYPE_DIVREM_1C },
#endif
#if HAVE_NATIVE_mpn_mod_1c
  { TRY(mpn_mod_1c),       TYPE_MOD_1C },
#endif
  { TRY(mpn_div_qr_1n_pi1), TYPE_DIV_QR_1N_PI1 },
#if GMP_NUMB_BITS % 4 == 0
  { TRY(mpn_mod_34lsub1),  TYPE_MOD_34LSUB1 },
#endif

  { TRY_FUNFUN(udiv_qrnnd), TYPE_UDIV_QRNND, 2 },
#if HAVE_NATIVE_mpn_udiv_qrnnd
  { TRY(mpn_udiv_qrnnd),    TYPE_UDIV_QRNND, 2 },
#endif
#if HAVE_NATIVE_mpn_udiv_qrnnd_r
  { TRY(mpn_udiv_qrnnd_r),  TYPE_UDIV_QRNND_R, 2 },
#endif

  { TRY(mpn_divexact_1),          TYPE_DIVEXACT_1 },
  { TRY(mpn_bdiv_q_1),            TYPE_BDIV_Q_1 },
  { TRY_FUNFUN(mpn_divexact_by3), TYPE_DIVEXACT_BY3 },
  { TRY(mpn_divexact_by3c),       TYPE_DIVEXACT_BY3C },

  { TRY_FUNFUN(mpn_modexact_1_odd), TYPE_MODEXACT_1_ODD },
  { TRY(mpn_modexact_1c_odd),       TYPE_MODEXACT_1C_ODD },


  { TRY(mpn_sbpi1_div_qr), TYPE_SBPI1_DIV_QR, 3},
  { TRY(mpn_tdiv_qr),      TYPE_TDIV_QR },

  { TRY(mpn_mul_1),      TYPE_MUL_1 },
#if HAVE_NATIVE_mpn_mul_1c
  { TRY(mpn_mul_1c),     TYPE_MUL_1C },
#endif
#if HAVE_NATIVE_mpn_mul_2
  { TRY(mpn_mul_2),      TYPE_MUL_2, 2 },
#endif
#if HAVE_NATIVE_mpn_mul_3
  { TRY(mpn_mul_3),      TYPE_MUL_3, 3 },
#endif
#if HAVE_NATIVE_mpn_mul_4
  { TRY(mpn_mul_4),      TYPE_MUL_4, 4 },
#endif
#if HAVE_NATIVE_mpn_mul_5
  { TRY(mpn_mul_5),      TYPE_MUL_5, 5 },
#endif
#if HAVE_NATIVE_mpn_mul_6
  { TRY(mpn_mul_6),      TYPE_MUL_6, 6 },
#endif

  { TRY(mpn_rshift),     TYPE_RSHIFT },
  { TRY(mpn_lshift),     TYPE_LSHIFT },
  { TRY(mpn_lshiftc),    TYPE_LSHIFTC },


  { TRY(mpn_mul_basecase), TYPE_MUL_MN },
  { TRY(mpn_mulmid_basecase), TYPE_MULMID_MN },
  { TRY(mpn_mullo_basecase), TYPE_MULLO_N },
  { TRY(mpn_sqrlo_basecase), TYPE_SQRLO },
  { TRY(mpn_sqrlo), TYPE_SQRLO },
#if SQR_TOOM2_THRESHOLD > 0
  { TRY(mpn_sqr_basecase), TYPE_SQR },
#endif

  { TRY(mpn_mul),    TYPE_MUL_MN },
  { TRY(mpn_mul_n),  TYPE_MUL_N },
  { TRY(mpn_sqr),    TYPE_SQR },

  { TRY_FUNFUN(umul_ppmm), TYPE_UMUL_PPMM, 2 },
#if HAVE_NATIVE_mpn_umul_ppmm
  { TRY(mpn_umul_ppmm),    TYPE_UMUL_PPMM, 2 },
#endif
#if HAVE_NATIVE_mpn_umul_ppmm_r
  { TRY(mpn_umul_ppmm_r),  TYPE_UMUL_PPMM_R, 2 },
#endif

  { TRY_FUNFUN(mpn_toom22_mul),  TYPE_MUL_N,  MPN_TOOM22_MUL_MINSIZE },
  { TRY_FUNFUN(mpn_toom2_sqr),   TYPE_SQR,    MPN_TOOM2_SQR_MINSIZE },
  { TRY_FUNFUN(mpn_toom33_mul),  TYPE_MUL_N,  MPN_TOOM33_MUL_MINSIZE },
  { TRY_FUNFUN(mpn_toom3_sqr),   TYPE_SQR,    MPN_TOOM3_SQR_MINSIZE },
  { TRY_FUNFUN(mpn_toom44_mul),  TYPE_MUL_N,  MPN_TOOM44_MUL_MINSIZE },
  { TRY_FUNFUN(mpn_toom4_sqr),   TYPE_SQR,    MPN_TOOM4_SQR_MINSIZE },

  { TRY(mpn_mulmid_n),  TYPE_MULMID_N, 1 },
  { TRY(mpn_mulmid),  TYPE_MULMID_MN, 1 },
  { TRY_FUNFUN(mpn_toom42_mulmid),  TYPE_MULMID_N,
    (2 * MPN_TOOM42_MULMID_MINSIZE - 1) },

  { TRY(mpn_gcd_1),        TYPE_GCD_1            },
  { TRY(mpn_gcd),          TYPE_GCD              },
  { TRY(mpz_legendre),     TYPE_MPZ_LEGENDRE     },
  { TRY(mpz_jacobi),       TYPE_MPZ_JACOBI       },
  { TRY(mpz_kronecker),    TYPE_MPZ_KRONECKER    },
  { TRY(mpz_kronecker_ui), TYPE_MPZ_KRONECKER_UI },
  { TRY(mpz_kronecker_si), TYPE_MPZ_KRONECKER_SI },
  { TRY(mpz_ui_kronecker), TYPE_MPZ_UI_KRONECKER },
  { TRY(mpz_si_kronecker), TYPE_MPZ_SI_KRONECKER },

  { TRY(mpn_popcount),   TYPE_POPCOUNT },
  { TRY(mpn_hamdist),    TYPE_HAMDIST },

  { TRY(mpn_sqrtrem),     TYPE_SQRTREM },
  { TRY_FUNFUN(mpn_sqrt), TYPE_SQRT },

  { TRY_FUNFUN(MPN_ZERO), TYPE_ZERO },

  { TRY(mpn_get_str),    TYPE_GET_STR },

  { TRY(mpn_binvert),    TYPE_BINVERT },
  { TRY(mpn_invert),     TYPE_INVERT  },

#ifdef EXTRA_ROUTINES
  EXTRA_ROUTINES
#endif
};

const struct choice_t *choice = NULL;


void
mprotect_maybe (void *addr, size_t len, int prot)
{
  if (!option_redzones)
    return;

#if HAVE_MPROTECT
  if (mprotect (addr, len, prot) != 0)
    {
      fprintf (stderr, "Cannot mprotect %p 0x%X 0x%X: %s\n",
	       addr, (unsigned) len, prot, strerror (errno));
      exit (1);
    }
#else
  {
    static int  warned = 0;
    if (!warned)
      {
	fprintf (stderr,
		 "mprotect not available, bounds testing not performed\n");
	warned = 1;
      }
  }
#endif
}

/* round "a" up to a multiple of "m" */
size_t
round_up_multiple (size_t a, size_t m)
{
  unsigned long  r;

  r = a % m;
  if (r == 0)
    return a;
  else
    return a + (m - r);
}


/* On some systems it seems that only an mmap'ed region can be mprotect'ed,
   for instance HP-UX 10.

   mmap will almost certainly return a pointer already aligned to a page
   boundary, but it's easy enough to share the alignment handling with the
   malloc case. */

void
malloc_region (struct region_t *r, mp_size_t n)
{
  mp_ptr  p;
  size_t  nbytes;

  ASSERT ((pagesize % GMP_LIMB_BYTES) == 0);

  n = round_up_multiple (n, PAGESIZE_LIMBS);
  r->size = n;

  nbytes = n*GMP_LIMB_BYTES + 2*REDZONE_BYTES + pagesize;

#if defined (MAP_ANONYMOUS) && ! defined (MAP_ANON)
#define MAP_ANON  MAP_ANONYMOUS
#endif

#if HAVE_MMAP && defined (MAP_ANON)
  /* note must pass fd=-1 for MAP_ANON on BSD */
  p = (mp_ptr) mmap (NULL, nbytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
  if (p == (void *) -1)
    {
      fprintf (stderr, "Cannot mmap %#x anon bytes: %s\n",
	       (unsigned) nbytes, strerror (errno));
      exit (1);
    }
#else
  p = (mp_ptr) malloc (nbytes);
  ASSERT_ALWAYS (p != NULL);
#endif

  p = (mp_ptr) align_pointer (p, pagesize);

  mprotect_maybe (p, REDZONE_BYTES, PROT_NONE);
  p += REDZONE_LIMBS;
  r->ptr = p;

  mprotect_maybe (p + n, REDZONE_BYTES, PROT_NONE);
}

void
mprotect_region (const struct region_t *r, int prot)
{
  mprotect_maybe (r->ptr, r->size, prot);
}


/* First four entries must be 0,1,2,3 for the benefit of CARRY_BIT, CARRY_3,
   and CARRY_4 */
mp_limb_t  carry_array[] = {
  0, 1, 2, 3,
  4,
  CNST_LIMB(1) << 8,
  CNST_LIMB(1) << 16,
  GMP_NUMB_MAX
};
int        carry_index;

#define CARRY_COUNT                                             \
  ((tr->carry == CARRY_BIT) ? 2                                 \
   : tr->carry == CARRY_3   ? 3                                 \
   : tr->carry == CARRY_4   ? 4                                 \
   : (tr->carry == CARRY_LIMB || tr->carry == CARRY_DIVISOR)    \
     ? numberof(carry_array) + CARRY_RANDOMS                    \
   : 1)

#define MPN_RANDOM_ALT(index,dst,size) \
  (((index) & 1) ? refmpn_random (dst, size) : refmpn_random2 (dst, size))

/* The dummy value after MPN_RANDOM_ALT ensures both sides of the ":" have
   the same type */
#define CARRY_ITERATION                                                 \
  for (carry_index = 0;                                                 \
       (carry_index < numberof (carry_array)                            \
	? (carry = carry_array[carry_index])                            \
	: (MPN_RANDOM_ALT (carry_index, &carry, 1), (mp_limb_t) 0)),    \
	 (tr->carry == CARRY_DIVISOR ? carry %= divisor : 0),           \
	 carry_index < CARRY_COUNT;                                     \
       carry_index++)


mp_limb_t  multiplier_array[] = {
  0, 1, 2, 3,
  CNST_LIMB(1) << 8,
  CNST_LIMB(1) << 16,
  GMP_NUMB_MAX - 2,
  GMP_NUMB_MAX - 1,
  GMP_NUMB_MAX
};
int        multiplier_index;

mp_limb_t  divisor_array[] = {
  1, 2, 3,
  CNST_LIMB(1) << 8,
  CNST_LIMB(1) << 16,
  CNST_LIMB(1) << (GMP_NUMB_BITS/2 - 1),
  GMP_NUMB_MAX >> (GMP_NUMB_BITS/2),
  GMP_NUMB_HIGHBIT,
  GMP_NUMB_HIGHBIT + 1,
  GMP_NUMB_MAX - 2,
  GMP_NUMB_MAX - 1,
  GMP_NUMB_MAX
};

int        divisor_index;

/* The dummy value after MPN_RANDOM_ALT ensures both sides of the ":" have
   the same type */
#define ARRAY_ITERATION(var, index, limit, array, randoms, cond)        \
  for (index = 0;                                                       \
       (index < numberof (array)                                        \
	? (var = array[index])                                          \
	: (MPN_RANDOM_ALT (index, &var, 1), (mp_limb_t) 0)),            \
       index < limit;                                                   \
       index++)

#define MULTIPLIER_COUNT                                \
  (tr->multiplier                                       \
    ? numberof (multiplier_array) + MULTIPLIER_RANDOMS  \
    : 1)

#define MULTIPLIER_ITERATION                                            \
  ARRAY_ITERATION(multiplier, multiplier_index, MULTIPLIER_COUNT,       \
		  multiplier_array, MULTIPLIER_RANDOMS, TRY_MULTIPLIER)

#define DIVISOR_COUNT                           \
  (tr->divisor                                  \
   ? numberof (divisor_array) + DIVISOR_RANDOMS \
   : 1)

#define DIVISOR_ITERATION                                               \
  ARRAY_ITERATION(divisor, divisor_index, DIVISOR_COUNT, divisor_array, \
		  DIVISOR_RANDOMS, TRY_DIVISOR)


/* overlap_array[].s[i] is where s[i] should be, 0 or 1 means overlapping
   d[0] or d[1] respectively, -1 means a separate (write-protected)
   location. */

struct overlap_t {
  int  s[NUM_SOURCES];
} overlap_array[] = {
  { { -1, -1, -1, -1, -1 } },
  { {  0, -1, -1, -1, -1 } },
  { { -1,  0, -1, -1, -1 } },
  { {  0,  0, -1, -1, -1 } },
  { {  1, -1, -1, -1, -1 } },
  { { -1,  1, -1, -1, -1 } },
  { {  1,  1, -1, -1, -1 } },
  { {  0,  1, -1, -1, -1 } },
  { {  1,  0, -1, -1, -1 } },
};

struct overlap_t  *overlap, *overlap_limit;

#define OVERLAP_COUNT                   \
  (tr->overlap & OVERLAP_NONE       ? 1 \
   : tr->overlap & OVERLAP_NOT_SRCS ? 3 \
   : tr->overlap & OVERLAP_NOT_SRC2 ? 2 \
   : tr->overlap & OVERLAP_NOT_DST2 ? 4	\
   : tr->dst[1]                     ? 9 \
   : tr->src[1]                     ? 4 \
   : tr->dst[0]                     ? 2 \
   : 1)

#define OVERLAP_ITERATION                               \
  for (overlap = &overlap_array[0],                     \
    overlap_limit = &overlap_array[OVERLAP_COUNT];      \
    overlap < overlap_limit;                            \
    overlap++)


int  base = 10;

#define T_RAND_COUNT  2
int  t_rand;

void
t_random (mp_ptr ptr, mp_size_t n)
{
  if (n == 0)
    return;

  switch (option_data) {
  case DATA_TRAND:
    switch (t_rand) {
    case 0: refmpn_random (ptr, n); break;
    case 1: refmpn_random2 (ptr, n); break;
    default: abort();
    }
    break;
  case DATA_SEQ:
    {
      static mp_limb_t  counter = 0;
      mp_size_t  i;
      for (i = 0; i < n; i++)
	ptr[i] = ++counter;
    }
    break;
  case DATA_ZEROS:
    refmpn_zero (ptr, n);
    break;
  case DATA_FFS:
    refmpn_fill (ptr, n, GMP_NUMB_MAX);
    break;
  case DATA_2FD:
    /* Special value 0x2FFF...FFFD, which divided by 3 gives 0xFFF...FFF,
       inducing the q1_ff special case in the mul-by-inverse part of some
       versions of divrem_1 and mod_1. */
    refmpn_fill (ptr, n, (mp_limb_t) -1);
    ptr[n-1] = 2;
    ptr[0] -= 2;
    break;

  default:
    abort();
  }
}
#define T_RAND_ITERATION \
  for (t_rand = 0; t_rand < T_RAND_COUNT; t_rand++)


void
print_each (const struct each_t *e)
{
  int  i;

  printf ("%s %s\n", e->name, e == &ref ? tr->reference_name : choice->name);
  if (tr->retval)
    mpn_trace ("   retval", &e->retval, 1);

  for (i = 0; i < NUM_DESTS; i++)
    {
      if (tr->dst[i])
	{
	  if (tr->dst_bytes[i])
	    byte_tracen ("   d[%d]", i, e->d[i].p, d[i].size);
	  else
	    mpn_tracen ("   d[%d]", i, e->d[i].p, d[i].size);
	  printf ("        located %p\n", (void *) (e->d[i].p));
	}
    }

  for (i = 0; i < NUM_SOURCES; i++)
    if (tr->src[i])
      printf ("   s[%d] located %p\n", i, (void *)  (e->s[i].p));
}


void
print_all (void)
{
  int  i;

  printf ("\n");
  printf ("size  %ld\n", (long) size);
  if (tr->size2)
    printf ("size2 %ld\n", (long) size2);

  for (i = 0; i < NUM_DESTS; i++)
    if (d[i].size != size)
      printf ("d[%d].size %ld\n", i, (long) d[i].size);

  if (tr->multiplier)
    mpn_trace ("   multiplier", &multiplier, 1);
  if (tr->divisor)
    mpn_trace ("   divisor", &divisor, 1);
  if (tr->shift)
    printf ("   shift %lu\n", shift);
  if (tr->carry)
    mpn_trace ("   carry", &carry, 1);
  if (tr->msize)
    mpn_trace ("   multiplier_N", multiplier_N, tr->msize);

  for (i = 0; i < NUM_DESTS; i++)
    if (tr->dst[i])
      printf ("   d[%d] %s, align %ld, size %ld\n",
	      i, d[i].high ? "high" : "low",
	      (long) d[i].align, (long) d[i].size);

  for (i = 0; i < NUM_SOURCES; i++)
    {
      if (tr->src[i])
	{
	  printf ("   s[%d] %s, align %ld, ",
		  i, s[i].high ? "high" : "low", (long) s[i].align);
	  switch (overlap->s[i]) {
	  case -1:
	    printf ("no overlap\n");
	    break;
	  default:
	    printf ("==d[%d]%s\n",
		    overlap->s[i],
		    tr->overlap == OVERLAP_LOW_TO_HIGH ? "+a"
		    : tr->overlap == OVERLAP_HIGH_TO_LOW ? "-a"
		    : "");
	    break;
	  }
	  printf ("   s[%d]=", i);
	  if (tr->carry_sign && (carry & (1 << i)))
	    printf ("-");
	  mpn_trace (NULL, s[i].p, SRC_SIZE(i));
	}
    }

  if (tr->dst0_from_src1)
    mpn_trace ("   d[0]", s[1].region.ptr, size);

  if (tr->reference)
    print_each (&ref);
  print_each (&fun);
}

void
compare (void)
{
  int  error = 0;
  int  i;

  if (tr->retval && ref.retval != fun.retval)
    {
      gmp_printf ("Different return values (%Mu, %Mu)\n",
		  ref.retval, fun.retval);
      error = 1;
    }

  for (i = 0; i < NUM_DESTS; i++)
    {
      switch (tr->dst_size[i]) {
      case SIZE_RETVAL:
      case SIZE_GET_STR:
	d[i].size = ref.retval;
	break;
      }
    }

  for (i = 0; i < NUM_DESTS; i++)
    {
      if (! tr->dst[i])
	continue;

      if (tr->dst_bytes[i])
	{
	  if (memcmp (ref.d[i].p, fun.d[i].p, d[i].size) != 0)
	    {
	      printf ("Different d[%d] data results, low diff at %ld, high diff at %ld\n",
		      i,
		      (long) byte_diff_lowest (ref.d[i].p, fun.d[i].p, d[i].size),
		      (long) byte_diff_highest (ref.d[i].p, fun.d[i].p, d[i].size));
	      error = 1;
	    }
	}
      else
	{
	  if (d[i].size != 0
	      && ! refmpn_equal_anynail (ref.d[i].p, fun.d[i].p, d[i].size))
	    {
	      printf ("Different d[%d] data results, low diff at %ld, high diff at %ld\n",
		      i,
		      (long) mpn_diff_lowest (ref.d[i].p, fun.d[i].p, d[i].size),
		      (long) mpn_diff_highest (ref.d[i].p, fun.d[i].p, d[i].size));
	      error = 1;
	    }
	}
    }

  if (error)
    {
      print_all();
      abort();
    }
}


/* The functions are cast if the return value should be a long rather than
   the default mp_limb_t.  This is necessary under _LONG_LONG_LIMB.  This
   might not be enough if some actual calling conventions checking is
   implemented on a long long limb system.  */

void
call (struct each_t *e, tryfun_t function)
{
  switch (choice->type) {
  case TYPE_ADD:
  case TYPE_SUB:
    e->retval = CALLING_CONVENTIONS (function)
      (e->d[0].p, e->s[0].p, size, e->s[1].p, size2);
    break;

  case TYPE_ADD_N:
  case TYPE_SUB_N:
  case TYPE_ADDLSH1_N:
  case TYPE_ADDLSH2_N:
  case TYPE_SUBLSH1_N:
  case TYPE_SUBLSH2_N:
  case TYPE_RSBLSH1_N:
  case TYPE_RSBLSH2_N:
  case TYPE_RSH1ADD_N:
  case TYPE_RSH1SUB_N:
    e->retval = CALLING_CONVENTIONS (function)
      (e->d[0].p, e->s[0].p, e->s[1].p, size);
    break;
  case TYPE_ADDLSH_N:
  case TYPE_SUBLSH_N:
  case TYPE_RSBLSH_N:
    e->retval = CALLING_CONVENTIONS (function)
      (e->d[0].p, e->s[0].p, e->s[1].p, size, shift);
    break;
  case TYPE_ADDLSH_NC:
  case TYPE_SUBLSH_NC:
  case TYPE_RSBLSH_NC:
    e->retval = CALLING_CONVENTIONS (function)
      (e->d[0].p, e->s[0].p, e->s[1].p, size, shift, carry);
    break;
  case TYPE_ADDLSH1_NC:
  case TYPE_ADDLSH2_NC:
  case TYPE_SUBLSH1_NC:
  case TYPE_SUBLSH2_NC:
  case TYPE_RSBLSH1_NC:
  case TYPE_RSBLSH2_NC:
  case TYPE_ADD_NC:
  case TYPE_SUB_NC:
    e->retval = CALLING_CONVENTIONS (function)
      (e->d[0].p, e->s[0].p, e->s[1].p, size, carry);
    break;
  case TYPE_ADDCND_N:
  case TYPE_SUBCND_N:
    e->retval = CALLING_CONVENTIONS (function)
      (carry, e->d[0].p, e->s[0].p, e->s[1].p, size);
    break;
  case TYPE_ADD_ERR1_N:
  case TYPE_SUB_ERR1_N:
    e->retval = CALLING_CONVENTIONS (function)
      (e->d[0].p, e->s[0].p, e->s[1].p, e->d[1].p, e->s[2].p, size, carry);
    break;
  case TYPE_ADD_ERR2_N:
  case TYPE_SUB_ERR2_N:
    e->retval = CALLING_CONVENTIONS (function)
      (e->d[0].p, e->s[0].p, e->s[1].p, e->d[1].p, e->s[2].p, e->s[3].p, size, carry);
    break;
  case TYPE_ADD_ERR3_N:
  case TYPE_SUB_ERR3_N:
    e->retval = CALLING_CONVENTIONS (function)
      (e->d[0].p, e->s[0].p, e->s[1].p, e->d[1].p, e->s[2].p, e->s[3].p, e->s[4].p, size, carry);
    break;

  case TYPE_MUL_1:
  case TYPE_ADDMUL_1:
  case TYPE_SUBMUL_1:
    e->retval = CALLING_CONVENTIONS (function)
      (e->d[0].p, e->s[0].p, size, multiplier);
    break;
  case TYPE_MUL_1C:
  case TYPE_ADDMUL_1C:
  case TYPE_SUBMUL_1C:
    e->retval = CALLING_CONVENTIONS (function)
      (e->d[0].p, e->s[0].p, size, multiplier, carry);
    break;

  case TYPE_MUL_2:
  case TYPE_MUL_3:
  case TYPE_MUL_4:
  case TYPE_MUL_5:
  case TYPE_MUL_6:
    if (size == 1)
      abort ();
    e->retval = CALLING_CONVENTIONS (function)
      (e->d[0].p, e->s[0].p, size, multiplier_N);
    break;

  case TYPE_ADDMUL_2:
  case TYPE_ADDMUL_3:
  case TYPE_ADDMUL_4:
  case TYPE_ADDMUL_5:
  case TYPE_ADDMUL_6:
  case TYPE_ADDMUL_7:
  case TYPE_ADDMUL_8:
    if (size == 1)
      abort ();
    e->retval = CALLING_CONVENTIONS (function)
      (e->d[0].p, e->s[0].p, size, multiplier_N);
    break;

  case TYPE_AND_N:
  case TYPE_ANDN_N:
  case TYPE_NAND_N:
  case TYPE_IOR_N:
  case TYPE_IORN_N:
  case TYPE_NIOR_N:
  case TYPE_XOR_N:
  case TYPE_XNOR_N:
    CALLING_CONVENTIONS (function) (e->d[0].p, e->s[0].p, e->s[1].p, size);
    break;

  case TYPE_ADDSUB_N:
    e->retval = CALLING_CONVENTIONS (function)
      (e->d[0].p, e->d[1].p, e->s[0].p, e->s[1].p, size);
    break;
  case TYPE_ADDSUB_NC:
    e->retval = CALLING_CONVENTIONS (function)
      (e->d[0].p, e->d[1].p, e->s[0].p, e->s[1].p, size, carry);
    break;

  case TYPE_COPY:
  case TYPE_COPYI:
  case TYPE_COPYD:
  case TYPE_COM:
    CALLING_CONVENTIONS (function) (e->d[0].p, e->s[0].p, size);
    break;

  case TYPE_ADDLSH1_N_IP1:
  case TYPE_ADDLSH2_N_IP1:
  case TYPE_ADDLSH1_N_IP2:
  case TYPE_ADDLSH2_N_IP2:
  case TYPE_SUBLSH1_N_IP1:
  case TYPE_SUBLSH2_N_IP1:
  case TYPE_DIVEXACT_BY3:
    e->retval = CALLING_CONVENTIONS (function) (e->d[0].p, e->s[0].p, size);
    break;
  case TYPE_DIVEXACT_BY3C:
    e->retval = CALLING_CONVENTIONS (function) (e->d[0].p, e->s[0].p, size,
						carry);
    break;


  case TYPE_DIVMOD_1:
  case TYPE_DIVEXACT_1:
  case TYPE_BDIV_Q_1:
    e->retval = CALLING_CONVENTIONS (function)
      (e->d[0].p, e->s[0].p, size, divisor);
    break;
  case TYPE_DIVMOD_1C:
    e->retval = CALLING_CONVENTIONS (function)
      (e->d[0].p, e->s[0].p, size, divisor, carry);
    break;
  case TYPE_DIVREM_1:
    e->retval = CALLING_CONVENTIONS (function)
      (e->d[0].p, size2, e->s[0].p, size, divisor);
    break;
  case TYPE_DIVREM_1C:
    e->retval = CALLING_CONVENTIONS (function)
      (e->d[0].p, size2, e->s[0].p, size, divisor, carry);
    break;
  case TYPE_PREINV_DIVREM_1:
    {
      mp_limb_t  dinv;
      unsigned   shift;
      shift = refmpn_count_leading_zeros (divisor);
      dinv = refmpn_invert_limb (divisor << shift);
      e->retval = CALLING_CONVENTIONS (function)
	(e->d[0].p, size2, e->s[0].p, size, divisor, dinv, shift);
    }
    break;
  case TYPE_MOD_1:
  case TYPE_MODEXACT_1_ODD:
    e->retval = CALLING_CONVENTIONS (function)
      (e->s[0].p, size, divisor);
    break;
  case TYPE_MOD_1C:
  case TYPE_MODEXACT_1C_ODD:
    e->retval = CALLING_CONVENTIONS (function)
      (e->s[0].p, size, divisor, carry);
    break;
  case TYPE_PREINV_MOD_1:
    e->retval = CALLING_CONVENTIONS (function)
      (e->s[0].p, size, divisor, refmpn_invert_limb (divisor));
    break;
  case TYPE_DIV_QR_1N_PI1:
    {
      mp_limb_t dinv = refmpn_invert_limb (divisor);
      e->retval = CALLING_CONVENTIONS (function)
	(e->d[0].p, e->s[0].p, size, e->s[1].p[0], divisor, dinv);
      break;
    }

  case TYPE_MOD_34LSUB1:
    e->retval = CALLING_CONVENTIONS (function) (e->s[0].p, size);
    break;

  case TYPE_UDIV_QRNND:
    e->retval = CALLING_CONVENTIONS (function)
      (e->d[0].p, e->s[0].p[1], e->s[0].p[0], divisor);
    break;
  case TYPE_UDIV_QRNND_R:
    e->retval = CALLING_CONVENTIONS (function)
      (e->s[0].p[1], e->s[0].p[0], divisor, e->d[0].p);
    break;

  case TYPE_SBPI1_DIV_QR:
    {
      gmp_pi1_t dinv;
      invert_pi1 (dinv, e->s[1].p[size2-1], e->s[1].p[size2-2]); /* FIXME: use refinvert_pi1 */
      refmpn_copyi (e->d[1].p, e->s[0].p, size);        /* dividend */
      refmpn_fill (e->d[0].p, size-size2, 0x98765432);  /* quotient */
      e->retval = CALLING_CONVENTIONS (function)
	(e->d[0].p, e->d[1].p, size, e->s[1].p, size2, dinv.inv32);
      refmpn_zero (e->d[1].p+size2, size-size2);    /* excess over remainder */
    }
    break;

  case TYPE_TDIV_QR:
    CALLING_CONVENTIONS (function) (e->d[0].p, e->d[1].p, 0,
				    e->s[0].p, size, e->s[1].p, size2);
    break;

  case TYPE_GCD_1:
    /* Must have a non-zero src, but this probably isn't the best way to do
       it. */
    if (refmpn_zero_p (e->s[0].p, size))
      e->retval = 0;
    else
      e->retval = CALLING_CONVENTIONS (function) (e->s[0].p, size, divisor);
    break;

  case TYPE_GCD:
    /* Sources are destroyed, so they're saved and replaced, but a general
       approach to this might be better.  Note that it's still e->s[0].p and
       e->s[1].p that are passed, to get the desired alignments. */
    {
      mp_ptr  s0 = refmpn_malloc_limbs (size);
      mp_ptr  s1 = refmpn_malloc_limbs (size2);
      refmpn_copyi (s0, e->s[0].p, size);
      refmpn_copyi (s1, e->s[1].p, size2);

      mprotect_region (&s[0].region, PROT_READ|PROT_WRITE);
      mprotect_region (&s[1].region, PROT_READ|PROT_WRITE);
      e->retval = CALLING_CONVENTIONS (function) (e->d[0].p,
						  e->s[0].p, size,
						  e->s[1].p, size2);
      refmpn_copyi (e->s[0].p, s0, size);
      refmpn_copyi (e->s[1].p, s1, size2);
      free (s0);
      free (s1);
    }
    break;

  case TYPE_GCD_FINDA:
    {
      /* FIXME: do this with a flag */
      mp_limb_t  c[2];
      c[0] = e->s[0].p[0];
      c[0] += (c[0] == 0);
      c[1] = e->s[0].p[0];
      c[1] += (c[1] == 0);
      e->retval = CALLING_CONVENTIONS (function) (c);
    }
    break;

  case TYPE_MPZ_LEGENDRE:
  case TYPE_MPZ_JACOBI:
    {
      mpz_t  a, b;
      PTR(a) = e->s[0].p; SIZ(a) = (carry==0 ? size : -size);
      PTR(b) = e->s[1].p; SIZ(b) = size2;
      e->retval = CALLING_CONVENTIONS (function) (a, b);
    }
    break;
  case TYPE_MPZ_KRONECKER:
    {
      mpz_t  a, b;
      PTR(a) = e->s[0].p; SIZ(a) = ((carry&1)==0 ? size : -size);
      PTR(b) = e->s[1].p; SIZ(b) = ((carry&2)==0 ? size2 : -size2);
      e->retval = CALLING_CONVENTIONS (function) (a, b);
    }
    break;
  case TYPE_MPZ_KRONECKER_UI:
    {
      mpz_t  a;
      PTR(a) = e->s[0].p; SIZ(a) = (carry==0 ? size : -size);
      e->retval = CALLING_CONVENTIONS(function) (a, (unsigned long)multiplier);
    }
    break;
  case TYPE_MPZ_KRONECKER_SI:
    {
      mpz_t  a;
      PTR(a) = e->s[0].p; SIZ(a) = (carry==0 ? size : -size);
      e->retval = CALLING_CONVENTIONS (function) (a, (long) multiplier);
    }
    break;
  case TYPE_MPZ_UI_KRONECKER:
    {
      mpz_t  b;
      PTR(b) = e->s[0].p; SIZ(b) = (carry==0 ? size : -size);
      e->retval = CALLING_CONVENTIONS(function) ((unsigned long)multiplier, b);
    }
    break;
  case TYPE_MPZ_SI_KRONECKER:
    {
      mpz_t  b;
      PTR(b) = e->s[0].p; SIZ(b) = (carry==0 ? size : -size);
      e->retval = CALLING_CONVENTIONS (function) ((long) multiplier, b);
    }
    break;

  case TYPE_MUL_MN:
  case TYPE_MULMID_MN:
    CALLING_CONVENTIONS (function)
      (e->d[0].p, e->s[0].p, size, e->s[1].p, size2);
    break;
  case TYPE_MUL_N:
  case TYPE_MULLO_N:
    CALLING_CONVENTIONS (function) (e->d[0].p, e->s[0].p, e->s[1].p, size);
    break;
  case TYPE_MULMID_N:
    CALLING_CONVENTIONS (function) (e->d[0].p, e->s[0].p, e->s[1].p,
				    (size + 1) / 2);
    break;
  case TYPE_SQR:
  case TYPE_SQRLO:
    CALLING_CONVENTIONS (function) (e->d[0].p, e->s[0].p, size);
    break;

  case TYPE_UMUL_PPMM:
    e->retval = CALLING_CONVENTIONS (function)
      (e->d[0].p, e->s[0].p[0], e->s[0].p[1]);
    break;
  case TYPE_UMUL_PPMM_R:
    e->retval = CALLING_CONVENTIONS (function)
      (e->s[0].p[0], e->s[0].p[1], e->d[0].p);
    break;

  case TYPE_ADDLSH_N_IP1:
  case TYPE_ADDLSH_N_IP2:
  case TYPE_SUBLSH_N_IP1:
  case TYPE_LSHIFT:
  case TYPE_LSHIFTC:
  case TYPE_RSHIFT:
    e->retval = CALLING_CONVENTIONS (function)
      (e->d[0].p, e->s[0].p, size, shift);
    break;

  case TYPE_POPCOUNT:
    e->retval = (* (unsigned long (*)(ANYARGS))
		 CALLING_CONVENTIONS (function)) (e->s[0].p, size);
    break;
  case TYPE_HAMDIST:
    e->retval = (* (unsigned long (*)(ANYARGS))
		 CALLING_CONVENTIONS (function)) (e->s[0].p, e->s[1].p, size);
    break;

  case TYPE_SQRTREM:
    e->retval = (* (long (*)(ANYARGS)) CALLING_CONVENTIONS (function))
      (e->d[0].p, e->d[1].p, e->s[0].p, size);
    break;

  case TYPE_SQRT:
    e->retval = (* (long (*)(ANYARGS)) CALLING_CONVENTIONS (function))
      (e->d[0].p, e->s[0].p, size);
    break;

  case TYPE_ZERO:
    CALLING_CONVENTIONS (function) (e->d[0].p, size);
    break;

  case TYPE_GET_STR:
    {
      size_t  sizeinbase, fill;
      char    *dst;
      MPN_SIZEINBASE (sizeinbase, e->s[0].p, size, base);
      ASSERT_ALWAYS (sizeinbase <= d[0].size);
      fill = d[0].size - sizeinbase;
      if (d[0].high)
	{
	  memset (e->d[0].p, 0xBA, fill);
	  dst = (char *) e->d[0].p + fill;
	}
      else
	{
	  dst = (char *) e->d[0].p;
	  memset (dst + sizeinbase, 0xBA, fill);
	}
      if (POW2_P (base))
	{
	  e->retval = CALLING_CONVENTIONS (function) (dst, base,
						      e->s[0].p, size);
	}
      else
	{
	  refmpn_copy (e->d[1].p, e->s[0].p, size);
	  e->retval = CALLING_CONVENTIONS (function) (dst, base,
						      e->d[1].p, size);
	}
      refmpn_zero (e->d[1].p, size);  /* clobbered or unused */
    }
    break;

 case TYPE_INVERT:
    {
      mp_ptr scratch;
      TMP_DECL;
      TMP_MARK;
      scratch = TMP_ALLOC_LIMBS (mpn_invert_itch (size));
      CALLING_CONVENTIONS (function) (e->d[0].p, e->s[0].p, size, scratch);
      TMP_FREE;
    }
    break;
  case TYPE_BINVERT:
    {
      mp_ptr scratch;
      TMP_DECL;
      TMP_MARK;
      scratch = TMP_ALLOC_LIMBS (mpn_binvert_itch (size));
      CALLING_CONVENTIONS (function) (e->d[0].p, e->s[0].p, size, scratch);
      TMP_FREE;
    }
    break;

#ifdef EXTRA_CALL
    EXTRA_CALL
#endif

  default:
    printf ("Unknown routine type %d\n", choice->type);
    abort ();
    break;
  }
}


void
pointer_setup (struct each_t *e)
{
  int  i, j;

  for (i = 0; i < NUM_DESTS; i++)
    {
      switch (tr->dst_size[i]) {
      case 0:
      case SIZE_RETVAL: /* will be adjusted later */
	d[i].size = size;
	break;

      case SIZE_1:
	d[i].size = 1;
	break;
      case SIZE_2:
	d[i].size = 2;
	break;
      case SIZE_3:
	d[i].size = 3;
	break;
      case SIZE_4:
	d[i].size = 4;
	break;
      case SIZE_6:
	d[i].size = 6;
	break;

      case SIZE_PLUS_1:
	d[i].size = size+1;
	break;
      case SIZE_PLUS_MSIZE_SUB_1:
	d[i].size = size + tr->msize - 1;
	break;

      case SIZE_SUM:
	if (tr->size2)
	  d[i].size = size + size2;
	else
	  d[i].size = 2*size;
	break;

      case SIZE_SIZE2:
	d[i].size = size2;
	break;

      case SIZE_DIFF:
	d[i].size = size - size2;
	break;

      case SIZE_DIFF_PLUS_1:
	d[i].size = size - size2 + 1;
	break;

      case SIZE_DIFF_PLUS_3:
	d[i].size = size - size2 + 3;
	break;

      case SIZE_CEIL_HALF:
	d[i].size = (size+1)/2;
	break;

      case SIZE_GET_STR:
	{
	  mp_limb_t ff = GMP_NUMB_MAX;
	  MPN_SIZEINBASE (d[i].size, &ff - (size-1), size, base);
	}
	break;

      default:
	printf ("Unrecognised dst_size type %d\n", tr->dst_size[i]);
	abort ();
      }
    }

  /* establish e->d[].p destinations */
  for (i = 0; i < NUM_DESTS; i++)
    {
      mp_size_t  offset = 0;

      /* possible room for overlapping sources */
      for (j = 0; j < numberof (overlap->s); j++)
	if (overlap->s[j] == i)
	  offset = MAX (offset, s[j].align);

      if (d[i].high)
	{
	  if (tr->dst_bytes[i])
	    {
	      e->d[i].p = (mp_ptr)
		((char *) (e->d[i].region.ptr + e->d[i].region.size)
		 - d[i].size - d[i].align);
	    }
	  else
	    {
	      e->d[i].p = e->d[i].region.ptr + e->d[i].region.size
		- d[i].size - d[i].align;
	      if (tr->overlap == OVERLAP_LOW_TO_HIGH)
		e->d[i].p -= offset;
	    }
	}
      else
	{
	  if (tr->dst_bytes[i])
	    {
	      e->d[i].p = (mp_ptr) ((char *) e->d[i].region.ptr + d[i].align);
	    }
	  else
	    {
	      e->d[i].p = e->d[i].region.ptr + d[i].align;
	      if (tr->overlap == OVERLAP_HIGH_TO_LOW)
		e->d[i].p += offset;
	    }
	}
    }

  /* establish e->s[].p sources */
  for (i = 0; i < NUM_SOURCES; i++)
    {
      int  o = overlap->s[i];
      switch (o) {
      case -1:
	/* no overlap */
	e->s[i].p = s[i].p;
	break;
      case 0:
      case 1:
	/* overlap with d[o] */
	if (tr->overlap == OVERLAP_HIGH_TO_LOW)
	  e->s[i].p = e->d[o].p - s[i].align;
	else if (tr->overlap == OVERLAP_LOW_TO_HIGH)
	  e->s[i].p = e->d[o].p + s[i].align;
	else if (tr->size2 == SIZE_FRACTION)
	  e->s[i].p = e->d[o].p + size2;
	else
	  e->s[i].p = e->d[o].p;
	break;
      default:
	abort();
	break;
      }
    }
}


void
validate_fail (void)
{
  if (tr->reference)
    {
      trap_location = TRAP_REF;
      call (&ref, tr->reference);
      trap_location = TRAP_NOWHERE;
    }

  print_all();
  abort();
}


void
try_one (void)
{
  int  i;

  if (option_spinner)
    spinner();
  spinner_count++;

  trap_location = TRAP_SETUPS;

  if (tr->divisor == DIVISOR_NORM)
    divisor |= GMP_NUMB_HIGHBIT;
  if (tr->divisor == DIVISOR_ODD)
    divisor |= 1;

  for (i = 0; i < NUM_SOURCES; i++)
    {
      if (s[i].high)
	s[i].p = s[i].region.ptr + s[i].region.size - SRC_SIZE(i) - s[i].align;
      else
	s[i].p = s[i].region.ptr + s[i].align;
    }

  pointer_setup (&ref);
  pointer_setup (&fun);

  ref.retval = 0x04152637;
  fun.retval = 0x8C9DAEBF;

  t_random (multiplier_N, tr->msize);

  for (i = 0; i < NUM_SOURCES; i++)
    {
      if (! tr->src[i])
	continue;

      mprotect_region (&s[i].region, PROT_READ|PROT_WRITE);
      t_random (s[i].p, SRC_SIZE(i));

      switch (tr->data) {
      case DATA_NON_ZERO:
	if (refmpn_zero_p (s[i].p, SRC_SIZE(i)))
	  s[i].p[0] = 1;
	break;

      case DATA_MULTIPLE_DIVISOR:
	/* same number of low zero bits as divisor */
	s[i].p[0] &= ~ LOW_ZEROS_MASK (divisor);
	refmpn_sub_1 (s[i].p, s[i].p, size,
		      refmpn_mod_1 (s[i].p, size, divisor));
	break;

      case DATA_GCD:
	/* s[1] no more bits than s[0] */
	if (i == 1 && size2 == size)
	  s[1].p[size-1] &= refmpn_msbone_mask (s[0].p[size-1]);

	/* high limb non-zero */
	s[i].p[SRC_SIZE(i)-1] += (s[i].p[SRC_SIZE(i)-1] == 0);

	/* odd */
	s[i].p[0] |= 1;
	break;

      case DATA_SRC0_ODD:
	if (i == 0)
	  s[i].p[0] |= 1;
	break;

      case DATA_SRC1_ODD:
	if (i == 1)
	  s[i].p[0] |= 1;
	break;

      case DATA_SRC1_ODD_PRIME:
	if (i == 1)
	  {
	    if (refmpn_zero_p (s[i].p+1, SRC_SIZE(i)-1)
		&& s[i].p[0] <=3)
	      s[i].p[0] = 3;
	    else
	      {
		mpz_t p;
		mpz_init (p);
		for (;;)
		  {
		    _mpz_realloc (p, SRC_SIZE(i));
		    MPN_COPY (PTR(p), s[i].p, SRC_SIZE(i));
		    SIZ(p) = SRC_SIZE(i);
		    MPN_NORMALIZE (PTR(p), SIZ(p));
		    mpz_nextprime (p, p);
		    if (mpz_size (p) <= SRC_SIZE(i))
		      break;

		    t_random (s[i].p, SRC_SIZE(i));
		  }
		MPN_COPY (s[i].p, PTR(p), SIZ(p));
		if (SIZ(p) < SRC_SIZE(i))
		  MPN_ZERO (s[i].p + SIZ(p), SRC_SIZE(i) - SIZ(p));
		mpz_clear (p);
	      }
	  }
	break;

      case DATA_SRC1_HIGHBIT:
	if (i == 1)
	  {
	    if (tr->size2)
	      s[i].p[size2-1] |= GMP_NUMB_HIGHBIT;
	    else
	      s[i].p[size-1] |= GMP_NUMB_HIGHBIT;
	  }
	break;

      case DATA_SRC0_HIGHBIT:
       if (i == 0)
	 {
	   s[i].p[size-1] |= GMP_NUMB_HIGHBIT;
	 }
       break;

      case DATA_UDIV_QRNND:
	s[i].p[1] %= divisor;
	break;
      case DATA_DIV_QR_1:
	if (i == 1)
	  s[i].p[0] %= divisor;
	break;
      }

      mprotect_region (&s[i].region, PROT_READ);
    }

  for (i = 0; i < NUM_DESTS; i++)
    {
      if (! tr->dst[i])
	continue;

      if (tr->dst0_from_src1 && i==0)
	{
	  mp_size_t  copy = MIN (d[0].size, SRC_SIZE(1));
	  mp_size_t  fill = MAX (0, d[0].size - copy);
	  MPN_COPY (fun.d[0].p, s[1].region.ptr, copy);
	  MPN_COPY (ref.d[0].p, s[1].region.ptr, copy);
	  refmpn_fill (fun.d[0].p + copy, fill, DEADVAL);
	  refmpn_fill (ref.d[0].p + copy, fill, DEADVAL);
	}
      else if (tr->dst_bytes[i])
	{
	  memset (ref.d[i].p, 0xBA, d[i].size);
	  memset (fun.d[i].p, 0xBA, d[i].size);
	}
      else
	{
	  refmpn_fill (ref.d[i].p, d[i].size, DEADVAL);
	  refmpn_fill (fun.d[i].p, d[i].size, DEADVAL);
	}
    }

  for (i = 0; i < NUM_SOURCES; i++)
    {
      if (! tr->src[i])
	continue;

      if (ref.s[i].p != s[i].p)
	{
	  refmpn_copyi (ref.s[i].p, s[i].p, SRC_SIZE(i));
	  refmpn_copyi (fun.s[i].p, s[i].p, SRC_SIZE(i));
	}
    }

  if (option_print)
    print_all();

  if (tr->validate != NULL)
    {
      trap_location = TRAP_FUN;
      call (&fun, choice->function);
      trap_location = TRAP_NOWHERE;

      if (! CALLING_CONVENTIONS_CHECK ())
	{
	  print_all();
	  abort();
	}

      (*tr->validate) ();
    }
  else
    {
      trap_location = TRAP_REF;
      call (&ref, tr->reference);
      trap_location = TRAP_FUN;
      call (&fun, choice->function);
      trap_location = TRAP_NOWHERE;

      if (! CALLING_CONVENTIONS_CHECK ())
	{
	  print_all();
	  abort();
	}

      compare ();
    }
}


#define SIZE_ITERATION                                          \
  for (size = MAX3 (option_firstsize,                           \
		    choice->minsize,                            \
		    (tr->size == SIZE_ALLOW_ZERO) ? 0 : 1),	\
	 size += (tr->size == SIZE_ODD) && !(size & 1);		\
       size <= option_lastsize;                                 \
       size += (tr->size == SIZE_ODD) ? 2 : 1)

#define SIZE2_FIRST                                     \
  (tr->size2 == SIZE_2 ? 2                              \
   : tr->size2 == SIZE_FRACTION ? option_firstsize2     \
   : tr->size2 == SIZE_CEIL_HALF ? ((size + 1) / 2)	\
   : tr->size2 ?                                        \
   MAX (choice->minsize, (option_firstsize2 != 0        \
			  ? option_firstsize2 : 1))     \
   : 0)

#define SIZE2_LAST                                      \
  (tr->size2 == SIZE_2 ? 2                              \
   : tr->size2 == SIZE_FRACTION ? FRACTION_COUNT-1      \
   : tr->size2 == SIZE_CEIL_HALF ? ((size + 1) / 2)	\
   : tr->size2 ? size                                   \
   : 0)

#define SIZE2_ITERATION \
  for (size2 = SIZE2_FIRST; size2 <= SIZE2_LAST; size2++)

#define ALIGN_COUNT(cond)  ((cond) ? ALIGNMENTS : 1)
#define ALIGN_ITERATION(w,n,cond) \
  for (w[n].align = 0; w[n].align < ALIGN_COUNT(cond); w[n].align++)

#define HIGH_LIMIT(cond)  ((cond) != 0)
#define HIGH_COUNT(cond)  (HIGH_LIMIT (cond) + 1)
#define HIGH_ITERATION(w,n,cond) \
  for (w[n].high = 0; w[n].high <= HIGH_LIMIT(cond); w[n].high++)

#define SHIFT_LIMIT                                     \
  ((unsigned long) (tr->shift ? GMP_NUMB_BITS -1 : 1))

#define SHIFT_ITERATION                                 \
  for (shift = 1; shift <= SHIFT_LIMIT; shift++)


void
try_many (void)
{
  int   i;

  {
    unsigned long  total = 1;

    total *= option_repetitions;
    total *= option_lastsize;
    if (tr->size2 == SIZE_FRACTION) total *= FRACTION_COUNT;
    else if (tr->size2)             total *= (option_lastsize+1)/2;

    total *= SHIFT_LIMIT;
    total *= MULTIPLIER_COUNT;
    total *= DIVISOR_COUNT;
    total *= CARRY_COUNT;
    total *= T_RAND_COUNT;

    total *= HIGH_COUNT (tr->dst[0]);
    total *= HIGH_COUNT (tr->dst[1]);
    total *= HIGH_COUNT (tr->src[0]);
    total *= HIGH_COUNT (tr->src[1]);

    total *= ALIGN_COUNT (tr->dst[0]);
    total *= ALIGN_COUNT (tr->dst[1]);
    total *= ALIGN_COUNT (tr->src[0]);
    total *= ALIGN_COUNT (tr->src[1]);

    total *= OVERLAP_COUNT;

    printf ("%s %lu\n", choice->name, total);
  }

  spinner_count = 0;

  for (i = 0; i < option_repetitions; i++)
    SIZE_ITERATION
      SIZE2_ITERATION

      SHIFT_ITERATION
      MULTIPLIER_ITERATION
      DIVISOR_ITERATION
      CARRY_ITERATION /* must be after divisor */
      T_RAND_ITERATION

      HIGH_ITERATION(d,0, tr->dst[0])
      HIGH_ITERATION(d,1, tr->dst[1])
      HIGH_ITERATION(s,0, tr->src[0])
      HIGH_ITERATION(s,1, tr->src[1])

      ALIGN_ITERATION(d,0, tr->dst[0])
      ALIGN_ITERATION(d,1, tr->dst[1])
      ALIGN_ITERATION(s,0, tr->src[0])
      ALIGN_ITERATION(s,1, tr->src[1])

      OVERLAP_ITERATION
      try_one();

  printf("\n");
}


/* Usually print_all() doesn't show much, but it might give a hint as to
   where the function was up to when it died. */
void
trap (int sig)
{
  const char *name = "noname";

  switch (sig) {
  case SIGILL:  name = "SIGILL";  break;
#ifdef SIGBUS
  case SIGBUS:  name = "SIGBUS";  break;
#endif
  case SIGSEGV: name = "SIGSEGV"; break;
  case SIGFPE:  name = "SIGFPE";  break;
  }

  printf ("\n\nSIGNAL TRAP: %s\n", name);

  switch (trap_location) {
  case TRAP_REF:
    printf ("  in reference function: %s\n", tr->reference_name);
    break;
  case TRAP_FUN:
    printf ("  in test function: %s\n", choice->name);
    print_all ();
    break;
  case TRAP_SETUPS:
    printf ("  in parameter setups\n");
    print_all ();
    break;
  default:
    printf ("  somewhere unknown\n");
    break;
  }
  exit (1);
}


void
try_init (void)
{
#if HAVE_GETPAGESIZE
  /* Prefer getpagesize() over sysconf(), since on SunOS 4 sysconf() doesn't
     know _SC_PAGESIZE. */
  pagesize = getpagesize ();
#else
#if HAVE_SYSCONF
  if ((pagesize = sysconf (_SC_PAGESIZE)) == -1)
    {
      /* According to the linux man page, sysconf doesn't set errno */
      fprintf (stderr, "Cannot get sysconf _SC_PAGESIZE\n");
      exit (1);
    }
#else
Error, error, cannot get page size
#endif
#endif

  printf ("pagesize is 0x%lX bytes\n", pagesize);

  signal (SIGILL,  trap);
#ifdef SIGBUS
  signal (SIGBUS,  trap);
#endif
  signal (SIGSEGV, trap);
  signal (SIGFPE,  trap);

  {
    int  i;

    for (i = 0; i < NUM_SOURCES; i++)
      {
	malloc_region (&s[i].region, 2*option_lastsize+ALIGNMENTS-1);
	printf ("s[%d] %p to %p (0x%lX bytes)\n",
		i, (void *) (s[i].region.ptr),
		(void *) (s[i].region.ptr + s[i].region.size),
		(long) s[i].region.size * GMP_LIMB_BYTES);
      }

#define INIT_EACH(e,es)                                                 \
    for (i = 0; i < NUM_DESTS; i++)                                     \
      {                                                                 \
	malloc_region (&e.d[i].region, 2*option_lastsize+ALIGNMENTS-1); \
	printf ("%s d[%d] %p to %p (0x%lX bytes)\n",                    \
		es, i, (void *) (e.d[i].region.ptr),			\
		(void *)  (e.d[i].region.ptr + e.d[i].region.size),	\
		(long) e.d[i].region.size * GMP_LIMB_BYTES);         \
      }

    INIT_EACH(ref, "ref");
    INIT_EACH(fun, "fun");
  }
}

int
strmatch_wild (const char *pattern, const char *str)
{
  size_t  plen, slen;

  /* wildcard at start */
  if (pattern[0] == '*')
    {
      pattern++;
      plen = strlen (pattern);
      slen = strlen (str);
      return (plen == 0
	      || (slen >= plen && memcmp (pattern, str+slen-plen, plen) == 0));
    }

  /* wildcard at end */
  plen = strlen (pattern);
  if (plen >= 1 && pattern[plen-1] == '*')
    return (memcmp (pattern, str, plen-1) == 0);

  /* no wildcards */
  return (strcmp (pattern, str) == 0);
}

void
try_name (const char *name)
{
  int  found = 0;
  int  i;

  for (i = 0; i < numberof (choice_array); i++)
    {
      if (strmatch_wild (name, choice_array[i].name))
	{
	  choice = &choice_array[i];
	  tr = &param[choice->type];
	  try_many ();
	  found = 1;
	}
    }

  if (!found)
    {
      printf ("%s unknown\n", name);
      /* exit (1); */
    }
}


void
usage (const char *prog)
{
  int  col = 0;
  int  i;

  printf ("Usage: %s [options] function...\n", prog);
  printf ("    -1        use limb data 1,2,3,etc\n");
  printf ("    -9        use limb data all 0xFF..FFs\n");
  printf ("    -a zeros  use limb data all zeros\n");
  printf ("    -a ffs    use limb data all 0xFF..FFs (same as -9)\n");
  printf ("    -a 2fd    use data 0x2FFF...FFFD\n");
  printf ("    -p        print each case tried (try this if seg faulting)\n");
  printf ("    -R        seed random numbers from time()\n");
  printf ("    -r reps   set repetitions (default %d)\n", DEFAULT_REPETITIONS);
  printf ("    -s size   starting size to test\n");
  printf ("    -S size2  starting size2 to test\n");
  printf ("    -s s1-s2  range of sizes to test\n");
  printf ("    -W        don't show the spinner (use this in gdb)\n");
  printf ("    -z        disable mprotect() redzones\n");
  printf ("Default data is refmpn_random() and refmpn_random2().\n");
  printf ("\n");
  printf ("Functions that can be tested:\n");

  for (i = 0; i < numberof (choice_array); i++)
    {
      if (col + 1 + strlen (choice_array[i].name) > 79)
	{
	  printf ("\n");
	  col = 0;
	}
      printf (" %s", choice_array[i].name);
      col += 1 + strlen (choice_array[i].name);
    }
  printf ("\n");

  exit(1);
}


int
main (int argc, char *argv[])
{
  int  i;

  /* unbuffered output */
  setbuf (stdout, NULL);
  setbuf (stderr, NULL);

  /* default trace in hex, and in upper-case so can paste into bc */
  mp_trace_base = -16;

  param_init ();

  {
    unsigned long  seed = 123;
    int   opt;

    while ((opt = getopt(argc, argv, "19a:b:E:pRr:S:s:Wz")) != EOF)
      {
	switch (opt) {
	case '1':
	  /* use limb data values 1, 2, 3, ... etc */
	  option_data = DATA_SEQ;
	  break;
	case '9':
	  /* use limb data values 0xFFF...FFF always */
	  option_data = DATA_FFS;
	  break;
	case 'a':
	  if (strcmp (optarg, "zeros") == 0)     option_data = DATA_ZEROS;
	  else if (strcmp (optarg, "seq") == 0)  option_data = DATA_SEQ;
	  else if (strcmp (optarg, "ffs") == 0)  option_data = DATA_FFS;
	  else if (strcmp (optarg, "2fd") == 0)  option_data = DATA_2FD;
	  else
	    {
	      fprintf (stderr, "unrecognised data option: %s\n", optarg);
	      exit (1);
	    }
	  break;
	case 'b':
	  mp_trace_base = atoi (optarg);
	  break;
	case 'E':
	  /* re-seed */
	  sscanf (optarg, "%lu", &seed);
	  printf ("Re-seeding with %lu\n", seed);
	  break;
	case 'p':
	  option_print = 1;
	  break;
	case 'R':
	  /* randomize */
	  seed = time (NULL);
	  printf ("Seeding with %lu, re-run using \"-E %lu\"\n", seed, seed);
	  break;
	case 'r':
	  option_repetitions = atoi (optarg);
	  break;
	case 's':
	  {
	    char  *p;
	    option_firstsize = strtol (optarg, 0, 0);
	    if ((p = strchr (optarg, '-')) != NULL)
	      option_lastsize = strtol (p+1, 0, 0);
	  }
	  break;
	case 'S':
	  /* -S <size> sets the starting size for the second of a two size
	     routine (like mpn_mul_basecase) */
	  option_firstsize2 = strtol (optarg, 0, 0);
	  break;
	case 'W':
	  /* use this when running in the debugger */
	  option_spinner = 0;
	  break;
	case 'z':
	  /* disable redzones */
	  option_redzones = 0;
	  break;
	case '?':
	  usage (argv[0]);
	  break;
	}
      }

    gmp_randinit_default (__gmp_rands);
    __gmp_rands_initialized = 1;
    gmp_randseed_ui (__gmp_rands, seed);
  }

  try_init();

  if (argc <= optind)
    usage (argv[0]);

  for (i = optind; i < argc; i++)
    try_name (argv[i]);

  return 0;
}
