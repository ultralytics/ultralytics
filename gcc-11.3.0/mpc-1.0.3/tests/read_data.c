/* read_data,c -- Read data file and check function.

Copyright (C) 2008, 2009, 2010, 2011, 2012 INRIA

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

#include <stdlib.h>
#include <string.h>
#include "mpc-tests.h"

char *pathname;
unsigned long line_number;
   /* file name with complete path and currently read line;
      kept globally to simplify parameter passing */
unsigned long test_line_number;
   /* start line of data test (which may extend over several lines) */
int nextchar;
   /* character appearing next in the file, may be EOF */

#define MPC_INEX_CMP(r, i, c)                                 \
  (((r) == TERNARY_NOT_CHECKED || (r) == MPC_INEX_RE(c))      \
   && ((i) == TERNARY_NOT_CHECKED || (i) == MPC_INEX_IM (c)))

#define MPFR_INEX_STR(inex)                     \
  (inex) == TERNARY_NOT_CHECKED ? "?"           \
    : (inex) == +1 ? "+1"                       \
    : (inex) == -1 ? "-1" : "0"

static const char *mpfr_rnd_mode [] =
  { "GMP_RNDN", "GMP_RNDZ", "GMP_RNDU", "GMP_RNDD" };

const char *rnd_mode[] =
  { "MPC_RNDNN", "MPC_RNDZN", "MPC_RNDUN", "MPC_RNDDN",
    "undefined", "undefined", "undefined", "undefined", "undefined",
    "undefined", "undefined", "undefined", "undefined", "undefined",
    "undefined", "undefined",
    "MPC_RNDNZ", "MPC_RNDZZ", "MPC_RNDUZ", "MPC_RNDDZ",
    "undefined", "undefined", "undefined", "undefined", "undefined",
    "undefined", "undefined", "undefined", "undefined", "undefined",
    "undefined", "undefined",
    "MPC_RNDNU", "MPC_RNDZU", "MPC_RNDUU", "MPC_RNDDU",
    "undefined", "undefined", "undefined", "undefined", "undefined",
    "undefined", "undefined", "undefined", "undefined", "undefined",
    "undefined", "undefined",
    "MPC_RNDND", "MPC_RNDZD", "MPC_RNDUD", "MPC_RNDDD",
    "undefined", "undefined", "undefined", "undefined", "undefined",
    "undefined", "undefined", "undefined", "undefined", "undefined",
    "undefined", "undefined",
  };

/* file functions */
FILE *
open_data_file (const char *file_name)
{
  FILE *fp;
  char *src_dir;
  char default_srcdir[] = ".";

  src_dir = getenv ("srcdir");
  if (src_dir == NULL)
    src_dir = default_srcdir;

  pathname = (char *) malloc ((strlen (src_dir)) + strlen (file_name) + 2);
  if (pathname == NULL)
    {
      printf ("Cannot allocate memory\n");
      exit (1);
    }
  sprintf (pathname, "%s/%s", src_dir, file_name);
  fp = fopen (pathname, "r");
  if (fp == NULL)
    {
      fprintf (stderr, "Unable to open %s\n", pathname);
      exit (1);
    }

  return fp;
}

void
close_data_file (FILE *fp)
{
  free (pathname);
  fclose (fp);
}

/* read primitives */
static void
skip_line (FILE *fp)
   /* skips characters until reaching '\n' or EOF; */
   /* '\n' is skipped as well                      */
{
   while (nextchar != EOF && nextchar != '\n')
     nextchar = getc (fp);
   if (nextchar != EOF)
     {
       line_number ++;
       nextchar = getc (fp);
     }
}

static void
skip_whitespace (FILE *fp)
   /* skips over whitespace if any until reaching EOF */
   /* or non-whitespace                               */
{
   while (isspace (nextchar))
     {
       if (nextchar == '\n')
         line_number ++;
       nextchar = getc (fp);
     }
}

void
skip_whitespace_comments (FILE *fp)
   /* skips over all whitespace and comments, if any */
{
   skip_whitespace (fp);
   while (nextchar == '#') {
      skip_line (fp);
      if (nextchar != EOF)
         skip_whitespace (fp);
   }
}


size_t
read_string (FILE *fp, char **buffer_ptr, size_t buffer_length, const char *name)
{
  size_t pos;
  char *buffer;

  pos = 0;
  buffer = *buffer_ptr;

  if (nextchar == '"')
    nextchar = getc (fp);
  else
    goto error;

  while (nextchar != EOF && nextchar != '"')
    {
      if (nextchar == '\n')
        line_number ++;
      if (pos + 1 > buffer_length)
        {
          buffer = (char *) realloc (buffer, 2 * buffer_length);
          if (buffer == NULL)
            {
              printf ("Cannot allocate memory\n");
              exit (1);
            }
          buffer_length *= 2;
        }
      buffer[pos++] = (char) nextchar;
      nextchar = getc (fp);
    }

  if (nextchar != '"')
    goto error;

  if (pos + 1 > buffer_length)
    {
      buffer = (char *) realloc (buffer, buffer_length + 1);
      if (buffer == NULL)
        {
          printf ("Cannot allocate memory\n");
          exit (1);
        }
      buffer_length *= 2;
    }
  buffer[pos] = '\0';

  nextchar = getc (fp);
  skip_whitespace_comments (fp);

  *buffer_ptr = buffer;

  return buffer_length;

 error:
  printf ("Error: Unable to read %s in file '%s' line '%lu'\n",
          name, pathname, line_number);
  exit (1);
}

/* All following read routines skip over whitespace and comments; */
/* so after calling them, nextchar is either EOF or the beginning */
/* of a non-comment token.                                        */
void
read_ternary (FILE *fp, int* ternary)
{
  switch (nextchar)
    {
    case '!':
      *ternary = TERNARY_ERROR;
      break;
    case '?':
      *ternary = TERNARY_NOT_CHECKED;
      break;
    case '+':
      *ternary = +1;
      break;
    case '0':
      *ternary = 0;
      break;
    case '-':
      *ternary = -1;
      break;
    default:
      printf ("Error: Unexpected ternary value '%c' in file '%s' line %lu\n",
              nextchar, pathname, line_number);
      exit (1);
    }

  nextchar = getc (fp);
  skip_whitespace_comments (fp);
}

void
read_mpfr_rounding_mode (FILE *fp, mpfr_rnd_t* rnd)
{
  switch (nextchar)
    {
    case 'n': case 'N':
      *rnd = GMP_RNDN;
      break;
    case 'z': case 'Z':
      *rnd = GMP_RNDZ;
      break;
    case 'u': case 'U':
      *rnd = GMP_RNDU;
      break;
    case 'd': case 'D':
      *rnd = GMP_RNDD;
      break;
    default:
      printf ("Error: Unexpected rounding mode '%c' in file '%s' line %lu\n",
              nextchar, pathname, line_number);
      exit (1);
    }

    nextchar = getc (fp);
    if (nextchar != EOF && !isspace (nextchar)) {
      printf ("Error: Rounding mode not followed by white space in file "
              "'%s' line %lu\n",
              pathname, line_number);
      exit (1);
    }
    skip_whitespace_comments (fp);
}

void
read_mpc_rounding_mode (FILE *fp, mpc_rnd_t* rnd)
{
   mpfr_rnd_t re, im;
   read_mpfr_rounding_mode (fp, &re);
   read_mpfr_rounding_mode (fp, &im);
   *rnd = MPC_RND (re, im);
}

void
read_int (FILE *fp, int *nread, const char *name)
{
  int n = 0;

  if (nextchar == EOF)
    {
      printf ("Error: Unexpected EOF when reading int "
              "in file '%s' line %lu\n",
              pathname, line_number);
      exit (1);
    }
  ungetc (nextchar, fp);
  n = fscanf (fp, "%i", nread);
  if (ferror (fp) || n == 0 || n == EOF)
    {
      printf ("Error: Cannot read %s in file '%s' line %lu\n",
              name, pathname, line_number);
      exit (1);
    }
  nextchar = getc (fp);
  skip_whitespace_comments (fp);
}

static void
read_uint (FILE *fp, unsigned long int *ui)
{
  int n = 0;

  if (nextchar == EOF)
    {
      printf ("Error: Unexpected EOF when reading uint "
              "in file '%s' line %lu\n",
              pathname, line_number);
      exit (1);
    }
  ungetc (nextchar, fp);
  n = fscanf (fp, "%lu", ui);
  if (ferror (fp) || n == 0 || n == EOF)
    {
      printf ("Error: Cannot read uint in file '%s' line %lu\n",
              pathname, line_number);
      exit (1);
    }
  nextchar = getc (fp);
  skip_whitespace_comments (fp);
}

static void
read_sint (FILE *fp, long int *si)
{
  int n = 0;

  if (nextchar == EOF)
    {
      printf ("Error: Unexpected EOF when reading sint "
              "in file '%s' line %lu\n",
              pathname, line_number);
      exit (1);
    }
  ungetc (nextchar, fp);
  n = fscanf (fp, "%li", si);
  if (ferror (fp) || n == 0 || n == EOF)
    {
      printf ("Error: Cannot read sint in file '%s' line %lu\n",
              pathname, line_number);
      exit (1);
    }
  nextchar = getc (fp);
  skip_whitespace_comments (fp);
}

mpfr_prec_t
read_mpfr_prec (FILE *fp)
{
   unsigned long prec;
   int n;

   if (nextchar == EOF) {
      printf ("Error: Unexpected EOF when reading mpfr precision "
              "in file '%s' line %lu\n",
              pathname, line_number);
      exit (1);
   }
   ungetc (nextchar, fp);
   n = fscanf (fp, "%lu", &prec);
   if (ferror (fp)) /* then also n == EOF */
      perror ("Error when reading mpfr precision");
   if (n == 0 || n == EOF || prec < MPFR_PREC_MIN || prec > MPFR_PREC_MAX) {
      printf ("Error: Impossible mpfr precision in file '%s' line %lu\n",
              pathname, line_number);
      exit (1);
   }
   nextchar = getc (fp);
   skip_whitespace_comments (fp);
   return (mpfr_prec_t) prec;
}

static void
read_mpfr_mantissa (FILE *fp, mpfr_ptr x)
{
   if (nextchar == EOF) {
      printf ("Error: Unexpected EOF when reading mpfr mantissa "
              "in file '%s' line %lu\n",
              pathname, line_number);
      exit (1);
   }
   ungetc (nextchar, fp);
   if (mpfr_inp_str (x, fp, 0, GMP_RNDN) == 0) {
      printf ("Error: Impossible to read mpfr mantissa "
              "in file '%s' line %lu\n",
              pathname, line_number);
      exit (1);
   }
   nextchar = getc (fp);
   skip_whitespace_comments (fp);
}

void
read_mpfr (FILE *fp, mpfr_ptr x, int *known_sign)
{
   int sign;
   mpfr_set_prec (x, read_mpfr_prec (fp));
   sign = nextchar;
   read_mpfr_mantissa (fp, x);

   /* the sign always matters for regular values ('+' is implicit),
      but when no sign appears before 0 or Inf in the data file, it means
      that only absolute value must be checked. */
   if (known_sign != NULL)
     *known_sign =
       (!mpfr_zero_p (x) && !mpfr_inf_p (x))
       || sign == '+' || sign == '-';
}

void
read_mpc (FILE *fp, mpc_ptr z, known_signs_t *ks)
{
  read_mpfr (fp, mpc_realref (z), ks == NULL ? NULL : &ks->re);
  read_mpfr (fp, mpc_imagref (z), ks == NULL ? NULL : &ks->im);
}

static void
check_compatible (int inex, mpfr_t expected, mpfr_rnd_t rnd, const char *s)
{
  if ((rnd == GMP_RNDU && inex == -1) ||
      (rnd == GMP_RNDD && inex == +1) ||
      (rnd == GMP_RNDZ && !mpfr_signbit (expected) && inex == +1) ||
      (rnd == GMP_RNDZ && mpfr_signbit (expected) && inex == -1))
    {
      if (s != NULL)
        printf ("Incompatible ternary value '%c' (%s part) in file '%s' line %lu\n",
              (inex == 1) ? '+' : '-', s, pathname, test_line_number);
      else
        printf ("Incompatible ternary value '%c' in file '%s' line %lu\n",
              (inex == 1) ? '+' : '-', pathname, test_line_number);
    }
}

/* read lines of data */
static void
read_cc (FILE *fp, int *inex_re, int *inex_im, mpc_ptr expected,
         known_signs_t *signs, mpc_ptr op, mpc_rnd_t *rnd)
{
  test_line_number = line_number;
  read_ternary (fp, inex_re);
  read_ternary (fp, inex_im);
  read_mpc (fp, expected, signs);
  read_mpc (fp, op, NULL);
  read_mpc_rounding_mode (fp, rnd);
  check_compatible (*inex_re, mpc_realref(expected), MPC_RND_RE(*rnd), "real");
  check_compatible (*inex_im, mpc_imagref(expected), MPC_RND_IM(*rnd), "imag");
}

static void
read_fc (FILE *fp, int *inex, mpfr_ptr expected, int *sign, mpc_ptr op,
         mpfr_rnd_t *rnd)
{
  test_line_number = line_number;
  read_ternary (fp, inex);
  read_mpfr (fp, expected, sign);
  read_mpc (fp, op, NULL);
  read_mpfr_rounding_mode (fp, rnd);
  check_compatible (*inex, expected, *rnd, NULL);
}

static void
read_ccc (FILE *fp, int *inex_re, int *inex_im, mpc_ptr expected,
          known_signs_t *signs, mpc_ptr op1, mpc_ptr op2, mpc_rnd_t *rnd)
{
  test_line_number = line_number;
  read_ternary (fp, inex_re);
  read_ternary (fp, inex_im);
  read_mpc (fp, expected, signs);
  read_mpc (fp, op1, NULL);
  read_mpc (fp, op2, NULL);
  read_mpc_rounding_mode (fp, rnd);
  check_compatible (*inex_re, mpc_realref(expected), MPC_RND_RE(*rnd), "real");
  check_compatible (*inex_im, mpc_imagref(expected), MPC_RND_IM(*rnd), "imag");
}

/* read lines of data for function with three mpc_t inputs and one mpc_t
   output like mpc_fma */
static void
read_cccc (FILE *fp, int *inex_re, int *inex_im, mpc_ptr expected,
	   known_signs_t *signs, mpc_ptr op1, mpc_ptr op2, mpc_ptr op3,
	   mpc_rnd_t *rnd)
{
  test_line_number = line_number;
  read_ternary (fp, inex_re);
  read_ternary (fp, inex_im);
  read_mpc (fp, expected, signs);
  read_mpc (fp, op1, NULL);
  read_mpc (fp, op2, NULL);
  read_mpc (fp, op3, NULL);
  read_mpc_rounding_mode (fp, rnd);
  check_compatible (*inex_re, mpc_realref(expected), MPC_RND_RE(*rnd), "real");
  check_compatible (*inex_im, mpc_imagref(expected), MPC_RND_IM(*rnd), "imag");
}

static void
read_cfc (FILE *fp, int *inex_re, int *inex_im, mpc_ptr expected,
          known_signs_t *signs, mpfr_ptr op1, mpc_ptr op2, mpc_rnd_t *rnd)
{
  test_line_number = line_number;
  read_ternary (fp, inex_re);
  read_ternary (fp, inex_im);
  read_mpc (fp, expected, signs);
  read_mpfr (fp, op1, NULL);
  read_mpc (fp, op2, NULL);
  read_mpc_rounding_mode (fp, rnd);
  check_compatible (*inex_re, mpc_realref(expected), MPC_RND_RE(*rnd), "real");
  check_compatible (*inex_im, mpc_imagref(expected), MPC_RND_IM(*rnd), "imag");
}

static void
read_ccf (FILE *fp, int *inex_re, int *inex_im, mpc_ptr expected,
          known_signs_t *signs, mpc_ptr op1, mpfr_ptr op2, mpc_rnd_t *rnd)
{
  test_line_number = line_number;
  read_ternary (fp, inex_re);
  read_ternary (fp, inex_im);
  read_mpc (fp, expected, signs);
  read_mpc (fp, op1, NULL);
  read_mpfr (fp, op2, NULL);
  read_mpc_rounding_mode (fp, rnd);
  check_compatible (*inex_re, mpc_realref(expected), MPC_RND_RE(*rnd), "real");
  check_compatible (*inex_im, mpc_imagref(expected), MPC_RND_IM(*rnd), "imag");
}

static void
read_ccu (FILE *fp, int *inex_re, int *inex_im, mpc_ptr expected,
          known_signs_t *signs, mpc_ptr op1, unsigned long int *op2, mpc_rnd_t *rnd)
{
  test_line_number = line_number;
  read_ternary (fp, inex_re);
  read_ternary (fp, inex_im);
  read_mpc (fp, expected, signs);
  read_mpc (fp, op1, NULL);
  read_uint (fp, op2);
  read_mpc_rounding_mode (fp, rnd);
  check_compatible (*inex_re, mpc_realref(expected), MPC_RND_RE(*rnd), "real");
  check_compatible (*inex_im, mpc_imagref(expected), MPC_RND_IM(*rnd), "imag");
}

static void
read_ccs (FILE *fp, int *inex_re, int *inex_im, mpc_ptr expected,
          known_signs_t *signs, mpc_ptr op1, long int *op2, mpc_rnd_t *rnd)
{
  test_line_number = line_number;
  read_ternary (fp, inex_re);
  read_ternary (fp, inex_im);
  read_mpc (fp, expected, signs);
  read_mpc (fp, op1, NULL);
  read_sint (fp, op2);
  read_mpc_rounding_mode (fp, rnd);
  check_compatible (*inex_re, mpc_realref(expected), MPC_RND_RE(*rnd), "real");
  check_compatible (*inex_im, mpc_imagref(expected), MPC_RND_IM(*rnd), "imag");
}

/* set MPFR flags to random values */
static void
set_mpfr_flags (int counter)
{
  if (counter & 1)
    mpfr_set_underflow ();
  else
    mpfr_clear_underflow ();
  if (counter & 2)
    mpfr_set_overflow ();
  else
    mpfr_clear_overflow ();
  /* the divide-by-0 flag was added in MPFR 3.1.0 */
#ifdef mpfr_set_divby0
  if (counter & 4)
    mpfr_set_divby0 ();
  else
    mpfr_clear_divby0 ();
#endif
  if (counter & 8)
    mpfr_set_nanflag ();
  else
    mpfr_clear_nanflag ();
  if (counter & 16)
    mpfr_set_inexflag ();
  else
    mpfr_clear_inexflag ();
  if (counter & 32)
    mpfr_set_erangeflag ();
  else
    mpfr_clear_erangeflag ();
}

/* Check MPFR flags: we allow that some flags are set internally by MPC,
   for example if MPC does internal computations (using MPFR) which yield
   an overflow, even if the final MPC result fits in the exponent range.
   However we don't allow MPC to *clear* the MPFR flags */
static void
check_mpfr_flags (int counter)
{
  int old, neu;

  old = (counter & 1) != 0;
  neu = mpfr_underflow_p () != 0;
  if (old && (neu == 0))
    {
      printf ("Error, underflow flag has been modified from %d to %d\n",
              old, neu);
      exit (1);
    }
  old = (counter & 2) != 0;
  neu = mpfr_overflow_p () != 0;
  if (old && (neu == 0))
    {
      printf ("Error, overflow flag has been modified from %d to %d\n",
              old, neu);
      exit (1);
    }
#ifdef mpfr_divby0_p
  old = (counter & 4) != 0;
  neu = mpfr_divby0_p () != 0;
  if (old && (neu == 0))
    {
      printf ("Error, divby0 flag has been modified from %d to %d\n",
              old, neu);
      exit (1);
    }
#endif
  old = (counter & 8) != 0;
  neu = mpfr_nanflag_p () != 0;
  if (old && (neu == 0))
    {
      printf ("Error, nanflag flag has been modified from %d to %d\n",
              old, neu);
      exit (1);
    }
  old = (counter & 16) != 0;
  neu = mpfr_inexflag_p () != 0;
  if (old && (neu == 0))
    {
      printf ("Error, inexflag flag has been modified from %d to %d\n",
              old, neu);
      exit (1);
    }
  old = (counter & 32) != 0;
  neu = mpfr_erangeflag_p () != 0;
  if (old && (neu == 0))
    {
      printf ("Error, erangeflag flag has been modified from %d to %d\n",
              old, neu);
      exit (1);
    }
}

/* data_check (function, data_file_name) checks function results against
 precomputed data in a file.*/
void
data_check (mpc_function function, const char *file_name)
{
  FILE *fp;

  int inex_re;
  mpfr_t x1, x2;
  mpfr_rnd_t mpfr_rnd = GMP_RNDN;
  int sign_real;

  int inex_im;
  mpc_t z1, z2, z3, z4, z5;
  mpc_rnd_t rnd = MPC_RNDNN;

  unsigned long int ui;
  long int si;

  known_signs_t signs;
  int inex = 0;

  static int rand_counter = 0;

  fp = open_data_file (file_name);

  /* 1. init needed variables */
  mpc_init2 (z1, 2);
  switch (function.type)
    {
    case FC:
      mpfr_init (x1);
      mpfr_init (x2);
      break;
    case CC: case CCU: case CCS:
      mpc_init2 (z2, 2);
      mpc_init2 (z3, 2);
      break;
    case C_CC:
      mpc_init2 (z2, 2);
      mpc_init2 (z3, 2);
      mpc_init2 (z4, 2);
      break;
    case CCCC:
      mpc_init2 (z2, 2);
      mpc_init2 (z3, 2);
      mpc_init2 (z4, 2);
      mpc_init2 (z5, 2);
      break;
    case CFC: case CCF:
      mpfr_init (x1);
      mpc_init2 (z2, 2);
      mpc_init2 (z3, 2);
      break;
    default:
      ;
    }

  /* 2. read data file */
  line_number = 1;
  nextchar = getc (fp);
  skip_whitespace_comments (fp);
  while (nextchar != EOF) {
      set_mpfr_flags (rand_counter);

      /* for each kind of function prototype: */
      /* 3.1 read a line of data: expected result, parameters, rounding mode */
      /* 3.2 compute function at the same precision as the expected result */
      /* 3.3 compare this result with the expected one */
      switch (function.type)
        {
        case FC: /* example mpc_norm */
          read_fc (fp, &inex_re, x1, &sign_real, z1, &mpfr_rnd);
          mpfr_set_prec (x2, mpfr_get_prec (x1));
          inex = function.pointer.FC (x2, z1, mpfr_rnd);
          if ((inex_re != TERNARY_NOT_CHECKED && inex_re != inex)
              || !same_mpfr_value (x1, x2, sign_real))
            {
              mpfr_t got, expected;
              mpc_t op;
              op[0] = z1[0];
              got[0] = x2[0];
              expected[0] = x1[0];
              printf ("%s(op) failed (%s:%lu)\nwith rounding mode %s\n",
                      function.name, file_name, test_line_number,
                      mpfr_rnd_mode[mpfr_rnd]);
              if (inex_re != TERNARY_NOT_CHECKED && inex_re != inex)
                printf("ternary value: got %s, expected %s\n",
                       MPFR_INEX_STR (inex), MPFR_INEX_STR (inex_re));
              MPC_OUT (op);
              printf ("     ");
              MPFR_OUT (got);
              MPFR_OUT (expected);

              exit (1);
            }
          break;

        case CC: /* example mpc_log */
          read_cc (fp, &inex_re, &inex_im, z1, &signs, z2, &rnd);
          mpfr_set_prec (mpc_realref (z3), MPC_PREC_RE (z1));
          mpfr_set_prec (mpc_imagref (z3), MPC_PREC_IM (z1));
          inex = function.pointer.CC (z3, z2, rnd);
          if (!MPC_INEX_CMP (inex_re, inex_im, inex)
              || !same_mpc_value (z3, z1, signs))
            {
              mpc_t op, got, expected; /* display sensible variable names */
              op[0] = z2[0];
              expected[0]= z1[0];
              got[0] = z3[0];
              printf ("%s(op) failed (line %lu)\nwith rounding mode %s\n",
                      function.name, test_line_number, rnd_mode[rnd]);
              if (!MPC_INEX_CMP (inex_re, inex_im, inex))
                printf("ternary value: got %s, expected (%s, %s)\n",
                       MPC_INEX_STR (inex),
                       MPFR_INEX_STR (inex_re), MPFR_INEX_STR (inex_im));
              MPC_OUT (op);
              printf ("     ");
              MPC_OUT (got);
              MPC_OUT (expected);

              exit (1);
            }
          break;

        case C_CC: /* example mpc_mul */
          read_ccc (fp, &inex_re, &inex_im, z1, &signs, z2, z3, &rnd);
          mpfr_set_prec (mpc_realref(z4), MPC_PREC_RE (z1));
          mpfr_set_prec (mpc_imagref(z4), MPC_PREC_IM (z1));
          inex = function.pointer.C_CC (z4, z2, z3, rnd);
          if (!MPC_INEX_CMP (inex_re, inex_im, inex)
              || !same_mpc_value (z4, z1, signs))
            {
              /* display sensible variable names */
              mpc_t op1, op2, got, expected;
              op1[0] = z2[0];
              op2[0] = z3[0];
              expected[0]= z1[0];
              got[0] = z4[0];
              printf ("%s(op) failed (line %lu)\nwith rounding mode %s\n",
                      function.name, test_line_number, rnd_mode[rnd]);
              if (!MPC_INEX_CMP (inex_re, inex_im, inex))
                printf("ternary value: got %s, expected (%s, %s)\n",
                       MPC_INEX_STR (inex),
                       MPFR_INEX_STR (inex_re), MPFR_INEX_STR (inex_im));
              MPC_OUT (op1);
              MPC_OUT (op2);
              printf ("     ");
              MPC_OUT (got);
              MPC_OUT (expected);

              exit (1);
            }
          if (function.properties & FUNC_PROP_SYMETRIC)
            {
              inex = function.pointer.C_CC (z4, z3, z2, rnd);
              if (!MPC_INEX_CMP (inex_re, inex_im, inex)
              || !same_mpc_value (z4, z1, signs))
                {
                  /* display sensible variable names */
                  mpc_t op1, op2, got, expected;
                  op1[0] = z3[0];
                  op2[0] = z2[0];
                  expected[0]= z1[0];
                  got[0] = z4[0];
                  printf ("%s(op) failed (line %lu/symetric test)\n"
                          "with rounding mode %s\n",
                          function.name, test_line_number, rnd_mode[rnd]);
                  if (!MPC_INEX_CMP (inex_re, inex_im, inex))
                    printf("ternary value: got %s, expected (%s, %s)\n",
                           MPC_INEX_STR (inex),
                           MPFR_INEX_STR (inex_re), MPFR_INEX_STR (inex_im));
                  MPC_OUT (op1);
                  MPC_OUT (op2);
                  printf ("     ");
                  MPC_OUT (got);
                  MPC_OUT (expected);

                  exit (1);
                }
            }
          break;

        case CCCC: /* example mpc_fma */
          read_cccc (fp, &inex_re, &inex_im, z1, &signs, z2, z3, z4, &rnd);
	  /* z1 is the expected value, z2, z3, z4 are the inputs, and z5 is
	     the computed value */
          mpfr_set_prec (mpc_realref(z5), MPC_PREC_RE (z1));
          mpfr_set_prec (mpc_imagref(z5), MPC_PREC_IM (z1));
          inex = function.pointer.CCCC (z5, z2, z3, z4, rnd);
          if (!MPC_INEX_CMP (inex_re, inex_im, inex)
              || !same_mpc_value (z5, z1, signs))
            {
              /* display sensible variable names */
              mpc_t op1, op2, op3, got, expected;
              op1[0] = z2[0];
              op2[0] = z3[0];
              op3[0] = z4[0];
              expected[0]= z1[0];
              got[0] = z5[0];
              printf ("%s(op) failed (line %lu)\nwith rounding mode %s\n",
                      function.name, test_line_number, rnd_mode[rnd]);
              if (!MPC_INEX_CMP (inex_re, inex_im, inex))
                printf("ternary value: got %s, expected (%s, %s)\n",
                       MPC_INEX_STR (inex),
                       MPFR_INEX_STR (inex_re), MPFR_INEX_STR (inex_im));
              MPC_OUT (op1);
              MPC_OUT (op2);
              MPC_OUT (op3);
              printf ("     ");
              MPC_OUT (got);
              MPC_OUT (expected);

              exit (1);
            }
          if (function.properties & FUNC_PROP_SYMETRIC)
            {
              inex = function.pointer.CCCC (z5, z3, z2, z4, rnd);
              if (!MPC_INEX_CMP (inex_re, inex_im, inex)
              || !same_mpc_value (z5, z1, signs))
                {
                  /* display sensible variable names */
                  mpc_t op1, op2, op3, got, expected;
                  op1[0] = z3[0];
                  op2[0] = z2[0];
		  op3[0] = z4[0];
                  expected[0]= z1[0];
                  got[0] = z5[0];
                  printf ("%s(op) failed (line %lu/symetric test)\n"
                          "with rounding mode %s\n",
                          function.name, test_line_number, rnd_mode[rnd]);
                  if (!MPC_INEX_CMP (inex_re, inex_im, inex))
                    printf("ternary value: got %s, expected (%s, %s)\n",
                           MPC_INEX_STR (inex),
                           MPFR_INEX_STR (inex_re), MPFR_INEX_STR (inex_im));
                  MPC_OUT (op1);
                  MPC_OUT (op2);
		  MPC_OUT (op3);
                  printf ("     ");
                  MPC_OUT (got);
                  MPC_OUT (expected);

                  exit (1);
                }
            }
          break;

        case CFC: /* example mpc_fr_div */
          read_cfc (fp, &inex_re, &inex_im, z1, &signs, x1, z2, &rnd);
          mpfr_set_prec (mpc_realref(z3), MPC_PREC_RE (z1));
          mpfr_set_prec (mpc_imagref(z3), MPC_PREC_IM (z1));
          inex = function.pointer.CFC (z3, x1, z2, rnd);
          if (!MPC_INEX_CMP (inex_re, inex_im, inex)
              || !same_mpc_value (z3, z1, signs))
            {
              /* display sensible variable names */
              mpc_t op2, got, expected;
              mpfr_t op1;
              op1[0] = x1[0];
              op2[0] = z2[0];
              expected[0]= z1[0];
              got[0] = z3[0];
              printf ("%s(op) failed (line %lu)\nwith rounding mode %s\n",
                      function.name, test_line_number, rnd_mode[rnd]);
              if (!MPC_INEX_CMP (inex_re, inex_im, inex))
                printf("ternary value: got %s, expected (%s, %s)\n",
                       MPC_INEX_STR (inex),
                       MPFR_INEX_STR (inex_re), MPFR_INEX_STR (inex_im));
              MPFR_OUT (op1);
              MPC_OUT (op2);
              printf ("     ");
              MPC_OUT (got);
              MPC_OUT (expected);

              exit (1);
            }
          break;

        case CCF: /* example mpc_mul_fr */
          read_ccf (fp, &inex_re, &inex_im, z1, &signs, z2, x1, &rnd);
          mpfr_set_prec (mpc_realref(z3), MPC_PREC_RE (z1));
          mpfr_set_prec (mpc_imagref(z3), MPC_PREC_IM (z1));
          inex = function.pointer.CCF (z3, z2, x1, rnd);
          if (!MPC_INEX_CMP (inex_re, inex_im, inex)
              || !same_mpc_value (z3, z1, signs))
            {
              /* display sensible variable names */
              mpc_t op1, got, expected;
              mpfr_t op2;
              op1[0] = z2[0];
              op2[0] = x1[0];
              expected[0]= z1[0];
              got[0] = z3[0];
              printf ("%s(op) failed (line %lu)\nwith rounding mode %s\n",
                      function.name, test_line_number, rnd_mode[rnd]);
              if (!MPC_INEX_CMP (inex_re, inex_im, inex))
                printf("ternary value: got %s, expected (%s, %s)\n",
                       MPC_INEX_STR (inex),
                       MPFR_INEX_STR (inex_re), MPFR_INEX_STR (inex_im));
              MPC_OUT (op1);
              MPFR_OUT (op2);
              printf ("     ");
              MPC_OUT (got);
              MPC_OUT (expected);

              exit (1);
            }
          break;

        case CCU: /* example mpc_pow_ui */
          read_ccu (fp, &inex_re, &inex_im, z1, &signs, z2, &ui, &rnd);
          mpfr_set_prec (mpc_realref(z3), MPC_PREC_RE (z1));
          mpfr_set_prec (mpc_imagref(z3), MPC_PREC_IM (z1));
          inex = function.pointer.CCU (z3, z2, ui, rnd);
          if (!MPC_INEX_CMP (inex_re, inex_im, inex)
              || !same_mpc_value (z3, z1, signs))
            {
              /* display sensible variable names */
              mpc_t op1, got, expected;
              op1[0] = z2[0];
              expected[0]= z1[0];
              got[0] = z3[0];
              printf ("%s(op) failed (line %lu)\nwith rounding mode %s\n",
                      function.name, test_line_number, rnd_mode[rnd]);
              if (!MPC_INEX_CMP (inex_re, inex_im, inex))
                printf("ternary value: got %s, expected (%s, %s)\n",
                       MPC_INEX_STR (inex),
                       MPFR_INEX_STR (inex_re), MPFR_INEX_STR (inex_im));
              MPC_OUT (op1);
              printf ("op2 %lu\n     ", ui);
              MPC_OUT (got);
              MPC_OUT (expected);

              exit (1);
            }
          break;

        case CCS: /* example mpc_pow_si */
          read_ccs (fp, &inex_re, &inex_im, z1, &signs, z2, &si, &rnd);
          mpfr_set_prec (mpc_realref(z3), MPC_PREC_RE (z1));
          mpfr_set_prec (mpc_imagref(z3), MPC_PREC_IM (z1));
          inex = function.pointer.CCS (z3, z2, si, rnd);
          if (!MPC_INEX_CMP (inex_re, inex_im, inex)
              || !same_mpc_value (z3, z1, signs))
            {
              /* display sensible variable names */
              mpc_t op1, got, expected;
              op1[0] = z2[0];
              expected[0]= z1[0];
              got[0] = z3[0];
              printf ("%s(op) failed (line %lu)\nwith rounding mode %s\n",
                      function.name, test_line_number, rnd_mode[rnd]);
              if (!MPC_INEX_CMP (inex_re, inex_im, inex))
                printf("ternary value: got %s, expected (%s, %s)\n",
                       MPC_INEX_STR (inex),
                       MPFR_INEX_STR (inex_re), MPFR_INEX_STR (inex_im));
              MPC_OUT (op1);
              printf ("op2 %li\n     ", si);
              MPC_OUT (got);
              MPC_OUT (expected);

              exit (1);
            }
          break;

        default:
          printf ("Unhandled function prototype %i in 'data_check'\n", function.type);
          exit (1);
        }

      /* check MPFR flags were not modified */
      check_mpfr_flags (rand_counter);
      rand_counter ++;
    }

  /* 3. Clear used variables */
  mpc_clear (z1);
  switch (function.type)
    {
    case FC:
      mpfr_clear (x1);
      mpfr_clear (x2);
      break;
    case CC: case CCU: case CCS:
      mpc_clear (z2);
      mpc_clear (z3);
      break;
    case C_CC:
      mpc_clear (z2);
      mpc_clear (z3);
      mpc_clear (z4);
      break;
    case CCCC:
      mpc_clear (z2);
      mpc_clear (z3);
      mpc_clear (z4);
      mpc_clear (z5);
      break;
    case CFC: case CCF:
      mpfr_clear (x1);
      mpc_clear (z2);
      mpc_clear (z3);
      break;
    default:
      ;
    }

  close_data_file (fp);
}
