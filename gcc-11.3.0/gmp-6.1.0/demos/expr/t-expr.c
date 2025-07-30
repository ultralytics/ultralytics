/* Test expression evaluation (print nothing and exit 0 if successful).

Copyright 2000-2004 Free Software Foundation, Inc.

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

#include <stdio.h>
#include <stdlib.h>

#include "gmp.h"
#include "tests.h"
#include "expr-impl.h"


int  option_trace = 0;


struct data_t {
  int         base;
  const char  *expr;
  const char  *want;
};

#define numberof(x)  (sizeof (x) / sizeof ((x)[0]))


/* These data_xxx[] arrays are tables to be tested with one or more of the
   mp?_t types.  z=mpz_t, q=mpz_t, f=mpf_t.  */

struct data_t  data_zqf[] = {

  /* various deliberately wrong expressions */
  { 0, "", NULL },
  { 0, "1+", NULL },
  { 0, "+2", NULL },
  { 0, "1,2", NULL },
  { 0, "foo(1,2)", NULL },
  { 0, "1+foo", NULL },
  { 10, "0fff", NULL },
  { 0, "!", NULL },
  { 0, "10!", NULL },
  { 0, "-10!", NULL },
  { 0, "gcd((4,6))", NULL },
  { 0, "()", NULL },
  { 0, "fac(2**1000)", NULL },
  { 0, "$", NULL },
  { 0, "$-", NULL },

  /* some basics */
  { 10, "123", "123" },
  { 10, "-123", "-123" },
  { 10, "1+2", "3" },
  { 10, "1+2+3", "6" },
  { 10, "1+2*3", "7" },
  { 10, "3*2+1", "7" },
  { 10, "$a", "55" },
  { 10, "b", "99" },
  { 16, "b", "11" },
  { 10, "4**3 * 2 + 1", "129" },
  { 10, "1<2", "1" },
  { 10, "1>2", "0" },

  { 10, "(123)", "123" },

  { 10, "sgn(-123)", "-1" },
  { 10, "5-7", "-2" },

  { 0, "cmp(0,0)", "0" },
  { 0, "cmp(1,0)", "1" },
  { 0, "cmp(0,1)", "-1" },
  { 0, "cmp(-1,0)", "-1" },
  { 0, "cmp(0,-1)", "1" },

  { 10, "0 ? 123 : 456", "456" },
  { 10, "1 ? 4+5 : 6+7", "9" },

  { 10, "(123)", "123" },
  { 10, "(2+3)", "5" },
  { 10, "(4+5)*(5+6)", "99" },

  { 0, "1 << 16", "65536" },
  { 0, "256 >> 4", "16" },
  { 0, "-256 >> 4", "-16" },

  { 0, "!1", "0" },
  { 0, "!9", "0" },
  { 0, "!0", "1" },

  { 0, "2**2**2", "16" },
  { 0, "-2**2**2", "-16" },

  { 0, "0x100", "256" },
  { 10, "0x100", NULL },
  { 10, "0x 100", NULL },

  { 0, " max ( 1, 2, 3, 4, 5, 6, 7, 8)", "8" },
  { 0, " max ( 1, 9, 2, 3, 4, 5, 6, 7, 8)", "9" },
  { 0, " min ( 1, 9, 2, 3, 4, 5, 6, 7, 8)", "1" },

  { 10, "abs(123)",  "123" },
  { 10, "abs(-123)", "123" },
  { 10, "abs(0)",    "0" },

  /* filling data stack */
  { 0, "1+(1+(1+(1+(1+(1+(1+(1+(1+(1+(1+(1+(1+(1+(1+1))))))))))))))", "16" },

  /* filling control stack */
  { 0, "----------------------------------------------------1", "1" },
};


const struct data_t  data_z[] = {
  { 0, "divisible_p(333,3)", "1" },
  { 0, "congruent_p(7,1,3)", "1" },

  { 0, "cmpabs(0,0)", "0" },
  { 0, "cmpabs(1,0)", "1" },
  { 0, "cmpabs(0,1)", "-1" },
  { 0, "cmpabs(-1,0)", "1" },
  { 0, "cmpabs(0,-1)", "-1" },

  { 0, "odd_p(1)", "1" },
  { 0, "odd_p(0)", "0" },
  { 0, "odd_p(-1)", "1" },

  { 0, "even_p(1)", "0" },
  { 0, "even_p(0)", "1" },
  { 0, "even_p(-1)", "0" },

  { 0, "fac(0)",  "1" },
  { 0, "fac(1)",  "1" },
  { 0, "fac(2)",  "2" },
  { 0, "fac(3)",  "6" },
  { 0, "fac(10)", "3628800" },

  { 10, "root(81,4)", "3" },

  { 10, "gcd(4,6)", "2" },
  { 10, "gcd(4,6,9)", "1" },

  { 10, "powm(3,2,9)", "0" },
  { 10, "powm(3,2,8)", "1" },

  /* filling data stack */
  { 0, "1 ? 1 : 1 || 1 && 1 | 1 ^ 1 & 1 == 1 >= 1 << 1 - 1 * 1 ** 1", "1" },

  /* filling control stack */
  { 0, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~1", "1" },

  { 0, "fib(10)", "55" },

  { 0, "setbit(0,5)", "32" },
  { 0, "clrbit(32,5)", "0" },
  { 0, "tstbit(32,5)", "1" },
  { 0, "tstbit(32,4)", "0" },
  { 0, "scan0(7,0)", "3" },
  { 0, "scan1(7,0)", "0" },
};

const struct data_t  data_zq[] = {
  /* expecting failure */
  { 0, "1.2", NULL },
};

const struct data_t  data_q[] = {
  { 10,  "(1/2 + 1/3 + 1/4 + 1/5 + 1/6)*20", "29" },
  { 0, "num(5/9)", "5" },
  { 0, "den(5/9)", "9" },
};

const struct data_t  data_zf[] = {
  { 10, "sqrt ( 49 )", "7" },
  { 10, "sqrt ( 49 ) + 1", "8" },
  { 10, "sqrt((49))", "7" },
  { 10, "sqrt((((((((49))))))))", "7" },
};

const struct data_t  data_f[] = {
  { 0, "1@10",    "10000000000" },
  { 0, "1.5@10",  "15000000000" },
  { 0, "1000@-1", "100" },
  { 0, "10.00@-1", "1" },

  { 0, "1e10",     "10000000000" },
  { 0, "1.5e10",   "15000000000" },
  { 0, "1000e-1",  "100" },
  { 0, "10.00e-1", "1" },

  { 16, "1@9",  "68719476736" },

  { 16,  "1@10", "18446744073709551616" },
  { -16, "1@10", "1099511627776" },

  { 0, "ceil(0)",           "0" },
  { 0, "ceil(0.25)",        "1" },
  { 0, "ceil(0.5)",         "1" },
  { 0, "ceil(1.5)",         "2" },
  { 0, "ceil(-0.5)",        "0" },
  { 0, "ceil(-1.5)",        "-1" },

  /* only simple cases because mpf_eq currently only works on whole limbs */
  { 0, "eq(0xFFFFFFFFFFFFFFFF1111111111111111,0xFFFFFFFFFFFFFFFF2222222222222222,64)", "1" },
  { 0, "eq(0xFFFFFFFFFFFFFFFF1111111111111111,0xFFFFFFFFFFFFFFFF2222222222222222,128)", "0" },

  { 0, "floor(0)",           "0" },
  { 0, "floor(0.25)",        "0" },
  { 0, "floor(0.5)",         "0" },
  { 0, "floor(1.5)",         "1" },
  { 0, "floor(-0.5)",        "-1" },
  { 0, "floor(-1.5)",        "-2" },

  { 0, "integer_p(1)",   "1" },
  { 0, "integer_p(0.5)", "0" },

  { 0, "trunc(0)",           "0" },
  { 0, "trunc(0.25)",        "0" },
  { 0, "trunc(0.5)",         "0" },
  { 0, "trunc(1.5)",         "1" },
  { 0, "trunc(-0.5)",        "0" },
  { 0, "trunc(-1.5)",        "-1" },
};

struct datalist_t {
  const struct data_t  *data;
  int                  num;
};

#define DATALIST(data)  { data, numberof (data) }

struct datalist_t  list_z[] = {
  DATALIST (data_z),
  DATALIST (data_zq),
  DATALIST (data_zf),
  DATALIST (data_zqf),
};

struct datalist_t  list_q[] = {
  DATALIST (data_q),
  DATALIST (data_zq),
  DATALIST (data_zqf),
};

struct datalist_t  list_f[] = {
  DATALIST (data_zf),
  DATALIST (data_zqf),
  DATALIST (data_f),
};


void
check_z (void)
{
  const struct data_t  *data;
  mpz_t  a, b, got, want;
  int    l, i, ret;

  mpz_init (got);
  mpz_init (want);
  mpz_init_set_ui (a, 55);
  mpz_init_set_ui (b, 99);

  for (l = 0; l < numberof (list_z); l++)
    {
      data = list_z[l].data;

      for (i = 0; i < list_z[l].num; i++)
        {
          if (option_trace)
            printf ("mpz_expr \"%s\"\n", data[i].expr);

          ret = mpz_expr (got, data[i].base, data[i].expr, a, b, NULL);

          if (data[i].want == NULL)
            {
              /* expect to fail */
              if (ret == MPEXPR_RESULT_OK)
                {
                  printf ("mpz_expr wrong return value, got %d, expected failure\n", ret);
                  goto error;
                }
            }
          else
            {
              if (mpz_set_str (want, data[i].want, 0) != 0)
                {
                  printf ("Cannot parse wanted value string\n");
                  goto error;
                }
              if (ret != MPEXPR_RESULT_OK)
                {
                  printf ("mpz_expr failed unexpectedly\n");
                  printf ("   return value %d\n", ret);
                  goto error;
                }
              if (mpz_cmp (got, want) != 0)
                {
                  printf ("mpz_expr wrong result\n");
                  printf ("   got  "); mpz_out_str (stdout, 10, got);
                  printf ("\n");
                  printf ("   want "); mpz_out_str (stdout, 10, want);
                  printf ("\n");
                  goto error;
                }
            }
        }
    }
  mpz_clear (a);
  mpz_clear (b);
  mpz_clear (got);
  mpz_clear (want);
  return;

 error:
  printf ("   base %d\n", data[i].base);
  printf ("   expr \"%s\"\n", data[i].expr);
  if (data[i].want != NULL)
    printf ("   want \"%s\"\n", data[i].want);
  abort ();
}

void
check_q (void)
{
  const struct data_t  *data;
  mpq_t  a, b, got, want;
  int    l, i, ret;

  mpq_init (got);
  mpq_init (want);
  mpq_init (a);
  mpq_init (b);

  mpq_set_ui (a, 55, 1);
  mpq_set_ui (b, 99, 1);

  for (l = 0; l < numberof (list_q); l++)
    {
      data = list_q[l].data;

      for (i = 0; i < list_q[l].num; i++)
        {
          if (option_trace)
            printf ("mpq_expr \"%s\"\n", data[i].expr);

          ret = mpq_expr (got, data[i].base, data[i].expr, a, b, NULL);

          if (data[i].want == NULL)
            {
              /* expect to fail */
              if (ret == MPEXPR_RESULT_OK)
                {
                  printf ("mpq_expr wrong return value, got %d, expected failure\n", ret);
                  goto error;
                }
            }
          else
            {
              if (mpz_set_str (mpq_numref(want), data[i].want, 0) != 0)
                {
                  printf ("Cannot parse wanted value string\n");
                  goto error;
                }
              mpz_set_ui (mpq_denref(want), 1);

              if (ret != MPEXPR_RESULT_OK)
                {
                  printf ("mpq_expr failed unexpectedly\n");
                  printf ("   return value %d\n", ret);
                  goto error;
                }
              if (mpq_cmp (got, want) != 0)
                {
                  printf ("mpq_expr wrong result\n");
                  printf ("   got  "); mpq_out_str (stdout, 10, got);
                  printf ("\n");
                  printf ("   want "); mpq_out_str (stdout, 10, want);
                  printf ("\n");
                  goto error;
                }
            }
        }
    }
  mpq_clear (a);
  mpq_clear (b);
  mpq_clear (got);
  mpq_clear (want);
  return;

 error:
  printf ("   base %d\n", data[i].base);
  printf ("   expr \"%s\"\n", data[i].expr);
  if (data[i].want != NULL)
    printf ("   want \"%s\"\n", data[i].want);
  abort ();
}

void
check_f (void)
{
  const struct data_t  *data;
  mpf_t  a, b, got, want;
  int    l, i, ret;

  mpf_set_default_prec (200L);

  mpf_init (got);
  mpf_init (want);
  mpf_init_set_ui (a, 55);
  mpf_init_set_ui (b, 99);

  for (l = 0; l < numberof (list_f); l++)
    {
      data = list_f[l].data;

      for (i = 0; i < list_f[l].num; i++)
        {
          if (option_trace)
            printf ("mpf_expr \"%s\"\n", data[i].expr);

          ret = mpf_expr (got, data[i].base, data[i].expr, a, b, NULL);

          if (data[i].want == NULL)
            {
              /* expect to fail */
              if (ret == MPEXPR_RESULT_OK)
                {
                  printf ("mpf_expr wrong return value, got %d, expected failure\n", ret);
                  goto error;
                }
            }
          else
            {
              if (mpf_set_str (want, data[i].want, 0) != 0)
                {
                  printf ("Cannot parse wanted value string\n");
                  goto error;
                }

              if (ret != MPEXPR_RESULT_OK)
                {
                  printf ("mpf_expr failed unexpectedly\n");
                  printf ("   return value %d\n", ret);
                  goto error;
                }
              if (mpf_cmp (got, want) != 0)
                {
                  printf ("mpf_expr wrong result\n");
                  printf ("   got  "); mpf_out_str (stdout, 10, 20, got);
                  printf ("\n");
                  printf ("   want "); mpf_out_str (stdout, 10, 20, want);
                  printf ("\n");
                  goto error;
                }
            }
        }
    }
  mpf_clear (a);
  mpf_clear (b);
  mpf_clear (got);
  mpf_clear (want);
  return;

 error:
  printf ("   base %d\n", data[i].base);
  printf ("   expr \"%s\"\n", data[i].expr);
  if (data[i].want != NULL)
    printf ("   want \"%s\"\n", data[i].want);
  abort ();
}


int
main (int argc, char *argv[])
{
  tests_start ();

  if (argc >= 2)
    option_trace = 1;

  check_z ();
  check_q ();
  check_f ();

  tests_end ();
  exit (0);
}
