/* Demo program to run expression evaluation.

Copyright 2000-2002, 2004 Free Software Foundation, Inc.

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


/* Usage: ./run-expr [-z] [-q] [-f] [-p prec] [-b base] expression...

   Evaluate each argument as a simple expression.  By default this is in mpz
   integers, but -q selects mpq or -f selects mpf.  For mpf the float
   precision can be set with -p.  In all cases the input base can be set
   with -b, or the default is "0" meaning decimal with "0x" allowed.

   This is a pretty trivial program, it's just an easy way to experiment
   with the evaluation functions.  */


#include <stdio.h>
#include <stdlib.h>

#include "gmp.h"
#include "expr.h"


void
run_expr (int type, int base, unsigned long prec, char *str)
{
  int  outbase = (base == 0 ? 10 : base);
  int  ret;

  switch (type) {
  case 'z':
  default:
    {
      mpz_t  res, var_a, var_b;

      mpz_init (res);
      mpz_init_set_ui (var_a, 55L);
      mpz_init_set_ui (var_b, 99L);

      ret = mpz_expr (res, base, str, var_a, var_b, NULL);
      printf ("\"%s\" base %d: ", str, base);
      if (ret == MPEXPR_RESULT_OK)
        {
          printf ("result ");
          mpz_out_str (stdout, outbase, res);
          printf ("\n");
        }
      else
        printf ("invalid (return code %d)\n", ret);

      mpz_clear (res);
      mpz_clear (var_a);
      mpz_clear (var_b);
    }
    break;

  case 'q':
    {
      mpq_t  res, var_a, var_b;

      mpq_init (res);
      mpq_init (var_a);
      mpq_init (var_b);

      mpq_set_ui (var_a, 55L, 1);
      mpq_set_ui (var_b, 99L, 1);

      ret = mpq_expr (res, base, str, var_a, var_b, NULL);
      printf ("\"%s\" base %d: ", str, base);
      if (ret == MPEXPR_RESULT_OK)
        {
          printf ("result ");
          mpq_out_str (stdout, outbase, res);
          printf ("\n");
        }
      else
        printf ("invalid (return code %d)\n", ret);

      mpq_clear (res);
      mpq_clear (var_a);
      mpq_clear (var_b);
    }
    break;

  case 'f':
    {
      mpf_t  res, var_a, var_b;

      mpf_init2 (res, prec);
      mpf_init_set_ui (var_a, 55L);
      mpf_init_set_ui (var_b, 99L);

      ret = mpf_expr (res, base, str, var_a, var_b, NULL);
      printf ("\"%s\" base %d: ", str, base);
      if (ret == MPEXPR_RESULT_OK)
        {
          printf ("result ");
          mpf_out_str (stdout, outbase, (size_t) 0, res);
          printf ("\n");
        }
      else
        printf ("invalid (return code %d)\n", ret);

      mpf_clear (res);
      mpf_clear (var_a);
      mpf_clear (var_b);
    }
    break;
  }
}

int
main (int argc, char *argv[])
{
  int            type = 'z';
  int            base = 0;
  unsigned long  prec = 64;
  int            seen_expr = 0;
  int            opt;
  char           *arg;

  for (;;)
    {
      argv++;
      arg = argv[0];
      if (arg == NULL)
        break;

      if (arg[0] == '-')
        {
          for (;;)
            {
              arg++;
              opt = arg[0];

              switch (opt) {
              case '\0':
                goto end_opt;

              case 'f':
              case 'q':
              case 'z':
                type = opt;
                break;

              case 'b':
                arg++;
                if (arg[0] == '\0')
                  {
                    argv++;
                    arg = argv[0];
                    if (arg == NULL)
                      {
                      need_arg:
                        fprintf (stderr, "Need argument for -%c\n", opt);
                        exit (1);
                      }
                  }
                base = atoi (arg);
                goto end_opt;

              case 'p':
                arg++;
                if (arg[0] == '\0')
                  {
                    argv++;
                    arg = argv[0];
                    if (arg == NULL)
                      goto need_arg;
                  }
                prec = atoi (arg);
                goto end_opt;

              case '-':
                arg++;
                if (arg[0] != '\0')
                  {
                    /* no "--foo" options */
                    fprintf (stderr, "Unrecognised option --%s\n", arg);
                    exit (1);
                  }
                /* stop option interpretation at "--" */
                for (;;)
                  {
                    argv++;
                    arg = argv[0];
                    if (arg == NULL)
                      goto done;
                    run_expr (type, base, prec, arg);
                    seen_expr = 1;
                  }

              default:
                fprintf (stderr, "Unrecognised option -%c\n", opt);
                exit (1);
              }
            }
        end_opt:
          ;
        }
      else
        {
          run_expr (type, base, prec, arg);
          seen_expr = 1;
        }
    }

 done:
  if (! seen_expr)
    {
      printf ("Usage: %s [-z] [-q] [-f] [-p prec] [-b base] expression...\n", argv[0]);
      exit (1);
    }

  return 0;
}
