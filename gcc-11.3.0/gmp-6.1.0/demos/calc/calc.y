%{
/* A simple integer desk calculator using yacc and gmp.

Copyright 2000-2002 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see https://www.gnu.org/licenses/.  */


/* This is a simple program, meant only to show one way to use GMP for this
   sort of thing.  There's few features, and error checking is minimal.
   Standard input is read, calc_help() below shows the inputs accepted.

   Expressions are evaluated as they're read.  If user defined functions
   were wanted it'd be necessary to build a parse tree like pexpr.c does, or
   a list of operations for a stack based evaluator.  That would also make
   it possible to detect and optimize evaluations "mod m" like pexpr.c does.

   A stack is used for intermediate values in the expression evaluation,
   separate from the yacc parser stack.  This is simple, makes error
   recovery easy, minimizes the junk around mpz calls in the rules, and
   saves initializing or clearing "mpz_t"s during a calculation.  A
   disadvantage though is that variables must be copied to the stack to be
   worked on.  A more sophisticated calculator or language system might be
   able to avoid that when executing a compiled or semi-compiled form.

   Avoiding repeated initializing and clearing of "mpz_t"s is important.  In
   this program the time spent parsing is obviously much greater than any
   possible saving from this, but a proper calculator or language should
   take some trouble over it.  Don't be surprised if an init/clear takes 3
   or more times as long as a 10 limb addition, depending on the system (see
   the mpz_init_realloc_clear example in tune/README).  */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gmp.h"
#define NO_CALC_H /* because it conflicts with normal calc.c stuff */
#include "calc-common.h"


#define numberof(x)  (sizeof (x) / sizeof ((x)[0]))


void
calc_help (void)
{
  printf ("Examples:\n");
  printf ("    2+3*4        expressions are evaluated\n");
  printf ("    x=5^6        variables a to z can be set and used\n");
  printf ("Operators:\n");
  printf ("    + - *        arithmetic\n");
  printf ("    / %%          division and remainder (rounding towards negative infinity)\n");
  printf ("    ^            exponentiation\n");
  printf ("    !            factorial\n");
  printf ("    << >>        left and right shifts\n");
  printf ("    <= >= >      \\ comparisons, giving 1 if true, 0 if false\n");
  printf ("    == != <      /\n");
  printf ("    && ||        logical and/or, giving 1 if true, 0 if false\n");
  printf ("Functions:\n");
  printf ("    abs(n)       absolute value\n");
  printf ("    bin(n,m)     binomial coefficient\n");
  printf ("    fib(n)       fibonacci number\n");
  printf ("    gcd(a,b,..)  greatest common divisor\n");
  printf ("    kron(a,b)    kronecker symbol\n");
  printf ("    lcm(a,b,..)  least common multiple\n");
  printf ("    lucnum(n)    lucas number\n");
  printf ("    nextprime(n) next prime after n\n");
  printf ("    powm(b,e,m)  modulo powering, b^e%%m\n");
  printf ("    root(n,r)    r-th root\n");
  printf ("    sqrt(n)      square root\n");
  printf ("Other:\n");
  printf ("    hex          \\ set hex or decimal for input and output\n");
  printf ("    decimal      /   (\"0x\" can be used for hex too)\n");
  printf ("    quit         exit program (EOF works too)\n");
  printf ("    ;            statements are separated with a ; or newline\n");
  printf ("    \\            continue expressions with \\ before newline\n");
  printf ("    # xxx        comments are # though to newline\n");
  printf ("Hex numbers must be entered in upper case, to distinguish them from the\n");
  printf ("variables a to f (like in bc).\n");
}


int  ibase = 0;
int  obase = 10;


/* The stack is a fixed size, which means there's a limit on the nesting
   allowed in expressions.  A more sophisticated program could let it grow
   dynamically.  */

mpz_t    stack[100];
mpz_ptr  sp = stack[0];

#define CHECK_OVERFLOW()                                                  \
  if (sp >= stack[numberof(stack)])	/* FIXME */			\
    {                                                                     \
      fprintf (stderr,                                                    \
               "Value stack overflow, too much nesting in expression\n"); \
      YYERROR;                                                            \
    }

#define CHECK_EMPTY()                                                   \
  if (sp != stack[0])                                                   \
    {                                                                   \
      fprintf (stderr, "Oops, expected the value stack to be empty\n"); \
      sp = stack[0];                                                    \
    }


mpz_t  variable[26];

#define CHECK_VARIABLE(var)                                             \
  if ((var) < 0 || (var) >= numberof (variable))                        \
    {                                                                   \
      fprintf (stderr, "Oops, bad variable somehow: %d\n", var);        \
      YYERROR;                                                          \
    }


#define CHECK_UI(name,z)                        \
  if (! mpz_fits_ulong_p (z))                   \
    {                                           \
      fprintf (stderr, "%s too big\n", name);   \
      YYERROR;                                  \
    }

%}

%union {
  char  *str;
  int   var;
}

%token EOS BAD
%token HELP HEX DECIMAL QUIT
%token ABS BIN FIB GCD KRON LCM LUCNUM NEXTPRIME POWM ROOT SQRT
%token <str> NUMBER
%token <var> VARIABLE

/* operators, increasing precedence */
%left     LOR
%left     LAND
%nonassoc '<' '>' EQ NE LE GE
%left     LSHIFT RSHIFT
%left     '+' '-'
%left     '*' '/' '%'
%nonassoc UMINUS
%right    '^'
%nonassoc '!'

%%

top:
  statement
  | statements statement;

statements:
  statement EOS
  | statements statement EOS
  | error EOS { sp = stack[0]; yyerrok; };

statement:
  /* empty */
  | e {
      mpz_out_str (stdout, obase, sp); putchar ('\n');
      sp--;
      CHECK_EMPTY ();
    }
  | VARIABLE '=' e {
      CHECK_VARIABLE ($1);
      mpz_swap (variable[$1], sp);
      sp--;
      CHECK_EMPTY ();
    }
  | HELP    { calc_help (); }
  | HEX     { ibase = 16; obase = -16; }
  | DECIMAL { ibase = 0;  obase = 10; }
  | QUIT    { exit (0); };

/* "e" leaves it's value on the top of the mpz stack.  A rule like "e '+' e"
   will have done a reduction for the first "e" first and the second "e"
   second, so the code receives the values in that order on the stack.  */
e:
    '(' e ')'     /* value on stack */
    | e '+' e     { sp--; mpz_add    (sp, sp, sp+1); }
    | e '-' e     { sp--; mpz_sub    (sp, sp, sp+1); }
    | e '*' e     { sp--; mpz_mul    (sp, sp, sp+1); }
    | e '/' e     { sp--; mpz_fdiv_q (sp, sp, sp+1); }
    | e '%' e     { sp--; mpz_fdiv_r (sp, sp, sp+1); }
    | e '^' e     { CHECK_UI ("Exponent", sp);
                    sp--; mpz_pow_ui (sp, sp, mpz_get_ui (sp+1)); }
    | e LSHIFT e  { CHECK_UI ("Shift count", sp);
                    sp--; mpz_mul_2exp (sp, sp, mpz_get_ui (sp+1)); }
    | e RSHIFT e  { CHECK_UI ("Shift count", sp);
                    sp--; mpz_fdiv_q_2exp (sp, sp, mpz_get_ui (sp+1)); }
    | e '!'       { CHECK_UI ("Factorial", sp);
                    mpz_fac_ui (sp, mpz_get_ui (sp)); }
    | '-' e %prec UMINUS   { mpz_neg (sp, sp); }

    | e '<' e     { sp--; mpz_set_ui (sp, mpz_cmp (sp, sp+1) <  0); }
    | e LE  e     { sp--; mpz_set_ui (sp, mpz_cmp (sp, sp+1) <= 0); }
    | e EQ  e     { sp--; mpz_set_ui (sp, mpz_cmp (sp, sp+1) == 0); }
    | e NE  e     { sp--; mpz_set_ui (sp, mpz_cmp (sp, sp+1) != 0); }
    | e GE  e     { sp--; mpz_set_ui (sp, mpz_cmp (sp, sp+1) >= 0); }
    | e '>' e     { sp--; mpz_set_ui (sp, mpz_cmp (sp, sp+1) >  0); }

    | e LAND e    { sp--; mpz_set_ui (sp, mpz_sgn (sp) && mpz_sgn (sp+1)); }
    | e LOR e     { sp--; mpz_set_ui (sp, mpz_sgn (sp) || mpz_sgn (sp+1)); }

    | ABS '(' e ')'              { mpz_abs (sp, sp); }
    | BIN '(' e ',' e ')'        { sp--; CHECK_UI ("Binomial base", sp+1);
                                   mpz_bin_ui (sp, sp, mpz_get_ui (sp+1)); }
    | FIB '(' e ')'              { CHECK_UI ("Fibonacci", sp);
                                   mpz_fib_ui (sp, mpz_get_ui (sp)); }
    | GCD '(' gcdlist ')'        /* value on stack */
    | KRON '(' e ',' e ')'       { sp--; mpz_set_si (sp,
                                         mpz_kronecker (sp, sp+1)); }
    | LCM '(' lcmlist ')'        /* value on stack */
    | LUCNUM '(' e ')'           { CHECK_UI ("Lucas number", sp);
                                   mpz_lucnum_ui (sp, mpz_get_ui (sp)); }
    | NEXTPRIME '(' e ')'        { mpz_nextprime (sp, sp); }
    | POWM '(' e ',' e ',' e ')' { sp -= 2; mpz_powm (sp, sp, sp+1, sp+2); }
    | ROOT '(' e ',' e ')'       { sp--; CHECK_UI ("Nth-root", sp+1);
                                   mpz_root (sp, sp, mpz_get_ui (sp+1)); }
    | SQRT '(' e ')'             { mpz_sqrt (sp, sp); }

    | VARIABLE {
        sp++;
        CHECK_OVERFLOW ();
        CHECK_VARIABLE ($1);
        mpz_set (sp, variable[$1]);
      }
    | NUMBER {
        sp++;
        CHECK_OVERFLOW ();
        if (mpz_set_str (sp, $1, ibase) != 0)
          {
            fprintf (stderr, "Invalid number: %s\n", $1);
            YYERROR;
          }
      };

gcdlist:
    e                /* value on stack */
    | gcdlist ',' e  { sp--; mpz_gcd (sp, sp, sp+1); };

lcmlist:
    e                /* value on stack */
    | lcmlist ',' e  { sp--; mpz_lcm (sp, sp, sp+1); };

%%

yyerror (char *s)
{
  fprintf (stderr, "%s\n", s);
}

int calc_option_readline = -1;

int
main (int argc, char *argv[])
{
  int  i;

  for (i = 1; i < argc; i++)
    {
      if (strcmp (argv[i], "--readline") == 0)
        calc_option_readline = 1;
      else if (strcmp (argv[i], "--noreadline") == 0)
        calc_option_readline = 0;
      else if (strcmp (argv[i], "--help") == 0)
        {
          printf ("Usage: calc [--option]...\n");
          printf ("  --readline    use readline\n");
          printf ("  --noreadline  don't use readline\n");
          printf ("  --help        this message\n");
          printf ("Readline is only available when compiled in,\n");
          printf ("and in that case it's the default on a tty.\n");
          exit (0);
        }
      else
        {
          fprintf (stderr, "Unrecognised option: %s\n", argv[i]);
          exit (1);
        }
    }

#if WITH_READLINE
  calc_init_readline ();
#else
  if (calc_option_readline == 1)
    {
      fprintf (stderr, "Readline support not available\n");
      exit (1);
    }
#endif

  for (i = 0; i < numberof (variable); i++)
    mpz_init (variable[i]);

  for (i = 0; i < numberof (stack); i++)
    mpz_init (stack[i]);

  return yyparse ();
}
