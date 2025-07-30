/* Program for computing integer expressions using the GNU Multiple Precision
   Arithmetic Library.

Copyright 1997, 1999-2002, 2005, 2008, 2012, 2015 Free Software Foundation, Inc.

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see https://www.gnu.org/licenses/.  */


/* This expressions evaluator works by building an expression tree (using a
   recursive descent parser) which is then evaluated.  The expression tree is
   useful since we want to optimize certain expressions (like a^b % c).

   Usage: pexpr [options] expr ...
   (Assuming you called the executable `pexpr' of course.)

   Command line options:

   -b        print output in binary
   -o        print output in octal
   -d        print output in decimal (the default)
   -x        print output in hexadecimal
   -b<NUM>   print output in base NUM
   -t        print timing information
   -html     output html
   -wml      output wml
   -split    split long lines each 80th digit
*/

/* Define LIMIT_RESOURCE_USAGE if you want to make sure the program doesn't
   use up extensive resources (cpu, memory).  Useful for the GMP demo on the
   GMP web site, since we cannot load the server too much.  */

#include "pexpr-config.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <setjmp.h>
#include <signal.h>
#include <ctype.h>

#include <time.h>
#include <sys/types.h>
#include <sys/time.h>
#if HAVE_SYS_RESOURCE_H
#include <sys/resource.h>
#endif

#include "gmp.h"

/* SunOS 4 and HPUX 9 don't define a canonical SIGSTKSZ, use a default. */
#ifndef SIGSTKSZ
#define SIGSTKSZ  4096
#endif


#define TIME(t,func)							\
  do { int __t0, __tmp;							\
    __t0 = cputime ();							\
    {func;}								\
    __tmp = cputime () - __t0;						\
    (t) = __tmp;							\
  } while (0)

/* GMP version 1.x compatibility.  */
#if ! (__GNU_MP_VERSION >= 2)
typedef MP_INT __mpz_struct;
typedef __mpz_struct mpz_t[1];
typedef __mpz_struct *mpz_ptr;
#define mpz_fdiv_q	mpz_div
#define mpz_fdiv_r	mpz_mod
#define mpz_tdiv_q_2exp	mpz_div_2exp
#define mpz_sgn(Z) ((Z)->size < 0 ? -1 : (Z)->size > 0)
#endif

/* GMP version 2.0 compatibility.  */
#if ! (__GNU_MP_VERSION > 2 || __GNU_MP_VERSION_MINOR >= 1)
#define mpz_swap(a,b) \
  do { __mpz_struct __t; __t = *a; *a = *b; *b = __t;} while (0)
#endif

jmp_buf errjmpbuf;

enum op_t {NOP, LIT, NEG, NOT, PLUS, MINUS, MULT, DIV, MOD, REM, INVMOD, POW,
	   AND, IOR, XOR, SLL, SRA, POPCNT, HAMDIST, GCD, LCM, SQRT, ROOT, FAC,
	   LOG, LOG2, FERMAT, MERSENNE, FIBONACCI, RANDOM, NEXTPRIME, BINOM,
	   TIMING};

/* Type for the expression tree.  */
struct expr
{
  enum op_t op;
  union
  {
    struct {struct expr *lhs, *rhs;} ops;
    mpz_t val;
  } operands;
};

typedef struct expr *expr_t;

void cleanup_and_exit (int);

char *skipspace (char *);
void makeexp (expr_t *, enum op_t, expr_t, expr_t);
void free_expr (expr_t);
char *expr (char *, expr_t *);
char *term (char *, expr_t *);
char *power (char *, expr_t *);
char *factor (char *, expr_t *);
int match (char *, char *);
int matchp (char *, char *);
int cputime (void);

void mpz_eval_expr (mpz_ptr, expr_t);
void mpz_eval_mod_expr (mpz_ptr, expr_t, mpz_ptr);

char *error;
int flag_print = 1;
int print_timing = 0;
int flag_html = 0;
int flag_wml = 0;
int flag_splitup_output = 0;
char *newline = "";
gmp_randstate_t rstate;



/* cputime() returns user CPU time measured in milliseconds.  */
#if ! HAVE_CPUTIME
#if HAVE_GETRUSAGE
int
cputime (void)
{
  struct rusage rus;

  getrusage (0, &rus);
  return rus.ru_utime.tv_sec * 1000 + rus.ru_utime.tv_usec / 1000;
}
#else
#if HAVE_CLOCK
int
cputime (void)
{
  if (CLOCKS_PER_SEC < 100000)
    return clock () * 1000 / CLOCKS_PER_SEC;
  return clock () / (CLOCKS_PER_SEC / 1000);
}
#else
int
cputime (void)
{
  return 0;
}
#endif
#endif
#endif


int
stack_downwards_helper (char *xp)
{
  char  y;
  return &y < xp;
}
int
stack_downwards_p (void)
{
  char  x;
  return stack_downwards_helper (&x);
}


void
setup_error_handler (void)
{
#if HAVE_SIGACTION
  struct sigaction act;
  act.sa_handler = cleanup_and_exit;
  sigemptyset (&(act.sa_mask));
#define SIGNAL(sig)  sigaction (sig, &act, NULL)
#else
  struct { int sa_flags; } act;
#define SIGNAL(sig)  signal (sig, cleanup_and_exit)
#endif
  act.sa_flags = 0;

  /* Set up a stack for signal handling.  A typical cause of error is stack
     overflow, and in such situation a signal can not be delivered on the
     overflown stack.  */
#if HAVE_SIGALTSTACK
  {
    /* AIX uses stack_t, MacOS uses struct sigaltstack, various other
       systems have both. */
#if HAVE_STACK_T
    stack_t s;
#else
    struct sigaltstack s;
#endif
    s.ss_sp = malloc (SIGSTKSZ);
    s.ss_size = SIGSTKSZ;
    s.ss_flags = 0;
    if (sigaltstack (&s, NULL) != 0)
      perror("sigaltstack");
    act.sa_flags = SA_ONSTACK;
  }
#else
#if HAVE_SIGSTACK
  {
    struct sigstack s;
    s.ss_sp = malloc (SIGSTKSZ);
    if (stack_downwards_p ())
      s.ss_sp += SIGSTKSZ;
    s.ss_onstack = 0;
    if (sigstack (&s, NULL) != 0)
      perror("sigstack");
    act.sa_flags = SA_ONSTACK;
  }
#else
#endif
#endif

#ifdef LIMIT_RESOURCE_USAGE
  {
    struct rlimit limit;

    limit.rlim_cur = limit.rlim_max = 0;
    setrlimit (RLIMIT_CORE, &limit);

    limit.rlim_cur = 3;
    limit.rlim_max = 4;
    setrlimit (RLIMIT_CPU, &limit);

    limit.rlim_cur = limit.rlim_max = 16 * 1024 * 1024;
    setrlimit (RLIMIT_DATA, &limit);

    getrlimit (RLIMIT_STACK, &limit);
    limit.rlim_cur = 4 * 1024 * 1024;
    setrlimit (RLIMIT_STACK, &limit);

    SIGNAL (SIGXCPU);
  }
#endif /* LIMIT_RESOURCE_USAGE */

  SIGNAL (SIGILL);
  SIGNAL (SIGSEGV);
#ifdef SIGBUS /* not in mingw */
  SIGNAL (SIGBUS);
#endif
  SIGNAL (SIGFPE);
  SIGNAL (SIGABRT);
}

int
main (int argc, char **argv)
{
  struct expr *e;
  int i;
  mpz_t r;
  int errcode = 0;
  char *str;
  int base = 10;

  setup_error_handler ();

  gmp_randinit (rstate, GMP_RAND_ALG_LC, 128);

  {
#if HAVE_GETTIMEOFDAY
    struct timeval tv;
    gettimeofday (&tv, NULL);
    gmp_randseed_ui (rstate, tv.tv_sec + tv.tv_usec);
#else
    time_t t;
    time (&t);
    gmp_randseed_ui (rstate, t);
#endif
  }

  mpz_init (r);

  while (argc > 1 && argv[1][0] == '-')
    {
      char *arg = argv[1];

      if (arg[1] >= '0' && arg[1] <= '9')
	break;

      if (arg[1] == 't')
	print_timing = 1;
      else if (arg[1] == 'b' && arg[2] >= '0' && arg[2] <= '9')
	{
	  base = atoi (arg + 2);
	  if (base < 2 || base > 62)
	    {
	      fprintf (stderr, "error: invalid output base\n");
	      exit (-1);
	    }
	}
      else if (arg[1] == 'b' && arg[2] == 0)
	base = 2;
      else if (arg[1] == 'x' && arg[2] == 0)
	base = 16;
      else if (arg[1] == 'X' && arg[2] == 0)
	base = -16;
      else if (arg[1] == 'o' && arg[2] == 0)
	base = 8;
      else if (arg[1] == 'd' && arg[2] == 0)
	base = 10;
      else if (arg[1] == 'v' && arg[2] == 0)
	{
	  printf ("pexpr linked to gmp %s\n", __gmp_version);
	}
      else if (strcmp (arg, "-html") == 0)
	{
	  flag_html = 1;
	  newline = "<br>";
	}
      else if (strcmp (arg, "-wml") == 0)
	{
	  flag_wml = 1;
	  newline = "<br/>";
	}
      else if (strcmp (arg, "-split") == 0)
	{
	  flag_splitup_output = 1;
	}
      else if (strcmp (arg, "-noprint") == 0)
	{
	  flag_print = 0;
	}
      else
	{
	  fprintf (stderr, "error: unknown option `%s'\n", arg);
	  exit (-1);
	}
      argv++;
      argc--;
    }

  for (i = 1; i < argc; i++)
    {
      int s;
      int jmpval;

      /* Set up error handler for parsing expression.  */
      jmpval = setjmp (errjmpbuf);
      if (jmpval != 0)
	{
	  fprintf (stderr, "error: %s%s\n", error, newline);
	  fprintf (stderr, "       %s%s\n", argv[i], newline);
	  if (! flag_html)
	    {
	      /* ??? Dunno how to align expression position with arrow in
		 HTML ??? */
	      fprintf (stderr, "       ");
	      for (s = jmpval - (long) argv[i]; --s >= 0; )
		putc (' ', stderr);
	      fprintf (stderr, "^\n");
	    }

	  errcode |= 1;
	  continue;
	}

      str = expr (argv[i], &e);

      if (str[0] != 0)
	{
	  fprintf (stderr,
		   "error: garbage where end of expression expected%s\n",
		   newline);
	  fprintf (stderr, "       %s%s\n", argv[i], newline);
	  if (! flag_html)
	    {
	      /* ??? Dunno how to align expression position with arrow in
		 HTML ??? */
	      fprintf (stderr, "        ");
	      for (s = str - argv[i]; --s; )
		putc (' ', stderr);
	      fprintf (stderr, "^\n");
	    }

	  errcode |= 1;
	  free_expr (e);
	  continue;
	}

      /* Set up error handler for evaluating expression.  */
      if (setjmp (errjmpbuf))
	{
	  fprintf (stderr, "error: %s%s\n", error, newline);
	  fprintf (stderr, "       %s%s\n", argv[i], newline);
	  if (! flag_html)
	    {
	      /* ??? Dunno how to align expression position with arrow in
		 HTML ??? */
	      fprintf (stderr, "       ");
	      for (s = str - argv[i]; --s >= 0; )
		putc (' ', stderr);
	      fprintf (stderr, "^\n");
	    }

	  errcode |= 2;
	  continue;
	}

      if (print_timing)
	{
	  int t;
	  TIME (t, mpz_eval_expr (r, e));
	  printf ("computation took %d ms%s\n", t, newline);
	}
      else
	mpz_eval_expr (r, e);

      if (flag_print)
	{
	  size_t out_len;
	  char *tmp, *s;

	  out_len = mpz_sizeinbase (r, base >= 0 ? base : -base) + 2;
#ifdef LIMIT_RESOURCE_USAGE
	  if (out_len > 100000)
	    {
	      printf ("result is about %ld digits, not printing it%s\n",
		      (long) out_len - 3, newline);
	      exit (-2);
	    }
#endif
	  tmp = malloc (out_len);

	  if (print_timing)
	    {
	      int t;
	      printf ("output conversion ");
	      TIME (t, mpz_get_str (tmp, base, r));
	      printf ("took %d ms%s\n", t, newline);
	    }
	  else
	    mpz_get_str (tmp, base, r);

	  out_len = strlen (tmp);
	  if (flag_splitup_output)
	    {
	      for (s = tmp; out_len > 80; s += 80)
		{
		  fwrite (s, 1, 80, stdout);
		  printf ("%s\n", newline);
		  out_len -= 80;
		}

	      fwrite (s, 1, out_len, stdout);
	    }
	  else
	    {
	      fwrite (tmp, 1, out_len, stdout);
	    }

	  free (tmp);
	  printf ("%s\n", newline);
	}
      else
	{
	  printf ("result is approximately %ld digits%s\n",
		  (long) mpz_sizeinbase (r, base >= 0 ? base : -base),
		  newline);
	}

      free_expr (e);
    }

  mpz_clear (r);

  exit (errcode);
}

char *
expr (char *str, expr_t *e)
{
  expr_t e2;

  str = skipspace (str);
  if (str[0] == '+')
    {
      str = term (str + 1, e);
    }
  else if (str[0] == '-')
    {
      str = term (str + 1, e);
      makeexp (e, NEG, *e, NULL);
    }
  else if (str[0] == '~')
    {
      str = term (str + 1, e);
      makeexp (e, NOT, *e, NULL);
    }
  else
    {
      str = term (str, e);
    }

  for (;;)
    {
      str = skipspace (str);
      switch (str[0])
	{
	case 'p':
	  if (match ("plus", str))
	    {
	      str = term (str + 4, &e2);
	      makeexp (e, PLUS, *e, e2);
	    }
	  else
	    return str;
	  break;
	case 'm':
	  if (match ("minus", str))
	    {
	      str = term (str + 5, &e2);
	      makeexp (e, MINUS, *e, e2);
	    }
	  else
	    return str;
	  break;
	case '+':
	  str = term (str + 1, &e2);
	  makeexp (e, PLUS, *e, e2);
	  break;
	case '-':
	  str = term (str + 1, &e2);
	  makeexp (e, MINUS, *e, e2);
	  break;
	default:
	  return str;
	}
    }
}

char *
term (char *str, expr_t *e)
{
  expr_t e2;

  str = power (str, e);
  for (;;)
    {
      str = skipspace (str);
      switch (str[0])
	{
	case 'm':
	  if (match ("mul", str))
	    {
	      str = power (str + 3, &e2);
	      makeexp (e, MULT, *e, e2);
	      break;
	    }
	  if (match ("mod", str))
	    {
	      str = power (str + 3, &e2);
	      makeexp (e, MOD, *e, e2);
	      break;
	    }
	  return str;
	case 'd':
	  if (match ("div", str))
	    {
	      str = power (str + 3, &e2);
	      makeexp (e, DIV, *e, e2);
	      break;
	    }
	  return str;
	case 'r':
	  if (match ("rem", str))
	    {
	      str = power (str + 3, &e2);
	      makeexp (e, REM, *e, e2);
	      break;
	    }
	  return str;
	case 'i':
	  if (match ("invmod", str))
	    {
	      str = power (str + 6, &e2);
	      makeexp (e, REM, *e, e2);
	      break;
	    }
	  return str;
	case 't':
	  if (match ("times", str))
	    {
	      str = power (str + 5, &e2);
	      makeexp (e, MULT, *e, e2);
	      break;
	    }
	  if (match ("thru", str))
	    {
	      str = power (str + 4, &e2);
	      makeexp (e, DIV, *e, e2);
	      break;
	    }
	  if (match ("through", str))
	    {
	      str = power (str + 7, &e2);
	      makeexp (e, DIV, *e, e2);
	      break;
	    }
	  return str;
	case '*':
	  str = power (str + 1, &e2);
	  makeexp (e, MULT, *e, e2);
	  break;
	case '/':
	  str = power (str + 1, &e2);
	  makeexp (e, DIV, *e, e2);
	  break;
	case '%':
	  str = power (str + 1, &e2);
	  makeexp (e, MOD, *e, e2);
	  break;
	default:
	  return str;
	}
    }
}

char *
power (char *str, expr_t *e)
{
  expr_t e2;

  str = factor (str, e);
  while (str[0] == '!')
    {
      str++;
      makeexp (e, FAC, *e, NULL);
    }
  str = skipspace (str);
  if (str[0] == '^')
    {
      str = power (str + 1, &e2);
      makeexp (e, POW, *e, e2);
    }
  return str;
}

int
match (char *s, char *str)
{
  char *ostr = str;
  int i;

  for (i = 0; s[i] != 0; i++)
    {
      if (str[i] != s[i])
	return 0;
    }
  str = skipspace (str + i);
  return str - ostr;
}

int
matchp (char *s, char *str)
{
  char *ostr = str;
  int i;

  for (i = 0; s[i] != 0; i++)
    {
      if (str[i] != s[i])
	return 0;
    }
  str = skipspace (str + i);
  if (str[0] == '(')
    return str - ostr + 1;
  return 0;
}

struct functions
{
  char *spelling;
  enum op_t op;
  int arity; /* 1 or 2 means real arity; 0 means arbitrary.  */
};

struct functions fns[] =
{
  {"sqrt", SQRT, 1},
#if __GNU_MP_VERSION >= 2
  {"root", ROOT, 2},
  {"popc", POPCNT, 1},
  {"hamdist", HAMDIST, 2},
#endif
  {"gcd", GCD, 0},
#if __GNU_MP_VERSION > 2 || __GNU_MP_VERSION_MINOR >= 1
  {"lcm", LCM, 0},
#endif
  {"and", AND, 0},
  {"ior", IOR, 0},
#if __GNU_MP_VERSION > 2 || __GNU_MP_VERSION_MINOR >= 1
  {"xor", XOR, 0},
#endif
  {"plus", PLUS, 0},
  {"pow", POW, 2},
  {"minus", MINUS, 2},
  {"mul", MULT, 0},
  {"div", DIV, 2},
  {"mod", MOD, 2},
  {"rem", REM, 2},
#if __GNU_MP_VERSION >= 2
  {"invmod", INVMOD, 2},
#endif
  {"log", LOG, 2},
  {"log2", LOG2, 1},
  {"F", FERMAT, 1},
  {"M", MERSENNE, 1},
  {"fib", FIBONACCI, 1},
  {"Fib", FIBONACCI, 1},
  {"random", RANDOM, 1},
  {"nextprime", NEXTPRIME, 1},
  {"binom", BINOM, 2},
  {"binomial", BINOM, 2},
  {"fac", FAC, 1},
  {"fact", FAC, 1},
  {"factorial", FAC, 1},
  {"time", TIMING, 1},
  {"", NOP, 0}
};

char *
factor (char *str, expr_t *e)
{
  expr_t e1, e2;

  str = skipspace (str);

  if (isalpha (str[0]))
    {
      int i;
      int cnt;

      for (i = 0; fns[i].op != NOP; i++)
	{
	  if (fns[i].arity == 1)
	    {
	      cnt = matchp (fns[i].spelling, str);
	      if (cnt != 0)
		{
		  str = expr (str + cnt, &e1);
		  str = skipspace (str);
		  if (str[0] != ')')
		    {
		      error = "expected `)'";
		      longjmp (errjmpbuf, (int) (long) str);
		    }
		  makeexp (e, fns[i].op, e1, NULL);
		  return str + 1;
		}
	    }
	}

      for (i = 0; fns[i].op != NOP; i++)
	{
	  if (fns[i].arity != 1)
	    {
	      cnt = matchp (fns[i].spelling, str);
	      if (cnt != 0)
		{
		  str = expr (str + cnt, &e1);
		  str = skipspace (str);

		  if (str[0] != ',')
		    {
		      error = "expected `,' and another operand";
		      longjmp (errjmpbuf, (int) (long) str);
		    }

		  str = skipspace (str + 1);
		  str = expr (str, &e2);
		  str = skipspace (str);

		  if (fns[i].arity == 0)
		    {
		      while (str[0] == ',')
			{
			  makeexp (&e1, fns[i].op, e1, e2);
			  str = skipspace (str + 1);
			  str = expr (str, &e2);
			  str = skipspace (str);
			}
		    }

		  if (str[0] != ')')
		    {
		      error = "expected `)'";
		      longjmp (errjmpbuf, (int) (long) str);
		    }

		  makeexp (e, fns[i].op, e1, e2);
		  return str + 1;
		}
	    }
	}
    }

  if (str[0] == '(')
    {
      str = expr (str + 1, e);
      str = skipspace (str);
      if (str[0] != ')')
	{
	  error = "expected `)'";
	  longjmp (errjmpbuf, (int) (long) str);
	}
      str++;
    }
  else if (str[0] >= '0' && str[0] <= '9')
    {
      expr_t res;
      char *s, *sc;

      res = malloc (sizeof (struct expr));
      res -> op = LIT;
      mpz_init (res->operands.val);

      s = str;
      while (isalnum (str[0]))
	str++;
      sc = malloc (str - s + 1);
      memcpy (sc, s, str - s);
      sc[str - s] = 0;

      mpz_set_str (res->operands.val, sc, 0);
      *e = res;
      free (sc);
    }
  else
    {
      error = "operand expected";
      longjmp (errjmpbuf, (int) (long) str);
    }
  return str;
}

char *
skipspace (char *str)
{
  while (str[0] == ' ')
    str++;
  return str;
}

/* Make a new expression with operation OP and right hand side
   RHS and left hand side lhs.  Put the result in R.  */
void
makeexp (expr_t *r, enum op_t op, expr_t lhs, expr_t rhs)
{
  expr_t res;
  res = malloc (sizeof (struct expr));
  res -> op = op;
  res -> operands.ops.lhs = lhs;
  res -> operands.ops.rhs = rhs;
  *r = res;
  return;
}

/* Free the memory used by expression E.  */
void
free_expr (expr_t e)
{
  if (e->op != LIT)
    {
      free_expr (e->operands.ops.lhs);
      if (e->operands.ops.rhs != NULL)
	free_expr (e->operands.ops.rhs);
    }
  else
    {
      mpz_clear (e->operands.val);
    }
}

/* Evaluate the expression E and put the result in R.  */
void
mpz_eval_expr (mpz_ptr r, expr_t e)
{
  mpz_t lhs, rhs;

  switch (e->op)
    {
    case LIT:
      mpz_set (r, e->operands.val);
      return;
    case PLUS:
      mpz_init (lhs); mpz_init (rhs);
      mpz_eval_expr (lhs, e->operands.ops.lhs);
      mpz_eval_expr (rhs, e->operands.ops.rhs);
      mpz_add (r, lhs, rhs);
      mpz_clear (lhs); mpz_clear (rhs);
      return;
    case MINUS:
      mpz_init (lhs); mpz_init (rhs);
      mpz_eval_expr (lhs, e->operands.ops.lhs);
      mpz_eval_expr (rhs, e->operands.ops.rhs);
      mpz_sub (r, lhs, rhs);
      mpz_clear (lhs); mpz_clear (rhs);
      return;
    case MULT:
      mpz_init (lhs); mpz_init (rhs);
      mpz_eval_expr (lhs, e->operands.ops.lhs);
      mpz_eval_expr (rhs, e->operands.ops.rhs);
      mpz_mul (r, lhs, rhs);
      mpz_clear (lhs); mpz_clear (rhs);
      return;
    case DIV:
      mpz_init (lhs); mpz_init (rhs);
      mpz_eval_expr (lhs, e->operands.ops.lhs);
      mpz_eval_expr (rhs, e->operands.ops.rhs);
      mpz_fdiv_q (r, lhs, rhs);
      mpz_clear (lhs); mpz_clear (rhs);
      return;
    case MOD:
      mpz_init (rhs);
      mpz_eval_expr (rhs, e->operands.ops.rhs);
      mpz_abs (rhs, rhs);
      mpz_eval_mod_expr (r, e->operands.ops.lhs, rhs);
      mpz_clear (rhs);
      return;
    case REM:
      /* Check if lhs operand is POW expression and optimize for that case.  */
      if (e->operands.ops.lhs->op == POW)
	{
	  mpz_t powlhs, powrhs;
	  mpz_init (powlhs);
	  mpz_init (powrhs);
	  mpz_init (rhs);
	  mpz_eval_expr (powlhs, e->operands.ops.lhs->operands.ops.lhs);
	  mpz_eval_expr (powrhs, e->operands.ops.lhs->operands.ops.rhs);
	  mpz_eval_expr (rhs, e->operands.ops.rhs);
	  mpz_powm (r, powlhs, powrhs, rhs);
	  if (mpz_cmp_si (rhs, 0L) < 0)
	    mpz_neg (r, r);
	  mpz_clear (powlhs);
	  mpz_clear (powrhs);
	  mpz_clear (rhs);
	  return;
	}

      mpz_init (lhs); mpz_init (rhs);
      mpz_eval_expr (lhs, e->operands.ops.lhs);
      mpz_eval_expr (rhs, e->operands.ops.rhs);
      mpz_fdiv_r (r, lhs, rhs);
      mpz_clear (lhs); mpz_clear (rhs);
      return;
#if __GNU_MP_VERSION >= 2
    case INVMOD:
      mpz_init (lhs); mpz_init (rhs);
      mpz_eval_expr (lhs, e->operands.ops.lhs);
      mpz_eval_expr (rhs, e->operands.ops.rhs);
      mpz_invert (r, lhs, rhs);
      mpz_clear (lhs); mpz_clear (rhs);
      return;
#endif
    case POW:
      mpz_init (lhs); mpz_init (rhs);
      mpz_eval_expr (lhs, e->operands.ops.lhs);
      if (mpz_cmpabs_ui (lhs, 1) <= 0)
	{
	  /* For 0^rhs and 1^rhs, we just need to verify that
	     rhs is well-defined.  For (-1)^rhs we need to
	     determine (rhs mod 2).  For simplicity, compute
	     (rhs mod 2) for all three cases.  */
	  expr_t two, et;
	  two = malloc (sizeof (struct expr));
	  two -> op = LIT;
	  mpz_init_set_ui (two->operands.val, 2L);
	  makeexp (&et, MOD, e->operands.ops.rhs, two);
	  e->operands.ops.rhs = et;
	}

      mpz_eval_expr (rhs, e->operands.ops.rhs);
      if (mpz_cmp_si (rhs, 0L) == 0)
	/* x^0 is 1 */
	mpz_set_ui (r, 1L);
      else if (mpz_cmp_si (lhs, 0L) == 0)
	/* 0^y (where y != 0) is 0 */
	mpz_set_ui (r, 0L);
      else if (mpz_cmp_ui (lhs, 1L) == 0)
	/* 1^y is 1 */
	mpz_set_ui (r, 1L);
      else if (mpz_cmp_si (lhs, -1L) == 0)
	/* (-1)^y just depends on whether y is even or odd */
	mpz_set_si (r, (mpz_get_ui (rhs) & 1) ? -1L : 1L);
      else if (mpz_cmp_si (rhs, 0L) < 0)
	/* x^(-n) is 0 */
	mpz_set_ui (r, 0L);
      else
	{
	  unsigned long int cnt;
	  unsigned long int y;
	  /* error if exponent does not fit into an unsigned long int.  */
	  if (mpz_cmp_ui (rhs, ~(unsigned long int) 0) > 0)
	    goto pow_err;

	  y = mpz_get_ui (rhs);
	  /* x^y == (x/(2^c))^y * 2^(c*y) */
#if __GNU_MP_VERSION >= 2
	  cnt = mpz_scan1 (lhs, 0);
#else
	  cnt = 0;
#endif
	  if (cnt != 0)
	    {
	      if (y * cnt / cnt != y)
		goto pow_err;
	      mpz_tdiv_q_2exp (lhs, lhs, cnt);
	      mpz_pow_ui (r, lhs, y);
	      mpz_mul_2exp (r, r, y * cnt);
	    }
	  else
	    mpz_pow_ui (r, lhs, y);
	}
      mpz_clear (lhs); mpz_clear (rhs);
      return;
    pow_err:
      error = "result of `pow' operator too large";
      mpz_clear (lhs); mpz_clear (rhs);
      longjmp (errjmpbuf, 1);
    case GCD:
      mpz_init (lhs); mpz_init (rhs);
      mpz_eval_expr (lhs, e->operands.ops.lhs);
      mpz_eval_expr (rhs, e->operands.ops.rhs);
      mpz_gcd (r, lhs, rhs);
      mpz_clear (lhs); mpz_clear (rhs);
      return;
#if __GNU_MP_VERSION > 2 || __GNU_MP_VERSION_MINOR >= 1
    case LCM:
      mpz_init (lhs); mpz_init (rhs);
      mpz_eval_expr (lhs, e->operands.ops.lhs);
      mpz_eval_expr (rhs, e->operands.ops.rhs);
      mpz_lcm (r, lhs, rhs);
      mpz_clear (lhs); mpz_clear (rhs);
      return;
#endif
    case AND:
      mpz_init (lhs); mpz_init (rhs);
      mpz_eval_expr (lhs, e->operands.ops.lhs);
      mpz_eval_expr (rhs, e->operands.ops.rhs);
      mpz_and (r, lhs, rhs);
      mpz_clear (lhs); mpz_clear (rhs);
      return;
    case IOR:
      mpz_init (lhs); mpz_init (rhs);
      mpz_eval_expr (lhs, e->operands.ops.lhs);
      mpz_eval_expr (rhs, e->operands.ops.rhs);
      mpz_ior (r, lhs, rhs);
      mpz_clear (lhs); mpz_clear (rhs);
      return;
#if __GNU_MP_VERSION > 2 || __GNU_MP_VERSION_MINOR >= 1
    case XOR:
      mpz_init (lhs); mpz_init (rhs);
      mpz_eval_expr (lhs, e->operands.ops.lhs);
      mpz_eval_expr (rhs, e->operands.ops.rhs);
      mpz_xor (r, lhs, rhs);
      mpz_clear (lhs); mpz_clear (rhs);
      return;
#endif
    case NEG:
      mpz_eval_expr (r, e->operands.ops.lhs);
      mpz_neg (r, r);
      return;
    case NOT:
      mpz_eval_expr (r, e->operands.ops.lhs);
      mpz_com (r, r);
      return;
    case SQRT:
      mpz_init (lhs);
      mpz_eval_expr (lhs, e->operands.ops.lhs);
      if (mpz_sgn (lhs) < 0)
	{
	  error = "cannot take square root of negative numbers";
	  mpz_clear (lhs);
	  longjmp (errjmpbuf, 1);
	}
      mpz_sqrt (r, lhs);
      return;
#if __GNU_MP_VERSION > 2 || __GNU_MP_VERSION_MINOR >= 1
    case ROOT:
      mpz_init (lhs); mpz_init (rhs);
      mpz_eval_expr (lhs, e->operands.ops.lhs);
      mpz_eval_expr (rhs, e->operands.ops.rhs);
      if (mpz_sgn (rhs) <= 0)
	{
	  error = "cannot take non-positive root orders";
	  mpz_clear (lhs); mpz_clear (rhs);
	  longjmp (errjmpbuf, 1);
	}
      if (mpz_sgn (lhs) < 0 && (mpz_get_ui (rhs) & 1) == 0)
	{
	  error = "cannot take even root orders of negative numbers";
	  mpz_clear (lhs); mpz_clear (rhs);
	  longjmp (errjmpbuf, 1);
	}

      {
	unsigned long int nth = mpz_get_ui (rhs);
	if (mpz_cmp_ui (rhs, ~(unsigned long int) 0) > 0)
	  {
	    /* If we are asked to take an awfully large root order, cheat and
	       ask for the largest order we can pass to mpz_root.  This saves
	       some error prone special cases.  */
	    nth = ~(unsigned long int) 0;
	  }
	mpz_root (r, lhs, nth);
      }
      mpz_clear (lhs); mpz_clear (rhs);
      return;
#endif
    case FAC:
      mpz_eval_expr (r, e->operands.ops.lhs);
      if (mpz_size (r) > 1)
	{
	  error = "result of `!' operator too large";
	  longjmp (errjmpbuf, 1);
	}
      mpz_fac_ui (r, mpz_get_ui (r));
      return;
#if __GNU_MP_VERSION >= 2
    case POPCNT:
      mpz_eval_expr (r, e->operands.ops.lhs);
      { long int cnt;
	cnt = mpz_popcount (r);
	mpz_set_si (r, cnt);
      }
      return;
    case HAMDIST:
      { long int cnt;
	mpz_init (lhs); mpz_init (rhs);
	mpz_eval_expr (lhs, e->operands.ops.lhs);
	mpz_eval_expr (rhs, e->operands.ops.rhs);
	cnt = mpz_hamdist (lhs, rhs);
	mpz_clear (lhs); mpz_clear (rhs);
	mpz_set_si (r, cnt);
      }
      return;
#endif
    case LOG2:
      mpz_eval_expr (r, e->operands.ops.lhs);
      { unsigned long int cnt;
	if (mpz_sgn (r) <= 0)
	  {
	    error = "logarithm of non-positive number";
	    longjmp (errjmpbuf, 1);
	  }
	cnt = mpz_sizeinbase (r, 2);
	mpz_set_ui (r, cnt - 1);
      }
      return;
    case LOG:
      { unsigned long int cnt;
	mpz_init (lhs); mpz_init (rhs);
	mpz_eval_expr (lhs, e->operands.ops.lhs);
	mpz_eval_expr (rhs, e->operands.ops.rhs);
	if (mpz_sgn (lhs) <= 0)
	  {
	    error = "logarithm of non-positive number";
	    mpz_clear (lhs); mpz_clear (rhs);
	    longjmp (errjmpbuf, 1);
	  }
	if (mpz_cmp_ui (rhs, 256) >= 0)
	  {
	    error = "logarithm base too large";
	    mpz_clear (lhs); mpz_clear (rhs);
	    longjmp (errjmpbuf, 1);
	  }
	cnt = mpz_sizeinbase (lhs, mpz_get_ui (rhs));
	mpz_set_ui (r, cnt - 1);
	mpz_clear (lhs); mpz_clear (rhs);
      }
      return;
    case FERMAT:
      {
	unsigned long int t;
	mpz_init (lhs);
	mpz_eval_expr (lhs, e->operands.ops.lhs);
	t = (unsigned long int) 1 << mpz_get_ui (lhs);
	if (mpz_cmp_ui (lhs, ~(unsigned long int) 0) > 0 || t == 0)
	  {
	    error = "too large Mersenne number index";
	    mpz_clear (lhs);
	    longjmp (errjmpbuf, 1);
	  }
	mpz_set_ui (r, 1);
	mpz_mul_2exp (r, r, t);
	mpz_add_ui (r, r, 1);
	mpz_clear (lhs);
      }
      return;
    case MERSENNE:
      mpz_init (lhs);
      mpz_eval_expr (lhs, e->operands.ops.lhs);
      if (mpz_cmp_ui (lhs, ~(unsigned long int) 0) > 0)
	{
	  error = "too large Mersenne number index";
	  mpz_clear (lhs);
	  longjmp (errjmpbuf, 1);
	}
      mpz_set_ui (r, 1);
      mpz_mul_2exp (r, r, mpz_get_ui (lhs));
      mpz_sub_ui (r, r, 1);
      mpz_clear (lhs);
      return;
    case FIBONACCI:
      { mpz_t t;
	unsigned long int n, i;
	mpz_init (lhs);
	mpz_eval_expr (lhs, e->operands.ops.lhs);
	if (mpz_sgn (lhs) <= 0 || mpz_cmp_si (lhs, 1000000000) > 0)
	  {
	    error = "Fibonacci index out of range";
	    mpz_clear (lhs);
	    longjmp (errjmpbuf, 1);
	  }
	n = mpz_get_ui (lhs);
	mpz_clear (lhs);

#if __GNU_MP_VERSION > 2 || __GNU_MP_VERSION_MINOR >= 1
	mpz_fib_ui (r, n);
#else
	mpz_init_set_ui (t, 1);
	mpz_set_ui (r, 1);

	if (n <= 2)
	  mpz_set_ui (r, 1);
	else
	  {
	    for (i = 3; i <= n; i++)
	      {
		mpz_add (t, t, r);
		mpz_swap (t, r);
	      }
	  }
	mpz_clear (t);
#endif
      }
      return;
    case RANDOM:
      {
	unsigned long int n;
	mpz_init (lhs);
	mpz_eval_expr (lhs, e->operands.ops.lhs);
	if (mpz_sgn (lhs) <= 0 || mpz_cmp_si (lhs, 1000000000) > 0)
	  {
	    error = "random number size out of range";
	    mpz_clear (lhs);
	    longjmp (errjmpbuf, 1);
	  }
	n = mpz_get_ui (lhs);
	mpz_clear (lhs);
	mpz_urandomb (r, rstate, n);
      }
      return;
    case NEXTPRIME:
      {
	mpz_eval_expr (r, e->operands.ops.lhs);
	mpz_nextprime (r, r);
      }
      return;
    case BINOM:
      mpz_init (lhs); mpz_init (rhs);
      mpz_eval_expr (lhs, e->operands.ops.lhs);
      mpz_eval_expr (rhs, e->operands.ops.rhs);
      {
	unsigned long int k;
	if (mpz_cmp_ui (rhs, ~(unsigned long int) 0) > 0)
	  {
	    error = "k too large in (n over k) expression";
	    mpz_clear (lhs); mpz_clear (rhs);
	    longjmp (errjmpbuf, 1);
	  }
	k = mpz_get_ui (rhs);
	mpz_bin_ui (r, lhs, k);
      }
      mpz_clear (lhs); mpz_clear (rhs);
      return;
    case TIMING:
      {
	int t0;
	t0 = cputime ();
	mpz_eval_expr (r, e->operands.ops.lhs);
	printf ("time: %d\n", cputime () - t0);
      }
      return;
    default:
      abort ();
    }
}

/* Evaluate the expression E modulo MOD and put the result in R.  */
void
mpz_eval_mod_expr (mpz_ptr r, expr_t e, mpz_ptr mod)
{
  mpz_t lhs, rhs;

  switch (e->op)
    {
      case POW:
	mpz_init (lhs); mpz_init (rhs);
	mpz_eval_mod_expr (lhs, e->operands.ops.lhs, mod);
	mpz_eval_expr (rhs, e->operands.ops.rhs);
	mpz_powm (r, lhs, rhs, mod);
	mpz_clear (lhs); mpz_clear (rhs);
	return;
      case PLUS:
	mpz_init (lhs); mpz_init (rhs);
	mpz_eval_mod_expr (lhs, e->operands.ops.lhs, mod);
	mpz_eval_mod_expr (rhs, e->operands.ops.rhs, mod);
	mpz_add (r, lhs, rhs);
	if (mpz_cmp_si (r, 0L) < 0)
	  mpz_add (r, r, mod);
	else if (mpz_cmp (r, mod) >= 0)
	  mpz_sub (r, r, mod);
	mpz_clear (lhs); mpz_clear (rhs);
	return;
      case MINUS:
	mpz_init (lhs); mpz_init (rhs);
	mpz_eval_mod_expr (lhs, e->operands.ops.lhs, mod);
	mpz_eval_mod_expr (rhs, e->operands.ops.rhs, mod);
	mpz_sub (r, lhs, rhs);
	if (mpz_cmp_si (r, 0L) < 0)
	  mpz_add (r, r, mod);
	else if (mpz_cmp (r, mod) >= 0)
	  mpz_sub (r, r, mod);
	mpz_clear (lhs); mpz_clear (rhs);
	return;
      case MULT:
	mpz_init (lhs); mpz_init (rhs);
	mpz_eval_mod_expr (lhs, e->operands.ops.lhs, mod);
	mpz_eval_mod_expr (rhs, e->operands.ops.rhs, mod);
	mpz_mul (r, lhs, rhs);
	mpz_mod (r, r, mod);
	mpz_clear (lhs); mpz_clear (rhs);
	return;
      default:
	mpz_init (lhs);
	mpz_eval_expr (lhs, e);
	mpz_mod (r, lhs, mod);
	mpz_clear (lhs);
	return;
    }
}

void
cleanup_and_exit (int sig)
{
  switch (sig) {
#ifdef LIMIT_RESOURCE_USAGE
  case SIGXCPU:
    printf ("expression took too long to evaluate%s\n", newline);
    break;
#endif
  case SIGFPE:
    printf ("divide by zero%s\n", newline);
    break;
  default:
    printf ("expression required too much memory to evaluate%s\n", newline);
    break;
  }
  exit (-2);
}
