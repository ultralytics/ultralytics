/* mpexpr_evaluate -- shared code for simple expression evaluation

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

#include <ctype.h>
#include <stdio.h>
#include <string.h>

#include "gmp.h"
#include "expr-impl.h"


/* Change this to "#define TRACE(x) x" to get some traces.  The trace
   printfs junk up the code a bit, but it's very hard to tell what's going
   on without them.  Set MPX_TRACE to a suitable output function for the
   mpz/mpq/mpf being run (if you have the wrong trace function it'll
   probably segv).  */

#define TRACE(x)
#define MPX_TRACE  mpz_trace


/* A few helper macros copied from gmp-impl.h */
#define ALLOCATE_FUNC_TYPE(n,type) \
  ((type *) (*allocate_func) ((n) * sizeof (type)))
#define ALLOCATE_FUNC_LIMBS(n)   ALLOCATE_FUNC_TYPE (n, mp_limb_t)
#define REALLOCATE_FUNC_TYPE(p, old_size, new_size, type) \
  ((type *) (*reallocate_func)                            \
   (p, (old_size) * sizeof (type), (new_size) * sizeof (type)))
#define REALLOCATE_FUNC_LIMBS(p, old_size, new_size) \
  REALLOCATE_FUNC_TYPE(p, old_size, new_size, mp_limb_t)
#define FREE_FUNC_TYPE(p,n,type) (*free_func) (p, (n) * sizeof (type))
#define FREE_FUNC_LIMBS(p,n)     FREE_FUNC_TYPE (p, n, mp_limb_t)
#define ASSERT(x)



/* All the error strings are just for diagnostic traces.  Only the error
   code is actually returned.  */
#define ERROR(str,code)                 \
  {                                     \
    TRACE (printf ("%s\n", str));       \
    p->error_code = (code);             \
    goto done;                          \
  }


#define REALLOC(ptr, alloc, incr, type)                         \
  do {                                                          \
    int  new_alloc = (alloc) + (incr);                          \
    ptr = REALLOCATE_FUNC_TYPE (ptr, alloc, new_alloc, type);   \
    (alloc) = new_alloc;                                        \
  } while (0)


/* data stack top element */
#define SP   (p->data_stack + p->data_top)

/* Make sure there's room for another data element above current top.
   reallocate_func is fetched for when this macro is used in lookahead(). */
#define DATA_SPACE()                                                    \
  do {                                                                  \
    if (p->data_top + 1 >= p->data_alloc)                               \
      {                                                                 \
	void *(*reallocate_func) (void *, size_t, size_t);              \
	mp_get_memory_functions (NULL, &reallocate_func, NULL);         \
	TRACE (printf ("grow stack from %d\n", p->data_alloc));         \
	REALLOC (p->data_stack, p->data_alloc, 20, union mpX_t);        \
      }                                                                 \
    ASSERT (p->data_top + 1 <= p->data_inited);                         \
    if (p->data_top + 1 == p->data_inited)                              \
      {                                                                 \
	TRACE (printf ("initialize %d\n", p->data_top + 1));            \
	(*p->mpX_init) (&p->data_stack[p->data_top + 1], p->prec);      \
	p->data_inited++;                                               \
      }                                                                 \
  } while (0)

#define DATA_PUSH()                             \
  do {                                          \
    p->data_top++;                              \
    ASSERT (p->data_top < p->data_alloc);       \
    ASSERT (p->data_top < p->data_inited);      \
  } while (0)

/* the last stack entry is never popped, so top>=0 will be true */
#define DATA_POP(n)             \
  do {                          \
    p->data_top -= (n);         \
    ASSERT (p->data_top >= 0);  \
  } while (0)


/* lookahead() parses the next token.  Return 1 if successful, with some
   extra data.  Return 0 if fail, with reason in p->error_code.

   "prefix" is MPEXPR_TYPE_PREFIX if an operator with that attribute is
   preferred, or 0 if an operator without is preferred. */

#define TOKEN_EOF         -1   /* no extra data */
#define TOKEN_VALUE       -2   /* pushed onto data stack */
#define TOKEN_OPERATOR    -3   /* stored in p->token_op */
#define TOKEN_FUNCTION    -4   /* stored in p->token_op */

#define TOKEN_NAME(n)                           \
  ((n) == TOKEN_EOF ? "TOKEN_EOF"               \
   : (n) == TOKEN_VALUE ? "TOKEN_VALUE"         \
   : (n) == TOKEN_OPERATOR ? "TOKEN_OPERATOR"   \
   : (n) == TOKEN_VALUE ? "TOKEN_FUNCTION"      \
   : "UNKNOWN TOKEN")

/* Functions default to being parsed as whole words, operators to match just
   at the start of the string.  The type flags override this. */
#define WHOLEWORD(op)                           \
  (op->precedence == 0                          \
   ? (! (op->type & MPEXPR_TYPE_OPERATOR))      \
   :   (op->type & MPEXPR_TYPE_WHOLEWORD))

#define isasciispace(c)   (isascii (c) && isspace (c))

static int
lookahead (struct mpexpr_parse_t *p, int prefix)
{
  const struct mpexpr_operator_t  *op, *op_found;
  size_t  oplen, oplen_found, wlen;
  int     i;

  /* skip white space */
  while (p->elen > 0 && isasciispace (*p->e))
    p->e++, p->elen--;

  if (p->elen == 0)
    {
      TRACE (printf ("lookahead EOF\n"));
      p->token = TOKEN_EOF;
      return 1;
    }

  DATA_SPACE ();

  /* Get extent of whole word. */
  for (wlen = 0; wlen < p->elen; wlen++)
    if (! isasciicsym (p->e[wlen]))
      break;

  TRACE (printf ("lookahead at: \"%.*s\" length %u, word %u\n",
		 (int) p->elen, p->e, p->elen, wlen));

  op_found = NULL;
  oplen_found = 0;
  for (op = p->table; op->name != NULL; op++)
    {
      if (op->type == MPEXPR_TYPE_NEW_TABLE)
	{
	  printf ("new\n");
	  op = (struct mpexpr_operator_t *) op->name - 1;
	  continue;
	}

      oplen = strlen (op->name);
      if (! ((WHOLEWORD (op) ? wlen == oplen : p->elen >= oplen)
	     && memcmp (p->e, op->name, oplen) == 0))
	continue;

      /* Shorter matches don't replace longer previous ones. */
      if (op_found && oplen < oplen_found)
	continue;

      /* On a match of equal length to a previous one, the old match isn't
	 replaced if it has the preferred prefix, and if it doesn't then
	 it's not replaced if the new one also doesn't.  */
      if (op_found && oplen == oplen_found
	  && ((op_found->type & MPEXPR_TYPE_PREFIX) == prefix
	      || (op->type & MPEXPR_TYPE_PREFIX) != prefix))
	continue;

      /* This is now either the first match seen, or a longer than previous
	 match, or an equal to previous one but with a preferred prefix. */
      op_found = op;
      oplen_found = oplen;
    }

  if (op_found)
    {
      p->e += oplen_found, p->elen -= oplen_found;

      if (op_found->type == MPEXPR_TYPE_VARIABLE)
	{
	  if (p->elen == 0)
	    ERROR ("end of string expecting a variable",
		   MPEXPR_RESULT_PARSE_ERROR);
	  i = p->e[0] - 'a';
	  if (i < 0 || i >= MPEXPR_VARIABLES)
	    ERROR ("bad variable name", MPEXPR_RESULT_BAD_VARIABLE);
	  goto variable;
	}

      if (op_found->precedence == 0)
	{
	  TRACE (printf ("lookahead function: %s\n", op_found->name));
	  p->token = TOKEN_FUNCTION;
	  p->token_op = op_found;
	  return 1;
	}
      else
	{
	  TRACE (printf ("lookahead operator: %s\n", op_found->name));
	  p->token = TOKEN_OPERATOR;
	  p->token_op = op_found;
	  return 1;
	}
    }

  oplen = (*p->mpX_number) (SP+1, p->e, p->elen, p->base);
  if (oplen != 0)
    {
      p->e += oplen, p->elen -= oplen;
      p->token = TOKEN_VALUE;
      DATA_PUSH ();
      TRACE (MPX_TRACE ("lookahead number", SP));
      return 1;
    }

  /* Maybe an unprefixed one character variable */
  i = p->e[0] - 'a';
  if (wlen == 1 && i >= 0 && i < MPEXPR_VARIABLES)
    {
    variable:
      p->e++, p->elen--;
      if (p->var[i] == NULL)
	ERROR ("NULL variable", MPEXPR_RESULT_BAD_VARIABLE);
      TRACE (printf ("lookahead variable: var[%d] = ", i);
	     MPX_TRACE ("", p->var[i]));
      p->token = TOKEN_VALUE;
      DATA_PUSH ();
      (*p->mpX_set) (SP, p->var[i]);
      return 1;
    }

  ERROR ("no token matched", MPEXPR_RESULT_PARSE_ERROR);

 done:
  return 0;
}


/* control stack current top element */
#define CP   (p->control_stack + p->control_top)

/* make sure there's room for another control element above current top */
#define CONTROL_SPACE()                                                    \
  do {                                                                     \
    if (p->control_top + 1 >= p->control_alloc)                            \
      {                                                                    \
	TRACE (printf ("grow control stack from %d\n", p->control_alloc)); \
	REALLOC (p->control_stack, p->control_alloc, 20,                   \
		 struct mpexpr_control_t);                                 \
      }                                                                    \
  } while (0)

/* Push an operator on the control stack, claiming currently to have the
   given number of args ready.  Local variable "op" is used in case opptr is
   a reference through CP.  */
#define CONTROL_PUSH(opptr,args)                        \
  do {                                                  \
    const struct mpexpr_operator_t *op = opptr;		\
    struct mpexpr_control_t *cp;                        \
    CONTROL_SPACE ();                                   \
    p->control_top++;                                   \
    ASSERT (p->control_top < p->control_alloc);         \
    cp = CP;                                            \
    cp->op = op;                                        \
    cp->argcount = (args);                              \
    TRACE_CONTROL("control stack push:");               \
  } while (0)

/* The special operator_done is never popped, so top>=0 will hold. */
#define CONTROL_POP()                           \
  do {                                          \
    p->control_top--;                           \
    ASSERT (p->control_top >= 0);               \
    TRACE_CONTROL ("control stack pop:");       \
  } while (0)

#define TRACE_CONTROL(str)                              \
  TRACE ({                                              \
    int  i;                                             \
    printf ("%s depth %d:", str, p->control_top);       \
    for (i = 0; i <= p->control_top; i++)               \
      printf (" \"%s\"(%d)",                            \
	      p->control_stack[i].op->name,             \
	      p->control_stack[i].argcount);            \
    printf ("\n");                                      \
  });


#define LOOKAHEAD(prefix)               \
  do {                                  \
    if (! lookahead (p, prefix))        \
      goto done;                        \
  } while (0)

#define CHECK_UI(n)                                                     \
  do {                                                                  \
    if (! (*p->mpX_ulong_p) (n))                                        \
      ERROR ("operand doesn't fit ulong", MPEXPR_RESULT_NOT_UI);        \
  } while (0)

#define CHECK_ARGCOUNT(str,n)                                              \
  do {                                                                     \
    if (CP->argcount != (n))                                               \
      {                                                                    \
	TRACE (printf ("wrong number of arguments for %s, got %d want %d", \
		       str, CP->argcount, n));                             \
	ERROR ("", MPEXPR_RESULT_PARSE_ERROR);                             \
      }                                                                    \
  } while (0)


/* There's two basic states here.  In both p->token is the next token.

   "another_expr" is when a whole expression should be parsed.  This means a
   literal or variable value possibly followed by an operator, or a function
   or prefix operator followed by a further whole expression.

   "another_operator" is when an expression has been parsed and its value is
   on the top of the data stack (SP) and an optional further postfix or
   infix operator should be parsed.

   In "another_operator" precedences determine whether to push the operator
   onto the control stack, or instead go to "apply_control" to reduce the
   operator currently on top of the control stack.

   When an operator has both a prefix and postfix/infix form, a LOOKAHEAD()
   for "another_expr" will seek the prefix form, a LOOKAHEAD() for
   "another_operator" will seek the postfix/infix form.  The grammar is
   simple enough that the next state is known before reading the next token.

   Argument count checking guards against functions consuming the wrong
   number of operands from the data stack.  The same checks are applied to
   operators, but will always pass since a UNARY or BINARY will only ever
   parse with the correct operands.  */

int
mpexpr_evaluate (struct mpexpr_parse_t *p)
{
  void *(*allocate_func) (size_t);
  void *(*reallocate_func) (void *, size_t, size_t);
  void (*free_func) (void *, size_t);

  mp_get_memory_functions (&allocate_func, &reallocate_func, &free_func);

  TRACE (printf ("mpexpr_evaluate() base %d \"%.*s\"\n",
		 p->base, (int) p->elen, p->e));

  /* "done" is a special sentinel at the bottom of the control stack,
     precedence -1 is lower than any normal operator.  */
  {
    static const struct mpexpr_operator_t  operator_done
      = { "DONE", NULL, MPEXPR_TYPE_DONE, -1 };

    p->control_alloc = 20;
    p->control_stack = ALLOCATE_FUNC_TYPE (p->control_alloc,
					   struct mpexpr_control_t);
    p->control_top = 0;
    CP->op = &operator_done;
    CP->argcount = 1;
  }

  p->data_inited = 0;
  p->data_alloc = 20;
  p->data_stack = ALLOCATE_FUNC_TYPE (p->data_alloc, union mpX_t);
  p->data_top = -1;

  p->error_code = MPEXPR_RESULT_OK;


 another_expr_lookahead:
  LOOKAHEAD (MPEXPR_TYPE_PREFIX);
  TRACE (printf ("another expr\n"));

  /*another_expr:*/
  switch (p->token) {
  case TOKEN_VALUE:
    goto another_operator_lookahead;

  case TOKEN_OPERATOR:
    TRACE (printf ("operator %s\n", p->token_op->name));
    if (! (p->token_op->type & MPEXPR_TYPE_PREFIX))
      ERROR ("expected a prefix operator", MPEXPR_RESULT_PARSE_ERROR);

    CONTROL_PUSH (p->token_op, 1);
    goto another_expr_lookahead;

  case TOKEN_FUNCTION:
    CONTROL_PUSH (p->token_op, 1);

    if (p->token_op->type & MPEXPR_TYPE_CONSTANT)
      goto apply_control_lookahead;

    LOOKAHEAD (MPEXPR_TYPE_PREFIX);
    if (! (p->token == TOKEN_OPERATOR
	   && p->token_op->type == MPEXPR_TYPE_OPENPAREN))
      ERROR ("expected open paren for function", MPEXPR_RESULT_PARSE_ERROR);

    TRACE (printf ("open paren for function \"%s\"\n", CP->op->name));

    if ((CP->op->type & MPEXPR_TYPE_MASK_ARGCOUNT) == MPEXPR_TYPE_NARY(0))
      {
	LOOKAHEAD (0);
	if (! (p->token == TOKEN_OPERATOR
	       && p->token_op->type == MPEXPR_TYPE_CLOSEPAREN))
	  ERROR ("expected close paren for 0ary function",
		 MPEXPR_RESULT_PARSE_ERROR);
	goto apply_control_lookahead;
      }

    goto another_expr_lookahead;
  }
  ERROR ("unrecognised start of expression", MPEXPR_RESULT_PARSE_ERROR);


 another_operator_lookahead:
  LOOKAHEAD (0);
 another_operator:
  TRACE (printf ("another operator maybe: %s\n", TOKEN_NAME(p->token)));

  switch (p->token) {
  case TOKEN_EOF:
    goto apply_control;

  case TOKEN_OPERATOR:
    /* The next operator is compared to the one on top of the control stack.
       If the next is lower precedence, or the same precedence and not
       right-associative, then reduce using the control stack and look at
       the next operator again later.  */

#define PRECEDENCE_TEST_REDUCE(tprec,cprec,ttype,ctype)                 \
    ((tprec) < (cprec)                                                  \
     || ((tprec) == (cprec) && ! ((ttype) & MPEXPR_TYPE_RIGHTASSOC)))

    if (PRECEDENCE_TEST_REDUCE (p->token_op->precedence, CP->op->precedence,
				p->token_op->type,       CP->op->type))
      {
	TRACE (printf ("defer operator: %s (prec %d vs %d, type 0x%X)\n",
		       p->token_op->name,
		       p->token_op->precedence, CP->op->precedence,
		       p->token_op->type));
	goto apply_control;
      }

    /* An argsep is a binary operator, but is never pushed on the control
       stack, it just accumulates an extra argument for a function. */
    if (p->token_op->type == MPEXPR_TYPE_ARGSEP)
      {
	if (CP->op->precedence != 0)
	  ERROR ("ARGSEP not in a function call", MPEXPR_RESULT_PARSE_ERROR);

	TRACE (printf ("argsep for function \"%s\"(%d)\n",
		       CP->op->name, CP->argcount));

#define IS_PAIRWISE(type)                                               \
	(((type) & (MPEXPR_TYPE_MASK_ARGCOUNT | MPEXPR_TYPE_PAIRWISE))  \
	 == (MPEXPR_TYPE_BINARY | MPEXPR_TYPE_PAIRWISE))

	if (IS_PAIRWISE (CP->op->type) && CP->argcount >= 2)
	  {
	    TRACE (printf ("    will reduce pairwise now\n"));
	    CP->argcount--;
	    CONTROL_PUSH (CP->op, 2);
	    goto apply_control;
	  }

	CP->argcount++;
	goto another_expr_lookahead;
      }

    switch (p->token_op->type & MPEXPR_TYPE_MASK_ARGCOUNT) {
    case MPEXPR_TYPE_NARY(1):
      /* Postfix unary operators can always be applied immediately.  The
	 easiest way to do this is just push it on the control stack and go
	 to the normal control stack reduction code. */

      TRACE (printf ("postfix unary operator: %s\n", p->token_op->name));
      if (p->token_op->type & MPEXPR_TYPE_PREFIX)
	ERROR ("prefix unary operator used postfix",
	       MPEXPR_RESULT_PARSE_ERROR);
      CONTROL_PUSH (p->token_op, 1);
      goto apply_control_lookahead;

    case MPEXPR_TYPE_NARY(2):
      CONTROL_PUSH (p->token_op, 2);
      goto another_expr_lookahead;

    case MPEXPR_TYPE_NARY(3):
      CONTROL_PUSH (p->token_op, 1);
      goto another_expr_lookahead;
    }

    TRACE (printf ("unrecognised operator \"%s\" type: 0x%X",
		   CP->op->name, CP->op->type));
    ERROR ("", MPEXPR_RESULT_PARSE_ERROR);
    break;

  default:
    TRACE (printf ("expecting an operator, got token %d", p->token));
    ERROR ("", MPEXPR_RESULT_PARSE_ERROR);
  }


 apply_control_lookahead:
  LOOKAHEAD (0);
 apply_control:
  /* Apply the top element CP of the control stack.  Data values are SP,
     SP-1, etc.  Result is left as stack top SP after popping consumed
     values.

     The use of sp as a duplicate of SP will help compilers that can't
     otherwise recognise the various uses of SP as common subexpressions.  */

  TRACE (printf ("apply control: nested %d, \"%s\" 0x%X, %d args\n",
		 p->control_top, CP->op->name, CP->op->type, CP->argcount));

  TRACE (printf ("apply 0x%X-ary\n",
		 CP->op->type & MPEXPR_TYPE_MASK_ARGCOUNT));
  switch (CP->op->type & MPEXPR_TYPE_MASK_ARGCOUNT) {
  case MPEXPR_TYPE_NARY(0):
    {
      mpX_ptr  sp;
      DATA_SPACE ();
      DATA_PUSH ();
      sp = SP;
      switch (CP->op->type & MPEXPR_TYPE_MASK_ARGSTYLE) {
      case 0:
	(* (mpexpr_fun_0ary_t) CP->op->fun) (sp);
	break;
      case MPEXPR_TYPE_RESULT_INT:
	(*p->mpX_set_si) (sp, (long) (* (mpexpr_fun_i_0ary_t) CP->op->fun) ());
	break;
      default:
	ERROR ("unrecognised 0ary argument calling style",
	       MPEXPR_RESULT_BAD_TABLE);
      }
    }
    break;

  case MPEXPR_TYPE_NARY(1):
    {
      mpX_ptr  sp = SP;
      CHECK_ARGCOUNT ("unary", 1);
      TRACE (MPX_TRACE ("before", sp));

      switch (CP->op->type & MPEXPR_TYPE_MASK_SPECIAL) {
      case 0:
	/* not a special */
	break;

      case MPEXPR_TYPE_DONE & MPEXPR_TYPE_MASK_SPECIAL:
	TRACE (printf ("special done\n"));
	goto done;

      case MPEXPR_TYPE_LOGICAL_NOT & MPEXPR_TYPE_MASK_SPECIAL:
	TRACE (printf ("special logical not\n"));
	(*p->mpX_set_si)
	  (sp, (long) ((* (mpexpr_fun_i_unary_t) CP->op->fun) (sp) == 0));
	goto apply_control_done;

      case MPEXPR_TYPE_CLOSEPAREN & MPEXPR_TYPE_MASK_SPECIAL:
	CONTROL_POP ();
	if (CP->op->type == MPEXPR_TYPE_OPENPAREN)
	  {
	    TRACE (printf ("close paren matching open paren\n"));
	    CONTROL_POP ();
	    goto another_operator;
	  }
	if (CP->op->precedence == 0)
	  {
	    TRACE (printf ("close paren for function\n"));
	    goto apply_control;
	  }
	ERROR ("unexpected close paren", MPEXPR_RESULT_PARSE_ERROR);

      default:
	TRACE (printf ("unrecognised special unary operator 0x%X",
		       CP->op->type & MPEXPR_TYPE_MASK_SPECIAL));
	ERROR ("", MPEXPR_RESULT_BAD_TABLE);
      }

      switch (CP->op->type & MPEXPR_TYPE_MASK_ARGSTYLE) {
      case 0:
	(* (mpexpr_fun_unary_t) CP->op->fun) (sp, sp);
	break;
      case MPEXPR_TYPE_LAST_UI:
	CHECK_UI (sp);
	(* (mpexpr_fun_unary_ui_t) CP->op->fun)
	  (sp, (*p->mpX_get_ui) (sp));
	break;
      case MPEXPR_TYPE_RESULT_INT:
	(*p->mpX_set_si)
	  (sp, (long) (* (mpexpr_fun_i_unary_t) CP->op->fun) (sp));
	break;
      case MPEXPR_TYPE_RESULT_INT | MPEXPR_TYPE_LAST_UI:
	CHECK_UI (sp);
	(*p->mpX_set_si)
	  (sp,
	   (long) (* (mpexpr_fun_i_unary_ui_t) CP->op->fun)
	   ((*p->mpX_get_ui) (sp)));
	break;
      default:
	ERROR ("unrecognised unary argument calling style",
	       MPEXPR_RESULT_BAD_TABLE);
      }
    }
    break;

  case MPEXPR_TYPE_NARY(2):
    {
      mpX_ptr  sp;

      /* pairwise functions are allowed to have just one argument */
      if ((CP->op->type & MPEXPR_TYPE_PAIRWISE)
	  && CP->op->precedence == 0
	  && CP->argcount == 1)
	goto apply_control_done;

      CHECK_ARGCOUNT ("binary", 2);
      DATA_POP (1);
      sp = SP;
      TRACE (MPX_TRACE ("lhs", sp);
	     MPX_TRACE ("rhs", sp+1));

      if (CP->op->type & MPEXPR_TYPE_MASK_CMP)
	{
	  int  type = CP->op->type;
	  int  cmp = (* (mpexpr_fun_i_binary_t) CP->op->fun)
	    (sp, sp+1);
	  (*p->mpX_set_si)
	    (sp,
	     (long)
	     ((  (cmp  < 0) & ((type & MPEXPR_TYPE_MASK_CMP_LT) != 0))
	      | ((cmp == 0) & ((type & MPEXPR_TYPE_MASK_CMP_EQ) != 0))
	      | ((cmp  > 0) & ((type & MPEXPR_TYPE_MASK_CMP_GT) != 0))));
	  goto apply_control_done;
	}

      switch (CP->op->type & MPEXPR_TYPE_MASK_SPECIAL) {
      case 0:
	/* not a special */
	break;

      case MPEXPR_TYPE_QUESTION & MPEXPR_TYPE_MASK_SPECIAL:
	ERROR ("'?' without ':'", MPEXPR_RESULT_PARSE_ERROR);

      case MPEXPR_TYPE_COLON & MPEXPR_TYPE_MASK_SPECIAL:
	TRACE (printf ("special colon\n"));
	CONTROL_POP ();
	if (CP->op->type != MPEXPR_TYPE_QUESTION)
	  ERROR ("':' without '?'", MPEXPR_RESULT_PARSE_ERROR);

	CP->argcount--;
	DATA_POP (1);
	sp--;
	TRACE (MPX_TRACE ("query", sp);
	       MPX_TRACE ("true",  sp+1);
	       MPX_TRACE ("false", sp+2));
	(*p->mpX_set)
	  (sp, (* (mpexpr_fun_i_unary_t) CP->op->fun) (sp)
	   ? sp+1 : sp+2);
	goto apply_control_done;

      case MPEXPR_TYPE_LOGICAL_AND & MPEXPR_TYPE_MASK_SPECIAL:
	TRACE (printf ("special logical and\n"));
	(*p->mpX_set_si)
	  (sp,
	   (long)
	   ((* (mpexpr_fun_i_unary_t) CP->op->fun) (sp)
	    && (* (mpexpr_fun_i_unary_t) CP->op->fun) (sp+1)));
	goto apply_control_done;

      case MPEXPR_TYPE_LOGICAL_OR & MPEXPR_TYPE_MASK_SPECIAL:
	TRACE (printf ("special logical and\n"));
	(*p->mpX_set_si)
	  (sp,
	   (long)
	   ((* (mpexpr_fun_i_unary_t) CP->op->fun) (sp)
	    || (* (mpexpr_fun_i_unary_t) CP->op->fun) (sp+1)));
	goto apply_control_done;

      case MPEXPR_TYPE_MAX & MPEXPR_TYPE_MASK_SPECIAL:
	TRACE (printf ("special max\n"));
	if ((* (mpexpr_fun_i_binary_t) CP->op->fun) (sp, sp+1) < 0)
	  (*p->mpX_swap) (sp, sp+1);
	goto apply_control_done;
      case MPEXPR_TYPE_MIN & MPEXPR_TYPE_MASK_SPECIAL:
	TRACE (printf ("special min\n"));
	if ((* (mpexpr_fun_i_binary_t) CP->op->fun) (sp, sp+1) > 0)
	  (*p->mpX_swap) (sp, sp+1);
	goto apply_control_done;

      default:
	ERROR ("unrecognised special binary operator",
	       MPEXPR_RESULT_BAD_TABLE);
      }

      switch (CP->op->type & MPEXPR_TYPE_MASK_ARGSTYLE) {
      case 0:
	(* (mpexpr_fun_binary_t) CP->op->fun) (sp, sp, sp+1);
	break;
      case MPEXPR_TYPE_LAST_UI:
	CHECK_UI (sp+1);
	(* (mpexpr_fun_binary_ui_t) CP->op->fun)
	  (sp, sp, (*p->mpX_get_ui) (sp+1));
	break;
      case MPEXPR_TYPE_RESULT_INT:
	(*p->mpX_set_si)
	  (sp,
	   (long) (* (mpexpr_fun_i_binary_t) CP->op->fun) (sp, sp+1));
	break;
      case MPEXPR_TYPE_LAST_UI | MPEXPR_TYPE_RESULT_INT:
	CHECK_UI (sp+1);
	(*p->mpX_set_si)
	  (sp,
	   (long) (* (mpexpr_fun_i_binary_ui_t) CP->op->fun)
	   (sp, (*p->mpX_get_ui) (sp+1)));
	break;
      default:
	ERROR ("unrecognised binary argument calling style",
	       MPEXPR_RESULT_BAD_TABLE);
      }
    }
    break;

  case MPEXPR_TYPE_NARY(3):
    {
      mpX_ptr  sp;

      CHECK_ARGCOUNT ("ternary", 3);
      DATA_POP (2);
      sp = SP;
      TRACE (MPX_TRACE ("arg1", sp);
	     MPX_TRACE ("arg2", sp+1);
	     MPX_TRACE ("arg3", sp+1));

      switch (CP->op->type & MPEXPR_TYPE_MASK_ARGSTYLE) {
      case 0:
	(* (mpexpr_fun_ternary_t) CP->op->fun) (sp, sp, sp+1, sp+2);
	break;
      case MPEXPR_TYPE_LAST_UI:
	CHECK_UI (sp+2);
	(* (mpexpr_fun_ternary_ui_t) CP->op->fun)
	  (sp, sp, sp+1, (*p->mpX_get_ui) (sp+2));
	break;
      case MPEXPR_TYPE_RESULT_INT:
	(*p->mpX_set_si)
	  (sp,
	   (long) (* (mpexpr_fun_i_ternary_t) CP->op->fun)
	   (sp, sp+1, sp+2));
	break;
      case MPEXPR_TYPE_LAST_UI | MPEXPR_TYPE_RESULT_INT:
	CHECK_UI (sp+2);
	(*p->mpX_set_si)
	  (sp,
	   (long) (* (mpexpr_fun_i_ternary_ui_t) CP->op->fun)
	   (sp, sp+1, (*p->mpX_get_ui) (sp+2)));
	break;
      default:
	ERROR ("unrecognised binary argument calling style",
	       MPEXPR_RESULT_BAD_TABLE);
      }
    }
    break;

  default:
    TRACE (printf ("unrecognised operator type: 0x%X\n", CP->op->type));
    ERROR ("", MPEXPR_RESULT_PARSE_ERROR);
  }

 apply_control_done:
  TRACE (MPX_TRACE ("result", SP));
  CONTROL_POP ();
  goto another_operator;

 done:
  if (p->error_code == MPEXPR_RESULT_OK)
    {
      if (p->data_top != 0)
	{
	  TRACE (printf ("data stack want top at 0, got %d\n", p->data_top));
	  p->error_code = MPEXPR_RESULT_PARSE_ERROR;
	}
      else
	(*p->mpX_set_or_swap) (p->res, SP);
    }

  {
    int  i;
    for (i = 0; i < p->data_inited; i++)
      {
	TRACE (printf ("clear %d\n", i));
	(*p->mpX_clear) (p->data_stack+i);
      }
  }

  FREE_FUNC_TYPE (p->data_stack, p->data_alloc, union mpX_t);
  FREE_FUNC_TYPE (p->control_stack, p->control_alloc, struct mpexpr_control_t);

  return p->error_code;
}
