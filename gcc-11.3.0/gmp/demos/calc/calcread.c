/* Readline support for calc program.

Copyright 2000, 2001 Free Software Foundation, Inc.

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

#include "calc-common.h"

#if WITH_READLINE
#include <stdio.h>   /* for FILE for old versions of readline/readline.h */
#include <stdlib.h>  /* for free */
#include <string.h>  /* for strdup */
#include <unistd.h>  /* for isatty */
#include <readline/readline.h>
#include <readline/history.h>

#include "gmp.h"


/* change this to "#define TRACE(x) x" for a few diagnostics */
#define TRACE(x)


#define MIN(x,y) ((x) < (y) ? (x) : (y))

char *
calc_completion_entry (const char *text, int state)
{
  static int  index, len;
  char  *name;

  if (!state)
    {
      index = 0;
      len = strlen (text);
    }
  TRACE (printf ("calc_completion_entry %s %d, index=%d len=%d\n",
		 text, state, index, len));
  while ((name = calc_keywords[index].name) != NULL)
    {
      index++;
      if (memcmp (name, text, len) == 0)
	return (strdup (name));
    }
  return NULL;
}

void
calc_init_readline (void)
{
  /* By default use readline when the input is a tty.  It's a bit contrary
     to the GNU interface conventions to make the behaviour depend on where
     the input is coming from, but this is pretty convenient.  */
  if (calc_option_readline == -1)
    {
      calc_option_readline = isatty (fileno (stdin));
      TRACE (printf ("calc_option_readline %d\n", calc_option_readline));
    }

  if (calc_option_readline)
    {
      printf ("GNU MP demo calculator program, gmp version %s\n", gmp_version);
      printf ("Type \"help\" for help.\n");
      rl_readline_name = "gmp-calc";
      rl_completion_entry_function = calc_completion_entry;
    }
}


/* This function is supposed to return YY_NULL to indicate EOF, but that
   constant is only in calclex.c and we don't want to clutter calclex.l with
   this readline stuff, so instead just hard code 0 for YY_NULL.  That's
   it's defined value on unix anyway.  */

int
calc_input (char *buf, size_t max_size)
{
  if (calc_option_readline)
    {
      static char    *line = NULL;
      static size_t  line_size = 0;
      static size_t  upto = 0;
      size_t         copy_size;

      if (upto >= line_size)
	{
	  if (line != NULL)
	    free (line);

	  line = readline (calc_more_input ? "more> " : "> ");
	  calc_more_input = 1;
	  if (line == NULL)
	    return 0;
	  TRACE (printf ("readline: %s\n", line));

	  if (line[0] != '\0')
	    add_history (line);

	  line_size = strlen (line);
	  line[line_size] = '\n';
	  line_size++;
	  upto = 0;
	}

      copy_size = MIN (line_size-upto, max_size);
      memcpy (buf, line+upto, copy_size);
      upto += copy_size;
      return copy_size;
    }
  else
    {
      /* not readline */
      return fread (buf, 1, max_size, stdin);
    }
}


/* This redefined input() might let a traditional lex use the readline
   support here.  Apparently POSIX doesn't specify whether an override like
   this will work, so maybe it'll work or maybe it won't.  This function is
   also not particularly efficient, but don't worry about that, since flex
   is the preferred parser.  */

int
input (void)
{
  char  c;
  if (calc_input (&c, 1) != 1)
    return EOF;
  else
    return (int) c;
}

#endif /* WITH_READLINE */
