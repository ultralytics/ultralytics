/* mpc_inp_str -- Input a complex number from a given stream.

Copyright (C) 2009, 2010, 2011 INRIA

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

#include <stdio.h> /* for FILE */
#include <ctype.h>
#include <string.h>
#include "mpc-impl.h"

static size_t
skip_whitespace (FILE *stream)
{
   int c = getc (stream);
   size_t size = 0;
   while (c != EOF && isspace ((unsigned char) c)) {
      c = getc (stream);
      size++;
   }
   if (c != EOF)
      ungetc (c, stream);
   return size;
}

/* Extract from stream the longest string made up of alphanumeric char and
   '_' (i.e. n-char-sequence).
   The user must free the returned string. */
static char *
extract_suffix (FILE *stream)
{
  int c;
  size_t nread = 0;
  size_t strsize = 100;
  char *str = mpc_alloc_str (strsize);

  c = getc (stream);
  while (isalnum ((unsigned char) c) || c == '_') {
    str [nread] = (char) c;
    nread++;
    if (nread == strsize) {
      str = mpc_realloc_str (str, strsize, 2 * strsize);
      strsize *= 2;
         }
    c = getc (stream);
  }

  str = mpc_realloc_str (str, strsize, nread + 1);
  strsize = nread + 1;
  str [nread] = '\0';

  if (c != EOF)
    ungetc (c, stream);
  return str;
}


/* Extract from the stream the longest string of characters which are neither
   whitespace nor brackets (except for an optional bracketed n-char_sequence
   directly following nan or @nan@ independently of case).
   The user must free the returned string.                                    */
static char *
extract_string (FILE *stream)
{
  int c;
  size_t nread = 0;
  size_t strsize = 100;
  char *str = mpc_alloc_str (strsize);
  size_t lenstr;

  c = getc (stream);
  while (c != EOF && c != '\n'
         && !isspace ((unsigned char) c)
         && c != '(' && c != ')') {
    str [nread] = (char) c;
    nread++;
    if (nread == strsize) {
      str = mpc_realloc_str (str, strsize, 2 * strsize);
      strsize *= 2;
    }
    c = getc (stream);
  }

  str = mpc_realloc_str (str, strsize, nread + 1);
  strsize = nread + 1;
  str [nread] = '\0';

  if (nread == 0)
    return str;

  lenstr = nread;

  if (c == '(') {
    size_t n;
    char *suffix;
    int ret;

    /* (n-char-sequence) only after a NaN */
    if ((nread != 3
         || tolower ((unsigned char) (str[0])) != 'n'
         || tolower ((unsigned char) (str[1])) != 'a'
         || tolower ((unsigned char) (str[2])) != 'n')
        && (nread != 5
            || str[0] != '@'
            || tolower ((unsigned char) (str[1])) != 'n'
            || tolower ((unsigned char) (str[2])) != 'a'
            || tolower ((unsigned char) (str[3])) != 'n'
            || str[4] != '@')) {
      ungetc (c, stream);
      return str;
    }

    suffix = extract_suffix (stream);
    nread += strlen (suffix) + 1;
    if (nread >= strsize) {
      str = mpc_realloc_str (str, strsize, nread + 1);
      strsize = nread + 1;
    }

    /* Warning: the sprintf does not allow overlap between arguments. */
    ret = sprintf (str + lenstr, "(%s", suffix);
    MPC_ASSERT (ret >= 0);
    n = lenstr + (size_t) ret;
    MPC_ASSERT (n == nread);

    c = getc (stream);
    if (c == ')') {
      str = mpc_realloc_str (str, strsize, nread + 2);
      strsize = nread + 2;
      str [nread] = (char) c;
      str [nread+1] = '\0';
      nread++;
    }
    else if (c != EOF)
      ungetc (c, stream);

    mpc_free_str (suffix);
  }
  else if (c != EOF)
    ungetc (c, stream);

  return str;
}


int
mpc_inp_str (mpc_ptr rop, FILE *stream, size_t *read, int base,
mpc_rnd_t rnd_mode)
{
   size_t white, nread = 0;
   int inex = -1;
   int c;
   char *str;

   if (stream == NULL)
      stream = stdin;

   white = skip_whitespace (stream);
   c = getc (stream);
   if (c != EOF) {
     if (c == '(') {
       char *real_str;
       char *imag_str;
       size_t n;
       int ret;

       nread++; /* the opening parenthesis */
       white = skip_whitespace (stream);
       real_str = extract_string (stream);
       nread += strlen(real_str);

       c = getc (stream);
       if (!isspace ((unsigned int) c)) {
         if (c != EOF)
           ungetc (c, stream);
         mpc_free_str (real_str);
         goto error;
       }
       else
         ungetc (c, stream);

       white += skip_whitespace (stream);
       imag_str = extract_string (stream);
       nread += strlen (imag_str);

       str = mpc_alloc_str (nread + 2);
       ret = sprintf (str, "(%s %s", real_str, imag_str);
       MPC_ASSERT (ret >= 0);
       n = (size_t) ret;
       MPC_ASSERT (n == nread + 1);
       mpc_free_str (real_str);
       mpc_free_str (imag_str);

       white += skip_whitespace (stream);
       c = getc (stream);
       if (c == ')') {
         str = mpc_realloc_str (str, nread +2, nread + 3);
         str [nread+1] = (char) c;
         str [nread+2] = '\0';
         nread++;
       }
       else if (c != EOF)
         ungetc (c, stream);
     }
     else {
       if (c != EOF)
         ungetc (c, stream);
       str = extract_string (stream);
       nread += strlen (str);
     }

     inex = mpc_set_str (rop, str, base, rnd_mode);

     mpc_free_str (str);
   }

error:
   if (inex == -1) {
      mpfr_set_nan (mpc_realref(rop));
      mpfr_set_nan (mpc_imagref(rop));
   }
   if (read != NULL)
     *read = white + nread;
   return inex;
}
