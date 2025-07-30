/* TMP_ALLOC routines for debugging.

Copyright 2000, 2001, 2004 Free Software Foundation, Inc.

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
#include <string.h>
#include "gmp.h"
#include "gmp-impl.h"


/* This method aims to help a malloc debugger find problems.  A linked list
   of allocated block is kept for TMP_FREE to release.  This is reentrant
   and thread safe.

   Each TMP_ALLOC is a separate malloced block, so redzones or sentinels
   applied by a malloc debugger either above or below can guard against
   accesses outside the allocated area.

   A marker is a "struct tmp_debug_t *" so that TMP_DECL can initialize it
   to NULL and we can detect TMP_ALLOC without TMP_MARK.

   It will work to realloc an MPZ_TMP_INIT variable, but when TMP_FREE comes
   to release the memory it will have the old size, thereby triggering an
   error from tests/memory.c.

   Possibilities:

   It'd be possible to keep a global list of active "struct tmp_debug_t"
   records, so at the end of a program any TMP leaks could be printed.  But
   if only a couple of routines are under test at any one time then the
   likely culprit should be easy enough to spot.  */


void
__gmp_tmp_debug_mark (const char *file, int line,
                      struct tmp_debug_t **markp, struct tmp_debug_t *mark,
                      const char *decl_name, const char *mark_name)
{
  if (strcmp (mark_name, decl_name) != 0)
    {
      __gmp_assert_header (file, line);
      fprintf (stderr, "GNU MP: TMP_MARK(%s) but TMP_DECL(%s) is in scope\n",
               mark_name, decl_name);
      abort ();
    }

  if (*markp != NULL)
    {
      __gmp_assert_header (file, line);
      fprintf (stderr, "GNU MP: Repeat of TMP_MARK(%s)\n", mark_name);
      if (mark->file != NULL && mark->file[0] != '\0' && mark->line != -1)
        {
          __gmp_assert_header (mark->file, mark->line);
          fprintf (stderr, "previous was here\n");
        }
      abort ();
    }

  *markp = mark;
  mark->file = file;
  mark->line = line;
  mark->list = NULL;
}

void *
__gmp_tmp_debug_alloc (const char *file, int line, int dummy,
                       struct tmp_debug_t **markp,
                       const char *decl_name, size_t size)
{
  struct tmp_debug_t        *mark = *markp;
  struct tmp_debug_entry_t  *p;

  ASSERT_ALWAYS (size >= 1);

  if (mark == NULL)
    {
      __gmp_assert_header (file, line);
      fprintf (stderr, "GNU MP: TMP_ALLOC without TMP_MARK(%s)\n", decl_name);
      abort ();
    }

  p = __GMP_ALLOCATE_FUNC_TYPE (1, struct tmp_debug_entry_t);
  p->size = size;
  p->block = (*__gmp_allocate_func) (size);
  p->next = mark->list;
  mark->list = p;
  return p->block;
}

void
__gmp_tmp_debug_free (const char *file, int line, int dummy,
                      struct tmp_debug_t **markp,
                      const char *decl_name, const char *free_name)
{
  struct tmp_debug_t        *mark = *markp;
  struct tmp_debug_entry_t  *p, *next;

  if (mark == NULL)
    {
      __gmp_assert_header (file, line);
      fprintf (stderr, "GNU MP: TMP_FREE(%s) without TMP_MARK(%s)\n",
               free_name, decl_name);
      abort ();
    }

  if (strcmp (free_name, decl_name) != 0)
    {
      __gmp_assert_header (file, line);
      fprintf (stderr, "GNU MP: TMP_FREE(%s) when TMP_DECL(%s) is in scope\n",
               free_name, decl_name);
      abort ();
    }

  p = mark->list;
  while (p != NULL)
    {
      next = p->next;
      (*__gmp_free_func) (p->block, p->size);
      __GMP_FREE_FUNC_TYPE (p, 1, struct tmp_debug_entry_t);
      p = next;
    }

  *markp = NULL;
}
