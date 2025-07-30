/* Memory allocation used during tests.

Copyright 2001-2003, 2006-2017 Free Software Foundation, Inc.
Contributed by the AriC and Caramba projects, INRIA.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

/* Note: this file comes from GMP's tests/memory.c */

#include <stdio.h>
#include <stdlib.h>  /* for abort */
#include <limits.h>

#include "mpfr-test.h"

/* Each block allocated is a separate malloc, for the benefit of a redzoning
   malloc debugger during development or when bug hunting.

   Sizes passed when reallocating or freeing are checked (the default
   routines don't care about these).

   Memory leaks are checked by requiring that all blocks have been freed
   when tests_memory_end() is called.  Test programs must be sure to have
   "clear"s for all temporary variables used.  */

struct header {
  void           *ptr;
  size_t         size;
  struct header  *next;
};

static struct header  *tests_memory_list;

/* Return a pointer to a pointer to the found block (so it can be updated
   when unlinking). */
static struct header **
tests_memory_find (void *ptr)
{
  struct header  **hp;

  for (hp = &tests_memory_list; *hp != NULL; hp = &((*hp)->next))
    if ((*hp)->ptr == ptr)
      return hp;

  return NULL;
}

/*
static int
tests_memory_valid (void *ptr)
{
  return (tests_memory_find (ptr) != NULL);
}
*/

void *
tests_allocate (size_t size)
{
  struct header  *h;

  if (size == 0)
    {
      printf ("tests_allocate(): attempt to allocate 0 bytes\n");
      abort ();
    }

  h = (struct header *) __gmp_default_allocate (sizeof (*h));
  h->next = tests_memory_list;
  tests_memory_list = h;

  h->size = size;
  h->ptr = __gmp_default_allocate (size);
  return h->ptr;
}

void *
tests_reallocate (void *ptr, size_t old_size, size_t new_size)
{
  struct header  **hp, *h;

  if (new_size == 0)
    {
      printf ("tests_reallocate(): attempt to reallocate 0x%lX to 0 bytes\n",
              (unsigned long) ptr);
      abort ();
    }

  hp = tests_memory_find (ptr);
  if (hp == NULL)
    {
      printf ("tests_reallocate(): attempt to reallocate bad pointer 0x%lX\n",
              (unsigned long) ptr);
      abort ();
    }
  h = *hp;

  if (h->size != old_size)
    {
      /* Note: we should use the standard %zu to print sizes, but
         this is not supported by old C implementations. */
      printf ("tests_reallocate(): bad old size %lu, should be %lu\n",
              (unsigned long) old_size, (unsigned long) h->size);
      abort ();
    }

  h->size = new_size;
  h->ptr = __gmp_default_reallocate (ptr, old_size, new_size);
  return h->ptr;
}

static struct header **
tests_free_find (void *ptr)
{
  struct header  **hp = tests_memory_find (ptr);
  if (hp == NULL)
    {
      printf ("tests_free(): attempt to free bad pointer 0x%lX\n",
              (unsigned long) ptr);
      abort ();
    }
  return hp;
}

static void
tests_free_nosize (void *ptr)
{
  struct header  **hp = tests_free_find (ptr);
  struct header  *h = *hp;

  *hp = h->next;  /* unlink */

  __gmp_default_free (ptr, h->size);
  __gmp_default_free (h, sizeof (*h));
}

void
tests_free (void *ptr, size_t size)
{
  struct header  **hp = tests_free_find (ptr);
  struct header  *h = *hp;

  if (h->size != size)
    {
      /* Note: we should use the standard %zu to print sizes, but
         this is not supported by old C implementations. */
      printf ("tests_free(): bad size %lu, should be %lu\n",
              (unsigned long) size, (unsigned long) h->size);
      abort ();
    }

  tests_free_nosize (ptr);
}

void
tests_memory_start (void)
{
  tests_memory_list = NULL;
  mp_set_memory_functions (tests_allocate, tests_reallocate, tests_free);
}

void
tests_memory_end (void)
{
  if (tests_memory_list != NULL)
    {
      struct header  *h;
      unsigned  count;

      printf ("tests_memory_end(): not all memory freed\n");

      count = 0;
      for (h = tests_memory_list; h != NULL; h = h->next)
        count++;

      printf ("    %u blocks remaining\n", count);
      abort ();
    }
}
