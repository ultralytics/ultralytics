/* Memory allocation used during tests.

Copyright 2001, 2002, 2007, 2013 Free Software Foundation, Inc.

This file is part of the GNU MP Library test suite.

The GNU MP Library test suite is free software; you can redistribute it
and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 3 of the License,
or (at your option) any later version.

The GNU MP Library test suite is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the GNU MP Library test suite.  If not, see https://www.gnu.org/licenses/.  */

#include <stdio.h>
#include <stdlib.h>		/* for abort */
#include <string.h>		/* for memcpy, memcmp */
#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"

#if GMP_LIMB_BITS == 64
#define PATTERN1 CNST_LIMB(0xcafebabedeadbeef)
#define PATTERN2 CNST_LIMB(0xabacadabaedeedab)
#else
#define PATTERN1 CNST_LIMB(0xcafebabe)
#define PATTERN2 CNST_LIMB(0xdeadbeef)
#endif

#if HAVE_INTPTR_T
#define PTRLIMB(p)  ((mp_limb_t) (intptr_t) p)
#else
#define PTRLIMB(p)  ((mp_limb_t) (size_t) p)
#endif

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

struct header  *tests_memory_list = NULL;

/* Return a pointer to a pointer to the found block (so it can be updated
   when unlinking). */
struct header **
tests_memory_find (void *ptr)
{
  struct header  **hp;

  for (hp = &tests_memory_list; *hp != NULL; hp = &((*hp)->next))
    if ((*hp)->ptr == ptr)
      return hp;

  return NULL;
}

int
tests_memory_valid (void *ptr)
{
  return (tests_memory_find (ptr) != NULL);
}

void *
tests_allocate (size_t size)
{
  struct header  *h;
  void *rptr, *ptr;
  mp_limb_t PATTERN2_var;

  if (size == 0)
    {
      fprintf (stderr, "tests_allocate(): attempt to allocate 0 bytes\n");
      abort ();
    }

  h = (struct header *) __gmp_default_allocate (sizeof (*h));
  h->next = tests_memory_list;
  tests_memory_list = h;

  rptr = __gmp_default_allocate (size + 2 * sizeof (mp_limb_t));
  ptr = (void *) ((gmp_intptr_t) rptr + sizeof (mp_limb_t));

  *((mp_limb_t *) ((gmp_intptr_t) ptr - sizeof (mp_limb_t)))
    = PATTERN1 - PTRLIMB (ptr);
  PATTERN2_var = PATTERN2 - PTRLIMB (ptr);
  memcpy ((void *) ((gmp_intptr_t) ptr + size), &PATTERN2_var, sizeof (mp_limb_t));

  h->size = size;
  h->ptr = ptr;
  return h->ptr;
}

void *
tests_reallocate (void *ptr, size_t old_size, size_t new_size)
{
  struct header  **hp, *h;
  void *rptr;
  mp_limb_t PATTERN2_var;

  if (new_size == 0)
    {
      fprintf (stderr, "tests_reallocate(): attempt to reallocate %p to 0 bytes\n",
	       ptr);
      abort ();
    }

  hp = tests_memory_find (ptr);
  if (hp == NULL)
    {
      fprintf (stderr, "tests_reallocate(): attempt to reallocate bad pointer %p\n",
	       ptr);
      abort ();
    }
  h = *hp;

  if (h->size != old_size)
    {
      fprintf (stderr, "tests_reallocate(): bad old size %lu, should be %lu\n",
	       (unsigned long) old_size, (unsigned long) h->size);
      abort ();
    }

  if (*((mp_limb_t *) ((gmp_intptr_t) ptr - sizeof (mp_limb_t)))
      != PATTERN1 - PTRLIMB (ptr))
    {
      fprintf (stderr, "in realloc: redzone clobbered before block\n");
      abort ();
    }
  PATTERN2_var = PATTERN2 - PTRLIMB (ptr);
  if (memcmp ((void *) ((gmp_intptr_t) ptr + h->size), &PATTERN2_var, sizeof (mp_limb_t)))
    {
      fprintf (stderr, "in realloc: redzone clobbered after block\n");
      abort ();
    }

  rptr = __gmp_default_reallocate ((void *) ((gmp_intptr_t) ptr - sizeof (mp_limb_t)),
				 old_size + 2 * sizeof (mp_limb_t),
				 new_size + 2 * sizeof (mp_limb_t));
  ptr = (void *) ((gmp_intptr_t) rptr + sizeof (mp_limb_t));

  *((mp_limb_t *) ((gmp_intptr_t) ptr - sizeof (mp_limb_t)))
    = PATTERN1 - PTRLIMB (ptr);
  PATTERN2_var = PATTERN2 - PTRLIMB (ptr);
  memcpy ((void *) ((gmp_intptr_t) ptr + new_size), &PATTERN2_var, sizeof (mp_limb_t));

  h->size = new_size;
  h->ptr = ptr;
  return h->ptr;
}

struct header **
tests_free_find (void *ptr)
{
  struct header  **hp = tests_memory_find (ptr);
  if (hp == NULL)
    {
      fprintf (stderr, "tests_free(): attempt to free bad pointer %p\n",
	       ptr);
      abort ();
    }
  return hp;
}

void
tests_free_nosize (void *ptr)
{
  struct header  **hp = tests_free_find (ptr);
  struct header  *h = *hp;
  mp_limb_t PATTERN2_var;

  *hp = h->next;  /* unlink */

  if (*((mp_limb_t *) ((gmp_intptr_t) ptr - sizeof (mp_limb_t)))
      != PATTERN1 - PTRLIMB (ptr))
    {
      fprintf (stderr, "in free: redzone clobbered before block\n");
      abort ();
    }
  PATTERN2_var = PATTERN2 - PTRLIMB (ptr);
  if (memcmp ((void *) ((gmp_intptr_t) ptr + h->size), &PATTERN2_var, sizeof (mp_limb_t)))
    {
      fprintf (stderr, "in free: redzone clobbered after block\n");
      abort ();
    }

  __gmp_default_free ((void *) ((gmp_intptr_t) ptr - sizeof(mp_limb_t)),
		      h->size + 2 * sizeof (mp_limb_t));
  __gmp_default_free (h, sizeof (*h));
}

void
tests_free (void *ptr, size_t size)
{
  struct header  **hp = tests_free_find (ptr);
  struct header  *h = *hp;

  if (h->size != size)
    {
      fprintf (stderr, "tests_free(): bad size %lu, should be %lu\n",
	       (unsigned long) size, (unsigned long) h->size);
      abort ();
    }

  tests_free_nosize (ptr);
}

void
tests_memory_start (void)
{
  mp_set_memory_functions (tests_allocate, tests_reallocate, tests_free);
}

void
tests_memory_end (void)
{
  if (tests_memory_list != NULL)
    {
      struct header  *h;
      unsigned  count;

      fprintf (stderr, "tests_memory_end(): not all memory freed\n");

      count = 0;
      for (h = tests_memory_list; h != NULL; h = h->next)
	count++;

      fprintf (stderr, "    %u blocks remaining\n", count);
      abort ();
    }
}
