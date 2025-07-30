/* Stack allocation routines.  This is intended for machines without support
   for the `alloca' function.

Copyright 1996, 1997, 1999-2001, 2006 Free Software Foundation, Inc.

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

#include "gmp.h"
#include "gmp-impl.h"


struct tmp_stack
{
  void *end;
  void *alloc_point;
  struct tmp_stack *prev;
};
typedef struct tmp_stack tmp_stack;


static unsigned long max_total_allocation = 0;
static unsigned long current_total_allocation = 0;

static tmp_stack xxx = {&xxx, &xxx, 0};
static tmp_stack *current = &xxx;

/* The rounded size of the header of each allocation block.  */
#define HSIZ   ROUND_UP_MULTIPLE (sizeof (tmp_stack), __TMP_ALIGN)


/* Allocate a block of exactly <size> bytes.  This should only be called
   through the TMP_ALLOC macro, which takes care of rounding/alignment.  */
void *
__gmp_tmp_alloc (unsigned long size)
{
  void *that;

  ASSERT ((size % __TMP_ALIGN) == 0);
  ASSERT (((unsigned) current->alloc_point % __TMP_ALIGN) == 0);

  if (size > (char *) current->end - (char *) current->alloc_point)
    {
      void *chunk;
      tmp_stack *header;
      unsigned long chunk_size;
      unsigned long now;

      /* Allocate a chunk that makes the total current allocation somewhat
	 larger than the maximum allocation ever.  If size is very large, we
	 allocate that much.  */

      now = current_total_allocation + size;
      if (now > max_total_allocation)
	{
	  /* We need more temporary memory than ever before.  Increase
	     for future needs.  */
	  now = (now * 3 / 2 + __TMP_ALIGN - 1) & -__TMP_ALIGN;
	  chunk_size = now - current_total_allocation + HSIZ;
	  current_total_allocation = now;
	  max_total_allocation = current_total_allocation;
	}
      else
	{
	  chunk_size = max_total_allocation - current_total_allocation + HSIZ;
	  current_total_allocation = max_total_allocation;
	}

      chunk = (*__gmp_allocate_func) (chunk_size);
      header = (tmp_stack *) chunk;
      header->end = (char *) chunk + chunk_size;
      header->alloc_point = (char *) chunk + HSIZ;
      header->prev = current;
      current = header;
    }

  that = current->alloc_point;
  current->alloc_point = (char *) that + size;
  ASSERT (((unsigned) that % __TMP_ALIGN) == 0);
  return that;
}

/* Typically called at function entry.  <mark> is assigned so that
   __gmp_tmp_free can later be used to reclaim all subsequently allocated
   storage.  */
void
__gmp_tmp_mark (struct tmp_marker *mark)
{
  mark->which_chunk = current;
  mark->alloc_point = current->alloc_point;
}

/* Free everything allocated since <mark> was assigned by __gmp_tmp_mark */
void
__gmp_tmp_free (struct tmp_marker *mark)
{
  while (mark->which_chunk != current)
    {
      tmp_stack *tmp;

      tmp = current;
      current = tmp->prev;
      current_total_allocation -= (((char *) (tmp->end) - (char *) tmp) - HSIZ);
      (*__gmp_free_func) (tmp, (char *) tmp->end - (char *) tmp);
    }
  current->alloc_point = mark->alloc_point;
}
