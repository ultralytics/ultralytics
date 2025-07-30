/* Memory allocation routines.

Copyright 1991, 1993, 1994, 2000-2002, 2012 Free Software Foundation, Inc.

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
#include <stdlib.h> /* for malloc, realloc, free */

#include "gmp.h"
#include "gmp-impl.h"


void * (*__gmp_allocate_func) (size_t) = __gmp_default_allocate;
void * (*__gmp_reallocate_func) (void *, size_t, size_t) = __gmp_default_reallocate;
void   (*__gmp_free_func) (void *, size_t) = __gmp_default_free;


/* Default allocation functions.  In case of failure to allocate/reallocate
   an error message is written to stderr and the program aborts.  */

void *
__gmp_default_allocate (size_t size)
{
  void *ret;
#ifdef DEBUG
  size_t req_size = size;
  size += 2 * GMP_LIMB_BYTES;
#endif
  ret = malloc (size);
  if (ret == 0)
    {
      fprintf (stderr, "GNU MP: Cannot allocate memory (size=%lu)\n", (long) size);
      abort ();
    }

#ifdef DEBUG
  {
    mp_ptr p = ret;
    p++;
    p[-1] = (0xdeadbeef << 31) + 0xdeafdeed;
    if (req_size % GMP_LIMB_BYTES == 0)
      p[req_size / GMP_LIMB_BYTES] = ~((0xdeadbeef << 31) + 0xdeafdeed);
    ret = p;
  }
#endif
  return ret;
}

void *
__gmp_default_reallocate (void *oldptr, size_t old_size, size_t new_size)
{
  void *ret;

#ifdef DEBUG
  size_t req_size = new_size;

  if (old_size != 0)
    {
      mp_ptr p = oldptr;
      if (p[-1] != (0xdeadbeef << 31) + 0xdeafdeed)
	{
	  fprintf (stderr, "gmp: (realloc) data clobbered before allocation block\n");
	  abort ();
	}
      if (old_size % GMP_LIMB_BYTES == 0)
	if (p[old_size / GMP_LIMB_BYTES] != ~((0xdeadbeef << 31) + 0xdeafdeed))
	  {
	    fprintf (stderr, "gmp: (realloc) data clobbered after allocation block\n");
	    abort ();
	  }
      oldptr = p - 1;
    }

  new_size += 2 * GMP_LIMB_BYTES;
#endif

  ret = realloc (oldptr, new_size);
  if (ret == 0)
    {
      fprintf (stderr, "GNU MP: Cannot reallocate memory (old_size=%lu new_size=%lu)\n", (long) old_size, (long) new_size);
      abort ();
    }

#ifdef DEBUG
  {
    mp_ptr p = ret;
    p++;
    p[-1] = (0xdeadbeef << 31) + 0xdeafdeed;
    if (req_size % GMP_LIMB_BYTES == 0)
      p[req_size / GMP_LIMB_BYTES] = ~((0xdeadbeef << 31) + 0xdeafdeed);
    ret = p;
  }
#endif
  return ret;
}

void
__gmp_default_free (void *blk_ptr, size_t blk_size)
{
#ifdef DEBUG
  {
    mp_ptr p = blk_ptr;
    if (blk_size != 0)
      {
	if (p[-1] != (0xdeadbeef << 31) + 0xdeafdeed)
	  {
	    fprintf (stderr, "gmp: (free) data clobbered before allocation block\n");
	    abort ();
	  }
	if (blk_size % GMP_LIMB_BYTES == 0)
	  if (p[blk_size / GMP_LIMB_BYTES] != ~((0xdeadbeef << 31) + 0xdeafdeed))
	    {
	      fprintf (stderr, "gmp: (free) data clobbered after allocation block\n");
	      abort ();
	    }
      }
    blk_ptr = p - 1;
  }
#endif
  free (blk_ptr);
}
