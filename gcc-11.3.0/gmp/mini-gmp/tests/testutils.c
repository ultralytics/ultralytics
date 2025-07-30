/*

Copyright 2013, 2014, Free Software Foundation, Inc.

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

#include "testutils.h"

/* Include it here, so we we could tweak, e.g., how MPZ_REALLOC
   works. */
#include "../mini-gmp.c"

static size_t total_alloc = 0;

/* Custom memory allocation to track memory usage, and add a small red
   zone.

   About alignment: In general, getting a block from malloc, and
   incrementing it by sizeof(size_t), like we do here, might give a
   pointer which is not properly aligned for all types. But the
   largest type we allocate space for is unsigned long (mp_limb_t),
   which shouldn't have stricter alignment requirements than
   size_t. */

static char block_end[8] =
  { 0x7c, 0x37, 0xd6, 0x12, 0xa8, 0x6c, 0x01, 0xd1 };

static void *
block_init (size_t *block, size_t size)
{
  char *p;
  *block++ = size;

  p = (char *) block;
  memcpy (p + size, block_end, sizeof(block_end));

  total_alloc += size;
  return p;
}

/* Check small redzone, return pointer to malloced block. */
static size_t *
block_check  (void *p)
{
  size_t *block = (size_t *) p - 1;
  size_t size = block[0];

  if (memcmp ((char *)p + size, block_end, sizeof(block_end)) != 0)
    {
      fprintf (stderr, "red zone overwritten.\n");
      abort ();
    }
  total_alloc -= size;
  return block;
}

static void *
tu_alloc (size_t size)
{
  size_t *block = (size_t *) malloc (sizeof(size_t) + size + sizeof(block_end));
  if (!block)
    {
      fprintf (stderr, "Virtual memory exhausted.\n");
      abort ();
    }

  return block_init (block, size);
}

static void *
tu_realloc (void *p, size_t old_size, size_t new_size)
{
  size_t *block = block_check (p);
  block = (size_t *) realloc (block, sizeof(size_t) + new_size + sizeof(block_end));
  if (!block)
    {
      fprintf (stderr, "Virtual memory exhausted.\n");
      abort ();
    }

  return block_init (block, new_size);
}

static void
tu_free (void *p, size_t old_size)
{
  free (block_check (p));
}

/* Free memory allocated via mini-gmp allocation function. */
void
testfree (void *p)
{
  void (*freefunc) (void *, size_t);
  mp_get_memory_functions (NULL, NULL, &freefunc);

  freefunc (p, 0);
}

int
main (int argc, char **argv)
{
  hex_random_init ();

  mp_set_memory_functions (tu_alloc, tu_realloc, tu_free);

  /* Currently, t-comb seems to be the only program accepting any
     arguments. It might make sense to parse common arguments here. */
  testmain (argc, argv);

  if (total_alloc != 0)
    {
      fprintf (stderr, "Memory leaked: %lu bytes.\n",
	       (unsigned long) total_alloc);
      abort ();
    }
  return 0;
}

void
testhalves (int count, void (*tested_fun) (int))
{
  void (*freefunc) (void *, size_t);
  void *(*reallocfunc) (void *, size_t, size_t);
  void *(*allocfunc) (size_t);
  size_t initial_alloc;

  mp_get_memory_functions (&allocfunc, &reallocfunc, &freefunc);
  initial_alloc = total_alloc;
  (*tested_fun) (count / 2);
  if (initial_alloc != total_alloc)
    {
      fprintf (stderr, "First half, memory leaked: %lu bytes.\n",
	       (unsigned long) total_alloc - initial_alloc);
      abort ();
    }
  mp_set_memory_functions (NULL, NULL, NULL);
  (*tested_fun) (count / 2);
  mp_set_memory_functions (allocfunc, reallocfunc, freefunc);
}

void
dump (const char *label, const mpz_t x)
{
  char *buf = mpz_get_str (NULL, 16, x);
  fprintf (stderr, "%s: %s\n", label, buf);
  testfree (buf);
}

void
mpz_set_str_or_abort (mpz_ptr z, const char *str, int base)
{
  if (mpz_set_str (z, str, base) != 0)
    {
      fprintf (stderr, "ERROR: mpz_set_str failed\n");
      fprintf (stderr, "   str  = \"%s\"\n", str);
      fprintf (stderr, "   base = %d\n", base);
      abort();
    }
}
