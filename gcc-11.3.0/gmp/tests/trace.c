/* Support for diagnostic traces.

Copyright 1999-2005 Free Software Foundation, Inc.

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


/* Future: Would like commas printed between limbs in hex or binary, but
   perhaps not always since it might upset cutting and pasting into bc or
   whatever.  */


#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* for strlen */

#include "gmp.h"
#include "gmp-impl.h"

#include "tests.h"


/* Number base for the various trace printing routines.
   Set this in main() or with the debugger.
   If hexadecimal is going to be fed into GNU bc, remember to use -16
   because bc requires upper case.  */

int  mp_trace_base = 10;


void
mp_trace_start (const char *name)
{
  if (name != NULL && name[0] != '\0')
    printf ("%s=", name);

  switch (ABS (mp_trace_base)) {
  case  2: printf ("bin:");                         break;
  case  8: printf ("oct:");                         break;
  case 10:                                          break;
  case 16: printf ("0x");                           break;
  default: printf ("base%d:", ABS (mp_trace_base)); break;
  }
}

/* Print "name=value\n" to stdout for an mpq_t value.  */
void
mpq_trace (const char *name, mpq_srcptr q)
{
  mp_trace_start (name);
  if (q == NULL)
    {
      printf ("NULL\n");
      return;
    }

  mpq_out_str (stdout, mp_trace_base, q);
  printf ("\n");
}


/* Print "name=value\n" to stdout for an mpz_t value.  */
void
mpz_trace (const char *name, mpz_srcptr z)
{
  mpq_t      q;
  mp_limb_t  one;

  if (z == NULL)
    {
      mpq_trace (name, NULL);
      return;
    }

  q->_mp_num._mp_alloc = ALLOC(z);
  q->_mp_num._mp_size = SIZ(z);
  q->_mp_num._mp_d = PTR(z);

  one = 1;
  q->_mp_den._mp_alloc = 1;
  q->_mp_den._mp_size = 1;
  q->_mp_den._mp_d = &one;

  mpq_trace(name, q);
}


/* Print "name=value\n" to stdout for an mpf_t value. */
void
mpf_trace (const char *name, mpf_srcptr f)
{
  mp_trace_start (name);
  if (f == NULL)
    {
      printf ("NULL\n");
      return;
    }

  mpf_out_str (stdout, ABS (mp_trace_base), 0, f);
  printf ("\n");
}


/* Print "namenum=value\n" to stdout for an mpz_t value.
   "name" should have a "%d" to get the number. */
void
mpz_tracen (const char *name, int num, mpz_srcptr z)
{
  if (name != NULL && name[0] != '\0')
    {
      printf (name, num);
      putchar ('=');
    }
  mpz_trace (NULL, z);
}


/* Print "name=value\n" to stdout for an mpn style ptr,size. */
void
mpn_trace (const char *name, mp_srcptr ptr, mp_size_t size)
{
  mpz_t  z;
  if (ptr == NULL)
    {
      mpz_trace (name, NULL);
      return;
    }
  MPN_NORMALIZE (ptr, size);
  PTR(z) = (mp_ptr) ptr;
  SIZ(z) = size;
  ALLOC(z) = size;
  mpz_trace (name, z);
}

/* Print "name=value\n" to stdout for a limb, nail doesn't have to be zero. */
void
mp_limb_trace (const char *name, mp_limb_t n)
{
#if GMP_NAIL_BITS != 0
  mp_limb_t  a[2];
  a[0] = n & GMP_NUMB_MASK;
  a[1] = n >> GMP_NUMB_BITS;
  mpn_trace (name, a, (mp_size_t) 2);
#else
  mpn_trace (name, &n, (mp_size_t) 1);
#endif
}


/* Print "namenum=value\n" to stdout for an mpn style ptr,size.
   "name" should have a "%d" to get the number.  */
void
mpn_tracen (const char *name, int num, mp_srcptr ptr, mp_size_t size)
{
  if (name != NULL && name[0] != '\0')
    {
      printf (name, num);
      putchar ('=');
    }
  mpn_trace (NULL, ptr, size);
}


/* Print "namenum=value\n" to stdout for an array of mpn style ptr,size.

   "a" is an array of pointers, each a[i] is a pointer to "size" many limbs.
   The formal parameter isn't mp_srcptr because that causes compiler
   warnings, but the values aren't modified.

   "name" should have a printf style "%d" to get the array index.  */

void
mpn_tracea (const char *name, const mp_ptr *a, int count, mp_size_t size)
{
  int i;
  for (i = 0; i < count; i++)
    mpn_tracen (name, i, a[i], size);
}


/* Print "value\n" to a file for an mpz_t value.  Any previous contents of
   the file are overwritten, so you need different file names each time this
   is called.

   Overwriting the file is a feature, it means you get old data replaced
   when you run a test program repeatedly.  */

void
mpn_trace_file (const char *filename, mp_srcptr ptr, mp_size_t size)
{
  FILE   *fp;
  mpz_t  z;

  fp = fopen (filename, "w");
  if (fp == NULL)
    {
      perror ("fopen");
      abort();
    }

  MPN_NORMALIZE (ptr, size);
  PTR(z) = (mp_ptr) ptr;
  SIZ(z) = (int) size;

  mpz_out_str (fp, mp_trace_base, z);
  fprintf (fp, "\n");

  if (ferror (fp) || fclose (fp) != 0)
    {
      printf ("error writing %s\n", filename);
      abort();
    }
}


/* Print "value\n" to a set of files, one file for each element of the given
   array of mpn style ptr,size.  Any previous contents of the files are
   overwritten, so you need different file names each time this is called.
   Each file is "filenameN" where N is 0 to count-1.

   "a" is an array of pointers, each a[i] is a pointer to "size" many limbs.
   The formal parameter isn't mp_srcptr because that causes compiler
   warnings, but the values aren't modified.

   Overwriting the files is a feature, it means you get old data replaced
   when you run a test program repeatedly.  The output style isn't
   particularly pretty, but at least it gets something out, and you can cat
   the files into bc, or whatever. */

void
mpn_tracea_file (const char *filename,
                 const mp_ptr *a, int count, mp_size_t size)
{
  char  *s;
  int   i;
  TMP_DECL;

  TMP_MARK;
  s = (char *) TMP_ALLOC (strlen (filename) + 50);

  for (i = 0; i < count; i++)
    {
      sprintf (s, "%s%d", filename, i);
      mpn_trace_file (s, a[i], size);
    }

  TMP_FREE;
}


void
byte_trace (const char *name, const void *ptr, mp_size_t size)
{
  const char *fmt;
  mp_size_t  i;

  mp_trace_start (name);

  switch (mp_trace_base) {
  case   8: fmt = " %o"; break;
  case  10: fmt = " %d"; break;
  case  16: fmt = " %x"; break;
  case -16: fmt = " %X"; break;
  default: printf ("Oops, unsupported base in byte_trace\n"); abort (); break;
  }

  for (i = 0; i < size; i++)
    printf (fmt, (int) ((unsigned char *) ptr)[i]);
  printf ("\n");
}

void
byte_tracen (const char *name, int num, const void *ptr, mp_size_t size)
{
  if (name != NULL && name[0] != '\0')
    {
      printf (name, num);
      putchar ('=');
    }
  byte_trace (NULL, ptr, size);
}


void
d_trace (const char *name, double d)
{
  union {
    double         d;
    unsigned char  b[sizeof(double)];
  } u;
  int  i;

  if (name != NULL && name[0] != '\0')
    printf ("%s=", name);

  u.d = d;
  printf ("[");
  for (i = 0; i < sizeof (u.b); i++)
    {
      if (i != 0)
        printf (" ");
      printf ("%02X", (int) u.b[i]);
    }
  printf ("] %.20g\n", d);
}
