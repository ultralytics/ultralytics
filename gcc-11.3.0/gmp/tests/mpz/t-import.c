/* Test mpz_import.

Copyright 2002, 2003 Free Software Foundation, Inc.

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
#include <stdlib.h>
#include <string.h>
#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"


void
check_data (void)
{
  static const struct {
    const char  *want;
    size_t      count;
    int         order;
    size_t      size;
    int         endian;
    int         nail;
    char        src[64];

  } data[] = {

    { "0", 0,1, 1,1, 0 },
    { "0", 1,1, 0,1, 0 },

    { "0x12345678", 4,1,  1,1, 0, { '\22', '\64', '\126', '\170' } },
    { "0x12345678", 1,1,  4,1, 0, { '\22', '\64', '\126', '\170' } },
    { "0x12345678", 1,-1, 4,1, 0, { '\22', '\64', '\126', '\170' } },

    { "0x12345678", 4,-1, 1,-1, 0, { '\170', '\126', '\064', '\22' } },
    { "0x12345678", 1,1,  4,-1, 0, { '\170', '\126', '\064', '\22' } },
    { "0x12345678", 1,-1, 4,-1, 0, { '\170', '\126', '\064', '\22' } },

    { "0",    5,1,  1,1, 7, { '\376', '\376', '\376', '\376', '\376' } },
    { "0",    5,-1, 1,1, 7, { '\376', '\376', '\376', '\376', '\376' } },
    { "0x15", 5,1,  1,1, 7, { '\377', '\376', '\377', '\376', '\377' } },

    { "0",    3,1,  2,1,   1, { '\200','\000', '\200','\000', '\200','\000' }},
    { "0",    3,1,  2,-1,  1, { '\000','\200', '\000','\200', '\000','\200' }},
    { "0",    3,1,  2,1,  15, { '\377','\376', '\377','\376', '\377','\376' }},

    { "0x2A", 3,1,  2,1, 14, { '\377','\376', '\377','\376', '\377','\376' } },
    { "0x06", 3,1,  2,1, 14, { '\377','\374', '\377','\375', '\377','\376' } },
    { "0x24", 3,-1, 2,1, 14, { '\377','\374', '\377','\375', '\377','\376' } },

    { "0x123456789ABC", 3,1,  2,1,  0, {
        '\022','\064', '\126','\170', '\232','\274' } },
    { "0x123456789ABC", 3,-1, 2,1,  0, {
        '\232','\274', '\126','\170', '\022','\064' } },
    { "0x123456789ABC", 3,1,  2,-1, 0, {
        '\064','\022', '\170','\126', '\274','\232' } },
    { "0x123456789ABC", 3,-1, 2,-1, 0, {
        '\274','\232', '\170','\126', '\064','\022' } },

    { "0x112233445566778899AABBCC", 3,1,  4,1,  0,
      { '\021','\042','\063','\104',
        '\125','\146','\167','\210',
        '\231','\252','\273','\314' } },
    { "0x112233445566778899AABBCC", 3,-1, 4,1,  0,
      { '\231','\252','\273','\314',
        '\125','\146','\167','\210',
        '\021','\042','\063','\104' } },
    { "0x112233445566778899AABBCC", 3,1,  4,-1, 0,
      { '\104','\063','\042','\021',
        '\210','\167','\146','\125',
        '\314','\273','\252','\231' } },
    { "0x112233445566778899AABBCC", 3,-1, 4,-1, 0,
      { '\314','\273','\252','\231',
        '\210','\167','\146','\125',
        '\104','\063','\042','\021' } },

    { "0x100120023003400450056006700780089009A00AB00BC00C", 3,1,  8,1,  0,
      { '\020','\001','\040','\002','\060','\003','\100','\004',
        '\120','\005','\140','\006','\160','\007','\200','\010',
        '\220','\011','\240','\012','\260','\013','\300','\014' } },
    { "0x100120023003400450056006700780089009A00AB00BC00C", 3,-1, 8,1,  0,
      { '\220','\011','\240','\012','\260','\013','\300','\014',
        '\120','\005','\140','\006','\160','\007','\200','\010',
        '\020','\001','\040','\002','\060','\003','\100','\004' } },
    { "0x100120023003400450056006700780089009A00AB00BC00C", 3,1,  8,-1, 0,
      { '\004','\100','\003','\060','\002','\040','\001','\020',
        '\010','\200','\007','\160','\006','\140','\005','\120',
        '\014','\300','\013','\260','\012','\240','\011','\220' } },
    { "0x100120023003400450056006700780089009A00AB00BC00C", 3,-1, 8,-1, 0,
      { '\014','\300','\013','\260','\012','\240','\011','\220',
        '\010','\200','\007','\160','\006','\140','\005','\120',
        '\004','\100','\003','\060','\002','\040','\001','\020' } },

    { "0x155555555555555555555555", 3,1,  4,1,  1,
      { '\325','\125','\125','\125',
        '\252','\252','\252','\252',
        '\325','\125','\125','\125' } },
    { "0x155555555555555555555555", 3,-1,  4,1,  1,
      { '\325','\125','\125','\125',
        '\252','\252','\252','\252',
        '\325','\125','\125','\125' } },
    { "0x155555555555555555555555", 3,1,  4,-1,  1,
      { '\125','\125','\125','\325',
        '\252','\252','\252','\252',
        '\125','\125','\125','\325' } },
    { "0x155555555555555555555555", 3,-1,  4,-1,  1,
      { '\125','\125','\125','\325',
        '\252','\252','\252','\252',
        '\125','\125','\125','\325' } },
  };

  char    buf[sizeof(data[0].src) + sizeof (mp_limb_t)];
  char    *src;
  size_t  align;
  int     i;
  mpz_t   got, want;

  mpz_init (got);
  mpz_init (want);

  for (i = 0; i < numberof (data); i++)
    {
      for (align = 0; align < sizeof (mp_limb_t); align++)
        {
          mpz_set_str_or_abort (want, data[i].want, 0);
          src = buf + align;
          memcpy (src, data[i].src, data[i].count * data[i].size);

          mpz_set_ui (got, 0L);
          mpz_import (got, data[i].count, data[i].order,
                      data[i].size, data[i].endian, data[i].nail, src);

          MPZ_CHECK_FORMAT (got);
          if (mpz_cmp (got, want) != 0)
            {
              printf ("wrong at data[%d]\n", i);
              printf ("    count=%lu order=%d  size=%lu endian=%d nail=%u  align=%lu\n",
                      (unsigned long) data[i].count, data[i].order,
                      (unsigned long) data[i].size, data[i].endian, data[i].nail,
                      (unsigned long) align);
              mpz_trace ("    got ", got);
              mpz_trace ("    want", want);
              abort ();
            }
        }
    }
  mpz_clear (got);
  mpz_clear (want);
}


int
main (void)
{
  tests_start ();

  mp_trace_base = -16;
  check_data ();

  tests_end ();
  exit (0);
}
