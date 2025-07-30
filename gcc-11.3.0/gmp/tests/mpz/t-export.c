/* Test mpz_export.

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
    const char  *src;
    size_t      want_count;
    int         order;
    size_t      size;
    int         endian;
    int         nail;
    char        want_data[64];

  } data[] = {

    { "0", 0,1, 1,1, 0 },
    { "0", 0,1, 2,1, 0 },
    { "0", 0,1, 3,1, 0 },

    { "0x12345678", 4,1,  1,1, 0, { '\022', '\064', '\126', '\170' } },
    { "0x12345678", 1,1,  4,1, 0, { '\022', '\064', '\126', '\170' } },
    { "0x12345678", 1,-1, 4,1, 0, { '\022', '\064', '\126', '\170' } },

    { "0x12345678", 4,-1, 1,-1, 0, { '\170', '\126', '\064', '\022' } },
    { "0x12345678", 1,1,  4,-1, 0, { '\170', '\126', '\064', '\022' } },
    { "0x12345678", 1,-1, 4,-1, 0, { '\170', '\126', '\064', '\022' } },

    { "0x15", 5,1,  1,1, 7, { '\001', '\000', '\001', '\000', '\001' } },

    { "0x1FFFFFFFFFFF", 3,1,  2,1,   1, {
	'\177','\377', '\177','\377', '\177','\377' } },
    { "0x1FFFFFFFFFFF", 3,1,  2,-1,  1, {
	'\377','\177', '\377','\177', '\377','\177' } },
    { "0x7",            3,1,  2,1,  15, {
	'\000','\001', '\000','\001', '\000','\001' } },
    { "0x7",            3,1,  2,-1, 15, {
	'\001','\000', '\001','\000', '\001','\000' } },

    { "0x24", 3,1,  2,1,  14, { '\000','\002', '\000','\001', '\000','\000' }},
    { "0x24", 3,1,  2,-1, 14, { '\002','\000', '\001','\000', '\000','\000' }},
    { "0x24", 3,-1, 2,-1, 14, { '\000','\000', '\001','\000', '\002','\000' }},
    { "0x24", 3,-1, 2,1,  14, { '\000','\000', '\000','\001', '\000','\002' }},

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
      { '\125','\125','\125','\125',
	'\052','\252','\252','\252',
	'\125','\125','\125','\125' } },
    { "0x155555555555555555555555", 3,-1,  4,1,  1,
      { '\125','\125','\125','\125',
	'\052','\252','\252','\252',
	'\125','\125','\125','\125' } },
    { "0x155555555555555555555555", 3,1,  4,-1,  1,
      { '\125','\125','\125','\125',
	'\252','\252','\252','\052',
	'\125','\125','\125','\125' } },
    { "0x155555555555555555555555", 3,-1,  4,-1,  1,
      { '\125','\125','\125','\125',
	'\252','\252','\252','\052',
	'\125','\125','\125','\125' } },
  };

  char    buf[sizeof(data[0].src) + sizeof (mp_limb_t) + 128];
  char    *got_data;
  void    *ret;
  size_t  align, got_count, j;
  int     i, error = 0;
  mpz_t   src;

  mpz_init (src);

  for (i = 0; i < numberof (data); i++)
    {
      for (align = 0; align < sizeof (mp_limb_t); align++)
	{
	  mpz_set_str_or_abort (src, data[i].src, 0);
	  MPZ_CHECK_FORMAT (src);
	  got_data = buf + align;

	  ASSERT_ALWAYS (data[i].want_count * data[i].size + align
			 <= sizeof (buf));

	  memset (got_data, '\0', data[i].want_count * data[i].size);
	  ret = mpz_export (got_data, &got_count, data[i].order,
			    data[i].size, data[i].endian, data[i].nail, src);

	  if (ret != got_data)
	    {
	      printf ("return doesn't equal given pointer\n");
	      error = 1;
	    }
	  if (got_count != data[i].want_count)
	    {
	      printf ("wrong count\n");
	      error = 1;
	    }
	  if (memcmp (got_data, data[i].want_data, got_count * data[i].size) != 0)
	    {
	      printf ("wrong result data\n");
	      error = 1;
	    }
	  if (error)
	    {
	      printf ("    at data[%d]  align=%d\n", i, (int) align);
	      printf ("    src \"%s\"\n", data[i].src);
	      mpz_trace ("    src", src);
	      printf ("    order=%d  size=%lu endian=%d nail=%u\n",
		      data[i].order,
		      (unsigned long) data[i].size, data[i].endian, data[i].nail);
	      printf ("    want count %lu\n", (unsigned long) data[i].want_count);
	      printf ("    got count  %lu\n", (unsigned long) got_count);
	      printf ("    want");
	      for (j = 0; j < data[i].want_count*data[i].size; j++)
		printf (" 0x%02X,", (unsigned) (unsigned char) data[i].want_data[j]);
	      printf ("\n");
	      printf ("    got ");
	      for (j = 0; j < got_count*data[i].size; j++)
		printf (" 0x%02X,", (unsigned) (unsigned char) got_data[j]);
	      printf ("\n");
	      abort ();
	    }
	}
    }
  mpz_clear (src);
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
