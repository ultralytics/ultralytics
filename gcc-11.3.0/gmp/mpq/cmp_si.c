/* _mpq_cmp_si -- compare mpq and long/ulong fraction.

Copyright 2001, 2013, 2014 Free Software Foundation, Inc.

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


/* Something like mpq_cmpabs_ui would be more useful for the neg/neg case,
   and perhaps a version accepting a parameter to reverse the test, to make
   it a tail call here.  */

int
_mpq_cmp_si (mpq_srcptr q, long n, unsigned long d)
{
  /* need canonical sign to get right result */
  ASSERT (SIZ(DEN(q)) > 0);

  if (n >= 0)
    return _mpq_cmp_ui (q, n, d);
  if (SIZ(NUM(q)) >= 0)
    {
      return 1;                                /* >=0 cmp <0 */
    }
  else
    {
      mpq_t  qabs;
      SIZ(NUM(qabs)) = -SIZ(NUM(q));
      PTR(NUM(qabs)) = PTR(NUM(q));
      SIZ(DEN(qabs)) = SIZ(DEN(q));
      PTR(DEN(qabs)) = PTR(DEN(q));

      return - _mpq_cmp_ui (qabs, NEG_CAST (unsigned long, n), d);    /* <0 cmp <0 */
    }
}
