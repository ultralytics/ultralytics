/* __gmp_doprnt_integer -- integer style formatted output.

   THE FUNCTIONS IN THIS FILE ARE FOR INTERNAL USE ONLY.  THEY'RE ALMOST
   CERTAIN TO BE SUBJECT TO INCOMPATIBLE CHANGES OR DISAPPEAR COMPLETELY IN
   FUTURE GNU MP RELEASES.

Copyright 2001 Free Software Foundation, Inc.

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

#include <stdarg.h>    /* for va_list and hence doprnt_funs_t */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "gmp.h"
#include "gmp-impl.h"


int
__gmp_doprnt_integer (const struct doprnt_funs_t *funs,
		      void *data,
		      const struct doprnt_params_t *p,
		      const char *s)
{
  int         retval = 0;
  int         slen, justlen, showbaselen, sign, signlen, slashlen, zeros;
  int         justify, den_showbaselen;
  const char  *slash, *showbase;

  /* '+' or ' ' if wanted, and don't already have '-' */
  sign = p->sign;
  if (s[0] == '-')
    {
      sign = s[0];
      s++;
    }
  signlen = (sign != '\0');

  /* if the precision was explicitly 0, print nothing for a 0 value */
  if (*s == '0' && p->prec == 0)
    s++;

  slen = strlen (s);
  slash = strchr (s, '/');

  showbase = NULL;
  showbaselen = 0;

  if (p->showbase != DOPRNT_SHOWBASE_NO)
    {
      switch (p->base) {
      case 16:  showbase = "0x"; showbaselen = 2; break;
      case -16: showbase = "0X"; showbaselen = 2; break;
      case 8:   showbase = "0";  showbaselen = 1; break;
      }
    }

  den_showbaselen = showbaselen;
  if (slash == NULL
      || (p->showbase == DOPRNT_SHOWBASE_NONZERO && slash[1] == '0'))
    den_showbaselen = 0;

  if (p->showbase == DOPRNT_SHOWBASE_NONZERO && s[0] == '0')
    showbaselen = 0;

  /* the influence of p->prec on mpq is currently undefined */
  zeros = MAX (0, p->prec - slen);

  /* space left over after actual output length */
  justlen = p->width
    - (strlen(s) + signlen + showbaselen + den_showbaselen + zeros);

  justify = p->justify;
  if (justlen <= 0) /* no justifying if exceed width */
    justify = DOPRNT_JUSTIFY_NONE;

  if (justify == DOPRNT_JUSTIFY_RIGHT)             /* pad right */
    DOPRNT_REPS (p->fill, justlen);

  DOPRNT_REPS_MAYBE (sign, signlen);               /* sign */

  DOPRNT_MEMORY_MAYBE (showbase, showbaselen);     /* base */

  DOPRNT_REPS_MAYBE ('0', zeros);                  /* zeros */

  if (justify == DOPRNT_JUSTIFY_INTERNAL)          /* pad internal */
    DOPRNT_REPS (p->fill, justlen);

  /* if there's a showbase on the denominator, then print the numerator
     separately so it can be inserted */
  if (den_showbaselen != 0)
    {
      ASSERT (slash != NULL);
      slashlen = slash+1 - s;
      DOPRNT_MEMORY (s, slashlen);                 /* numerator and slash */
      slen -= slashlen;
      s += slashlen;
      DOPRNT_MEMORY (showbase, den_showbaselen);
    }

  DOPRNT_MEMORY (s, slen);                         /* number, or denominator */

  if (justify == DOPRNT_JUSTIFY_LEFT)              /* pad left */
    DOPRNT_REPS (p->fill, justlen);

 done:
  return retval;

 error:
  retval = -1;
  goto done;
}
