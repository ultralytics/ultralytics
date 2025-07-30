/* Support for operator<< routines.

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

#include <iostream>
#include <stdarg.h>    /* for va_list and hence doprnt_funs_t */
#include <string.h>

#include "gmp.h"
#include "gmp-impl.h"

using namespace std;


/* Don't need "format" for operator<< routines, just "memory" and "reps".
   Omitting gmp_asprintf_format lets us avoid dragging vsnprintf into the
   link.  __gmp_asprintf_final will be called directly and doesn't need to
   be in the struct.  */

const struct doprnt_funs_t  __gmp_asprintf_funs_noformat = {
  NULL,
  (doprnt_memory_t) __gmp_asprintf_memory,
  (doprnt_reps_t)   __gmp_asprintf_reps,
  NULL
};


void
__gmp_doprnt_params_from_ios (struct doprnt_params_t *p, ios &o)
{
  if ((o.flags() & ios::basefield) == ios::hex)
    {
      p->expfmt = "@%c%02d";
      p->base = (o.flags() & ios::uppercase ? -16 : 16);
    }
  else
    {
      p->expfmt = (o.flags() & ios::uppercase ? "E%c%02d" : "e%c%02d");
      if ((o.flags() & ios::basefield) == ios::oct)
        p->base = 8;
      else
        p->base = 10;
    }

  /* "general" if none or more than one bit set */
  if ((o.flags() & ios::floatfield) == ios::fixed)
    p->conv = DOPRNT_CONV_FIXED;
  else if ((o.flags() & ios::floatfield) == ios::scientific)
    p->conv = DOPRNT_CONV_SCIENTIFIC;
  else
    p->conv = DOPRNT_CONV_GENERAL;

  p->exptimes4 = 0;

  p->fill = o.fill();

  /* "right" if more than one bit set */
  if ((o.flags() & ios::adjustfield) == ios::left)
    p->justify = DOPRNT_JUSTIFY_LEFT;
  else if ((o.flags() & ios::adjustfield) == ios::internal)
    p->justify = DOPRNT_JUSTIFY_INTERNAL;
  else
    p->justify = DOPRNT_JUSTIFY_RIGHT;

  /* ios::fixed allows prec==0, others take 0 as the default 6.
     Don't allow negatives (they do bad things to __gmp_doprnt_float_cxx).  */
  p->prec = MAX (0, o.precision());
  if (p->prec == 0 && p->conv != DOPRNT_CONV_FIXED)
    p->prec = 6;

  /* for hex showbase is always, for octal only non-zero */
  if (o.flags() & ios::showbase)
    p->showbase = ((o.flags() & ios::basefield) == ios::hex
                   ? DOPRNT_SHOWBASE_YES : DOPRNT_SHOWBASE_NONZERO);
  else
    p->showbase = DOPRNT_SHOWBASE_NO;

  p->showpoint = ((o.flags() & ios::showpoint) != 0);

  /* in fixed and scientific always show trailing zeros, in general format
     show them if showpoint is set (or so it seems) */
  if ((o.flags() & ios::floatfield) == ios::fixed
      || (o.flags() & ios::floatfield) == ios::scientific)
    p->showtrailing = 1;
  else
    p->showtrailing = p->showpoint;

  p->sign = (o.flags() & ios::showpos ? '+' : '\0');

  p->width = o.width();

  /* reset on each output */
  o.width (0);
}
