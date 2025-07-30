/* __gmp_fscanf_funs -- support for formatted input from a FILE.

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

#include <stdio.h>
#include "gmp.h"
#include "gmp-impl.h"


/* SunOS 4 stdio.h doesn't provide prototypes for these */
#if ! HAVE_DECL_FGETC
int fgetc (FILE *);
#endif
#if ! HAVE_DECL_FSCANF
int fscanf (FILE *, const char *, ...);
#endif
#if ! HAVE_DECL_UNGETC
int ungetc (int, FILE *);
#endif


static void
step (FILE *fp, int n)
{
}

const struct gmp_doscan_funs_t  __gmp_fscanf_funs = {
  (gmp_doscan_scan_t)  fscanf,
  (gmp_doscan_step_t)  step,
  (gmp_doscan_get_t)   fgetc,
  (gmp_doscan_unget_t) ungetc,
};
