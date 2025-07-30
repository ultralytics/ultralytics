/* __gmp_invalid_operation -- invalid floating point operation.

   THE FUNCTIONS IN THIS FILE ARE FOR INTERNAL USE ONLY.  THEY'RE ALMOST
   CERTAIN TO BE SUBJECT TO INCOMPATIBLE CHANGES OR DISAPPEAR COMPLETELY IN
   FUTURE GNU MP RELEASES.

Copyright 2003 Free Software Foundation, Inc.

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

#include "config.h"

#include <signal.h>
#include <stdlib.h>

#if HAVE_UNISTD_H
#include <unistd.h>  /* for getpid */
#endif

#include "gmp.h"
#include "gmp-impl.h"


/* Incidentally, kill is not available on mingw, but that's ok, it has raise
   and we'll be using that.  */
#if ! HAVE_RAISE
#define raise(sig)   kill (getpid(), sig)
#endif


/* __gmp_invalid_operation is for an invalid floating point operation, like
   mpz_set_d on a NaN or Inf.  It's done as a subroutine to minimize code in
   places raising an exception.

   feraiseexcept(FE_INVALID) is not used here, since unfortunately on most
   systems it would require libm.

   Alternatives:

   It might be possible to check whether a hardware "invalid operation" trap
   is enabled or not before raising a signal.  This would require all
   callers to be prepared to continue with some bogus result.  Bogus returns
   are bad, but presumably an application disabling the trap is prepared for
   that.

   On some systems (eg. BSD) the signal handler can find out the reason for
   a SIGFPE (overflow, invalid, div-by-zero, etc).  Perhaps we could get
   that into our raise too.

   i386 GLIBC implements feraiseexcept(FE_INVALID) with an asm fdiv 0/0.
   That would both respect the exceptions mask and give a reason code in a
   BSD signal.  */

void
__gmp_invalid_operation (void)
{
  raise (SIGFPE);
  abort ();
}
