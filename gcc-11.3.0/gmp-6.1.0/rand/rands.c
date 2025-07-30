/* __gmp_rands -- global random state for old-style random functions.

   EVERYTHING IN THIS FILE IS FOR INTERNAL USE ONLY.  IT'S ALMOST CERTAIN TO
   BE SUBJECT TO INCOMPATIBLE CHANGES OR DISAPPEAR COMPLETELY IN FUTURE GNU
   MP RELEASES.  */

/*
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

#include "gmp.h"
#include "gmp-impl.h"


/* Use this via the RANDS macro in gmp-impl.h */
char             __gmp_rands_initialized = 0;
gmp_randstate_t  __gmp_rands;
