/* mpc_out_str -- Output a complex number on a given stream.

Copyright (C) 2009, 2011 INRIA

This file is part of GNU MPC.

GNU MPC is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

GNU MPC is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see http://www.gnu.org/licenses/ .
*/

#include <stdio.h> /* for FILE */
#include <ctype.h>
#include "mpc-impl.h"

size_t
mpc_out_str (FILE *stream, int base, size_t n, mpc_srcptr op, mpc_rnd_t rnd) {
   size_t size = 3; /* for '(', ' ' and ')' */

   if (stream == NULL)
      stream = stdout; /* fprintf does not allow NULL as first argument */

   fprintf (stream, "(");
   size += mpfr_out_str (stream, base, n, mpc_realref(op), MPC_RND_RE(rnd));
   fprintf (stream, " ");
   size += mpfr_out_str (stream, base, n, mpc_imagref(op), MPC_RND_RE(rnd));
   fprintf (stream, ")");

   return size;
}
