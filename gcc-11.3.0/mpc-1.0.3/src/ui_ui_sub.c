/* mpc_ui_ui_sub -- Subtract a complex number from another one given
   implicitly by its real and imaginary parts of type unsigned long int.

Copyright (C) 2011 INRIA

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

#include "mpc-impl.h"

int
mpc_ui_ui_sub (mpc_ptr rop, unsigned long int re, unsigned long int im,
               mpc_srcptr op, mpc_rnd_t rnd)
{
   int inex_re, inex_im;

   inex_re = mpfr_ui_sub (mpc_realref (rop), re, mpc_realref (op), MPC_RND_RE (rnd));
   inex_im = mpfr_ui_sub (mpc_imagref (rop), im, mpc_imagref (op), MPC_RND_IM (rnd));

   return MPC_INEX (inex_re, inex_im);
}
