divert(-1)

dnl  m4 macros for Alpha assembler.

dnl  Copyright 2003, 2004 Free Software Foundation, Inc.

dnl  This file is part of the GNU MP Library.
dnl
dnl  The GNU MP Library is free software; you can redistribute it and/or modify
dnl  it under the terms of either:
dnl
dnl    * the GNU Lesser General Public License as published by the Free
dnl      Software Foundation; either version 3 of the License, or (at your
dnl      option) any later version.
dnl
dnl  or
dnl
dnl    * the GNU General Public License as published by the Free Software
dnl      Foundation; either version 2 of the License, or (at your option) any
dnl      later version.
dnl
dnl  or both in parallel, as here.
dnl
dnl  The GNU MP Library is distributed in the hope that it will be useful, but
dnl  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
dnl  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
dnl  for more details.
dnl
dnl  You should have received copies of the GNU General Public License and the
dnl  GNU Lesser General Public License along with the GNU MP Library.  If not,
dnl  see https://www.gnu.org/licenses/.


dnl  Usage: ASSERT([reg] [,code])
dnl
dnl  Require that the given reg is non-zero after executing the test code.
dnl  For example,
dnl
dnl         ASSERT(r8,
dnl         `       cmpult r16, r17, r8')
dnl
dnl  If the register argument is empty then nothing is tested, the code is
dnl  just executed.  This can be used for setups required by later ASSERTs.
dnl  If the code argument is omitted then the register is just tested, with
dnl  no special setup code.

define(ASSERT,
m4_assert_numargs_range(1,2)
m4_assert_defined(`WANT_ASSERT')
`ifelse(WANT_ASSERT,1,
`ifelse(`$2',,,`$2')
ifelse(`$1',,,
`	bne	$1, L(ASSERTok`'ASSERT_label_counter)
	.long	0	C halt
L(ASSERTok`'ASSERT_label_counter):
define(`ASSERT_label_counter',eval(ASSERT_label_counter+1))
')
')')
define(`ASSERT_label_counter',1)


dnl  Usage: bigend(`code')
dnl
dnl  Emit the given code only for a big-endian system, like Unicos.  This
dnl  can be used for instance for extra stuff needed by extwl.

define(bigend,
m4_assert_numargs(1)
`ifdef(`HAVE_LIMB_BIG_ENDIAN',`$1',
`ifdef(`HAVE_LIMB_LITTLE_ENDIAN',`',
`m4_error(`Cannot assemble, unknown limb endianness')')')')


dnl  Usage: bwx_available_p
dnl
dnl  Evaluate to 1 if the BWX byte memory instructions are available, or to
dnl  0 if not.
dnl
dnl  Listing the chips which do have BWX means anything we haven't looked at
dnl  will use safe non-BWX code.  The only targets without BWX currently are
dnl  plain alpha (ie. ev4) and alphaev5.

define(bwx_available_p,
m4_assert_numargs(-1)
`m4_ifdef_anyof_p(
	`HAVE_HOST_CPU_alphaev56',
	`HAVE_HOST_CPU_alphapca56',
	`HAVE_HOST_CPU_alphapca57',
	`HAVE_HOST_CPU_alphaev6',
	`HAVE_HOST_CPU_alphaev67',
	`HAVE_HOST_CPU_alphaev68',
	`HAVE_HOST_CPU_alphaev69',
	`HAVE_HOST_CPU_alphaev7',
	`HAVE_HOST_CPU_alphaev79')')


dnl  Usage: unop
dnl
dnl  The Cray Unicos assembler lacks unop, so give the equivalent ldq_u
dnl  explicitly.

define(unop,
m4_assert_numargs(-1)
`ldq_u	r31, 0(r30)')


divert
