dnl  Pentium-4 mpn_copyd -- copy limb vector, decrementing.

dnl  Copyright 1999-2001 Free Software Foundation, Inc.

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


dnl  The std/rep/movsl/cld is very slow for small blocks on pentium4.  Its
dnl  startup time seems to be about 165 cycles.  It then needs 2.6 c/l.
dnl  We therefore use an open-coded 2 c/l copying loop.

dnl  Ultimately, we may want to use 64-bit movq or 128-bit movdqu in some
dnl  nifty unrolled arrangement.  Clearly, that could reach much higher
dnl  speeds, at least for large blocks.

include(`../config.m4')


defframe(PARAM_SIZE, 12)
defframe(PARAM_SRC, 8)
defframe(PARAM_DST,  4)

	TEXT
	ALIGN(8)

PROLOGUE(mpn_copyd)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %ecx

	movl	PARAM_SRC, %eax
	movl	PARAM_DST, %edx
	movl	%ebx, PARAM_SIZE
	addl	$-1, %ecx
	js	L(end)

L(loop):
	movl	(%eax,%ecx,4), %ebx
	movl	%ebx, (%edx,%ecx,4)
	addl	$-1, %ecx

	jns	L(loop)
L(end):
	movl	PARAM_SIZE, %ebx
	ret

EPILOGUE()
