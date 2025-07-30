dnl  S/390-32 mpn_copyi

dnl  Copyright 2011 Free Software Foundation, Inc.

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


include(`../config.m4')

C            cycles/limb
C z900		 0.75
C z990           0.375
C z9		 ?
C z10		 ?
C z196		 ?

C NOTE
C  * This is based on GNU libc memcpy which was written by Martin Schwidefsky.

C INPUT PARAMETERS
define(`rp',	`%r2')
define(`up',	`%r3')
define(`n',	`%r4')

ASM_START()
PROLOGUE(mpn_copyi)
	ltr	%r4, %r4
	sll	%r4, 2
	je	L(rtn)
	ahi	%r4, -1
	lr	%r5, %r4
	srl	%r5, 8
	ltr	%r5, %r5		C < 256 bytes to copy?
	je	L(1)

L(top):	mvc	0(256, rp), 0(up)
	la	rp, 256(rp)
	la	up, 256(up)
	brct	%r5, L(top)

L(1):	bras	%r5, L(2)		C make r5 point to mvc insn
	mvc	0(1, rp), 0(up)
L(2):	ex	%r4, 0(%r5)		C execute mvc with length ((n-1) mod 256)+1
L(rtn):	br	%r14
EPILOGUE()
