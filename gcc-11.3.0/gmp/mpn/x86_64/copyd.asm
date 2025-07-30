dnl  AMD64 mpn_copyd -- copy limb vector, decrementing.

dnl  Copyright 2003, 2005, 2007, 2011, 2012 Free Software Foundation, Inc.

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

C	     cycles/limb
C AMD K8,K9	 1
C AMD K10	 1
C AMD bd1	 1.36
C AMD bobcat	 1.71
C Intel P4	 2-3
C Intel core2	 1
C Intel NHM	 1
C Intel SBR	 1
C Intel atom	 2
C VIA nano	 2


IFSTD(`define(`rp',`%rdi')')
IFSTD(`define(`up',`%rsi')')
IFSTD(`define(`n', `%rdx')')

IFDOS(`define(`rp',`%rcx')')
IFDOS(`define(`up',`%rdx')')
IFDOS(`define(`n', `%r8')')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(64)
PROLOGUE(mpn_copyd)
	lea	-8(up,n,8), up
	lea	(rp,n,8), rp
	sub	$4, n
	jc	L(end)
	nop

L(top):	mov	(up), %rax
	mov	-8(up), %r9
	lea	-32(rp), rp
	mov	-16(up), %r10
	mov	-24(up), %r11
	lea	-32(up), up
	mov	%rax, 24(rp)
	mov	%r9, 16(rp)
	sub	$4, n
	mov	%r10, 8(rp)
	mov	%r11, (rp)
	jnc	L(top)

L(end):	shr	R32(n)
	jnc	1f
	mov	(up), %rax
	mov	%rax, -8(rp)
	lea	-8(rp), rp
	lea	-8(up), up
1:	shr	R32(n)
	jnc	1f
	mov	(up), %rax
	mov	-8(up), %r9
	mov	%rax, -8(rp)
	mov	%r9, -16(rp)
1:	ret
EPILOGUE()
