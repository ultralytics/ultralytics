dnl  x86-64 mpn_lshift optimized for Pentium 4.

dnl  Copyright 2003, 2005, 2007, 2008, 2012 Free Software Foundation, Inc.

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
C AMD K8,K9	 2.5
C AMD K10	 ?
C Intel P4	 3.29
C Intel core2	 2.1 (fluctuates, presumably cache related)
C Intel corei	 ?
C Intel atom	14.3
C VIA nano	 ?

C INPUT PARAMETERS
define(`rp',`%rdi')
define(`up',`%rsi')
define(`n',`%rdx')
define(`cnt',`%cl')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(mpn_lshift)
	FUNC_ENTRY(4)
	mov	-8(up,n,8), %rax
	movd	R32(%rcx), %mm4
	neg	R32(%rcx)		C put rsh count in cl
	and	$63, R32(%rcx)
	movd	R32(%rcx), %mm5

	lea	1(n), R32(%r8)

	shr	R8(%rcx), %rax		C function return value

	and	$3, R32(%r8)
	je	L(rol)			C jump for n = 3, 7, 11, ...

	dec	R32(%r8)
	jne	L(1)
C	n = 4, 8, 12, ...
	movq	-8(up,n,8), %mm2
	psllq	%mm4, %mm2
	movq	-16(up,n,8), %mm0
	psrlq	%mm5, %mm0
	por	%mm0, %mm2
	movq	%mm2, -8(rp,n,8)
	dec	n
	jmp	L(rol)

L(1):	dec	R32(%r8)
	je	L(1x)			C jump for n = 1, 5, 9, 13, ...
C	n = 2, 6, 10, 16, ...
	movq	-8(up,n,8), %mm2
	psllq	%mm4, %mm2
	movq	-16(up,n,8), %mm0
	psrlq	%mm5, %mm0
	por	%mm0, %mm2
	movq	%mm2, -8(rp,n,8)
	dec	n
L(1x):
	cmp	$1, n
	je	L(ast)
	movq	-8(up,n,8), %mm2
	psllq	%mm4, %mm2
	movq	-16(up,n,8), %mm3
	psllq	%mm4, %mm3
	movq	-16(up,n,8), %mm0
	movq	-24(up,n,8), %mm1
	psrlq	%mm5, %mm0
	por	%mm0, %mm2
	psrlq	%mm5, %mm1
	por	%mm1, %mm3
	movq	%mm2, -8(rp,n,8)
	movq	%mm3, -16(rp,n,8)
	sub	$2, n

L(rol):	movq	-8(up,n,8), %mm2
	psllq	%mm4, %mm2
	movq	-16(up,n,8), %mm3
	psllq	%mm4, %mm3

	sub	$4, n			C				      4
	jb	L(end)			C				      2
	ALIGN(32)
L(top):
	C finish stuff from lsh block
	movq	16(up,n,8), %mm0
	movq	8(up,n,8), %mm1
	psrlq	%mm5, %mm0
	por	%mm0, %mm2
	psrlq	%mm5, %mm1
	movq	(up,n,8), %mm0
	por	%mm1, %mm3
	movq	-8(up,n,8), %mm1
	movq	%mm2, 24(rp,n,8)
	movq	%mm3, 16(rp,n,8)
	C start two new rsh
	psrlq	%mm5, %mm0
	psrlq	%mm5, %mm1

	C finish stuff from rsh block
	movq	8(up,n,8), %mm2
	movq	(up,n,8), %mm3
	psllq	%mm4, %mm2
	por	%mm2, %mm0
	psllq	%mm4, %mm3
	movq	-8(up,n,8), %mm2
	por	%mm3, %mm1
	movq	-16(up,n,8), %mm3
	movq	%mm0, 8(rp,n,8)
	movq	%mm1, (rp,n,8)
	C start two new lsh
	sub	$4, n
	psllq	%mm4, %mm2
	psllq	%mm4, %mm3

	jae	L(top)			C				      2
L(end):
	movq	8(up), %mm0
	psrlq	%mm5, %mm0
	por	%mm0, %mm2
	movq	(up), %mm1
	psrlq	%mm5, %mm1
	por	%mm1, %mm3
	movq	%mm2, 16(rp)
	movq	%mm3, 8(rp)

L(ast):	movq	(up), %mm2
	psllq	%mm4, %mm2
	movq	%mm2, (rp)
	emms
	FUNC_EXIT()
	ret
EPILOGUE()
