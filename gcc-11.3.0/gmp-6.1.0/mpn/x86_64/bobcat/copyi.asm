dnl  AMD64 mpn_copyi optimised for AMD bobcat.

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
C AMD K10	 1-2  (alignment fluctuations)
C AMD bd1	 ?
C AMD bobcat	 1.5
C Intel P4	 2.8
C Intel core2	 1
C Intel NHM	 1-1.25
C Intel SBR	 1
C Intel atom	 2.87
C VIA nano	 2

C INPUT PARAMETERS
C rp	rdi
C up	rsi
C n	rdx

define(`rp',`%rdi')
define(`up',`%rsi')
define(`n',`%rdx')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_copyi)
	FUNC_ENTRY(3)
	lea	-32(up,n,8), up
	lea	-32(rp,n,8), rp
	neg	n
	add	$4, n
	jg	L(end)
	ALIGN(16)
L(top):	mov	(up,n,8), %r8
	mov	%r8, (rp,n,8)
	mov	8(up,n,8), %r8
	mov	%r8, 8(rp,n,8)
	mov	16(up,n,8), %r8
	mov	%r8, 16(rp,n,8)
	mov	24(up,n,8), %r8
	mov	%r8, 24(rp,n,8)
L(ent):	add	$4, n
	jle	L(top)

L(end):	cmp	$4, R32(n)
	jz	L(ret)
	mov	(up,n,8), %r8
	mov	%r8, (rp,n,8)
	cmp	$3, R32(n)
	jz	L(ret)
	mov	8(up,n,8), %r8
	mov	%r8, 8(rp,n,8)
	cmp	$2, R32(n)
	jz	L(ret)
	mov	16(up,n,8), %r8
	mov	%r8, 16(rp,n,8)

L(ret):	FUNC_EXIT()
	ret
EPILOGUE()
