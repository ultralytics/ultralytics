dnl  AMD64 mpn_lshift -- mpn left shift, optimised for Atom.

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2011, 2012 Free Software Foundation, Inc.

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
C AMD K8,K9	 ?
C AMD K10	 ?
C Intel P4	 ?
C Intel core2	 ?
C Intel NHM	 ?
C Intel SBR	 ?
C Intel atom	 4.5
C VIA nano	 ?

C TODO
C  * Consider using 4-way unrolling.  We reach 4 c/l, but the code is 2.5 times
C    larger.

C INPUT PARAMETERS
define(`rp',	`%rdi')
define(`up',	`%rsi')
define(`n',	`%rdx')
define(`cnt',	`%rcx')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_lshift)
	FUNC_ENTRY(4)
	lea	-8(up,n,8), up
	lea	-8(rp,n,8), rp
	shr	R32(n)
	mov	(up), %rax
	jnc	L(evn)

	mov	%rax, %r11
	shl	R8(%rcx), %r11
	neg	R8(%rcx)
	shr	R8(%rcx), %rax
	test	n, n
	jnz	L(gt1)
	mov	%r11, (rp)
	FUNC_EXIT()
	ret

L(gt1):	mov	-8(up), %r8
	mov	%r8, %r10
	shr	R8(%rcx), %r8
	jmp	L(lo1)

L(evn):	mov	%rax, %r10
	neg	R8(%rcx)
	shr	R8(%rcx), %rax
	mov	-8(up), %r9
	mov	%r9, %r11
	shr	R8(%rcx), %r9
	neg	R8(%rcx)
	dec	n
	lea	8(rp), rp
	lea	-8(up), up
	jz	L(end)

	ALIGN(8)
L(top):	shl	R8(%rcx), %r10
	or	%r10, %r9
	shl	R8(%rcx), %r11
	neg	R8(%rcx)
	mov	-8(up), %r8
	mov	%r8, %r10
	mov	%r9, -8(rp)
	shr	R8(%rcx), %r8
	lea	-16(rp), rp
L(lo1):	mov	-16(up), %r9
	or	%r11, %r8
	mov	%r9, %r11
	shr	R8(%rcx), %r9
	lea	-16(up), up
	neg	R8(%rcx)
	mov	%r8, (rp)
	dec	n
	jg	L(top)

L(end):	shl	R8(%rcx), %r10
	or	%r10, %r9
	shl	R8(%rcx), %r11
	mov	%r9, -8(rp)
	mov	%r11, -16(rp)
	FUNC_EXIT()
	ret
EPILOGUE()
