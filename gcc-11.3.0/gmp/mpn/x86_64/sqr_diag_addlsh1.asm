dnl  AMD64 mpn_sqr_diag_addlsh1

dnl  Contributed to the GNU project by Torbjörn Granlund.

dnl  Copyright 2011-2013 Free Software Foundation, Inc.

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
C AMD K10	 2.5
C AMD bull	 3.6
C AMD pile	 3.6
C AMD steam	 ?
C AMD bobcat	 4
C AMD jaguar	 ?
C Intel P4	 ?
C Intel core	 4
C Intel NHM	 3.6
C Intel SBR	 3.15
C Intel IBR	 3.2
C Intel HWL	 2.6
C Intel BWL	 ?
C Intel atom	14
C VIA nano	 3.5

C When playing with pointers, set this to $2 to fall back to conservative
C indexing in wind-down code.
define(`I',`$1')

define(`rp',     `%rdi')
define(`tp',     `%rsi')
define(`up_arg', `%rdx')
define(`n',      `%rcx')

define(`up',     `%r11')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(mpn_sqr_diag_addlsh1)
	FUNC_ENTRY(4)
	push	%rbx

	dec	n
	shl	n

	mov	(up_arg), %rax

	lea	(rp,n,8), rp
	lea	(tp,n,8), tp
	lea	(up_arg,n,4), up
	neg	n

	mul	%rax
	mov	%rax, (rp,n,8)

	xor	R32(%rbx), R32(%rbx)
	jmp	L(mid)

	ALIGN(16)
L(top):	add	%r10, %r8
	adc	%rax, %r9
	mov	%r8, -8(rp,n,8)
	mov	%r9, (rp,n,8)
L(mid):	mov	8(up,n,4), %rax
	mov	(tp,n,8), %r8
	mov	8(tp,n,8), %r9
	adc	%r8, %r8
	adc	%r9, %r9
	lea	(%rdx,%rbx), %r10
	setc	R8(%rbx)
	mul	%rax
	add	$2, n
	js	L(top)

L(end):	add	%r10, %r8
	adc	%rax, %r9
	mov	%r8, I(-8(rp),-8(rp,n,8))
	mov	%r9, I((rp),(rp,n,8))
	adc	%rbx, %rdx
	mov	%rdx, I(8(rp),8(rp,n,8))

	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
