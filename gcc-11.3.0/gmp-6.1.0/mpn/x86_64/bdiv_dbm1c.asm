dnl  x86_64 mpn_bdiv_dbm1.

dnl  Copyright 2008, 2011, 2012 Free Software Foundation, Inc.

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
C AMD K8,K9	 2.25
C AMD K10	 2.25
C Intel P4	12.5
C Intel core2	 4
C Intel NHM	 3.75
C Intel SBR	 3.6
C Intel atom	20
C VIA nano	 4

C TODO
C  * Optimise feed-in code.

C INPUT PARAMETERS
define(`qp',	  `%rdi')
define(`up',	  `%rsi')
define(`n_param', `%rdx')
define(`bd',	  `%rcx')
define(`cy',	  `%r8')

define(`n',       `%r9')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_bdiv_dbm1c)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8	')
	mov	(up), %rax
	mov	n_param, n
	mov	R32(n_param), R32(%r11)
	mul	bd
	lea	(up,n,8), up
	lea	(qp,n,8), qp
	neg	n
	and	$3, R32(%r11)
	jz	L(lo0)
	lea	-4(n,%r11), n
	cmp	$2, R32(%r11)
	jc	L(lo1)
	jz	L(lo2)
	jmp	L(lo3)

	ALIGN(16)
L(top):	mov	(up,n,8), %rax
	mul	bd
L(lo0):	sub	%rax, %r8
	mov	%r8, (qp,n,8)
	sbb	%rdx, %r8
	mov	8(up,n,8), %rax
	mul	bd
L(lo3):	sub	%rax, %r8
	mov	%r8, 8(qp,n,8)
	sbb	%rdx, %r8
	mov	16(up,n,8), %rax
	mul	bd
L(lo2):	sub	%rax, %r8
	mov	%r8, 16(qp,n,8)
	sbb	%rdx, %r8
	mov	24(up,n,8), %rax
	mul	bd
L(lo1):	sub	%rax, %r8
	mov	%r8, 24(qp,n,8)
	sbb	%rdx, %r8
	add	$4, n
	jnz	L(top)

	mov	%r8, %rax
	FUNC_EXIT()
	ret
EPILOGUE()
