dnl  AMD64 mpn_mul_1 optimised for Intel Atom.

dnl  Copyright 2003-2005, 2007, 2008, 2012, 2013 Free Software Foundation, Inc.

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

C	     cycles/limb	best
C AMD K8,K9
C AMD K10
C AMD bd1
C AMD bd2
C AMD bobcat
C AMD jaguar
C Intel P4
C Intel PNR
C Intel NHM
C Intel SBR
C Intel IBR
C Intel HWL
C Intel BWL
C Intel atom	17.3		this
C VIA nano

C The loop of this code is the result of running a code generation and
C optimisation tool suite written by David Harvey and Torbjorn Granlund.

define(`rp',      `%rdi')   C rcx
define(`up',      `%rsi')   C rdx
define(`n_param', `%rdx')   C r8
define(`v0',      `%rcx')   C r9

define(`n',       `%r11')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_mul_1)
	FUNC_ENTRY(4)
	xor	%r8, %r8
L(com):	mov	(up), %rax
	lea	-16(up,n_param,8), up
	lea	-8(rp,n_param,8), rp
	test	$1, R8(n_param)
	jnz	L(bx1)

L(bx0):	mov	%r8, %r9
	test	$2, R8(n_param)
	jnz	L(b10)

L(b00):	mov	$2, R32(n)
	sub	n_param, n
	jmp	L(lo0)

L(bx1):	test	$2, R8(n_param)
	jnz	L(b11)

L(b01):	mov	$3, R32(n)
	sub	n_param, n
	mul	v0
	cmp	$2, n
	jnz	L(lo1)
	jmp	L(cj1)

L(b11):	mov	$1, R32(n)
	sub	n_param, n
	jmp	L(lo3)

L(b10):	xor	R32(n), R32(n)
	sub	n_param, n
	jmp	L(lo2)

L(top):	mul	v0
	mov	%r9, -24(rp,n,8)
L(lo1):	xor	%r9d, %r9d
	add	%rax, %r8
	mov	(up,n,8), %rax
	adc	%rdx, %r9
	mov	%r8, -16(rp,n,8)
L(lo0):	xor	%r8d, %r8d
	mul	v0
	add	%rax, %r9
	mov	8(up,n,8), %rax
	adc	%rdx, %r8
	mov	%r9, -8(rp,n,8)
L(lo3):	xor	%r9d, %r9d
	mul	v0
	add	%rax, %r8
	mov	16(up,n,8), %rax
	adc	%rdx, %r9
	mov	%r8, (rp,n,8)
L(lo2):	xor	%r8d, %r8d
	mul	v0
	add	%rax, %r9
	mov	24(up,n,8), %rax
	adc	%rdx, %r8
	add	$4, n
	js	L(top)

L(end):	mul	v0
	mov	%r9, -8(rp)
L(cj1):	add	%rax, %r8
	mov	$0, R32(%rax)
	adc	%rdx, %rax
	mov	%r8, (rp)
	FUNC_EXIT()
	ret
EPILOGUE()

PROLOGUE(mpn_mul_1c)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8	')
	jmp	L(com)
EPILOGUE()
ASM_END()
