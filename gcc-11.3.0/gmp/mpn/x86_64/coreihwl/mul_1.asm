dnl  AMD64 mpn_mul_1 using mulx optimised for Intel Haswell.

dnl  Contributed to the GNU project by Torbjörn Granlund.

dnl  Copyright 2012, 2013 Free Software Foundation, Inc.

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
C AMD K8,K9	n/a
C AMD K10	n/a
C AMD bd1	n/a
C AMD bd2	 ?
C AMD bobcat	n/a
C AMD jaguar	 ?
C Intel P4	n/a
C Intel PNR	n/a
C Intel NHM	n/a
C Intel SBR	n/a
C Intel IBR	n/a
C Intel HWL	 1.57		this
C Intel BWL	 ?
C Intel atom	n/a
C VIA nano	n/a

C The loop of this code is the result of running a code generation and
C optimisation tool suite written by David Harvey and Torbjorn Granlund.

define(`rp',      `%rdi')   C rcx
define(`up',      `%rsi')   C rdx
define(`n_param', `%rdx')   C r8
define(`v0_param',`%rcx')   C r9

define(`n',       `%rbp')
define(`v0',      `%rdx')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(mpn_mul_1)
	FUNC_ENTRY(4)
	push	%rbx
	push	%rbp
	push	%r12

	mov	n_param, n
	shr	$2, n

	test	$1, R8(n_param)
	jnz	L(bx1)

L(bx0):	test	$2, R8(n_param)
	mov	v0_param, v0
	jnz	L(b10)

L(b00):	mulx(	(up), %r9, %r8)
	mulx(	8,(up), %r11, %r10)
	mulx(	16,(up), %rcx, %r12)
	lea	-32(rp), rp
	jmp	L(lo0)

L(b10):	mulx(	(up), %rcx, %r12)
	mulx(	8,(up), %rbx, %rax)
	lea	-16(rp), rp
	test	n, n
	jz	L(cj2)
	mulx(	16,(up), %r9, %r8)
	lea	16(up), up
	jmp	L(lo2)

L(bx1):	test	$2, R8(n_param)
	mov	v0_param, v0
	jnz	L(b11)

L(b01):	mulx(	(up), %rbx, %rax)
	lea	-24(rp), rp
	test	n, n
	jz	L(cj1)
	mulx(	8,(up), %r9, %r8)
	lea	8(up), up
	jmp	L(lo1)

L(b11):	mulx(	(up), %r11, %r10)
	mulx(	8,(up), %rcx, %r12)
	mulx(	16,(up), %rbx, %rax)
	lea	-8(rp), rp
	test	n, n
	jz	L(cj3)
	lea	24(up), up
	jmp	L(lo3)

	ALIGN(32)
L(top):	lea	32(rp), rp
	mov	%r9, (rp)
	adc	%r8, %r11
L(lo3):	mulx(	(up), %r9, %r8)
	mov	%r11, 8(rp)
	adc	%r10, %rcx
L(lo2):	mov	%rcx, 16(rp)
	adc	%r12, %rbx
L(lo1):	mulx(	8,(up), %r11, %r10)
	adc	%rax, %r9
	mulx(	16,(up), %rcx, %r12)
	mov	%rbx, 24(rp)
L(lo0):	mulx(	24,(up), %rbx, %rax)
	lea	32(up), up
	dec	n
	jnz	L(top)

L(end):	lea	32(rp), rp
	mov	%r9, (rp)
	adc	%r8, %r11
L(cj3):	mov	%r11, 8(rp)
	adc	%r10, %rcx
L(cj2):	mov	%rcx, 16(rp)
	adc	%r12, %rbx
L(cj1):	mov	%rbx, 24(rp)
	adc	$0, %rax

	pop	%r12
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
ASM_END()
