dnl  AMD64 mpn_mul_2 optimised for Intel Haswell.

dnl  Contributed to the GNU project by Torbjörn Granlund.

dnl  Copyright 2013 Free Software Foundation, Inc.

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
C AMD K8,K9	n/a
C AMD K10	n/a
C AMD bull	n/a
C AMD pile	n/a
C AMD steam	 ?
C AMD bobcat	n/a
C AMD jaguar	 ?
C Intel P4	n/a
C Intel core	n/a
C Intel NHM	n/a
C Intel SBR	n/a
C Intel IBR	n/a
C Intel HWL	 1.86
C Intel BWL	 ?
C Intel atom	n/a
C VIA nano	n/a

C The loop of this code is the result of running a code generation and
C optimisation tool suite written by David Harvey and Torbjörn Granlund.

C TODO
C  * Move test and jcc together, for insn fusion.

define(`rp',     `%rdi')
define(`up',     `%rsi')
define(`n_param',`%rdx')
define(`vp',     `%rcx')

define(`v0', `%r8')
define(`v1', `%r9')
define(`w0', `%rbx')
define(`w1', `%rcx')
define(`w2', `%rbp')
define(`w3', `%r10')
define(`n',  `%r11')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(mpn_mul_2)
	FUNC_ENTRY(4)
	push	%rbx
	push	%rbp

	mov	(vp), v0
	mov	8(vp), v1

	lea	3(n_param), n
	shr	$2, n

	test	$1, R8(n_param)
	jnz	L(bx1)

L(bx0):	xor	w0, w0
	test	$2, R8(n_param)
	mov	(up), %rdx
	mulx(	v0, w2, w1)
	jz	L(lo0)

L(b10):	lea	-16(rp), rp
	lea	-16(up), up
	jmp	L(lo2)

L(bx1):	xor	w2, w2
	test	$2, R8(n_param)
	mov	(up), %rdx
	mulx(	v0, w0, w3)
	jnz	L(b11)

L(b01):	lea	-24(rp), rp
	lea	8(up), up
	jmp	L(lo1)

L(b11):	lea	-8(rp), rp
	lea	-8(up), up
	jmp	L(lo3)

	ALIGN(16)
L(top):	mulx(	v1, %rax, w0)
	add	%rax, w2		C 0
	mov	(up), %rdx
	mulx(	v0, %rax, w1)
	adc	$0, w0			C 1
	add	%rax, w2		C 0
	adc	$0, w1			C 1
	add	w3, w2			C 0
L(lo0):	mov	w2, (rp)		C 0
	adc	$0, w1			C 1
	mulx(	v1, %rax, w2)
	add	%rax, w0		C 1
	mov	8(up), %rdx
	adc	$0, w2			C 2
	mulx(	v0, %rax, w3)
	add	%rax, w0		C 1
	adc	$0, w3			C 2
	add	w1, w0			C 1
L(lo3):	mov	w0, 8(rp)		C 1
	adc	$0, w3			C 2
	mulx(	v1, %rax, w0)
	add	%rax, w2		C 2
	mov	16(up), %rdx
	mulx(	v0, %rax, w1)
	adc	$0, w0			C 3
	add	%rax, w2		C 2
	adc	$0, w1			C 3
	add	w3, w2			C 2
L(lo2):	mov	w2, 16(rp)		C 2
	adc	$0, w1			C 3
	mulx(	v1, %rax, w2)
	add	%rax, w0		C 3
	mov	24(up), %rdx
	adc	$0, w2			C 4
	mulx(	v0, %rax, w3)
	add	%rax, w0		C 3
	adc	$0, w3			C 4
	add	w1, w0			C 3
	lea	32(up), up
L(lo1):	mov	w0, 24(rp)		C 3
	adc	$0, w3			C 4
	dec	n
	lea	32(rp), rp
	jnz	L(top)

L(end):	mulx(	v1, %rdx, %rax)
	add	%rdx, w2
	adc	$0, %rax
	add	w3, w2
	mov	w2, (rp)
	adc	$0, %rax

	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
