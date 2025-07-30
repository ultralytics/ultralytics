dnl  AMD64 mpn_addmul_2 optimised for Intel Haswell.

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
C Intel HWL	 2.15
C Intel BWL	 ?
C Intel atom	n/a
C VIA nano	n/a

C The loop of this code is the result of running a code generation and
C optimisation tool suite written by David Harvey and Torbjörn Granlund.

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
define(`X0', `%r12')
define(`X1', `%r13')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(mpn_addmul_2)
	FUNC_ENTRY(4)
	push	%rbx
	push	%rbp
	push	%r12
	push	%r13

	mov	(vp), v0
	mov	8(vp), v1

	mov	n_param, n
	shr	$2, n

	test	$1, R8(n_param)
	jnz	L(bx1)

L(bx0):	mov	(rp), X0
	mov	8(rp), X1
	test	$2, R8(n_param)
	jnz	L(b10)

L(b00):	mov	(up), %rdx
	lea	16(up), up
	mulx(	v0, %rax, w1)
	add	%rax, X0
	mulx(	v1, %rax, w2)
	adc	$0, w1
	mov	X0, (rp)
	add	%rax, X1
	adc	$0, w2
	mov	-8(up), %rdx
	lea	16(rp), rp
	jmp	L(lo0)

L(b10):	mov	(up), %rdx
	inc	n
	mulx(	v0, %rax, w1)
	add	%rax, X0
	adc	$0, w1
	mulx(	v1, %rax, w2)
	mov	X0, (rp)
	mov	16(rp), X0
	add	%rax, X1
	adc	$0, w2
	xor	w0, w0
	jmp	L(lo2)

L(bx1):	mov	(rp), X1
	mov	8(rp), X0
	test	$2, R8(n_param)
	jnz	L(b11)

L(b01):	mov	(up), %rdx
	mulx(	v0, %rax, w3)
	add	%rax, X1
	adc	$0, w3
	mulx(	v1, %rax, w0)
	add	%rax, X0
	adc	$0, w0
	mov	8(up), %rdx
	mov	X1, (rp)
	mov	16(rp), X1
	mulx(	v0, %rax, w1)
	lea	24(rp), rp
	lea	24(up), up
	jmp	L(lo1)

L(b11):	mov	(up), %rdx
	inc	n
	mulx(	v0, %rax, w3)
	add	%rax, X1
	adc	$0, w3
	mulx(	v1, %rax, w0)
	add	%rax, X0
	adc	$0, w0
	mov	X1, (rp)
	mov	8(up), %rdx
	mulx(	v0, %rax, w1)
	lea	8(rp), rp
	lea	8(up), up
	jmp	L(lo3)

	ALIGN(16)
L(top):	mulx(	v0, %rax, w3)
	add	w0, X1
	adc	$0, w2
	add	%rax, X1
	adc	$0, w3
	mulx(	v1, %rax, w0)
	add	%rax, X0
	adc	$0, w0
	lea	32(rp), rp
	add	w1, X1
	mov	-16(up), %rdx
	mov	X1, -24(rp)
	adc	$0, w3
	add	w2, X0
	mov	-8(rp), X1
	mulx(	v0, %rax, w1)
	adc	$0, w0
L(lo1):	add	%rax, X0
	mulx(	v1, %rax, w2)
	adc	$0, w1
	add	w3, X0
	mov	X0, -16(rp)
	adc	$0, w1
	add	%rax, X1
	adc	$0, w2
	add	w0, X1
	mov	-8(up), %rdx
	adc	$0, w2
L(lo0):	mulx(	v0, %rax, w3)
	add	%rax, X1
	adc	$0, w3
	mov	(rp), X0
	mulx(	v1, %rax, w0)
	add	%rax, X0
	adc	$0, w0
	add	w1, X1
	mov	X1, -8(rp)
	adc	$0, w3
	mov	(up), %rdx
	add	w2, X0
	mulx(	v0, %rax, w1)
	adc	$0, w0
L(lo3):	add	%rax, X0
	adc	$0, w1
	mulx(	v1, %rax, w2)
	add	w3, X0
	mov	8(rp), X1
	mov	X0, (rp)
	mov	16(rp), X0
	adc	$0, w1
	add	%rax, X1
	adc	$0, w2
L(lo2):	mov	8(up), %rdx
	lea	32(up), up
	dec	n
	jnz	L(top)

L(end):	mulx(	v0, %rax, w3)
	add	w0, X1
	adc	$0, w2
	add	%rax, X1
	adc	$0, w3
	mulx(	v1, %rdx, %rax)
	add	w1, X1
	mov	X1, 8(rp)
	adc	$0, w3
	add	w2, %rdx
	adc	$0, %rax
	add	w3, %rdx
	mov	%rdx, 16(rp)
	adc	$0, %rax

	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
