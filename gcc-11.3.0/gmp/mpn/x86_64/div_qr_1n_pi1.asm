dnl  x86-64 mpn_div_qr_1n_pi1
dnl  -- Divide an mpn number by a normalized single-limb number,
dnl     using a single-limb inverse.

dnl  Contributed to the GNU project by Niels Möller

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


C		c/l
C AMD K8,K9	13
C AMD K10	13
C AMD bull	16.5
C AMD pile	15
C AMD steam	 ?
C AMD bobcat	16
C AMD jaguar	 ?
C Intel P4	47	poor
C Intel core	19.25
C Intel NHM	18
C Intel SBR	15	poor
C Intel IBR	13
C Intel HWL	11.7
C Intel BWL	 ?
C Intel atom	52	very poor
C VIA nano	19


C INPUT Parameters
define(`QP', `%rdi')
define(`UP', `%rsi')
define(`UN_INPUT', `%rdx')
define(`U1', `%rcx')	C Also in %rax
define(`D', `%r8')
define(`DINV', `%r9')

C Invariants
define(`B2', `%rbp')
define(`B2md', `%rbx')

C Variables
define(`UN', `%r8')	C Overlaps D input
define(`T', `%r10')
define(`U0', `%r11')
define(`U2', `%r12')
define(`Q0', `%r13')
define(`Q1', `%r14')
define(`Q2', `%r15')

ABI_SUPPORT(STD64)

	ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_div_qr_1n_pi1)
	FUNC_ENTRY(6)
IFDOS(`	mov	56(%rsp), %r8	')
IFDOS(`	mov	64(%rsp), %r9	')
	dec	UN_INPUT
	jnz	L(first)

	C Just a single 2/1 division.
	C T, U0 are allocated in scratch registers
	lea	1(U1), T
	mov	U1, %rax
	mul	DINV
	mov	(UP), U0
	add	U0, %rax
	adc	T, %rdx
	mov	%rdx, T
	imul	D, %rdx
	sub	%rdx, U0
	cmp	U0, %rax
	lea	(U0, D), %rax
	cmovnc	U0, %rax
	sbb	$0, T
	cmp	D, %rax
	jc	L(single_div_done)
	sub	D, %rax
	add	$1, T
L(single_div_done):
	mov	T, (QP)
	FUNC_EXIT
	ret
L(first):
	C FIXME: Could delay some of these until we enter the loop.
	push	%r15
	push	%r14
	push	%r13
	push	%r12
	push	%rbx
	push	%rbp

	mov	D, B2
	imul	DINV, B2
	neg	B2
	mov	B2, B2md
	sub	D, B2md

	C D not needed until final reduction
	push	D
	mov	UN_INPUT, UN	C Clobbers D

	mov	DINV, %rax
	mul	U1
	mov	%rax, Q0
	add	U1, %rdx
	mov	%rdx, T

	mov	B2, %rax
	mul	U1
	mov	-8(UP, UN, 8), U0
	mov	(UP, UN, 8), U1
	mov	T, (QP, UN, 8)
	add	%rax, U0
	adc	%rdx, U1
	sbb	U2, U2
	dec	UN
	mov	U1, %rax
	jz	L(final)

	ALIGN(16)

	C Loop is 28 instructions, 30 decoder slots, should run in 10 cycles.
	C At entry, %rax holds an extra copy of U1
L(loop):
	C {Q2, Q1, Q0} <-- DINV * U1 + B (Q0 + U2 DINV) + B^2 U2
	C Remains to add in B (U1 + c)
	mov	DINV, Q1
	mov	U2, Q2
	and	U2, Q1
	neg	Q2
	mul	DINV
	add	%rdx, Q1
	adc	$0, Q2
	add	Q0, Q1
	mov	%rax, Q0
	mov	B2, %rax
	lea	(B2md, U0), T
	adc	$0, Q2

	C {U2, U1, U0} <-- (U0 + U2 B2 -c U) B + U1 B2 + u
	mul	U1
	and	B2, U2
	add	U2, U0
	cmovnc	U0, T

	C {QP+UN, ...} <-- {QP+UN, ...} + {Q2, Q1} + U1 + c
	adc	U1, Q1
	mov	-8(UP, UN, 8), U0
	adc	Q2, 8(QP, UN, 8)
	jc	L(q_incr)
L(q_incr_done):
	add	%rax, U0
	mov	T, %rax
	adc	%rdx, %rax
	mov	Q1, (QP, UN, 8)
	sbb	U2, U2
	dec	UN
	mov	%rax, U1
	jnz	L(loop)

L(final):
	pop	D

	mov	U2, Q1
	and	D, U2
	sub	U2, %rax
	neg	Q1

	mov	%rax, U1
	sub	D, %rax
	cmovc	U1, %rax
	sbb	$-1, Q1

	lea	1(%rax), T
	mul	DINV
	add	U0, %rax
	adc	T, %rdx
	mov	%rdx, T
	imul	D, %rdx
	sub	%rdx, U0
	cmp	U0, %rax
	lea	(U0, D), %rax
	cmovnc	U0, %rax
	sbb	$0, T
	cmp	D, %rax
	jc	L(div_done)
	sub	D, %rax
	add	$1, T
L(div_done):
	add	T, Q0
	mov	Q0, (QP)
	adc	Q1, 8(QP)
	jnc	L(done)
L(final_q_incr):
	addq	$1, 16(QP)
	lea	8(QP), QP
	jc	L(final_q_incr)

L(done):
	pop	%rbp
	pop	%rbx
	pop	%r12
	pop	%r13
	pop	%r14
	pop	%r15
	FUNC_EXIT
	ret

L(q_incr):
	C U1 is not live, so use it for indexing
	lea	16(QP, UN, 8), U1
L(q_incr_loop):
	addq	$1, (U1)
	jnc	L(q_incr_done)
	lea	8(U1), U1
	jmp	L(q_incr_loop)
EPILOGUE()
