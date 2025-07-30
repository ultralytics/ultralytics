dnl  AMD64 mpn_mullo_basecase optimised for Intel Haswell.

dnl  Contributed to the GNU project by Torbjörn Granlund.

dnl  Copyright 2008, 2009, 2011-2013 Free Software Foundation, Inc.

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

C cycles/limb	mul_2		addmul_2
C AMD K8,K9	n/a		n/a
C AMD K10	n/a		n/a
C AMD bull	n/a		n/a
C AMD pile	n/a		n/a
C AMD steam	 ?		 ?
C AMD bobcat	n/a		n/a
C AMD jaguar	 ?		 ?
C Intel P4	n/a		n/a
C Intel core	n/a		n/a
C Intel NHM	n/a		n/a
C Intel SBR	n/a		n/a
C Intel IBR	n/a		n/a
C Intel HWL	 1.86		 2.15
C Intel BWL	 ?		 ?
C Intel atom	n/a		n/a
C VIA nano	n/a		n/a

C The inner loops of this code are the result of running a code generation and
C optimisation tool suite written by David Harvey and Torbjörn Granlund.

C TODO
C   * Implement proper cor2, replacing current cor0.
C   * Micro-optimise.

C When playing with pointers, set this to $2 to fall back to conservative
C indexing in wind-down code.
define(`I',`$1')

define(`rp',       `%rdi')
define(`up',       `%rsi')
define(`vp_param', `%rdx')
define(`n',        `%rcx')

define(`vp',       `%r8')
define(`X0',       `%r14')
define(`X1',       `%r15')

define(`w0',       `%r10')
define(`w1',       `%r11')
define(`w2',       `%r12')
define(`w3',       `%r13')
define(`i',        `%rbp')
define(`v0',       `%r9')
define(`v1',       `%rbx')

C rax rbx rcx rdx rdi rsi rbp r8 r9 r10 r11 r12 r13 r14 r15

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(mpn_mullo_basecase)
	FUNC_ENTRY(4)

	mov	vp_param, vp
	mov	(up), %rdx

	cmp	$4, n
	jb	L(small)

	push	%rbx
	push	%rbp
	push	%r12
	push	%r13

	mov	(vp), v0
	mov	8(vp), v1

	lea	2(n), i
	shr	$2, i
	neg	n
	add	$2, n

	push	up			C put entry `up' on stack

	test	$1, R8(n)
	jnz	L(m2x1)

L(m2x0):mulx(	v0, w0, w3)
	xor	R32(w2), R32(w2)
	test	$2, R8(n)
	jz	L(m2b2)

L(m2b0):lea	-8(rp), rp
	lea	-8(up), up
	jmp	L(m2e0)

L(m2b2):lea	-24(rp), rp
	lea	8(up), up
	jmp	L(m2e2)

L(m2x1):mulx(	v0, w2, w1)
	xor	R32(w0), R32(w0)
	test	$2, R8(n)
	jnz	L(m2b3)

L(m2b1):jmp	L(m2e1)

L(m2b3):lea	-16(rp), rp
	lea	-16(up), up
	jmp	L(m2e3)

	ALIGN(16)
L(m2tp):mulx(	v1, %rax, w0)
	add	%rax, w2
	mov	(up), %rdx
	mulx(	v0, %rax, w1)
	adc	$0, w0
	add	%rax, w2
	adc	$0, w1
	add	w3, w2
L(m2e1):mov	w2, (rp)
	adc	$0, w1
	mulx(	v1, %rax, w2)
	add	%rax, w0
	mov	8(up), %rdx
	adc	$0, w2
	mulx(	v0, %rax, w3)
	add	%rax, w0
	adc	$0, w3
	add	w1, w0
L(m2e0):mov	w0, 8(rp)
	adc	$0, w3
	mulx(	v1, %rax, w0)
	add	%rax, w2
	mov	16(up), %rdx
	mulx(	v0, %rax, w1)
	adc	$0, w0
	add	%rax, w2
	adc	$0, w1
	add	w3, w2
L(m2e3):mov	w2, 16(rp)
	adc	$0, w1
	mulx(	v1, %rax, w2)
	add	%rax, w0
	mov	24(up), %rdx
	adc	$0, w2
	mulx(	v0, %rax, w3)
	add	%rax, w0
	adc	$0, w3
	add	w1, w0
	lea	32(up), up
L(m2e2):mov	w0, 24(rp)
	adc	$0, w3
	dec	i
	lea	32(rp), rp
	jnz	L(m2tp)

L(m2ed):mulx(	v1, %rax, w0)
	add	%rax, w2
	mov	(up), %rdx
	mulx(	v0, %rax, w1)
	add	w2, %rax
	add	w3, %rax
	mov	%rax, (rp)

	mov	(%rsp), up		C restore `up' to beginning
	lea	16(vp), vp
	lea	8(rp,n,8), rp		C put back rp to old rp + 2
	add	$2, n
	jge	L(cor1)

	push	%r14
	push	%r15

L(outer):
	mov	(vp), v0
	mov	8(vp), v1

	lea	(n), i
	sar	$2, i

	mov	(up), %rdx
	test	$1, R8(n)
	jnz	L(bx1)

L(bx0):	mov	(rp), X1
	mov	8(rp), X0
	mulx(	v0, %rax, w3)
	add	%rax, X1
	adc	$0, w3
	mulx(	v1, %rax, w0)
	add	%rax, X0
	adc	$0, w0
	mov	8(up), %rdx
	mov	X1, (rp)
	mulx(	v0, %rax, w1)
	test	$2, R8(n)
	jz	L(b2)

L(b0):	lea	8(rp), rp
	lea	8(up), up
	jmp	L(lo0)

L(b2):	mov	16(rp), X1
	lea	24(rp), rp
	lea	24(up), up
	jmp	L(lo2)

L(bx1):	mov	(rp), X0
	mov	8(rp), X1
	mulx(	v0, %rax, w1)
	add	%rax, X0
	mulx(	v1, %rax, w2)
	adc	$0, w1
	mov	X0, (rp)
	add	%rax, X1
	adc	$0, w2
	mov	8(up), %rdx
	test	$2, R8(n)
	jnz	L(b3)

L(b1):	lea	16(up), up
	lea	16(rp), rp
	jmp	L(lo1)

L(b3):	mov	16(rp), X0
	lea	32(up), up
	mulx(	v0, %rax, w3)
	inc	i
	jz	L(cj3)
	jmp	L(lo3)

	ALIGN(16)
L(top):	mulx(	v0, %rax, w3)
	add	w0, X1
	adc	$0, w2
L(lo3):	add	%rax, X1
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
L(lo2):	add	%rax, X0
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
L(lo1):	mulx(	v0, %rax, w3)
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
L(lo0):	add	%rax, X0
	adc	$0, w1
	mulx(	v1, %rax, w2)
	add	w3, X0
	mov	8(rp), X1
	mov	X0, (rp)
	mov	16(rp), X0
	adc	$0, w1
	add	%rax, X1
	adc	$0, w2
	mov	8(up), %rdx
	lea	32(up), up
	inc	i
	jnz	L(top)

L(end):	mulx(	v0, %rax, w3)
	add	w0, X1
	adc	$0, w2
L(cj3):	add	%rax, X1
	adc	$0, w3
	mulx(	v1, %rax, w0)
	add	%rax, X0
	add	w1, X1
	mov	-16(up), %rdx
	mov	X1, 8(rp)
	adc	$0, w3
	add	w2, X0
	mulx(	v0, %rax, w1)
	add	X0, %rax
	add	w3, %rax
	mov	%rax, 16(rp)

	mov	16(%rsp), up		C restore `up' to beginning
	lea	16(vp), vp
	lea	24(rp,n,8), rp		C put back rp to old rp + 2
	add	$2, n
	jl	L(outer)

	pop	%r15
	pop	%r14

	jnz	L(cor0)

L(cor1):mov	(vp), v0
	mov	8(vp), v1
	mov	(up), %rdx
	mulx(	v0, %r12, %rbp)		C u0 x v2
	add	(rp), %r12		C FIXME: rp[0] still available in reg?
	adc	%rax, %rbp
	mov	8(up), %r10
	imul	v0, %r10
	imul	v1, %rdx
	mov	%r12, (rp)
	add	%r10, %rdx
	add	%rbp, %rdx
	mov	%rdx, 8(rp)
	pop	%rax			C deallocate `up' copy
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret

L(cor0):mov	(vp), %r11
	imul	(up), %r11
	add	%rax, %r11
	mov	%r11, (rp)
	pop	%rax			C deallocate `up' copy
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret

	ALIGN(16)
L(small):
	cmp	$2, n
	jae	L(gt1)
L(n1):	imul	(vp), %rdx
	mov	%rdx, (rp)
	FUNC_EXIT()
	ret
L(gt1):	ja	L(gt2)
L(n2):	mov	(vp), %r9
	mulx(	%r9, %rax, %rdx)
	mov	%rax, (rp)
	mov	8(up), %rax
	imul	%r9, %rax
	add	%rax, %rdx
	mov	8(vp), %r9
	mov	(up), %rcx
	imul	%r9, %rcx
	add	%rcx, %rdx
	mov	%rdx, 8(rp)
	FUNC_EXIT()
	ret
L(gt2):
L(n3):	mov	(vp), %r9
	mulx(	%r9, %rax, %r10)	C u0 x v0
	mov	%rax, (rp)
	mov	8(up), %rdx
	mulx(	%r9, %rax, %rdx)	C u1 x v0
	imul	16(up), %r9		C u2 x v0
	add	%rax, %r10
	adc	%rdx, %r9
	mov	8(vp), %r11
	mov	(up), %rdx
	mulx(	%r11, %rax, %rdx)	C u0 x v1
	add	%rax, %r10
	adc	%rdx, %r9
	imul	8(up), %r11		C u1 x v1
	add	%r11, %r9
	mov	%r10, 8(rp)
	mov	16(vp), %r10
	mov	(up), %rax
	imul	%rax, %r10		C u0 x v2
	add	%r10, %r9
	mov	%r9, 16(rp)
	FUNC_EXIT()
	ret
EPILOGUE()
