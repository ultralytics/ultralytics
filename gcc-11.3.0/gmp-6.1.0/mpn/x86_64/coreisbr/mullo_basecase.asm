dnl  AMD64 mpn_mullo_basecase optimised for Intel Sandy bridge and Ivy bridge.

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
C AMD K8,K9
C AMD K10
C AMD bull
C AMD pile
C AMD steam
C AMD bobcat
C AMD jaguar
C Intel P4
C Intel core
C Intel NHM
C Intel SBR	 2.5		 2.95
C Intel IBR	 2.3		 2.68
C Intel HWL	 2.0		 2.5
C Intel BWL
C Intel atom
C VIA nano

C The inner loops of this code are the result of running a code generation and
C optimisation tool suite written by David Harvey and Torbjörn Granlund.

C TODO
C   * Implement proper cor2, replacing current cor0.
C   * Offset n by 2 in order to avoid the outer loop cmp.  (And sqr_basecase?)
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

	mov	(up), %rax
	mov	vp_param, vp

	cmp	$4, n
	jb	L(small)

	mov	(vp_param), v0
	push	%rbx
	lea	(rp,n,8), rp		C point rp at R[un]
	push	%rbp
	lea	(up,n,8), up		C point up right after U's end
	push	%r12
	neg	n
	push	%r13
	mul	v0
	mov	8(vp), v1

	test	$1, R8(n)
	jnz	L(m2b1)

L(m2b0):lea	(n), i
	xor	w0, w0
	mov	%rax, w2
	mov	%rdx, w1
	jmp	L(m2l0)

L(m2b1):lea	1(n), i
	xor	w1, w1
	xor	w2, w2
	mov	%rax, w0
	mov	%rdx, w3
	jmp	L(m2l1)

	ALIGN(32)
L(m2tp):mul	v0
	add	%rax, w0
	mov	%rdx, w3
	adc	$0, w3
L(m2l1):mov	-8(up,i,8), %rax
	mul	v1
	add	w1, w0
	adc	$0, w3
	add	%rax, w2
	mov	w0, -8(rp,i,8)
	mov	%rdx, w0
	adc	$0, w0
	mov	(up,i,8), %rax
	mul	v0
	add	%rax, w2
	mov	%rdx, w1
	adc	$0, w1
	add	w3, w2
L(m2l0):mov	(up,i,8), %rax
	adc	$0, w1
	mul	v1
	mov	w2, (rp,i,8)
	add	%rax, w0
	mov	%rdx, w2		C FIXME: dead in last iteration
	mov	8(up,i,8), %rax
	adc	$0, w2			C FIXME: dead in last iteration
	add	$2, i
	jnc	L(m2tp)

L(m2ed):imul	v0, %rax
	add	w0, %rax
	add	w1, %rax
	mov	%rax, I(-8(rp),-8(rp,i,8))

	add	$2, n
	lea	16(vp), vp
	lea	-16(up), up
	cmp	$-2, n
	jge	L(cor1)

	push	%r14
	push	%r15

L(outer):
	mov	(vp), v0
	mov	8(vp), v1
	mov	(up,n,8), %rax
	mul	v0
	test	$1, R8(n)
	jnz	L(a1x1)

L(a1x0):mov	(rp,n,8), X1
	xor	w2, w2
	xor	w1, w1
	test	$2, R8(n)
	jnz	L(a110)

L(a100):lea	1(n), i
	jmp	L(lo0)

L(a110):lea	3(n), i
	mov	%rdx, w3
	add	%rax, X1
	mov	(up,n,8), %rax
	mov	8(rp,n,8), X0
	adc	$0, w3
	jmp	L(lo2)

L(a1x1):mov	(rp,n,8), X0
	xor	w0, w0
	mov	%rdx, w1
	test	$2, R8(n)
	jz	L(a111)

L(a101):lea	2(n), i
	add	%rax, X0
	adc	$0, w1
	mov	(up,n,8), %rax
	mul	v1
	mov	8(rp,n,8), X1
	jmp	L(lo1)

L(a111):lea	(n), i
	xor	w3, w3
	jmp	L(lo3)

	ALIGN(32)
L(top):
L(lo2):	mul	v1
	mov	%rdx, w0
	add	%rax, X0
	adc	$0, w0
	add	w1, X1
	adc	$0, w3
	add	w2, X0
	adc	$0, w0
	mov	-16(up,i,8), %rax
	mul	v0
	add	%rax, X0
	mov	%rdx, w1
	adc	$0, w1
	mov	-16(up,i,8), %rax
	mul	v1
	mov	X1, -24(rp,i,8)
	mov	-8(rp,i,8), X1
	add	w3, X0
	adc	$0, w1
L(lo1):	mov	%rdx, w2
	mov	X0, -16(rp,i,8)
	add	%rax, X1
	adc	$0, w2
	mov	-8(up,i,8), %rax
	add	w0, X1
	adc	$0, w2
	mul	v0
L(lo0):	add	%rax, X1
	mov	%rdx, w3
	adc	$0, w3
	mov	-8(up,i,8), %rax
	mul	v1
	add	w1, X1
	mov	(rp,i,8), X0
	adc	$0, w3
	mov	%rdx, w0
	add	%rax, X0
	adc	$0, w0
	mov	(up,i,8), %rax
	mul	v0
	add	w2, X0
	mov	X1, -8(rp,i,8)
	mov	%rdx, w1
	adc	$0, w0
L(lo3):	add	%rax, X0
	adc	$0, w1
	mov	(up,i,8), %rax
	add	w3, X0
	adc	$0, w1
	mul	v1
	mov	8(rp,i,8), X1
	add	%rax, X1
	mov	%rdx, w2
	adc	$0, w2
	mov	8(up,i,8), %rax
	mov	X0, (rp,i,8)
	mul	v0
	add	w0, X1
	mov	%rdx, w3
	adc	$0, w2
	add	%rax, X1
	mov	8(up,i,8), %rax
	mov	16(rp,i,8), X0
	adc	$0, w3
	add	$4, i
	jnc	L(top)

L(end):	imul	v1, %rax
	add	%rax, X0
	add	w1, X1
	adc	$0, w3
	add	w2, X0
	mov	I(-8(up),-16(up,i,8)), %rax
	imul	v0, %rax
	add	X0, %rax
	mov	X1, I(-16(rp),-24(rp,i,8))
	add	w3, %rax
	mov	%rax, I(-8(rp),-16(rp,i,8))

	add	$2, n
	lea	16(vp), vp
	lea	-16(up), up
	cmp	$-2, n
	jl	L(outer)

	pop	%r15
	pop	%r14

	jnz	L(cor0)

L(cor1):mov	(vp), v0
	mov	8(vp), v1
	mov	-16(up), %rax
	mul	v0			C u0 x v2
	add	-16(rp), %rax		C FIXME: rp[0] still available in reg?
	adc	-8(rp), %rdx		C FIXME: rp[1] still available in reg?
	mov	-8(up), %r10
	imul	v0, %r10
	mov	-16(up), %r11
	imul	v1, %r11
	mov	%rax, -16(rp)
	add	%r10, %r11
	add	%rdx, %r11
	mov	%r11, -8(rp)
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret

L(cor0):mov	(vp), %r11
	imul	-8(up), %r11
	add	%rax, %r11
	mov	%r11, -8(rp)
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
L(n1):	imul	(vp_param), %rax
	mov	%rax, (rp)
	FUNC_EXIT()
	ret
L(gt1):	ja	L(gt2)
L(n2):	mov	(vp_param), %r9
	mul	%r9
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
L(n3):	mov	(vp_param), %r9
	mul	%r9		C u0 x v0
	mov	%rax, (rp)
	mov	%rdx, %r10
	mov	8(up), %rax
	mul	%r9		C u1 x v0
	imul	16(up), %r9	C u2 x v0
	add	%rax, %r10
	adc	%rdx, %r9
	mov	8(vp), %r11
	mov	(up), %rax
	mul	%r11		C u0 x v1
	add	%rax, %r10
	adc	%rdx, %r9
	imul	8(up), %r11	C u1 x v1
	add	%r11, %r9
	mov	%r10, 8(rp)
	mov	16(vp), %r10
	mov	(up), %rax
	imul	%rax, %r10	C u0 x v2
	add	%r10, %r9
	mov	%r9, 16(rp)
	FUNC_EXIT()
	ret
EPILOGUE()
