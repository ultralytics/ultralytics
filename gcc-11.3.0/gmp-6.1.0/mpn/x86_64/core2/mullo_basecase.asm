dnl  AMD64 mpn_mullo_basecase optimised for Conroe/Wolfdale/Nehalem/Westmere.

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
C Intel core	 4.0		4.18-4.25
C Intel NHM	 3.75		4.06-4.2
C Intel SBR
C Intel IBR
C Intel HWL
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
define(`n_param',  `%rcx')

define(`v0',       `%r10')
define(`v1',       `%r11')
define(`w0',       `%rbx')
define(`w1',       `%rcx')
define(`w2',       `%rbp')
define(`w3',       `%r12')
define(`n',        `%r9')
define(`i',        `%r13')
define(`vp',       `%r8')

define(`X0',       `%r14')
define(`X1',       `%r15')

C rax rbx rcx rdx rdi rsi rbp r8 r9 r10 r11 r12 r13 r14 r15

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

define(`ALIGNx', `ALIGN(16)')

define(`N', 85)
ifdef(`N',,`define(`N',0)')
define(`MOV', `ifelse(eval(N & $3),0,`mov	$1, $2',`lea	($1), $2')')

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(mpn_mullo_basecase)
	FUNC_ENTRY(4)

	mov	(up), %rax
	mov	vp_param, vp

	cmp	$4, n_param
	jb	L(small)

	mov	(vp_param), v0
	push	%rbx
	lea	(rp,n_param,8), rp	C point rp at R[un]
	push	%rbp
	lea	(up,n_param,8), up	C point up right after U's end
	push	%r12
	mov	$0, R32(n)		C FIXME
	sub	n_param, n
	push	%r13
	mul	v0
	mov	8(vp), v1

	test	$1, R8(n_param)
	jnz	L(m2x1)

L(m2x0):test	$2, R8(n_param)
	jnz	L(m2b2)

L(m2b0):lea	(n), i
	mov	%rax, (rp,n,8)
	mov	%rdx, w1
	mov	(up,n,8), %rax
	xor	R32(w2), R32(w2)
	jmp	L(m2e0)

L(m2b2):lea	-2(n), i
	mov	%rax, w2
	mov	(up,n,8), %rax
	mov	%rdx, w3
	xor	R32(w0), R32(w0)
	jmp	L(m2e2)

L(m2x1):test	$2, R8(n_param)
	jnz	L(m2b3)

L(m2b1):lea	1(n), i
	mov	%rax, (rp,n,8)
	mov	(up,n,8), %rax
	mov	%rdx, w0
	xor	R32(w1), R32(w1)
	jmp	L(m2e1)

L(m2b3):lea	-1(n), i
	xor	R32(w3), R32(w3)
	mov	%rax, w1
	mov	%rdx, w2
	mov	(up,n,8), %rax
	jmp	L(m2e3)

	ALIGNx
L(m2tp):mul	v0
	add	%rax, w3
	mov	-8(up,i,8), %rax
	mov	w3, -8(rp,i,8)
	adc	%rdx, w0
	adc	$0, R32(w1)
L(m2e1):mul	v1
	add	%rax, w0
	adc	%rdx, w1
	mov	$0, R32(w2)
	mov	(up,i,8), %rax
	mul	v0
	add	%rax, w0
	mov	w0, (rp,i,8)
	adc	%rdx, w1
	mov	(up,i,8), %rax
	adc	$0, R32(w2)
L(m2e0):mul	v1
	add	%rax, w1
	adc	%rdx, w2
	mov	8(up,i,8), %rax
	mul	v0
	mov	$0, R32(w3)
	add	%rax, w1
	adc	%rdx, w2
	adc	$0, R32(w3)
	mov	8(up,i,8), %rax
L(m2e3):mul	v1
	add	%rax, w2
	mov	w1, 8(rp,i,8)
	adc	%rdx, w3
	mov	$0, R32(w0)
	mov	16(up,i,8), %rax
	mul	v0
	add	%rax, w2
	mov	16(up,i,8), %rax
	adc	%rdx, w3
	adc	$0, R32(w0)
L(m2e2):mul	v1
	mov	$0, R32(w1)		C FIXME: dead in last iteration
	add	%rax, w3
	mov	24(up,i,8), %rax
	mov	w2, 16(rp,i,8)
	adc	%rdx, w0		C FIXME: dead in last iteration
	add	$4, i
	js	L(m2tp)

L(m2ed):imul	v0, %rax
	add	w3, %rax
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

L(a1x0):mov	%rax, X1
	MOV(	%rdx, X0, 8)
	mov	(up,n,8), %rax
	mul	v1
	test	$2, R8(n)
	jnz	L(a110)

L(a100):lea	(n), i
	mov	(rp,n,8), w3
	mov	%rax, w0
	MOV(	%rdx, w1, 16)
	jmp	L(lo0)

L(a110):lea	2(n), i
	mov	(rp,n,8), w1
	mov	%rax, w2
	mov	8(up,n,8), %rax
	MOV(	%rdx, w3, 1)
	jmp	L(lo2)

L(a1x1):mov	%rax, X0
	MOV(	%rdx, X1, 2)
	mov	(up,n,8), %rax
	mul	v1
	test	$2, R8(n)
	jz	L(a111)

L(a101):lea	1(n), i
	MOV(	%rdx, w0, 4)
	mov	(rp,n,8), w2
	mov	%rax, w3
	jmp	L(lo1)

L(a111):lea	-1(n), i
	MOV(	%rdx, w2, 64)
	mov	%rax, w1
	mov	(rp,n,8), w0
	mov	8(up,n,8), %rax
	jmp	L(lo3)

	ALIGNx
L(top):	mul	v1
	add	w0, w1
	adc	%rax, w2
	mov	-8(up,i,8), %rax
	MOV(	%rdx, w3, 1)
	adc	$0, w3
L(lo2):	mul	v0
	add	w1, X1
	mov	X1, -16(rp,i,8)
	adc	%rax, X0
	MOV(	%rdx, X1, 2)
	adc	$0, X1
	mov	-8(up,i,8), %rax
	mul	v1
	MOV(	%rdx, w0, 4)
	mov	-8(rp,i,8), w1
	add	w1, w2
	adc	%rax, w3
	adc	$0, w0
L(lo1):	mov	(up,i,8), %rax
	mul	v0
	add	w2, X0
	adc	%rax, X1
	mov	X0, -8(rp,i,8)
	MOV(	%rdx, X0, 8)
	adc	$0, X0
	mov	(up,i,8), %rax
	mov	(rp,i,8), w2
	mul	v1
	add	w2, w3
	adc	%rax, w0
	MOV(	%rdx, w1, 16)
	adc	$0, w1
L(lo0):	mov	8(up,i,8), %rax
	mul	v0
	add	w3, X1
	mov	X1, (rp,i,8)
	adc	%rax, X0
	MOV(	%rdx, X1, 32)
	mov	8(rp,i,8), w3
	adc	$0, X1
	mov	8(up,i,8), %rax
	mul	v1
	add	w3, w0
	MOV(	%rdx, w2, 64)
	adc	%rax, w1
	mov	16(up,i,8), %rax
	adc	$0, w2
L(lo3):	mul	v0
	add	w0, X0
	mov	X0, 8(rp,i,8)
	MOV(	%rdx, X0, 128)
	adc	%rax, X1
	mov	16(up,i,8), %rax
	mov	16(rp,i,8), w0
	adc	$0, X0
	add	$4, i
	jnc	L(top)

L(end):	imul	v1, %rax
	add	w0, w1
	adc	%rax, w2
	mov	I(-8(up),-8(up,i,8)), %rax
	imul	v0, %rax
	add	w1, X1
	mov	X1, I(-16(rp),-16(rp,i,8))
	adc	X0, %rax
	mov	I(-8(rp),-8(rp,i,8)), w1
	add	w1, w2
	add	w2, %rax
	mov	%rax, I(-8(rp),-8(rp,i,8))

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
	mov	-8(up), %rbx
	imul	v0, %rbx
	mov	-16(up), %rcx
	imul	v1, %rcx
	mov	%rax, -16(rp)
	add	%rbx, %rcx
	add	%rdx, %rcx
	mov	%rcx, -8(rp)
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
	cmp	$2, n_param
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
