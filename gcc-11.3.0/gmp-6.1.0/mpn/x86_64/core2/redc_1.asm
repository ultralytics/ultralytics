dnl  X86-64 mpn_redc_1 optimised for Intel Conroe and Wolfdale.

dnl  Contributed to the GNU project by Torbjörn Granlund.

dnl  Copyright 2003-2005, 2007, 2008, 2011-2013 Free Software Foundation, Inc.

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
C AMD K8,K9	 ?
C AMD K10	 ?
C AMD bull	 ?
C AMD pile	 ?
C AMD steam	 ?
C AMD bobcat	 ?
C AMD jaguar	 ?
C Intel P4	 ?
C Intel core	 4.5  (fluctuating)
C Intel NHM	 ?
C Intel SBR	 ?
C Intel IBR	 ?
C Intel HWL	 ?
C Intel BWL	 ?
C Intel atom	 ?
C VIA nano	 ?

C The inner loops of this code are the result of running a code generation and
C optimisation tool suite written by David Harvey and Torbjörn Granlund.

C TODO
C  * Micro-optimise, none performed thus far.
C  * Consider inlining mpn_add_n.
C  * Single basecases out before the pushes.
C  * Keep up[i] in registers for basecases (might require pushes).

C When playing with pointers, set this to $2 to fall back to conservative
C indexing in wind-down code.
define(`I',`$1')

define(`rp',          `%rdi')   C rcx
define(`up',          `%rsi')   C rdx
define(`mp_param',    `%rdx')   C r8
define(`n',           `%rcx')   C r9
define(`u0inv',       `%r8')    C stack

define(`i',           `%r14')
define(`j',           `%r15')
define(`mp',          `%r12')
define(`q0',          `%r13')

C rax rbx rcx rdx rdi rsi rbp r8 r9 r10 r11 r12 r13 r14 r15
C  X  q0'  n   X  rp  up      u0i           mp   q0 i   j

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

define(`ALIGNx', `ALIGN(16)')

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(mpn_redc_1)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8	')
	push	%rbx
	push	%rbp
	push	%r12
	push	%r13
	push	%r14
	push	%r15

	mov	(up), q0
	mov	n, j			C outer loop induction var
	lea	(mp_param,n,8), mp
	lea	-16(up,n,8), up
	neg	n
	imul	u0inv, q0		C first iteration q0

	test	$1, R8(n)
	jz	L(b0)

L(b1):	cmp	$-1, R32(n)
	jz	L(n1)
	cmp	$-3, R32(n)
	jz	L(n3)

	push	rp

L(otp1):lea	3(n), i
	mov	(mp,n,8), %rax
	mul	q0
	lea	(%rax), %rbp
	mov	8(mp,n,8), %rax
	lea	(%rdx), %r9
	mul	q0
	lea	(%rax), %r11
	mov	16(mp,n,8), %rax
	mov	16(up,n,8), %r10
	lea	(%rdx), %rdi
	mul	q0
	add	%rbp, %r10
	lea	(%rax), %rbp
	mov	24(mp,n,8), %rax
	adc	%r9, %r11
	mov	24(up,n,8), %rbx
	lea	(%rdx), %r9
	adc	$0, %rdi
	mul	q0
	add	%r11, %rbx
	lea	(%rax), %r11
	mov	32(mp,n,8), %rax
	adc	%rdi, %rbp
	mov	%rbx, 24(up,n,8)
	mov	32(up,n,8), %r10
	lea	(%rdx), %rdi
	adc	$0, %r9
	imul	u0inv, %rbx		C next q limb
	add	$2, i
	jns	L(ed1)

	ALIGNx
L(tp1):	mul	q0
	add	%rbp, %r10
	lea	(%rax), %rbp
	mov	(mp,i,8), %rax
	adc	%r9, %r11
	mov	%r10, -8(up,i,8)
	mov	(up,i,8), %r10
	lea	(%rdx), %r9
	adc	$0, %rdi
	mul	q0
	add	%r11, %r10
	lea	(%rax), %r11
	mov	8(mp,i,8), %rax
	adc	%rdi, %rbp
	mov	%r10, (up,i,8)
	mov	8(up,i,8), %r10
	lea	(%rdx), %rdi
	adc	$0, %r9
	add	$2, i
	js	L(tp1)

L(ed1):	mul	q0
	add	%rbp, %r10
	adc	%r9, %r11
	mov	%r10, I(-8(up),-8(up,i,8))
	mov	I((up),(up,i,8)), %r10
	adc	$0, %rdi
	add	%r11, %r10
	adc	%rdi, %rax
	mov	%r10, I((up),(up,i,8))
	mov	I(8(up),8(up,i,8)), %r10
	adc	$0, %rdx
	add	%rax, %r10
	mov	%r10, I(8(up),8(up,i,8))
	adc	$0, %rdx
	mov	%rdx, 16(up,n,8)	C up[0]
	mov	%rbx, q0		C previously computed q limb -> q0
	lea	8(up), up		C up++
	dec	j
	jnz	L(otp1)
	jmp	L(cj)

L(b0):	cmp	$-2, R32(n)
	jz	L(n2)
	cmp	$-4, R32(n)
	jz	L(n4)

	push	rp

L(otp0):lea	4(n), i
	mov	(mp,n,8), %rax
	mul	q0
	lea	(%rax), %r11
	mov	8(mp,n,8), %rax
	lea	(%rdx), %rdi
	mul	q0
	lea	(%rax), %rbp
	mov	16(mp,n,8), %rax
	mov	16(up,n,8), %r10
	lea	(%rdx), %r9
	mul	q0
	add	%r11, %r10
	lea	(%rax), %r11
	mov	24(mp,n,8), %rax
	adc	%rdi, %rbp
	mov	24(up,n,8), %rbx
	lea	(%rdx), %rdi
	adc	$0, %r9
	mul	q0
	add	%rbp, %rbx
	lea	(%rax), %rbp
	mov	32(mp,n,8), %rax
	adc	%r9, %r11
	mov	%rbx, 24(up,n,8)
	mov	32(up,n,8), %r10
	lea	(%rdx), %r9
	adc	$0, %rdi
	imul	u0inv, %rbx		C next q limb
	jmp	L(e0)

	ALIGNx
L(tp0):	mul	q0
	add	%rbp, %r10
	lea	(%rax), %rbp
	mov	(mp,i,8), %rax
	adc	%r9, %r11
	mov	%r10, -8(up,i,8)
	mov	(up,i,8), %r10
	lea	(%rdx), %r9
	adc	$0, %rdi
L(e0):	mul	q0
	add	%r11, %r10
	lea	(%rax), %r11
	mov	8(mp,i,8), %rax
	adc	%rdi, %rbp
	mov	%r10, (up,i,8)
	mov	8(up,i,8), %r10
	lea	(%rdx), %rdi
	adc	$0, %r9
	add	$2, i
	js	L(tp0)

L(ed0):	mul	q0
	add	%rbp, %r10
	adc	%r9, %r11
	mov	%r10, I(-8(up),-8(up,i,8))
	mov	I((up),(up,i,8)), %r10
	adc	$0, %rdi
	add	%r11, %r10
	adc	%rdi, %rax
	mov	%r10, I((up),(up,i,8))
	mov	I(8(up),8(up,i,8)), %r10
	adc	$0, %rdx
	add	%rax, %r10
	mov	%r10, I(8(up),8(up,i,8))
	adc	$0, %rdx
	mov	%rdx, 16(up,n,8)	C up[0]
	mov	%rbx, q0		C previously computed q limb -> q0
	lea	8(up), up		C up++
	dec	j
	jnz	L(otp0)

L(cj):	lea	16(up), up		C FIXME
	pop	rp
L(add_n):
IFSTD(`	lea	(up,n,8), up		C param 2: up
	lea	(up,n,8), %rdx		C param 3: up - n
	neg	R32(n)		')	C param 4: n

IFDOS(`	lea	(up,n,8), %rdx		C param 2: up
	lea	(%rdx,n,8), %r8		C param 3: up - n
	neg	R32(n)
	mov	n, %r9			C param 4: n
	mov	rp, %rcx	')	C param 1: rp

IFSTD(`	sub	$8, %rsp	')
IFDOS(`	sub	$40, %rsp	')
	ASSERT(nz, `test $15, %rsp')
	CALL(	mpn_add_n)
IFSTD(`	add	$8, %rsp	')
IFDOS(`	add	$40, %rsp	')

L(ret):	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret

L(n1):	mov	(mp_param), %rax
	mul	q0
	add	8(up), %rax
	adc	16(up), %rdx
	mov	%rdx, (rp)
	mov	$0, R32(%rax)
	adc	R32(%rax), R32(%rax)
	jmp	L(ret)

L(n2):	mov	(mp_param), %rax
	mov	(up), %rbp
	mul	q0
	add	%rax, %rbp
	mov	%rdx, %r9
	adc	$0, %r9
	mov	-8(mp), %rax
	mov	8(up), %r10
	mul	q0
	add	%rax, %r10
	mov	%rdx, %r11
	adc	$0, %r11
	add	%r9, %r10
	adc	$0, %r11
	mov	%r10, q0
	imul	u0inv, q0		C next q0
	mov	-16(mp), %rax
	mul	q0
	add	%rax, %r10
	mov	%rdx, %r9
	adc	$0, %r9
	mov	-8(mp), %rax
	mov	16(up), %r14
	mul	q0
	add	%rax, %r14
	adc	$0, %rdx
	add	%r9, %r14
	adc	$0, %rdx
	xor	R32(%rax), R32(%rax)
	add	%r11, %r14
	adc	24(up), %rdx
	mov	%r14, (rp)
	mov	%rdx, 8(rp)
	adc	R32(%rax), R32(%rax)
	jmp	L(ret)

	ALIGNx
L(n3):	mov	-24(mp), %rax
	mov	-8(up), %r10
	mul	q0
	add	%rax, %r10
	mov	-16(mp), %rax
	mov	%rdx, %r11
	adc	$0, %r11
	mov	(up), %rbp
	mul	q0
	add	%rax, %rbp
	mov	%rdx, %r9
	adc	$0, %r9
	mov	-8(mp), %rax
	add	%r11, %rbp
	mov	8(up), %r10
	adc	$0, %r9
	mul	q0
	mov	%rbp, q0
	imul	u0inv, q0		C next q0
	add	%rax, %r10
	mov	%rdx, %r11
	adc	$0, %r11
	mov	%rbp, (up)
	add	%r9, %r10
	adc	$0, %r11
	mov	%r10, 8(up)
	mov	%r11, -8(up)		C up[0]
	lea	8(up), up		C up++
	dec	j
	jnz	L(n3)

	mov	-32(up), %rdx
	mov	-24(up), %rbx
	xor	R32(%rax), R32(%rax)
	add	%rbp, %rdx
	adc	%r10, %rbx
	adc	8(up), %r11
	mov	%rdx, (rp)
	mov	%rbx, 8(rp)
	mov	%r11, 16(rp)
	adc	R32(%rax), R32(%rax)
	jmp	L(ret)

	ALIGNx
L(n4):	mov	-32(mp), %rax
	mul	q0
	lea	(%rax), %r11
	mov	-24(mp), %rax
	lea	(%rdx), %r14
	mul	q0
	lea	(%rax), %rbp
	mov	-16(mp), %rax
	mov	-16(up), %r10
	lea	(%rdx), %r9
	mul	q0
	add	%r11, %r10
	lea	(%rax), %r11
	mov	-8(mp), %rax
	adc	%r14, %rbp
	mov	-8(up), %rbx
	lea	(%rdx), %r14
	adc	$0, %r9
	mul	q0
	add	%rbp, %rbx
	adc	%r9, %r11
	mov	%rbx, -8(up)
	mov	(up), %r10
	adc	$0, %r14
	imul	u0inv, %rbx		C next q limb
	add	%r11, %r10
	adc	%r14, %rax
	mov	%r10, (up)
	mov	8(up), %r10
	adc	$0, %rdx
	add	%rax, %r10
	mov	%r10, 8(up)
	adc	$0, %rdx
	mov	%rdx, -16(up)		C up[0]
	mov	%rbx, q0		C previously computed q limb -> q0
	lea	8(up), up		C up++
	dec	j
	jnz	L(n4)
	lea	16(up), up
	jmp	L(add_n)
EPILOGUE()
ASM_END()
