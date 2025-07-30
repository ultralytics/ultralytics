dnl  X86-64 mpn_redc_1 optimised for Intel Atom.

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
C AMD bobcat	 5.0
C AMD jaguar	 ?
C Intel P4	 ?
C Intel core	 ?
C Intel NHM	 ?
C Intel SBR	 ?
C Intel IBR	 ?
C Intel HWL	 ?
C Intel BWL	 ?
C Intel atom	 ?
C VIA nano	 ?

C TODO
C  * Micro-optimise, none performed thus far.
C  * Consider inlining mpn_add_n.
C  * Single basecases out before the pushes.
C  * Make lead-in code for the inner loops be more similar.

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
define(`w0',          `%rbp')
define(`w1',          `%r9')
define(`w2',          `%r10')
define(`w3',          `%r11')

C rax rbx rcx rdx rdi rsi rbp r8 r9 r10 r11 r12 r13 r14 r15

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
	lea	(up,n,8), up
	neg	n
	imul	u0inv, q0		C first iteration q0

	test	$1, R8(n)
	jz	L(bx0)

L(bx1):	test	$2, R8(n)
	jz	L(b3)

L(b1):	cmp	$-1, R32(n)
	jz	L(n1)

L(otp1):lea	1(n), i
	mov	(mp,n,8), %rax
	mul	q0
	mov	%rax, %rbp
	mov	8(mp,n,8), %rax
	mov	%rdx, %r9
	mul	q0
	mov	%rax, %rbx
	mov	16(mp,n,8), %rax
	mov	%rdx, %r10
	mul	q0
	add	(up,n,8), %rbp
	mov	%rax, %rbp
	adc	%r9, %rbx
	mov	24(mp,n,8), %rax
	adc	$0, %r10
	mov	%rdx, %r9
	mul	q0
	add	8(up,n,8), %rbx
	mov	%rbx, 8(up,n,8)
	mov	%rax, %r11
	adc	%r10, %rbp
	mov	32(mp,n,8), %rax
	adc	$0, %r9
	imul	u0inv, %rbx		C next q limb
	jmp	L(e1)

	ALIGNx
L(tp1):	mul	q0
	add	%rbp, -24(up,i,8)
	mov	%rax, %rbp
	mov	(mp,i,8), %rax
	adc	%r9, %r11
	mov	%rdx, %r9
	adc	$0, %r10
	mul	q0
	add	%r11, -16(up,i,8)
	mov	%rax, %r11
	mov	8(mp,i,8), %rax
	adc	%r10, %rbp
	mov	%rdx, %r10
	adc	$0, %r9
	mul	q0
	add	%rbp, -8(up,i,8)
	mov	%rax, %rbp
	adc	%r9, %r11
	mov	16(mp,i,8), %rax
	adc	$0, %r10
	mov	%rdx, %r9
	mul	q0
	add	%r11, (up,i,8)
	mov	%rax, %r11
	adc	%r10, %rbp
	mov	24(mp,i,8), %rax
	adc	$0, %r9
L(e1):	add	$4, i
	mov	%rdx, %r10
	js	L(tp1)

L(ed1):	mul	q0
	add	%rbp, I(-24(up),-24(up,i,8))
	adc	%r9, %r11
	adc	$0, %r10
	add	%r11, I(-16(up),-16(up,i,8))
	adc	%r10, %rax
	adc	$0, %rdx
	add	%rax, I(-8(up),-8(up,i,8))
	adc	$0, %rdx
	mov	%rdx, (up,n,8)		C up[0]
	mov	%rbx, q0		C previously computed q limb -> q0
	lea	8(up), up		C up++
	dec	j
	jnz	L(otp1)
	jmp	L(cj)

L(b3):	cmp	$-3, R32(n)
	jz	L(n3)

L(otp3):lea	3(n), i
	mov	(mp,n,8), %rax
	mul	q0
	mov	%rax, %rbp
	mov	8(mp,n,8), %rax
	mov	%rdx, %r9
	mul	q0
	mov	%rax, %rbx
	mov	16(mp,n,8), %rax
	mov	%rdx, %r10
	mul	q0
	add	(up,n,8), %rbp
	mov	%rax, %rbp
	mov	24(mp,n,8), %rax
	adc	%r9, %rbx
	mov	%rdx, %r9
	adc	$0, %r10
	mul	q0
	add	8(up,n,8), %rbx
	mov	%rbx, 8(up,n,8)
	mov	%rax, %r11
	mov	32(mp,n,8), %rax
	adc	%r10, %rbp
	mov	%rdx, %r10
	adc	$0, %r9
	imul	u0inv, %rbx		C next q limb
	jmp	L(e3)

	ALIGNx
L(tp3):	mul	q0
	add	%rbp, -24(up,i,8)
	mov	%rax, %rbp
	mov	(mp,i,8), %rax
	adc	%r9, %r11
	mov	%rdx, %r9
	adc	$0, %r10
	mul	q0
	add	%r11, -16(up,i,8)
	mov	%rax, %r11
	mov	8(mp,i,8), %rax
	adc	%r10, %rbp
	mov	%rdx, %r10
	adc	$0, %r9
L(e3):	mul	q0
	add	%rbp, -8(up,i,8)
	mov	%rax, %rbp
	adc	%r9, %r11
	mov	16(mp,i,8), %rax
	adc	$0, %r10
	mov	%rdx, %r9
	mul	q0
	add	%r11, (up,i,8)
	mov	%rax, %r11
	adc	%r10, %rbp
	mov	24(mp,i,8), %rax
	adc	$0, %r9
	add	$4, i
	mov	%rdx, %r10
	js	L(tp3)

L(ed3):	mul	q0
	add	%rbp, I(-24(up),-24(up,i,8))
	adc	%r9, %r11
	adc	$0, %r10
	add	%r11, I(-16(up),-16(up,i,8))
	adc	%r10, %rax
	adc	$0, %rdx
	add	%rax, I(-8(up),-8(up,i,8))
	adc	$0, %rdx
	mov	%rdx, (up,n,8)		C up[0]
	mov	%rbx, q0		C previously computed q limb -> q0
	lea	8(up), up		C up++
	dec	j
	jnz	L(otp3)
C	jmp	L(cj)

L(cj):
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

L(bx0):	test	$2, R8(n)
	jnz	L(b2)

L(b0):	cmp	$-4, R32(n)
	jz	L(n4)

L(otp0):lea	4(n), i
	mov	(mp,n,8), %rax
	mul	q0
	mov	%rax, %r11
	mov	8(mp,n,8), %rax
	mov	%rdx, %r10
	mul	q0
	mov	%rax, %rbx
	mov	16(mp,n,8), %rax
	mov	%rdx, %r9
	mul	q0
	add	(up,n,8), %r11
	mov	%rax, %r11
	adc	%r10, %rbx
	mov	24(mp,n,8), %rax
	adc	$0, %r9
	mov	%rdx, %r10
	mul	q0
	add	8(up,n,8), %rbx
	mov	%rbx, 8(up,n,8)
	mov	%rax, %rbp
	mov	32(mp,n,8), %rax
	adc	%r9, %r11
	mov	%rdx, %r9
	adc	$0, %r10
	imul	u0inv, %rbx		C next q limb
	jmp	L(e0)

	ALIGNx
L(tp0):	mul	q0
	add	%rbp, -24(up,i,8)
	mov	%rax, %rbp
	mov	(mp,i,8), %rax
	adc	%r9, %r11
	mov	%rdx, %r9
	adc	$0, %r10
L(e0):	mul	q0
	add	%r11, -16(up,i,8)
	mov	%rax, %r11
	mov	8(mp,i,8), %rax
	adc	%r10, %rbp
	mov	%rdx, %r10
	adc	$0, %r9
	mul	q0
	add	%rbp, -8(up,i,8)
	mov	%rax, %rbp
	adc	%r9, %r11
	mov	16(mp,i,8), %rax
	adc	$0, %r10
	mov	%rdx, %r9
	mul	q0
	add	%r11, (up,i,8)
	mov	%rax, %r11
	adc	%r10, %rbp
	mov	24(mp,i,8), %rax
	adc	$0, %r9
	add	$4, i
	mov	%rdx, %r10
	js	L(tp0)

L(ed0):	mul	q0
	add	%rbp, I(-24(up),-24(up,i,8))
	adc	%r9, %r11
	adc	$0, %r10
	add	%r11, I(-16(up),-16(up,i,8))
	adc	%r10, %rax
	adc	$0, %rdx
	add	%rax, I(-8(up),-8(up,i,8))
	adc	$0, %rdx
	mov	%rdx, (up,n,8)		C up[0]
	mov	%rbx, q0		C previously computed q limb -> q0
	lea	8(up), up		C up++
	dec	j
	jnz	L(otp0)
	jmp	L(cj)

L(b2):	cmp	$-2, R32(n)
	jz	L(n2)

L(otp2):lea	2(n), i
	mov	(mp,n,8), %rax
	mul	q0
	mov	%rax, %r11
	mov	8(mp,n,8), %rax
	mov	%rdx, %r10
	mul	q0
	mov	%rax, %rbx
	mov	16(mp,n,8), %rax
	mov	%rdx, %r9
	mul	q0
	add	(up,n,8), %r11
	mov	%rax, %r11
	adc	%r10, %rbx
	mov	24(mp,n,8), %rax
	adc	$0, %r9
	mov	%rdx, %r10
	mul	q0
	add	8(up,n,8), %rbx
	mov	%rbx, 8(up,n,8)
	mov	%rax, %rbp
	mov	32(mp,n,8), %rax
	adc	%r9, %r11
	mov	%rdx, %r9
	adc	$0, %r10
	imul	u0inv, %rbx		C next q limb
	jmp	L(e2)

	ALIGNx
L(tp2):	mul	q0
	add	%rbp, -24(up,i,8)
	mov	%rax, %rbp
	mov	(mp,i,8), %rax
	adc	%r9, %r11
	mov	%rdx, %r9
	adc	$0, %r10
	mul	q0
	add	%r11, -16(up,i,8)
	mov	%rax, %r11
	mov	8(mp,i,8), %rax
	adc	%r10, %rbp
	mov	%rdx, %r10
	adc	$0, %r9
	mul	q0
	add	%rbp, -8(up,i,8)
	mov	%rax, %rbp
	adc	%r9, %r11
	mov	16(mp,i,8), %rax
	adc	$0, %r10
	mov	%rdx, %r9
L(e2):	mul	q0
	add	%r11, (up,i,8)
	mov	%rax, %r11
	adc	%r10, %rbp
	mov	24(mp,i,8), %rax
	adc	$0, %r9
	add	$4, i
	mov	%rdx, %r10
	js	L(tp2)

L(ed2):	mul	q0
	add	%rbp, I(-24(up),-24(up,i,8))
	adc	%r9, %r11
	adc	$0, %r10
	add	%r11, I(-16(up),-16(up,i,8))
	adc	%r10, %rax
	adc	$0, %rdx
	add	%rax, I(-8(up),-8(up,i,8))
	adc	$0, %rdx
	mov	%rdx, (up,n,8)		C up[0]
	mov	%rbx, q0		C previously computed q limb -> q0
	lea	8(up), up		C up++
	dec	j
	jnz	L(otp2)
	jmp	L(cj)

L(n1):	mov	(mp_param), %rax
	mul	q0
	add	-8(up), %rax
	adc	(up), %rdx
	mov	%rdx, (rp)
	mov	$0, R32(%rax)
	adc	R32(%rax), R32(%rax)
	jmp	L(ret)

L(n2):	mov	(mp_param), %rax
	mov	-16(up), %rbp
	mul	q0
	add	%rax, %rbp
	mov	%rdx, %r9
	adc	$0, %r9
	mov	-8(mp), %rax
	mov	-8(up), %r10
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
	mov	(up), %r14
	mul	q0
	add	%rax, %r14
	adc	$0, %rdx
	add	%r9, %r14
	adc	$0, %rdx
	xor	R32(%rax), R32(%rax)
	add	%r11, %r14
	adc	8(up), %rdx
	mov	%r14, (rp)
	mov	%rdx, 8(rp)
	adc	R32(%rax), R32(%rax)
	jmp	L(ret)

	ALIGNx
L(n3):	mov	-24(mp), %rax
	mov	-24(up), %r10
	mul	q0
	add	%rax, %r10
	mov	-16(mp), %rax
	mov	%rdx, %r11
	adc	$0, %r11
	mov	-16(up), %rbp
	mul	q0
	add	%rax, %rbp
	mov	%rdx, %r9
	adc	$0, %r9
	mov	-8(mp), %rax
	add	%r11, %rbp
	mov	-8(up), %r10
	adc	$0, %r9
	mul	q0
	mov	%rbp, q0
	imul	u0inv, q0		C next q0
	add	%rax, %r10
	mov	%rdx, %r11
	adc	$0, %r11
	mov	%rbp, -16(up)
	add	%r9, %r10
	adc	$0, %r11
	mov	%r10, -8(up)
	mov	%r11, -24(up)		C up[0]
	lea	8(up), up		C up++
	dec	j
	jnz	L(n3)

	mov	-48(up), %rdx
	mov	-40(up), %rbx
	xor	R32(%rax), R32(%rax)
	add	%rbp, %rdx
	adc	%r10, %rbx
	adc	-8(up), %r11
	mov	%rdx, (rp)
	mov	%rbx, 8(rp)
	mov	%r11, 16(rp)
	adc	R32(%rax), R32(%rax)
	jmp	L(ret)

L(n4):	mov	-32(mp), %rax
	mul	q0
	mov	%rax, %r11
	mov	-24(mp), %rax
	mov	%rdx, %r10
	mul	q0
	mov	%rax, %rbx
	mov	-16(mp), %rax
	mov	%rdx, %r9
	mul	q0
	add	-32(up), %r11
	mov	%rax, %r11
	adc	%r10, %rbx
	mov	-8(mp), %rax
	adc	$0, %r9
	mov	%rdx, %r10
	mul	q0
	add	-24(up), %rbx
	mov	%rbx, -24(up)
	adc	%r9, %r11
	adc	$0, %r10
	imul	u0inv, %rbx		C next q limb
	add	%r11, -16(up)
	adc	%r10, %rax
	adc	$0, %rdx
	add	%rax, -8(up)
	adc	$0, %rdx
	mov	%rdx, -32(up)		C up[0]
	mov	%rbx, q0		C previously computed q limb -> q0
	dec	j
	lea	8(up), up		C up++
	jnz	L(n4)
	jmp	L(cj)
EPILOGUE()
ASM_END()
