dnl  AMD64 mpn_mul_basecase optimised for Intel Haswell.

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

C cycles/limb	mul_1		mul_2		mul_3		addmul_2
C AMD K8,K9	n/a		n/a		 -		n/a
C AMD K10	n/a		n/a		 -		n/a
C AMD bull	n/a		n/a		 -		n/a
C AMD pile	n/a		n/a		 -		n/a
C AMD steam	 ?		 ?		 -		 ?
C AMD bobcat	n/a		n/a		 -		n/a
C AMD jaguar	 ?		 ?		 -		 ?
C Intel P4	n/a		n/a		 -		n/a
C Intel core	n/a		n/a		 -		n/a
C Intel NHM	n/a		n/a		 -		n/a
C Intel SBR	n/a		n/a		 -		n/a
C Intel IBR	n/a		n/a		 -		n/a
C Intel HWL	 1.77		 1.86		 -		 2.15
C Intel BWL	 ?		 ?		 -		 ?
C Intel atom	n/a		n/a		 -		n/a
C VIA nano	n/a		n/a		 -		n/a

C The inner loops of this code are the result of running a code generation and
C optimisation tool suite written by David Harvey and Torbjörn Granlund.

C TODO
C  * Adjoin a mul_3.
C  * Further micro-optimise.

define(`rp',      `%rdi')
define(`up',      `%rsi')
define(`un_param',`%rdx')
define(`vp',      `%rcx')
define(`vn',      `%r8')

define(`un',      `%rbx')

define(`w0',	`%r10')
define(`w1',	`%r11')
define(`w2',	`%r12')
define(`w3',	`%r13')
define(`n',	`%rbp')
define(`v0',	`%r9')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_mul_basecase)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8d	')
	push	%rbx
	push	%rbp
	push	%r12
	push	%r13
	push	%r14
	mov	un_param, un		C free up rdx
	neg	un

	mov	un_param, n		C FIXME: share
	sar	$2, n			C FIXME: share

	test	$1, R8(vn)
	jz	L(do_mul_2)

define(`w4',	`%r9')
define(`w5',	`%r14')

	mov	(vp), %rdx

L(do_mul_1):
	test	$1, R8(un)
	jnz	L(m1x1)

L(m1x0):test	$2, R8(un)
	jnz	L(m110)

L(m100):
	mulx(	(up), w5, w2)
	mulx(	8,(up), w1, w3)
	lea	-24(rp), rp
	jmp	L(m1l0)

L(m110):
	mulx(	(up), w3, w4)
	mulx(	8,(up), w1, w5)
	lea	-8(rp), rp
	test	n, n
	jz	L(cj2)
	mulx(	16,(up), w0, w2)
	lea	16(up), up
	jmp	L(m1l2)

L(m1x1):test	$2, R8(un)
	jz	L(m111)

L(m101):
	mulx(	(up), w4, w5)
	lea	-16(rp), rp
	test	n, n
	jz	L(cj1)
	mulx(	8,(up), w0, w2)
	lea	8(up), up
	jmp	L(m1l1)

L(m111):
	mulx(	(up), w2, w3)
	mulx(	8,(up), w0, w4)
	mulx(	16,(up), w1, w5)
	lea	24(up), up
	test	n, n
	jnz	L(gt3)
	add	w0, w3
	jmp	L(cj3)
L(gt3):	add	w0, w3
	jmp	L(m1l3)

	ALIGN(32)
L(m1tp):lea	32(rp), rp
L(m1l3):mov	w2, (rp)
	mulx(	(up), w0, w2)
L(m1l2):mov	w3, 8(rp)
	adc	w1, w4
L(m1l1):adc	w0, w5
	mov	w4, 16(rp)
	mulx(	8,(up), w1, w3)
L(m1l0):mov	w5, 24(rp)
	mulx(	16,(up), w0, w4)
	adc	w1, w2
	mulx(	24,(up), w1, w5)
	adc	w0, w3
	lea	32(up), up
	dec	n
	jnz	L(m1tp)

L(m1ed):lea	32(rp), rp
L(cj3):	mov	w2, (rp)
L(cj2):	mov	w3, 8(rp)
	adc	w1, w4
L(cj1):	mov	w4, 16(rp)
	adc	$0, w5
	mov	w5, 24(rp)

	dec	R32(vn)
	jz	L(ret5)

	lea	8(vp), vp
	lea	32(rp), rp
C	push	%r12
C	push	%r13
C	push	%r14
	jmp	L(do_addmul)

L(do_mul_2):
define(`v1',	`%r14')
C	push	%r12
C	push	%r13
C	push	%r14

	mov	(vp), v0
	mov	8(vp), v1

	lea	(un), n
	sar	$2, n

	test	$1, R8(un)
	jnz	L(m2x1)

L(m2x0):xor	w0, w0
	test	$2, R8(un)
	mov	(up), %rdx
	mulx(	v0, w2, w1)
	jz	L(m2l0)

L(m210):lea	-16(rp), rp
	lea	-16(up), up
	jmp	L(m2l2)

L(m2x1):xor	w2, w2
	test	$2, R8(un)
	mov	(up), %rdx
	mulx(	v0, w0, w3)
	jz	L(m211)

L(m201):lea	-24(rp), rp
	lea	8(up), up
	jmp	L(m2l1)

L(m211):lea	-8(rp), rp
	lea	-8(up), up
	jmp	L(m2l3)

	ALIGN(16)
L(m2tp):mulx(	v1, %rax, w0)
	add	%rax, w2
	mov	(up), %rdx
	mulx(	v0, %rax, w1)
	adc	$0, w0
	add	%rax, w2
	adc	$0, w1
	add	w3, w2
L(m2l0):mov	w2, (rp)
	adc	$0, w1
	mulx(	v1, %rax, w2)
	add	%rax, w0
	mov	8(up), %rdx
	adc	$0, w2
	mulx(	v0, %rax, w3)
	add	%rax, w0
	adc	$0, w3
	add	w1, w0
L(m2l3):mov	w0, 8(rp)
	adc	$0, w3
	mulx(	v1, %rax, w0)
	add	%rax, w2
	mov	16(up), %rdx
	mulx(	v0, %rax, w1)
	adc	$0, w0
	add	%rax, w2
	adc	$0, w1
	add	w3, w2
L(m2l2):mov	w2, 16(rp)
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
L(m2l1):mov	w0, 24(rp)
	adc	$0, w3
	inc	n
	lea	32(rp), rp
	jnz	L(m2tp)

L(m2ed):mulx(	v1, %rdx, %rax)
	add	%rdx, w2
	adc	$0, %rax
	add	w3, w2
	mov	w2, (rp)
	adc	$0, %rax
	mov	%rax, 8(rp)

	add	$-2, R32(vn)
	jz	L(ret5)
	lea	16(vp), vp
	lea	16(rp), rp


L(do_addmul):
	push	%r15
	push	vn			C save vn in new stack slot
define(`vn',	`(%rsp)')
define(`X0',	`%r14')
define(`X1',	`%r15')
define(`v1',	`%r8')

	lea	(rp,un,8), rp
	lea	(up,un,8), up

L(outer):
	mov	(vp), v0
	mov	8(vp), v1

	lea	2(un), n
	sar	$2, n

	mov	(up), %rdx
	test	$1, R8(un)
	jnz	L(bx1)

L(bx0):	mov	(rp), X0
	mov	8(rp), X1
	mulx(	v0, %rax, w1)
	add	%rax, X0
	mulx(	v1, %rax, w2)
	adc	$0, w1
	mov	X0, (rp)
	add	%rax, X1
	adc	$0, w2
	mov	8(up), %rdx
	test	$2, R8(un)
	jnz	L(b10)

L(b00):	lea	16(up), up
	lea	16(rp), rp
	jmp	L(lo0)

L(b10):	mov	16(rp), X0
	lea	32(up), up
	mulx(	v0, %rax, w3)
	jmp	L(lo2)

L(bx1):	mov	(rp), X1
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
	test	$2, R8(un)
	jz	L(b11)

L(b01):	mov	16(rp), X1
	lea	24(rp), rp
	lea	24(up), up
	jmp	L(lo1)

L(b11):	lea	8(rp), rp
	lea	8(up), up
	jmp	L(lo3)

	ALIGN(16)
L(top):	mulx(	v0, %rax, w3)
	add	w0, X1
	adc	$0, w2
L(lo2):	add	%rax, X1
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
	mov	8(up), %rdx
	lea	32(up), up
	inc	n
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
	mov	%rax, 24(rp)

	addl	$-2, vn
	lea	16(vp), vp
	lea	-16(up,un,8), up
	lea	32(rp,un,8), rp
	jnz	L(outer)

	pop	%rax		C deallocate vn slot
	pop	%r15
L(ret5):pop	%r14
L(ret4):pop	%r13
L(ret3):pop	%r12
L(ret2):pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
