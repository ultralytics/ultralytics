dnl  AMD64 mpn_mul_basecase optimised for Intel Sandy bridge and Ivy bridge.

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
C Intel SBR	 2.5		 2.5		 -		 2.95
C Intel IBR	 2.4		 2.3		 -		 2.68
C Intel HWL	 2.35		 2.0		 -		 2.5
C Intel BWL
C Intel atom
C VIA nano

C The inner loops of this code are the result of running a code generation and
C optimisation tool suite written by David Harvey and Torbjorn Granlund.

C TODO
C  * Fix the addmul_2 fluctuation affecting SBR.
C  * Improve feed-in code, avoiding zeroing of many registers and dummy adds in
C    the loops at the expense of code size.
C  * Adjoin a mul_3, avoiding slow mul_1 for odd vn.
C  * Consider replacing the 2-way mul_2 code with 4-way code, for a very slight
C    speedup.
C  * Further micro-optimise.

C When playing with pointers, set this to $2 to fall back to conservative
C indexing in wind-down code.
define(`I',`$1')


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
	mov	un_param, un		C free up rdx
	neg	un

	mov	(up), %rax		C shared for mul_1 and mul_2
	lea	(up,un_param,8), up	C point at operand end
	lea	(rp,un_param,8), rp	C point at rp[un-1]

	mov	(vp), v0		C shared for mul_1 and mul_2
	mul	v0			C shared for mul_1 and mul_2

	test	$1, R8(vn)
	jz	L(do_mul_2)

L(do_mul_1):
	test	$1, R8(un)
	jnz	L(m1x1)

L(m1x0):mov	%rax, w0		C un = 2, 4, 6, 8, ...
	mov	%rdx, w1
	mov	8(up,un,8), %rax
	test	$2, R8(un)
	jnz	L(m110)

L(m100):lea	2(un), n		C un = 4, 8, 12, ...
	jmp	L(m1l0)

L(m110):lea	(un), n			C un = 2, 6, 10, ...
	jmp	L(m1l2)

L(m1x1):mov	%rax, w1		C un = 1, 3, 5, 7, ...
	mov	%rdx, w0
	test	$2, R8(un)
	jz	L(m111)

L(m101):lea	3(un), n		C un = 1, 5, 9, ...
	test	n, n
	js	L(m1l1)
	mov	%rax, -8(rp)
	mov	%rdx, (rp)
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret

L(m111):lea	1(un), n		C un = 3, 7, 11, ...
	mov	8(up,un,8), %rax
	jmp	L(m1l3)

	ALIGN(16)		C FIXME
L(m1tp):mov	%rdx, w0
	add	%rax, w1
L(m1l1):mov	-16(up,n,8), %rax
	adc	$0, w0
	mul	v0
	add	%rax, w0
	mov	w1, -24(rp,n,8)
	mov	-8(up,n,8), %rax
	mov	%rdx, w1
	adc	$0, w1
L(m1l0):mul	v0
	mov	w0, -16(rp,n,8)
	add	%rax, w1
	mov	%rdx, w0
	mov	(up,n,8), %rax
	adc	$0, w0
L(m1l3):mul	v0
	mov	w1, -8(rp,n,8)
	mov	%rdx, w1
	add	%rax, w0
	mov	8(up,n,8), %rax
	adc	$0, w1
L(m1l2):mul	v0
	mov	w0, (rp,n,8)
	add	$4, n
	jnc	L(m1tp)

L(m1ed):add	%rax, w1
	adc	$0, %rdx
	mov	w1, I(-8(rp),-24(rp,n,8))
	mov	%rdx, I((rp),-16(rp,n,8))

	dec	R32(vn)
	jz	L(ret2)

	lea	8(vp), vp
	lea	8(rp), rp
	push	%r12
	push	%r13
	push	%r14
	jmp	L(do_addmul)

L(do_mul_2):
define(`v1',	`%r14')
	push	%r12
	push	%r13
	push	%r14

	mov	8(vp), v1

	test	$1, R8(un)
	jnz	L(m2b1)

L(m2b0):lea	(un), n
	xor	w0, w0
	mov	%rax, w2
	mov	%rdx, w1
	jmp	L(m2l0)

L(m2b1):lea	1(un), n
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
L(m2l1):mov	-8(up,n,8), %rax
	mul	v1
	add	w1, w0
	adc	$0, w3
	add	%rax, w2
	mov	w0, -8(rp,n,8)
	mov	%rdx, w0
	adc	$0, w0
	mov	(up,n,8), %rax
	mul	v0
	add	%rax, w2
	mov	%rdx, w1
	adc	$0, w1
	add	w3, w2
L(m2l0):mov	(up,n,8), %rax
	adc	$0, w1
	mul	v1
	mov	w2, (rp,n,8)
	add	%rax, w0
	mov	%rdx, w2
	mov	8(up,n,8), %rax
	adc	$0, w2
	add	$2, n
	jnc	L(m2tp)

L(m2ed):mul	v0
	add	%rax, w0
	mov	%rdx, w3
	adc	$0, w3
	mov	I(-8(up),-8(up,n,8)), %rax
	mul	v1
	add	w1, w0
	adc	$0, w3
	add	%rax, w2
	mov	w0, I(-8(rp),-8(rp,n,8))
	adc	$0, %rdx
	add	w3, w2
	mov	w2, I((rp),(rp,n,8))
	adc	$0, %rdx
	mov	%rdx, I(8(rp),8(rp,n,8))

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

L(outer):
	mov	(vp), v0
	mov	8(vp), v1
	mov	(up,un,8), %rax
	mul	v0
	test	$1, R8(un)
	jnz	L(a1x1)

L(a1x0):mov	(rp,un,8), X0
	xor	w0, w0
	mov	%rdx, w1
	test	$2, R8(un)
	jnz	L(a110)

L(a100):lea	2(un), n		C un = 4, 8, 12, ...
	add	%rax, X0
	adc	$0, w1
	mov	(up,un,8), %rax
	mul	v1
	mov	8(rp,un,8), X1
	jmp	L(lo0)

L(a110):lea	(un), n			C un = 2, 6, 10, ...
	xor	w3, w3
	jmp	L(lo2)

L(a1x1):mov	(rp,un,8), X1
	xor	w2, w2
	xor	w1, w1
	test	$2, R8(un)
	jz	L(a111)

L(a101):lea	3(un), n		C un = 1, 5, 9, ...
	mov	%rdx, w3
	add	%rax, X1
	mov	(up,un,8), %rax
	mov	8(rp,un,8), X0
	adc	$0, w3
	jmp	L(top)

L(a111):lea	1(un), n		C un = 3, 7, 11, ...
	jmp	L(lo3)

	ALIGN(32)
L(top):	mul	v1
	mov	%rdx, w0
	add	%rax, X0
	adc	$0, w0
	add	w1, X1
	adc	$0, w3
	add	w2, X0
	adc	$0, w0
	mov	-16(up,n,8), %rax
	mul	v0
	add	%rax, X0
	mov	%rdx, w1
	adc	$0, w1
	mov	-16(up,n,8), %rax
	mul	v1
	mov	X1, -24(rp,n,8)
	mov	-8(rp,n,8), X1
	add	w3, X0
	adc	$0, w1
L(lo0):	mov	%rdx, w2
	mov	X0, -16(rp,n,8)
	add	%rax, X1
	adc	$0, w2
	mov	-8(up,n,8), %rax
	add	w0, X1
	adc	$0, w2
	mul	v0
L(lo3):	add	%rax, X1
	mov	%rdx, w3
	adc	$0, w3
	mov	-8(up,n,8), %rax
	mul	v1
	add	w1, X1
	mov	(rp,n,8), X0
	adc	$0, w3
	mov	%rdx, w0
	add	%rax, X0
	adc	$0, w0
	mov	(up,n,8), %rax
	mul	v0
	add	w2, X0
	mov	X1, -8(rp,n,8)
	mov	%rdx, w1
	adc	$0, w0
L(lo2):	add	%rax, X0
	adc	$0, w1
	mov	(up,n,8), %rax
	add	w3, X0
	adc	$0, w1
	mul	v1
	mov	8(rp,n,8), X1
	add	%rax, X1
	mov	%rdx, w2
	adc	$0, w2
	mov	8(up,n,8), %rax
	mov	X0, (rp,n,8)
	mul	v0
	add	w0, X1
	mov	%rdx, w3
	adc	$0, w2
	add	%rax, X1
	mov	8(up,n,8), %rax
	mov	16(rp,n,8), X0		C useless but harmless in final iter
	adc	$0, w3
	add	$4, n
	jnc	L(top)

L(end):	mul	v1
	add	w1, X1
	adc	$0, w3
	add	w2, %rax
	adc	$0, %rdx
	mov	X1, I(-8(rp),-24(rp,n,8))
	add	w3, %rax
	adc	$0, %rdx
	mov	%rax, I((rp),-16(rp,n,8))
	mov	%rdx, I(8(rp),-8(rp,n,8))

	addl	$-2, vn
	lea	16(vp), vp
	lea	16(rp), rp
	jnz	L(outer)

	pop	%rax		C deallocate vn slot
	pop	%r15
L(ret5):pop	%r14
	pop	%r13
	pop	%r12
L(ret2):pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
