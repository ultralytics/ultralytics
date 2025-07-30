dnl  AMD64 mpn_redc_1 optimised for Intel Haswell.

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
C Intel HWL	 2.32
C Intel BWL	 ?
C Intel atom	n/a
C VIA nano	n/a

C The inner loops of this code are the result of running a code generation and
C optimisation tool suite written by David Harvey and Torbjörn Granlund.

C TODO
C  * Micro-optimise.
C  * Consider inlining mpn_add_n.  Tests indicate that this saves just 1-2
C    cycles, though.

define(`rp',          `%rdi')   C rcx
define(`up',          `%rsi')   C rdx
define(`mp_param',    `%rdx')   C r8
define(`n',           `%rcx')   C r9
define(`u0inv_param', `%r8')    C stack

define(`i',           `%r14')
define(`j',           `%r15')
define(`mp',          `%rdi')
define(`u0inv',       `(%rsp)')  C stack

ABI_SUPPORT(DOS64)    C FIXME: needs verification
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_redc_1)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8	')
	push	%rbx
	push	%rbp
	push	%r12
	push	%r13
	push	%r14
	push	%r15
	push	rp
	mov	mp_param, mp		C note that rp and mp shares register
	mov	(up), %rdx

	neg	n
	push	%r8			C put u0inv on stack
	imul	u0inv_param, %rdx	C first iteration q0
	mov	n, j			C outer loop induction var

	test	$1, R8(n)
	jnz	L(bx1)

L(bx0):	test	$2, R8(n)
	jz	L(o0b)

	cmp	$-2, R32(n)
	jnz	L(o2)

C Special code for n = 2 since general code cannot handle it
	mov	8(%rsp), %rbx		C rp
	lea	16(%rsp), %rsp		C deallocate two slots
	mulx(	(mp), %r9, %r12)
	mulx(	8,(mp), %r11, %r10)
	add	%r12, %r11
	adc	$0, %r10
	add	(up), %r9		C = 0
	adc	8(up), %r11		C r11 = up[1]
	adc	$0, %r10		C -> up[0]
	mov	%r11, %rdx
	imul	u0inv_param, %rdx
	mulx(	(mp), %r13, %r12)
	mulx(	8,(mp), %r14, %r15)
	xor	R32(%rax), R32(%rax)
	add	%r12, %r14
	adc	$0, %r15
	add	%r11, %r13		C = 0
	adc	16(up), %r14		C rp[2]
	adc	$0, %r15		C -> up[1]
	add	%r14, %r10
	adc	24(up), %r15
	mov	%r10, (%rbx)
	mov	%r15, 8(%rbx)
	setc	R8(%rax)
	jmp	L(ret)

L(o2):	lea	2(n), i			C inner loop induction var
	mulx(	(mp), %r9, %r8)
	mulx(	8,(mp), %r11, %r10)
	sar	$2, i
	add	%r8, %r11
	jmp	L(lo2)

	ALIGN(16)
L(tp2):	adc	%rax, %r9
	lea	32(up), up
	adc	%r8, %r11
L(lo2):	mulx(	16,(mp), %r13, %r12)
	mov	(up), %r8
	mulx(	24,(mp), %rbx, %rax)
	lea	32(mp), mp
	adc	%r10, %r13
	adc	%r12, %rbx
	adc	$0, %rax
	mov	8(up), %r10
	mov	16(up), %r12
	add	%r9, %r8
	mov	24(up), %rbp
	mov	%r8, (up)
	adc	%r11, %r10
	mulx(	(mp), %r9, %r8)
	mov	%r10, 8(up)
	adc	%r13, %r12
	mov	%r12, 16(up)
	adc	%rbx, %rbp
	mulx(	8,(mp), %r11, %r10)
	mov	%rbp, 24(up)
	inc	i
	jnz	L(tp2)

L(ed2):	mov	56(up,n,8), %rdx	C next iteration up[0]
	lea	16(mp,n,8), mp		C mp = (last starting mp)
	adc	%rax, %r9
	adc	%r8, %r11
	mov	32(up), %r8
	adc	$0, %r10
	imul	u0inv, %rdx		C next iteration q0
	mov	40(up), %rax
	add	%r9, %r8
	mov	%r8, 32(up)
	adc	%r11, %rax
	mov	%rax, 40(up)
	lea	56(up,n,8), up		C up = (last starting up) + 1
	adc	$0, %r10
	mov	%r10, -8(up)
	inc	j
	jnz	L(o2)

	jmp	L(cj)


L(bx1):	test	$2, R8(n)
	jz	L(o3a)

L(o1a):	cmp	$-1, R32(n)
	jnz	L(o1b)

C Special code for n = 1 since general code cannot handle it
	mov	8(%rsp), %rbx		C rp
	lea	16(%rsp), %rsp		C deallocate two slots
	mulx(	(mp), %r11, %r10)
	add	(up), %r11
	adc	8(up), %r10
	mov	%r10, (%rbx)
	mov	$0, R32(%rax)
	setc	R8(%rax)
	jmp	L(ret)

L(o1b):	lea	24(mp), mp
L(o1):	lea	1(n), i			C inner loop induction var
	mulx(	-24,(mp), %r11, %r10)
	mulx(	-16,(mp), %r13, %r12)
	mulx(	-8,(mp), %rbx, %rax)
	sar	$2, i
	add	%r10, %r13
	adc	%r12, %rbx
	adc	$0, %rax
	mov	(up), %r10
	mov	8(up), %r12
	mov	16(up), %rbp
	add	%r11, %r10
	jmp	L(lo1)

	ALIGN(16)
L(tp1):	adc	%rax, %r9
	lea	32(up), up
	adc	%r8, %r11
	mulx(	16,(mp), %r13, %r12)
	mov	-8(up), %r8
	mulx(	24,(mp), %rbx, %rax)
	lea	32(mp), mp
	adc	%r10, %r13
	adc	%r12, %rbx
	adc	$0, %rax
	mov	(up), %r10
	mov	8(up), %r12
	add	%r9, %r8
	mov	16(up), %rbp
	mov	%r8, -8(up)
	adc	%r11, %r10
L(lo1):	mulx(	(mp), %r9, %r8)
	mov	%r10, (up)
	adc	%r13, %r12
	mov	%r12, 8(up)
	adc	%rbx, %rbp
	mulx(	8,(mp), %r11, %r10)
	mov	%rbp, 16(up)
	inc	i
	jnz	L(tp1)

L(ed1):	mov	48(up,n,8), %rdx	C next iteration up[0]
	lea	40(mp,n,8), mp		C mp = (last starting mp)
	adc	%rax, %r9
	adc	%r8, %r11
	mov	24(up), %r8
	adc	$0, %r10
	imul	u0inv, %rdx		C next iteration q0
	mov	32(up), %rax
	add	%r9, %r8
	mov	%r8, 24(up)
	adc	%r11, %rax
	mov	%rax, 32(up)
	lea	48(up,n,8), up		C up = (last starting up) + 1
	adc	$0, %r10
	mov	%r10, -8(up)
	inc	j
	jnz	L(o1)

	jmp	L(cj)

L(o3a):	cmp	$-3, R32(n)
	jnz	L(o3b)

C Special code for n = 3 since general code cannot handle it
L(n3):	mulx(	(mp), %rbx, %rax)
	mulx(	8,(mp), %r9, %r14)
	add	(up), %rbx
	mulx(	16,(mp), %r11, %r10)
	adc	%rax, %r9		C W 1
	adc	%r14, %r11		C W 2
	mov	8(up), %r14
	mov	u0inv_param, %rdx
	adc	$0, %r10		C W 3
	mov	16(up), %rax
	add	%r9, %r14		C W 1
	mov	%r14, 8(up)
	mulx(	%r14, %rdx, %r13)	C next iteration q0
	adc	%r11, %rax		C W 2
	mov	%rax, 16(up)
	adc	$0, %r10		C W 3
	mov	%r10, (up)
	lea	8(up), up		C up = (last starting up) + 1
	inc	j
	jnz	L(n3)

	jmp	L(cj)

L(o3b):	lea	8(mp), mp
L(o3):	lea	4(n), i			C inner loop induction var
	mulx(	-8,(mp), %rbx, %rax)
	mulx(	(mp), %r9, %r8)
	mov	(up), %rbp
	mulx(	8,(mp), %r11, %r10)
	sar	$2, i
	add	%rbx, %rbp
	nop
	adc	%rax, %r9
	jmp	L(lo3)

	ALIGN(16)
L(tp3):	adc	%rax, %r9
	lea	32(up), up
L(lo3):	adc	%r8, %r11
	mulx(	16,(mp), %r13, %r12)
	mov	8(up), %r8
	mulx(	24,(mp), %rbx, %rax)
	lea	32(mp), mp
	adc	%r10, %r13
	adc	%r12, %rbx
	adc	$0, %rax
	mov	16(up), %r10
	mov	24(up), %r12
	add	%r9, %r8
	mov	32(up), %rbp
	mov	%r8, 8(up)
	adc	%r11, %r10
	mulx(	(mp), %r9, %r8)
	mov	%r10, 16(up)
	adc	%r13, %r12
	mov	%r12, 24(up)
	adc	%rbx, %rbp
	mulx(	8,(mp), %r11, %r10)
	mov	%rbp, 32(up)
	inc	i
	jnz	L(tp3)

L(ed3):	mov	64(up,n,8), %rdx	C next iteration up[0]
	lea	24(mp,n,8), mp		C mp = (last starting mp)
	adc	%rax, %r9
	adc	%r8, %r11
	mov	40(up), %r8
	adc	$0, %r10
	imul	u0inv, %rdx		C next iteration q0
	mov	48(up), %rax
	add	%r9, %r8
	mov	%r8, 40(up)
	adc	%r11, %rax
	mov	%rax, 48(up)
	lea	64(up,n,8), up		C up = (last starting up) + 1
	adc	$0, %r10
	mov	%r10, -8(up)
	inc	j
	jnz	L(o3)

	jmp	L(cj)

L(o0b):	lea	16(mp), mp
L(o0):	mov	n, i			C inner loop induction var
	mulx(	-16,(mp), %r13, %r12)
	mulx(	-8,(mp), %rbx, %rax)
	sar	$2, i
	add	%r12, %rbx
	adc	$0, %rax
	mov	(up), %r12
	mov	8(up), %rbp
	mulx(	(mp), %r9, %r8)
	add	%r13, %r12
	jmp	L(lo0)

	ALIGN(16)
L(tp0):	adc	%rax, %r9
	lea	32(up), up
	adc	%r8, %r11
	mulx(	16,(mp), %r13, %r12)
	mov	-16(up), %r8
	mulx(	24,(mp), %rbx, %rax)
	lea	32(mp), mp
	adc	%r10, %r13
	adc	%r12, %rbx
	adc	$0, %rax
	mov	-8(up), %r10
	mov	(up), %r12
	add	%r9, %r8
	mov	8(up), %rbp
	mov	%r8, -16(up)
	adc	%r11, %r10
	mulx(	(mp), %r9, %r8)
	mov	%r10, -8(up)
	adc	%r13, %r12
	mov	%r12, (up)
L(lo0):	adc	%rbx, %rbp
	mulx(	8,(mp), %r11, %r10)
	mov	%rbp, 8(up)
	inc	i
	jnz	L(tp0)

L(ed0):	mov	40(up,n,8), %rdx	C next iteration up[0]
	lea	32(mp,n,8), mp		C mp = (last starting mp)
	adc	%rax, %r9
	adc	%r8, %r11
	mov	16(up), %r8
	adc	$0, %r10
	imul	u0inv, %rdx		C next iteration q0
	mov	24(up), %rax
	add	%r9, %r8
	mov	%r8, 16(up)
	adc	%r11, %rax
	mov	%rax, 24(up)
	lea	40(up,n,8), up		C up = (last starting up) + 1
	adc	$0, %r10
	mov	%r10, -8(up)
	inc	j
	jnz	L(o0)

L(cj):
IFSTD(`	mov	8(%rsp), %rdi		C param 1: rp
	lea	16-8(%rsp), %rsp	C deallocate 2, add back for alignment
	lea	(up,n,8), %rdx		C param 3: up - n
	neg	R32(n)		')	C param 4: n

IFDOS(`	mov	up, %rdx		C param 2: up
	lea	(up,n,8), %r8		C param 3: up - n
	neg	R32(n)
	mov	n, %r9			C param 4: n
	mov	8(%rsp), %rcx		C param 1: rp
	lea	16-32-8(%rsp), %rsp')	C deallocate 2, allocate shadow, align

	ASSERT(nz, `test $15, %rsp')
	CALL(	mpn_add_n)

IFSTD(`	lea	8(%rsp), %rsp	')
IFDOS(`	lea	32+8(%rsp), %rsp')

L(ret):	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
