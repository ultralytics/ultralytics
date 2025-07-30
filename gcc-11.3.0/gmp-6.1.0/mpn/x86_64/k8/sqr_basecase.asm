dnl  AMD64 mpn_sqr_basecase.

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2008, 2009, 2011, 2012 Free Software Foundation, Inc.

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

C The inner loops of this code are the result of running a code generation and
C optimization tool suite written by David Harvey and Torbjorn Granlund.

C NOTES
C   * There is a major stupidity in that we call mpn_mul_1 initially, for a
C     large trip count.  Instead, we should follow the generic/sqr_basecase.c
C     code which uses addmul_2s from the start, conditionally leaving a 1x1
C     multiply to the end.  (In assembly code, one would stop invoking
C     addmul_2s loops when perhaps 3x2s respectively a 2x2s remains.)
C   * Another stupidity is in the sqr_diag_addlsh1 code.  It does not need to
C     save/restore carry, instead it can propagate into the high product word.
C   * Align more labels, should shave off a few cycles.
C   * We can safely use 32-bit size operations, since operands with (2^32)
C     limbs will lead to non-termination in practice.
C   * The jump table could probably be optimized, at least for non-pic.
C   * The special code for n <= 4 was quickly written.  It is probably too
C     large and unnecessarily slow.
C   * Consider combining small cases code so that the n=k-1 code jumps into the
C     middle of the n=k code.
C   * Avoid saving registers for small cases code.
C   * Needed variables:
C    n   r11  input size
C    i   r8   work left, initially n
C    j   r9   inner loop count
C        r15  unused
C    v0  r13
C    v1  r14
C    rp  rdi
C    up  rsi
C    w0  rbx
C    w1  rcx
C    w2  rbp
C    w3  r10
C    tp  r12
C    lo  rax
C    hi  rdx
C        rsp

C INPUT PARAMETERS
define(`rp',	  `%rdi')
define(`up',	  `%rsi')
define(`n_param', `%rdx')

define(`n',	`%r11')
define(`tp',	`%r12')
define(`i',	`%r8')
define(`j',	`%r9')
define(`v0',	`%r13')
define(`v1',	`%r14')
define(`w0',	`%rbx')
define(`w1',	`%rcx')
define(`w2',	`%rbp')
define(`w3',	`%r10')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_sqr_basecase)
	FUNC_ENTRY(3)
	mov	R32(n_param), R32(%rcx)
	mov	R32(n_param), R32(n)		C free original n register (rdx)

	add	$-40, %rsp

	and	$3, R32(%rcx)
	cmp	$4, R32(n_param)
	lea	4(%rcx), %r8

	mov	%rbx, 32(%rsp)
	mov	%rbp, 24(%rsp)
	mov	%r12, 16(%rsp)
	mov	%r13, 8(%rsp)
	mov	%r14, (%rsp)

	cmovg	%r8, %rcx

	lea	L(tab)(%rip), %rax
ifdef(`PIC',
`	movslq	(%rax,%rcx,4), %r10
	add	%r10, %rax
	jmp	*%rax
',`
	jmp	*(%rax,%rcx,8)
')
	JUMPTABSECT
	ALIGN(8)
L(tab):	JMPENT(	L(4), L(tab))
	JMPENT(	L(1), L(tab))
	JMPENT(	L(2), L(tab))
	JMPENT(	L(3), L(tab))
	JMPENT(	L(0m4), L(tab))
	JMPENT(	L(1m4), L(tab))
	JMPENT(	L(2m4), L(tab))
	JMPENT(	L(3m4), L(tab))
	TEXT

L(1):	mov	(up), %rax
	mul	%rax
	add	$40, %rsp
	mov	%rax, (rp)
	mov	%rdx, 8(rp)
	FUNC_EXIT()
	ret

L(2):	mov	(up), %rax
	mov	%rax, %r8
	mul	%rax
	mov	8(up), %r11
	mov	%rax, (rp)
	mov	%r11, %rax
	mov	%rdx, %r9
	mul	%rax
	add	$40, %rsp
	mov	%rax, %r10
	mov	%r11, %rax
	mov	%rdx, %r11
	mul	%r8
	xor	%r8, %r8
	add	%rax, %r9
	adc	%rdx, %r10
	adc	%r8, %r11
	add	%rax, %r9
	mov	%r9, 8(rp)
	adc	%rdx, %r10
	mov	%r10, 16(rp)
	adc	%r8, %r11
	mov	%r11, 24(rp)
	FUNC_EXIT()
	ret

L(3):	mov	(up), %rax
	mov	%rax, %r10
	mul	%rax
	mov	8(up), %r11
	mov	%rax, (rp)
	mov	%r11, %rax
	mov	%rdx, 8(rp)
	mul	%rax
	mov	16(up), %rcx
	mov	%rax, 16(rp)
	mov	%rcx, %rax
	mov	%rdx, 24(rp)
	mul	%rax
	mov	%rax, 32(rp)
	mov	%rdx, 40(rp)

	mov	%r11, %rax
	mul	%r10
	mov	%rax, %r8
	mov	%rcx, %rax
	mov	%rdx, %r9
	mul	%r10
	xor	%r10, %r10
	add	%rax, %r9
	mov	%r11, %rax
	mov	%r10, %r11
	adc	%rdx, %r10

	mul	%rcx
	add	$40, %rsp
	add	%rax, %r10
	adc	%r11, %rdx
	add	%r8, %r8
	adc	%r9, %r9
	adc	%r10, %r10
	adc	%rdx, %rdx
	adc	%r11, %r11
	add	%r8, 8(rp)
	adc	%r9, 16(rp)
	adc	%r10, 24(rp)
	adc	%rdx, 32(rp)
	adc	%r11, 40(rp)
	FUNC_EXIT()
	ret

L(4):	mov	(up), %rax
	mov	%rax, %r11
	mul	%rax
	mov	8(up), %rbx
	mov	%rax, (rp)
	mov	%rbx, %rax
	mov	%rdx, 8(rp)
	mul	%rax
	mov	%rax, 16(rp)
	mov	%rdx, 24(rp)
	mov	16(up), %rax
	mul	%rax
	mov	%rax, 32(rp)
	mov	%rdx, 40(rp)
	mov	24(up), %rax
	mul	%rax
	mov	%rax, 48(rp)
	mov	%rbx, %rax
	mov	%rdx, 56(rp)

	mul	%r11
	add	$32, %rsp
	mov	%rax, %r8
	mov	%rdx, %r9
	mov	16(up), %rax
	mul	%r11
	xor	%r10, %r10
	add	%rax, %r9
	adc	%rdx, %r10
	mov	24(up), %rax
	mul	%r11
	xor	%r11, %r11
	add	%rax, %r10
	adc	%rdx, %r11
	mov	16(up), %rax
	mul	%rbx
	xor	%rcx, %rcx
	add	%rax, %r10
	adc	%rdx, %r11
	adc	$0, %rcx
	mov	24(up), %rax
	mul	%rbx
	pop	%rbx
	add	%rax, %r11
	adc	%rdx, %rcx
	mov	16(up), %rdx
	mov	24(up), %rax
	mul	%rdx
	add	%rax, %rcx
	adc	$0, %rdx

	add	%r8, %r8
	adc	%r9, %r9
	adc	%r10, %r10
	adc	%r11, %r11
	adc	%rcx, %rcx
	mov	$0, R32(%rax)
	adc	%rdx, %rdx

	adc	%rax, %rax
	add	%r8, 8(rp)
	adc	%r9, 16(rp)
	adc	%r10, 24(rp)
	adc	%r11, 32(rp)
	adc	%rcx, 40(rp)
	adc	%rdx, 48(rp)
	adc	%rax, 56(rp)
	FUNC_EXIT()
	ret


L(0m4):
	lea	-16(rp,n,8), tp		C point tp in middle of result operand
	mov	(up), v0
	mov	8(up), %rax
	lea	(up,n,8), up		C point up at end of input operand

	lea	-4(n), i
C Function mpn_mul_1_m3(tp, up - i, i, up[-i - 1])
	xor	R32(j), R32(j)
	sub	n, j

	mul	v0
	xor	R32(w2), R32(w2)
	mov	%rax, w0
	mov	16(up,j,8), %rax
	mov	%rdx, w3
	jmp	L(L3)

	ALIGN(16)
L(mul_1_m3_top):
	add	%rax, w2
	mov	w3, (tp,j,8)
	mov	(up,j,8), %rax
	adc	%rdx, w1
	xor	R32(w0), R32(w0)
	mul	v0
	xor	R32(w3), R32(w3)
	mov	w2, 8(tp,j,8)
	add	%rax, w1
	adc	%rdx, w0
	mov	8(up,j,8), %rax
	mov	w1, 16(tp,j,8)
	xor	R32(w2), R32(w2)
	mul	v0
	add	%rax, w0
	mov	16(up,j,8), %rax
	adc	%rdx, w3
L(L3):	xor	R32(w1), R32(w1)
	mul	v0
	add	%rax, w3
	mov	24(up,j,8), %rax
	adc	%rdx, w2
	mov	w0, 24(tp,j,8)
	mul	v0
	add	$4, j
	js	L(mul_1_m3_top)

	add	%rax, w2
	mov	w3, (tp)
	adc	%rdx, w1
	mov	w2, 8(tp)
	mov	w1, 16(tp)

	lea	eval(2*8)(tp), tp	C tp += 2
	lea	-8(up), up
	jmp	L(dowhile)


L(1m4):
	lea	8(rp,n,8), tp		C point tp in middle of result operand
	mov	(up), v0		C u0
	mov	8(up), %rax		C u1
	lea	8(up,n,8), up		C point up at end of input operand

	lea	-3(n), i
C Function mpn_mul_2s_m0(tp, up - i, i, up - i - 1)
	lea	-3(n), j
	neg	j

	mov	%rax, v1		C u1
	mul	v0			C u0 * u1
	mov	%rdx, w1
	xor	R32(w2), R32(w2)
	mov	%rax, 8(rp)
	jmp	L(m0)

	ALIGN(16)
L(mul_2_m0_top):
	mul	v1
	add	%rax, w0
	adc	%rdx, w1
	mov	-24(up,j,8), %rax
	mov	$0, R32(w2)
	mul	v0
	add	%rax, w0
	mov	-24(up,j,8), %rax
	adc	%rdx, w1
	adc	$0, R32(w2)
	mul	v1			C v1 * u0
	add	%rax, w1
	mov	w0, -24(tp,j,8)
	adc	%rdx, w2
L(m0):	mov	-16(up,j,8), %rax	C u2, u6 ...
	mul	v0			C u0 * u2
	mov	$0, R32(w3)
	add	%rax, w1
	adc	%rdx, w2
	mov	-16(up,j,8), %rax
	adc	$0, R32(w3)
	mov	$0, R32(w0)
	mov	w1, -16(tp,j,8)
	mul	v1
	add	%rax, w2
	mov	-8(up,j,8), %rax
	adc	%rdx, w3
	mov	$0, R32(w1)
	mul	v0
	add	%rax, w2
	mov	-8(up,j,8), %rax
	adc	%rdx, w3
	adc	$0, R32(w0)
	mul	v1
	add	%rax, w3
	mov	w2, -8(tp,j,8)
	adc	%rdx, w0
L(m2x):	mov	(up,j,8), %rax
	mul	v0
	add	%rax, w3
	adc	%rdx, w0
	adc	$0, R32(w1)
	add	$4, j
	mov	-32(up,j,8), %rax
	mov	w3, -32(tp,j,8)
	js	L(mul_2_m0_top)

	mul	v1
	add	%rax, w0
	adc	%rdx, w1
	mov	w0, -8(tp)
	mov	w1, (tp)

	lea	-16(up), up
	lea	eval(3*8-24)(tp), tp	C tp += 3
	jmp	L(dowhile_end)


L(2m4):
	lea	-16(rp,n,8), tp		C point tp in middle of result operand
	mov	(up), v0
	mov	8(up), %rax
	lea	(up,n,8), up		C point up at end of input operand

	lea	-4(n), i
C Function mpn_mul_1_m1(tp, up - (i - 1), i - 1, up[-i])
	lea	-2(n), j
	neg	j

	mul	v0
	mov	%rax, w2
	mov	(up,j,8), %rax
	mov	%rdx, w1
	jmp	L(L1)

	ALIGN(16)
L(mul_1_m1_top):
	add	%rax, w2
	mov	w3, (tp,j,8)
	mov	(up,j,8), %rax
	adc	%rdx, w1
L(L1):	xor	R32(w0), R32(w0)
	mul	v0
	xor	R32(w3), R32(w3)
	mov	w2, 8(tp,j,8)
	add	%rax, w1
	adc	%rdx, w0
	mov	8(up,j,8), %rax
	mov	w1, 16(tp,j,8)
	xor	R32(w2), R32(w2)
	mul	v0
	add	%rax, w0
	mov	16(up,j,8), %rax
	adc	%rdx, w3
	xor	R32(w1), R32(w1)
	mul	v0
	add	%rax, w3
	mov	24(up,j,8), %rax
	adc	%rdx, w2
	mov	w0, 24(tp,j,8)
	mul	v0
	add	$4, j
	js	L(mul_1_m1_top)

	add	%rax, w2
	mov	w3, (tp)
	adc	%rdx, w1
	mov	w2, 8(tp)
	mov	w1, 16(tp)

	lea	eval(2*8)(tp), tp	C tp += 2
	lea	-8(up), up
	jmp	L(dowhile_mid)


L(3m4):
	lea	8(rp,n,8), tp		C point tp in middle of result operand
	mov	(up), v0		C u0
	mov	8(up), %rax		C u1
	lea	8(up,n,8), up		C point up at end of input operand

	lea	-5(n), i
C Function mpn_mul_2s_m2(tp, up - i + 1, i - 1, up - i)
	lea	-1(n), j
	neg	j

	mov	%rax, v1		C u1
	mul	v0			C u0 * u1
	mov	%rdx, w3
	xor	R32(w0), R32(w0)
	xor	R32(w1), R32(w1)
	mov	%rax, 8(rp)
	jmp	L(m2)

	ALIGN(16)
L(mul_2_m2_top):
	mul	v1
	add	%rax, w0
	adc	%rdx, w1
	mov	-24(up,j,8), %rax
	mov	$0, R32(w2)
	mul	v0
	add	%rax, w0
	mov	-24(up,j,8), %rax
	adc	%rdx, w1
	adc	$0, R32(w2)
	mul	v1			C v1 * u0
	add	%rax, w1
	mov	w0, -24(tp,j,8)
	adc	%rdx, w2
	mov	-16(up,j,8), %rax
	mul	v0
	mov	$0, R32(w3)
	add	%rax, w1
	adc	%rdx, w2
	mov	-16(up,j,8), %rax
	adc	$0, R32(w3)
	mov	$0, R32(w0)
	mov	w1, -16(tp,j,8)
	mul	v1
	add	%rax, w2
	mov	-8(up,j,8), %rax
	adc	%rdx, w3
	mov	$0, R32(w1)
	mul	v0
	add	%rax, w2
	mov	-8(up,j,8), %rax
	adc	%rdx, w3
	adc	$0, R32(w0)
	mul	v1
	add	%rax, w3
	mov	w2, -8(tp,j,8)
	adc	%rdx, w0
L(m2):	mov	(up,j,8), %rax
	mul	v0
	add	%rax, w3
	adc	%rdx, w0
	adc	$0, R32(w1)
	add	$4, j
	mov	-32(up,j,8), %rax
	mov	w3, -32(tp,j,8)
	js	L(mul_2_m2_top)

	mul	v1
	add	%rax, w0
	adc	%rdx, w1
	mov	w0, -8(tp)
	mov	w1, (tp)

	lea	-16(up), up
	jmp	L(dowhile_mid)

L(dowhile):
C Function mpn_addmul_2s_m2(tp, up - (i - 1), i - 1, up - i)
	lea	4(i), j
	neg	j

	mov	16(up,j,8), v0
	mov	24(up,j,8), v1
	mov	24(up,j,8), %rax
	mul	v0
	xor	R32(w3), R32(w3)
	add	%rax, 24(tp,j,8)
	adc	%rdx, w3
	xor	R32(w0), R32(w0)
	xor	R32(w1), R32(w1)
	jmp	L(am2)

	ALIGN(16)
L(addmul_2_m2_top):
	add	w3, (tp,j,8)
	adc	%rax, w0
	mov	8(up,j,8), %rax
	adc	%rdx, w1
	mov	$0, R32(w2)
	mul	v0
	add	%rax, w0
	mov	8(up,j,8), %rax
	adc	%rdx, w1
	adc	$0, R32(w2)
	mul	v1				C v1 * u0
	add	w0, 8(tp,j,8)
	adc	%rax, w1
	adc	%rdx, w2
	mov	16(up,j,8), %rax
	mov	$0, R32(w3)
	mul	v0				C v0 * u1
	add	%rax, w1
	mov	16(up,j,8), %rax
	adc	%rdx, w2
	adc	$0, R32(w3)
	mul	v1				C v1 * u1
	add	w1, 16(tp,j,8)
	adc	%rax, w2
	mov	24(up,j,8), %rax
	adc	%rdx, w3
	mul	v0
	mov	$0, R32(w0)
	add	%rax, w2
	adc	%rdx, w3
	mov	$0, R32(w1)
	mov	24(up,j,8), %rax
	adc	$0, R32(w0)
	mul	v1
	add	w2, 24(tp,j,8)
	adc	%rax, w3
	adc	%rdx, w0
L(am2):	mov	32(up,j,8), %rax
	mul	v0
	add	%rax, w3
	mov	32(up,j,8), %rax
	adc	%rdx, w0
	adc	$0, R32(w1)
	mul	v1
	add	$4, j
	js	L(addmul_2_m2_top)

	add	w3, (tp)
	adc	%rax, w0
	adc	%rdx, w1
	mov	w0, 8(tp)
	mov	w1, 16(tp)

	lea	eval(2*8)(tp), tp	C tp += 2

	add	$-2, R32(i)		C i -= 2

L(dowhile_mid):
C Function mpn_addmul_2s_m0(tp, up - (i - 1), i - 1, up - i)
	lea	2(i), j
	neg	j

	mov	(up,j,8), v0
	mov	8(up,j,8), v1
	mov	8(up,j,8), %rax
	mul	v0
	xor	R32(w1), R32(w1)
	add	%rax, 8(tp,j,8)
	adc	%rdx, w1
	xor	R32(w2), R32(w2)
	jmp	L(20)

	ALIGN(16)
L(addmul_2_m0_top):
	add	w3, (tp,j,8)
	adc	%rax, w0
	mov	8(up,j,8), %rax
	adc	%rdx, w1
	mov	$0, R32(w2)
	mul	v0
	add	%rax, w0
	mov	8(up,j,8), %rax
	adc	%rdx, w1
	adc	$0, R32(w2)
	mul	v1				C v1 * u0
	add	w0, 8(tp,j,8)
	adc	%rax, w1
	adc	%rdx, w2
L(20):	mov	16(up,j,8), %rax
	mov	$0, R32(w3)
	mul	v0				C v0 * u1
	add	%rax, w1
	mov	16(up,j,8), %rax
	adc	%rdx, w2
	adc	$0, R32(w3)
	mul	v1				C v1 * u1
	add	w1, 16(tp,j,8)
	adc	%rax, w2
	mov	24(up,j,8), %rax
	adc	%rdx, w3
	mul	v0
	mov	$0, R32(w0)
	add	%rax, w2
	adc	%rdx, w3
	mov	$0, R32(w1)
	mov	24(up,j,8), %rax
	adc	$0, R32(w0)
	mul	v1
	add	w2, 24(tp,j,8)
	adc	%rax, w3
	adc	%rdx, w0
	mov	32(up,j,8), %rax
	mul	v0
	add	%rax, w3
	mov	32(up,j,8), %rax
	adc	%rdx, w0
	adc	$0, R32(w1)
	mul	v1
	add	$4, j
	js	L(addmul_2_m0_top)

	add	w3, (tp)
	adc	%rax, w0
	adc	%rdx, w1
	mov	w0, 8(tp)
	mov	w1, 16(tp)

	lea	eval(2*8)(tp), tp	C tp += 2
L(dowhile_end):

	add	$-2, R32(i)		C i -= 2
	jne	L(dowhile)

C Function mpn_addmul_2s_2
	mov	-16(up), v0
	mov	-8(up), v1
	mov	-8(up), %rax
	mul	v0
	xor	R32(w3), R32(w3)
	add	%rax, -8(tp)
	adc	%rdx, w3
	xor	R32(w0), R32(w0)
	xor	R32(w1), R32(w1)
	mov	(up), %rax
	mul	v0
	add	%rax, w3
	mov	(up), %rax
	adc	%rdx, w0
	mul	v1
	add	w3, (tp)
	adc	%rax, w0
	adc	%rdx, w1
	mov	w0, 8(tp)
	mov	w1, 16(tp)

C Function mpn_sqr_diag_addlsh1
	lea	-4(n,n), j

	mov	8(rp), %r11
	lea	-8(up), up
	lea	(rp,j,8), rp
	neg	j
	mov	(up,j,4), %rax
	mul	%rax
	test	$2, R8(j)
	jnz	L(odd)

L(evn):	add	%r11, %r11
	sbb	R32(%rbx), R32(%rbx)		C save CF
	add	%rdx, %r11
	mov	%rax, (rp,j,8)
	jmp	L(d0)

L(odd):	add	%r11, %r11
	sbb	R32(%rbp), R32(%rbp)		C save CF
	add	%rdx, %r11
	mov	%rax, (rp,j,8)
	lea	-2(j), j
	jmp	L(d1)

	ALIGN(16)
L(top):	mov	(up,j,4), %rax
	mul	%rax
	add	R32(%rbp), R32(%rbp)		C restore carry
	adc	%rax, %r10
	adc	%rdx, %r11
	mov	%r10, (rp,j,8)
L(d0):	mov	%r11, 8(rp,j,8)
	mov	16(rp,j,8), %r10
	adc	%r10, %r10
	mov	24(rp,j,8), %r11
	adc	%r11, %r11
	nop
	sbb	R32(%rbp), R32(%rbp)		C save CF
	mov	8(up,j,4), %rax
	mul	%rax
	add	R32(%rbx), R32(%rbx)		C restore carry
	adc	%rax, %r10
	adc	%rdx, %r11
	mov	%r10, 16(rp,j,8)
L(d1):	mov	%r11, 24(rp,j,8)
	mov	32(rp,j,8), %r10
	adc	%r10, %r10
	mov	40(rp,j,8), %r11
	adc	%r11, %r11
	sbb	R32(%rbx), R32(%rbx)		C save CF
	add	$4, j
	js	L(top)

	mov	(up), %rax
	mul	%rax
	add	R32(%rbp), R32(%rbp)		C restore carry
	adc	%rax, %r10
	adc	%rdx, %r11
	mov	%r10, (rp)
	mov	%r11, 8(rp)
	mov	16(rp), %r10
	adc	%r10, %r10
	sbb	R32(%rbp), R32(%rbp)		C save CF
	neg	R32(%rbp)
	mov	8(up), %rax
	mul	%rax
	add	R32(%rbx), R32(%rbx)		C restore carry
	adc	%rax, %r10
	adc	%rbp, %rdx
	mov	%r10, 16(rp)
	mov	%rdx, 24(rp)

	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
