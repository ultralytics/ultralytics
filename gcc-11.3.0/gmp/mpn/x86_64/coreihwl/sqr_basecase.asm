dnl  AMD64 mpn_sqr_basecase optimised for Intel Haswell.

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

C cycles/limb	mul_2		addmul_2	sqr_diag_addlsh1
C AMD K8,K9	n/a		n/a			n/a
C AMD K10	n/a		n/a			n/a
C AMD bull	n/a		n/a			n/a
C AMD pile	n/a		n/a			n/a
C AMD steam	 ?		 ?			 ?
C AMD bobcat	n/a		n/a			n/a
C AMD jaguar	 ?		 ?			 ?
C Intel P4	n/a		n/a			n/a
C Intel core	n/a		n/a			n/a
C Intel NHM	n/a		n/a			n/a
C Intel SBR	n/a		n/a			n/a
C Intel IBR	n/a		n/a			n/a
C Intel HWL	 1.86		 2.15			~2.5
C Intel BWL	 ?		 ?			 ?
C Intel atom	n/a		n/a			n/a
C VIA nano	n/a		n/a			n/a

C The inner loops of this code are the result of running a code generation and
C optimisation tool suite written by David Harvey and Torbjörn Granlund, except
C that the sqr_diag_addlsh1 loop was manually written.

C TODO
C  * Replace current unoptimised sqr_diag_addlsh1 loop; 1.75 c/l might be
C    possible.
C  * Consider splitting outer loop into 2, one for n = 1 (mod 2) and one for
C    n = 0 (mod 2).  These loops could fall into specific "corner" code.
C  * Consider splitting outer loop into 4.
C  * Streamline pointer updates.
C  * Perhaps suppress a few more xor insns in feed-in code.
C  * Make sure we write no dead registers in feed-in code.
C  * We might use 32-bit size ops, since n >= 2^32 is non-terminating.  Watch
C    out for negative sizes being zero-extended, though.
C  * Provide straight-line code for n = 4; then look for simplifications in
C    main code.

define(`rp',	  `%rdi')
define(`up',	  `%rsi')
define(`un_param',`%rdx')


ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(mpn_sqr_basecase)
	FUNC_ENTRY(3)

	cmp	$2, un_param
	jae	L(gt1)

	mov	(up), %rdx
	mulx(	%rdx, %rax, %rdx)
	mov	%rax, (rp)
	mov	%rdx, 8(rp)
	FUNC_EXIT()
	ret

L(gt1):	jne	L(gt2)

	mov	(up), %rdx
	mov	8(up), %rcx
	mulx(	%rcx, %r9, %r10)	C v0 * v1	W 1 2
	mulx(	%rdx, %rax, %r8)	C v0 * v0	W 0 1
	mov	%rcx, %rdx
	mulx(	%rdx, %r11, %rdx)	C v1 * v1	W 2 3
	add	%r9, %r9		C		W 1
	adc	%r10, %r10		C		W 2
	adc	$0, %rdx		C		W 3
	add	%r9, %r8		C W 1
	adc	%r11, %r10		C W 2
	adc	$0, %rdx		C W 3
	mov	%rax, (rp)
	mov	%r8, 8(rp)
	mov	%r10, 16(rp)
	mov	%rdx, 24(rp)
	FUNC_EXIT()
	ret

L(gt2):	cmp	$4, un_param
	jae	L(gt3)
define(`v0', `%r8')
define(`v1', `%r9')
define(`w0', `%r10')
define(`w2', `%r11')

	mov	(up), v0
	mov	8(up), %rdx
	mov	%rdx, v1
	mulx(	v0, w2, %rax)
	mov	16(up), %rdx
	mulx(	v0, w0, %rcx)
	mov	w2, %r8
	add	%rax, w0
	adc	$0, %rcx
	mulx(	v1, %rdx, %rax)
	add	%rcx, %rdx
	mov	%rdx, 24(rp)
	adc	$0, %rax
	mov	%rax, 32(rp)
	xor	R32(%rcx), R32(%rcx)
	mov	(up), %rdx
	mulx(	%rdx, %rax, w2)
	mov	%rax, (rp)
	add	%r8, %r8
	adc	w0, w0
	setc	R8(%rcx)
	mov	8(up), %rdx
	mulx(	%rdx, %rax, %rdx)
	add	w2, %r8
	adc	%rax, w0
	mov	%r8, 8(rp)
	mov	w0, 16(rp)
	mov	24(rp), %r8
	mov	32(rp), w0
	lea	(%rdx,%rcx), w2
	adc	%r8, %r8
	adc	w0, w0
	setc	R8(%rcx)
	mov	16(up), %rdx
	mulx(	%rdx, %rax, %rdx)
	add	w2, %r8
	adc	%rax, w0
	mov	%r8, 24(rp)
	mov	w0, 32(rp)
	adc	%rcx, %rdx
	mov	%rdx, 40(rp)
	FUNC_EXIT()
	ret

L(gt3):

define(`v0', `%r8')
define(`v1', `%r9')
define(`w0', `%r10')
define(`w1', `%r11')
define(`w2', `%rbx')
define(`w3', `%rbp')
define(`un', `%r12')
define(`n',  `%rcx')

define(`X0', `%r13')
define(`X1', `%r14')

L(do_mul_2):
	push	%rbx
	push	%rbp
	push	%r12
	push	%r13
	push	%r14
	mov	$0, R32(un)
	sub	un_param, un		C free up rdx
	push	un
	mov	(up), v0
	mov	8(up), %rdx
	lea	2(un), n
	sar	$2, n			C FIXME: suppress, change loop?
	inc	un			C decrement |un|
	mov	%rdx, v1

	test	$1, R8(un)
	jnz	L(mx1)

L(mx0):	mulx(	v0, w2, w1)
	mov	16(up), %rdx
	mov	w2, 8(rp)
	xor	w2, w2
	mulx(	v0, w0, w3)
	test	$2, R8(un)
	jz	L(m00)

L(m10):	lea	-8(rp), rp
	lea	-8(up), up
	jmp	L(mlo2)

L(m00):	lea	8(up), up
	lea	8(rp), rp
	jmp	L(mlo0)

L(mx1):	mulx(	v0, w0, w3)
	mov	16(up), %rdx
	mov	w0, 8(rp)
	xor	w0, w0
	mulx(	v0, w2, w1)
	test	$2, R8(un)
	jz	L(mlo3)

L(m01):	lea	16(rp), rp
	lea	16(up), up
	jmp	L(mlo1)

	ALIGN(32)
L(mtop):mulx(	v1, %rax, w0)
	add	%rax, w2		C 0
	mov	(up), %rdx
	mulx(	v0, %rax, w1)
	adc	$0, w0			C 1
	add	%rax, w2		C 0
L(mlo1):adc	$0, w1			C 1
	add	w3, w2			C 0
	mov	w2, (rp)		C 0
	adc	$0, w1			C 1
	mulx(	v1, %rax, w2)
	add	%rax, w0		C 1
	mov	8(up), %rdx
	adc	$0, w2			C 2
	mulx(	v0, %rax, w3)
	add	%rax, w0		C 1
	adc	$0, w3			C 2
L(mlo0):add	w1, w0			C 1
	mov	w0, 8(rp)		C 1
	adc	$0, w3			C 2
	mulx(	v1, %rax, w0)
	add	%rax, w2		C 2
	mov	16(up), %rdx
	mulx(	v0, %rax, w1)
	adc	$0, w0			C 3
	add	%rax, w2		C 2
	adc	$0, w1			C 3
L(mlo3):add	w3, w2			C 2
	mov	w2, 16(rp)		C 2
	adc	$0, w1			C 3
	mulx(	v1, %rax, w2)
	add	%rax, w0		C 3
	mov	24(up), %rdx
	adc	$0, w2			C 4
	mulx(	v0, %rax, w3)
	add	%rax, w0		C 3
	adc	$0, w3			C 4
L(mlo2):add	w1, w0			C 3
	lea	32(up), up
	mov	w0, 24(rp)		C 3
	adc	$0, w3			C 4
	inc	n
	lea	32(rp), rp
	jnz	L(mtop)

L(mend):mulx(	v1, %rdx, %rax)
	add	%rdx, w2
	adc	$0, %rax
	add	w3, w2
	mov	w2, (rp)
	adc	$0, %rax
	mov	%rax, 8(rp)

	lea	16(up), up
	lea	-16(rp), rp

L(do_addmul_2):
L(outer):
	lea	(up,un,8), up		C put back up to 2 positions above last time
	lea	48(rp,un,8), rp		C put back rp to 4 positions above last time

	mov	-8(up), v0		C shared between addmul_2 and corner

	add	$2, un			C decrease |un|
	cmp	$-2, un
	jge	L(corner)

	mov	(up), v1

	lea	1(un), n
	sar	$2, n			C FIXME: suppress, change loop?

	mov	v1, %rdx
	test	$1, R8(un)
	jnz	L(bx1)

L(bx0):	mov	(rp), X0
	mov	8(rp), X1
	mulx(	v0, %rax, w1)
	add	%rax, X0
	adc	$0, w1
	mov	X0, (rp)
	xor	w2, w2
	test	$2, R8(un)
	jnz	L(b10)

L(b00):	mov	8(up), %rdx
	lea	16(rp), rp
	lea	16(up), up
	jmp	L(lo0)

L(b10):	mov	8(up), %rdx
	mov	16(rp), X0
	lea	32(up), up
	inc	n
	mulx(	v0, %rax, w3)
	jz	L(ex)
	jmp	L(lo2)

L(bx1):	mov	(rp), X1
	mov	8(rp), X0
	mulx(	v0, %rax, w3)
	mov	8(up), %rdx
	add	%rax, X1
	adc	$0, w3
	xor	w0, w0
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

	ALIGN(32)
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
L(ex):	add	%rax, X1
	adc	$0, w3
	mulx(	v1, %rdx, %rax)
	add	w1, X1
	mov	X1, 8(rp)
	adc	$0, w3
	add	w2, %rdx
	adc	$0, %rax
	add	%rdx, w3
	mov	w3, 16(rp)
	adc	$0, %rax
	mov	%rax, 24(rp)

	jmp	L(outer)		C loop until a small corner remains

L(corner):
	pop	un
	mov	(up), %rdx
	jg	L(small_corner)

	mov	%rdx, v1
	mov	(rp), X0
	mov	%rax, X1		C Tricky rax reuse of last iteration
	mulx(	v0, %rax, w1)
	add	%rax, X0
	adc	$0, w1
	mov	X0, (rp)
	mov	8(up), %rdx
	mulx(	v0, %rax, w3)
	add	%rax, X1
	adc	$0, w3
	mulx(	v1, %rdx, %rax)
	add	w1, X1
	mov	X1, 8(rp)
	adc	$0, w3
	add	w3, %rdx
	mov	%rdx, 16(rp)
	adc	$0, %rax
	mov	%rax, 24(rp)
	lea	32(rp), rp
	lea	16(up), up
	jmp	L(com)

L(small_corner):
	mulx(	v0, X1, w3)
	add	%rax, X1		C Tricky rax reuse of last iteration
	adc	$0, w3
	mov	X1, (rp)
	mov	w3, 8(rp)
	lea	16(rp), rp
	lea	8(up), up

L(com):

L(sqr_diag_addlsh1):
	lea	8(up,un,8), up		C put back up at its very beginning
	lea	(rp,un,8), rp
	lea	(rp,un,8), rp		C put back rp at its very beginning
	inc	un

	mov	-8(up), %rdx
	xor	R32(%rbx), R32(%rbx)	C clear CF as side effect
	mulx(	%rdx, %rax, %r10)
	mov	%rax, 8(rp)
	mov	16(rp), %r8
	mov	24(rp), %r9
	jmp	L(dm)

	ALIGN(16)
L(dtop):mov	32(rp), %r8
	mov	40(rp), %r9
	lea	16(rp), rp
	lea	(%rdx,%rbx), %r10
L(dm):	adc	%r8, %r8
	adc	%r9, %r9
	setc	R8(%rbx)
	mov	(up), %rdx
	lea	8(up), up
	mulx(	%rdx, %rax, %rdx)
	add	%r10, %r8
	adc	%rax, %r9
	mov	%r8, 16(rp)
	mov	%r9, 24(rp)
	inc	un
	jnz	L(dtop)

L(dend):adc	%rbx, %rdx
	mov	%rdx, 32(rp)

	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
