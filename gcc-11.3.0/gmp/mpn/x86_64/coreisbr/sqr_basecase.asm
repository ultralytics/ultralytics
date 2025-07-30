dnl  AMD64 mpn_sqr_basecase optimised for Intel Sandy bridge and Ivy bridge.

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
C AMD K8,K9	 ?		 ?			 ?
C AMD K10	 ?		 ?			 ?
C AMD bull	 ?		 ?			 ?
C AMD pile	 ?		 ?			 ?
C AMD steam	 ?		 ?			 ?
C AMD bobcat	 ?		 ?			 ?
C AMD jaguar	 ?		 ?			 ?
C Intel P4	 ?		 ?			 ?
C Intel core	 ?		 ?			 ?
C Intel NHM	 ?		 ?			 ?
C Intel SBR	 2.57		 2.93			 3.0
C Intel IBR	 2.35		 2.66			 3.0
C Intel HWL	 2.02		 2.5			 2.5
C Intel BWL	 ?		 ?			 ?
C Intel atom	 ?		 ?			 ?
C VIA nano	 ?		 ?			 ?

C The inner loops of this code are the result of running a code generation and
C optimisation tool suite written by David Harvey and Torbjörn Granlund, except
C that the sqr_diag_addlsh1 loop was manually written.

C TODO
C  * Replace current unoptimised sqr_diag_addlsh1 loop, 2.5 c/l should be easy.
C  * Streamline pointer updates.
C  * Perhaps suppress a few more xor insns in feed-in code.
C  * Make sure we write no dead registers in feed-in code.
C  * We might use 32-bit size ops, since n >= 2^32 is non-terminating.  Watch
C    out for negative sizes being zero-extended, though.
C  * The straight-line code for n <= 3 comes from the K8 code, and might be
C    quite sub-optimal here.  Write specific code, and add code for n = 4.
C  * The mul_2 loop has a 10 insn common sequence in the loop start and the
C    wind-down code.  Try re-rolling it.
C  * This file has been the subject to just basic micro-optimisation.

C When playing with pointers, set this to $2 to fall back to conservative
C indexing in wind-down code.
define(`I',`$1')

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

	mov	(up), %rax
	mul	%rax
	mov	%rax, (rp)
	mov	%rdx, 8(rp)
	FUNC_EXIT()
	ret

L(gt1):	jne	L(gt2)

	mov	(up), %rax
	mov	%rax, %r8
	mul	%rax
	mov	8(up), %r11
	mov	%rax, (rp)
	mov	%r11, %rax
	mov	%rdx, %r9
	mul	%rax
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

L(gt2):	cmp	$4, un_param
	jae	L(gt3)
define(`v0', `%r8')
define(`v1', `%r9')
define(`w0', `%r10')
define(`w2', `%r11')

	mov	(up), %rax
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
	mov	(up), v0
	push	%rbx
	lea	(rp,un_param,8), rp	C point rp at R[un]
	mov	8(up), %rax
	push	%rbp
	lea	(up,un_param,8), up	C point up right after U's end
	mov	%rax, v1
	push	%r12
	mov	$1, R32(un)		C free up rdx
	push	%r13
	sub	un_param, un
	push	%r14
	push	un
	mul	v0
	mov	%rax, (rp,un,8)
	mov	8(up,un,8), %rax
	test	$1, R8(un)
	jnz	L(m2b1)

L(m2b0):lea	2(un), n
	xor	R32(w1), R32(w1)	C FIXME
	xor	R32(w2), R32(w2)	C FIXME
	mov	%rdx, w0
	jmp	L(m2l0)

L(m2b1):lea	1(un), n
	xor	R32(w3), R32(w3)	C FIXME
	xor	R32(w0), R32(w0)	C FIXME
	mov	%rdx, w2
	jmp	L(m2l1)

	ALIGN(32)
L(m2tp):
L(m2l0):mul	v0
	add	%rax, w0
	mov	%rdx, w3
	adc	$0, w3
	mov	-8(up,n,8), %rax
	mul	v1
	add	w1, w0
	adc	$0, w3
	add	%rax, w2
	mov	w0, -8(rp,n,8)
	mov	%rdx, w0
	adc	$0, w0
	mov	(up,n,8), %rax
L(m2l1):mul	v0
	add	%rax, w2
	mov	%rdx, w1
	adc	$0, w1
	add	w3, w2
	mov	(up,n,8), %rax
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

	add	$2, un			C decrease |un|

L(do_addmul_2):
L(outer):
	lea	16(rp), rp
	cmp	$-2, R32(un)		C jump if un C {-1,0}  FIXME jump if un C {-2,1}
	jge	L(corner)		C FIXME: move to before the lea above

	mov	-8(up,un,8), v0
	mov	(up,un,8), %rax
	mov	%rax, v1
	mul	v0
	test	$1, R8(un)
	jnz	L(a1x1)

L(a1x0):mov	(rp,un,8), X0
	xor	w0, w0
	mov	8(rp,un,8), X1
	add	%rax, X0
	mov	%rdx, w1
	adc	$0, w1
	xor	w2, w2
	mov	X0, (rp,un,8)
	mov	8(up,un,8), %rax
	test	$2, R8(un)
	jnz	L(a110)

L(a100):lea	2(un), n		C un = 4, 8, 12, ...
	jmp	L(lo0)

L(a110):lea	(un), n			C un = 2, 6, 10, ...
	jmp	L(lo2)

L(a1x1):mov	(rp,un,8), X1
	xor	w2, w2
	mov	8(rp,un,8), X0
	add	%rax, X1
	mov	%rdx, w3
	adc	$0, w3
	xor	w0, w0
	mov	8(up,un,8), %rax
	test	$2, R8(un)
	jz	L(a111)

L(a101):lea	3(un), n		C un = 1, 5, 9, ...
	jmp	L(lo1)

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
L(lo1):	mul	v0
	add	%rax, X0
	mov	%rdx, w1
	adc	$0, w1
	mov	-16(up,n,8), %rax
	mul	v1
	mov	X1, -24(rp,n,8)
	mov	-8(rp,n,8), X1
	add	w3, X0
	adc	$0, w1
	mov	%rdx, w2
	mov	X0, -16(rp,n,8)
	add	%rax, X1
	adc	$0, w2
	mov	-8(up,n,8), %rax
	add	w0, X1
	adc	$0, w2
L(lo0):	mul	v0
	add	%rax, X1
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
L(lo3):	mul	v0
	add	w2, X0
	mov	X1, -8(rp,n,8)
	mov	%rdx, w1
	adc	$0, w0
	add	%rax, X0
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
L(lo2):	mul	v0
	add	w0, X1
	mov	%rdx, w3
	adc	$0, w2
	add	%rax, X1
	mov	8(up,n,8), %rax
	mov	16(rp,n,8), X0
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

	add	$2, un			C decrease |un|
	jmp	L(outer)		C loop until a small corner remains

L(corner):
	pop	n
	jg	L(small_corner)

	lea	8(rp), rp
	mov	-24(up), v0
	mov	-16(up), %rax
	mov	%rax, v1
	mul	v0
	mov	-24(rp), X0
	mov	-16(rp), X1
	add	%rax, X0
	mov	%rdx, w1
	adc	$0, w1
	xor	w2, w2
	mov	X0, -24(rp)
	mov	-8(up), %rax
	mul	v0
	add	$0, X1
	mov	%rdx, w3
	adc	$0, w2
	add	%rax, X1
	mov	-8(up), %rax
	adc	$0, w3
	mul	v1
	add	w1, X1
	adc	$0, w3
	add	w2, %rax
	adc	$0, %rdx
	mov	X1, -16(rp)
	jmp	L(com)

L(small_corner):
	mov	-8(rp), w3
	mov	-16(up), v0
	mov	-8(up), %rax
	mul	v0
L(com):	add	w3, %rax
	adc	$0, %rdx
	mov	%rax, -8(rp)
	mov	%rdx, (rp)

L(sqr_diag_addlsh1):
	mov	-8(up,n,8), %rax
	shl	n
	mul	%rax
	mov	%rax, (rp,n,8)

	xor	R32(%rbx), R32(%rbx)
	mov	8(rp,n,8), %r8
	mov	16(rp,n,8), %r9
	jmp	L(dm)

	ALIGN(32)
L(dtop):add	%r8, %r10
	adc	%r9, %rax
	mov	8(rp,n,8), %r8
	mov	16(rp,n,8), %r9
	mov	%r10, -8(rp,n,8)
	mov	%rax, (rp,n,8)
L(dm):	adc	%r8, %r8
	adc	%r9, %r9
	mov	(up,n,4), %rax
	lea	(%rdx,%rbx), %r10
	setc	R8(%rbx)
	mul	%rax
	add	$2, n
	js	L(dtop)

L(dend):add	%r8, %r10
	adc	%r9, %rax
	mov	%r10, I(-8(rp),-8(rp,n,8))
	mov	%rax, I((rp),(rp,n,8))
	adc	%rbx, %rdx
	mov	%rdx, I(8(rp),8(rp,n,8))

	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
