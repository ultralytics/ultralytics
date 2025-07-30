dnl  x86-64 mpn_div_qr_2n_pi1
dnl  -- Divide an mpn number by a normalized 2-limb number,
dnl     using a single-limb inverse.

dnl  Copyright 2007, 2008, 2010-2012 Free Software Foundation, Inc.

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


C		c/l
C INPUT PARAMETERS
define(`qp',		`%rdi')
define(`rp',		`%rsi')
define(`up_param',	`%rdx')
define(`un',		`%rcx')
define(`d1',		`%r8')
define(`d0',		`%r9')
define(`di_param',	`8(%rsp)')

define(`di',		`%r10')
define(`up',		`%r11')
define(`u2',		`%rbx')
define(`u1',		`%r12')
define(`t1',		`%r13')
define(`t0',		`%r14')
define(`md1',		`%r15')

C TODO
C * Store qh in the same stack slot as di_param, instead of pushing
C   it. (we could put it in register %rbp, but then we would need to
C   save and restore that instead, which doesn't seem like a win).

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_div_qr_2n_pi1)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8	')
IFDOS(`	mov	64(%rsp), %r9	')
IFDOS(`define(`di_param', `72(%rsp)')')
	mov	di_param, di
	mov	up_param, up
	push	%r15
	push	%r14
	push	%r13
	push	%r12
	push	%rbx

	mov	-16(up, un, 8), u1
	mov	-8(up, un, 8), u2

	mov	u1, t0
	mov	u2, t1
	sub	d0, t0
	sbb	d1, t1
	cmovnc  t0, u1
	cmovnc	t1, u2
	C push qh which is !carry
	sbb	%rax, %rax
	inc	%rax
	push	%rax
	lea	-2(un), un
	mov	d1, md1
	neg	md1

	jmp	L(next)

	ALIGN(16)
L(loop):
	C udiv_qr_3by2 (q,u2,u1,u2,u1,n0, d1,d0,di)
	C Based on the optimized divrem_2.asm code.

	mov	di, %rax
	mul	u2
	mov	u1, t0
	add	%rax, t0	C q0 in t0
	adc	u2, %rdx
	mov	%rdx, t1	C q in t1
	imul	md1, %rdx
	mov	d0, %rax
	lea	(%rdx, u1), u2
	mul	t1
	mov	(up, un, 8), u1
	sub	d0, u1
	sbb	d1, u2
	sub	%rax, u1
	sbb	%rdx, u2
	xor	R32(%rax), R32(%rax)
	xor	R32(%rdx), R32(%rdx)
	cmp	t0, u2
	cmovnc	d0, %rax
	cmovnc	d1, %rdx
	adc	$0, t1
	nop
	add	%rax, u1
	adc	%rdx, u2
	cmp	d1, u2
	jae	L(fix)
L(bck):
	mov	t1, (qp, un, 8)
L(next):
	sub	$1, un
	jnc	L(loop)
L(end):
	mov	u2, 8(rp)
	mov	u1, (rp)

	C qh on stack
	pop	%rax

	pop	%rbx
	pop	%r12
	pop	%r13
	pop	%r14
	pop	%r15
	FUNC_EXIT()
	ret

L(fix):	C Unlikely update. u2 >= d1
	seta	%dl
	cmp	d0, u1
	setae	%al
	orb	%dl, %al		C "orb" form to placate Sun tools
	je	L(bck)
	inc	t1
	sub	d0, u1
	sbb	d1, u2
	jmp	L(bck)
EPILOGUE()
