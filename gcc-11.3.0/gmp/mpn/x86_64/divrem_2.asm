dnl  x86-64 mpn_divrem_2 -- Divide an mpn number by a normalized 2-limb number.

dnl  Copyright 2007, 2008, 2010, 2014 Free Software Foundation, Inc.

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


C	     cycles/limb	best
C AMD K8,K9	18
C AMD K10	18
C AMD bull
C AMD pile
C AMD bobcat
C AMD jaguar
C Intel P4	68
C Intel core	34
C Intel NHM	30.25
C Intel SBR	21.3
C Intel IBR	21.4
C Intel HWL	20.6
C Intel BWL
C Intel atom	73
C VIA nano	33


C INPUT PARAMETERS
define(`qp',		`%rdi')
define(`fn',		`%rsi')
define(`up_param',	`%rdx')
define(`un_param',	`%rcx')
define(`dp',		`%r8')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_divrem_2)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8	')
	push	%r15
	push	%r14
	push	%r13
	push	%r12
	lea	-24(%rdx,%rcx,8), %r12	C r12 = &up[un-1]
	mov	%rsi, %r13
	push	%rbp
	mov	%rdi, %rbp
	push	%rbx
	mov	8(%r8), %r11		C d1
	mov	16(%r12), %rbx
	mov	(%r8), %r8		C d0
	mov	8(%r12), %r10

	xor	R32(%r15), R32(%r15)
	cmp	%rbx, %r11
	ja	L(2)
	setb	%dl
	cmp	%r10, %r8
	setbe	%al
	orb	%al, %dl		C "orb" form to placate Sun tools
	je	L(2)
	inc	R32(%r15)
	sub	%r8, %r10
	sbb	%r11, %rbx
L(2):
	lea	-3(%rcx,%r13), %r14	C un + fn - 3
	test	%r14, %r14
	js	L(end)

	push	%r8
	push	%r10
	push	%r11
IFSTD(`	mov	%r11, %rdi	')
IFDOS(`	mov	%r11, %rcx	')
	ASSERT(nz, `test $15, %rsp')
	CALL(	mpn_invert_limb)
	pop	%r11
	pop	%r10
	pop	%r8

	mov	%r11, %rdx
	mov	%rax, %rdi
	imul	%rax, %rdx
	mov	%rdx, %r9
	mul	%r8
	xor	R32(%rcx), R32(%rcx)
	add	%r8, %r9
	adc	$-1, %rcx
	add	%rdx, %r9
	adc	$0, %rcx
	js	2f
1:	dec	%rdi
	sub	%r11, %r9
	sbb	$0, %rcx
	jns	1b
2:

	lea	(%rbp,%r14,8), %rbp
	mov	%r11, %rsi
	neg	%rsi			C -d1

C rax rbx rcx rdx rsi rdi  rbp r8 r9 r10 r11 r12 r13 r14 r15
C     n2  un      -d1 dinv qp  d0 q0     d1  up  fn      msl

	ALIGN(16)
L(top):	mov	%rdi, %rax		C di		ncp
	mul	%rbx			C		0, 17
	mov	%r10, %rcx		C
	add	%rax, %rcx		C		4
	adc	%rbx, %rdx		C		5
	mov	%rdx, %r9		C q		6
	imul	%rsi, %rdx		C		6
	mov	%r8, %rax		C		ncp
	lea	(%rdx, %r10), %rbx	C n1 -= ...	10
	xor	R32(%r10), R32(%r10)	C
	mul	%r9			C		7
	cmp	%r14, %r13		C
	jg	L(19)			C
	mov	(%r12), %r10		C
	sub	$8, %r12		C
L(19):	sub	%r8, %r10		C		ncp
	sbb	%r11, %rbx		C		11
	sub	%rax, %r10		C		11
	sbb	%rdx, %rbx		C		12
	xor	R32(%rax), R32(%rax)	C
	xor	R32(%rdx), R32(%rdx)	C
	cmp	%rcx, %rbx		C		13
	cmovnc	%r8, %rax		C		14
	cmovnc	%r11, %rdx		C		14
	adc	$0, %r9			C adjust q	14
	nop
	add	%rax, %r10		C		15
	adc	%rdx, %rbx		C		16
	cmp	%r11, %rbx		C
	jae	L(fix)			C
L(bck):	mov	%r9, (%rbp)		C
	sub	$8, %rbp		C
	dec	%r14
	jns	L(top)

L(end):	mov	%r10, 8(%r12)
	mov	%rbx, 16(%r12)
	pop	%rbx
	pop	%rbp
	pop	%r12
	pop	%r13
	pop	%r14
	mov	%r15, %rax
	pop	%r15
	FUNC_EXIT()
	ret

L(fix):	seta	%dl
	cmp	%r8, %r10
	setae	%al
	orb	%dl, %al		C "orb" form to placate Sun tools
	je	L(bck)
	inc	%r9
	sub	%r8, %r10
	sbb	%r11, %rbx
	jmp	L(bck)
EPILOGUE()
