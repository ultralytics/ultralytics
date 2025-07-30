dnl  AMD64 mpn_sublsh1_n optimised for Intel Atom.
dnl  Used also for AMD bd1.

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2011, 2012 Free Software Foundation, Inc.

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

C TODO
C  * This code is slightly large at 501 bytes.
C  * aorrlsh1_n.asm and this file use the same basic pattern.

C	     cycles/limb
C AMD K8,K9	 ?
C AMD K10	 ?
C AMD bd1	 2.3
C AMD bobcat	 ?
C Intel P4	 ?
C Intel core2	 ?
C Intel NHM	 ?
C Intel SBR	 ?
C Intel atom	 5	(4.875 is probably possible)
C VIA nano	 ?

C INPUT PARAMETERS
define(`rp',       `%rdi')
define(`up',       `%rsi')
define(`vp',       `%rdx')
define(`n',        `%rcx')
define(`cy',       `%r8')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_sublsh1_n)
	FUNC_ENTRY(4)
	push	%rbp
	push	%r15
	xor	R32(%rbp), R32(%rbp)
L(ent):	mov	R32(n), R32(%rax)
	and	$3, R32(%rax)
	jz	L(b0)
	cmp	$2, R32(%rax)
	jz	L(b2)
	jg	L(b3)

L(b1):	mov	(vp), %r8
	add	%r8, %r8
	lea	8(vp), vp
	sbb	R32(%rax), R32(%rax)	C save scy
	add	R32(%rbp), R32(%rbp)	C restore acy
	mov	(up), %r15
	sbb	%r8, %r15
	mov	%r15, (rp)
	sbb	R32(%rbp), R32(%rbp)	C save acy
	lea	8(up), up
	lea	8(rp), rp
	jmp	L(b0)

L(b2):	mov	(vp), %r8
	add	%r8, %r8
	mov	8(vp), %r9
	adc	%r9, %r9
	lea	16(vp), vp
	sbb	R32(%rax), R32(%rax)	C save scy
	add	R32(%rbp), R32(%rbp)	C restore acy
	mov	(up), %r15
	sbb	%r8, %r15
	mov	%r15, (rp)
	mov	8(up), %r15
	sbb	%r9, %r15
	mov	%r15, 8(rp)
	sbb	R32(%rbp), R32(%rbp)	C save acy
	lea	16(up), up
	lea	16(rp), rp
	jmp	L(b0)

L(b3):	mov	(vp), %r8
	add	%r8, %r8
	mov	8(vp), %r9
	adc	%r9, %r9
	mov	16(vp), %r10
	adc	%r10, %r10
	lea	24(vp), vp
	sbb	R32(%rax), R32(%rax)	C save scy
	add	R32(%rbp), R32(%rbp)	C restore acy
	mov	(up), %r15
	sbb	%r8, %r15
	mov	%r15, (rp)
	mov	8(up), %r15
	sbb	%r9, %r15
	mov	%r15, 8(rp)
	mov	16(up), %r15
	sbb	%r10, %r15
	mov	%r15, 16(rp)
	sbb	R32(%rbp), R32(%rbp)	C save acy
	lea	24(up), up
	lea	24(rp), rp

L(b0):	test	$4, R8(n)
	jz	L(skp)
	add	R32(%rax), R32(%rax)	C restore scy
	mov	(vp), %r8
	adc	%r8, %r8
	mov	8(vp), %r9
	adc	%r9, %r9
	mov	16(vp), %r10
	adc	%r10, %r10
	mov	24(vp), %r11
	adc	%r11, %r11
	lea	32(vp), vp
	sbb	R32(%rax), R32(%rax)	C save scy
	add	R32(%rbp), R32(%rbp)	C restore acy
	mov	(up), %r15
	sbb	%r8, %r15
	mov	%r15, (rp)
	mov	8(up), %r15
	sbb	%r9, %r15
	mov	%r15, 8(rp)
	mov	16(up), %r15
	sbb	%r10, %r15
	mov	%r15, 16(rp)
	mov	24(up), %r15
	sbb	%r11, %r15
	mov	%r15, 24(rp)
	lea	32(up), up
	lea	32(rp), rp
	sbb	R32(%rbp), R32(%rbp)	C save acy

L(skp):	cmp	$8, n
	jl	L(rtn)

	push	%r12
	push	%r13
	push	%r14
	push	%rbx
	lea	-64(rp), rp
	jmp	L(x)

	ALIGN(16)
L(top):	mov	(vp), %r8
	add	R32(%rax), R32(%rax)
	lea	64(vp), vp
	adc	%r8, %r8
	mov	-56(vp), %r9
	adc	%r9, %r9
	mov	-48(vp), %r10
	adc	%r10, %r10
	mov	-40(vp), %r11
	adc	%r11, %r11
	mov	-32(vp), %r12
	adc	%r12, %r12
	mov	-24(vp), %r13
	adc	%r13, %r13
	mov	-16(vp), %r14
	adc	%r14, %r14
	mov	-8(vp), %r15
	adc	%r15, %r15
	sbb	R32(%rax), R32(%rax)
	add	R32(%rbp), R32(%rbp)
	mov	(up), %rbp
	lea	64(rp), rp
	mov	8(up), %rbx
	sbb	%r8, %rbp
	mov	32(up), %r8
	mov	%rbp, (rp)
	sbb	%r9, %rbx
	mov	16(up), %rbp
	mov	%rbx, 8(rp)
	sbb	%r10, %rbp
	mov	24(up), %rbx
	mov	%rbp, 16(rp)
	sbb	%r11, %rbx
	mov	%rbx, 24(rp)
	sbb	%r12, %r8
	mov	40(up), %r9
	mov	%r8, 32(rp)
	sbb	%r13, %r9
	mov	48(up), %rbp
	mov	%r9, 40(rp)
	sbb	%r14, %rbp
	mov	56(up), %rbx
	mov	%rbp, 48(rp)
	sbb	%r15, %rbx
	lea	64(up), up
	mov	%rbx, 56(rp)
	sbb	R32(%rbp), R32(%rbp)
L(x):	sub	$8, n
	jge	L(top)

L(end):	pop	%rbx
	pop	%r14
	pop	%r13
	pop	%r12
L(rtn):
	add	R32(%rbp), R32(%rax)
	neg	R32(%rax)

	pop	%r15
	pop	%rbp
	FUNC_EXIT()
	ret
EPILOGUE()
PROLOGUE(mpn_sublsh1_nc)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8	')
	push	%rbp
	push	%r15
	neg	%r8			C set CF
	sbb	R32(%rbp), R32(%rbp)	C save acy
	jmp	L(ent)
EPILOGUE()
