dnl  AMD64 mpn_addlsh1_n, mpn_rsblsh1_n optimised for Intel Atom.
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
C  * This code is slightly large at 433 bytes.
C  * sublsh1_n.asm and this file use the same basic pattern.

C	     cycles/limb
C AMD K8,K9	 ?
C AMD K10	 ?
C AMD bd1	 2.3
C AMD bobcat	 ?
C Intel P4	 ?
C Intel core2	 ?
C Intel NHM	 ?
C Intel SBR	 ?
C Intel atom	 4.875	(4.75 is probably possible)
C VIA nano	 ?

C INPUT PARAMETERS
define(`rp',       `%rdi')
define(`up',       `%rsi')
define(`vp',       `%rdx')
define(`n',        `%rcx')
define(`cy',       `%r8')

ifdef(`OPERATION_addlsh1_n', `
  define(ADDSUB,	add)
  define(ADCSBB,	adc)
  define(func_n,	mpn_addlsh1_n)
  define(func_nc,	mpn_addlsh1_nc)')
ifdef(`OPERATION_rsblsh1_n', `
  define(ADDSUB,	sub)
  define(ADCSBB,	sbb)
  define(func_n,	mpn_rsblsh1_n)
  define(func_nc,	mpn_rsblsh1_nc)')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

MULFUNC_PROLOGUE(mpn_addlsh1_n mpn_addlsh1_nc mpn_rsblsh1_n mpn_rsblsh1_nc)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(func_n)
	FUNC_ENTRY(4)
	push	%rbp
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
	ADCSBB	(up), %r8
	mov	%r8, (rp)
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
	ADCSBB	(up), %r8
	mov	%r8, (rp)
	ADCSBB	8(up), %r9
	mov	%r9, 8(rp)
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
	ADCSBB	(up), %r8
	mov	%r8, (rp)
	ADCSBB	8(up), %r9
	mov	%r9, 8(rp)
	ADCSBB	16(up), %r10
	mov	%r10, 16(rp)
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
	ADCSBB	(up), %r8
	mov	%r8, (rp)
	ADCSBB	8(up), %r9
	mov	%r9, 8(rp)
	ADCSBB	16(up), %r10
	mov	%r10, 16(rp)
	ADCSBB	24(up), %r11
	mov	%r11, 24(rp)
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
L(top):	add	R32(%rax), R32(%rax)	C restore scy
	lea	64(rp), rp
	mov	(vp), %r8
	adc	%r8, %r8
	mov	8(vp), %r9
	adc	%r9, %r9
	mov	16(vp), %r10
	adc	%r10, %r10
	mov	24(vp), %r11
	adc	%r11, %r11
	mov	32(vp), %r12
	adc	%r12, %r12
	mov	40(vp), %r13
	adc	%r13, %r13
	mov	48(vp), %r14
	adc	%r14, %r14
	mov	56(vp), %rbx
	adc	%rbx, %rbx
	lea	64(vp), vp
	sbb	R32(%rax), R32(%rax)	C save scy
	add	R32(%rbp), R32(%rbp)	C restore acy
	ADCSBB	(up), %r8
	mov	%r8, (rp)
	ADCSBB	8(up), %r9
	mov	%r9, 8(rp)
	ADCSBB	16(up), %r10
	mov	%r10, 16(rp)
	ADCSBB	24(up), %r11
	mov	%r11, 24(rp)
	ADCSBB	32(up), %r12
	mov	%r12, 32(rp)
	ADCSBB	40(up), %r13
	mov	%r13, 40(rp)
	ADCSBB	48(up), %r14
	mov	%r14, 48(rp)
	ADCSBB	56(up), %rbx
	mov	%rbx, 56(rp)
	sbb	R32(%rbp), R32(%rbp)	C save acy
	lea	64(up), up
L(x):	sub	$8, n
	jge	L(top)

L(end):	pop	%rbx
	pop	%r14
	pop	%r13
	pop	%r12
L(rtn):
ifdef(`OPERATION_addlsh1_n',`
	add	R32(%rbp), R32(%rax)
	neg	R32(%rax)')
ifdef(`OPERATION_rsblsh1_n',`
	sub	R32(%rax), R32(%rbp)
	movslq	R32(%rbp), %rax')

	pop	%rbp
	FUNC_EXIT()
	ret
EPILOGUE()
PROLOGUE(func_nc)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8	')
	push	%rbp
	neg	%r8			C set CF
	sbb	R32(%rbp), R32(%rbp)	C save acy
	jmp	L(ent)
EPILOGUE()
