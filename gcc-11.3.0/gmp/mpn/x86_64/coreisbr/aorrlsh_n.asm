dnl  AMD64 mpn_addlsh_n -- rp[] = up[] + (vp[] << k)
dnl  AMD64 mpn_rsblsh_n -- rp[] = (vp[] << k) - up[]
dnl  Optimised for Sandy Bridge.

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

C	     cycles/limb
C AMD K8,K9	 ?
C AMD K10	 5.25
C Intel P4	 ?
C Intel core2	 3.1
C Intel NHM	 3.95
C Intel SBR	 2.75
C Intel atom	 ?
C VIA nano	 ?

C The inner-loop probably runs close to optimally on Sandy Bridge (using 4-way
C unrolling).  The rest of the code is quite crude, and could perhaps be made
C both smaller and faster.

C INPUT PARAMETERS
define(`rp',	`%rdi')
define(`up',	`%rsi')
define(`vp',	`%rdx')
define(`n',	`%rcx')
define(`cnt',	`%r8')
define(`cy',	`%r9')			C for _nc variant

ifdef(`OPERATION_addlsh_n', `
	define(ADDSUB,	add)
	define(ADCSBB,	adc)
	define(IFRSB,	)
	define(func_n,	mpn_addlsh_n)
	define(func_nc,	mpn_addlsh_nc)')
ifdef(`OPERATION_rsblsh_n', `
	define(ADDSUB,	sub)
	define(ADCSBB,	sbb)
	define(IFRSB,	`$1')
	define(func_n,	mpn_rsblsh_n)
	define(func_nc,	mpn_rsblsh_nc)')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

C mpn_rsblsh_nc removed below, its idea of carry-in is inconsistent with
C refmpn_rsblsh_nc
MULFUNC_PROLOGUE(mpn_addlsh_n mpn_addlsh_nc mpn_rsblsh_n)

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(func_n)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8d	')	C cnt
	push	%rbx
	xor	R32(%rbx), R32(%rbx)	C clear CF save register
L(ent):	push	%rbp
	mov	R32(n), R32(%rbp)
	mov	n, %rax
	mov	R32(cnt), R32(%rcx)
	neg	R32(%rcx)
	and	$3, R32(%rbp)
	jz	L(b0)
	lea	-32(vp,%rbp,8), vp
	lea	-32(up,%rbp,8), up
	lea	-32(rp,%rbp,8), rp
	cmp	$2, R32(%rbp)
	jc	L(b1)
	jz	L(b2)

L(b3):	xor	%r8, %r8
	mov	8(vp), %r9
	mov	16(vp), %r10
	shrd	R8(%rcx), %r9, %r8
	shrd	R8(%rcx), %r10, %r9
	mov	24(vp), %r11
	shrd	R8(%rcx), %r11, %r10
	sub	$3, %rax
	jz	L(3)
	add	R32(%rbx), R32(%rbx)
	lea	32(vp), vp
	ADCSBB	8(up), %r8
	ADCSBB	16(up), %r9
	ADCSBB	24(up), %r10
	lea	32(up), up
	jmp	L(lo3)
L(3):	add	R32(%rbx), R32(%rbx)
	lea	32(vp), vp
	ADCSBB	8(up), %r8
	ADCSBB	16(up), %r9
	ADCSBB	24(up), %r10
	jmp	L(wd3)

L(b0):	mov	(vp), %r8
	mov	8(vp), %r9
	xor	R32(%rbp), R32(%rbp)
	jmp	L(lo0)

L(b1):	xor	%r10, %r10
	mov	24(vp), %r11
	shrd	R8(%rcx), %r11, %r10
	sub	$1, %rax
	jz	L(1)
	add	R32(%rbx), R32(%rbx)
	lea	32(vp), vp
	ADCSBB	24(up), %r10
	lea	32(up), up
	mov	(vp), %r8
	jmp	L(lo1)
L(1):	add	R32(%rbx), R32(%rbx)
	ADCSBB	24(up), %r10
	jmp	L(wd1)

L(b2):	xor	%r9, %r9
	mov	16(vp), %r10
	shrd	R8(%rcx), %r10, %r9
	mov	24(vp), %r11
	shrd	R8(%rcx), %r11, %r10
	sub	$2, %rax
	jz	L(2)
	add	R32(%rbx), R32(%rbx)
	lea	32(vp), vp
	ADCSBB	16(up), %r9
	ADCSBB	24(up), %r10
	lea	32(up), up
	jmp	L(lo2)
L(2):	add	R32(%rbx), R32(%rbx)
	ADCSBB	16(up), %r9
	ADCSBB	24(up), %r10
	jmp	L(wd2)

	ALIGN(32)			C 16-byte alignment is not enough!
L(top):	shrd	R8(%rcx), %r11, %r10
	add	R32(%rbx), R32(%rbx)
	lea	32(vp), vp
	ADCSBB	(up), %rbp
	ADCSBB	8(up), %r8
	ADCSBB	16(up), %r9
	ADCSBB	24(up), %r10
	mov	%rbp, (rp)
	lea	32(up), up
L(lo3):	mov	%r8, 8(rp)
L(lo2):	mov	%r9, 16(rp)
	mov	(vp), %r8
L(lo1):	mov	%r10, 24(rp)
	mov	8(vp), %r9
	mov	%r11, %rbp
	lea	32(rp), rp
	sbb	R32(%rbx), R32(%rbx)
L(lo0):	shrd	R8(%rcx), %r8, %rbp
	mov	16(vp), %r10
	shrd	R8(%rcx), %r9, %r8
	shrd	R8(%rcx), %r10, %r9
	mov	24(vp), %r11
	sub	$4, %rax
	jg	L(top)

	shrd	R8(%rcx), %r11, %r10
	add	R32(%rbx), R32(%rbx)
	ADCSBB	(up), %rbp
	ADCSBB	8(up), %r8
	ADCSBB	16(up), %r9
	ADCSBB	24(up), %r10
	mov	%rbp, (rp)
L(wd3):	mov	%r8, 8(rp)
L(wd2):	mov	%r9, 16(rp)
L(wd1):	mov	%r10, 24(rp)
	adc	R32(%rax), R32(%rax)	C rax is zero after loop
	shr	R8(%rcx), %r11
	ADDSUB	%r11, %rax
IFRSB(	neg	%rax)
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
PROLOGUE(func_nc)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8d	')	C cnt
IFDOS(`	mov	64(%rsp), %r9	')	C cy
	push	%rbx
	neg	cy
	sbb	R32(%rbx), R32(%rbx)	C initialise CF save register
	jmp	L(ent)
EPILOGUE()
