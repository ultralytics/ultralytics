dnl  AMD64 mpn_addlsh_n -- rp[] = up[] + (vp[] << k)
dnl  AMD64 mpn_rsblsh_n -- rp[] = (vp[] << k) - up[]
dnl  Optimised for Nehalem.

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
C AMD K10	 4.75
C Intel P4	 ?
C Intel core2	 2.8-3
C Intel NHM	 2.8
C Intel SBR	 3.55
C Intel atom	 ?
C VIA nano	 ?

C The inner-loop probably runs close to optimally on Nehalem (using 4-way
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

C mpn_rsblsh_nc removed below, its idea of carry-in is inconsistent with
C refmpn_rsblsh_nc
MULFUNC_PROLOGUE(mpn_addlsh_n mpn_addlsh_nc mpn_rsblsh_n)

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

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

	lea	-8(up,%rax,8), up
	lea	-8(vp,%rax,8), vp
	lea	-40(rp,%rax,8), rp
	neg	%rax

	and	$3, R32(%rbp)
	jz	L(b0)
	cmp	$2, R32(%rbp)
	jc	L(b1)
	jz	L(b2)

L(b3):	xor	R32(%r9), R32(%r9)
	mov	8(vp,%rax,8), %r10
	mov	16(vp,%rax,8), %r11
	shrd	%cl, %r10, %r9
	shrd	%cl, %r11, %r10
	add	R32(%rbx), R32(%rbx)
	ADCSBB	8(up,%rax,8), %r9
	mov	24(vp,%rax,8), %r8
	ADCSBB	16(up,%rax,8), %r10
	sbb	R32(%rbx), R32(%rbx)
	add	$3, %rax
	jmp	L(lo3)

L(b0):	mov	8(vp,%rax,8), %r9
	xor	R32(%r8), R32(%r8)
	shrd	%cl, %r9, %r8
	mov	16(vp,%rax,8), %r10
	mov	24(vp,%rax,8), %r11
	shrd	%cl, %r10, %r9
	shrd	%cl, %r11, %r10
	add	R32(%rbx), R32(%rbx)
	ADCSBB	8(up,%rax,8), %r8
	mov	%r8, 40(rp,%rax,8)	C offset 40
	ADCSBB	16(up,%rax,8), %r9
	mov	32(vp,%rax,8), %r8
	ADCSBB	24(up,%rax,8), %r10
	sbb	R32(%rbx), R32(%rbx)
	add	$4, %rax
	jmp	L(lo0)

L(b1):	mov	8(vp,%rax,8), %r8
	add	$1, %rax
	jz	L(1)
	mov	8(vp,%rax,8), %r9
	xor	R32(%rbp), R32(%rbp)
	jmp	L(lo1)
L(1):	xor	R32(%r11), R32(%r11)
	jmp	L(wd1)

L(b2):	xor	%r10, %r10
	mov	8(vp,%rax,8), %r11
	shrd	%cl, %r11, %r10
	add	R32(%rbx), R32(%rbx)
	mov	16(vp,%rax,8), %r8
	ADCSBB	8(up,%rax,8), %r10
	sbb	R32(%rbx), R32(%rbx)
	add	$2, %rax
	jz	L(end)

	ALIGN(16)
L(top):	mov	8(vp,%rax,8), %r9
	mov	%r11, %rbp
L(lo2):	mov	%r10, 24(rp,%rax,8)	C offset 24
L(lo1):	shrd	%cl, %r8, %rbp
	shrd	%cl, %r9, %r8
	mov	16(vp,%rax,8), %r10
	mov	24(vp,%rax,8), %r11
	shrd	%cl, %r10, %r9
	shrd	%cl, %r11, %r10
	add	R32(%rbx), R32(%rbx)
	ADCSBB	(up,%rax,8), %rbp
	ADCSBB	8(up,%rax,8), %r8
	mov	%r8, 40(rp,%rax,8)	C offset 40
	ADCSBB	16(up,%rax,8), %r9
	mov	32(vp,%rax,8), %r8
	ADCSBB	24(up,%rax,8), %r10
	sbb	R32(%rbx), R32(%rbx)
	add	$4, %rax
	mov	%rbp, (rp,%rax,8)	C offset 32
L(lo0):
L(lo3):	mov	%r9, 16(rp,%rax,8)	C offset 48
	jnz	L(top)

L(end):	mov	%r10, 24(rp,%rax,8)
L(wd1):	shrd	%cl, %r8, %r11
	add	R32(%rbx), R32(%rbx)
	ADCSBB	(up,%rax,8), %r11
	mov	%r11, 32(rp,%rax,8)	C offset 32
	adc	R32(%rax), R32(%rax)	C rax is zero after loop
	shr	R8(%rcx), %r8
	ADDSUB	%r8, %rax
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
