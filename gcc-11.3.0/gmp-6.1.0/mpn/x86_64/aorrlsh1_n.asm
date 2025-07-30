dnl  AMD64 mpn_addlsh1_n -- rp[] = up[] + (vp[] << 1)
dnl  AMD64 mpn_rsblsh1_n -- rp[] = (vp[] << 1) - up[]

dnl  Copyright 2003, 2005-2009, 2011, 2012 Free Software Foundation, Inc.

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
C AMD K8,K9	 2
C AMD K10	 2
C AMD bd1	 ?
C AMD bobcat	 ?
C Intel P4	 13
C Intel core2	 3.45
C Intel NHM	 ?
C Intel SBR	 ?
C Intel atom	 ?
C VIA nano	 ?


C Sometimes speed degenerates, supposedly related to that some operand
C alignments cause cache conflicts.

C The speed is limited by decoding/issue bandwidth.  There are 22 instructions
C in the loop, which corresponds to ceil(22/3)/4 = 1.83 c/l.

C INPUT PARAMETERS
define(`rp',`%rdi')
define(`up',`%rsi')
define(`vp',`%rdx')
define(`n', `%rcx')

ifdef(`OPERATION_addlsh1_n', `
  define(ADDSUB,	add)
  define(ADCSBB,	adc)
  define(func,		mpn_addlsh1_n)')
ifdef(`OPERATION_rsblsh1_n', `
  define(ADDSUB,	sub)
  define(ADCSBB,	sbb)
  define(func,		mpn_rsblsh1_n)')

MULFUNC_PROLOGUE(mpn_addlsh1_n mpn_rsblsh1_n)

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(func)
	FUNC_ENTRY(4)
	push	%rbp

	mov	(vp), %r8
	mov	R32(n), R32(%rax)
	lea	(rp,n,8), rp
	lea	(up,n,8), up
	lea	(vp,n,8), vp
	neg	n
	xor	R32(%rbp), R32(%rbp)
	and	$3, R32(%rax)
	je	L(b00)
	cmp	$2, R32(%rax)
	jc	L(b01)
	je	L(b10)

L(b11):	add	%r8, %r8
	mov	8(vp,n,8), %r9
	adc	%r9, %r9
	mov	16(vp,n,8), %r10
	adc	%r10, %r10
	sbb	R32(%rax), R32(%rax)	C save scy
	ADDSUB	(up,n,8), %r8
	ADCSBB	8(up,n,8), %r9
	mov	%r8, (rp,n,8)
	mov	%r9, 8(rp,n,8)
	ADCSBB	16(up,n,8), %r10
	mov	%r10, 16(rp,n,8)
	sbb	R32(%rbp), R32(%rbp)	C save acy
	add	$3, n
	jmp	L(ent)

L(b10):	add	%r8, %r8
	mov	8(vp,n,8), %r9
	adc	%r9, %r9
	sbb	R32(%rax), R32(%rax)	C save scy
	ADDSUB	(up,n,8), %r8
	ADCSBB	8(up,n,8), %r9
	mov	%r8, (rp,n,8)
	mov	%r9, 8(rp,n,8)
	sbb	R32(%rbp), R32(%rbp)	C save acy
	add	$2, n
	jmp	L(ent)

L(b01):	add	%r8, %r8
	sbb	R32(%rax), R32(%rax)	C save scy
	ADDSUB	(up,n,8), %r8
	mov	%r8, (rp,n,8)
	sbb	R32(%rbp), R32(%rbp)	C save acy
	inc	n
L(ent):	jns	L(end)

	ALIGN(16)
L(top):	add	R32(%rax), R32(%rax)	C restore scy

	mov	(vp,n,8), %r8
L(b00):	adc	%r8, %r8
	mov	8(vp,n,8), %r9
	adc	%r9, %r9
	mov	16(vp,n,8), %r10
	adc	%r10, %r10
	mov	24(vp,n,8), %r11
	adc	%r11, %r11

	sbb	R32(%rax), R32(%rax)	C save scy
	add	R32(%rbp), R32(%rbp)	C restore acy

	ADCSBB	(up,n,8), %r8
	nop				C Hammer speedup!
	ADCSBB	8(up,n,8), %r9
	mov	%r8, (rp,n,8)
	mov	%r9, 8(rp,n,8)
	ADCSBB	16(up,n,8), %r10
	ADCSBB	24(up,n,8), %r11
	mov	%r10, 16(rp,n,8)
	mov	%r11, 24(rp,n,8)

	sbb	R32(%rbp), R32(%rbp)	C save acy
	add	$4, n
	js	L(top)

L(end):
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
