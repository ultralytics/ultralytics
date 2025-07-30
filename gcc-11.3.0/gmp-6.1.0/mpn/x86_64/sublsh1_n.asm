dnl  AMD64 mpn_sublsh1_n -- rp[] = up[] - (vp[] << 1)

dnl  Copyright 2003, 2005-2007, 2011, 2012 Free Software Foundation, Inc.

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
C AMD K8,K9	 2.2
C AMD K10	 2.2
C Intel P4	12.75
C Intel core2	 3.45
C Intel corei	 ?
C Intel atom	 ?
C VIA nano	 3.25

C Sometimes speed degenerates, supposedly related to that some operand
C alignments cause cache conflicts.

C The speed is limited by decoding/issue bandwidth.  There are 26 instructions
C in the loop, which corresponds to 26/3/4 = 2.167 c/l.

C INPUT PARAMETERS
define(`rp',`%rdi')
define(`up',`%rsi')
define(`vp',`%rdx')
define(`n', `%rcx')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_sublsh1_n)
	FUNC_ENTRY(4)
	push	%rbx
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
	mov	(up,n,8), %rbp
	mov	8(up,n,8), %rbx
	sub	%r8, %rbp
	sbb	%r9, %rbx
	mov	%rbp, (rp,n,8)
	mov	%rbx, 8(rp,n,8)
	mov	16(up,n,8), %rbp
	sbb	%r10, %rbp
	mov	%rbp, 16(rp,n,8)
	sbb	R32(%rbp), R32(%rbp)	C save acy
	add	$3, n
	jmp	L(ent)

L(b10):	add	%r8, %r8
	mov	8(vp,n,8), %r9
	adc	%r9, %r9
	sbb	R32(%rax), R32(%rax)	C save scy
	mov	(up,n,8), %rbp
	mov	8(up,n,8), %rbx
	sub	%r8, %rbp
	sbb	%r9, %rbx
	mov	%rbp, (rp,n,8)
	mov	%rbx, 8(rp,n,8)
	sbb	R32(%rbp), R32(%rbp)	C save acy
	add	$2, n
	jmp	L(ent)

L(b01):	add	%r8, %r8
	sbb	R32(%rax), R32(%rax)	C save scy
	mov	(up,n,8), %rbp
	sub	%r8, %rbp
	mov	%rbp, (rp,n,8)
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

	mov	(up,n,8), %rbp
	mov	8(up,n,8), %rbx
	sbb	%r8, %rbp
	sbb	%r9, %rbx
	mov	%rbp, (rp,n,8)
	mov	%rbx, 8(rp,n,8)
	mov	16(up,n,8), %rbp
	mov	24(up,n,8), %rbx
	sbb	%r10, %rbp
	sbb	%r11, %rbx
	mov	%rbp, 16(rp,n,8)
	mov	%rbx, 24(rp,n,8)

	sbb	R32(%rbp), R32(%rbp)	C save acy
	add	$4, n
	js	L(top)

L(end):	add	R32(%rbp), R32(%rax)
	neg	R32(%rax)

	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
