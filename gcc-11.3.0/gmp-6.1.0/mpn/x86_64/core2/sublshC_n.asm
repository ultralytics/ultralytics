dnl  AMD64 mpn_sublshC_n -- rp[] = up[] - (vp[] << 1), optimised for Core 2 and
dnl  Core iN.

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2008, 2010-2012 Free Software Foundation, Inc.

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

C	     cycles/limb
C AMD K8,K9	 4.25
C AMD K10	 ?
C Intel P4	 ?
C Intel core2	 3
C Intel NHM	 3.1
C Intel SBR	 2.47
C Intel atom	 ?
C VIA nano	 ?

C INPUT PARAMETERS
define(`rp',`%rdi')
define(`up',`%rsi')
define(`vp',`%rdx')
define(`n', `%rcx')

ASM_START()
	TEXT
	ALIGN(8)
PROLOGUE(func)
	FUNC_ENTRY(4)
	push	%rbx
	push	%r12

	mov	R32(%rcx), R32(%rax)
	lea	24(up,n,8), up
	lea	24(vp,n,8), vp
	lea	24(rp,n,8), rp
	neg	n

	xor	R32(%r11), R32(%r11)

	mov	-24(vp,n,8), %r8	C do first limb early
	shrd	$RSH, %r8, %r11

	and	$3, R32(%rax)
	je	L(b0)
	cmp	$2, R32(%rax)
	jc	L(b1)
	je	L(b2)

L(b3):	mov	-16(vp,n,8), %r9
	shrd	$RSH, %r9, %r8
	mov	-8(vp,n,8), %r10
	shrd	$RSH, %r10, %r9
	mov	-24(up,n,8), %r12
	ADDSUB	%r11, %r12
	mov	%r12, -24(rp,n,8)
	mov	-16(up,n,8), %r12
	ADCSBB	%r8, %r12
	mov	%r12, -16(rp,n,8)
	mov	-8(up,n,8), %r12
	ADCSBB	%r9, %r12
	mov	%r12, -8(rp,n,8)
	mov	%r10, %r11
	sbb	R32(%rax), R32(%rax)	C save cy
	add	$3, n
	js	L(top)
	jmp	L(end)

L(b1):	mov	-24(up,n,8), %r12
	ADDSUB	%r11, %r12
	mov	%r12, -24(rp,n,8)
	mov	%r8, %r11
	sbb	R32(%rax), R32(%rax)	C save cy
	inc	n
	js	L(top)
	jmp	L(end)

L(b2):	mov	-16(vp,n,8), %r9
	shrd	$RSH, %r9, %r8
	mov	-24(up,n,8), %r12
	ADDSUB	%r11, %r12
	mov	%r12, -24(rp,n,8)
	mov	-16(up,n,8), %r12
	ADCSBB	%r8, %r12
	mov	%r12, -16(rp,n,8)
	mov	%r9, %r11
	sbb	R32(%rax), R32(%rax)	C save cy
	add	$2, n
	js	L(top)
	jmp	L(end)

	ALIGN(16)
L(top):	mov	-24(vp,n,8), %r8
	shrd	$RSH, %r8, %r11
L(b0):	mov	-16(vp,n,8), %r9
	shrd	$RSH, %r9, %r8
	mov	-8(vp,n,8), %r10
	shrd	$RSH, %r10, %r9
	mov	(vp,n,8), %rbx
	shrd	$RSH, %rbx, %r10

	add	R32(%rax), R32(%rax)	C restore cy

	mov	-24(up,n,8), %r12
	ADCSBB	%r11, %r12
	mov	%r12, -24(rp,n,8)

	mov	-16(up,n,8), %r12
	ADCSBB	%r8, %r12
	mov	%r12, -16(rp,n,8)

	mov	-8(up,n,8), %r12
	ADCSBB	%r9, %r12
	mov	%r12, -8(rp,n,8)

	mov	(up,n,8), %r12
	ADCSBB	%r10, %r12
	mov	%r12, (rp,n,8)

	mov	%rbx, %r11
	sbb	R32(%rax), R32(%rax)	C save cy

	add	$4, n
	js	L(top)

L(end):	shr	$RSH, %r11
	pop	%r12
	pop	%rbx
	sub	R32(%r11), R32(%rax)
	neg	R32(%rax)
	FUNC_EXIT()
	ret
EPILOGUE()
