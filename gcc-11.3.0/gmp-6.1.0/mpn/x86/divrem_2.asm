dnl  x86 mpn_divrem_2 -- Divide an mpn number by a normalized 2-limb number.

dnl  Copyright 2007, 2008 Free Software Foundation, Inc.

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


C		norm	frac
C 486
C P5
C P6-13		29.2
C P6-15		*26
C K6
C K7		22
C K8		*19
C P4-f1
C P4-f2		*65
C P4-f3
C P4-f4		*72

C A star means numbers not updated for the latest version of the code.


C TODO
C  * Perhaps keep ecx or esi in stack slot, freeing up a reg for q0.
C  * The loop has not been carefully tuned.  We should at the very least do
C    some local insn swapping.
C  * The code outside the main loop is what gcc generated.  Clean up!
C  * Clean up stack slot usage.

C INPUT PARAMETERS
C qp
C fn
C up_param
C un_param
C dp


C eax ebx ecx edx esi edi ebp
C         cnt         qp

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_divrem_2)
	push	%ebp
	push	%edi
	push	%esi
	push	%ebx
	sub	$36, %esp
	mov	68(%esp), %ecx		C un
	mov	72(%esp), %esi		C dp
	movl	$0, 32(%esp)
	lea	0(,%ecx,4), %edi
	add	64(%esp), %edi		C up
	mov	(%esi), %ebx
	mov	4(%esi), %eax
	mov	%ebx, 20(%esp)
	sub	$12, %edi
	mov	%eax, 24(%esp)
	mov	%edi, 12(%esp)
	mov	8(%edi), %ebx
	mov	4(%edi), %ebp
	cmp	%eax, %ebx
	jb	L(8)
	seta	%dl
	cmp	20(%esp), %ebp
	setae	%al
	orb	%dl, %al		C "orb" form to placate Sun tools
	jne	L(35)
L(8):
	mov	60(%esp), %esi		C fn
	lea	-3(%esi,%ecx), %edi
	test	%edi, %edi
	js	L(9)
	mov	24(%esp), %edx
	mov	$-1, %esi
	mov	%esi, %eax
	mov	%esi, %ecx
	not	%edx
	divl	24(%esp)
	mov	%eax, %esi
	imul	24(%esp), %eax
	mov	%eax, (%esp)
	mov	%esi, %eax
	mull	20(%esp)
	mov	(%esp), %eax
	add	20(%esp), %eax
	adc	$0, %ecx
	add	%eax, %edx
	adc	$0, %ecx
	mov	%ecx, %eax
	js	L(32)
L(36):	dec	%esi
	sub	24(%esp), %edx
	sbb	$0, %eax
	jns	L(36)
L(32):
	mov	%esi, 16(%esp)		C di
	mov	%edi, %ecx		C un
	mov	12(%esp), %esi		C up
	mov	24(%esp), %eax
	neg	%eax
	mov	%eax, 4(%esp)		C -d1
	ALIGN(16)
	nop

C eax ebx ecx edx esi edi ebp  0    4   8   12  16  20  24  28  32   56  60
C     n2  un      up      n1   q0  -d1          di  d0  d1      msl  qp  fn

L(loop):
	mov	16(%esp), %eax		C di
	mul	%ebx
	add	%ebp, %eax
	mov	%eax, (%esp)		C q0
	adc	%ebx, %edx
	mov	%edx, %edi		C q
	imul	4(%esp), %edx
	mov	20(%esp), %eax
	lea	(%edx, %ebp), %ebx	C n1 -= ...
	mul	%edi
	xor	%ebp, %ebp
	cmp	60(%esp), %ecx
	jl	L(19)
	mov	(%esi), %ebp
	sub	$4, %esi
L(19):	sub	20(%esp), %ebp
	sbb	24(%esp), %ebx
	sub	%eax, %ebp
	sbb	%edx, %ebx
	mov	20(%esp), %eax		C d1
	inc	%edi
	xor	%edx, %edx
	cmp	(%esp), %ebx
	adc	$-1, %edx		C mask
	add	%edx, %edi		C q--
	and	%edx, %eax		C d0 or 0
	and	24(%esp), %edx		C d1 or 0
	add	%eax, %ebp
	adc	%edx, %ebx
	cmp	24(%esp), %ebx
	jae	L(fix)
L(bck):	mov	56(%esp), %edx
	mov	%edi, (%edx, %ecx, 4)
	dec	%ecx
	jns	L(loop)

L(9):	mov	64(%esp), %esi		C up
	mov	%ebp, (%esi)
	mov	%ebx, 4(%esi)
	mov	32(%esp), %eax
	add	$36, %esp
	pop	%ebx
	pop	%esi
	pop	%edi
	pop	%ebp
	ret

L(fix):	seta	%dl
	cmp	20(%esp), %ebp
	setae	%al
	orb	%dl, %al		C "orb" form to placate Sun tools
	je	L(bck)
	inc	%edi
	sub	20(%esp), %ebp
	sbb	24(%esp), %ebx
	jmp	L(bck)

L(35):	sub	20(%esp), %ebp
	sbb	24(%esp), %ebx
	movl	$1, 32(%esp)
	jmp	L(8)
EPILOGUE()
