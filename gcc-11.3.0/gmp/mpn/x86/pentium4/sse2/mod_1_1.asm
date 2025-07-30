dnl  x86-32 mpn_mod_1_1p for Pentium 4 and P6 models with SSE2 (i.e., 9,D,E,F).

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2009, 2010 Free Software Foundation, Inc.

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

C TODO:
C  * Optimize.  The present code was written quite straightforwardly.
C  * Optimize post-loop reduction code; it is from mod_1s_4p, thus overkill.
C  * Write a cps function that uses sse2 insns.

C                           cycles/limb
C P6 model 0-8,10-12		-
C P6 model 9   (Banias)		?
C P6 model 13  (Dothan)		?
C P4 model 0-1 (Willamette)	?
C P4 model 2   (Northwood)     16
C P4 model 3-4 (Prescott)      18

C INPUT PARAMETERS
C ap		sp + 4
C n		sp + 8
C b		sp + 12
C cps		sp + 16

define(`B1modb', `%mm1')
define(`B2modb', `%mm2')
define(`ap',     `%edx')
define(`n',      `%eax')

	TEXT
	ALIGN(16)
PROLOGUE(mpn_mod_1_1p)
	push	%ebx
	mov	8(%esp), ap
	mov	12(%esp), n
	mov	20(%esp), %ecx
	movd	8(%ecx), B1modb
	movd	12(%ecx), B2modb

	lea	-4(ap,n,4), ap

C FIXME: See comment in generic/mod_1_1.c.
	movd	(ap), %mm7
	movd	-4(ap), %mm4
	pmuludq B1modb, %mm7
	paddq	%mm4, %mm7
	add	$-2, n
	jz	L(end)

	ALIGN(8)
L(top):	movq	%mm7, %mm6
	psrlq	$32, %mm7		C rh
	movd	-8(ap), %mm0
	add	$-4, ap
	pmuludq	B2modb, %mm7
	pmuludq	B1modb, %mm6
	add	$-1, n
	paddq	%mm0, %mm7
	paddq	%mm6, %mm7
	jnz	L(top)

L(end):	pcmpeqd	%mm4, %mm4
	psrlq	$32, %mm4		C 0x00000000FFFFFFFF
	pand	%mm7, %mm4		C rl
	psrlq	$32, %mm7		C rh
	pmuludq	B1modb, %mm7		C rh,cl
	paddq	%mm4, %mm7		C rh,rl
	movd	4(%ecx), %mm4		C cnt
	psllq	%mm4, %mm7		C rh,rl normalized
	movq	%mm7, %mm2		C rl in low half
	psrlq	$32, %mm7		C rh
	movd	(%ecx), %mm1		C bi
	pmuludq	%mm7, %mm1		C qh,ql
	paddq	%mm2, %mm1		C qh-1,ql
	movd	%mm1, %ecx		C ql
	psrlq	$32, %mm1		C qh-1
	movd	16(%esp), %mm3		C b
	pmuludq	%mm1, %mm3		C (qh-1) * b
	psubq	%mm3, %mm2		C r in low half (could use psubd)
	movd	%mm2, %eax		C r
	mov	16(%esp), %ebx
	sub	%ebx, %eax		C r
	cmp	%eax, %ecx
	lea	(%eax,%ebx), %edx
	cmovc(	%edx, %eax)
	movd	%mm4, %ecx		C cnt
	cmp	%ebx, %eax
	jae	L(fix)
	emms
	pop	%ebx
	shr	%cl, %eax
	ret

L(fix):	sub	%ebx, %eax
	emms
	pop	%ebx
	shr	%cl, %eax
	ret
EPILOGUE()

PROLOGUE(mpn_mod_1_1p_cps)
C CAUTION: This is the same code as in k7/mod_1_1.asm
	push	%ebp
	mov	12(%esp), %ebp
	push	%esi
	bsr	%ebp, %ecx
	push	%ebx
	xor	$31, %ecx
	mov	16(%esp), %esi
	sal	%cl, %ebp
	mov	%ebp, %edx
	not	%edx
	mov	$-1, %eax
	div	%ebp
	mov	%eax, (%esi)		C store bi
	mov	%ecx, 4(%esi)		C store cnt
	xor	%ebx, %ebx
	sub	%ebp, %ebx
	mov	$1, %edx
	shld	%cl, %eax, %edx
	imul	%edx, %ebx
	mul	%ebx
	add	%ebx, %edx
	not	%edx
	imul	%ebp, %edx
	add	%edx, %ebp
	cmp	%edx, %eax
	cmovc(	%ebp, %edx)
	shr	%cl, %ebx
	mov	%ebx, 8(%esi)		C store B1modb
	shr	%cl, %edx
	mov	%edx, 12(%esi)		C store B2modb
	pop	%ebx
	pop	%esi
	pop	%ebp
	ret
EPILOGUE()
