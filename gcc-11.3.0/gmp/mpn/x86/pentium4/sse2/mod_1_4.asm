dnl  x86-32 mpn_mod_1s_4p for Pentium 4 and P6 models with SSE2 (i.e. 9,D,E,F).

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
C  * Optimize post-loop reduction code.
C  * Write a cps function that uses sse2 insns.

C			    cycles/limb
C P6 model 0-8,10-12		-
C P6 model 9   (Banias)		?
C P6 model 13  (Dothan)		3.4
C P4 model 0-1 (Willamette)	?
C P4 model 2   (Northwood)	4
C P4 model 3-4 (Prescott)	4.5

C INPUT PARAMETERS
C ap		sp + 4
C n		sp + 8
C b		sp + 12
C cps		sp + 16

define(`B1modb', `%mm1')
define(`B2modb', `%mm2')
define(`B3modb', `%mm3')
define(`B4modb', `%mm4')
define(`B5modb', `%mm5')
define(`ap',     `%edx')
define(`n',      `%eax')

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_mod_1s_4p)
	push	%ebx
	mov	8(%esp), ap
	mov	12(%esp), n
	mov	20(%esp), %ecx

	movd	8(%ecx), B1modb
	movd	12(%ecx), B2modb
	movd	16(%ecx), B3modb
	movd	20(%ecx), B4modb
	movd	24(%ecx), B5modb

	mov	n, %ebx
	lea	-4(ap,n,4), ap
	and	$3, %ebx
	je	L(b0)
	cmp	$2, %ebx
	jc	L(b1)
	je	L(b2)

L(b3):	movd	-4(ap), %mm7
	pmuludq	B1modb, %mm7
	movd	-8(ap), %mm6
	paddq	%mm6, %mm7
	movd	(ap), %mm6
	pmuludq	B2modb, %mm6
	paddq	%mm6, %mm7
	lea	-24(ap), ap
	add	$-3, n
	jz	L(end)
	jmp	L(top)

L(b0):	movd	-8(ap), %mm7
	pmuludq	B1modb, %mm7
	movd	-12(ap), %mm6
	paddq	%mm6, %mm7
	movd	-4(ap), %mm6
	pmuludq	B2modb, %mm6
	paddq	%mm6, %mm7
	movd	(ap), %mm6
	pmuludq	B3modb, %mm6
	paddq	%mm6, %mm7
	lea	-28(ap), ap
	add	$-4, n
	jz	L(end)
	jmp	L(top)

L(b1):	movd	(ap), %mm7
	lea	-16(ap), ap
	dec	n
	jz	L(x)
	jmp	L(top)

L(b2):	movd	-4(ap), %mm7		C rl
	punpckldq (ap), %mm7		C rh
	lea	-20(ap), ap
	add	$-2, n
	jz	L(end)

	ALIGN(8)
L(top):	movd	4(ap), %mm0
	pmuludq	B1modb, %mm0
	movd	0(ap), %mm6
	paddq	%mm6, %mm0

	movd	8(ap), %mm6
	pmuludq	B2modb, %mm6
	paddq	%mm6, %mm0

	movd	12(ap), %mm6
	pmuludq	B3modb, %mm6
	paddq	%mm6, %mm0

	movq	%mm7, %mm6
	psrlq	$32, %mm7		C rh
	pmuludq	B5modb, %mm7
	pmuludq	B4modb, %mm6

	paddq	%mm0, %mm7
	paddq	%mm6, %mm7

	add	$-16, ap
	add	$-4, n
	jnz	L(top)

L(end):	pcmpeqd	%mm4, %mm4
	psrlq	$32, %mm4		C 0x00000000FFFFFFFF
	pand	%mm7, %mm4		C rl
	psrlq	$32, %mm7		C rh
	pmuludq	B1modb, %mm7		C rh,cl
	paddq	%mm4, %mm7		C rh,rl
L(x):	movd	4(%ecx), %mm4		C cnt
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

	ALIGN(16)
PROLOGUE(mpn_mod_1s_4p_cps)
C CAUTION: This is the same code as in k7/mod_1_4.asm
	push	%ebp
	push	%edi
	push	%esi
	push	%ebx
	mov	20(%esp), %ebp		C FIXME: avoid bp for 0-idx
	mov	24(%esp), %ebx
	bsr	%ebx, %ecx
	xor	$31, %ecx
	sal	%cl, %ebx		C b << cnt
	mov	%ebx, %edx
	not	%edx
	mov	$-1, %eax
	div	%ebx
	xor	%edi, %edi
	sub	%ebx, %edi
	mov	$1, %esi
	mov	%eax, (%ebp)		C store bi
	mov	%ecx, 4(%ebp)		C store cnt
	shld	%cl, %eax, %esi
	imul	%edi, %esi
	mov	%eax, %edi
	mul	%esi

	add	%esi, %edx
	shr	%cl, %esi
	mov	%esi, 8(%ebp)		C store B1modb

	not	%edx
	imul	%ebx, %edx
	lea	(%edx,%ebx), %esi
	cmp	%edx, %eax
	cmovnc(	%edx, %esi)
	mov	%edi, %eax
	mul	%esi

	add	%esi, %edx
	shr	%cl, %esi
	mov	%esi, 12(%ebp)		C store B2modb

	not	%edx
	imul	%ebx, %edx
	lea	(%edx,%ebx), %esi
	cmp	%edx, %eax
	cmovnc(	%edx, %esi)
	mov	%edi, %eax
	mul	%esi

	add	%esi, %edx
	shr	%cl, %esi
	mov	%esi, 16(%ebp)		C store B3modb

	not	%edx
	imul	%ebx, %edx
	lea	(%edx,%ebx), %esi
	cmp	%edx, %eax
	cmovnc(	%edx, %esi)
	mov	%edi, %eax
	mul	%esi

	add	%esi, %edx
	shr	%cl, %esi
	mov	%esi, 20(%ebp)		C store B4modb

	not	%edx
	imul	%ebx, %edx
	add	%edx, %ebx
	cmp	%edx, %eax
	cmovnc(	%edx, %ebx)

	shr	%cl, %ebx
	mov	%ebx, 24(%ebp)		C store B5modb

	pop	%ebx
	pop	%esi
	pop	%edi
	pop	%ebp
	ret
EPILOGUE()
