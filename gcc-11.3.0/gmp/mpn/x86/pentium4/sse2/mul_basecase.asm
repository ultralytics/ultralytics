dnl  mpn_mul_basecase for Pentium 4 and P6 models with SSE2 (i.e., 9,D,E,F).

dnl  Copyright 2001, 2002, 2005, 2007 Free Software Foundation, Inc.

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
C  * Improve ad-hoc outer loop code and register handling.  Some feed-in
C    scheduling could improve things by several cycles per outer iteration.
C  * In code for un <= 3, try keeping accumulation operands in registers,
C    without storing intermediates to rp.
C  * We might want to keep 32 in a free mm register, since the register form is
C    3 bytes and the immediate form is 4 bytes.  About 70 bytes to save.
C  * Look into different loop alignment, we now expand the code about 50 bytes
C    with possibly needless alignment.
C  * Perhaps rewrap loops 00,01,02 (6 loops) to allow fall-through entry.
C  * Use OSP, should solve feed-in latency problems.
C  * Save a few tens of bytes by doing cross-jumping for Loel0, etc.
C  * Save around 120 bytes by remapping "m 0", "m 1", "m 2" and "m 3" registers
C    so that they can share feed-in code, and changing the branch targets from
C    L<n> to Lm<nn>.

C                           cycles/limb
C P6 model 9   (Banias)         ?
C P6 model 13  (Dothan)         5.24
C P6 model 14  (Yonah)          ?
C P4 model 0-1 (Willamette):    5
C P4 model 2   (Northwood):     4.60 at 32 limbs
C P4 model 3-4 (Prescott):      4.94 at 32 limbs

C INPUT PARAMETERS
C rp		sp + 4
C up		sp + 8
C un		sp + 12
C vp		sp + 16
C vn		sp + 20

	TEXT
	ALIGN(16)
PROLOGUE(mpn_mul_basecase)
	push	%esi
	push	%ebx
	mov	12(%esp), %edx		C rp
	mov	16(%esp), %eax		C up
	mov	20(%esp), %ecx		C un
	mov	24(%esp), %esi		C vp
	mov	28(%esp), %ebx		C vn
	movd	(%esi), %mm7		C
L(ent):	cmp	$3, %ecx
	ja	L(big)
	movd	(%eax), %mm6
	pmuludq	%mm7, %mm6
	jz	L(un3)
	cmp	$2, %ecx
	jz	L(un2)

L(un1):	movd	%mm6, (%edx)		C				un=1
	psrlq	$32, %mm6		C				un=1
	movd	%mm6, 4(%edx)		C				un=1
	jmp	L(rtr)			C				un=1

L(un2):	movd	4(%eax), %mm1		C				un=2
	pmuludq	%mm7, %mm1		C				un=2
	movd	%mm6, (%edx)		C				un=2
	psrlq	$32, %mm6		C				un=2
	paddq	%mm1, %mm6		C				un=2
	movd	%mm6, 4(%edx)		C				un=2
	psrlq	$32, %mm6		C				un=2
	movd	%mm6, 8(%edx)		C				un=2
      dec	%ebx			C				un=2
      jz	L(rtr)			C				un=2
	movd	4(%esi), %mm7		C				un=2
	movd	(%eax), %mm6		C				un=2
	pmuludq	%mm7, %mm6		C				un=2
	movd	4(%eax), %mm1		C				un=2
	movd	4(%edx), %mm4		C				un=2
	pmuludq	%mm7, %mm1		C				un=2
	movd	8(%edx), %mm5		C				un=2
	paddq	%mm4, %mm6		C				un=2
	paddq	%mm1, %mm5		C				un=2
	movd	%mm6, 4(%edx)		C				un=2
	psrlq	$32, %mm6		C				un=2
	paddq	%mm5, %mm6		C				un=2
	movd	%mm6, 8(%edx)		C				un=2
	psrlq	$32, %mm6		C				un=2
	movd	%mm6, 12(%edx)		C				un=2
L(rtr):	emms
	pop	%ebx
	pop	%esi
	ret

L(un3):	movd	4(%eax), %mm1		C				un=3
	pmuludq	%mm7, %mm1		C				un=3
	movd	8(%eax), %mm2		C				un=3
	pmuludq	%mm7, %mm2		C				un=3
	movd	%mm6, (%edx)		C				un=3
	psrlq	$32, %mm6		C				un=3
	paddq	%mm1, %mm6		C				un=3
	movd	%mm6, 4(%edx)		C				un=3
	psrlq	$32, %mm6		C				un=3
	paddq	%mm2, %mm6		C				un=3
	movd	%mm6, 8(%edx)		C				un=3
	psrlq	$32, %mm6		C				un=3
	movd	%mm6, 12(%edx)		C				un=3
      dec	%ebx			C				un=3
      jz	L(rtr)			C				un=3
	movd	4(%esi), %mm7		C				un=3
	movd	(%eax), %mm6		C				un=3
	pmuludq	%mm7, %mm6		C				un=3
	movd	4(%eax), %mm1		C				un=3
	movd	4(%edx), %mm4		C				un=3
	pmuludq	%mm7, %mm1		C				un=3
	movd	8(%eax), %mm2		C				un=3
	movd	8(%edx), %mm5		C				un=3
	pmuludq	%mm7, %mm2		C				un=3
	paddq	%mm4, %mm6		C				un=3
	paddq	%mm1, %mm5		C				un=3
	movd	12(%edx), %mm4		C				un=3
	movd	%mm6, 4(%edx)		C				un=3
	psrlq	$32, %mm6		C				un=3
	paddq	%mm5, %mm6		C				un=3
	paddq	%mm2, %mm4		C				un=3
	movd	%mm6, 8(%edx)		C				un=3
	psrlq	$32, %mm6		C				un=3
	paddq	%mm4, %mm6		C				un=3
	movd	%mm6, 12(%edx)		C				un=3
	psrlq	$32, %mm6		C				un=3
	movd	%mm6, 16(%edx)		C				un=3
      dec	%ebx			C				un=3
      jz	L(rtr)			C				un=3
	movd	8(%esi), %mm7		C				un=3
	movd	(%eax), %mm6		C				un=3
	pmuludq	%mm7, %mm6		C				un=3
	movd	4(%eax), %mm1		C				un=3
	movd	8(%edx), %mm4		C				un=3
	pmuludq	%mm7, %mm1		C				un=3
	movd	8(%eax), %mm2		C				un=3
	movd	12(%edx), %mm5		C				un=3
	pmuludq	%mm7, %mm2		C				un=3
	paddq	%mm4, %mm6		C				un=3
	paddq	%mm1, %mm5		C				un=3
	movd	16(%edx), %mm4		C				un=3
	movd	%mm6, 8(%edx)		C				un=3
	psrlq	$32, %mm6		C				un=3
	paddq	%mm5, %mm6		C				un=3
	paddq	%mm2, %mm4		C				un=3
	movd	%mm6, 12(%edx)		C				un=3
	psrlq	$32, %mm6		C				un=3
	paddq	%mm4, %mm6		C				un=3
	movd	%mm6, 16(%edx)		C				un=3
	psrlq	$32, %mm6		C				un=3
	movd	%mm6, 20(%edx)		C				un=3
	jmp	L(rtr)


L(big):	push	%edi
	pxor	%mm6, %mm6
	lea	4(%esi), %esi
	and	$3, %ecx
	jz	L(0)
	cmp	$2, %ecx
	jc	L(1)
	jz	L(2)
	jmp	L(3)			C FIXME: one case should fall through


L(0):	movd	(%eax), %mm3		C				m 0
	sub	24(%esp), %ecx		C inner loop count		m 0
	mov	%ecx, 24(%esp)		C update loop count for later	m 0
	pmuludq	%mm7, %mm3		C				m 0
	movd	4(%eax), %mm0		C				m 0
	pmuludq	%mm7, %mm0		C				m 0
	movd	8(%eax), %mm1		C				m 0
	jmp	L(m00)			C				m 0
	ALIGN(16)			C				m 0
L(lpm0):
	pmuludq	%mm7, %mm4		C				m 0
	paddq	%mm0, %mm6		C				m 0
	movd	(%eax), %mm3		C				m 0
	movd	%mm6, -12(%edx)		C				m 0
	psrlq	$32, %mm6		C				m 0
	pmuludq	%mm7, %mm3		C				m 0
	paddq	%mm1, %mm6		C				m 0
	movd	4(%eax), %mm0		C				m 0
	movd	%mm6, -8(%edx)		C				m 0
	psrlq	$32, %mm6		C				m 0
	pmuludq	%mm7, %mm0		C				m 0
	paddq	%mm4, %mm6		C				m 0
	movd	8(%eax), %mm1		C				m 0
	movd	%mm6, -4(%edx)		C				m 0
	psrlq	$32, %mm6		C				m 0
L(m00):	pmuludq	%mm7, %mm1		C				m 0
	paddq	%mm3, %mm6		C				m 0
	movd	12(%eax), %mm4		C				m 0
	movd	%mm6, (%edx)		C				m 0
	psrlq	$32, %mm6		C				m 0
	lea	16(%eax), %eax		C				m 0
	lea	16(%edx), %edx		C				m 0
	add	$4, %ecx		C				m 0
	ja	L(lpm0)			C				m 0
	pmuludq	%mm7, %mm4		C				m 0
	paddq	%mm0, %mm6		C				m 0
	movd	%mm6, -12(%edx)		C				m 0
	psrlq	$32, %mm6		C				m 0
	paddq	%mm1, %mm6		C				m 0
	mov	16(%esp), %edi		C rp				  0
	jmp	L(x0)

L(olp0):
	lea	4(%edi), %edi		C				am 0
	movd	(%esi), %mm7		C				am 0
	lea	4(%esi), %esi		C				am 0
	mov	%edi, %edx		C rp				am 0
	mov	20(%esp), %eax		C up				am 0
	movd	(%eax), %mm3		C				am 0
	mov	24(%esp), %ecx		C inner loop count		am 0
	pxor	%mm6, %mm6		C				am 0
	pmuludq	%mm7, %mm3		C				am 0
	movd	4(%eax), %mm0		C				am 0
	movd	(%edx), %mm5		C				am 0
	pmuludq	%mm7, %mm0		C				am 0
	movd	8(%eax), %mm1		C				am 0
	paddq	%mm3, %mm5		C				am 0
	movd	4(%edx), %mm4		C				am 0
	jmp	L(am00)			C				am 0
	ALIGN(16)			C				mm 0
L(lam0):
	pmuludq	%mm7, %mm2		C				am 0
	paddq	%mm4, %mm6		C				am 0
	movd	(%eax), %mm3		C				am 0
	paddq	%mm1, %mm5		C				am 0
	movd	-4(%edx), %mm4		C				am 0
	movd	%mm6, -12(%edx)		C				am 0
	psrlq	$32, %mm6		C				am 0
	pmuludq	%mm7, %mm3		C				am 0
	paddq	%mm5, %mm6		C				am 0
	movd	4(%eax), %mm0		C				am 0
	paddq	%mm2, %mm4		C				am 0
	movd	(%edx), %mm5		C				am 0
	movd	%mm6, -8(%edx)		C				am 0
	psrlq	$32, %mm6		C				am 0
	pmuludq	%mm7, %mm0		C				am 0
	paddq	%mm4, %mm6		C				am 0
	movd	8(%eax), %mm1		C				am 0
	paddq	%mm3, %mm5		C				am 0
	movd	4(%edx), %mm4		C				am 0
	movd	%mm6, -4(%edx)		C				am 0
	psrlq	$32, %mm6		C				am 0
L(am00):
	pmuludq	%mm7, %mm1		C				am 0
	paddq	%mm5, %mm6		C				am 0
	movd	12(%eax), %mm2		C				am 0
	paddq	%mm0, %mm4		C				am 0
	movd	8(%edx), %mm5		C				am 0
	movd	%mm6, (%edx)		C				am 0
	psrlq	$32, %mm6		C				am 0
	lea	16(%eax), %eax		C				am 0
	lea	16(%edx), %edx		C				am 0
	add	$4, %ecx		C				am 0
	jnz	L(lam0)			C				am 0
	pmuludq	%mm7, %mm2		C				am 0
	paddq	%mm4, %mm6		C				am 0
	paddq	%mm1, %mm5		C				am 0
	movd	-4(%edx), %mm4		C				am 0
	movd	%mm6, -12(%edx)		C				am 0
	psrlq	$32, %mm6		C				am 0
	paddq	%mm5, %mm6		C				am 0
	paddq	%mm2, %mm4		C				am 0
L(x0):	movd	%mm6, -8(%edx)		C				am 0
	psrlq	$32, %mm6		C				am 0
	paddq	%mm4, %mm6		C				am 0
	movd	%mm6, -4(%edx)		C				am 0
	psrlq	$32, %mm6		C				am 0
	movd	%mm6, (%edx)		C				am 0
	dec	%ebx			C				am 0
	jnz	L(olp0)			C				am 0
L(oel0):
	emms				C				   0
	pop	%edi			C				   0
	pop	%ebx			C				   0
	pop	%esi			C				   0
	ret				C				   0


L(1):	movd	(%eax), %mm4		C				m 1
	sub	24(%esp), %ecx		C				m 1
	mov	%ecx, 24(%esp)		C update loop count for later	m 1
	pmuludq	%mm7, %mm4		C				m 1
	movd	4(%eax), %mm3		C				m 1
	pmuludq	%mm7, %mm3		C				m 1
	movd	8(%eax), %mm0		C				m 1
	jmp	L(m01)			C				m 1
	ALIGN(16)			C				m 1
L(lpm1):
	pmuludq	%mm7, %mm4		C				m 1
	paddq	%mm0, %mm6		C				m 1
	movd	4(%eax), %mm3		C				m 1
	movd	%mm6, -8(%edx)		C				m 1
	psrlq	$32, %mm6		C				m 1
	pmuludq	%mm7, %mm3		C				m 1
	paddq	%mm1, %mm6		C				m 1
	movd	8(%eax), %mm0		C				m 1
	movd	%mm6, -4(%edx)		C				m 1
	psrlq	$32, %mm6		C				m 1
L(m01):	pmuludq	%mm7, %mm0		C				m 1
	paddq	%mm4, %mm6		C				m 1
	movd	12(%eax), %mm1		C				m 1
	movd	%mm6, (%edx)		C				m 1
	psrlq	$32, %mm6		C				m 1
	pmuludq	%mm7, %mm1		C				m 1
	paddq	%mm3, %mm6		C				m 1
	movd	16(%eax), %mm4		C				m 1
	movd	%mm6, 4(%edx)		C				m 1
	psrlq	$32, %mm6		C				m 1
	lea	16(%eax), %eax		C				m 1
	lea	16(%edx), %edx		C				m 1
	add	$4, %ecx		C				m 1
	ja	L(lpm1)			C				m 1
	pmuludq	%mm7, %mm4		C				m 1
	paddq	%mm0, %mm6		C				m 1
	movd	%mm6, -8(%edx)		C				m 1
	psrlq	$32, %mm6		C				m 1
	paddq	%mm1, %mm6		C				m 1
	mov	16(%esp), %edi		C rp				  1
	jmp	L(x1)

L(olp1):
	lea	4(%edi), %edi		C				am 1
	movd	(%esi), %mm7		C				am 1
	lea	4(%esi), %esi		C				am 1
	mov	%edi, %edx		C rp				am 1
	mov	20(%esp), %eax		C up				am 1
	movd	(%eax), %mm2		C				am 1
	mov	24(%esp), %ecx		C inner loop count		am 1
	pxor	%mm6, %mm6		C				am 1
	pmuludq	%mm7, %mm2		C				am 1
	movd	4(%eax), %mm3		C				am 1
	movd	(%edx), %mm4		C				am 1
	pmuludq	%mm7, %mm3		C				am 1
	movd	8(%eax), %mm0		C				am 1
	paddq	%mm2, %mm4		C				am 1
	movd	4(%edx), %mm5		C				am 1
	jmp	L(am01)			C				am 1
	ALIGN(16)			C				am 1
L(lam1):
	pmuludq	%mm7, %mm2		C				am 1
	paddq	%mm4, %mm6		C				am 1
	movd	4(%eax), %mm3		C				am 1
	paddq	%mm1, %mm5		C				am 1
	movd	(%edx), %mm4		C				am 1
	movd	%mm6, -8(%edx)		C				am 1
	psrlq	$32, %mm6		C				am 1
	pmuludq	%mm7, %mm3		C				am 1
	paddq	%mm5, %mm6		C				am 1
	movd	8(%eax), %mm0		C				am 1
	paddq	%mm2, %mm4		C				am 1
	movd	4(%edx), %mm5		C				am 1
	movd	%mm6, -4(%edx)		C				am 1
	psrlq	$32, %mm6		C				am 1
L(am01):
	pmuludq	%mm7, %mm0		C				am 1
	paddq	%mm4, %mm6		C				am 1
	movd	12(%eax), %mm1		C				am 1
	paddq	%mm3, %mm5		C				am 1
	movd	8(%edx), %mm4		C				am 1
	movd	%mm6, (%edx)		C				am 1
	psrlq	$32, %mm6		C				am 1
	pmuludq	%mm7, %mm1		C				am 1
	paddq	%mm5, %mm6		C				am 1
	movd	16(%eax), %mm2		C				am 1
	paddq	%mm0, %mm4		C				am 1
	movd	12(%edx), %mm5		C				am 1
	movd	%mm6, 4(%edx)		C				am 1
	psrlq	$32, %mm6		C				am 1
	lea	16(%eax), %eax		C				am 1
	lea	16(%edx), %edx		C				am 1
	add	$4, %ecx		C				am 1
	jnz	L(lam1)			C				am 1
	pmuludq	%mm7, %mm2		C				am 1
	paddq	%mm4, %mm6		C				am 1
	paddq	%mm1, %mm5		C				am 1
	movd	(%edx), %mm4		C				am 1
	movd	%mm6, -8(%edx)		C				am 1
	psrlq	$32, %mm6		C				am 1
	paddq	%mm5, %mm6		C				am 1
	paddq	%mm2, %mm4		C				am 1
L(x1):	movd	%mm6, -4(%edx)		C				am 1
	psrlq	$32, %mm6		C				am 1
	paddq	%mm4, %mm6		C				am 1
	movd	%mm6, (%edx)		C				am 1
	psrlq	$32, %mm6		C				am 1
	movd	%mm6, 4(%edx)		C				am 1
	dec	%ebx			C				am 1
	jnz	L(olp1)			C				am 1
L(oel1):
	emms				C				   1
	pop	%edi			C				   1
	pop	%ebx			C				   1
	pop	%esi			C				   1
	ret				C				   1


L(2):	movd	(%eax), %mm1		C				m 2
	sub	24(%esp), %ecx		C				m 2
	mov	%ecx, 24(%esp)		C update loop count for later	m 2
	pmuludq	%mm7, %mm1		C				m 2
	movd	4(%eax), %mm4		C				m 2
	pmuludq	%mm7, %mm4		C				m 2
	movd	8(%eax), %mm3		C				m 2
	jmp	L(m10)			C				m 2
	ALIGN(16)			C				m 2
L(lpm2):
	pmuludq	%mm7, %mm4		C				m 2
	paddq	%mm0, %mm6		C				m 2
	movd	8(%eax), %mm3		C				m 2
	movd	%mm6, -4(%edx)		C				m 2
	psrlq	$32, %mm6		C				m 2
L(m10):	pmuludq	%mm7, %mm3		C				m 2
	paddq	%mm1, %mm6		C				m 2
	movd	12(%eax), %mm0		C				m 2
	movd	%mm6, (%edx)		C				m 2
	psrlq	$32, %mm6		C				m 2
	pmuludq	%mm7, %mm0		C				m 2
	paddq	%mm4, %mm6		C				m 2
	movd	16(%eax), %mm1		C				m 2
	movd	%mm6, 4(%edx)		C				m 2
	psrlq	$32, %mm6		C				m 2
	pmuludq	%mm7, %mm1		C				m 2
	paddq	%mm3, %mm6		C				m 2
	movd	20(%eax), %mm4		C				m 2
	movd	%mm6, 8(%edx)		C				m 2
	psrlq	$32, %mm6		C				m 2
	lea	16(%eax), %eax		C				m 2
	lea	16(%edx), %edx		C				m 2
	add	$4, %ecx		C				m 2
	ja	L(lpm2)			C				m 2
	pmuludq	%mm7, %mm4		C				m 2
	paddq	%mm0, %mm6		C				m 2
	movd	%mm6, -4(%edx)		C				m 2
	psrlq	$32, %mm6		C				m 2
	paddq	%mm1, %mm6		C				m 2
	mov	16(%esp), %edi		C rp				  2
	jmp	L(x2)

L(olp2):
	lea	4(%edi), %edi		C				am 2
	movd	(%esi), %mm7		C				am 2
	lea	4(%esi), %esi		C				am 2
	mov	%edi, %edx		C rp				am 2
	mov	20(%esp), %eax		C up				am 2
	movd	(%eax), %mm1		C				am 2
	mov	24(%esp), %ecx		C inner loop count		am 2
	pxor	%mm6, %mm6		C				am 2
	pmuludq	%mm7, %mm1		C				am 2
	movd	4(%eax), %mm2		C				am 2
	movd	(%edx), %mm5		C				am 2
	pmuludq	%mm7, %mm2		C				am 2
	movd	8(%eax), %mm3		C				am 2
	paddq	%mm1, %mm5		C				am 2
	movd	4(%edx), %mm4		C				am 2
	jmp	L(am10)			C				am 2
	ALIGN(16)			C				am 2
L(lam2):
	pmuludq	%mm7, %mm2		C				am 2
	paddq	%mm4, %mm6		C				am 2
	movd	8(%eax), %mm3		C				am 2
	paddq	%mm1, %mm5		C				am 2
	movd	4(%edx), %mm4		C				am 2
	movd	%mm6, -4(%edx)		C				am 2
	psrlq	$32, %mm6		C				am 2
L(am10):
	pmuludq	%mm7, %mm3		C				am 2
	paddq	%mm5, %mm6		C				am 2
	movd	12(%eax), %mm0		C				am 2
	paddq	%mm2, %mm4		C				am 2
	movd	8(%edx), %mm5		C				am 2
	movd	%mm6, (%edx)		C				am 2
	psrlq	$32, %mm6		C				am 2
	pmuludq	%mm7, %mm0		C				am 2
	paddq	%mm4, %mm6		C				am 2
	movd	16(%eax), %mm1		C				am 2
	paddq	%mm3, %mm5		C				am 2
	movd	12(%edx), %mm4		C				am 2
	movd	%mm6, 4(%edx)		C				am 2
	psrlq	$32, %mm6		C				am 2
	pmuludq	%mm7, %mm1		C				am 2
	paddq	%mm5, %mm6		C				am 2
	movd	20(%eax), %mm2		C				am 2
	paddq	%mm0, %mm4		C				am 2
	movd	16(%edx), %mm5		C				am 2
	movd	%mm6, 8(%edx)		C				am 2
	psrlq	$32, %mm6		C				am 2
	lea	16(%eax), %eax		C				am 2
	lea	16(%edx), %edx		C				am 2
	add	$4, %ecx		C				am 2
	jnz	L(lam2)			C				am 2
	pmuludq	%mm7, %mm2		C				am 2
	paddq	%mm4, %mm6		C				am 2
	paddq	%mm1, %mm5		C				am 2
	movd	4(%edx), %mm4		C				am 2
	movd	%mm6, -4(%edx)		C				am 2
	psrlq	$32, %mm6		C				am 2
	paddq	%mm5, %mm6		C				am 2
	paddq	%mm2, %mm4		C				am 2
L(x2):	movd	%mm6, (%edx)		C				am 2
	psrlq	$32, %mm6		C				am 2
	paddq	%mm4, %mm6		C				am 2
	movd	%mm6, 4(%edx)		C				am 2
	psrlq	$32, %mm6		C				am 2
	movd	%mm6, 8(%edx)		C				am 2
	dec	%ebx			C				am 2
	jnz	L(olp2)			C				am 2
L(oel2):
	emms				C				   2
	pop	%edi			C				   2
	pop	%ebx			C				   2
	pop	%esi			C				   2
	ret				C				   2


L(3):	movd	(%eax), %mm0		C				m 3
	sub	24(%esp), %ecx		C				m 3
	mov	%ecx, 24(%esp)		C update loop count for later	m 3
	pmuludq	%mm7, %mm0		C				m 3
	movd	4(%eax), %mm1		C				m 3
	pmuludq	%mm7, %mm1		C				m 3
	movd	8(%eax), %mm4		C				m 3
	jmp	L(lpm3)			C				m 3
	ALIGN(16)			C				m 3
L(lpm3):
	pmuludq	%mm7, %mm4		C				m 3
	paddq	%mm0, %mm6		C				m 3
	movd	12(%eax), %mm3		C				m 3
	movd	%mm6, (%edx)		C				m 3
	psrlq	$32, %mm6		C				m 3
	pmuludq	%mm7, %mm3		C				m 3
	paddq	%mm1, %mm6		C				m 3
	movd	16(%eax), %mm0		C				m 3
	movd	%mm6, 4(%edx)		C				m 3
	psrlq	$32, %mm6		C				m 3
	pmuludq	%mm7, %mm0		C				m 3
	paddq	%mm4, %mm6		C				m 3
	movd	20(%eax), %mm1		C				m 3
	movd	%mm6, 8(%edx)		C				m 3
	psrlq	$32, %mm6		C				m 3
	pmuludq	%mm7, %mm1		C				m 3
	paddq	%mm3, %mm6		C				m 3
	movd	24(%eax), %mm4		C				m 3
	movd	%mm6, 12(%edx)		C				m 3
	psrlq	$32, %mm6		C				m 3
	lea	16(%eax), %eax		C				m 3
	lea	16(%edx), %edx		C				m 3
	add	$4, %ecx		C				m 3
	ja	L(lpm3)			C				m 3
	pmuludq	%mm7, %mm4		C				m 3
	paddq	%mm0, %mm6		C				m 3
	movd	%mm6, (%edx)		C				m 3
	psrlq	$32, %mm6		C				m 3
	paddq	%mm1, %mm6		C				m 3
	mov	16(%esp), %edi		C rp				  3
	jmp	L(x3)

L(olp3):
	lea	4(%edi), %edi		C				am 3
	movd	(%esi), %mm7		C				am 3
	lea	4(%esi), %esi		C				am 3
	mov	%edi, %edx		C rp				am 3
	mov	20(%esp), %eax		C up				am 3
	movd	(%eax), %mm0		C				am 3
	mov	24(%esp), %ecx		C inner loop count		am 3
	pxor	%mm6, %mm6		C				am 3
	pmuludq	%mm7, %mm0		C				am 3
	movd	4(%eax), %mm1		C				am 3
	movd	(%edx), %mm4		C				am 3
	pmuludq	%mm7, %mm1		C				am 3
	movd	8(%eax), %mm2		C				am 3
	paddq	%mm0, %mm4		C				am 3
	movd	4(%edx), %mm5		C				am 3
	jmp	L(lam3)			C				am 3
	ALIGN(16)			C				am 3
L(lam3):
	pmuludq	%mm7, %mm2		C				am 3
	paddq	%mm4, %mm6		C				am 3
	movd	12(%eax), %mm3		C				am 3
	paddq	%mm1, %mm5		C				am 3
	movd	8(%edx), %mm4		C				am 3
	movd	%mm6, (%edx)		C				am 3
	psrlq	$32, %mm6		C				am 3
	pmuludq	%mm7, %mm3		C				am 3
	paddq	%mm5, %mm6		C				am 3
	movd	16(%eax), %mm0		C				am 3
	paddq	%mm2, %mm4		C				am 3
	movd	12(%edx), %mm5		C				am 3
	movd	%mm6, 4(%edx)		C				am 3
	psrlq	$32, %mm6		C				am 3
	pmuludq	%mm7, %mm0		C				am 3
	paddq	%mm4, %mm6		C				am 3
	movd	20(%eax), %mm1		C				am 3
	paddq	%mm3, %mm5		C				am 3
	movd	16(%edx), %mm4		C				am 3
	movd	%mm6, 8(%edx)		C				am 3
	psrlq	$32, %mm6		C				am 3
	pmuludq	%mm7, %mm1		C				am 3
	paddq	%mm5, %mm6		C				am 3
	movd	24(%eax), %mm2		C				am 3
	paddq	%mm0, %mm4		C				am 3
	movd	20(%edx), %mm5		C				am 3
	movd	%mm6, 12(%edx)		C				am 3
	psrlq	$32, %mm6		C				am 3
	lea	16(%eax), %eax		C				am 3
	lea	16(%edx), %edx		C				am 3
	add	$4, %ecx		C				am 3
	jnz	L(lam3)			C				am 3
	pmuludq	%mm7, %mm2		C				am 3
	paddq	%mm4, %mm6		C				am 3
	paddq	%mm1, %mm5		C				am 3
	movd	8(%edx), %mm4		C				am 3
	movd	%mm6, (%edx)		C				am 3
	psrlq	$32, %mm6		C				am 3
	paddq	%mm5, %mm6		C				am 3
	paddq	%mm2, %mm4		C				am 3
L(x3):	movd	%mm6, 4(%edx)		C				am 3
	psrlq	$32, %mm6		C				am 3
	paddq	%mm4, %mm6		C				am 3
	movd	%mm6, 8(%edx)		C				am 3
	psrlq	$32, %mm6		C				am 3
	movd	%mm6, 12(%edx)		C				am 3
	dec	%ebx			C				am 3
	jnz	L(olp3)			C				am 3
L(oel3):
	emms				C				   3
	pop	%edi			C				   3
	pop	%ebx			C				   3
	pop	%esi			C				   3
	ret				C				   3
EPILOGUE()
