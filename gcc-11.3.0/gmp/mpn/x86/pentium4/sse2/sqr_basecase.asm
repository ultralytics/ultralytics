dnl  mpn_sqr_basecase for Pentium 4 and P6 models with SSE2 (i.e., 9,D,E,F).

dnl  Copyright 2001, 2002, 2007 Free Software Foundation, Inc.

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
C  * In Lam3...Lam1 code for, keep accumulation operands in registers, without
C    storing intermediates to rp.
C  * We might want to keep 32 in a free mm register, since the register form is
C    3 bytes and the immediate form is 4 bytes.  About 80 bytes to save.
C  * Look into different loop alignment, we now expand the code about 50 bytes
C    with possibly needless alignment.
C  * Use OSP, should solve feed-in latency problems.
C  * Address relative slowness for un<=3 for Pentium M.  The old code is there
C    considerably faster.  (1:20/14, 2:34:32, 3:66/57)

C INPUT PARAMETERS
C rp		sp + 4
C up		sp + 8
C un		sp + 12

	TEXT
	ALIGN(16)
PROLOGUE(mpn_sqr_basecase)
	mov	4(%esp), %edx		C rp
	mov	8(%esp), %eax		C up
	mov	12(%esp), %ecx		C un

	cmp	$2, %ecx
	jc	L(un1)
	jz	L(un2)
	cmp	$4, %ecx
	jc	L(un3)
	jz	L(un4)
	jmp	L(big)

L(un1):	mov	(%eax), %eax
	mov	%edx, %ecx
	mul	%eax
	mov	%eax, (%ecx)
	mov	%edx, 4(%ecx)
	ret
L(un2):	movd	(%eax), %mm0		C				un=2
	movd	(%eax), %mm2		C				un=2
	movd	4(%eax), %mm1		C				un=2
	pmuludq	%mm0, %mm0		C 64b weight 0			un=2
	pmuludq	%mm1, %mm2		C 64b weight 32			un=2
	pmuludq	%mm1, %mm1		C 64b weight 64			un=2
	movd	%mm0, (%edx)		C				un=2
	psrlq	$32, %mm0		C 32b weight 32			un=2
	pcmpeqd	%mm7, %mm7		C				un=2
	psrlq	$33, %mm7		C 0x000000007FFFFFFF		un=2
	pand	%mm2, %mm7		C 31b weight 32			un=2
	psrlq	$31, %mm2		C 33b weight 65			un=2
	psllq	$1, %mm7		C 31b weight 33			un=2
	paddq	%mm7, %mm0		C				un=2
	movd	%mm0, 4(%edx)		C				un=2
	psrlq	$32, %mm0		C				un=2
	paddq	%mm2, %mm1		C				un=2
	paddq	%mm0, %mm1		C				un=2
	movd	%mm1, 8(%edx)		C				un=2
	psrlq	$32, %mm1		C				un=2
	movd	%mm1, 12(%edx)		C				un=2
	emms
	ret
L(un3):	movd	(%eax), %mm7		C				un=3
	movd	4(%eax), %mm6		C				un=3
	pmuludq	%mm7, %mm6		C				un=3
	movd	8(%eax), %mm2		C				un=3
	pmuludq	%mm7, %mm2		C				un=3
	movd	%mm6, 4(%edx)		C				un=3
	psrlq	$32, %mm6		C				un=3
	paddq	%mm2, %mm6		C				un=3
	movd	%mm6, 8(%edx)		C				un=3
	psrlq	$32, %mm6		C				un=3
	movd	%mm6, 12(%edx)		C				un=3
	lea	4(%edx), %edx		C				un=3
	lea	4(%eax), %eax		C				un=3
	jmp	L(am1)
L(un4):	movd	(%eax), %mm7		C				un=4
	movd	4(%eax), %mm6		C				un=4
	pmuludq	%mm7, %mm6		C				un=4
	movd	8(%eax), %mm0		C				un=4
	pmuludq	%mm7, %mm0		C				un=4
	movd	12(%eax), %mm1		C				un=4
	pmuludq	%mm7, %mm1		C				un=4
	movd	%mm6, 4(%edx)		C				un=4
	psrlq	$32, %mm6		C				un=4
	paddq	%mm0, %mm6		C				un=4
	movd	%mm6, 8(%edx)		C				un=4
	psrlq	$32, %mm6		C				un=4
	paddq	%mm1, %mm6		C				un=4
	movd	%mm6, 12(%edx)		C				un=4
	psrlq	$32, %mm6		C				un=4
	movd	%mm6, 16(%edx)		C				un=4
	lea	4(%edx), %edx		C				un=4
	lea	4(%eax), %eax		C				un=4
	jmp	L(am2)

L(big):	push	%esi
	push	%ebx
	push	%edi
	pxor	%mm6, %mm6
	movd	(%eax), %mm7		C
	lea	4(%eax), %esi		C init up, up++
	lea	4(%eax), %eax		C up2++  FIXME: should fix offsets
	lea	4(%edx), %edi		C init rp, rp++
	lea	4(%edx), %edx		C rp2++
	lea	-4(%ecx), %ebx		C loop count
	and	$3, %ecx
	jz	L(3m)
	cmp	$2, %ecx
	ja	L(2m)
	jb	L(0m)

L(1m):
	movd	(%eax), %mm4		C				m 1
	lea	(%ebx), %ecx		C inner loop count		m 1
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
	sub	$4, %ecx		C				m 1
	ja	L(lpm1)			C				m 1
	pmuludq	%mm7, %mm4		C				m 1
	paddq	%mm0, %mm6		C				m 1
	movd	%mm6, -8(%edx)		C				m 1
	psrlq	$32, %mm6		C				m 1
	paddq	%mm1, %mm6		C				m 1
	jmp	L(0)

L(2m):
	movd	(%eax), %mm1		C				m 2
	lea	(%ebx), %ecx		C inner loop count		m 2
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
	sub	$4, %ecx		C				m 2
	ja	L(lpm2)			C				m 2
	pmuludq	%mm7, %mm4		C				m 2
	paddq	%mm0, %mm6		C				m 2
	movd	%mm6, -4(%edx)		C				m 2
	psrlq	$32, %mm6		C				m 2
	paddq	%mm1, %mm6		C				m 2
	jmp	L(1)

L(3m):
	movd	(%eax), %mm0		C				m 3
	lea	(%ebx), %ecx		C inner loop count		m 3
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
	sub	$4, %ecx		C				m 3
	ja	L(lpm3)			C				m 3
	pmuludq	%mm7, %mm4		C				m 3
	paddq	%mm0, %mm6		C				m 3
	movd	%mm6, (%edx)		C				m 3
	psrlq	$32, %mm6		C				m 3
	paddq	%mm1, %mm6		C				m 3
	jmp	L(2)

L(0m):
	movd	(%eax), %mm3		C				m 0
	lea	(%ebx), %ecx		C inner loop count		m 0
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
	sub	$4, %ecx		C				m 0
	ja	L(lpm0)			C				m 0
	pmuludq	%mm7, %mm4		C				m 0
	paddq	%mm0, %mm6		C				m 0
	movd	%mm6, -12(%edx)		C				m 0
	psrlq	$32, %mm6		C				m 0
	paddq	%mm1, %mm6		C				m 0
	jmp	L(3)

L(outer):
	lea	8(%edi), %edi		C rp += 2
	movd	(%esi), %mm7		C				am 3
	mov	%edi, %edx		C rp2 = rp			am 3
	lea	4(%esi), %esi		C up++				am 3
	lea	(%esi), %eax		C up2 = up			am 3
	movd	(%eax), %mm0		C				am 3
	lea	(%ebx), %ecx		C inner loop count		am 3
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
	sub	$4, %ecx		C				am 3
	ja	L(lam3)			C				am 3
	pmuludq	%mm7, %mm2		C				am 3
	paddq	%mm4, %mm6		C				am 3
	paddq	%mm1, %mm5		C				am 3
	movd	8(%edx), %mm4		C				am 3
	movd	%mm6, (%edx)		C				am 3
	psrlq	$32, %mm6		C				am 3
	paddq	%mm5, %mm6		C				am 3
	paddq	%mm2, %mm4		C				am 3
L(2):	movd	%mm6, 4(%edx)		C				am 3
	psrlq	$32, %mm6		C				am 3
	paddq	%mm4, %mm6		C				am 3
	movd	%mm6, 8(%edx)		C				am 3
	psrlq	$32, %mm6		C				am 3
	movd	%mm6, 12(%edx)		C				am 3

	lea	8(%edi), %edi		C rp += 2
	movd	(%esi), %mm7		C				am 2
	mov	%edi, %edx		C rp2 = rp			am 2
	lea	4(%esi), %esi		C up++				am 2
	lea	(%esi), %eax		C up2 = up			am 2
	movd	(%eax), %mm1		C				am 2
	lea	(%ebx), %ecx		C inner loop count		am 2
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
	sub	$4, %ecx		C				am 2
	ja	L(lam2)			C				am 2
	pmuludq	%mm7, %mm2		C				am 2
	paddq	%mm4, %mm6		C				am 2
	paddq	%mm1, %mm5		C				am 2
	movd	4(%edx), %mm4		C				am 2
	movd	%mm6, -4(%edx)		C				am 2
	psrlq	$32, %mm6		C				am 2
	paddq	%mm5, %mm6		C				am 2
	paddq	%mm2, %mm4		C				am 2
L(1):	movd	%mm6, (%edx)		C				am 2
	psrlq	$32, %mm6		C				am 2
	paddq	%mm4, %mm6		C				am 2
	movd	%mm6, 4(%edx)		C				am 2
	psrlq	$32, %mm6		C				am 2
	movd	%mm6, 8(%edx)		C				am 2

	lea	8(%edi), %edi		C rp += 2
	movd	(%esi), %mm7		C				am 1
	mov	%edi, %edx		C rp2 = rp			am 1
	lea	4(%esi), %esi		C up++				am 1
	lea	(%esi), %eax		C up2 = up			am 1
	movd	(%eax), %mm2		C				am 1
	lea	(%ebx), %ecx		C inner loop count		am 1
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
	sub	$4, %ecx		C				am 1
	ja	L(lam1)			C				am 1
	pmuludq	%mm7, %mm2		C				am 1
	paddq	%mm4, %mm6		C				am 1
	paddq	%mm1, %mm5		C				am 1
	movd	(%edx), %mm4		C				am 1
	movd	%mm6, -8(%edx)		C				am 1
	psrlq	$32, %mm6		C				am 1
	paddq	%mm5, %mm6		C				am 1
	paddq	%mm2, %mm4		C				am 1
L(0):	movd	%mm6, -4(%edx)		C				am 1
	psrlq	$32, %mm6		C				am 1
	paddq	%mm4, %mm6		C				am 1
	movd	%mm6, (%edx)		C				am 1
	psrlq	$32, %mm6		C				am 1
	movd	%mm6, 4(%edx)		C				am 1

	lea	8(%edi), %edi		C rp += 2
	movd	(%esi), %mm7		C				am 0
	mov	%edi, %edx		C rp2 = rp			am 0
	lea	4(%esi), %esi		C up++				am 0
	lea	(%esi), %eax		C up2 = up			am 0
	movd	(%eax), %mm3		C				am 0
	lea	(%ebx), %ecx		C inner loop count		am 0
	pxor	%mm6, %mm6		C				am 0
	pmuludq	%mm7, %mm3		C				am 0
	movd	4(%eax), %mm0		C				am 0
	movd	(%edx), %mm5		C				am 0
	pmuludq	%mm7, %mm0		C				am 0
	movd	8(%eax), %mm1		C				am 0
	paddq	%mm3, %mm5		C				am 0
	movd	4(%edx), %mm4		C				am 0
	jmp	L(am00)			C				am 0
	ALIGN(16)			C				am 0
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
	sub	$4, %ecx		C				am 0
	ja	L(lam0)			C				am 0
	pmuludq	%mm7, %mm2		C				am 0
	paddq	%mm4, %mm6		C				am 0
	paddq	%mm1, %mm5		C				am 0
	movd	-4(%edx), %mm4		C				am 0
	movd	%mm6, -12(%edx)		C				am 0
	psrlq	$32, %mm6		C				am 0
	paddq	%mm5, %mm6		C				am 0
	paddq	%mm2, %mm4		C				am 0
L(3):	movd	%mm6, -8(%edx)		C				am 0
	psrlq	$32, %mm6		C				am 0
	paddq	%mm4, %mm6		C				am 0
	movd	%mm6, -4(%edx)		C				am 0
	psrlq	$32, %mm6		C				am 0
	movd	%mm6, (%edx)		C				am 0
	sub	$4, %ebx		C				am 0
	ja	L(outer)			C				am 0

	mov	%edi, %edx
	mov	%esi, %eax
	pop	%edi
	pop	%ebx
	pop	%esi

L(am3):	C up[un-1..un-3] x up[un-4]
	lea	8(%edx), %edx		C rp2 += 2
	movd	(%eax), %mm7
	movd	4(%eax), %mm1
	movd	8(%eax), %mm2
	movd	12(%eax), %mm3
	movd	(%edx), %mm4
	pmuludq	%mm7, %mm1
	movd	4(%edx), %mm5
	pmuludq	%mm7, %mm2
	movd	8(%edx), %mm6
	pmuludq	%mm7, %mm3
	paddq	%mm1, %mm4
	paddq	%mm2, %mm5
	paddq	%mm3, %mm6
	movd	%mm4, (%edx)
	psrlq	$32, %mm4
	paddq	%mm5, %mm4
	movd	%mm4, 4(%edx)
	psrlq	$32, %mm4
	paddq	%mm6, %mm4
	movd	%mm4, 8(%edx)
	psrlq	$32, %mm4
	movd	%mm4, 12(%edx)		C FIXME feed through!
	lea	4(%eax), %eax

L(am2):	C up[un-1..un-2] x up[un-3]
	lea	8(%edx), %edx		C rp2 += 2
	movd	(%eax), %mm7
	movd	4(%eax), %mm1
	movd	8(%eax), %mm2
	movd	(%edx), %mm4
	movd	4(%edx), %mm5
	pmuludq	%mm7, %mm1
	pmuludq	%mm7, %mm2
	paddq	%mm1, %mm4
	paddq	%mm2, %mm5
	movd	%mm4, (%edx)
	psrlq	$32, %mm4
	paddq	%mm5, %mm4
	movd	%mm4, 4(%edx)
	psrlq	$32, %mm4
	movd	%mm4, 8(%edx)		C FIXME feed through!
	lea	4(%eax), %eax

L(am1):	C up[un-1] x up[un-2]
	lea	8(%edx), %edx		C rp2 += 2
	movd	(%eax), %mm7
	movd	4(%eax), %mm2
	movd	(%edx), %mm4
	pmuludq	%mm7, %mm2
	paddq	%mm2, %mm4
	movd	%mm4, (%edx)
	psrlq	$32, %mm4
	movd	%mm4, 4(%edx)

C *** diag stuff, use elementary code for now

	mov	4(%esp), %edx		C rp
	mov	8(%esp), %eax		C up
	mov	12(%esp), %ecx		C un

	movd	(%eax), %mm2
	pmuludq	%mm2, %mm2		C src[0]^2

	pcmpeqd	%mm7, %mm7
	psrlq	$32, %mm7

	movd	4(%edx), %mm3		C dst[1]

	movd	%mm2, (%edx)
	psrlq	$32, %mm2

	psllq	$1, %mm3		C 2*dst[1]
	paddq	%mm3, %mm2
	movd	%mm2, 4(%edx)
	psrlq	$32, %mm2

	sub	$2, %ecx

L(diag):
	movd	4(%eax), %mm0		C src limb
	add	$4, %eax
	pmuludq	%mm0, %mm0
	movq	%mm7, %mm1
	pand	%mm0, %mm1		C diagonal low
	psrlq	$32, %mm0		C diagonal high

	movd	8(%edx), %mm3
	psllq	$1, %mm3		C 2*dst[i]
	paddq	%mm3, %mm1
	paddq	%mm1, %mm2
	movd	%mm2, 8(%edx)
	psrlq	$32, %mm2

	movd	12(%edx), %mm3
	psllq	$1, %mm3		C 2*dst[i+1]
	paddq	%mm3, %mm0
	paddq	%mm0, %mm2
	movd	%mm2, 12(%edx)
	add	$8, %edx
	psrlq	$32, %mm2

	sub	$1, %ecx
	jnz	L(diag)

	movd	4(%eax), %mm0		C src[size-1]
	pmuludq	%mm0, %mm0
	pand	%mm0, %mm7		C diagonal low
	psrlq	$32, %mm0		C diagonal high

	movd	8(%edx), %mm3		C dst[2*size-2]
	psllq	$1, %mm3
	paddq	%mm3, %mm7
	paddq	%mm7, %mm2
	movd	%mm2, 8(%edx)
	psrlq	$32, %mm2

	paddq	%mm0, %mm2
	movd	%mm2, 12(%edx)		C dst[2*size-1]

	emms
	ret

EPILOGUE()
