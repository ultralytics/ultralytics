dnl  x86 mpn_sqr_basecase -- square an mpn number, optimised for atom.

dnl  Contributed to the GNU project by Torbjorn Granlund and Marco Bodrato.

dnl  Copyright 2011 Free Software Foundation, Inc.

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

C TODO
C  * Check if 'jmp N(%esp)' is well-predicted enough to allow us to combine the
C    4 large loops into one; we could use it for the outer loop branch.
C  * Optimise code outside of inner loops.
C  * Write combined addmul_1 feed-in a wind-down code, and use when iterating
C    outer each loop.  ("Overlapping software pipelining")
C  * Perhaps use caller-saves regs for inlined mul_1, allowing us to postpone
C    all pushes.
C  * Perhaps write special code for n < M, for some small M.
C  * Replace inlined addmul_1 with smaller code from aorsmul_1.asm, or perhaps
C    with even less pipelined code.
C  * We run the outer loop until we have a 2-limb by 1-limb addmul_1 left.
C    Consider breaking out earlier, saving high the cost of short loops.

C void mpn_sqr_basecase (mp_ptr wp,
C                        mp_srcptr xp, mp_size_t xn);

define(`rp',  `%edi')
define(`up',  `%esi')
define(`n',   `%ecx')

define(`un',  `%ebp')

	TEXT
	ALIGN(16)
PROLOGUE(mpn_sqr_basecase)
	push	%edi
	push	%esi
	mov	12(%esp), rp
	mov	16(%esp), up
	mov	20(%esp), n

	lea	4(rp), rp	C write triangular product starting at rp[1]
	dec	n
	movd	(up), %mm7

	jz	L(one)
	lea	4(up), up
	push	%ebx
	push	%ebp
	mov	n, %eax

	movd	(up), %mm0
	neg	n
	pmuludq	%mm7, %mm0
	pxor	%mm6, %mm6
	mov	n, un

	and	$3, %eax
	jz	L(of0)
	cmp	$2, %eax
	jc	L(of1)
	jz	L(of2)

C ================================================================
	jmp	L(m3)
	ALIGN(16)
L(lm3):	movd	-4(up), %mm0
	pmuludq	%mm7, %mm0
	psrlq	$32, %mm6
	lea	16(rp), rp
	paddq	%mm0, %mm6
	movd	(up), %mm0
	pmuludq	%mm7, %mm0
	movd	%mm6, -4(rp)
	psrlq	$32, %mm6
L(m3):	paddq	%mm0, %mm6
	movd	4(up), %mm0
	pmuludq	%mm7, %mm0
	movd	%mm6, (rp)
	psrlq	$32, %mm6
	paddq	%mm0, %mm6
	movd	8(up), %mm0
	pmuludq	%mm7, %mm0
	movd	%mm6, 4(rp)
	psrlq	$32, %mm6
	paddq	%mm0, %mm6
	add	$4, un
	movd	%mm6, 8(rp)
	lea	16(up), up
	js	L(lm3)

	psrlq	$32, %mm6
	movd	%mm6, 12(rp)

	inc	n
C	jz	L(done)
  lea	-12(up), up
  lea	4(rp), rp
	jmp	L(ol2)

C ================================================================
	ALIGN(16)
L(lm0):	movd	(up), %mm0
	pmuludq	%mm7, %mm0
	psrlq	$32, %mm6
	lea	16(rp), rp
L(of0):	paddq	%mm0, %mm6
	movd	4(up), %mm0
	pmuludq	%mm7, %mm0
	movd	%mm6, (rp)
	psrlq	$32, %mm6
	paddq	%mm0, %mm6
	movd	8(up), %mm0
	pmuludq	%mm7, %mm0
	movd	%mm6, 4(rp)
	psrlq	$32, %mm6
	paddq	%mm0, %mm6
	movd	12(up), %mm0
	pmuludq	%mm7, %mm0
	movd	%mm6, 8(rp)
	psrlq	$32, %mm6
	paddq	%mm0, %mm6
	add	$4, un
	movd	%mm6, 12(rp)
	lea	16(up), up
	js	L(lm0)

	psrlq	$32, %mm6
	movd	%mm6, 16(rp)

	inc	n
C	jz	L(done)
  lea	-8(up), up
  lea	8(rp), rp
	jmp	L(ol3)

C ================================================================
	ALIGN(16)
L(lm1):	movd	-12(up), %mm0
	pmuludq	%mm7, %mm0
	psrlq	$32, %mm6
	lea	16(rp), rp
	paddq	%mm0, %mm6
	movd	-8(up), %mm0
	pmuludq	%mm7, %mm0
	movd	%mm6, -12(rp)
	psrlq	$32, %mm6
	paddq	%mm0, %mm6
	movd	-4(up), %mm0
	pmuludq	%mm7, %mm0
	movd	%mm6, -8(rp)
	psrlq	$32, %mm6
	paddq	%mm0, %mm6
	movd	(up), %mm0
	pmuludq	%mm7, %mm0
	movd	%mm6, -4(rp)
	psrlq	$32, %mm6
L(of1):	paddq	%mm0, %mm6
	add	$4, un
	movd	%mm6, (rp)
	lea	16(up), up
	js	L(lm1)

	psrlq	$32, %mm6
	movd	%mm6, 4(rp)

	inc	n
	jz	L(done)		C goes away when we add special n=2 code
  lea	-20(up), up
  lea	-4(rp), rp
	jmp	L(ol0)

C ================================================================
	ALIGN(16)
L(lm2):	movd	-8(up), %mm0
	pmuludq	%mm7, %mm0
	psrlq	$32, %mm6
	lea	16(rp), rp
	paddq	%mm0, %mm6
	movd	-4(up), %mm0
	pmuludq	%mm7, %mm0
	movd	%mm6, -8(rp)
	psrlq	$32, %mm6
	paddq	%mm0, %mm6
	movd	(up), %mm0
	pmuludq	%mm7, %mm0
	movd	%mm6, -4(rp)
	psrlq	$32, %mm6
L(of2):	paddq	%mm0, %mm6
	movd	4(up), %mm0
	pmuludq	%mm7, %mm0
	movd	%mm6, (rp)
	psrlq	$32, %mm6
	paddq	%mm0, %mm6
	add	$4, un
	movd	%mm6, 4(rp)
	lea	16(up), up
	js	L(lm2)

	psrlq	$32, %mm6
	movd	%mm6, 8(rp)

	inc	n
C	jz	L(done)
  lea	-16(up), up
C  lea	(rp), rp
C	jmp	L(ol1)

C ================================================================

L(ol1):	lea	4(up,n,4), up
	movd	(up), %mm7	C read next U invariant limb
	lea	8(rp,n,4), rp
	mov	n, un

	movd	4(up), %mm1
	pmuludq	%mm7, %mm1
	sar	$2, un
	movd	%mm1, %ebx
	inc	un
	jz	L(re1)

	movd	8(up), %mm0
	pmuludq	%mm7, %mm0
	xor	%edx, %edx	C zero edx and CF
	jmp	L(a1)

L(la1):	adc	$0, %edx
	add	%ebx, 12(rp)
	movd	%mm0, %eax
	pmuludq	%mm7, %mm1
	lea	16(rp), rp
	psrlq	$32, %mm0
	adc	%edx, %eax
	movd	%mm0, %edx
	movd	%mm1, %ebx
	movd	8(up), %mm0
	pmuludq	%mm7, %mm0
	adc	$0, %edx
	add	%eax, (rp)
L(a1):	psrlq	$32, %mm1
	adc	%edx, %ebx
	movd	%mm1, %edx
	movd	%mm0, %eax
	movd	12(up), %mm1
	pmuludq	%mm7, %mm1
	adc	$0, %edx
	add	%ebx, 4(rp)
	psrlq	$32, %mm0
	adc	%edx, %eax
	movd	%mm0, %edx
	movd	%mm1, %ebx
	lea	16(up), up
	movd	(up), %mm0
	adc	$0, %edx
	add	%eax, 8(rp)
	psrlq	$32, %mm1
	adc	%edx, %ebx
	movd	%mm1, %edx
	pmuludq	%mm7, %mm0
	inc	un
	movd	4(up), %mm1
	jnz	L(la1)

	adc	un, %edx	C un is zero here
	add	%ebx, 12(rp)
	movd	%mm0, %eax
	pmuludq	%mm7, %mm1
	lea	16(rp), rp
	psrlq	$32, %mm0
	adc	%edx, %eax
	movd	%mm0, %edx
	movd	%mm1, %ebx
	adc	un, %edx
	add	%eax, (rp)
	psrlq	$32, %mm1
	adc	%edx, %ebx
	movd	%mm1, %eax
	adc	un, %eax
	add	%ebx, 4(rp)
	adc	un, %eax
	mov	%eax, 8(rp)

	inc	n

C ================================================================

L(ol0):	lea	(up,n,4), up
	movd	4(up), %mm7	C read next U invariant limb
	lea	4(rp,n,4), rp
	mov	n, un

	movd	8(up), %mm0
	pmuludq	%mm7, %mm0
	sar	$2, un
	movd	12(up), %mm1
	movd	%mm0, %eax
	pmuludq	%mm7, %mm1
	xor	%edx, %edx	C zero edx and CF
	jmp	L(a0)

L(la0):	adc	$0, %edx
	add	%ebx, 12(rp)
	movd	%mm0, %eax
	pmuludq	%mm7, %mm1
	lea	16(rp), rp
	psrlq	$32, %mm0
	adc	%edx, %eax
	movd	%mm0, %edx
	movd	%mm1, %ebx
	movd	8(up), %mm0
	pmuludq	%mm7, %mm0
	adc	$0, %edx
	add	%eax, (rp)
	psrlq	$32, %mm1
	adc	%edx, %ebx
	movd	%mm1, %edx
	movd	%mm0, %eax
	movd	12(up), %mm1
	pmuludq	%mm7, %mm1
	adc	$0, %edx
	add	%ebx, 4(rp)
L(a0):	psrlq	$32, %mm0
	adc	%edx, %eax
	movd	%mm0, %edx
	movd	%mm1, %ebx
	lea	16(up), up
	movd	(up), %mm0
	adc	$0, %edx
	add	%eax, 8(rp)
	psrlq	$32, %mm1
	adc	%edx, %ebx
	movd	%mm1, %edx
	pmuludq	%mm7, %mm0
	inc	un
	movd	4(up), %mm1
	jnz	L(la0)

	adc	un, %edx	C un is zero here
	add	%ebx, 12(rp)
	movd	%mm0, %eax
	pmuludq	%mm7, %mm1
	lea	16(rp), rp
	psrlq	$32, %mm0
	adc	%edx, %eax
	movd	%mm0, %edx
	movd	%mm1, %ebx
	adc	un, %edx
	add	%eax, (rp)
	psrlq	$32, %mm1
	adc	%edx, %ebx
	movd	%mm1, %eax
	adc	un, %eax
	add	%ebx, 4(rp)
	adc	un, %eax
	mov	%eax, 8(rp)

	inc	n

C ================================================================

L(ol3):	lea	12(up,n,4), up
	movd	-8(up), %mm7	C read next U invariant limb
	lea	(rp,n,4), rp	C put rp back
	mov	n, un

	movd	-4(up), %mm1
	pmuludq	%mm7, %mm1
	sar	$2, un
	movd	%mm1, %ebx
	movd	(up), %mm0
	xor	%edx, %edx	C zero edx and CF
	jmp	L(a3)

L(la3):	adc	$0, %edx
	add	%ebx, 12(rp)
	movd	%mm0, %eax
	pmuludq	%mm7, %mm1
	lea	16(rp), rp
	psrlq	$32, %mm0
	adc	%edx, %eax
	movd	%mm0, %edx
	movd	%mm1, %ebx
	movd	8(up), %mm0
	pmuludq	%mm7, %mm0
	adc	$0, %edx
	add	%eax, (rp)
	psrlq	$32, %mm1
	adc	%edx, %ebx
	movd	%mm1, %edx
	movd	%mm0, %eax
	movd	12(up), %mm1
	pmuludq	%mm7, %mm1
	adc	$0, %edx
	add	%ebx, 4(rp)
	psrlq	$32, %mm0
	adc	%edx, %eax
	movd	%mm0, %edx
	movd	%mm1, %ebx
	lea	16(up), up
	movd	(up), %mm0
	adc	$0, %edx
	add	%eax, 8(rp)
L(a3):	psrlq	$32, %mm1
	adc	%edx, %ebx
	movd	%mm1, %edx
	pmuludq	%mm7, %mm0
	inc	un
	movd	4(up), %mm1
	jnz	L(la3)

	adc	un, %edx	C un is zero here
	add	%ebx, 12(rp)
	movd	%mm0, %eax
	pmuludq	%mm7, %mm1
	lea	16(rp), rp
	psrlq	$32, %mm0
	adc	%edx, %eax
	movd	%mm0, %edx
	movd	%mm1, %ebx
	adc	un, %edx
	add	%eax, (rp)
	psrlq	$32, %mm1
	adc	%edx, %ebx
	movd	%mm1, %eax
	adc	un, %eax
	add	%ebx, 4(rp)
	adc	un, %eax
	mov	%eax, 8(rp)

	inc	n

C ================================================================

L(ol2):	lea	8(up,n,4), up
	movd	-4(up), %mm7	C read next U invariant limb
	lea	12(rp,n,4), rp
	mov	n, un

	movd	(up), %mm0
	pmuludq	%mm7, %mm0
	xor	%edx, %edx
	sar	$2, un
	movd	4(up), %mm1
	test	un, un		C clear carry
	movd	%mm0, %eax
	pmuludq	%mm7, %mm1
	inc	un
	jnz	L(a2)
	jmp	L(re2)

L(la2):	adc	$0, %edx
	add	%ebx, 12(rp)
	movd	%mm0, %eax
	pmuludq	%mm7, %mm1
	lea	16(rp), rp
L(a2):	psrlq	$32, %mm0
	adc	%edx, %eax
	movd	%mm0, %edx
	movd	%mm1, %ebx
	movd	8(up), %mm0
	pmuludq	%mm7, %mm0
	adc	$0, %edx
	add	%eax, (rp)
	psrlq	$32, %mm1
	adc	%edx, %ebx
	movd	%mm1, %edx
	movd	%mm0, %eax
	movd	12(up), %mm1
	pmuludq	%mm7, %mm1
	adc	$0, %edx
	add	%ebx, 4(rp)
	psrlq	$32, %mm0
	adc	%edx, %eax
	movd	%mm0, %edx
	movd	%mm1, %ebx
	lea	16(up), up
	movd	(up), %mm0
	adc	$0, %edx
	add	%eax, 8(rp)
	psrlq	$32, %mm1
	adc	%edx, %ebx
	movd	%mm1, %edx
	pmuludq	%mm7, %mm0
	inc	un
	movd	4(up), %mm1
	jnz	L(la2)

	adc	un, %edx	C un is zero here
	add	%ebx, 12(rp)
	movd	%mm0, %eax
	pmuludq	%mm7, %mm1
	lea	16(rp), rp
	psrlq	$32, %mm0
	adc	%edx, %eax
	movd	%mm0, %edx
	movd	%mm1, %ebx
	adc	un, %edx
	add	%eax, (rp)
	psrlq	$32, %mm1
	adc	%edx, %ebx
	movd	%mm1, %eax
	adc	un, %eax
	add	%ebx, 4(rp)
	adc	un, %eax
	mov	%eax, 8(rp)

	inc	n
	jmp	L(ol1)

C ================================================================
L(re2):	psrlq	$32, %mm0
	movd	(up), %mm7	C read next U invariant limb
	adc	%edx, %eax
	movd	%mm0, %edx
	movd	%mm1, %ebx
	adc	un, %edx
	add	%eax, (rp)
	lea	4(rp), rp
	psrlq	$32, %mm1
	adc	%edx, %ebx
	movd	%mm1, %eax
	movd	4(up), %mm1
	adc	un, %eax
	add	%ebx, (rp)
	pmuludq	%mm7, %mm1
	adc	un, %eax
	mov	%eax, 4(rp)
	movd	%mm1, %ebx

L(re1):	psrlq	$32, %mm1
	add	%ebx, 4(rp)
	movd	%mm1, %eax
	adc	un, %eax
	xor	n, n		C make n zeroness assumption below true
	mov	%eax, 8(rp)

L(done):			C n is zero here
	mov	24(%esp), up
	mov	28(%esp), %eax

	movd	(up), %mm0
	inc	%eax
	pmuludq	%mm0, %mm0
	lea	4(up), up
	mov	20(%esp), rp
	shr	%eax
	movd	%mm0, (rp)
	psrlq	$32, %mm0
	lea	-12(rp), rp
	mov	%eax, 28(%esp)
	jnc	L(odd)

	movd	%mm0, %ebp
	movd	(up), %mm0
	lea	8(rp), rp
	pmuludq	%mm0, %mm0
	lea	-4(up), up
	add	8(rp), %ebp
	movd	%mm0, %edx
	adc	12(rp), %edx
	rcr	n
	jmp	L(ent)

C	ALIGN(16)		C alignment seems irrelevant
L(top):	movd	(up), %mm1
	adc	n, n
	movd	%mm0, %eax
	pmuludq	%mm1, %mm1
	movd	4(up), %mm0
	adc	(rp), %eax
	movd	%mm1, %ebx
	pmuludq	%mm0, %mm0
	psrlq	$32, %mm1
	adc	4(rp), %ebx
	movd	%mm1, %ebp
	movd	%mm0, %edx
	adc	8(rp), %ebp
	adc	12(rp), %edx
	rcr	n		C FIXME: isn't this awfully slow on atom???
	adc	%eax, (rp)
	adc	%ebx, 4(rp)
L(ent):	lea	8(up), up
	adc	%ebp, 8(rp)
	psrlq	$32, %mm0
	adc	%edx, 12(rp)
L(odd):	decl	28(%esp)
	lea	16(rp), rp
	jnz	L(top)

L(end):	adc	n, n
	movd	%mm0, %eax
	adc	n, %eax
	mov	%eax, (rp)

L(rtn):	emms
	pop	%ebp
	pop	%ebx
	pop	%esi
	pop	%edi
	ret

L(one):	pmuludq	%mm7, %mm7
	movq	%mm7, -4(rp)
	emms
	pop	%esi
	pop	%edi
	ret
EPILOGUE()
