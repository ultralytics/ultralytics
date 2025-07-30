dnl  x86 mpn_mul_basecase -- Multiply two limb vectors and store the result in
dnl  a third limb vector.

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
C  * Postpone push of ebx until we know vn > 1.  Perhaps use caller-saves regs
C    for inlined mul_1, allowing us to postpone all pushes.
C  * Perhaps write special code for vn <= un < M, for some small M.

C void mpn_mul_basecase (mp_ptr wp,
C                        mp_srcptr xp, mp_size_t xn,
C                        mp_srcptr yp, mp_size_t yn);
C

define(`rp',  `%edi')
define(`up',  `%esi')
define(`un',  `%ecx')
define(`vp',  `%ebp')
define(`vn',  `36(%esp)')

	TEXT
	ALIGN(16)
PROLOGUE(mpn_mul_basecase)
	push	%edi
	push	%esi
	push	%ebx
	push	%ebp
	mov	20(%esp), rp
	mov	24(%esp), up
	mov	28(%esp), un
	mov	32(%esp), vp

	movd	(up), %mm0
	movd	(vp), %mm7
	pmuludq	%mm7, %mm0
	pxor	%mm6, %mm6

	mov	un, %eax
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
	sub	$4, un
	movd	%mm6, 8(rp)
	lea	16(up), up
	ja	L(lm3)

	psrlq	$32, %mm6
	movd	%mm6, 12(rp)

	decl	vn
	jz	L(done)
	lea	-8(rp), rp

L(ol3):	mov	28(%esp), un
	neg	un
	lea	4(vp), vp
	movd	(vp), %mm7	C read next V limb
	mov	24(%esp), up
	lea	16(rp,un,4), rp

	movd	(up), %mm0
	pmuludq	%mm7, %mm0
	sar	$2, un
	movd	4(up), %mm1
	movd	%mm0, %ebx
	pmuludq	%mm7, %mm1
	lea	-8(up), up
	xor	%edx, %edx	C zero edx and CF
	jmp	L(a3)

L(la3):	movd	4(up), %mm1
	adc	$0, %edx
	add	%eax, 12(rp)
	movd	%mm0, %ebx
	pmuludq	%mm7, %mm1
	lea	16(rp), rp
	psrlq	$32, %mm0
	adc	%edx, %ebx
	movd	%mm0, %edx
	movd	%mm1, %eax
	movd	8(up), %mm0
	pmuludq	%mm7, %mm0
	adc	$0, %edx
	add	%ebx, (rp)
	psrlq	$32, %mm1
	adc	%edx, %eax
	movd	%mm1, %edx
	movd	%mm0, %ebx
	movd	12(up), %mm1
	pmuludq	%mm7, %mm1
	adc	$0, %edx
	add	%eax, 4(rp)
L(a3):	psrlq	$32, %mm0
	adc	%edx, %ebx
	movd	%mm0, %edx
	movd	%mm1, %eax
	lea	16(up), up
	movd	(up), %mm0
	adc	$0, %edx
	add	%ebx, 8(rp)
	psrlq	$32, %mm1
	adc	%edx, %eax
	movd	%mm1, %edx
	pmuludq	%mm7, %mm0
	inc	un
	jnz	L(la3)

	adc	un, %edx	C un is zero here
	add	%eax, 12(rp)
	movd	%mm0, %ebx
	psrlq	$32, %mm0
	adc	%edx, %ebx
	movd	%mm0, %eax
	adc	un, %eax
	add	%ebx, 16(rp)
	adc	un, %eax
	mov	%eax, 20(rp)

	decl	vn
	jnz	L(ol3)
	jmp	L(done)

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
	sub	$4, un
	movd	%mm6, 12(rp)
	lea	16(up), up
	ja	L(lm0)

	psrlq	$32, %mm6
	movd	%mm6, 16(rp)

	decl	vn
	jz	L(done)
	lea	-4(rp), rp

L(ol0):	mov	28(%esp), un
	neg	un
	lea	4(vp), vp
	movd	(vp), %mm7	C read next V limb
	mov	24(%esp), up
	lea	20(rp,un,4), rp

	movd	(up), %mm1
	pmuludq	%mm7, %mm1
	sar	$2, un
	movd	4(up), %mm0
	lea	-4(up), up
	movd	%mm1, %eax
	pmuludq	%mm7, %mm0
	xor	%edx, %edx	C zero edx and CF
	jmp	L(a0)

L(la0):	movd	4(up), %mm1
	adc	$0, %edx
	add	%eax, 12(rp)
	movd	%mm0, %ebx
	pmuludq	%mm7, %mm1
	lea	16(rp), rp
	psrlq	$32, %mm0
	adc	%edx, %ebx
	movd	%mm0, %edx
	movd	%mm1, %eax
	movd	8(up), %mm0
	pmuludq	%mm7, %mm0
	adc	$0, %edx
	add	%ebx, (rp)
L(a0):	psrlq	$32, %mm1
	adc	%edx, %eax
	movd	%mm1, %edx
	movd	%mm0, %ebx
	movd	12(up), %mm1
	pmuludq	%mm7, %mm1
	adc	$0, %edx
	add	%eax, 4(rp)
	psrlq	$32, %mm0
	adc	%edx, %ebx
	movd	%mm0, %edx
	movd	%mm1, %eax
	lea	16(up), up
	movd	(up), %mm0
	adc	$0, %edx
	add	%ebx, 8(rp)
	psrlq	$32, %mm1
	adc	%edx, %eax
	movd	%mm1, %edx
	pmuludq	%mm7, %mm0
	inc	un
	jnz	L(la0)

	adc	un, %edx	C un is zero here
	add	%eax, 12(rp)
	movd	%mm0, %ebx
	psrlq	$32, %mm0
	adc	%edx, %ebx
	movd	%mm0, %eax
	adc	un, %eax
	add	%ebx, 16(rp)
	adc	un, %eax
	mov	%eax, 20(rp)

	decl	vn
	jnz	L(ol0)
	jmp	L(done)

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
	sub	$4, un
	movd	%mm6, (rp)
	lea	16(up), up
	ja	L(lm1)

	psrlq	$32, %mm6
	movd	%mm6, 4(rp)

	decl	vn
	jz	L(done)
	lea	-16(rp), rp

L(ol1):	mov	28(%esp), un
	neg	un
	lea	4(vp), vp
	movd	(vp), %mm7	C read next V limb
	mov	24(%esp), up
	lea	24(rp,un,4), rp

	movd	(up), %mm0
	pmuludq	%mm7, %mm0
	sar	$2, un
	movd	%mm0, %ebx
	movd	4(up), %mm1
	pmuludq	%mm7, %mm1
	xor	%edx, %edx	C zero edx and CF
	inc	un
	jmp	L(a1)

L(la1):	movd	4(up), %mm1
	adc	$0, %edx
	add	%eax, 12(rp)
	movd	%mm0, %ebx
	pmuludq	%mm7, %mm1
	lea	16(rp), rp
L(a1):	psrlq	$32, %mm0
	adc	%edx, %ebx
	movd	%mm0, %edx
	movd	%mm1, %eax
	movd	8(up), %mm0
	pmuludq	%mm7, %mm0
	adc	$0, %edx
	add	%ebx, (rp)
	psrlq	$32, %mm1
	adc	%edx, %eax
	movd	%mm1, %edx
	movd	%mm0, %ebx
	movd	12(up), %mm1
	pmuludq	%mm7, %mm1
	adc	$0, %edx
	add	%eax, 4(rp)
	psrlq	$32, %mm0
	adc	%edx, %ebx
	movd	%mm0, %edx
	movd	%mm1, %eax
	lea	16(up), up
	movd	(up), %mm0
	adc	$0, %edx
	add	%ebx, 8(rp)
	psrlq	$32, %mm1
	adc	%edx, %eax
	movd	%mm1, %edx
	pmuludq	%mm7, %mm0
	inc	un
	jnz	L(la1)

	adc	un, %edx	C un is zero here
	add	%eax, 12(rp)
	movd	%mm0, %ebx
	psrlq	$32, %mm0
	adc	%edx, %ebx
	movd	%mm0, %eax
	adc	un, %eax
	add	%ebx, 16(rp)
	adc	un, %eax
	mov	%eax, 20(rp)

	decl	vn
	jnz	L(ol1)
	jmp	L(done)

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
	sub	$4, un
	movd	%mm6, 4(rp)
	lea	16(up), up
	ja	L(lm2)

	psrlq	$32, %mm6
	movd	%mm6, 8(rp)

	decl	vn
	jz	L(done)
	lea	-12(rp), rp

L(ol2):	mov	28(%esp), un
	neg	un
	lea	4(vp), vp
	movd	(vp), %mm7	C read next V limb
	mov	24(%esp), up
	lea	12(rp,un,4), rp

	movd	(up), %mm1
	pmuludq	%mm7, %mm1
	sar	$2, un
	movd	4(up), %mm0
	lea	4(up), up
	movd	%mm1, %eax
	xor	%edx, %edx	C zero edx and CF
	jmp	L(lo2)

L(la2):	movd	4(up), %mm1
	adc	$0, %edx
	add	%eax, 12(rp)
	movd	%mm0, %ebx
	pmuludq	%mm7, %mm1
	lea	16(rp), rp
	psrlq	$32, %mm0
	adc	%edx, %ebx
	movd	%mm0, %edx
	movd	%mm1, %eax
	movd	8(up), %mm0
	pmuludq	%mm7, %mm0
	adc	$0, %edx
	add	%ebx, (rp)
	psrlq	$32, %mm1
	adc	%edx, %eax
	movd	%mm1, %edx
	movd	%mm0, %ebx
	movd	12(up), %mm1
	pmuludq	%mm7, %mm1
	adc	$0, %edx
	add	%eax, 4(rp)
	psrlq	$32, %mm0
	adc	%edx, %ebx
	movd	%mm0, %edx
	movd	%mm1, %eax
	lea	16(up), up
	movd	(up), %mm0
	adc	$0, %edx
	add	%ebx, 8(rp)
L(lo2):	psrlq	$32, %mm1
	adc	%edx, %eax
	movd	%mm1, %edx
	pmuludq	%mm7, %mm0
	inc	un
	jnz	L(la2)

	adc	un, %edx	C un is zero here
	add	%eax, 12(rp)
	movd	%mm0, %ebx
	psrlq	$32, %mm0
	adc	%edx, %ebx
	movd	%mm0, %eax
	adc	un, %eax
	add	%ebx, 16(rp)
	adc	un, %eax
	mov	%eax, 20(rp)

	decl	vn
	jnz	L(ol2)
C	jmp	L(done)

C ================================================================
L(done):
	emms
	pop	%ebp
	pop	%ebx
	pop	%esi
	pop	%edi
	ret
EPILOGUE()
