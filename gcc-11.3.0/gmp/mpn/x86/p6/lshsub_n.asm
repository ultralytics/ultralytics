dnl  Intel P6 mpn_lshsub_n -- mpn papillion support.

dnl  Copyright 2006 Free Software Foundation, Inc.

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

C P6/13: 3.35 cycles/limb	(separate mpn_sub_n + mpn_lshift needs 4.12)

C (1) The loop is not scheduled in any way, and scheduling attempts have not
C     improved speed on P6/13.  Presumably, the K7 will want scheduling, if it
C     at all wants to use MMX.
C (2) We could save a register by not alternatingly using eax and edx in the
C     loop.

define(`rp',	`%edi')
define(`up',	`%esi')
define(`vp',	`%ebx')
define(`n',	`%ecx')
define(`cnt',	`%mm7')

ASM_START()

	TEXT
	ALIGN(16)

PROLOGUE(mpn_lshsub_n)
	push	%edi
	push	%esi
	push	%ebx

	mov	16(%esp), rp
	mov	20(%esp), up
	mov	24(%esp), vp
	mov	28(%esp), n
	mov	$32, %eax
	sub	32(%esp), %eax
	movd	%eax, cnt

	lea	(up,n,4), up
	lea	(vp,n,4), vp
	lea	(rp,n,4), rp

	neg	n
	mov	n, %eax
	and	$-8, n
	and	$7, %eax
	shl	%eax				C eax = 2x
	lea	(%eax,%eax,4), %edx		C edx = 10x
ifdef(`PIC',`
	call	L(pic_calc)
L(here):
',`
	lea	L(ent)(%eax,%edx,2), %eax	C eax = 22x
')

	pxor	%mm1, %mm1
	pxor	%mm0, %mm0

	jmp	*%eax

ifdef(`PIC',`
L(pic_calc):
	C See mpn/x86/README about old gas bugs
	lea	(%eax,%edx,2), %eax
	add	$L(ent)-L(here), %eax
	add	(%esp), %eax
	ret_internal
')

L(end):	C compute (cy<<cnt) | (edx>>(32-cnt))
	sbb	%eax, %eax
	neg	%eax
	mov	32(%esp), %ecx
	shld	%cl, %edx, %eax

	emms

	pop	%ebx
	pop	%esi
	pop	%edi
	ret
	ALIGN(16)
L(top):	jecxz	L(end)
L(ent):	mov	   0(up,n,4), %eax
	sbb	   0(vp,n,4), %eax
	movd	   %eax, %mm0
	punpckldq  %mm0, %mm1
	psrlq	   %mm7, %mm1
	movd	   %mm1, 0(rp,n,4)

	mov	   4(up,n,4), %edx
	sbb	   4(vp,n,4), %edx
	movd	   %edx, %mm1
	punpckldq  %mm1, %mm0
	psrlq	   %mm7, %mm0
	movd	   %mm0, 4(rp,n,4)

	mov	   8(up,n,4), %eax
	sbb	   8(vp,n,4), %eax
	movd	   %eax, %mm0
	punpckldq  %mm0, %mm1
	psrlq	   %mm7, %mm1
	movd	   %mm1, 8(rp,n,4)

	mov	   12(up,n,4), %edx
	sbb	   12(vp,n,4), %edx
	movd	   %edx, %mm1
	punpckldq  %mm1, %mm0
	psrlq	   %mm7, %mm0
	movd	   %mm0, 12(rp,n,4)

	mov	   16(up,n,4), %eax
	sbb	   16(vp,n,4), %eax
	movd	   %eax, %mm0
	punpckldq  %mm0, %mm1
	psrlq	   %mm7, %mm1
	movd	   %mm1, 16(rp,n,4)

	mov	   20(up,n,4), %edx
	sbb	   20(vp,n,4), %edx
	movd	   %edx, %mm1
	punpckldq  %mm1, %mm0
	psrlq	   %mm7, %mm0
	movd	   %mm0, 20(rp,n,4)

	mov	   24(up,n,4), %eax
	sbb	   24(vp,n,4), %eax
	movd	   %eax, %mm0
	punpckldq  %mm0, %mm1
	psrlq	   %mm7, %mm1
	movd	   %mm1, 24(rp,n,4)

	mov	   28(up,n,4), %edx
	sbb	   28(vp,n,4), %edx
	movd	   %edx, %mm1
	punpckldq  %mm1, %mm0
	psrlq	   %mm7, %mm0
	movd	   %mm0, 28(rp,n,4)

	lea	   8(n), n
	jmp	   L(top)

EPILOGUE()
