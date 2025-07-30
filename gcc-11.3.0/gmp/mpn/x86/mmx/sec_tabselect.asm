dnl  X86 MMX mpn_sec_tabselect.

dnl  Contributed to the GNU project by Torbjörn Granlund.

dnl  Copyright 2011-2013 Free Software Foundation, Inc.

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

C			     cycles/limb     cycles/limb
C			      ali,evn n	     unal,evn n
C P5
C P6 model 0-8,10-12
C P6 model 9  (Banias)
C P6 model 13 (Dothan)		 1.33		 1.87
C P4 model 0  (Willamette)
C P4 model 1  (?)
C P4 model 2  (Northwood)	 2.1		 2.63
C P4 model 3  (Prescott)
C P4 model 4  (Nocona)		 1.7		 2.57
C Intel Atom			 1.85		 2.7
C AMD K6
C AMD K7			 1.33		 1.33
C AMD K8
C AMD K10

define(`rp',     `%edi')
define(`tp',     `%esi')
define(`n',      `%edx')
define(`nents',  `%ecx')
define(`which',  `')

define(`i',      `%ebp')
define(`j',      `%ebx')

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_sec_tabselect)
	push	%ebx
	push	%esi
	push	%edi
	push	%ebp

	mov	20(%esp), rp
	mov	24(%esp), tp
	mov	28(%esp), n
	mov	32(%esp), nents

	movd	36(%esp), %mm6
	punpckldq %mm6, %mm6		C 2 copies of `which'

	mov	$1, %ebx
	movd	%ebx, %mm7
	punpckldq %mm7, %mm7		C 2 copies of 1

	mov	n, j
	add	$-4, j
	js	L(outer_end)

L(outer_top):
	mov	nents, i
	mov	tp, %eax
	pxor	%mm1, %mm1
	pxor	%mm4, %mm4
	pxor	%mm5, %mm5
	ALIGN(16)
L(top):	movq	%mm6, %mm0
	pcmpeqd	%mm1, %mm0
	paddd	%mm7, %mm1
	movq	(tp), %mm2
	movq	8(tp), %mm3
	pand	%mm0, %mm2
	pand	%mm0, %mm3
	por	%mm2, %mm4
	por	%mm3, %mm5
	lea	(tp,n,4), tp
	add	$-1, i
	jne	L(top)

	movq	%mm4, (rp)
	movq	%mm5, 8(rp)

	lea	16(%eax), tp
	lea	16(rp), rp
	add	$-4, j
	jns	L(outer_top)
L(outer_end):

	test	$2, %dl
	jz	L(b0x)

L(b1x):	mov	nents, i
	mov	tp, %eax
	pxor	%mm1, %mm1
	pxor	%mm4, %mm4
	ALIGN(16)
L(tp2):	movq	%mm6, %mm0
	pcmpeqd	%mm1, %mm0
	paddd	%mm7, %mm1
	movq	(tp), %mm2
	pand	%mm0, %mm2
	por	%mm2, %mm4
	lea	(tp,n,4), tp
	add	$-1, i
	jne	L(tp2)

	movq	%mm4, (rp)

	lea	8(%eax), tp
	lea	8(rp), rp

L(b0x):	test	$1, %dl
	jz	L(b00)

L(b01):	mov	nents, i
	pxor	%mm1, %mm1
	pxor	%mm4, %mm4
	ALIGN(16)
L(tp1):	movq	%mm6, %mm0
	pcmpeqd	%mm1, %mm0
	paddd	%mm7, %mm1
	movd	(tp), %mm2
	pand	%mm0, %mm2
	por	%mm2, %mm4
	lea	(tp,n,4), tp
	add	$-1, i
	jne	L(tp1)

	movd	%mm4, (rp)

L(b00):	pop	%ebp
	pop	%edi
	pop	%esi
	pop	%ebx
	emms
	ret
EPILOGUE()
