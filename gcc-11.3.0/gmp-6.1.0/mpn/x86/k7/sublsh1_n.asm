dnl  AMD K7 mpn_sublsh1_n_ip1 -- rp[] = rp[] - (up[] << 1)

dnl  Copyright 2011 Free Software Foundation, Inc.

dnl  Contributed to the GNU project by Torbjorn Granlund and Marco Bodrato.

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

C This is an attempt at a sublsh1_n for x86-32, not relying on sse2 insns.  The
C innerloop is 2*3-way unrolled, which is best we can do with the available
C registers.  It seems tricky to use the same structure for rsblsh1_n, since we
C cannot feed carry between operations there.

C			    cycles/limb
C P5
C P6 model 0-8,10-12
C P6 model 9  (Banias)
C P6 model 13 (Dothan)
C P4 model 0  (Willamette)
C P4 model 1  (?)
C P4 model 2  (Northwood)
C P4 model 3  (Prescott)
C P4 model 4  (Nocona)
C Intel Atom			 6.75
C AMD K6
C AMD K7
C AMD K8

C This is a basic sublsh1_n for k7, atom, and perhaps some other x86-32
C processors.  It uses 2*4-way unrolling, for good reasons.
C
C Breaking carry recurrency might be a good idea.  We would then need separate
C registers for the shift carry and add/subtract carry, which in turn would
C force is to 2*2-way unrolling.

defframe(PARAM_SIZE,	12)
defframe(PARAM_SRC,	 8)
defframe(PARAM_DST,	 4)

dnl  re-use parameter space
define(VAR_COUNT,`PARAM_SIZE')
define(SAVE_EBX,`PARAM_SRC')
define(SAVE_EBP,`PARAM_DST')

ASM_START()
	TEXT
	ALIGN(8)
PROLOGUE(mpn_sublsh1_n_ip1)
deflit(`FRAME',0)

define(`rp',  `%edi')
define(`up',  `%esi')

	mov	PARAM_SIZE, %eax	C size
	push	up			FRAME_pushl()
	push	rp			FRAME_pushl()
	xor	%edx, %edx
	mov	PARAM_SRC, up
	mov	PARAM_DST, rp
	mov	%ebx, SAVE_EBX
	mov	%eax, %ebx
	shr	$3, %eax

	not	%eax			C count = -(size\8)-i
	and	$7, %ebx		C size % 8
	jz	L(exact)

L(oop):
ifdef(`CPU_P6',`
	shr	%edx ')			C restore 2nd saved carry bit
	mov	(up), %ecx
	adc	%ecx, %ecx
	rcr	%edx			C restore 1st saved carry bit
	lea	4(up), up
	sbb	%ecx, (rp)
	lea	4(rp), rp
	adc	%edx, %edx		C save a carry bit in edx
ifdef(`CPU_P6',`
	adc	%edx, %edx ')		C save another carry bit in edx
	dec	%ebx
	jnz	L(oop)
L(exact):
	inc	%eax
	jz	L(end)
	mov	%eax, VAR_COUNT
	mov	%ebp, SAVE_EBP

	ALIGN(16)
L(top):
ifdef(`CPU_P6',`
	shr	%edx ')			C restore 2nd saved carry bit
	mov	(up), %eax
	adc	%eax, %eax
	mov	4(up), %ebx
	adc	%ebx, %ebx
	mov	8(up), %ecx
	adc	%ecx, %ecx
	mov	12(up), %ebp
	adc	%ebp, %ebp

	rcr	%edx			C restore 1st saved carry bit

	sbb	%eax, (rp)
	sbb	%ebx, 4(rp)
	sbb	%ecx, 8(rp)
	sbb	%ebp, 12(rp)

	mov	16(up), %eax
	adc	%eax, %eax
	mov	20(up), %ebx
	adc	%ebx, %ebx
	mov	24(up), %ecx
	adc	%ecx, %ecx
	mov	28(up), %ebp
	adc	%ebp, %ebp

	lea	32(up), up
	adc	%edx, %edx		C save a carry bit in edx

	sbb	%eax, 16(rp)
	sbb	%ebx, 20(rp)
	sbb	%ecx, 24(rp)
	sbb	%ebp, 28(rp)

ifdef(`CPU_P6',`
	adc	%edx, %edx ')		C save another carry bit in edx
	incl	VAR_COUNT
	lea	32(rp), rp
	jne	L(top)

	mov	SAVE_EBP, %ebp
L(end):
	mov	SAVE_EBX, %ebx

ifdef(`CPU_P6',`
	xor	%eax, %eax
	shr	$1, %edx
	adc	%edx, %eax
',`
	adc	$0, %edx
	mov	%edx, %eax
')
	pop	rp			FRAME_popl()
	pop	up			FRAME_popl()
	ret
EPILOGUE()
ASM_END()
