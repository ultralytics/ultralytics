dnl  AMD K7 mpn_addlsh1_n -- rp[] = up[] + (vp[] << 1)

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

C This is an attempt at an addlsh1_n for x86-32, not relying on sse2 insns.
C The innerloop is 2*3-way unrolled, which is best we can do with the available
C registers.  It seems tricky to use the same structure for rsblsh1_n, since we
C cannot feed carry between operations there.

C			    cycles/limb
C P5
C P6 model 0-8,10-12
C P6 model 9  (Banias)
C P6 model 13 (Dothan)		 5.4	(worse than add_n + lshift)
C P4 model 0  (Willamette)
C P4 model 1  (?)
C P4 model 2  (Northwood)
C P4 model 3  (Prescott)
C P4 model 4  (Nocona)
C Intel Atom			 6
C AMD K6			 ?
C AMD K7			 2.5
C AMD K8

C This is a basic addlsh1_n for k7, atom, and perhaps some other x86-32
C processors.  It uses 2*3-way unrolling, for good reasons.  Unfortunately,
C that means we need an initial magic multiply.
C
C It is not clear how to do sublsh1_n or rsblsh1_n using the same pattern.  We
C cannot do rsblsh1_n since we feed carry from the shift blocks to the
C add/subtract blocks, which is right for addition but reversed for
C subtraction.  We could perhaps do sublsh1_n, with some extra move insns,
C without losing any time, since we're not issue limited but carry recurrency
C latency.
C
C Breaking carry recurrency might be a good idea.  We would then need separate
C registers for the shift carry and add/subtract carry, which in turn would
C force is to 2*2-way unrolling.

defframe(PARAM_SIZE,	16)
defframe(PARAM_DBLD,	12)
defframe(PARAM_SRC,	 8)
defframe(PARAM_DST,	 4)

dnl  re-use parameter space
define(VAR_COUNT,`PARAM_DST')
define(VAR_TMP,`PARAM_DBLD')

ASM_START()
	TEXT
	ALIGN(8)
PROLOGUE(mpn_addlsh1_n)
deflit(`FRAME',0)

define(`rp',  `%edi')
define(`up',  `%esi')
define(`vp',  `%ebp')

	mov	$0x2aaaaaab, %eax

	push	%ebx			FRAME_pushl()
	mov	PARAM_SIZE, %ebx	C size

	push	rp			FRAME_pushl()
	mov	PARAM_DST, rp

	mul	%ebx

	push	up			FRAME_pushl()
	mov	PARAM_SRC, up

	not	%edx			C count = -(size\8)-1
	mov	%edx, VAR_COUNT

	push	vp			FRAME_pushl()
	mov	PARAM_DBLD, vp

	lea	3(%edx,%edx,2), %ecx	C count*3+3 = -(size\6)*3
	xor	%edx, %edx
	lea	(%ebx,%ecx,2), %ebx	C size + (count*3+3)*2 = size % 6
	or	%ebx, %ebx
	jz	L(exact)

L(oop):
ifdef(`CPU_P6',`
	shr	%edx ')			C restore 2nd saved carry bit
	mov	(vp), %eax
	adc	%eax, %eax
	rcr	%edx			C restore 1st saved carry bit
	lea	4(vp), vp
	adc	(up), %eax
	lea	4(up), up
	adc	%edx, %edx		C save a carry bit in edx
ifdef(`CPU_P6',`
	adc	%edx, %edx ')		C save another carry bit in edx
	dec	%ebx
	mov	%eax, (rp)
	lea	4(rp), rp
	jnz	L(oop)
	mov	vp, VAR_TMP
L(exact):
	incl	VAR_COUNT
	jz	L(end)

	ALIGN(16)
L(top):
ifdef(`CPU_P6',`
	shr	%edx ')			C restore 2nd saved carry bit
	mov	(vp), %eax
	adc	%eax, %eax
	mov	4(vp), %ebx
	adc	%ebx, %ebx
	mov	8(vp), %ecx
	adc	%ecx, %ecx

	rcr	%edx			C restore 1st saved carry bit

	adc	(up), %eax
	mov	%eax, (rp)
	adc	4(up), %ebx
	mov	%ebx, 4(rp)
	adc	8(up), %ecx
	mov	%ecx, 8(rp)

	mov	12(vp), %eax
	adc	%eax, %eax
	mov	16(vp), %ebx
	adc	%ebx, %ebx
	mov	20(vp), %ecx
	adc	%ecx, %ecx

	lea	24(vp), vp
	adc	%edx, %edx		C save a carry bit in edx

	adc	12(up), %eax
	mov	%eax, 12(rp)
	adc	16(up), %ebx
	mov	%ebx, 16(rp)
	adc	20(up), %ecx

	lea	24(up), up

ifdef(`CPU_P6',`
	adc	%edx, %edx ')		C save another carry bit in edx
	mov	%ecx, 20(rp)
	incl	VAR_COUNT
	lea	24(rp), rp
	jne	L(top)

L(end):
	pop	vp			FRAME_popl()
	pop	up			FRAME_popl()

ifdef(`CPU_P6',`
	xor	%eax, %eax
	shr	$1, %edx
	adc	%edx, %eax
',`
	adc	$0, %edx
	mov	%edx, %eax
')
	pop	rp			FRAME_popl()
	pop	%ebx			FRAME_popl()
	ret
EPILOGUE()
ASM_END()
