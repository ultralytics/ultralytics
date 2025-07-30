dnl  X86-32 and X86-64 mpn_popcount using SSE2.

dnl  Copyright 2006, 2007, 2011, 2015 Free Software Foundation, Inc.

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


C 32-bit		     popcount	     hamdist
C			    cycles/limb	    cycles/limb
C P5				-
C P6 model 0-8,10-12		-
C P6 model 9  (Banias)		?
C P6 model 13 (Dothan)		4
C P4 model 0  (Willamette)	?
C P4 model 1  (?)		?
C P4 model 2  (Northwood)	3.9
C P4 model 3  (Prescott)	?
C P4 model 4  (Nocona)		?
C AMD K6			-
C AMD K7			-
C AMD K8			?

C 64-bit		     popcount	     hamdist
C			    cycles/limb	    cycles/limb
C P4 model 4 (Nocona):		8
C AMD K8,K9			7.5
C AMD K10			3.5
C Intel core2			3.68
C Intel corei			3.15
C Intel atom		       10.8
C VIA nano			6.5

C TODO
C  * Make an mpn_hamdist based on this.  Alignment could either be handled by
C    using movdqu for one operand and movdqa for the other, or by painfully
C    shifting as we go.  Unfortunately, there seem to be no usable shift
C    instruction, except for one that takes an immediate count.
C  * It would probably be possible to cut a few cycles/limb using software
C    pipelining.
C  * There are 35 decode slots unused by the SSE2 instructions.  Loop control
C    needs just 2 or 3 slots, leaving around 32 slots.  This allows a parallel
C    integer based popcount.  Such a combined loop would handle 6 limbs in
C    about 30 cycles on K8.
C  * We could save a byte or two by using 32-bit operations on areg.
C  * Check if using movdqa to a temp of and then register-based pand is faster.

ifelse(GMP_LIMB_BITS,`32',
`	define(`up',  `%edx')
	define(`n',   `%ecx')
	define(`areg',`%eax')
	define(`breg',`%ebx')
	define(`zero',`%xmm4')
	define(`LIMB32',`	$1')
	define(`LIMB64',`dnl')
',`
	define(`up',  `%rdi')
	define(`n',   `%rsi')
	define(`areg',`%rax')
	define(`breg',`%rdx')
	define(`zero',`%xmm8')
	define(`LIMB32',`dnl')
	define(`LIMB64',`	$1')
')

define(`mm01010101',`%xmm6')
define(`mm00110011',`%xmm7')
define(`mm00001111',`%xmm2')

define(`GMP_LIMB_BYTES', eval(GMP_LIMB_BITS/8))
define(`LIMBS_PER_XMM',  eval(16/GMP_LIMB_BYTES))
define(`LIMBS_PER_2XMM', eval(32/GMP_LIMB_BYTES))

undefine(`psadbw')			C override inherited m4 version

C This file is shared between 32-bit and 64-bit builds.  Only the former has
C LEAL.  Default LEAL as an alias of LEA.
ifdef(`LEAL',,`define(`LEAL', `LEA($1,$2)')')

ASM_START()

C Make cnsts global to work around Apple relocation bug.
ifdef(`DARWIN',`
	define(`cnsts', MPN(popccnsts))
	GLOBL	cnsts')

	TEXT
	ALIGN(32)
PROLOGUE(mpn_popcount)

LIMB32(`mov	4(%esp), up	')
LIMB32(`mov	8(%esp), n	')
LIMB32(`push	%ebx		')

	pxor	%xmm3, %xmm3		C zero grand total count
LIMB64(`pxor	zero, zero	')
ifdef(`PIC',`
	LEAL(	cnsts, breg)
',`
LIMB32(`mov	$cnsts, breg	')
LIMB64(`movabs	$cnsts, breg	')
')

	movdqa	-48(breg), mm01010101
	movdqa	-32(breg), mm00110011
	movdqa	-16(breg), mm00001111

	mov	up, areg
	and	$-16, up		C round `up' down to 128-bit boundary
	and	$12, areg		C 32:areg = 0, 4, 8, 12
					C 64:areg = 0, 8
	movdqa	(up), %xmm0
	pand	64(breg,areg,4), %xmm0
	shr	$m4_log2(GMP_LIMB_BYTES), %eax
	add	areg, n			C compensate n for rounded down `up'

	pxor	%xmm4, %xmm4
	sub	$LIMBS_PER_XMM, n
	jbe	L(sum)

	sub	$LIMBS_PER_XMM, n
	ja	L(ent)
	jmp	L(lsum)

	ALIGN(16)
L(top):	movdqa	(up), %xmm0
L(ent):	movdqa	16(up), %xmm4

	movdqa	%xmm0, %xmm1
	movdqa	%xmm4, %xmm5
	psrld	$1, %xmm0
	psrld	$1, %xmm4
	pand	mm01010101, %xmm0
	pand	mm01010101, %xmm4
	psubd	%xmm0, %xmm1
	psubd	%xmm4, %xmm5

	movdqa	%xmm1, %xmm0
	movdqa	%xmm5, %xmm4
	psrlq	$2, %xmm1
	psrlq	$2, %xmm5
	pand	mm00110011, %xmm0
	pand	mm00110011, %xmm4
	pand	mm00110011, %xmm1
	pand	mm00110011, %xmm5
	paddq	%xmm0, %xmm1
	paddq	%xmm4, %xmm5

LIMB32(`pxor	zero, zero	')

	add	$32, up
	sub	$LIMBS_PER_2XMM, n

	paddq	%xmm5, %xmm1
	movdqa	%xmm1, %xmm0
	psrlq	$4, %xmm1
	pand	mm00001111, %xmm0
	pand	mm00001111, %xmm1
	paddq	%xmm0, %xmm1

	psadbw	zero, %xmm1
	paddq	%xmm1, %xmm3		C add to grand total

	jnc	L(top)
L(end):
	add	$LIMBS_PER_2XMM, n
	jz	L(rt)
	movdqa	(up), %xmm0
	pxor	%xmm4, %xmm4
	sub	$LIMBS_PER_XMM, n
	jbe	L(sum)
L(lsum):
	movdqa	%xmm0, %xmm4
	movdqa	16(up), %xmm0
L(sum):
	shl	$m4_log2(GMP_LIMB_BYTES), n
	and	$12, n
	pand	(breg,n,4), %xmm0

	movdqa	%xmm0, %xmm1
	movdqa	%xmm4, %xmm5
	psrld	$1, %xmm0
	psrld	$1, %xmm4
	pand	mm01010101, %xmm0
	pand	mm01010101, %xmm4
	psubd	%xmm0, %xmm1
	psubd	%xmm4, %xmm5

	movdqa	%xmm1, %xmm0
	movdqa	%xmm5, %xmm4
	psrlq	$2, %xmm1
	psrlq	$2, %xmm5
	pand	mm00110011, %xmm0
	pand	mm00110011, %xmm4
	pand	mm00110011, %xmm1
	pand	mm00110011, %xmm5
	paddq	%xmm0, %xmm1
	paddq	%xmm4, %xmm5

LIMB32(`pxor	zero, zero	')

	paddq	%xmm5, %xmm1
	movdqa	%xmm1, %xmm0
	psrlq	$4, %xmm1
	pand	mm00001111, %xmm0
	pand	mm00001111, %xmm1
	paddq	%xmm0, %xmm1

	psadbw	zero, %xmm1
	paddq	%xmm1, %xmm3		C add to grand total


C Add the two 64-bit halves of the grand total counter
L(rt):	movdqa	%xmm3, %xmm0
	psrldq	$8, %xmm3
	paddq	%xmm3, %xmm0
	movd	%xmm0, areg		C movq avoided due to gas bug

LIMB32(`pop	%ebx		')
	ret

EPILOGUE()
DEF_OBJECT(dummy,16)
C Three magic constants used for masking out bits
	.byte	0x55,0x55,0x55,0x55,0x55,0x55,0x55,0x55
	.byte	0x55,0x55,0x55,0x55,0x55,0x55,0x55,0x55

	.byte	0x33,0x33,0x33,0x33,0x33,0x33,0x33,0x33
	.byte	0x33,0x33,0x33,0x33,0x33,0x33,0x33,0x33

	.byte	0x0f,0x0f,0x0f,0x0f,0x0f,0x0f,0x0f,0x0f
	.byte	0x0f,0x0f,0x0f,0x0f,0x0f,0x0f,0x0f,0x0f
cnsts:
C Masks for high end of number
	.byte	0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff
	.byte	0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff

	.byte	0xff,0xff,0xff,0xff,0x00,0x00,0x00,0x00
	.byte	0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00

	.byte	0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff
	.byte	0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00

	.byte	0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff
	.byte	0xff,0xff,0xff,0xff,0x00,0x00,0x00,0x00
C Masks for low end of number
	.byte	0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff
	.byte	0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff

	.byte	0x00,0x00,0x00,0x00,0xff,0xff,0xff,0xff
	.byte	0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff

	.byte	0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00
	.byte	0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff

	.byte	0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00
	.byte	0x00,0x00,0x00,0x00,0xff,0xff,0xff,0xff
END_OBJECT(dummy)
ASM_END()
