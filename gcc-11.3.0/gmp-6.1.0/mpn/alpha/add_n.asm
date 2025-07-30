dnl  Alpha mpn_add_n -- Add two limb vectors of the same length > 0 and
dnl  store sum in a third limb vector.

dnl  Copyright 1995, 1999, 2000, 2005, 2011 Free Software Foundation, Inc.

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

C      cycles/limb
C EV4:     ?
C EV5:     4.75
C EV6:     3

dnl  INPUT PARAMETERS
dnl  res_ptr	r16
dnl  s1_ptr	r17
dnl  s2_ptr	r18
dnl  size	r19

ASM_START()
PROLOGUE(mpn_add_nc)
	bis	r20,r31,r25
	br	L(com)
EPILOGUE()
PROLOGUE(mpn_add_n)
	bis	r31,r31,r25		C clear cy
L(com):	subq	r19,4,r19		C decr loop cnt
	blt	r19,$Lend2		C if less than 4 limbs, goto 2nd loop
C Start software pipeline for 1st loop
	ldq	r0,0(r18)
	ldq	r4,0(r17)
	ldq	r1,8(r18)
	ldq	r5,8(r17)
	addq	r17,32,r17		C update s1_ptr
	addq	r0,r4,r28		C 1st main add
	ldq	r2,16(r18)
	addq	r25,r28,r20		C 1st carry add
	ldq	r3,24(r18)
	cmpult	r28,r4,r8		C compute cy from last add
	ldq	r6,-16(r17)
	cmpult	r20,r28,r25		C compute cy from last add
	ldq	r7,-8(r17)
	bis	r8,r25,r25		C combine cy from the two adds
	subq	r19,4,r19		C decr loop cnt
	addq	r1,r5,r28		C 2nd main add
	addq	r18,32,r18		C update s2_ptr
	addq	r28,r25,r21		C 2nd carry add
	cmpult	r28,r5,r8		C compute cy from last add
	blt	r19,$Lend1		C if less than 4 limbs remain, jump
C 1st loop handles groups of 4 limbs in a software pipeline
	ALIGN(16)
$Loop:	cmpult	r21,r28,r25		C compute cy from last add
	ldq	r0,0(r18)
	bis	r8,r25,r25		C combine cy from the two adds
	ldq	r1,8(r18)
	addq	r2,r6,r28		C 3rd main add
	ldq	r4,0(r17)
	addq	r28,r25,r22		C 3rd carry add
	ldq	r5,8(r17)
	cmpult	r28,r6,r8		C compute cy from last add
	cmpult	r22,r28,r25		C compute cy from last add
	stq	r20,0(r16)
	bis	r8,r25,r25		C combine cy from the two adds
	stq	r21,8(r16)
	addq	r3,r7,r28		C 4th main add
	addq	r28,r25,r23		C 4th carry add
	cmpult	r28,r7,r8		C compute cy from last add
	cmpult	r23,r28,r25		C compute cy from last add
		addq	r17,32,r17		C update s1_ptr
	bis	r8,r25,r25		C combine cy from the two adds
		addq	r16,32,r16		C update res_ptr
	addq	r0,r4,r28		C 1st main add
	ldq	r2,16(r18)
	addq	r25,r28,r20		C 1st carry add
	ldq	r3,24(r18)
	cmpult	r28,r4,r8		C compute cy from last add
	ldq	r6,-16(r17)
	cmpult	r20,r28,r25		C compute cy from last add
	ldq	r7,-8(r17)
	bis	r8,r25,r25		C combine cy from the two adds
	subq	r19,4,r19		C decr loop cnt
	stq	r22,-16(r16)
	addq	r1,r5,r28		C 2nd main add
	stq	r23,-8(r16)
	addq	r25,r28,r21		C 2nd carry add
		addq	r18,32,r18		C update s2_ptr
	cmpult	r28,r5,r8		C compute cy from last add
	bge	r19,$Loop
C Finish software pipeline for 1st loop
$Lend1:	cmpult	r21,r28,r25		C compute cy from last add
	bis	r8,r25,r25		C combine cy from the two adds
	addq	r2,r6,r28		C 3rd main add
	addq	r28,r25,r22		C 3rd carry add
	cmpult	r28,r6,r8		C compute cy from last add
	cmpult	r22,r28,r25		C compute cy from last add
	stq	r20,0(r16)
	bis	r8,r25,r25		C combine cy from the two adds
	stq	r21,8(r16)
	addq	r3,r7,r28		C 4th main add
	addq	r28,r25,r23		C 4th carry add
	cmpult	r28,r7,r8		C compute cy from last add
	cmpult	r23,r28,r25		C compute cy from last add
	bis	r8,r25,r25		C combine cy from the two adds
	addq	r16,32,r16		C update res_ptr
	stq	r22,-16(r16)
	stq	r23,-8(r16)
$Lend2:	addq	r19,4,r19		C restore loop cnt
	beq	r19,$Lret
C Start software pipeline for 2nd loop
	ldq	r0,0(r18)
	ldq	r4,0(r17)
	subq	r19,1,r19
	beq	r19,$Lend0
C 2nd loop handles remaining 1-3 limbs
	ALIGN(16)
$Loop0:	addq	r0,r4,r28		C main add
	ldq	r0,8(r18)
	cmpult	r28,r4,r8		C compute cy from last add
	ldq	r4,8(r17)
	addq	r28,r25,r20		C carry add
	addq	r18,8,r18
	addq	r17,8,r17
	stq	r20,0(r16)
	cmpult	r20,r28,r25		C compute cy from last add
	subq	r19,1,r19		C decr loop cnt
	bis	r8,r25,r25		C combine cy from the two adds
	addq	r16,8,r16
	bne	r19,$Loop0
$Lend0:	addq	r0,r4,r28		C main add
	addq	r28,r25,r20		C carry add
	cmpult	r28,r4,r8		C compute cy from last add
	cmpult	r20,r28,r25		C compute cy from last add
	stq	r20,0(r16)
	bis	r8,r25,r25		C combine cy from the two adds

$Lret:	bis	r25,r31,r0		C return cy
	ret	r31,(r26),1
EPILOGUE()
ASM_END()
