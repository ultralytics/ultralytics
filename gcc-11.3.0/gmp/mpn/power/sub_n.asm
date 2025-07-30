dnl  IBM POWER mpn_sub_n -- Subtract two limb vectors of equal, non-zero
dnl  length.

dnl  Copyright 1992, 1994-1996, 1999-2001, 2005 Free Software Foundation, Inc.

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


dnl  INPUT PARAMETERS
dnl  res_ptr	r3
dnl  s1_ptr	r4
dnl  s2_ptr	r5
dnl  size	r6

include(`../config.m4')

ASM_START()
PROLOGUE(mpn_sub_n)
	andil.	10,6,1		C odd or even number of limbs?
	l	8,0(4)		C load least significant s1 limb
	l	0,0(5)		C load least significant s2 limb
	cal	3,-4(3)		C offset res_ptr, it's updated before it's used
	sri	10,6,1		C count for unrolled loop
	sf	7,0,8		C subtract least significant limbs, set cy
	mtctr	10		C copy count into CTR
	beq	0,Leven		C branch if even # of limbs (# of limbs >= 2)

C We have an odd # of limbs.  Add the first limbs separately.
	cmpi	1,10,0		C is count for unrolled loop zero?
	bc	4,6,L1		C bne cr1,L1 (misassembled by gas)
	st	7,4(3)
	sfe	3,0,0		C load !cy into ...
	sfi	3,3,0		C ... return value register
	br			C return

C We added least significant limbs.  Now reload the next limbs to enter loop.
L1:	lu	8,4(4)		C load s1 limb and update s1_ptr
	lu	0,4(5)		C load s2 limb and update s2_ptr
	stu	7,4(3)
	sfe	7,0,8		C subtract limbs, set cy
Leven:	lu	9,4(4)		C load s1 limb and update s1_ptr
	lu	10,4(5)		C load s2 limb and update s2_ptr
	bdz	Lend		C If done, skip loop

Loop:	lu	8,4(4)		C load s1 limb and update s1_ptr
	lu	0,4(5)		C load s2 limb and update s2_ptr
	sfe	11,10,9		C subtract previous limbs with cy, set cy
	stu	7,4(3)		C
	lu	9,4(4)		C load s1 limb and update s1_ptr
	lu	10,4(5)		C load s2 limb and update s2_ptr
	sfe	7,0,8		C subtract previous limbs with cy, set cy
	stu	11,4(3)		C
	bdn	Loop		C decrement CTR and loop back

Lend:	sfe	11,10,9		C subtract limbs with cy, set cy
	st	7,4(3)		C
	st	11,8(3)		C
	sfe	3,0,0		C load !cy into ...
	sfi	3,3,0		C ... return value register
	br
EPILOGUE(mpn_sub_n)
