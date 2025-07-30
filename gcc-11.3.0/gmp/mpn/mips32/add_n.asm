dnl  MIPS32 mpn_add_n -- Add two limb vectors of the same length > 0 and store
dnl  sum in a third limb vector.

dnl  Copyright 1995, 2000, 2002 Free Software Foundation, Inc.

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

C INPUT PARAMETERS
C res_ptr	$4
C s1_ptr	$5
C s2_ptr	$6
C size		$7

ASM_START()
PROLOGUE(mpn_add_n)

	lw	$10,0($5)
	lw	$11,0($6)

	addiu	$7,$7,-1
	and	$9,$7,4-1	C number of limbs in first loop
	beq	$9,$0,.L0	C if multiple of 4 limbs, skip first loop
	 move	$2,$0

	subu	$7,$7,$9

.Loop0:	addiu	$9,$9,-1
	lw	$12,4($5)
	addu	$11,$11,$2
	lw	$13,4($6)
	sltu	$8,$11,$2
	addu	$11,$10,$11
	sltu	$2,$11,$10
	sw	$11,0($4)
	or	$2,$2,$8

	addiu	$5,$5,4
	addiu	$6,$6,4
	move	$10,$12
	move	$11,$13
	bne	$9,$0,.Loop0
	 addiu	$4,$4,4

.L0:	beq	$7,$0,.Lend
	 nop

.Loop:	addiu	$7,$7,-4

	lw	$12,4($5)
	addu	$11,$11,$2
	lw	$13,4($6)
	sltu	$8,$11,$2
	addu	$11,$10,$11
	sltu	$2,$11,$10
	sw	$11,0($4)
	or	$2,$2,$8

	lw	$10,8($5)
	addu	$13,$13,$2
	lw	$11,8($6)
	sltu	$8,$13,$2
	addu	$13,$12,$13
	sltu	$2,$13,$12
	sw	$13,4($4)
	or	$2,$2,$8

	lw	$12,12($5)
	addu	$11,$11,$2
	lw	$13,12($6)
	sltu	$8,$11,$2
	addu	$11,$10,$11
	sltu	$2,$11,$10
	sw	$11,8($4)
	or	$2,$2,$8

	lw	$10,16($5)
	addu	$13,$13,$2
	lw	$11,16($6)
	sltu	$8,$13,$2
	addu	$13,$12,$13
	sltu	$2,$13,$12
	sw	$13,12($4)
	or	$2,$2,$8

	addiu	$5,$5,16
	addiu	$6,$6,16

	bne	$7,$0,.Loop
	 addiu	$4,$4,16

.Lend:	addu	$11,$11,$2
	sltu	$8,$11,$2
	addu	$11,$10,$11
	sltu	$2,$11,$10
	sw	$11,0($4)
	j	$31
	or	$2,$2,$8
EPILOGUE(mpn_add_n)
