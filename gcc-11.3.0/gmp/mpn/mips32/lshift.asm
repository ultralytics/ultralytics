dnl  MIPS32 mpn_lshift -- Left shift.

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
C src_ptr	$5
C size		$6
C cnt		$7

ASM_START()
PROLOGUE(mpn_lshift)
	sll	$2,$6,2
	addu	$5,$5,$2	C make r5 point at end of src
	lw	$10,-4($5)	C load first limb
	subu	$13,$0,$7
	addu	$4,$4,$2	C make r4 point at end of res
	addiu	$6,$6,-1
	and	$9,$6,4-1	C number of limbs in first loop
	beq	$9,$0,.L0	C if multiple of 4 limbs, skip first loop
	 srl	$2,$10,$13	C compute function result

	subu	$6,$6,$9

.Loop0:	lw	$3,-8($5)
	addiu	$4,$4,-4
	addiu	$5,$5,-4
	addiu	$9,$9,-1
	sll	$11,$10,$7
	srl	$12,$3,$13
	move	$10,$3
	or	$8,$11,$12
	bne	$9,$0,.Loop0
	 sw	$8,0($4)

.L0:	beq	$6,$0,.Lend
	 nop

.Loop:	lw	$3,-8($5)
	addiu	$4,$4,-16
	addiu	$6,$6,-4
	sll	$11,$10,$7
	srl	$12,$3,$13

	lw	$10,-12($5)
	sll	$14,$3,$7
	or	$8,$11,$12
	sw	$8,12($4)
	srl	$9,$10,$13

	lw	$3,-16($5)
	sll	$11,$10,$7
	or	$8,$14,$9
	sw	$8,8($4)
	srl	$12,$3,$13

	lw	$10,-20($5)
	sll	$14,$3,$7
	or	$8,$11,$12
	sw	$8,4($4)
	srl	$9,$10,$13

	addiu	$5,$5,-16
	or	$8,$14,$9
	bgtz	$6,.Loop
	 sw	$8,0($4)

.Lend:	sll	$8,$10,$7
	j	$31
	sw	$8,-4($4)
EPILOGUE(mpn_lshift)
