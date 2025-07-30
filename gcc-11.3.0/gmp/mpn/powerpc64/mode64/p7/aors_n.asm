dnl  PowerPC-64 mpn_add_n, mpn_sub_n optimised for POWER7.

dnl  Copyright 2013 Free Software Foundation, Inc.

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

C		   cycles/limb
C POWER3/PPC630		 ?
C POWER4/PPC970		 ?
C POWER5		 ?
C POWER6		 ?
C POWER7		 2.18

C This is a tad bit slower than the cnd_aors_n.asm code, which is of course an
C anomaly.

ifdef(`OPERATION_add_n',`
  define(ADDSUBC,	adde)
  define(ADDSUB,	addc)
  define(func,		mpn_add_n)
  define(func_nc,	mpn_add_nc)
  define(GENRVAL,	`addi	r3, r3, 1')
  define(SETCBR,	`addic	r0, $1, -1')
  define(CLRCB,		`addic	r0, r0, 0')
')
ifdef(`OPERATION_sub_n',`
  define(ADDSUBC,	subfe)
  define(ADDSUB,	subfc)
  define(func,		mpn_sub_n)
  define(func_nc,	mpn_sub_nc)
  define(GENRVAL,	`neg	r3, r3')
  define(SETCBR,	`subfic	r0, $1, 0')
  define(CLRCB,		`addic	r0, r1, -1')
')

MULFUNC_PROLOGUE(mpn_add_n mpn_add_nc mpn_sub_n mpn_sub_nc)

C INPUT PARAMETERS
define(`rp',	`r3')
define(`up',	`r4')
define(`vp',	`r5')
define(`n',	`r6')

ASM_START()
PROLOGUE(func_nc)
	SETCBR(r7)
	b	L(ent)
EPILOGUE()

PROLOGUE(func)
	CLRCB
L(ent):
	andi.	r7, n, 1
	beq	L(bx0)

L(bx1):	ld	r7, 0(up)
	ld	r9, 0(vp)
	ADDSUBC	r11, r9, r7
	std	r11, 0(rp)
	cmpldi	cr6, n, 1
	beq	cr6, L(end)
	addi	up, up, 8
	addi	vp, vp, 8
	addi	rp, rp, 8

L(bx0):	addi	r0, n, 2	C compute branch...
	srdi	r0, r0, 2	C ...count
	mtctr	r0

	andi.	r7, n, 2
	bne	L(mid)

	addi	up, up, 16
	addi	vp, vp, 16
	addi	rp, rp, 16

	ALIGN(32)
L(top):	ld	r6, -16(up)
	ld	r7, -8(up)
	ld	r8, -16(vp)
	ld	r9, -8(vp)
	ADDSUBC	r10, r8, r6
	ADDSUBC	r11, r9, r7
	std	r10, -16(rp)
	std	r11, -8(rp)
L(mid):	ld	r6, 0(up)
	ld	r7, 8(up)
	ld	r8, 0(vp)
	ld	r9, 8(vp)
	ADDSUBC	r10, r8, r6
	ADDSUBC	r11, r9, r7
	std	r10, 0(rp)
	std	r11, 8(rp)
	addi	up, up, 32
	addi	vp, vp, 32
	addi	rp, rp, 32
	bdnz	L(top)

L(end):	subfe	r3, r0, r0	C -cy
	GENRVAL
	blr
EPILOGUE()
