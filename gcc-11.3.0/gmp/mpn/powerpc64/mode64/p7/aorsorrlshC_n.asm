dnl  PowerPC-64 mpn_addlshC_n, mpn_sublshC_n, mpn_rsblshC_n.

dnl  Copyright 2003, 2005, 2009, 2010, 2013 Free Software Foundation, Inc.

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

C                  cycles/limb
C POWER3/PPC630          ?
C POWER4/PPC970          ?
C POWER5                 ?
C POWER6                 ?
C POWER7                 2.5

C INPUT PARAMETERS
define(`rp', `r3')
define(`up', `r4')
define(`vp', `r5')
define(`n',  `r6')

ifdef(`DO_add', `
  define(`ADDSUBC',	`addc	$1, $2, $3')
  define(`ADDSUBE',	`adde	$1, $2, $3')
  define(INITCY,	`addic	$1, r1, 0')
  define(RETVAL,	`addze	r3, $1')
  define(`func',	mpn_addlsh`'LSH`'_n)')
ifdef(`DO_sub', `
  define(`ADDSUBC',	`subfc	$1, $2, $3')
  define(`ADDSUBE',	`subfe	$1, $2, $3')
  define(INITCY,	`addic	$1, r1, -1')
  define(RETVAL,	`subfze	r3, $1
			neg	r3, r3')
  define(`func',	mpn_sublsh`'LSH`'_n)')
ifdef(`DO_rsb', `
  define(`ADDSUBC',	`subfc	$1, $3, $2')
  define(`ADDSUBE',	`subfe	$1, $3, $2')
  define(INITCY,	`addic	$1, r1, -1')
  define(RETVAL,	`addme	r3, $1')
  define(`func',	mpn_rsblsh`'LSH`'_n)')

define(`s0', `r0')  define(`s1', `r9')
define(`u0', `r6')  define(`u1', `r7')
define(`v0', `r10') define(`v1', `r11')


ASM_START()
PROLOGUE(func)
	rldic	r7, n, 3, 59
	add	up, up, r7
	add	vp, vp, r7
	add	rp, rp, r7

ifdef(`DO_add', `
	addic	r0, n, 3	C set cy flag as side effect
',`
	subfc	r0, r0, r0	C set cy flag
	addi	r0, n, 3
')
	srdi	r0, r0, 2
	mtctr	r0

	andi.	r0, n, 1
	beq	L(bx0)

L(bx1):	andi.	r0, n, 2
	li	s0, 0
	bne	L(lo3)
	b	L(lo1)

L(bx0):	andi.	r0, n, 2
	li	s1, 0
	bne	L(lo2)

	ALIGN(32)
L(top):	addi	rp, rp, 32
	ld	v0, 0(vp)
	addi	vp, vp, 32
	rldimi	s1, v0, LSH, 0
	ld	u0, 0(up)
	addi	up, up, 32
	srdi	s0, v0, RSH
	ADDSUBE(s1, s1, u0)
	std	s1, -32(rp)
L(lo3):	ld	v1, -24(vp)
	rldimi	s0, v1, LSH, 0
	ld	u1, -24(up)
	srdi	s1, v1, RSH
	ADDSUBE(s0, s0, u1)
	std	s0, -24(rp)
L(lo2):	ld	v0, -16(vp)
	rldimi	s1, v0, LSH, 0
	ld	u0, -16(up)
	srdi	s0, v0, RSH
	ADDSUBE(s1, s1, u0)
	std	s1, -16(rp)
L(lo1):	ld	v1, -8(vp)
	rldimi	s0, v1, LSH, 0
	ld	u1, -8(up)
	srdi	s1, v1, RSH
	ADDSUBE(s0, s0, u1)
	std	s0, -8(rp)
	bdnz	L(top)		C decrement CTR and loop back

	RETVAL(	s1)
	blr
EPILOGUE()
