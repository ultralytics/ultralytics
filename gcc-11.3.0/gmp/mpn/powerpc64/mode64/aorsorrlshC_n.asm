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
C POWER3/PPC630          1.83   (1.5 c/l should be possible)
C POWER4/PPC970          3      (2.0 c/l should be possible)
C POWER5                 3
C POWER6              3.5-47
C POWER7                 3

C STATUS
C  * Try combining upx+up, and vpx+vp.
C  * The worst case 47 c/l for POWER6 happens if the 3rd operand for ldx is
C    greater than the 2nd operand.  Yes, this addition is non-commutative wrt
C    performance.

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

define(`rpx', `r6')
define(`upx', `r7')
define(`vpx', `r12')

define(`s0', `r0')  define(`s1', `r9')
define(`u0', `r8')
define(`v0', `r10') define(`v1', `r11')


ASM_START()
PROLOGUE(func)
	cmpldi	cr0, n, 13
	bgt	L(big)

	mtctr	n		C copy n in ctr
	INITCY(	r0)		C clear cy

	ld	v0, 0(vp)	C load v limb
	ld	u0, 0(up)	C load u limb
	addi	up, up, -8	C update up
	addi	rp, rp, -8	C update rp
	sldi	s1, v0, LSH
	bdz	L(ex1)		C If done, skip loop

	ALIGN(16)
L(lo0):	ld	v1, 8(vp)	C load v limb
	ADDSUBE(s1, s1, u0)	C add limbs with cy, set cy
	ldu	u0, 16(up)	C load u limb and update up
	srdi	s0, v0, RSH	C shift down previous v limb
	std	s1, 8(rp)	C store result limb
	rldimi	s0, v1, LSH, 0	C left shift v limb and merge with prev v limb
	bdz	L(ex0)		C decrement ctr and exit if done
	ldu	v0, 16(vp)	C load v limb and update vp
	ADDSUBE(s0, s0, u0)	C add limbs with cy, set cy
	ld	u0, 8(up)	C load u limb
	srdi	s1, v1, RSH	C shift down previous v limb
	stdu	s0, 16(rp)	C store result limb and update rp
	rldimi	s1, v0, LSH, 0	C left shift v limb and merge with prev v limb
	bdnz	L(lo0)		C decrement ctr and loop back

L(ex1):	ADDSUBE(r7, s1, u0)
	std	r7, 8(rp)	C store last result limb
	srdi	r0, v0, RSH
	RETVAL(	r0)
	blr
L(ex0):	ADDSUBE(r7, s0, u0)
	std	r7, 16(rp)	C store last result limb
	srdi	r0, v1, RSH
	RETVAL(	r0)
	blr


L(big):	rldicl.	r0, n, 0,63	C r0 = n & 1, set cr0
	addi	r6, n, -1	C ...for ctr
	srdi	r6, r6, 1	C ...for ctr
	mtctr	r6		C copy count into ctr
	beq	cr0, L(b0)

L(b1):	ld	v1, 0(vp)
	ld	u0, 0(up)
	sldi	s1, v1, LSH
	srdi	s0, v1, RSH
	ld	v0, 8(vp)
	ADDSUBC(s1, s1, u0)	C add limbs without cy, set cy
	addi	rpx, rp, -16
	addi	rp, rp, -8
	sub	upx, up, rp
	sub	vpx, vp, rp
	sub	up, up, rpx
	sub	vp, vp, rpx
	addi	up, up, 8
	addi	upx, upx, 16
	addi	vp, vp, 16
	addi	vpx, vpx, 24
	b	L(mid)

L(b0):	ld	v0, 0(vp)
	ld	u0, 0(up)
	sldi	s0, v0, LSH
	srdi	s1, v0, RSH
	ld	v1, 8(vp)
	ADDSUBC(s0, s0, u0)	C add limbs without cy, set cy
	addi	rpx, rp, -8
	addi	rp, rp, -16
	sub	upx, up, rpx
	sub	vpx, vp, rpx
	sub	up, up, rp
	sub	vp, vp, rp
	addi	up, up, 8
	addi	upx, upx, 16
	addi	vp, vp, 16
	addi	vpx, vpx, 24

	ALIGN(32)
L(top):	ldx	u0, rp, up
	ldx	v0, rp, vp
	rldimi	s1, v1, LSH, 0
	stdu	s0, 16(rp)
	srdi	s0, v1, RSH
	ADDSUBE(s1, s1, u0)	C add limbs with cy, set cy
L(mid):	ldx	u0, rpx, upx
	ldx	v1, rpx, vpx
	rldimi	s0, v0, LSH, 0
	stdu	s1, 16(rpx)
	srdi	s1, v0, RSH
	ADDSUBE(s0, s0, u0)	C add limbs with cy, set cy
	bdnz	L(top)		C decrement CTR and loop back

	ldx	u0, rp, up
	rldimi	s1, v1, LSH, 0
	std	s0, 16(rp)
	srdi	s0, v1, RSH
	ADDSUBE(s1, s1, u0)	C add limbs with cy, set cy
	std	s1, 24(rp)

	RETVAL(	s0)
	blr
EPILOGUE()
