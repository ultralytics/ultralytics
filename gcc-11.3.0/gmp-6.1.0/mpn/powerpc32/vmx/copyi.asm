dnl  PowerPC-32/VMX and PowerPC-64/VMX mpn_copyi.

dnl  Copyright 2006 Free Software Foundation, Inc.

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

C                16-byte coaligned      unaligned
C                   cycles/limb        cycles/limb
C 7400,7410 (G4):       0.5                0.64
C 744x,745x (G4+):      0.75               0.82
C 970 (G5):             0.78               1.02		(64-bit limbs)

C STATUS
C  * Works for all sizes and alignments.

C TODO
C  * Optimize unaligned case.  Some basic tests with 2-way and 4-way unrolling
C    indicate that we can reach 0.56 c/l for 7400, 0.75 c/l for 745x, and 0.80
C    c/l for 970.
C  * Consider using VMX instructions also for head and tail, by using some
C    read-modify-write tricks.
C  * The VMX code is used from the smallest sizes it handles, but measurements
C    show a large speed bump at the cutoff points.  Small copying (perhaps
C    using some read-modify-write technique) should be optimized.
C  * Make an mpn_com based on this code.

define(`GMP_LIMB_BYTES', eval(GMP_LIMB_BITS/8))
define(`LIMBS_PER_VR',  eval(16/GMP_LIMB_BYTES))
define(`LIMBS_PER_2VR', eval(32/GMP_LIMB_BYTES))


ifelse(GMP_LIMB_BITS,32,`
	define(`LIMB32',`	$1')
	define(`LIMB64',`')
',`
	define(`LIMB32',`')
	define(`LIMB64',`	$1')
')

C INPUT PARAMETERS
define(`rp',	`r3')
define(`up',	`r4')
define(`n',	`r5')

define(`us',	`v4')


ASM_START()
PROLOGUE(mpn_copyi)

LIMB32(`cmpi	cr7, n, 11	')
LIMB64(`cmpdi	cr7, n, 5	')
	bge	cr7, L(big)

	or.	r0, n, n
	beqlr	cr0

C Handle small cases with plain operations
	mtctr	n
L(topS):
LIMB32(`lwz	r0, 0(up)	')
LIMB64(`ld	r0, 0(up)	')
	addi	up, up, GMP_LIMB_BYTES
LIMB32(`stw	r0, 0(rp)	')
LIMB64(`std	r0, 0(rp)	')
	addi	rp, rp, GMP_LIMB_BYTES
	bdnz	L(topS)
	blr

C Handle large cases with VMX operations
L(big):
	mfspr	r12, 256
	oris	r0, r12, 0xf800		C Set VRSAVE bit 0-4
	mtspr	256, r0

LIMB32(`rlwinm.	r7, rp, 30,30,31')	C (rp >> 2) mod 4
LIMB64(`rlwinm.	r7, rp, 29,31,31')	C (rp >> 3) mod 2
	beq	L(rp_aligned)

	subfic	r7, r7, LIMBS_PER_VR
	subf	n, r7, n
L(top0):
LIMB32(`lwz	r0, 0(up)	')
LIMB64(`ld	r0, 0(up)	')
	addi	up, up, GMP_LIMB_BYTES
LIMB32(`addic.	r7, r7, -1	')
LIMB32(`stw	r0, 0(rp)	')
LIMB64(`std	r0, 0(rp)	')
	addi	rp, rp, GMP_LIMB_BYTES
LIMB32(`bne	L(top0)		')

L(rp_aligned):

LIMB32(`rlwinm.	r0, up, 30,30,31')	C (up >> 2) mod 4
LIMB64(`rlwinm.	r0, up, 29,31,31')	C (up >> 3) mod 2

LIMB64(`srdi	r7, n, 2	')	C loop count corresponding to n
LIMB32(`srwi	r7, n, 3	')	C loop count corresponding to n
	mtctr	r7			C copy n to count register

	li	r10, 16

	beq	L(up_aligned)

	lvsl	us, 0, up

LIMB32(`andi.	r0, n, 0x4	')
LIMB64(`andi.	r0, n, 0x2	')
	beq	L(1)
	lvx	v0, 0, up
	lvx	v2, r10, up
	vperm	v3, v0, v2, us
	stvx	v3, 0, rp
	addi	up, up, 32
	addi	rp, rp, 16
	b	L(lpu)
L(1):	lvx	v2, 0, up
	addi	up, up, 16
	b	L(lpu)

	ALIGN(32)
L(lpu):	lvx	v0, 0, up
	vperm	v3, v2, v0, us
	stvx	v3, 0, rp
	lvx	v2, r10, up
	addi	up, up, 32
	vperm	v3, v0, v2, us
	stvx	v3, r10, rp
	addi	rp, rp, 32
	bdnz	L(lpu)

	addi	up, up, -16
	b	L(tail)

L(up_aligned):

LIMB32(`andi.	r0, n, 0x4	')
LIMB64(`andi.	r0, n, 0x2	')
	beq	L(lpa)
	lvx	v0, 0,   up
	stvx	v0, 0,   rp
	addi	up, up, 16
	addi	rp, rp, 16
	b	L(lpa)

	ALIGN(32)
L(lpa):	lvx	v0, 0,   up
	lvx	v1, r10, up
	addi	up, up, 32
	nop
	stvx	v0, 0,   rp
	stvx	v1, r10, rp
	addi	rp, rp, 32
	bdnz	L(lpa)

L(tail):
LIMB32(`rlwinm.	r7, n, 0,30,31	')	C r7 = n mod 4
LIMB64(`rlwinm.	r7, n, 0,31,31	')	C r7 = n mod 2
	beq	L(ret)
LIMB32(`li	r10, 0		')
L(top2):
LIMB32(`lwzx	r0, r10, up	')
LIMB64(`ld	r0, 0(up)	')
LIMB32(`addic.	r7, r7, -1	')
LIMB32(`stwx	r0, r10, rp	')
LIMB64(`std	r0, 0(rp)	')
LIMB32(`addi	r10, r10, GMP_LIMB_BYTES')
LIMB32(`bne	L(top2)		')

L(ret):	mtspr	256, r12
	blr
EPILOGUE()
