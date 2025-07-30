dnl  PowerPC-32/VMX and PowerPC-64/VMX mpn_popcount.

dnl  Copyright 2006, 2010 Free Software Foundation, Inc.

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

C                   cycles/limb
C 7400,7410 (G4):       ?
C 744x,745x (G4+):      1.125
C 970 (G5):             2.25

C TODO
C  * Rewrite the awkward huge n outer loop code.
C  * Two lvx, two vperm, and two vxor could make us a similar hamdist.
C  * Compress cnsts table in 64-bit mode, only half the values are needed.

define(`GMP_LIMB_BYTES', eval(GMP_LIMB_BITS/8))
define(`LIMBS_PER_VR',  eval(16/GMP_LIMB_BYTES))
define(`LIMBS_PER_2VR', eval(32/GMP_LIMB_BYTES))

define(`OPERATION_popcount')

define(`ap',	`r3')
define(`n',	`r4')

define(`rtab',	`v10')
define(`cnt4',	`v11')

ifelse(GMP_LIMB_BITS,32,`
	define(`LIMB32',`	$1')
	define(`LIMB64',`')
',`
	define(`LIMB32',`')
	define(`LIMB64',`	$1')
')

C The inner loop handles up to 2^34 bits, i.e., 2^31 64-limbs, due to overflow
C in vsum4ubs.  For large operands, we work in chunks, of size LIMBS_PER_CHUNK.
define(`LIMBS_PER_CHUNK', 0x1000)
define(`LIMBS_CHUNK_THRES', 0x1001)

ASM_START()
PROLOGUE(mpn_popcount,toc)
	mfspr	r10, 256
	oris	r0, r10, 0xfffc		C Set VRSAVE bit 0-13
	mtspr	256, r0

ifdef(`HAVE_ABI_mode32',
`	rldicl	n, n, 0, 32')		C zero extend n

C Load various constants into vector registers
	LEAL(	r11, cnsts)
	li	r12, 16
	vspltisb cnt4, 4		C 0x0404...04 used as shift count

	li	r7, 160
	lvx	rtab, 0, r11

LIMB64(`lis	r0, LIMBS_CHUNK_THRES	')
LIMB64(`cmpd	cr7, n, r0		')

	lvx	v0, 0, ap
	addi	r7, r11, 80
	rlwinm	r6, ap, 2,26,29
	lvx	v8, r7, r6
	vand	v0, v0, v8

LIMB32(`rlwinm	r8, ap, 30,30,31	')
LIMB64(`rlwinm	r8, ap, 29,31,31	')
	add	n, n, r8		C compensate n for rounded down `ap'

	vxor	v1, v1, v1
	li	r8, 0			C grand total count

	vxor	v12, v12, v12		C zero total count
	vxor	v13, v13, v13		C zero total count

	addic.	n, n, -LIMBS_PER_VR
	ble	L(sum)

	addic.	n, n, -LIMBS_PER_VR
	ble	L(lsum)

C For 64-bit machines, handle huge n that would overflow vsum4ubs
LIMB64(`ble	cr7, L(small)		')
LIMB64(`addis	r9, n, -LIMBS_PER_CHUNK	') C remaining n
LIMB64(`lis	n, LIMBS_PER_CHUNK	')

	ALIGN(16)
L(small):
LIMB32(`srwi	r7, n, 3	')	C loop count corresponding to n
LIMB64(`srdi	r7, n, 2	')	C loop count corresponding to n
	addi	r7, r7, 1
	mtctr	r7			C copy n to count register
	b	L(ent)

	ALIGN(16)
L(top):
	lvx	v0, 0, ap
L(ent):	lvx	v1, r12, ap
	addi	ap, ap, 32
	vsrb	v8, v0, cnt4
	vsrb	v9, v1, cnt4
	vperm	v2, rtab, rtab, v0
	vperm	v3, rtab, rtab, v8
	vperm	v4, rtab, rtab, v1
	vperm	v5, rtab, rtab, v9
	vaddubm	v6, v2, v3
	vaddubm	v7, v4, v5
	vsum4ubs v12, v6, v12
	vsum4ubs v13, v7, v13
	bdnz	L(top)

	andi.	n, n, eval(LIMBS_PER_2VR-1)
	beq	L(rt)

	lvx	v0, 0, ap
	vxor	v1, v1, v1
	cmpwi	n, LIMBS_PER_VR
	ble	L(sum)
L(lsum):
	vor	v1, v0, v0
	lvx	v0, r12, ap
L(sum):
LIMB32(`rlwinm	r6, n, 4,26,27	')
LIMB64(`rlwinm	r6, n, 5,26,26	')
	addi	r7, r11, 16
	lvx	v8, r7, r6
	vand	v0, v0, v8
	vsrb	v8, v0, cnt4
	vsrb	v9, v1, cnt4
	vperm	v2, rtab, rtab, v0
	vperm	v3, rtab, rtab, v8
	vperm	v4, rtab, rtab, v1
	vperm	v5, rtab, rtab, v9
	vaddubm	v6, v2, v3
	vaddubm	v7, v4, v5
	vsum4ubs v12, v6, v12
	vsum4ubs v13, v7, v13

	ALIGN(16)
L(rt):	vadduwm	v3, v12, v13
	li	r7, -16			C FIXME: does all ppc32 and ppc64 ABIs
	stvx	v3, r7, r1		C FIXME: ...support storing below sp?

	lwz	r7, -16(r1)
	add	r8, r8, r7
	lwz	r7, -12(r1)
	add	r8, r8, r7
	lwz	r7, -8(r1)
	add	r8, r8, r7
	lwz	r7, -4(r1)
	add	r8, r8, r7

C Handle outer loop for huge n.  We inherit cr7 and r0 from above.
LIMB64(`ble	cr7, L(ret)
	vxor	v12, v12, v12		C zero total count
	vxor	v13, v13, v13		C zero total count
	mr	n, r9
	cmpd	cr7, n, r0
	ble	cr7, L(2)
	addis	r9, n, -LIMBS_PER_CHUNK	C remaining n
	lis	n, LIMBS_PER_CHUNK
L(2):	srdi	r7, n, 2		C loop count corresponding to n
	mtctr	r7			C copy n to count register
	b	L(top)
')

	ALIGN(16)
L(ret):	mr	r3, r8
	mtspr	256, r10
	blr
EPILOGUE()

DEF_OBJECT(cnsts,16)
C Counts for vperm
	.byte	0x00,0x01,0x01,0x02,0x01,0x02,0x02,0x03
	.byte	0x01,0x02,0x02,0x03,0x02,0x03,0x03,0x04
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
END_OBJECT(cnsts)
ASM_END()
