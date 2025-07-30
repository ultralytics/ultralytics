dnl  PowerPC-32 mpn_mod_34lsub1 -- mpn remainder mod 2^24-1.

dnl  Copyright 2002, 2003, 2005-2007, 2012 Free Software Foundation, Inc.

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


C                cycles/limb
C 603e:              -
C 604e:              -
C 75x (G3):          -
C 7400,7410 (G4):    1          simple load-use scheduling results in 0.75
C 744x,745x (G4+):   0.75
C ppc970:            0.75
C power4:            -
C power5:            -

C TODO
C  * Either start using the low-end masking constants, or remove them.
C  * Merge multiple feed-in cases into a parameterized code block.
C  * Reduce register usage.  It should be possible to almost halve it.

define(`up', `r3')
define(`n', `r4')

define(`a0', `v3')
define(`a1', `v4')
define(`a2', `v5')
define(`c0', `v6')
define(`c1', `v7')
define(`c2', `v8')
define(`z',  `v9')
define(`x0', `v10')
define(`x1', `v11')
define(`x2', `v12')
define(`x3', `v13')
define(`pv', `v14')
define(`y0', `v0')
define(`y1', `v1')
define(`y2', `v2')
define(`y3', `v15')

ASM_START()
PROLOGUE(mpn_mod_34lsub1)
	cmpwi	cr0, n, 20		C tuned cutoff point
	bge	L(large)

	li	r9, 0			C result accumulator
	mulli	r10, n, 0xb		C 0xb = ceil(32/3)
	srwi.	r10, r10, 5		C r10 = floor(n/3), n < 32
	beq	L(small_tail)
	mtctr	r10
	lwz	r6, 0(up)
	lwz	r7, 4(up)
	lwzu	r8, 8(up)
	subf	n, r10, n
	subf	n, r10, n
	subf	n, r10, n
	bdz	L(small_end)

	ALIGN(16)
L(los):	rlwinm	r0, r6, 0,8,31
	add	r9, r9, r0		C add 24b from u0
	srwi	r0, r6, 24
	lwz	r6, 4(up)
	rlwimi	r0, r7, 8, 0x00ffff00	C --111100
	add	r9, r9, r0		C add 8b from u0 and 16b from u1
	srwi	r0, r7, 16
	lwz	r7, 8(up)
	rlwimi	r0, r8, 16, 0x00ff0000	C --221111
	add	r9, r9, r0		C add 16b from u1 and 8b from u2
	srwi	r0, r8, 8		C --222222
	lwzu	r8, 12(up)
	add	r9, r9, r0		C add 24b from u2
	bdnz	L(los)
L(small_end):
	rlwinm	r0, r6, 0,8,31
	add	r9, r9, r0		C add 24b from u0
	srwi	r0, r6, 24
	rlwimi	r0, r7, 8, 0x00ffff00	C --111100
	add	r9, r9, r0		C add 8b from u0 and 16b from u1
	srwi	r0, r7, 16
	rlwimi	r0, r8, 16, 0x00ff0000	C --221111
	add	r9, r9, r0		C add 16b from u1 and 8b from u2
	srwi	r0, r8, 8		C --222222
	add	r9, r9, r0		C add 24b from u2

	addi	up, up, 4
	rlwinm	r0, r9, 0,8,31
	srwi	r9, r9, 24
	add	r9, r9, r0

L(small_tail):
	cmpi	cr0, n, 1
	blt	L(ret)

	lwz	r6, 0(up)
	rlwinm	r0, r6, 0,8,31
	srwi	r6, r6, 24
	add	r9, r9, r0
	add	r9, r9, r6

	beq	L(ret)

	lwz	r6, 4(up)
	rlwinm	r0, r6, 8,8,23
	srwi	r6, r6, 16
	add	r9, r9, r0
	add	r9, r9, r6

L(ret):	mr	r3, r9
	blr


L(large):
	stwu	r1, -32(r1)
	mfspr	r10, 256
	oris	r0, r10, 0xffff		C Set VRSAVE bit 0-15
	mtspr	256, r0

	andi.	r7, up, 15
	vxor	a0, v0, v0
	lis	r9, 0xaaaa
	vxor	a1, v0, v0
	ori	r9, r9, 0xaaab
	vxor	a2, v0, v0
	li	r5, 16
	vxor	c0, v0, v0
	li	r6, 32
	vxor	c1, v0, v0
	LEAL(	r11, cnsts)		C CAUTION clobbers r0 for elf, darwin
	vxor	c2, v0, v0
	vxor	z, v0, v0

	beq	L(aligned16)

	cmpwi	cr7, r7, 8
	bge	cr7, L(na4)

	lvx	a2, 0, up
	addi	up, up, 16
	vsldoi	a2, a2, z, 4
	vsldoi	a2, z, a2, 12

	addi	n, n, 9
	mulhwu	r0, n, r9
	srwi	r0, r0, 3		C r0 = floor(n/12)
	mtctr	r0

	mulli	r8, r0, 12
	subf	n, r8, n
	b	L(2)

L(na4):	bne	cr7, L(na8)

	lvx	a1, 0, up
	addi	up, up, -16
	vsldoi	a1, a1, z, 8
	vsldoi	a1, z, a1, 8

	addi	n, n, 6
	mulhwu	r0, n, r9
	srwi	r0, r0, 3		C r0 = floor(n/12)
	mtctr	r0

	mulli	r8, r0, 12
	subf	n, r8, n
	b	L(1)

L(na8):
	lvx	a0, 0, up
	vsldoi	a0, a0, z, 12
	vsldoi	a0, z, a0, 4

	addi	n, n, 3
	mulhwu	r0, n, r9
	srwi	r0, r0, 3		C r0 = floor(n/12)
	mtctr	r0

	mulli	r8, r0, 12
	subf	n, r8, n
	b	L(0)

L(aligned16):
	mulhwu	r0, n, r9
	srwi	r0, r0, 3		C r0 = floor(n/12)
	mtctr	r0

	mulli	r8, r0, 12
	subf	n, r8, n

	lvx	a0, 0, up
L(0):	lvx	a1, r5, up
L(1):	lvx	a2, r6, up
	addi	up, up, 48
L(2):	bdz	L(end)
	li	r12, 256
	li	r9, 288
	ALIGN(32)
L(top):
	lvx	v0, 0, up
	vaddcuw	v10, a0, v0
	vadduwm	a0, a0, v0
	vadduwm	c0, c0, v10

	lvx	v1, r5, up
	vaddcuw	v10, a1, v1
	vadduwm	a1, a1, v1
	vadduwm	c1, c1, v10

	lvx	v2, r6, up
	dcbt	up, r12
	dcbt	up, r9
	addi	up, up, 48
	vaddcuw	v10, a2, v2
	vadduwm	a2, a2, v2
	vadduwm	c2, c2, v10
	bdnz	L(top)

L(end):
C n = 0...11
	cmpwi	cr0, n, 0
	beq	L(sum)
	cmpwi	cr0, n, 4
	ble	L(tail.1..4)
	cmpwi	cr0, n, 8
	ble	L(tail.5..8)

L(tail.9..11):
	lvx	v0, 0, up
	vaddcuw	v10, a0, v0
	vadduwm	a0, a0, v0
	vadduwm	c0, c0, v10

	lvx	v1, r5, up
	vaddcuw	v10, a1, v1
	vadduwm	a1, a1, v1
	vadduwm	c1, c1, v10

	lvx	v2, r6, up

	addi	r8, r11, 96
	rlwinm	r3, n ,4,26,27
	lvx	v11, r3, r8
	vand	v2, v2, v11

	vaddcuw	v10, a2, v2
	vadduwm	a2, a2, v2
	vadduwm	c2, c2, v10
	b	L(sum)

L(tail.5..8):
	lvx	v0, 0, up
	vaddcuw	v10, a0, v0
	vadduwm	a0, a0, v0
	vadduwm	c0, c0, v10

	lvx	v1, r5, up

	addi	r8, r11, 96
	rlwinm	r3, n ,4,26,27
	lvx	v11, r3, r8
	vand	v1, v1, v11

	vaddcuw	v10, a1, v1
	vadduwm	a1, a1, v1
	vadduwm	c1, c1, v10
	b	L(sum)

L(tail.1..4):
	lvx	v0, 0, up

	addi	r8, r11, 96
	rlwinm	r3, n ,4,26,27
	lvx	v11, r3, r8
	vand	v0, v0, v11

	vaddcuw	v10, a0, v0
	vadduwm	a0, a0, v0
	vadduwm	c0, c0, v10

L(sum):	lvx	pv, 0, r11
	vperm	x0, a0, z, pv		C extract 4 24-bit field from a0
	vperm	y0, c2, z, pv
	lvx	pv, r5, r11
	vperm	x1, a1, z, pv		C extract 4 24-bit field from a1
	vperm	y1, c0, z, pv		C extract 4 24-bit field from a1
	lvx	pv, r6, r11
	vperm	x2, a2, z, pv		C extract 4 24-bit field from a1
	vperm	y2, c1, z, pv		C extract 4 24-bit field from a1
	li	r10,  48
	lvx	pv, r10, r11
	vperm	x3, a0, z, pv		C extract remaining/partial a0 fields
	vperm	y3, c2, z, pv		C extract remaining/partial a0 fields
	li	r10,  64
	lvx	pv, r10, r11
	vperm	x3, a1, x3, pv		C insert remaining/partial a1 fields
	vperm	y3, c0, y3, pv		C insert remaining/partial a1 fields
	li	r10,  80
	lvx	pv, r10, r11
	vperm	x3, a2, x3, pv		C insert remaining/partial a2 fields
	vperm	y3, c1, y3, pv		C insert remaining/partial a2 fields

C We now have 4 128-bit accumulators to sum
	vadduwm	x0, x0, x1
	vadduwm	x2, x2, x3
	vadduwm	x0, x0, x2

	vadduwm	y0, y0, y1
	vadduwm	y2, y2, y3
	vadduwm	y0, y0, y2

	vadduwm	x0, x0, y0

C Reduce 32-bit fields
	vsumsws	x0, x0, z

	li	r7, 16
	stvx	x0, r7, r1
	lwz	r3, 28(r1)

	mtspr	256, r10
	addi	r1, r1, 32
	blr
EPILOGUE()

C load	|      v0       |      v1       |      v2       |
C acc	|      a0       |      a1       |      a2       |
C carry	|      c0       |      c1       |      c2       |
C	| 0   1   2   3 | 4   5   6   7 | 8   9  10  11 |  128
C	|---|---|---|---|---|---|---|---|---|---|---|---|   32
C	|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |   24
C	|     |     |     |     |     |     |     |     |   48

C       $---------------$---------------$---------------$---------------$
C       |   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   |
C       |_______________________________________________________________|
C   |           |           |           |           |           |           |
C       <-hi16-> <--- 24 --> <--- 24 --> <--- 24 --> <--- 24 --> <-lo16->


DEF_OBJECT(cnsts,16)
C Permutation vectors in the order they are used above
C #      00   01   02   03    04   05   06   07    08   09   0a   0b    0c   0d   0e   0f
 .byte 0x10,0x01,0x02,0x03, 0x10,0x06,0x07,0x00, 0x10,0x0b,0x04,0x05, 0x10,0x08,0x09,0x0a C a0
 .byte 0x10,0x07,0x00,0x01, 0x10,0x04,0x05,0x06, 0x10,0x09,0x0a,0x0b, 0x10,0x0e,0x0f,0x08 C a1
 .byte 0x10,0x00,0x01,0x02, 0x10,0x05,0x06,0x07, 0x10,0x0a,0x0b,0x04, 0x10,0x0f,0x08,0x09 C a2
 .byte 0x10,0x0d,0x0e,0x0f, 0x10,0x10,0x10,0x0c, 0x10,0x10,0x10,0x10, 0x10,0x10,0x10,0x10 C part a0
 .byte 0x10,0x11,0x12,0x13, 0x10,0x02,0x03,0x17, 0x10,0x10,0x0c,0x0d, 0x10,0x10,0x10,0x10 C part a1
 .byte 0x10,0x11,0x12,0x13, 0x10,0x15,0x16,0x17, 0x10,0x03,0x1a,0x1b, 0x10,0x0c,0x0d,0x0e C part a2
C Masks for high end of number
 .byte 0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff
 .byte 0xff,0xff,0xff,0xff,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00
 .byte 0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00
 .byte 0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0x00,0x00,0x00,0x00
C Masks for low end of number
C .byte	0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff
C .byte	0x00,0x00,0x00,0x00,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff
C .byte	0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff
C .byte	0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xff,0xff,0xff,0xff
END_OBJECT(cnsts)
