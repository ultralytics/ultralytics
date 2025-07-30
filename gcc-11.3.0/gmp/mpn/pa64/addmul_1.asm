dnl  HP-PA 2.0 64-bit mpn_addmul_1 -- Multiply a limb vector with a limb and
dnl  add the result to a second limb vector.

dnl  Copyright 1998-2000, 2002, 2003 Free Software Foundation, Inc.

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

C		    cycles/limb
C 8000,8200:		7
C 8500,8600,8700:	6.375

C  The feed-in and wind-down code has not yet been scheduled.  Many cycles
C  could be saved there per call.

C  DESCRIPTION:
C  The main loop "BIG" is 4-way unrolled, mainly to allow
C  effective use of ADD,DC.  Delays in moving data via the cache from the FP
C  registers to the IU registers, have demanded a deep software pipeline, and
C  a lot of stack slots for partial products in flight.
C
C  CODE STRUCTURE:
C  save-some-registers
C  do 0, 1, 2, or 3 limbs
C  if done, restore-some-regs and return
C  save-many-regs
C  do 4, 8, ... limb
C  restore-all-regs

C  STACK LAYOUT:
C  HP-PA stack grows upwards.  We could allocate 8 fewer slots by using the
C  slots marked FREE, as well as some slots in the caller's "frame marker".
C
C -00 <- r30
C -08  FREE
C -10  tmp
C -18  tmp
C -20  tmp
C -28  tmp
C -30  tmp
C -38  tmp
C -40  tmp
C -48  tmp
C -50  tmp
C -58  tmp
C -60  tmp
C -68  tmp
C -70  tmp
C -78  tmp
C -80  tmp
C -88  tmp
C -90  FREE
C -98  FREE
C -a0  FREE
C -a8  FREE
C -b0  r13
C -b8  r12
C -c0  r11
C -c8  r10
C -d0  r8
C -d8  r8
C -e0  r7
C -e8  r6
C -f0  r5
C -f8  r4
C -100 r3
C  Previous frame:
C  [unused area]
C -38/-138 vlimb home slot.  For 2.0N, the vlimb arg will arrive here.


include(`../config.m4')

C INPUT PARAMETERS:
define(`rp',`%r26')	C
define(`up',`%r25')	C
define(`n',`%r24')	C
define(`vlimb',`%r23')	C

define(`climb',`%r23')	C

ifdef(`HAVE_ABI_2_0w',
`	.level	2.0w
',`	.level	2.0
')
PROLOGUE(mpn_addmul_1)

ifdef(`HAVE_ABI_2_0w',
`	std		vlimb, -0x38(%r30)	C store vlimb into "home" slot
')
	std,ma		%r3, 0x100(%r30)
	std		%r4, -0xf8(%r30)
	std		%r5, -0xf0(%r30)
	ldo		0(%r0), climb		C clear climb
	fldd		-0x138(%r30), %fr8	C put vlimb in fp register

define(`p032a1',`%r1')	C
define(`p032a2',`%r19')	C

define(`m032',`%r20')	C
define(`m096',`%r21')	C

define(`p000a',`%r22')	C
define(`p064a',`%r29')	C

define(`s000',`%r31')	C

define(`ma000',`%r4')	C
define(`ma064',`%r20')	C

define(`r000',`%r3')	C

	extrd,u		n, 63, 2, %r5
	cmpb,=		%r5, %r0, L(BIG)
	nop

	fldd		0(up), %fr4
	ldo		8(up), up
	xmpyu		%fr8R, %fr4L, %fr22
	xmpyu		%fr8L, %fr4R, %fr23
	fstd		%fr22, -0x78(%r30)	C mid product to  -0x78..-0x71
	xmpyu		%fr8R, %fr4R, %fr24
	xmpyu		%fr8L, %fr4L, %fr25
	fstd		%fr23, -0x70(%r30)	C mid product to  -0x70..-0x69
	fstd		%fr24, -0x80(%r30)	C low product to  -0x80..-0x79
	addib,<>	-1, %r5, L(two_or_more)
	fstd		%fr25, -0x68(%r30)	C high product to -0x68..-0x61
LDEF(one)
	ldd		-0x78(%r30), p032a1
	ldd		-0x70(%r30), p032a2
	ldd		-0x80(%r30), p000a
	b		L(0_one_out)
	ldd		-0x68(%r30), p064a

LDEF(two_or_more)
	fldd		0(up), %fr4
	ldo		8(up), up
	xmpyu		%fr8R, %fr4L, %fr22
	xmpyu		%fr8L, %fr4R, %fr23
	ldd		-0x78(%r30), p032a1
	fstd		%fr22, -0x78(%r30)	C mid product to  -0x78..-0x71
	xmpyu		%fr8R, %fr4R, %fr24
	xmpyu		%fr8L, %fr4L, %fr25
	ldd		-0x70(%r30), p032a2
	fstd		%fr23, -0x70(%r30)	C mid product to  -0x70..-0x69
	ldd		-0x80(%r30), p000a
	fstd		%fr24, -0x80(%r30)	C low product to  -0x80..-0x79
	ldd		-0x68(%r30), p064a
	addib,<>	-1, %r5, L(three_or_more)
	fstd		%fr25, -0x68(%r30)	C high product to -0x68..-0x61
LDEF(two)
	add		p032a1, p032a2, m032
	add,dc		%r0, %r0, m096
	depd,z		m032, 31, 32, ma000
	extrd,u		m032, 31, 32, ma064
	ldd		0(rp), r000
	b		L(0_two_out)
	depd		m096, 31, 32, ma064

LDEF(three_or_more)
	fldd		0(up), %fr4
	add		p032a1, p032a2, m032
	add,dc		%r0, %r0, m096
	depd,z		m032, 31, 32, ma000
	extrd,u		m032, 31, 32, ma064
	ldd		0(rp), r000
C	addib,=		-1, %r5, L(0_out)
	depd		m096, 31, 32, ma064
LDEF(loop0)
C	xmpyu		%fr8R, %fr4L, %fr22
C	xmpyu		%fr8L, %fr4R, %fr23
C	ldd		-0x78(%r30), p032a1
C	fstd		%fr22, -0x78(%r30)	C mid product to  -0x78..-0x71
C
C	xmpyu		%fr8R, %fr4R, %fr24
C	xmpyu		%fr8L, %fr4L, %fr25
C	ldd		-0x70(%r30), p032a2
C	fstd		%fr23, -0x70(%r30)	C mid product to  -0x70..-0x69
C
C	ldo		8(rp), rp
C	add		climb, p000a, s000
C	ldd		-0x80(%r30), p000a
C	fstd		%fr24, -0x80(%r30)	C low product to  -0x80..-0x79
C
C	add,dc		p064a, %r0, climb
C	ldo		8(up), up
C	ldd		-0x68(%r30), p064a
C	fstd		%fr25, -0x68(%r30)	C high product to -0x68..-0x61
C
C	add		ma000, s000, s000
C	add,dc		ma064, climb, climb
C	fldd		0(up), %fr4
C
C	add		r000, s000, s000
C	add,dc		%r0, climb, climb
C	std		s000, -8(rp)
C
C	add		p032a1, p032a2, m032
C	add,dc		%r0, %r0, m096
C
C	depd,z		m032, 31, 32, ma000
C	extrd,u		m032, 31, 32, ma064
C	ldd		0(rp), r000
C	addib,<>	-1, %r5, L(loop0)
C	depd		m096, 31, 32, ma064
LDEF(0_out)
	ldo		8(up), up
	xmpyu		%fr8R, %fr4L, %fr22
	xmpyu		%fr8L, %fr4R, %fr23
	ldd		-0x78(%r30), p032a1
	fstd		%fr22, -0x78(%r30)	C mid product to  -0x78..-0x71
	xmpyu		%fr8R, %fr4R, %fr24
	xmpyu		%fr8L, %fr4L, %fr25
	ldd		-0x70(%r30), p032a2
	fstd		%fr23, -0x70(%r30)	C mid product to  -0x70..-0x69
	ldo		8(rp), rp
	add		climb, p000a, s000
	ldd		-0x80(%r30), p000a
	fstd		%fr24, -0x80(%r30)	C low product to  -0x80..-0x79
	add,dc		p064a, %r0, climb
	ldd		-0x68(%r30), p064a
	fstd		%fr25, -0x68(%r30)	C high product to -0x68..-0x61
	add		ma000, s000, s000
	add,dc		ma064, climb, climb
	add		r000, s000, s000
	add,dc		%r0, climb, climb
	std		s000, -8(rp)
	add		p032a1, p032a2, m032
	add,dc		%r0, %r0, m096
	depd,z		m032, 31, 32, ma000
	extrd,u		m032, 31, 32, ma064
	ldd		0(rp), r000
	depd		m096, 31, 32, ma064
LDEF(0_two_out)
	ldd		-0x78(%r30), p032a1
	ldd		-0x70(%r30), p032a2
	ldo		8(rp), rp
	add		climb, p000a, s000
	ldd		-0x80(%r30), p000a
	add,dc		p064a, %r0, climb
	ldd		-0x68(%r30), p064a
	add		ma000, s000, s000
	add,dc		ma064, climb, climb
	add		r000, s000, s000
	add,dc		%r0, climb, climb
	std		s000, -8(rp)
LDEF(0_one_out)
	add		p032a1, p032a2, m032
	add,dc		%r0, %r0, m096
	depd,z		m032, 31, 32, ma000
	extrd,u		m032, 31, 32, ma064
	ldd		0(rp), r000
	depd		m096, 31, 32, ma064

	add		climb, p000a, s000
	add,dc		p064a, %r0, climb
	add		ma000, s000, s000
	add,dc		ma064, climb, climb
	add		r000, s000, s000
	add,dc		%r0, climb, climb
	std		s000, 0(rp)

	cmpib,>=	4, n, L(done)
	ldo		8(rp), rp

C 4-way unrolled code.

LDEF(BIG)

define(`p032a1',`%r1')	C
define(`p032a2',`%r19')	C
define(`p096b1',`%r20')	C
define(`p096b2',`%r21')	C
define(`p160c1',`%r22')	C
define(`p160c2',`%r29')	C
define(`p224d1',`%r31')	C
define(`p224d2',`%r3')	C
			C
define(`m032',`%r4')	C
define(`m096',`%r5')	C
define(`m160',`%r6')	C
define(`m224',`%r7')	C
define(`m288',`%r8')	C
			C
define(`p000a',`%r1')	C
define(`p064a',`%r19')	C
define(`p064b',`%r20')	C
define(`p128b',`%r21')	C
define(`p128c',`%r22')	C
define(`p192c',`%r29')	C
define(`p192d',`%r31')	C
define(`p256d',`%r3')	C
			C
define(`s000',`%r10')	C
define(`s064',`%r11')	C
define(`s128',`%r12')	C
define(`s192',`%r13')	C
			C
define(`ma000',`%r9')	C
define(`ma064',`%r4')	C
define(`ma128',`%r5')	C
define(`ma192',`%r6')	C
define(`ma256',`%r7')	C
			C
define(`r000',`%r1')	C
define(`r064',`%r19')	C
define(`r128',`%r20')	C
define(`r192',`%r21')	C

	std		%r6, -0xe8(%r30)
	std		%r7, -0xe0(%r30)
	std		%r8, -0xd8(%r30)
	std		%r9, -0xd0(%r30)
	std		%r10, -0xc8(%r30)
	std		%r11, -0xc0(%r30)
	std		%r12, -0xb8(%r30)
	std		%r13, -0xb0(%r30)

ifdef(`HAVE_ABI_2_0w',
`	extrd,u		n, 61, 62, n		C right shift 2
',`	extrd,u		n, 61, 30, n		C right shift 2, zero extend
')

LDEF(4_or_more)
	fldd		0(up), %fr4
	fldd		8(up), %fr5
	fldd		16(up), %fr6
	fldd		24(up), %fr7
	xmpyu		%fr8R, %fr4L, %fr22
	xmpyu		%fr8L, %fr4R, %fr23
	xmpyu		%fr8R, %fr5L, %fr24
	xmpyu		%fr8L, %fr5R, %fr25
	xmpyu		%fr8R, %fr6L, %fr26
	xmpyu		%fr8L, %fr6R, %fr27
	fstd		%fr22, -0x78(%r30)	C mid product to  -0x78..-0x71
	xmpyu		%fr8R, %fr7L, %fr28
	xmpyu		%fr8L, %fr7R, %fr29
	fstd		%fr23, -0x70(%r30)	C mid product to  -0x70..-0x69
	xmpyu		%fr8R, %fr4R, %fr30
	xmpyu		%fr8L, %fr4L, %fr31
	fstd		%fr24, -0x38(%r30)	C mid product to  -0x38..-0x31
	xmpyu		%fr8R, %fr5R, %fr22
	xmpyu		%fr8L, %fr5L, %fr23
	fstd		%fr25, -0x30(%r30)	C mid product to  -0x30..-0x29
	xmpyu		%fr8R, %fr6R, %fr24
	xmpyu		%fr8L, %fr6L, %fr25
	fstd		%fr26, -0x58(%r30)	C mid product to  -0x58..-0x51
	xmpyu		%fr8R, %fr7R, %fr26
	fstd		%fr27, -0x50(%r30)	C mid product to  -0x50..-0x49
	addib,<>	-1, n, L(8_or_more)
	xmpyu		%fr8L, %fr7L, %fr27
	fstd		%fr28, -0x18(%r30)	C mid product to  -0x18..-0x11
	fstd		%fr29, -0x10(%r30)	C mid product to  -0x10..-0x09
	fstd		%fr30, -0x80(%r30)	C low product to  -0x80..-0x79
	fstd		%fr31, -0x68(%r30)	C high product to -0x68..-0x61
	fstd		%fr22, -0x40(%r30)	C low product to  -0x40..-0x39
	fstd		%fr23, -0x28(%r30)	C high product to -0x28..-0x21
	fstd		%fr24, -0x60(%r30)	C low product to  -0x60..-0x59
	fstd		%fr25, -0x48(%r30)	C high product to -0x48..-0x41
	fstd		%fr26, -0x20(%r30)	C low product to  -0x20..-0x19
	fstd		%fr27, -0x88(%r30)	C high product to -0x88..-0x81
	ldd		-0x78(%r30), p032a1
	ldd		-0x70(%r30), p032a2
	ldd		-0x38(%r30), p096b1
	ldd		-0x30(%r30), p096b2
	ldd		-0x58(%r30), p160c1
	ldd		-0x50(%r30), p160c2
	ldd		-0x18(%r30), p224d1
	ldd		-0x10(%r30), p224d2
	b		L(end1)
	nop

LDEF(8_or_more)
	fstd		%fr28, -0x18(%r30)	C mid product to  -0x18..-0x11
	fstd		%fr29, -0x10(%r30)	C mid product to  -0x10..-0x09
	ldo		32(up), up
	fstd		%fr30, -0x80(%r30)	C low product to  -0x80..-0x79
	fstd		%fr31, -0x68(%r30)	C high product to -0x68..-0x61
	fstd		%fr22, -0x40(%r30)	C low product to  -0x40..-0x39
	fstd		%fr23, -0x28(%r30)	C high product to -0x28..-0x21
	fstd		%fr24, -0x60(%r30)	C low product to  -0x60..-0x59
	fstd		%fr25, -0x48(%r30)	C high product to -0x48..-0x41
	fstd		%fr26, -0x20(%r30)	C low product to  -0x20..-0x19
	fstd		%fr27, -0x88(%r30)	C high product to -0x88..-0x81
	fldd		0(up), %fr4
	fldd		8(up), %fr5
	fldd		16(up), %fr6
	fldd		24(up), %fr7
	xmpyu		%fr8R, %fr4L, %fr22
	ldd		-0x78(%r30), p032a1
	xmpyu		%fr8L, %fr4R, %fr23
	xmpyu		%fr8R, %fr5L, %fr24
	ldd		-0x70(%r30), p032a2
	xmpyu		%fr8L, %fr5R, %fr25
	xmpyu		%fr8R, %fr6L, %fr26
	ldd		-0x38(%r30), p096b1
	xmpyu		%fr8L, %fr6R, %fr27
	fstd		%fr22, -0x78(%r30)	C mid product to  -0x78..-0x71
	xmpyu		%fr8R, %fr7L, %fr28
	ldd		-0x30(%r30), p096b2
	xmpyu		%fr8L, %fr7R, %fr29
	fstd		%fr23, -0x70(%r30)	C mid product to  -0x70..-0x69
	xmpyu		%fr8R, %fr4R, %fr30
	ldd		-0x58(%r30), p160c1
	xmpyu		%fr8L, %fr4L, %fr31
	fstd		%fr24, -0x38(%r30)	C mid product to  -0x38..-0x31
	xmpyu		%fr8R, %fr5R, %fr22
	ldd		-0x50(%r30), p160c2
	xmpyu		%fr8L, %fr5L, %fr23
	fstd		%fr25, -0x30(%r30)	C mid product to  -0x30..-0x29
	xmpyu		%fr8R, %fr6R, %fr24
	ldd		-0x18(%r30), p224d1
	xmpyu		%fr8L, %fr6L, %fr25
	fstd		%fr26, -0x58(%r30)	C mid product to  -0x58..-0x51
	xmpyu		%fr8R, %fr7R, %fr26
	ldd		-0x10(%r30), p224d2
	fstd		%fr27, -0x50(%r30)	C mid product to  -0x50..-0x49
	addib,=		-1, n, L(end2)
	xmpyu		%fr8L, %fr7L, %fr27
LDEF(loop)
	add		p032a1, p032a2, m032
	ldd		-0x80(%r30), p000a
	add,dc		p096b1, p096b2, m096
	fstd		%fr28, -0x18(%r30)	C mid product to  -0x18..-0x11

	add,dc		p160c1, p160c2, m160
	ldd		-0x68(%r30), p064a
	add,dc		p224d1, p224d2, m224
	fstd		%fr29, -0x10(%r30)	C mid product to  -0x10..-0x09

	add,dc		%r0, %r0, m288
	ldd		-0x40(%r30), p064b
	ldo		32(up), up
	fstd		%fr30, -0x80(%r30)	C low product to  -0x80..-0x79

	depd,z		m032, 31, 32, ma000
	ldd		-0x28(%r30), p128b
	extrd,u		m032, 31, 32, ma064
	fstd		%fr31, -0x68(%r30)	C high product to -0x68..-0x61

	depd		m096, 31, 32, ma064
	ldd		-0x60(%r30), p128c
	extrd,u		m096, 31, 32, ma128
	fstd		%fr22, -0x40(%r30)	C low product to  -0x40..-0x39

	depd		m160, 31, 32, ma128
	ldd		-0x48(%r30), p192c
	extrd,u		m160, 31, 32, ma192
	fstd		%fr23, -0x28(%r30)	C high product to -0x28..-0x21

	depd		m224, 31, 32, ma192
	ldd		-0x20(%r30), p192d
	extrd,u		m224, 31, 32, ma256
	fstd		%fr24, -0x60(%r30)	C low product to  -0x60..-0x59

	depd		m288, 31, 32, ma256
	ldd		-0x88(%r30), p256d
	add		climb, p000a, s000
	fstd		%fr25, -0x48(%r30)	C high product to -0x48..-0x41

	add,dc		p064a, p064b, s064
	ldd		0(rp), r000
	add,dc		p128b, p128c, s128
	fstd		%fr26, -0x20(%r30)	C low product to  -0x20..-0x19

	add,dc		p192c, p192d, s192
	ldd		8(rp), r064
	add,dc		p256d, %r0, climb
	fstd		%fr27, -0x88(%r30)	C high product to -0x88..-0x81

	ldd		16(rp), r128
	add		ma000, s000, s000	C accum mid 0
	ldd		24(rp), r192
	add,dc		ma064, s064, s064	C accum mid 1

	add,dc		ma128, s128, s128	C accum mid 2
	fldd		0(up), %fr4
	add,dc		ma192, s192, s192	C accum mid 3
	fldd		8(up), %fr5

	add,dc		ma256, climb, climb
	fldd		16(up), %fr6
	add		r000, s000, s000	C accum rlimb 0
	fldd		24(up), %fr7

	add,dc		r064, s064, s064	C accum rlimb 1
	add,dc		r128, s128, s128	C accum rlimb 2
	std		s000, 0(rp)

	add,dc		r192, s192, s192	C accum rlimb 3
	add,dc		%r0, climb, climb
	std		s064, 8(rp)

	xmpyu		%fr8R, %fr4L, %fr22
	ldd		-0x78(%r30), p032a1
	xmpyu		%fr8L, %fr4R, %fr23
	std		s128, 16(rp)

	xmpyu		%fr8R, %fr5L, %fr24
	ldd		-0x70(%r30), p032a2
	xmpyu		%fr8L, %fr5R, %fr25
	std		s192, 24(rp)

	xmpyu		%fr8R, %fr6L, %fr26
	ldd		-0x38(%r30), p096b1
	xmpyu		%fr8L, %fr6R, %fr27
	fstd		%fr22, -0x78(%r30)	C mid product to  -0x78..-0x71

	xmpyu		%fr8R, %fr7L, %fr28
	ldd		-0x30(%r30), p096b2
	xmpyu		%fr8L, %fr7R, %fr29
	fstd		%fr23, -0x70(%r30)	C mid product to  -0x70..-0x69

	xmpyu		%fr8R, %fr4R, %fr30
	ldd		-0x58(%r30), p160c1
	xmpyu		%fr8L, %fr4L, %fr31
	fstd		%fr24, -0x38(%r30)	C mid product to  -0x38..-0x31

	xmpyu		%fr8R, %fr5R, %fr22
	ldd		-0x50(%r30), p160c2
	xmpyu		%fr8L, %fr5L, %fr23
	fstd		%fr25, -0x30(%r30)	C mid product to  -0x30..-0x29

	xmpyu		%fr8R, %fr6R, %fr24
	ldd		-0x18(%r30), p224d1
	xmpyu		%fr8L, %fr6L, %fr25
	fstd		%fr26, -0x58(%r30)	C mid product to  -0x58..-0x51

	xmpyu		%fr8R, %fr7R, %fr26
	ldd		-0x10(%r30), p224d2
	fstd		%fr27, -0x50(%r30)	C mid product to  -0x50..-0x49
	xmpyu		%fr8L, %fr7L, %fr27

	addib,<>	-1, n, L(loop)
	ldo		32(rp), rp

LDEF(end2)
	add		p032a1, p032a2, m032
	ldd		-0x80(%r30), p000a
	add,dc		p096b1, p096b2, m096
	fstd		%fr28, -0x18(%r30)	C mid product to  -0x18..-0x11
	add,dc		p160c1, p160c2, m160
	ldd		-0x68(%r30), p064a
	add,dc		p224d1, p224d2, m224
	fstd		%fr29, -0x10(%r30)	C mid product to  -0x10..-0x09
	add,dc		%r0, %r0, m288
	ldd		-0x40(%r30), p064b
	fstd		%fr30, -0x80(%r30)	C low product to  -0x80..-0x79
	depd,z		m032, 31, 32, ma000
	ldd		-0x28(%r30), p128b
	extrd,u		m032, 31, 32, ma064
	fstd		%fr31, -0x68(%r30)	C high product to -0x68..-0x61
	depd		m096, 31, 32, ma064
	ldd		-0x60(%r30), p128c
	extrd,u		m096, 31, 32, ma128
	fstd		%fr22, -0x40(%r30)	C low product to  -0x40..-0x39
	depd		m160, 31, 32, ma128
	ldd		-0x48(%r30), p192c
	extrd,u		m160, 31, 32, ma192
	fstd		%fr23, -0x28(%r30)	C high product to -0x28..-0x21
	depd		m224, 31, 32, ma192
	ldd		-0x20(%r30), p192d
	extrd,u		m224, 31, 32, ma256
	fstd		%fr24, -0x60(%r30)	C low product to  -0x60..-0x59
	depd		m288, 31, 32, ma256
	ldd		-0x88(%r30), p256d
	add		climb, p000a, s000
	fstd		%fr25, -0x48(%r30)	C high product to -0x48..-0x41
	add,dc		p064a, p064b, s064
	ldd		0(rp), r000
	add,dc		p128b, p128c, s128
	fstd		%fr26, -0x20(%r30)	C low product to  -0x20..-0x19
	add,dc		p192c, p192d, s192
	ldd		8(rp), r064
	add,dc		p256d, %r0, climb
	fstd		%fr27, -0x88(%r30)	C high product to -0x88..-0x81
	ldd		16(rp), r128
	add		ma000, s000, s000	C accum mid 0
	ldd		24(rp), r192
	add,dc		ma064, s064, s064	C accum mid 1
	add,dc		ma128, s128, s128	C accum mid 2
	add,dc		ma192, s192, s192	C accum mid 3
	add,dc		ma256, climb, climb
	add		r000, s000, s000	C accum rlimb 0
	add,dc		r064, s064, s064	C accum rlimb 1
	add,dc		r128, s128, s128	C accum rlimb 2
	std		s000, 0(rp)
	add,dc		r192, s192, s192	C accum rlimb 3
	add,dc		%r0, climb, climb
	std		s064, 8(rp)
	ldd		-0x78(%r30), p032a1
	std		s128, 16(rp)
	ldd		-0x70(%r30), p032a2
	std		s192, 24(rp)
	ldd		-0x38(%r30), p096b1
	ldd		-0x30(%r30), p096b2
	ldd		-0x58(%r30), p160c1
	ldd		-0x50(%r30), p160c2
	ldd		-0x18(%r30), p224d1
	ldd		-0x10(%r30), p224d2
	ldo		32(rp), rp

LDEF(end1)
	add		p032a1, p032a2, m032
	ldd		-0x80(%r30), p000a
	add,dc		p096b1, p096b2, m096
	add,dc		p160c1, p160c2, m160
	ldd		-0x68(%r30), p064a
	add,dc		p224d1, p224d2, m224
	add,dc		%r0, %r0, m288
	ldd		-0x40(%r30), p064b
	depd,z		m032, 31, 32, ma000
	ldd		-0x28(%r30), p128b
	extrd,u		m032, 31, 32, ma064
	depd		m096, 31, 32, ma064
	ldd		-0x60(%r30), p128c
	extrd,u		m096, 31, 32, ma128
	depd		m160, 31, 32, ma128
	ldd		-0x48(%r30), p192c
	extrd,u		m160, 31, 32, ma192
	depd		m224, 31, 32, ma192
	ldd		-0x20(%r30), p192d
	extrd,u		m224, 31, 32, ma256
	depd		m288, 31, 32, ma256
	ldd		-0x88(%r30), p256d
	add		climb, p000a, s000
	add,dc		p064a, p064b, s064
	ldd		0(rp), r000
	add,dc		p128b, p128c, s128
	add,dc		p192c, p192d, s192
	ldd		8(rp), r064
	add,dc		p256d, %r0, climb
	ldd		16(rp), r128
	add		ma000, s000, s000	C accum mid 0
	ldd		24(rp), r192
	add,dc		ma064, s064, s064	C accum mid 1
	add,dc		ma128, s128, s128	C accum mid 2
	add,dc		ma192, s192, s192	C accum mid 3
	add,dc		ma256, climb, climb
	add		r000, s000, s000	C accum rlimb 0
	add,dc		r064, s064, s064	C accum rlimb 1
	add,dc		r128, s128, s128	C accum rlimb 2
	std		s000, 0(rp)
	add,dc		r192, s192, s192	C accum rlimb 3
	add,dc		%r0, climb, climb
	std		s064, 8(rp)
	std		s128, 16(rp)
	std		s192, 24(rp)

	ldd		-0xb0(%r30), %r13
	ldd		-0xb8(%r30), %r12
	ldd		-0xc0(%r30), %r11
	ldd		-0xc8(%r30), %r10
	ldd		-0xd0(%r30), %r9
	ldd		-0xd8(%r30), %r8
	ldd		-0xe0(%r30), %r7
	ldd		-0xe8(%r30), %r6
LDEF(done)
ifdef(`HAVE_ABI_2_0w',
`	copy		climb, %r28
',`	extrd,u		climb, 63, 32, %r29
	extrd,u		climb, 31, 32, %r28
')
	ldd		-0xf0(%r30), %r5
	ldd		-0xf8(%r30), %r4
	bve		(%r2)
	ldd,mb		-0x100(%r30), %r3
EPILOGUE(mpn_addmul_1)
