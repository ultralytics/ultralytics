dnl  Alpha mpn_modexact_1c_odd -- mpn exact remainder

dnl  Copyright 2003, 2004 Free Software Foundation, Inc.

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


C      cycles/limb
C EV4:    47
C EV5:    30
C EV6:    15


C mp_limb_t mpn_modexact_1c_odd (mp_srcptr src, mp_size_t size, mp_limb_t d,
C                                mp_limb_t c)
C
C This code follows the "alternate" code in mpn/generic/mode1o.c,
C eliminating cbit+climb from the dependent chain.  This leaves,
C
C        ev4   ev5   ev6
C         1     3     1    subq   y = x - h
C        23    13     7    mulq   q = y * inverse
C        23    14     7    umulh  h = high (q * d)
C        --    --    --
C        47    30    15
C
C In each case, the load latency, loop control, and extra carry bit handling
C hide under the multiply latencies.  Those latencies are long enough that
C we don't need to worry about alignment or pairing to squeeze out
C performance.
C
C For the first limb, some of the loop code is broken out and scheduled back
C since it can be done earlier.
C
C   - The first ldq src[0] is near the start of the routine, for maximum
C     time from memory.
C
C   - The subq y=x-climb can be done without waiting for the inverse.
C
C   - The mulq y*inverse is replicated after the final subq for the inverse,
C     instead of branching to the mulq in the main loop.  On ev4 a branch
C     there would cost cycles, but we can hide them under the mulq latency.
C
C For the last limb, high<divisor is tested and if that's true a subtract
C and addback is done, as per the main mpn/generic/mode1o.c code.  This is a
C data-dependent branch, but we're waiting for umulh so any penalty should
C hide there.  The multiplies saved would be worth the cost anyway.
C
C Enhancements:
C
C For size==1, a plain division (done bitwise say) might be faster than
C calculating an inverse, the latter taking about 130 cycles on ev4 or 70 on
C ev5.  A call to gcc __remqu might be a possibility.

ASM_START()
PROLOGUE(mpn_modexact_1c_odd,gp)

	C r16	src
	C r17	size
	C r18	d
	C r19	c

	LEA(r0, binvert_limb_table)
	srl	r18, 1, r20		C d >> 1

	and	r20, 127, r20		C idx = d>>1 & 0x7F

	addq	r0, r20, r21		C table + idx

ifelse(bwx_available_p,1,
`	ldbu	r20, 0(r21)		C table[idx], inverse 8 bits
',`
	ldq_u	r20, 0(r21)		C table[idx] qword
	extbl	r20, r21, r20		C table[idx], inverse 8 bits
')

	mull	r20, r20, r7		C i*i
	addq	r20, r20, r20		C 2*i

	ldq	r2, 0(r16)		C x = s = src[0]
	lda	r17, -1(r17)		C size--
	clr	r0			C initial cbit=0

	mull	r7, r18, r7		C i*i*d

	subq	r20, r7, r20		C 2*i-i*i*d, inverse 16 bits

	mull	r20, r20, r7		C i*i
	addq	r20, r20, r20		C 2*i

	mull	r7, r18, r7		C i*i*d

	subq	r20, r7, r20		C 2*i-i*i*d, inverse 32 bits

	mulq	r20, r20, r7		C i*i
	addq	r20, r20, r20		C 2*i

	mulq	r7, r18, r7		C i*i*d
	subq	r2, r19, r3		C y = x - climb

	subq	r20, r7, r20		C inv = 2*i-i*i*d, inverse 64 bits

ASSERT(r7, C should have d*inv==1 mod 2^64
`	mulq	r18, r20, r7
	cmpeq	r7, 1, r7')

	mulq	r3, r20, r4		C first q = y * inv

	beq	r17, L(one)		C if size==1
	br	L(entry)


L(top):
	C r0	cbit
	C r16	src, incrementing
	C r17	size, decrementing
	C r18	d
	C r19	climb
	C r20	inv

	ldq	r1, 0(r16)		C s = src[i]
	subq	r1, r0, r2		C x = s - cbit
	cmpult	r1, r0, r0		C new cbit = s < cbit

	subq	r2, r19, r3		C y = x - climb

	mulq	r3, r20, r4		C q = y * inv
L(entry):
	cmpult	r2, r19, r5		C cbit2 = x < climb
	addq	r5, r0, r0		C cbit += cbit2
	lda	r16, 8(r16)		C src++
	lda	r17, -1(r17)		C size--

	umulh	r4, r18, r19		C climb = q * d
	bne	r17, L(top)		C while 2 or more limbs left



	C r0	cbit
	C r18	d
	C r19	climb
	C r20	inv

	ldq	r1, 0(r16)		C s = src[size-1] high limb

	cmpult	r1, r18, r2		C test high<divisor
	bne	r2, L(skip)		C skip if so

	C can't skip a division, repeat loop code

	subq	r1, r0, r2		C x = s - cbit
	cmpult	r1, r0, r0		C new cbit = s < cbit

	subq	r2, r19, r3		C y = x - climb

	mulq	r3, r20, r4		C q = y * inv
L(one):
	cmpult	r2, r19, r5		C cbit2 = x < climb
	addq	r5, r0, r0		C cbit += cbit2

	umulh	r4, r18, r19		C climb = q * d

	addq	r19, r0, r0		C return climb + cbit
	ret	r31, (r26), 1


	ALIGN(8)
L(skip):
	C with high<divisor, the final step can be just (cbit+climb)-s and
	C an addback of d if that underflows

	addq	r19, r0, r19		C c = climb + cbit

	subq	r19, r1, r2		C c - s
	cmpult	r19, r1, r3		C c < s

	addq	r2, r18, r0		C return c-s + divisor

	cmoveq	r3, r2, r0		C return c-s if no underflow
	ret	r31, (r26), 1

EPILOGUE()
ASM_END()
