dnl  Itanium-2 mpn_gcd_1 -- mpn by 1 gcd.

dnl  Contributed to the GNU project by Kevin Ryde, innerloop by Torbjorn
dnl  Granlund.

dnl  Copyright 2002-2005, 2012, 2013, 2015 Free Software Foundation, Inc.

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


C           cycles/bitpair (1x1 gcd)
C Itanium:       ?
C Itanium 2:     5.1


C mpn_gcd_1 (mp_srcptr xp, mp_size_t xsize, mp_limb_t y);
C
C The entry sequence is designed to expect xsize>1 and hence a modexact
C call.  This ought to be more common than a 1x1 operation.  Our critical
C path is thus stripping factors of 2 from y, calling modexact, then
C stripping factors of 2 from the x remainder returned.
C
C The common factors of 2 between x and y must be determined using the
C original x, not the remainder from the modexact.  This is done with
C x_orig which is xp[0].  There's plenty of time to do this while the rest
C of the modexact etc is happening.
C
C It's possible xp[0] is zero.  In this case the trailing zeros calculation
C popc((x-1)&~x) gives 63, and that's clearly no less than what y will
C have, making min(x_twos,y_twos) == y_twos.
C
C The main loop consists of transforming x,y to abs(x-y),min(x,y), and then
C stripping factors of 2 from abs(x-y).  Those factors of two are
C determined from just y-x, without the abs(), since there's the same
C number of trailing zeros on n or -n in twos complement.  That makes the
C dependent chain 8 cycles deep.
C
C The selection of x-y versus y-x for abs(x-y), and the selection of the
C minimum of x and y, is done in parallel with the critical path.
C
C The algorithm takes about 0.68 iterations per bit (two N bit operands) on
C average, hence the final 5.8 cycles/bitpair.
C
C Not done:
C
C An alternate algorithm which didn't strip all twos, but instead applied
C tbit and predicated extr on x, and then y, was attempted.  The loop was 6
C cycles, but the algorithm is an average 1.25 iterations per bitpair for a
C total 7.25 c/bp, which is slower than the current approach.
C
C Alternatives:
C
C Perhaps we could do something tricky by extracting a few high bits and a
C few low bits from the operands, and looking up a table which would give a
C set of predicates to control some shifts or subtracts or whatever.  That
C could knock off multiple bits per iteration.
C
C The right shifts are a bit of a bottleneck (shr at 2 or 3 cycles, or extr
C only going down I0), perhaps it'd be possible to shift left instead,
C using add.  That would mean keeping track of the lowest not-yet-zeroed
C bit, using some sort of mask.
C
C TODO:
C  * Once mod_1_N exists in assembly for Itanium, add conditional calls.
C  * Call bmod_1 even for n=1 when up[0] >> v0 (like other gcd_1 impls).
C  * Probably avoid popcnt also outside of loop, instead use ctz_table.

ASM_START()
	.explicit				C What does this mean?

C HP's assembler requires these declarations for importing mpn_modexact_1c_odd
	.global	mpn_modexact_1c_odd
	.type	mpn_modexact_1c_odd,@function

C ctz_table[n] is the number of trailing zeros on n, or MAXSHIFT if n==0.

deflit(MAXSHIFT, 7)
deflit(MASK, eval((m4_lshift(1,MAXSHIFT))-1))

C	.section	".rodata"
	.rodata
	ALIGN(m4_lshift(1,MAXSHIFT))	C align table to allow using dep
ctz_table:
	data1	MAXSHIFT
forloop(i,1,MASK,
`	data1	m4_count_trailing_zeros(i)
')

PROLOGUE(mpn_gcd_1)

		C r32	xp
		C r33	xsize
		C r34	y

define(x,           r8)
define(xp_orig,     r32)
define(xsize,       r33)
define(y,           r34)  define(inputs, 3)
define(save_rp,     r35)
define(save_pfs,    r36)
define(x_orig,      r37)
define(x_orig_one,  r38)
define(y_twos,      r39)  define(locals, 5)
define(out_xp,      r40)
define(out_xsize,   r41)
define(out_divisor, r42)
define(out_carry,   r43)  define(outputs, 4)

	.prologue
 {.mmi;
ifdef(`HAVE_ABI_32',
`		addp4	r9 = 0, xp_orig   define(xp,r9)',	C M0
`					  define(xp,xp_orig)')
	.save ar.pfs, save_pfs
		alloc	save_pfs = ar.pfs, inputs, locals, outputs, 0 C M2
	.save rp, save_rp
		mov	save_rp = b0			C I0
}{.mbb;	.body
		add	r10 = -1, y			C M3  y-1
		nop.b	0				C B0
		nop.b	0				C B1
	;;

}{.mmi;		ld8	x = [xp]			C M0  x = xp[0] if no modexact
		ld8	x_orig = [xp]			C M1  orig x for common twos
		cmp.ne	p6,p0 = 1, xsize		C I0
}{.mmi;		andcm	y_twos = r10, y			C M2  (y-1)&~y
		mov	out_xp = xp_orig		C M3
		mov	out_xsize = xsize		C I1
	;;
}{.mmi;		mov	out_carry = 0			C M0
		nop.m	0				C M1
		popcnt	y_twos = y_twos			C I0  y twos
	;;
}{.mmi;		add	x_orig_one = -1, x_orig		C M0  orig x-1
		nop.m	0				C M1
		shr.u	out_divisor = y, y_twos		C I0  y without twos
}{.mib;		nop.m	0				C M2
		shr.u	y = y, y_twos			C I1  y without twos
	(p6)	br.call.sptk.many b0 = mpn_modexact_1c_odd  C if xsize>1
	;;
}
	C modexact can leave x==0
 {.mmi;		cmp.eq	p6,p0 = 0, x			C M0  if {xp,xsize} % y == 0
		andcm	x_orig = x_orig_one, x_orig	C M1  orig (x-1)&~x
		add	r9 = -1, x			C I0  x-1
	;;
}{.mmi;		andcm	r9 = r9, x			C M0  (x-1)&~x
		nop.m	0				C M1
		mov	b0 = save_rp			C I0
	;;
}{.mii;		nop.m	0				C M0
		popcnt	x_orig = x_orig			C I0  orig x twos
		popcnt	r9 = r9				C I0  x twos
	;;
}{.mmi;		cmp.lt	p7,p0 = x_orig, y_twos		C M0  orig x_twos < y_twos
		addl	r22 = @ltoff(ctz_table), r1
		shr.u	x = x, r9			C I0  x odd
	;;
}{.mib;
	(p7)	mov	y_twos = x_orig		C M0  common twos
		add	r10 = -1, y		C I0  y-1
	(p6)	br.dpnt.few L(done_y)		C B0  x%y==0 then result y
	;;
}
		mov	r25 = m4_lshift(MASK, MAXSHIFT)
		ld8	r22 = [r22]
		br	L(ent)
	;;

	ALIGN(32)
L(top):
	.pred.rel "mutex", p6,p7
 {.mmi;	(p7)	mov	y = x
	(p6)	sub	x = x, y
		dep	r21 = r19, r22, 0, MAXSHIFT	C concat(table,lowbits)
}{.mmi;		and	r20 = MASK, r19
	(p7)	mov	x = r19
		nop	0
	;;
}
L(mid):
{.mmb;		ld1	r16 = [r21]
		cmp.eq	p10,p0 = 0, r20
	(p10)	br.spnt.few.clr	 L(shift_alot)
	;;
}{.mmi;		nop	0
		nop	0
		shr.u	x = x, r16
	;;
}
L(ent):
 {.mmi;		sub	r19 = y, x
		cmp.gtu	p6,p7 = x, y
		cmp.ne	p8,p0 = x, y
}{.mmb;		nop	0
		nop	0
	(p8)	br.sptk.few.clr L(top)
}

L(done_y):	C result is y
		mov	ar.pfs = save_pfs	C I0
		shl	r8 = y, y_twos		C I   common factors of 2
		br.ret.sptk.many b0

L(shift_alot):
		and	r20 = x, r25
		shr.u	x = x, MAXSHIFT
	;;
		dep	r21 = x, r22, 0, MAXSHIFT
		br	L(mid)
EPILOGUE()
