dnl  PowerPC-32/VMX and PowerPC-64/VMX mpn_and_n, mpn_andn_n, mpn_nand_n,
dnl  mpn_ior_n, mpn_iorn_n, mpn_nior_n, mpn_xor_n, mpn_xnor_n -- mpn bitwise
dnl  logical operations.

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


C               and,ior,andn,nior,xor    iorn,xnor         nand
C                   cycles/limb         cycles/limb    cycles/limb
C 7400,7410 (G4):       1.39                 ?              ?
C 744x,745x (G4+):      1.14                1.39           1.39
C 970:                  1.7                 2.0            2.0

C STATUS
C  * Works for all sizes and alignment for 32-bit limbs.
C  * Works for n >= 4 for 64-bit limbs; untested for smaller operands.
C  * Current performance makes this pointless for 970

C TODO
C  * Might want to make variants when just one of the source operands needs
C    vperm, and when neither needs it.  The latter runs 50% faster on 7400.
C  * Idea: If the source operands are equally aligned, we could do the logops
C    first, then vperm before storing!  That means we never need more than one
C    vperm, ever!
C  * Perhaps align `rp' after initial alignment loop?
C  * Instead of having scalar code in the beginning and end, consider using
C    read-modify-write vector code.
C  * Software pipeline?  Hopefully not too important, this is hairy enough
C    already.
C  * At least be more clever about operand loading, i.e., load v operands before
C    u operands, since v operands are sometimes negated.

define(`GMP_LIMB_BYTES', eval(GMP_LIMB_BITS/8))
define(`LIMBS_PER_VR',  eval(16/GMP_LIMB_BYTES))
define(`LIMBS_PER_2VR', eval(32/GMP_LIMB_BYTES))

define(`vnegb', `')		C default neg-before to null
define(`vnega', `')		C default neg-before to null

ifdef(`OPERATION_and_n',
`	define(`func',	`mpn_and_n')
	define(`logopS',`and	$1,$2,$3')
	define(`logop',	`vand	$1,$2,$3')')
ifdef(`OPERATION_andn_n',
`	define(`func',	`mpn_andn_n')
	define(`logopS',`andc	$1,$2,$3')
	define(`logop',	`vandc	$1,$2,$3')')
ifdef(`OPERATION_nand_n',
`	define(`func',	`mpn_nand_n')
	define(`logopS',`nand	$1,$2,$3')
	define(`logop',	`vand	$1,$2,$3')
	define(`vnega',	`vnor	$1,$2,$2')')
ifdef(`OPERATION_ior_n',
`	define(`func',	`mpn_ior_n')
	define(`logopS',`or	$1,$2,$3')
	define(`logop',	`vor	$1,$2,$3')')
ifdef(`OPERATION_iorn_n',
`	define(`func',	`mpn_iorn_n')
	define(`logopS',`orc	$1,$2,$3')
	define(`vnegb',	`vnor	$1,$2,$2')
	define(`logop',	`vor	$1,$2,$3')')
ifdef(`OPERATION_nior_n',
`	define(`func',	`mpn_nior_n')
	define(`logopS',`nor	$1,$2,$3')
	define(`logop',	`vnor	$1,$2,$3')')
ifdef(`OPERATION_xor_n',
`	define(`func',	`mpn_xor_n')
	define(`logopS',`xor	$1,$2,$3')
	define(`logop',	`vxor	$1,$2,$3')')
ifdef(`OPERATION_xnor_n',
`	define(`func',`mpn_xnor_n')
	define(`logopS',`eqv	$1,$2,$3')
	define(`vnegb',	`vnor	$1,$2,$2')
	define(`logop',	`vxor	$1,$2,$3')')

ifelse(GMP_LIMB_BITS,`32',`
	define(`LIMB32',`	$1')
	define(`LIMB64',`')
',`
	define(`LIMB32',`')
	define(`LIMB64',`	$1')
')

C INPUT PARAMETERS
define(`rp',	`r3')
define(`up',	`r4')
define(`vp',	`r5')
define(`n',	`r6')

define(`us',	`v8')
define(`vs',	`v9')

MULFUNC_PROLOGUE(mpn_and_n mpn_andn_n mpn_nand_n mpn_ior_n mpn_iorn_n mpn_nior_n mpn_xor_n mpn_xnor_n)

ASM_START()
PROLOGUE(func)

LIMB32(`cmpwi	cr0, n, 8	')
LIMB64(`cmpdi	cr0, n, 4	')
	bge	L(big)

	mtctr	n

LIMB32(`lwz	r8, 0(up)	')
LIMB32(`lwz	r9, 0(vp)	')
LIMB32(`logopS(	r0, r8, r9)	')
LIMB32(`stw	r0, 0(rp)	')
LIMB32(`bdz	L(endS)		')

L(topS):
LIMB32(`lwzu	r8, 4(up)	')
LIMB64(`ld	r8, 0(up)	')
LIMB64(`addi	up, up, GMP_LIMB_BYTES	')
LIMB32(`lwzu	r9, 4(vp)	')
LIMB64(`ld	r9, 0(vp)	')
LIMB64(`addi	vp, vp, GMP_LIMB_BYTES	')
	logopS(	r0, r8, r9)
LIMB32(`stwu	r0, 4(rp)	')
LIMB64(`std	r0, 0(rp)	')
LIMB64(`addi	rp, rp, GMP_LIMB_BYTES	')
	bdnz	L(topS)
L(endS):
	blr

L(big):	mfspr	r12, 256
	oris	r0, r12, 0xfffc		C Set VRSAVE bit 0-13 FIXME
	mtspr	256, r0

C First loop until the destination is 16-byte aligned.  This will execute 0 or 1
C times for 64-bit machines, and 0 to 3 times for 32-bit machines.

LIMB32(`rlwinm.	r0, rp, 30,30,31')	C (rp >> 2) mod 4
LIMB64(`rlwinm.	r0, rp, 29,31,31')	C (rp >> 3) mod 2
	beq	L(aligned)

	subfic	r7, r0, LIMBS_PER_VR
LIMB32(`li	r10, 0		')
	subf	n, r7, n
L(top0):
LIMB32(`lwz	r8, 0(up)	')
LIMB64(`ld	r8, 0(up)	')
	addi	up, up, GMP_LIMB_BYTES
LIMB32(`lwz	r9, 0(vp)	')
LIMB64(`ld	r9, 0(vp)	')
	addi	vp, vp, GMP_LIMB_BYTES
LIMB32(`addic.	r7, r7, -1	')
	logopS(	r0, r8, r9)
LIMB32(`stwx	r0, r10, rp	')
LIMB64(`std	r0, 0(rp)	')
LIMB32(`addi	r10, r10, GMP_LIMB_BYTES')
LIMB32(`bne	L(top0)		')

	addi	rp, rp, 16		C update rp, but preserve its alignment

L(aligned):
LIMB64(`srdi	r7, n, 1	')	C loop count corresponding to n
LIMB32(`srwi	r7, n, 2	')	C loop count corresponding to n
	mtctr	r7			C copy n to count register

	li	r10, 16
	lvsl	us, 0, up
	lvsl	vs, 0, vp

	lvx	v2, 0, up
	lvx	v3, 0, vp
	bdnz	L(gt1)
	lvx	v0, r10, up
	lvx	v1, r10, vp
	vperm	v4, v2, v0, us
	vperm	v5, v3, v1, vs
	vnegb(	v5, v5)
	logop(	v6, v4, v5)
	vnega(	v6, v6)
	stvx	v6, 0, rp
	addi	up, up, 16
	addi	vp, vp, 16
	addi	rp, rp, 4
	b	L(tail)

L(gt1):	addi	up, up, 16
	addi	vp, vp, 16

L(top):	lvx	v0, 0, up
	lvx	v1, 0, vp
	vperm	v4, v2, v0, us
	vperm	v5, v3, v1, vs
	vnegb(	v5, v5)
	logop(	v6, v4, v5)
	vnega(	v6, v6)
	stvx	v6, 0, rp
	bdz	L(end)
	lvx	v2, r10, up
	lvx	v3, r10, vp
	vperm	v4, v0, v2, us
	vperm	v5, v1, v3, vs
	vnegb(	v5, v5)
	logop(	v6, v4, v5)
	vnega(	v6, v6)
	stvx	v6, r10, rp
	addi	up, up, 32
	addi	vp, vp, 32
	addi	rp, rp, 32
	bdnz	L(top)

	andi.	r0, up, 15
	vxor	v0, v0, v0
	beq	1f
	lvx	v0, 0, up
1:	andi.	r0, vp, 15
	vxor	v1, v1, v1
	beq	1f
	lvx	v1, 0, vp
1:	vperm	v4, v2, v0, us
	vperm	v5, v3, v1, vs
	vnegb(	v5, v5)
	logop(	v6, v4, v5)
	vnega(	v6, v6)
	stvx	v6, 0, rp
	addi	rp, rp, 4
	b	L(tail)

L(end):	andi.	r0, up, 15
	vxor	v2, v2, v2
	beq	1f
	lvx	v2, r10, up
1:	andi.	r0, vp, 15
	vxor	v3, v3, v3
	beq	1f
	lvx	v3, r10, vp
1:	vperm	v4, v0, v2, us
	vperm	v5, v1, v3, vs
	vnegb(	v5, v5)
	logop(	v6, v4, v5)
	vnega(	v6, v6)
	stvx	v6, r10, rp

	addi	up, up, 16
	addi	vp, vp, 16
	addi	rp, rp, 20

L(tail):
LIMB32(`rlwinm.	r7, n, 0,30,31	')	C r7 = n mod 4
LIMB64(`rlwinm.	r7, n, 0,31,31	')	C r7 = n mod 2
	beq	L(ret)
	addi	rp, rp, 15
LIMB32(`rlwinm	rp, rp, 0,0,27	')
LIMB64(`rldicr	rp, rp, 0,59	')
	li	r10, 0
L(top2):
LIMB32(`lwzx	r8, r10, up	')
LIMB64(`ldx	r8, r10, up	')
LIMB32(`lwzx	r9, r10, vp	')
LIMB64(`ldx	r9, r10, vp	')
LIMB32(`addic.	r7, r7, -1	')
	logopS(	r0, r8, r9)
LIMB32(`stwx	r0, r10, rp	')
LIMB64(`std	r0, 0(rp)	')
LIMB32(`addi	r10, r10, GMP_LIMB_BYTES')
LIMB32(`bne	L(top2)		')

L(ret):	mtspr	256, r12
	blr
EPILOGUE()

C This works for 64-bit PowerPC, since a limb ptr can only be aligned
C in 2 relevant ways, which means we can always find a pair of aligned
C pointers of rp, up, and vp.
C process words until rp is 16-byte aligned
C if (((up | vp) & 15) == 0)
C   process with VMX without any vperm
C else if ((up & 15) != 0 && (vp & 15) != 0)
C   process with VMX using vperm on store data
C else if ((up & 15) != 0)
C   process with VMX using vperm on up data
C else
C   process with VMX using vperm on vp data
C
C	rlwinm,	r0, up, 0,28,31
C	rlwinm	r0, vp, 0,28,31
C	cmpwi	cr7, r0, 0
C	cror	cr6, cr0, cr7
C	crand	cr0, cr0, cr7
