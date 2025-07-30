dnl  Intel P5 mpn_lshift -- mpn left shift.

dnl  Copyright 2000-2002 Free Software Foundation, Inc.

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


C P5: 1.75 cycles/limb.


C mp_limb_t mpn_lshift (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                       unsigned shift);
C
C Shift src,size left by shift many bits and store the result in dst,size.
C Zeros are shifted in at the right.  Return the bits shifted out at the
C left.
C
C The comments in mpn_rshift apply here too.

defframe(PARAM_SHIFT,16)
defframe(PARAM_SIZE, 12)
defframe(PARAM_SRC,  8)
defframe(PARAM_DST,  4)
deflit(`FRAME',0)

dnl  minimum 5, because the unrolled loop can't handle less
deflit(UNROLL_THRESHOLD, 5)

	TEXT
	ALIGN(8)

PROLOGUE(mpn_lshift)

	pushl	%ebx
	pushl	%edi
deflit(`FRAME',8)

	movl	PARAM_SIZE, %eax
	movl	PARAM_DST, %edx

	movl	PARAM_SRC, %ebx
	movl	PARAM_SHIFT, %ecx

	cmp	$UNROLL_THRESHOLD, %eax
	jae	L(unroll)

	movl	-4(%ebx,%eax,4), %edi	C src high limb
	decl	%eax

	jnz	L(simple)

	shldl(	%cl, %edi, %eax)	C eax was decremented to zero

	shll	%cl, %edi

	movl	%edi, (%edx)		C dst low limb
	popl	%edi			C risk of data cache bank clash

	popl	%ebx

	ret


C -----------------------------------------------------------------------------
L(simple):
	C eax	size-1
	C ebx	src
	C ecx	shift
	C edx	dst
	C esi
	C edi
	C ebp
deflit(`FRAME',8)

	movd	(%ebx,%eax,4), %mm5	C src high limb

	movd	%ecx, %mm6		C lshift
	negl	%ecx

	psllq	%mm6, %mm5
	addl	$32, %ecx

	movd	%ecx, %mm7
	psrlq	$32, %mm5		C retval


L(simple_top):
	C eax	counter, limbs, negative
	C ebx	src
	C ecx
	C edx	dst
	C esi
	C edi
	C
	C mm0	scratch
	C mm5	return value
	C mm6	shift
	C mm7	32-shift

	movq	-4(%ebx,%eax,4), %mm0
	decl	%eax

	psrlq	%mm7, %mm0

	C

	movd	%mm0, 4(%edx,%eax,4)
	jnz	L(simple_top)


	movd	(%ebx), %mm0

	movd	%mm5, %eax
	psllq	%mm6, %mm0

	popl	%edi
	popl	%ebx

	movd	%mm0, (%edx)

	emms

	ret


C -----------------------------------------------------------------------------
	ALIGN(8)
L(unroll):
	C eax	size
	C ebx	src
	C ecx	shift
	C edx	dst
	C esi
	C edi
	C ebp
deflit(`FRAME',8)

	movd	-4(%ebx,%eax,4), %mm5	C src high limb
	leal	(%ebx,%eax,4), %edi

	movd	%ecx, %mm6		C lshift
	andl	$4, %edi

	psllq	%mm6, %mm5
	jz	L(start_src_aligned)


	C src isn't aligned, process high limb separately (marked xxx) to
	C make it so.
	C
	C  source     -8(ebx,%eax,4)
	C                  |
	C  +-------+-------+-------+--
	C  |               |
	C  +-------+-------+-------+--
	C        0mod8   4mod8   0mod8
	C
	C  dest
	C     -4(edx,%eax,4)
	C          |
	C  +-------+-------+--
	C  |  xxx  |       |
	C  +-------+-------+--

	movq	-8(%ebx,%eax,4), %mm0	C unaligned load

	psllq	%mm6, %mm0
	decl	%eax

	psrlq	$32, %mm0

	C

	movd	%mm0, (%edx,%eax,4)
L(start_src_aligned):

	movq	-8(%ebx,%eax,4), %mm1	C src high qword
	leal	(%edx,%eax,4), %edi

	andl	$4, %edi
	psrlq	$32, %mm5		C return value

	movq	-16(%ebx,%eax,4), %mm3	C src second highest qword
	jz	L(start_dst_aligned)

	C dst isn't aligned, subtract 4 to make it so, and pretend the shift
	C is 32 bits extra.  High limb of dst (marked xxx) handled here
	C separately.
	C
	C  source     -8(ebx,%eax,4)
	C                  |
	C  +-------+-------+--
	C  |      mm1      |
	C  +-------+-------+--
	C                0mod8   4mod8
	C
	C  dest
	C     -4(edx,%eax,4)
	C          |
	C  +-------+-------+-------+--
	C  |  xxx  |               |
	C  +-------+-------+-------+--
	C        0mod8   4mod8   0mod8

	movq	%mm1, %mm0
	addl	$32, %ecx		C new shift

	psllq	%mm6, %mm0

	movd	%ecx, %mm6
	psrlq	$32, %mm0

	C wasted cycle here waiting for %mm0

	movd	%mm0, -4(%edx,%eax,4)
	subl	$4, %edx
L(start_dst_aligned):


	psllq	%mm6, %mm1
	negl	%ecx			C -shift

	addl	$64, %ecx		C 64-shift
	movq	%mm3, %mm2

	movd	%ecx, %mm7
	subl	$8, %eax		C size-8

	psrlq	%mm7, %mm3

	por	%mm1, %mm3		C mm3 ready to store
	jc	L(finish)


	C The comments in mpn_rshift apply here too.

	ALIGN(8)
L(unroll_loop):
	C eax	counter, limbs
	C ebx	src
	C ecx
	C edx	dst
	C esi
	C edi
	C
	C mm0
	C mm1
	C mm2	src qword from 16(%ebx,%eax,4)
	C mm3	dst qword ready to store to 24(%edx,%eax,4)
	C
	C mm5	return value
	C mm6	lshift
	C mm7	rshift

	movq	8(%ebx,%eax,4), %mm0
	psllq	%mm6, %mm2

	movq	%mm0, %mm1
	psrlq	%mm7, %mm0

	movq	%mm3, 24(%edx,%eax,4)	C prev
	por	%mm2, %mm0

	movq	(%ebx,%eax,4), %mm3	C
	psllq	%mm6, %mm1		C

	movq	%mm0, 16(%edx,%eax,4)
	movq	%mm3, %mm2		C

	psrlq	%mm7, %mm3		C
	subl	$4, %eax

	por	%mm1, %mm3		C
	jnc	L(unroll_loop)



L(finish):
	C eax	-4 to -1 representing respectively 0 to 3 limbs remaining

	testb	$2, %al

	jz	L(finish_no_two)

	movq	8(%ebx,%eax,4), %mm0
	psllq	%mm6, %mm2

	movq	%mm0, %mm1
	psrlq	%mm7, %mm0

	movq	%mm3, 24(%edx,%eax,4)	C prev
	por	%mm2, %mm0

	movq	%mm1, %mm2
	movq	%mm0, %mm3

	subl	$2, %eax
L(finish_no_two):


	C eax	-4 or -3 representing respectively 0 or 1 limbs remaining
	C
	C mm2	src prev qword, from 16(%ebx,%eax,4)
	C mm3	dst qword, for 24(%edx,%eax,4)

	testb	$1, %al
	movd	%mm5, %eax	C retval

	popl	%edi
	jz	L(finish_zero)


	C One extra src limb, destination was aligned.
	C
	C                 source                  ebx
	C                 --+---------------+-------+
	C                   |      mm2      |       |
	C                 --+---------------+-------+
	C
	C dest         edx+12           edx+4     edx
	C --+---------------+---------------+-------+
	C   |      mm3      |               |       |
	C --+---------------+---------------+-------+
	C
	C mm6 = shift
	C mm7 = ecx = 64-shift


	C One extra src limb, destination was unaligned.
	C
	C                 source                  ebx
	C                 --+---------------+-------+
	C                   |      mm2      |       |
	C                 --+---------------+-------+
	C
	C         dest         edx+12           edx+4
	C         --+---------------+---------------+
	C           |      mm3      |               |
	C         --+---------------+---------------+
	C
	C mm6 = shift+32
	C mm7 = ecx = 64-(shift+32)


	C In both cases there's one extra limb of src to fetch and combine
	C with mm2 to make a qword at 4(%edx), and in the aligned case
	C there's an extra limb of dst to be formed from that extra src limb
	C left shifted.


	movd	(%ebx), %mm0
	psllq	%mm6, %mm2

	movq	%mm3, 12(%edx)
	psllq	$32, %mm0

	movq	%mm0, %mm1
	psrlq	%mm7, %mm0

	por	%mm2, %mm0
	psllq	%mm6, %mm1

	movq	%mm0, 4(%edx)
	psrlq	$32, %mm1

	andl	$32, %ecx
	popl	%ebx

	jz	L(finish_one_unaligned)

	movd	%mm1, (%edx)
L(finish_one_unaligned):

	emms

	ret


L(finish_zero):

	C No extra src limbs, destination was aligned.
	C
	C                 source          ebx
	C                 --+---------------+
	C                   |      mm2      |
	C                 --+---------------+
	C
	C dest          edx+8             edx
	C --+---------------+---------------+
	C   |      mm3      |               |
	C --+---------------+---------------+
	C
	C mm6 = shift
	C mm7 = ecx = 64-shift


	C No extra src limbs, destination was unaligned.
	C
	C               source            ebx
	C                 --+---------------+
	C                   |      mm2      |
	C                 --+---------------+
	C
	C         dest          edx+8   edx+4
	C         --+---------------+-------+
	C           |      mm3      |       |
	C         --+---------------+-------+
	C
	C mm6 = shift+32
	C mm7 = ecx = 64-(shift+32)


	C The movd for the unaligned case writes the same data to 4(%edx)
	C that the movq does for the aligned case.


	movq	%mm3, 8(%edx)
	andl	$32, %ecx

	psllq	%mm6, %mm2
	jz	L(finish_zero_unaligned)

	movq	%mm2, (%edx)
L(finish_zero_unaligned):

	psrlq	$32, %mm2
	popl	%ebx

	movd	%mm5, %eax	C retval

	movd	%mm2, 4(%edx)

	emms

	ret

EPILOGUE()
