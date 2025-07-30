dnl  AMD K6-2 mpn_and_n, mpn_andn_n, mpn_nand_n, mpn_ior_n, mpn_iorn_n,
dnl  mpn_nior_n, mpn_xor_n, mpn_xnor_n -- mpn bitwise logical operations.

dnl  Copyright 1999-2002 Free Software Foundation, Inc.

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

NAILS_SUPPORT(0-31)


C         alignment dst/src1/src2, A=0mod8, N=4mod8
C      A/A/A A/A/N A/N/A A/N/N N/A/A N/A/N N/N/A N/N/N
C
C K6-2  1.2   1.5   1.5   1.2   1.2   1.5   1.5   1.2   and,andn,ior,xor
C K6-2  1.5   1.75  2.0   1.75  1.75  2.0   1.75  1.5   iorn,xnor
C K6-2  1.75  2.0   2.0   2.0   2.0   2.0   2.0   1.75  nand,nior
C
C K6    1.5   1.68  1.75  1.2   1.75  1.75  1.68  1.5   and,andn,ior,xor
C K6    2.0   2.0   2.25  2.25  2.25  2.25  2.0   2.0   iorn,xnor
C K6    2.0   2.25  2.25  2.25  2.25  2.25  2.25  2.0   nand,nior


dnl  M4_p and M4_i are the MMX and integer instructions
dnl  M4_*_neg_dst means whether to negate the final result before writing
dnl  M4_*_neg_src2 means whether to negate the src2 values before using them

define(M4_choose_op,
m4_assert_numargs(7)
`ifdef(`OPERATION_$1',`
define(`M4_function',  `mpn_$1')
define(`M4_operation', `$1')
define(`M4_p',         `$2')
define(`M4_p_neg_dst', `$3')
define(`M4_p_neg_src2',`$4')
define(`M4_i',         `$5')
define(`M4_i_neg_dst', `$6')
define(`M4_i_neg_src2',`$7')
')')

dnl  xnor is done in "iorn" style because it's a touch faster than "nior"
dnl  style (the two are equivalent for xor).
dnl
dnl  pandn can't be used with nails.

M4_choose_op( and_n,  pand,0,0,  andl,0,0)
ifelse(GMP_NAIL_BITS,0,
`M4_choose_op(andn_n, pandn,0,0, andl,0,1)',
`M4_choose_op(andn_n, pand,0,1,  andl,0,1)')
M4_choose_op( nand_n, pand,1,0,  andl,1,0)
M4_choose_op( ior_n,  por,0,0,   orl,0,0)
M4_choose_op( iorn_n, por,0,1,   orl,0,1)
M4_choose_op( nior_n, por,1,0,   orl,1,0)
M4_choose_op( xor_n,  pxor,0,0,  xorl,0,0)
M4_choose_op( xnor_n, pxor,0,1,  xorl,0,1)

ifdef(`M4_function',,
`m4_error(`Unrecognised or undefined OPERATION symbol
')')

MULFUNC_PROLOGUE(mpn_and_n mpn_andn_n mpn_nand_n mpn_ior_n mpn_iorn_n mpn_nior_n mpn_xor_n mpn_xnor_n)


C void M4_function (mp_ptr dst, mp_srcptr src1, mp_srcptr src2,
C                   mp_size_t size);
C
C Do src1,size M4_operation src2,size, storing the result in dst,size.
C
C Unaligned movq loads and stores are a bit slower than aligned ones.  The
C test at the start of the routine checks the alignment of src1 and if
C necessary processes one limb separately at the low end to make it aligned.
C
C The raw speeds without this alignment switch are as follows.
C
C           alignment dst/src1/src2, A=0mod8, N=4mod8
C     A/A/A  A/A/N  A/N/A  A/N/N  N/A/A  N/A/N  N/N/A  N/N/N
C
C K6                 1.5    2.0                 1.5    2.0    and,andn,ior,xor
C K6                 1.75   2.2                 2.0    2.28   iorn,xnor
C K6                 2.0    2.25                2.35   2.28   nand,nior
C
C
C Future:
C
C K6 can do one 64-bit load per cycle so each of these routines should be
C able to approach 1.0 c/l, if aligned.  The basic and/andn/ior/xor might be
C able to get 1.0 with just a 4 limb loop, being 3 instructions per 2 limbs.
C The others are 4 instructions per 2 limbs, and so can only approach 1.0
C because there's nowhere to hide some loop control.

defframe(PARAM_SIZE,16)
defframe(PARAM_SRC2,12)
defframe(PARAM_SRC1,8)
defframe(PARAM_DST, 4)
deflit(`FRAME',0)

	TEXT
	ALIGN(32)
PROLOGUE(M4_function)
			movl	PARAM_SIZE, %ecx
			pushl	%ebx		FRAME_pushl()

			movl	PARAM_SRC1, %eax

			movl	PARAM_SRC2, %ebx
			cmpl	$1, %ecx

			movl	PARAM_DST, %edx
			ja	L(two_or_more)


			movl	(%ebx), %ecx
			popl	%ebx
ifelse(M4_i_neg_src2,1,`notl_or_xorl_GMP_NUMB_MASK(	%ecx)')
			M4_i	(%eax), %ecx
ifelse(M4_i_neg_dst,1,`	notl_or_xorl_GMP_NUMB_MASK(	%ecx)')
			movl	%ecx, (%edx)

			ret


L(two_or_more):
			C eax	src1
			C ebx	src2
			C ecx	size
			C edx	dst
			C esi
			C edi
			C ebp

			pushl	%esi		FRAME_pushl()
			testl	$4, %eax
			jz	L(alignment_ok)

			movl	(%ebx), %esi
			addl	$4, %ebx
ifelse(M4_i_neg_src2,1,`notl_or_xorl_GMP_NUMB_MASK(	%esi)')
			M4_i	(%eax), %esi
			addl	$4, %eax
ifelse(M4_i_neg_dst,1,`	notl_or_xorl_GMP_NUMB_MASK(	%esi)')
			movl	%esi, (%edx)
			addl	$4, %edx
			decl	%ecx

L(alignment_ok):
			movl	%ecx, %esi
			shrl	%ecx
			jnz	L(still_two_or_more)

			movl	(%ebx), %ecx
			popl	%esi
ifelse(M4_i_neg_src2,1,`notl_or_xorl_GMP_NUMB_MASK(	%ecx)')
			M4_i	(%eax), %ecx
ifelse(M4_i_neg_dst,1,`	notl_or_xorl_GMP_NUMB_MASK(	%ecx)')
			popl	%ebx
			movl	%ecx, (%edx)
			ret


L(still_two_or_more):
ifelse(eval(M4_p_neg_src2 || M4_p_neg_dst),1,`
			pcmpeqd	%mm7, %mm7		C all ones
ifelse(GMP_NAIL_BITS,0,,`psrld	$GMP_NAIL_BITS, %mm7')	C clear nails
')

			ALIGN(16)
L(top):
			C eax	src1
			C ebx	src2
			C ecx	counter
			C edx	dst
			C esi
			C edi
			C ebp
			C
			C carry bit is low of size

			movq	-8(%ebx,%ecx,8), %mm0
ifelse(M4_p_neg_src2,1,`pxor	%mm7, %mm0')
			M4_p	-8(%eax,%ecx,8), %mm0
ifelse(M4_p_neg_dst,1,`	pxor	%mm7, %mm0')
			movq	%mm0, -8(%edx,%ecx,8)

			loop	L(top)


			jnc	L(no_extra)

			movl	-4(%ebx,%esi,4), %ebx
ifelse(M4_i_neg_src2,1,`notl_or_xorl_GMP_NUMB_MASK(	%ebx)')
			M4_i	-4(%eax,%esi,4), %ebx
ifelse(M4_i_neg_dst,1,`	notl_or_xorl_GMP_NUMB_MASK(	%ebx)')
			movl	%ebx, -4(%edx,%esi,4)
L(no_extra):

			popl	%esi
			popl	%ebx
			emms_or_femms
			ret

EPILOGUE()
