dnl Add a set of flags to WARNING_FLAGS, that enable compiler warnings for
dnl isl. The warnings that are enabled vary with the compiler and only include
dnl warnings that did not trigger at the time of adding these flags.
AC_DEFUN([AX_SET_WARNING_FLAGS],[dnl
	AX_COMPILER_VENDOR

	WARNING_FLAGS=""

	if test "${ax_cv_c_compiler_vendor}" = "clang"; then
		dnl isl is at the moment clean of -Wall warnings.  If clang adds
		dnl new warnings to -Wall which cause false positives, the
		dnl specific warning types will be disabled explicitally (by
		dnl adding for example -Wno-return-type). To temporarily disable
		dnl all warnings run configure with CFLAGS=-Wno-all.
		WARNING_FLAGS="-Wall"
	fi
])
