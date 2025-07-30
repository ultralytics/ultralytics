# valgrind-tests.m4 serial 3
dnl Copyright (C) 2008-2014 Free Software Foundation, Inc.
dnl This file is free software; the Free Software Foundation
dnl gives unlimited permission to copy and/or distribute it,
dnl with or without modifications, as long as this notice is preserved.

dnl From Simon Josefsson
dnl with adaptations to MPC

# gl_VALGRIND_TESTS()
# -------------------
# Check if valgrind is available, and set VALGRIND to it if available.
AC_DEFUN([gl_VALGRIND_TESTS],
[
  # Run self-tests under valgrind?
  if test "$cross_compiling" = no; then
    AC_CHECK_PROGS(VALGRIND, valgrind)
  fi

  OPTS="-q --error-exitcode=1 --leak-check=full"

  if test -n "$VALGRIND" \
     && $VALGRIND $OPTS $SHELL -c 'exit 0' > /dev/null 2>&1; then
    opt_valgrind_tests=yes
    VALGRIND="$VALGRIND $OPTS"
# Addition AE: enable suppression file through a shell variable
    AC_MSG_CHECKING([for valgrind suppression file])
    if test -n "$VALGRIND_SUPPRESSION"; then
       AC_MSG_RESULT($VALGRIND_SUPPRESSION)
       VALGRIND="$VALGRIND --suppressions=$VALGRIND_SUPPRESSION"
    else
       AC_MSG_RESULT([no])
    fi
    AC_DEFINE([MPC_USE_VALGRIND], 1, [Use valgrind for make check])
  else
    opt_valgrind_tests=no
    VALGRIND=
  fi

  AC_MSG_CHECKING([whether self tests are run under valgrind])
  AC_MSG_RESULT($opt_valgrind_tests)
])

