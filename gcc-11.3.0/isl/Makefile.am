if HAVE_CLANG
    MAYBE_INTERFACE = interface
endif
SUBDIRS = . $(MAYBE_INTERFACE) doc
DIST_SUBDIRS = $(MAYBE_INTERFACE) doc

ACLOCAL_AMFLAGS = -I m4
AUTOMAKE_OPTIONS = nostdinc subdir-objects

lib_LTLIBRARIES = libisl.la
noinst_PROGRAMS = isl_test isl_polyhedron_sample isl_pip \
	isl_polyhedron_minimize isl_polytope_scan \
	isl_polyhedron_detect_equalities isl_cat \
	isl_closure isl_bound isl_schedule isl_codegen isl_test_int
TESTS = isl_test codegen_test.sh pip_test.sh bound_test.sh isl_test_int

if IMATH_FOR_MP

MP_SRC = \
	isl_hide_deprecated.h \
	isl_imath.c \
	isl_imath.h \
	isl_int_imath.h \
	imath_wrap/gmp_compat.h \
	imath_wrap/imath.h \
	imath_wrap/imrat.h \
	imath_wrap/wrap.h \
	imath_wrap/gmp_compat.c \
	imath_wrap/imath.c \
	imath_wrap/imrat.c

noinst_PROGRAMS += isl_test_imath
TESTS += isl_test_imath

if SMALL_INT_OPT
MP_SRC += isl_int_sioimath.h \
	isl_int_sioimath.c \
	isl_val_sioimath.c
else
MP_SRC += isl_val_imath.c
endif

DEPRECATED_SRC =
MP_INCLUDE_H =
endif

if GMP_FOR_MP
if NEED_GET_MEMORY_FUNCTIONS
GET_MEMORY_FUNCTIONS=mp_get_memory_functions.c
endif

MP_SRC = \
	$(GET_MEMORY_FUNCTIONS) \
	isl_int_gmp.h \
	isl_gmp.c \
	isl_val_gmp.c

DEPRECATED_SRC = isl_ast_int.c
MP_INCLUDE_H = include/isl/val_gmp.h
endif

AM_CPPFLAGS = -I. -I$(srcdir) -I$(srcdir)/include -Iinclude/ @MP_CPPFLAGS@
AM_CFLAGS = @WARNING_FLAGS@

libisl_la_SOURCES = \
	$(MP_SRC) \
	$(DEPRECATED_SRC) \
	isl_aff.c \
	isl_aff_private.h \
	isl_affine_hull.c \
	isl_arg.c \
	isl_ast.c \
	isl_ast_private.h \
	isl_ast_build.c \
	isl_ast_build_private.h \
	isl_ast_build_expr.c \
	isl_ast_build_expr.h \
	isl_ast_codegen.c \
	isl_ast_graft.c \
	isl_ast_graft_private.h \
	isl_band.c \
	isl_band_private.h \
	isl_basis_reduction.h \
	basis_reduction_tab.c \
	isl_bernstein.c \
	isl_bernstein.h \
	isl_blk.c \
	isl_blk.h \
	isl_bound.c \
	isl_bound.h \
	isl_coalesce.c \
	isl_constraint.c \
	isl_constraint_private.h \
	isl_convex_hull.c \
	isl_ctx.c \
	isl_ctx_private.h \
	isl_deprecated.c \
	isl_dim_map.h \
	isl_dim_map.c \
	isl_equalities.c \
	isl_equalities.h \
	isl_factorization.c \
	isl_factorization.h \
	isl_farkas.c \
	isl_ffs.c \
	isl_flow.c \
	isl_fold.c \
	isl_hash.c \
	isl_hash_private.h \
	isl_id_to_ast_expr.c \
	isl_id_to_id.c \
	isl_id_to_pw_aff.c \
	isl_ilp.c \
	isl_ilp_private.h \
	isl_input.c \
	isl_int.h \
	isl_local.h \
	isl_local.c \
	isl_local_space_private.h \
	isl_local_space.c \
	isl_lp.c \
	isl_lp_private.h \
	isl_map.c \
	isl_map_list.c \
	isl_map_simplify.c \
	isl_map_subtract.c \
	isl_map_private.h \
	isl_map_to_basic_set.c \
	isl_mat.c \
	isl_mat_private.h \
	isl_morph.c \
	isl_morph.h \
	isl_id.c \
	isl_id_private.h \
	isl_obj.c \
	isl_options.c \
	isl_options_private.h \
	isl_output.c \
	isl_output_private.h \
	isl_point_private.h \
	isl_point.c \
	isl_polynomial_private.h \
	isl_polynomial.c \
	isl_printer_private.h \
	isl_printer.c \
	print.c \
	isl_range.c \
	isl_range.h \
	isl_reordering.c \
	isl_reordering.h \
	isl_sample.h \
	isl_sample.c \
	isl_scan.c \
	isl_scan.h \
	isl_schedule.c \
	isl_schedule_band.c \
	isl_schedule_band.h \
	isl_schedule_node.c \
	isl_schedule_node_private.h \
	isl_schedule_read.c \
	isl_schedule_tree.c \
	isl_schedule_tree.h \
	isl_schedule_private.h \
	isl_schedule_constraints.c \
	isl_schedule_constraints.h \
	isl_scheduler.c \
	isl_set_list.c \
	isl_sort.c \
	isl_sort.h \
	isl_space.c \
	isl_space_private.h \
	isl_stream.c \
	isl_stream_private.h \
	isl_seq.c \
	isl_seq.h \
	isl_tab.c \
	isl_tab.h \
	isl_tab_pip.c \
	isl_tarjan.c \
	isl_tarjan.h \
	isl_transitive_closure.c \
	isl_union_map.c \
	isl_union_map_private.h \
	isl_val.c \
	isl_val_private.h \
	isl_vec_private.h \
	isl_vec.c \
	isl_version.c \
	isl_vertices_private.h \
	isl_vertices.c \
	isl_yaml.h
libisl_la_LIBADD = @MP_LIBS@
libisl_la_LDFLAGS = -version-info @versioninfo@ \
	@MP_LDFLAGS@

isl_test_LDFLAGS = @MP_LDFLAGS@
isl_test_LDADD = libisl.la @MP_LIBS@

isl_test_int_LDFLAGS = @MP_LDFLAGS@
isl_test_int_LDADD = libisl.la @MP_LIBS@

if IMATH_FOR_MP
isl_test_imath_LDFLAGS = @MP_LDFLAGS@
isl_test_imath_LDADD = libisl.la @MP_LIBS@
endif

isl_polyhedron_sample_LDADD = libisl.la
isl_polyhedron_sample_SOURCES = \
	polyhedron_sample.c

isl_pip_LDFLAGS = @MP_LDFLAGS@
isl_pip_LDADD = libisl.la @MP_LIBS@
isl_pip_SOURCES = \
	pip.c

isl_schedule_LDFLAGS = @MP_LDFLAGS@
isl_schedule_LDADD = libisl.la @MP_LIBS@
isl_schedule_SOURCES = \
	schedule.c

isl_codegen_LDFLAGS = @MP_LDFLAGS@
isl_codegen_LDADD = libisl.la @MP_LIBS@
isl_codegen_SOURCES = \
	codegen.c

isl_bound_LDFLAGS = @MP_LDFLAGS@
isl_bound_LDADD = libisl.la @MP_LIBS@
isl_bound_SOURCES = \
	bound.c

isl_polyhedron_minimize_LDFLAGS = @MP_LDFLAGS@
isl_polyhedron_minimize_LDADD = libisl.la @MP_LIBS@
isl_polyhedron_minimize_SOURCES = \
	polyhedron_minimize.c

isl_polytope_scan_LDADD = libisl.la
isl_polytope_scan_SOURCES = \
	polytope_scan.c

isl_polyhedron_detect_equalities_LDADD = libisl.la
isl_polyhedron_detect_equalities_SOURCES = \
	polyhedron_detect_equalities.c

isl_cat_LDADD = libisl.la
isl_cat_SOURCES = \
	cat.c

isl_closure_LDADD = libisl.la
isl_closure_SOURCES = \
	closure.c

nodist_pkginclude_HEADERS = \
	include/isl/stdint.h
pkginclude_HEADERS = \
	$(MP_INCLUDE_H) \
	include/isl/aff.h \
	include/isl/aff_type.h \
	include/isl/arg.h \
	include/isl/ast.h \
	include/isl/ast_type.h \
	include/isl/ast_build.h \
	include/isl/band.h \
	include/isl/constraint.h \
	include/isl/ctx.h \
	include/isl/flow.h \
	include/isl/id.h \
	include/isl/id_to_ast_expr.h \
	include/isl/id_to_id.h \
	include/isl/id_to_pw_aff.h \
	include/isl/ilp.h \
	include/isl/hash.h \
	include/isl/hmap.h \
	include/isl/hmap_templ.c \
	include/isl/list.h \
	include/isl/local_space.h \
	include/isl/lp.h \
	include/isl/mat.h \
	include/isl/map.h \
	include/isl/map_to_basic_set.h \
	include/isl/map_type.h \
	include/isl/maybe.h \
	include/isl/maybe_ast_expr.h \
	include/isl/maybe_basic_set.h \
	include/isl/maybe_id.h \
	include/isl/maybe_pw_aff.h \
	include/isl/maybe_templ.h \
	include/isl/multi.h \
	include/isl/obj.h \
	include/isl/options.h \
	include/isl/point.h \
	include/isl/polynomial.h \
	include/isl/polynomial_type.h \
	include/isl/printer.h \
	include/isl/printer_type.h \
	include/isl/schedule.h \
	include/isl/schedule_node.h \
	include/isl/schedule_type.h \
	include/isl/set.h \
	include/isl/set_type.h \
	include/isl/space.h \
	include/isl/stream.h \
	include/isl/union_map.h \
	include/isl/union_map_type.h \
	include/isl/union_set.h \
	include/isl/union_set_type.h \
	include/isl/val.h \
	include/isl/vec.h \
	include/isl/version.h \
	include/isl/vertices.h
deprecateddir = $(pkgincludedir)/deprecated
deprecated_HEADERS = \
	include/isl/deprecated/int.h \
	include/isl/deprecated/aff_int.h \
	include/isl/deprecated/ast_int.h \
	include/isl/deprecated/constraint_int.h \
	include/isl/deprecated/ilp_int.h \
	include/isl/deprecated/map_int.h \
	include/isl/deprecated/mat_int.h \
	include/isl/deprecated/point_int.h \
	include/isl/deprecated/polynomial_int.h \
	include/isl/deprecated/set_int.h \
	include/isl/deprecated/union_map_int.h \
	include/isl/deprecated/val_int.h \
	include/isl/deprecated/vec_int.h

BUILT_SOURCES = gitversion.h

CLEANFILES = \
	gitversion.h

DISTCLEANFILES = \
	isl-uninstalled.sh \
	isl-uninstalled.pc \
	isl.pc \
	isl.pc.in \
	include/isl/stdint.h

EXTRA_DIST = \
	LICENSE \
	isl_config_post.h \
	basis_reduction_templ.c \
	bset_to_bmap.c \
	bset_from_bmap.c \
	isl_list_templ.c \
	isl_list_templ.h \
	isl_map_lexopt_templ.c \
	isl_multi_macro.h \
	isl_multi_templ.c \
	isl_multi_templ.h \
	isl_multi_apply_templ.c \
	isl_multi_apply_set.c \
	isl_multi_apply_union_set.c \
	isl_multi_cmp.c \
	isl_multi_coalesce.c \
	isl_multi_floor.c \
	isl_multi_gist.c \
	isl_multi_hash.c \
	isl_multi_intersect.c \
	print_templ.c \
	print_templ_yaml.c \
	isl_power_templ.c \
	isl_pw_macro.h \
	isl_pw_templ.c \
	isl_pw_hash.c \
	isl_pw_union_opt.c \
	read_in_string_templ.c \
	set_to_map.c \
	set_from_map.c \
	isl_tab_lexopt_templ.c \
	isl_union_macro.h \
	isl_union_templ.c \
	isl_union_single.c \
	isl_union_multi.c \
	isl_union_eval.c \
	isl_union_neg.c \
	isl.py \
	doc/CodingStyle \
	doc/SubmittingPatches \
	doc/implementation.tex \
	doc/isl.bib \
	doc/mypod2latex \
	doc/manual.tex \
	doc/reading.tex \
	doc/user.pod \
	imath/gmp_compat.c \
	imath/gmp_compat.h \
	imath/imath.c \
	imath/imath.h \
	imath/imrat.c \
	imath/imrat.h \
	interface/all.h \
	interface/isl.py.top \
	test_inputs

dist-hook:
	echo @GIT_HEAD_VERSION@ > $(distdir)/GIT_HEAD_ID
	(cd doc; make manual.pdf)
	cp doc/manual.pdf $(distdir)/doc/

pkgconfigdir=$(pkgconfig_libdir)
pkgconfig_DATA = $(pkgconfig_libfile)

gitversion.h: @GIT_HEAD@
	$(AM_V_GEN)echo '#define GIT_HEAD_ID "'@GIT_HEAD_VERSION@'"' > $@

install-data-local: $(srcdir)/isl.py
	@libisl=`sed -ne "/^library_names=/{s/.*='//;s/'$$//;s/ .*//;p;}" \
		 $(builddir)/libisl.la`; \
	case $$libisl in \
	'') echo Cannot find isl library name. GDB bindings not installed.;; \
	*) echo $(INSTALL_DATA) $(srcdir)/isl.py \
		$(DESTDIR)$(libdir)/$$libisl-gdb.py; \
	test -z "$(libdir)" || $(MKDIR_P) "$(DESTDIR)$(libdir)"; \
	$(INSTALL_DATA) $(srcdir)/isl.py $(DESTDIR)$(libdir)/$$libisl-gdb.py; esac

uninstall-local:
	@libisl=`sed -ne "/^library_names=/{s/.*='//;s/'$$//;s/ .*//;p;}" \
		 $(builddir)/libisl.la`; \
	if test -n "$${libisl}"; then \
		rm -f $(DESTDIR)$(libdir)/$$libisl-gdb.py; \
	fi
