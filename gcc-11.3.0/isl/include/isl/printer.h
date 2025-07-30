#ifndef ISL_PRINTER_H
#define ISL_PRINTER_H

#include <stdio.h>
#include <isl/ctx.h>
#include <isl/printer_type.h>
#include <isl/id.h>

#if defined(__cplusplus)
extern "C" {
#endif

__isl_give isl_printer *isl_printer_to_file(isl_ctx *ctx, FILE *file);
__isl_give isl_printer *isl_printer_to_str(isl_ctx *ctx);
__isl_null isl_printer *isl_printer_free(__isl_take isl_printer *printer);

isl_ctx *isl_printer_get_ctx(__isl_keep isl_printer *printer);
FILE *isl_printer_get_file(__isl_keep isl_printer *printer);

__isl_give char *isl_printer_get_str(__isl_keep isl_printer *printer);

__isl_give isl_printer *isl_printer_set_indent(__isl_take isl_printer *p,
	int indent);
__isl_give isl_printer *isl_printer_indent(__isl_take isl_printer *p,
	int indent);

#define ISL_FORMAT_ISL			0
#define ISL_FORMAT_POLYLIB		1
#define ISL_FORMAT_POLYLIB_CONSTRAINTS	2
#define ISL_FORMAT_OMEGA		3
#define ISL_FORMAT_C			4
#define ISL_FORMAT_LATEX		5
#define ISL_FORMAT_EXT_POLYLIB		6
__isl_give isl_printer *isl_printer_set_output_format(__isl_take isl_printer *p,
	int output_format);
int isl_printer_get_output_format(__isl_keep isl_printer *p);

#define ISL_YAML_STYLE_BLOCK		0
#define ISL_YAML_STYLE_FLOW		1
__isl_give isl_printer *isl_printer_set_yaml_style(__isl_take isl_printer *p,
	int yaml_style);
int isl_printer_get_yaml_style(__isl_keep isl_printer *p);

__isl_give isl_printer *isl_printer_set_indent_prefix(__isl_take isl_printer *p,
	const char *prefix);
__isl_give isl_printer *isl_printer_set_prefix(__isl_take isl_printer *p,
	const char *prefix);
__isl_give isl_printer *isl_printer_set_suffix(__isl_take isl_printer *p,
	const char *suffix);
__isl_give isl_printer *isl_printer_set_isl_int_width(__isl_take isl_printer *p,
	int width);

isl_bool isl_printer_has_note(__isl_keep isl_printer *p,
	__isl_keep isl_id *id);
__isl_give isl_id *isl_printer_get_note(__isl_keep isl_printer *p,
	__isl_take isl_id *id);
__isl_give isl_printer *isl_printer_set_note(__isl_take isl_printer *p,
	__isl_take isl_id *id, __isl_take isl_id *note);

__isl_give isl_printer *isl_printer_start_line(__isl_take isl_printer *p);
__isl_give isl_printer *isl_printer_end_line(__isl_take isl_printer *p);
__isl_give isl_printer *isl_printer_print_double(__isl_take isl_printer *p,
	double d);
__isl_give isl_printer *isl_printer_print_int(__isl_take isl_printer *p, int i);
__isl_give isl_printer *isl_printer_print_str(__isl_take isl_printer *p,
	const char *s);

__isl_give isl_printer *isl_printer_yaml_start_mapping(
	__isl_take isl_printer *p);
__isl_give isl_printer *isl_printer_yaml_end_mapping(
	__isl_take isl_printer *p);
__isl_give isl_printer *isl_printer_yaml_start_sequence(
	__isl_take isl_printer *p);
__isl_give isl_printer *isl_printer_yaml_end_sequence(
	__isl_take isl_printer *p);
__isl_give isl_printer *isl_printer_yaml_next(__isl_take isl_printer *p);

__isl_give isl_printer *isl_printer_flush(__isl_take isl_printer *p);

#if defined(__cplusplus)
}
#endif

#endif
