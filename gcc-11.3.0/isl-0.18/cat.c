#include <assert.h>
#include <isl/obj.h>
#include <isl/printer.h>
#include <isl/stream.h>
#include <isl/options.h>

struct isl_arg_choice cat_format[] = {
	{"isl",		ISL_FORMAT_ISL},
	{"omega",	ISL_FORMAT_OMEGA},
	{"polylib",	ISL_FORMAT_POLYLIB},
	{"ext-polylib",	ISL_FORMAT_EXT_POLYLIB},
	{"latex",	ISL_FORMAT_LATEX},
	{0}
};

struct isl_arg_choice cat_yaml_style[] = {
	{ "block",	ISL_YAML_STYLE_BLOCK },
	{ "flow",	ISL_YAML_STYLE_FLOW },
	{ 0 }
};

struct cat_options {
	struct isl_options	*isl;
	unsigned		 format;
	unsigned		 yaml_style;
};

ISL_ARGS_START(struct cat_options, cat_options_args)
ISL_ARG_CHILD(struct cat_options, isl, "isl", &isl_options_args, "isl options")
ISL_ARG_CHOICE(struct cat_options, format, 0, "format", \
	cat_format,	ISL_FORMAT_ISL, "output format")
ISL_ARG_CHOICE(struct cat_options, yaml_style, 0, "yaml-style", \
	cat_yaml_style, ISL_YAML_STYLE_BLOCK, "output YAML style")
ISL_ARGS_END

ISL_ARG_DEF(cat_options, struct cat_options, cat_options_args)

int main(int argc, char **argv)
{
	struct isl_ctx *ctx;
	isl_stream *s;
	struct isl_obj obj;
	struct cat_options *options;
	isl_printer *p;

	options = cat_options_new_with_defaults();
	assert(options);
	argc = cat_options_parse(options, argc, argv, ISL_ARG_ALL);

	ctx = isl_ctx_alloc_with_options(&cat_options_args, options);

	s = isl_stream_new_file(ctx, stdin);
	obj = isl_stream_read_obj(s);
	isl_stream_free(s);

	p = isl_printer_to_file(ctx, stdout);
	p = isl_printer_set_output_format(p, options->format);
	p = isl_printer_set_yaml_style(p, options->yaml_style);
	p = obj.type->print(p, obj.v);
	p = isl_printer_end_line(p);
	isl_printer_free(p);

	obj.type->free(obj.v);

	isl_ctx_free(ctx);

	return 0;
}
