#include <assert.h>
#include <isl/stream.h>
#include <isl_map_private.h>
#include <isl_polynomial_private.h>
#include <isl_scan.h>
#include <isl/options.h>
#include <isl/deprecated/point_int.h>
#include <isl/deprecated/polynomial_int.h>

struct bound_options {
	struct isl_options	*isl;
	unsigned		 verify;
	int			 print_all;
	int			 continue_on_error;
};

ISL_ARGS_START(struct bound_options, bound_options_args)
ISL_ARG_CHILD(struct bound_options, isl, "isl", &isl_options_args,
	"isl options")
ISL_ARG_BOOL(struct bound_options, verify, 'T', "verify", 0, NULL)
ISL_ARG_BOOL(struct bound_options, print_all, 'A', "print-all", 0, NULL)
ISL_ARG_BOOL(struct bound_options, continue_on_error, '\0', "continue-on-error", 0, NULL)
ISL_ARGS_END

ISL_ARG_DEF(bound_options, struct bound_options, bound_options_args)

static __isl_give isl_set *set_bounds(__isl_take isl_set *set)
{
	unsigned nparam;
	int i, r;
	isl_point *pt, *pt2;
	isl_set *box;

	nparam = isl_set_dim(set, isl_dim_param);
	r = nparam >= 8 ? 5 : nparam >= 5 ? 15 : 50;

	pt = isl_set_sample_point(isl_set_copy(set));
	pt2 = isl_point_copy(pt);

	for (i = 0; i < nparam; ++i) {
		pt = isl_point_add_ui(pt, isl_dim_param, i, r);
		pt2 = isl_point_sub_ui(pt2, isl_dim_param, i, r);
	}

	box = isl_set_box_from_points(pt, pt2);

	return isl_set_intersect(set, box);
}

struct verify_point_bound {
	struct bound_options *options;
	int stride;
	int n;
	int exact;
	int error;

	isl_pw_qpolynomial_fold *pwf;
	isl_pw_qpolynomial_fold *bound;
};

static isl_stat verify_point(__isl_take isl_point *pnt, void *user)
{
	int i;
	unsigned nvar;
	unsigned nparam;
	struct verify_point_bound *vpb = (struct verify_point_bound *) user;
	isl_int t;
	isl_ctx *ctx;
	isl_pw_qpolynomial_fold *pwf;
	isl_val *bound = NULL;
	isl_val *opt = NULL;
	isl_set *dom = NULL;
	isl_printer *p;
	const char *minmax;
	int bounded;
	int sign;
	int ok;
	FILE *out = vpb->options->print_all ? stdout : stderr;

	vpb->n--;

	if (1) {
		minmax = "ub";
		sign = 1;
	} else {
		minmax = "lb";
		sign = -1;
	}

	ctx = isl_point_get_ctx(pnt);
	p = isl_printer_to_file(ctx, out);

	isl_int_init(t);

	pwf = isl_pw_qpolynomial_fold_copy(vpb->pwf);

	nparam = isl_pw_qpolynomial_fold_dim(pwf, isl_dim_param);
	for (i = 0; i < nparam; ++i) {
		isl_point_get_coordinate(pnt, isl_dim_param, i, &t);
		pwf = isl_pw_qpolynomial_fold_fix_dim(pwf, isl_dim_param, i, t);
	}

	bound = isl_pw_qpolynomial_fold_eval(
				    isl_pw_qpolynomial_fold_copy(vpb->bound),
				    isl_point_copy(pnt));

	dom = isl_pw_qpolynomial_fold_domain(isl_pw_qpolynomial_fold_copy(pwf));
	bounded = isl_set_is_bounded(dom);

	if (bounded < 0)
		goto error;

	if (!bounded)
		opt = isl_pw_qpolynomial_fold_eval(
				    isl_pw_qpolynomial_fold_copy(pwf),
				    isl_set_sample_point(isl_set_copy(dom)));
	else if (sign > 0)
		opt = isl_pw_qpolynomial_fold_max(isl_pw_qpolynomial_fold_copy(pwf));
	else
		opt = isl_pw_qpolynomial_fold_min(isl_pw_qpolynomial_fold_copy(pwf));

	nvar = isl_set_dim(dom, isl_dim_set);
	if (vpb->exact && bounded)
		ok = isl_val_eq(opt, bound);
	else if (sign > 0)
		ok = isl_val_le(opt, bound);
	else
		ok = isl_val_le(bound, opt);
	if (ok < 0)
		goto error;

	if (vpb->options->print_all || !ok) {
		p = isl_printer_print_str(p, minmax);
		p = isl_printer_print_str(p, "(");
		for (i = 0; i < nparam; ++i) {
			if (i)
				p = isl_printer_print_str(p, ", ");
			isl_point_get_coordinate(pnt, isl_dim_param, i, &t);
			p = isl_printer_print_isl_int(p, t);
		}
		p = isl_printer_print_str(p, ") = ");
		p = isl_printer_print_val(p, bound);
		p = isl_printer_print_str(p, ", ");
		p = isl_printer_print_str(p, bounded ? "opt" : "sample");
		p = isl_printer_print_str(p, " = ");
		p = isl_printer_print_val(p, opt);
		if (ok)
			p = isl_printer_print_str(p, ". OK");
		else
			p = isl_printer_print_str(p, ". NOT OK");
		p = isl_printer_end_line(p);
	} else if ((vpb->n % vpb->stride) == 0) {
		p = isl_printer_print_str(p, "o");
		p = isl_printer_flush(p);
	}

	if (0) {
error:
		ok = 0;
	}

	isl_pw_qpolynomial_fold_free(pwf);
	isl_val_free(bound);
	isl_val_free(opt);
	isl_point_free(pnt);
	isl_set_free(dom);

	isl_int_clear(t);

	isl_printer_free(p);

	if (!ok)
		vpb->error = 1;

	if (vpb->options->continue_on_error)
		ok = 1;

	return (vpb->n >= 1 && ok) ? isl_stat_ok : isl_stat_error;
}

static int check_solution(__isl_take isl_pw_qpolynomial_fold *pwf,
	__isl_take isl_pw_qpolynomial_fold *bound, int exact,
	struct bound_options *options)
{
	struct verify_point_bound vpb;
	isl_int count, max;
	isl_set *dom;
	isl_set *context;
	int i, r, n;

	dom = isl_pw_qpolynomial_fold_domain(isl_pw_qpolynomial_fold_copy(pwf));
	context = isl_set_params(isl_set_copy(dom));
	context = isl_set_remove_divs(context);
	context = set_bounds(context);

	isl_int_init(count);
	isl_int_init(max);

	isl_int_set_si(max, 200);
	r = isl_set_count_upto(context, max, &count);
	assert(r >= 0);
	n = isl_int_get_si(count);

	isl_int_clear(max);
	isl_int_clear(count);

	vpb.options = options;
	vpb.pwf = pwf;
	vpb.bound = bound;
	vpb.n = n;
	vpb.stride = n > 70 ? 1 + (n + 1)/70 : 1;
	vpb.error = 0;
	vpb.exact = exact;

	if (!options->print_all) {
		for (i = 0; i < vpb.n; i += vpb.stride)
			printf(".");
		printf("\r");
		fflush(stdout);
	}

	isl_set_foreach_point(context, verify_point, &vpb);

	isl_set_free(context);
	isl_set_free(dom);
	isl_pw_qpolynomial_fold_free(pwf);
	isl_pw_qpolynomial_fold_free(bound);

	if (!options->print_all)
		printf("\n");

	if (vpb.error) {
		fprintf(stderr, "Check failed !\n");
		return -1;
	}

	return 0;
}

int main(int argc, char **argv)
{
	isl_ctx *ctx;
	isl_pw_qpolynomial_fold *copy;
	isl_pw_qpolynomial_fold *pwf;
	isl_stream *s;
	struct isl_obj obj;
	struct bound_options *options;
	int exact;
	int r = 0;

	options = bound_options_new_with_defaults();
	assert(options);
	argc = bound_options_parse(options, argc, argv, ISL_ARG_ALL);

	ctx = isl_ctx_alloc_with_options(&bound_options_args, options);

	s = isl_stream_new_file(ctx, stdin);
	obj = isl_stream_read_obj(s);
	if (obj.type == isl_obj_pw_qpolynomial)
		pwf = isl_pw_qpolynomial_fold_from_pw_qpolynomial(isl_fold_max,
								  obj.v);
	else if (obj.type == isl_obj_pw_qpolynomial_fold)
		pwf = obj.v;
	else {
		obj.type->free(obj.v);
		isl_die(ctx, isl_error_invalid, "invalid input", goto error);
	}

	if (options->verify)
		copy = isl_pw_qpolynomial_fold_copy(pwf);

	pwf = isl_pw_qpolynomial_fold_bound(pwf, &exact);
	pwf = isl_pw_qpolynomial_fold_coalesce(pwf);

	if (options->verify) {
		r = check_solution(copy, pwf, exact, options);
	} else {
		if (!exact)
			printf("# NOT exact\n");
		isl_pw_qpolynomial_fold_print(pwf, stdout, 0);
		fprintf(stdout, "\n");
		isl_pw_qpolynomial_fold_free(pwf);
	}

error:
	isl_stream_free(s);

	isl_ctx_free(ctx);

	return r;
}
