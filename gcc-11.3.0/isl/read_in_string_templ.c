#include <isl/stream.h>

#define xCAT(A,B) A ## B
#define CAT(A,B) xCAT(A,B)
#undef TYPE
#define TYPE CAT(isl_,BASE)
#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

/* Read an object of type TYPE from "s", where the object may
 * either be specified directly or as a string.
 *
 * First check if the next token in "s" is a string.  If so, try and
 * extract the object from the string.
 * Otherwise, try and read the object directly from "s".
 */
static __isl_give TYPE *FN(read,BASE)(__isl_keep isl_stream *s)
{
	struct isl_token *tok;
	int type;

	tok = isl_stream_next_token(s);
	type = isl_token_get_type(tok);
	if (type == ISL_TOKEN_STRING) {
		char *str;
		isl_ctx *ctx;
		TYPE *res;

		ctx = isl_stream_get_ctx(s);
		str = isl_token_get_str(ctx, tok);
		res = FN(TYPE,read_from_str)(ctx, str);
		free(str);
		isl_token_free(tok);
		return res;
	}
	isl_stream_push_token(s, tok);
	return FN(isl_stream_read,BASE)(s);
}
