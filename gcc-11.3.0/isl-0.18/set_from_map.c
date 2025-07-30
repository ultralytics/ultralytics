#include <isl/map_type.h>

/* Return the set that was treated as the map "map".
 */
static __isl_give isl_set *set_from_map(__isl_take isl_map *map)
{
	return (isl_set *) map;
}
