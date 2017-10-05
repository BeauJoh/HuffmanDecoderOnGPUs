#ifdef __APPLE__
#include <err.h>
#include <string.h>
#define error(x,y,z) err(x,z)
#else
#include <error.h>
#endif
