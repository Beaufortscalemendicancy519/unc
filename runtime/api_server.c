// api_server.c — OpenAI-compatible HTTP API (stub)
// Full implementation requires an HTTP library (e.g., mongoose or libuv).

#include "runtime.h"
#include <stdio.h>

void unc_serve(struct UNCRuntime* rt, int port) {
    (void)rt;
    fprintf(stderr, "[unc] HTTP API server on port %d (stub — not yet implemented)\n", port);
    fprintf(stderr, "[unc] In future releases this will expose:\n");
    fprintf(stderr, "[unc]   POST /v1/chat/completions\n");
    fprintf(stderr, "[unc]   POST /v1/completions\n");
}
