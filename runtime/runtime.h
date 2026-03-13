// runtime.h — UNC runtime shim API
#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct UNCRuntime UNCRuntime;

/**
 * Load a compiled .unc model file.
 * Returns NULL on failure.
 */
UNCRuntime* unc_load(const char* unc_path);

/**
 * Run text generation.
 * @param rt       Runtime handle
 * @param prompt   Input text (UTF-8)
 * @param max_tokens  Maximum tokens to generate
 * @param out_buf  Output buffer for generated text (caller-allocated)
 * @param out_size Size of output buffer
 */
void unc_generate(
    UNCRuntime*  rt,
    const char*  prompt,
    int          max_tokens,
    char*        out_buf,
    size_t       out_size
);

/**
 * Start an OpenAI-compatible HTTP server.
 * Blocks until the server exits.
 */
void unc_serve(UNCRuntime* rt, int port);

/**
 * Free a runtime handle.
 */
void unc_free(UNCRuntime* rt);

#ifdef __cplusplus
}
#endif
