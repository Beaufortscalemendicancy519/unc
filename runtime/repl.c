// repl.c — Interactive REPL interface (stub)

#include "runtime.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void unc_repl(struct UNCRuntime* rt) {
    char buf[4096];
    char out[16384];

    printf("unc REPL (type 'quit' to exit)\n> ");
    fflush(stdout);

    while (fgets(buf, sizeof(buf), stdin)) {
        buf[strcspn(buf, "\n")] = '\0';
        if (strcmp(buf, "quit") == 0 || strcmp(buf, "exit") == 0) break;
        if (buf[0] == '\0') { printf("> "); fflush(stdout); continue; }

        unc_generate(rt, buf, 256, out, sizeof(out));
        printf("%s\n> ", out);
        fflush(stdout);
    }
}
