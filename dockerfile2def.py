#!/usr/bin/env python3
"""Convert a (single-stage) toolkit Dockerfile into an Apptainer definition file.

Vulcan has no Docker daemon, so each reconstruction module image is rebuilt as a
SIF with `apptainer build --fakeroot` inside a Slurm job (compute nodes reach
PyPI, GitHub, Docker Hub and Ubuntu apt through the cluster proxy).

Usage: dockerfile2def.py DOCKERFILE CONTEXT_DIR OUT_DEF [--base-sif PATH]
Also writes OUT_DEF.workdir containing the final WORKDIR (Docker's default cwd).
"""
import argparse
import re
import shlex
from pathlib import Path

PYTORCH_BASE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
# nvcr.io is unreachable from Vulcan; these images only need a CUDA PyTorch stack.
BASE_SUBSTITUTIONS = {
    "nvcr.io/nvidia/tensorrt:24.12-py3": PYTORCH_BASE,
}


def logical_lines(text: str):
    """Yield Dockerfile instructions with backslash continuations joined."""
    buf = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if not buf and (not line.strip() or line.lstrip().startswith("#")):
            continue
        if line.lstrip().startswith("#") and buf:
            continue  # comment inside a continued RUN
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        buf += line
        yield buf.strip()
        buf = ""
    if buf.strip():
        yield buf.strip()


def parse_env(rest: str):
    """ENV k=v [k2=v2 ...]  or legacy  ENV k v."""
    if "=" in rest.split()[0]:
        return [tuple(tok.split("=", 1)) for tok in shlex.split(rest)]
    key, _, value = rest.partition(" ")
    return [(key, value.strip())]


def convert(dockerfile: Path, context: Path, base_sif: str | None):
    text = dockerfile.read_text()
    if "<<" in text:
        raise SystemExit(f"{dockerfile}: heredoc RUN not supported")
    froms = [l for l in logical_lines(text) if l.upper().startswith("FROM ")]
    if len(froms) != 1:
        raise SystemExit(f"{dockerfile}: expected exactly one FROM, got {len(froms)}")

    post = [
        "set -e",
        "export DEBIAN_FRONTEND=noninteractive",
        # L40S = Ada (sm_89): avoids compiling CUDA extensions for every arch.
        'export TORCH_CUDA_ARCH_LIST="8.9"',
        'export MAX_JOBS="${V2D_BUILD_JOBS:-8}"',
        "export PIP_NO_CACHE_DIR=1",
    ]
    env_lines = []
    workdir = "/"
    bootstrap = None

    for line in logical_lines(text):
        instr, _, rest = line.partition(" ")
        instr = instr.upper()
        rest = rest.strip()
        if instr == "FROM":
            image = rest.split()[0]
            image = BASE_SUBSTITUTIONS.get(image, image)
            if image == PYTORCH_BASE and base_sif:
                bootstrap = f"Bootstrap: localimage\nFrom: {base_sif}"
            else:
                bootstrap = f"Bootstrap: docker\nFrom: {image}"
        elif instr == "ARG":
            name, eq, default = rest.partition("=")
            post.append(f'{name}="${{{name}:-{default}}}"' if eq else f'{name}="${{{name}:-}}"')
        elif instr == "ENV":
            for key, value in parse_env(rest):
                # Double quotes so references like $PYTHONPATH still expand.
                escaped = value.replace('"', '\\"')
                env_lines.append(f'export {key}="{escaped}"')
                post.append(f'export {key}="{escaped}"')
        elif instr == "WORKDIR":
            workdir = rest if rest.startswith("/") else f"{workdir.rstrip('/')}/{rest}"
            post.append(f"mkdir -p {workdir} && cd {workdir}")
        elif instr == "RUN":
            # Subshell: like Docker, each RUN starts in WORKDIR and a `cd` or
            # `export` inside it does not leak into the next instruction.
            post.append(f"( cd {workdir} && {rest} )")
        elif instr in ("COPY", "ADD"):
            args = [a for a in shlex.split(rest) if not a.startswith("--")]
            *srcs, dst = args
            if not dst.startswith("/"):
                dst = f"{workdir.rstrip('/')}/{dst}"
            for src in srcs:
                s = "/opt/v2d_ctx" if src in (".", "./") else f"/opt/v2d_ctx/{src.rstrip('/')}"
                # Docker copies a directory's *contents* into dst (never nesting), and
                # a file into dst/ when dst ends with '/' or there are several sources.
                into_dir = dst.endswith("/") or len(srcs) > 1
                post.append(
                    f'if [ -d {s} ]; then mkdir -p {dst} && cp -a {s}/. {dst}/; '
                    f'elif {"true" if into_dir else "false"}; then mkdir -p {dst} && cp -a {s} {dst}/; '
                    f'else mkdir -p $(dirname {dst}) && cp -a {s} {dst}; fi')
        elif instr in ("LABEL", "EXPOSE", "CMD", "ENTRYPOINT", "USER", "SHELL", "VOLUME", "HEALTHCHECK"):
            continue  # not meaningful for `apptainer exec python -m ...`
        else:
            raise SystemExit(f"{dockerfile}: unsupported instruction {instr}")

    post.append("rm -rf /opt/v2d_ctx")
    body = [
        bootstrap,
        "",
        "%files",
        f"    {context} /opt/v2d_ctx",
        "",
        "%environment",
        *[f"    {l}" for l in env_lines],
        "",
        "%post",
        *[f"    {l}" for l in post],
        "",
    ]
    return "\n".join(body), workdir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dockerfile", type=Path)
    p.add_argument("context", type=Path)
    p.add_argument("out_def", type=Path)
    p.add_argument("--base-sif", default=None)
    a = p.parse_args()
    text, workdir = convert(a.dockerfile, a.context.resolve(), a.base_sif)
    a.out_def.write_text(text)
    Path(str(a.out_def) + ".workdir").write_text(workdir + "\n")
    print(f"wrote {a.out_def} (workdir={workdir})")


if __name__ == "__main__":
    main()
