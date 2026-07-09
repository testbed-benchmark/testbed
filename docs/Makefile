# Minimal Makefile for Sphinx documentation

SPHINXOPTS  ?=
SPHINXBUILD ?= sphinx-build
SOURCEDIR   = .
BUILDDIR    = _build

HOST ?= 127.0.0.1
PORT ?= 8080

.PHONY: help livehtml Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Live-rebuilding development server.
# Ignores the autoapi-generated .rst files, which are rewritten on every
# build and would otherwise trigger an infinite rebuild loop.
livehtml:
	sphinx-autobuild \
		-b html \
		--host "$(HOST)" \
		--port "$(PORT)" \
		--ignore "*/autoapi/*" \
		"$(SOURCEDIR)" "$(BUILDDIR)/html" \
		$(SPHINXOPTS) $(O)

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)