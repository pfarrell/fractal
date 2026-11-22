# Shortcuts for the usual cargo commands. `make help` lists them.

# Find cargo even in shells that haven't sourced ~/.cargo/env.
export PATH := $(HOME)/.cargo/bin:$(PATH)

BIN     := fractal-explorer
RELEASE := target/release/$(BIN)
PREFIX  ?= $(HOME)/.local
ARGS    ?=

.DEFAULT_GOAL := help
.PHONY: help build debug run run-debug test fmt fmt-check lint check png clean install uninstall

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  make %-10s %s\n", $$1, $$2}'

build: ## Build the optimized release binary
	cargo build --release

debug: ## Build an unoptimized debug binary
	cargo build

run: ## Build and run the release binary (pass flags with ARGS="...")
	cargo run --release -- $(ARGS)

run-debug: ## Build and run the debug binary
	cargo run -- $(ARGS)

test: ## Run the test suite
	cargo test

fmt: ## Format the code
	cargo fmt

fmt-check: ## Check formatting without changing files
	cargo fmt --check

lint: ## Run clippy with warnings treated as errors
	cargo clippy --all-targets -- -D warnings

check: fmt-check lint test ## Everything to run before committing: fmt-check, lint, test

png: ## Render a still image (e.g. make png ARGS="mandelbrot -0.5 0 3 out.png")
	cargo run --release --example render_png -- $(ARGS)

clean: ## Delete build output (target/)
	cargo clean

install: build ## Copy the release binary to $(PREFIX)/bin (default ~/.local/bin)
	install -Dm755 $(RELEASE) $(PREFIX)/bin/$(BIN)

uninstall: ## Remove the installed binary
	rm -f $(PREFIX)/bin/$(BIN)
