# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM Engineering book project using Python 3.11.8. The project uses `uv` for dependency management.

## Development Setup

Python version is managed via `.python-version` (currently 3.11.8).

Install dependencies:
```bash
uv sync
```

Run the sample application:
```bash
uv run hello.py
```

## Project Structure

This is an early-stage project with minimal structure. The main entry point is `hello.py` with a basic `main()` function.

## Package Management

- Uses `uv` for Python package management
- Project metadata is in `pyproject.toml`
- No external dependencies currently defined
