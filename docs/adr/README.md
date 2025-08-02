# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the photonic-nn-foundry project.

## What is an ADR?

An Architecture Decision Record (ADR) is a document that captures an important architectural decision made along with its context and consequences.

## Format

We use the format proposed by Michael Nygard in his article ["Documenting Architecture Decisions"](http://thinkrelevance.com/blog/2011/11/15/documenting-architecture-decisions).

Each ADR should include:

1. **Status**: What is the status, such as proposed, accepted, rejected, deprecated, superseded, etc.?
2. **Context**: What is the issue that we're seeing that is motivating this decision or change?
3. **Decision**: What is the change that we're proposing or have agreed to implement?
4. **Consequences**: What becomes easier or more difficult to do and any risks introduced by this change?

## Naming Convention

ADRs should be numbered sequentially and include a short descriptive title:
- `001-use-pytorch-as-primary-framework.md`
- `002-implement-photonic-mac-units.md`
- `003-adopt-containerized-development.md`

## Current ADRs

- [001 - Use PyTorch as Primary Deep Learning Framework](./001-use-pytorch-as-primary-framework.md)
- [002 - Implement Photonic MAC Units](./002-implement-photonic-mac-units.md)
- [003 - Adopt Containerized Development Environment](./003-adopt-containerized-development.md)