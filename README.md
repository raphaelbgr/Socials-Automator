# Socials Automator

AI-powered Instagram carousel content generator.

## Installation

```bash
pip install -e .
```

## Usage

```bash
# List profiles
socials list-profiles

# Generate a post
socials generate ai.for.mortals --topic "5 ChatGPT tricks" --slides 4
```

## Configuration

- Copy `.env.example` to `.env` and add your API keys
- Edit `config/providers.yaml` to customize AI provider priorities
