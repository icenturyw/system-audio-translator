# Security Policy

## Supported Versions

The default branch receives security fixes.

## Reporting a Vulnerability

Please do not open a public issue for a security vulnerability.

Report privately through GitHub Security Advisories if available, or contact the maintainer through the GitHub profile linked from this repository.

Please include:

- Affected version or commit.
- Steps to reproduce.
- Expected and actual behavior.
- Any logs that do not contain private audio, transcripts, tokens, or local model data.

## Data Handling Notes

System Audio Translator may process live system audio, microphone audio, transcripts, and translated text. Users should choose translation providers according to their privacy needs:

- Google Translate mode sends text to the translation provider.
- LM Studio mode can keep translation local when the selected local model and server are local.
- Whisper transcription runs through the configured local `faster-whisper` model.

Do not include private audio, meeting transcripts, API keys, or local model prompts in public issues.
