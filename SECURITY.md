# Security

Danglish is designed to run locally. The web server has no authentication and binds to
`127.0.0.1` by default. Do not expose it to a network without an authenticated reverse
proxy and appropriate access controls. The microphone path processes audio in memory and
does not write it to disk; model and voice downloads use the Hugging Face cache.

To report a vulnerability, open a private security advisory on the repository rather than
a public issue. Include the affected version, a minimal reproduction, and whether the
issue is in Danglish or an upstream model/runtime. Do not include private audio or model
files in a report.
