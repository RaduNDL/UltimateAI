# English training data

`english_seed.jsonl` is the small, hand-reviewed, application-specific part of the
training set. Add examples here instead of putting raw project files into the SFT
corpus. Each line must be a JSON object with `system`, `user`, and `assistant`
messages. Keep all prose in English and write the answer you want the model to
produce.

Good examples are concrete: FastAPI endpoints, Angular components, TypeScript
errors, PostgreSQL design, security, tests, Docker, Git, debugging, and decisions
for this application. Do not add chain-of-thought, generic apologies, fake test
claims, secrets, or copied dependency folders.

`prepare_data.py` combines this seed file with a small filtered stream from the
external code datasets and creates `app-train.jsonl` plus `app-eval.jsonl`. The
evaluation file is held out and must never be used as training input.
